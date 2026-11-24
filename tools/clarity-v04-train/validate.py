"""Evaluate identical luma-stage outputs. Device-only timing and Motion tests are in README.md."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

LIMITS = dict(y_psnr_min_gain=.15, edge_psnr_min_gain=0., hud_rmse_max_regression=.15,
              flicker_max_regression=.10)


def edges(y):
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[...,1:-1] = .5*(y[...,2:]-y[...,:-2])
    gy[...,1:-1,:] = .5*(y[...,2:,:]-y[...,:-2,:])
    return np.hypot(gx,gy) >= 12


def test_psnr(mse):
    return 20*math.log10(876)-10*math.log10(max(mse,1e-12))


def group_key(r, by):
    parts = [str(r["h"])]
    if "mode" in by:
        parts.append(r["mode"][:3])
    if "clip" in by:
        parts.append(f"clip{r['clip']}")
    if "rate" in by:
        # Observed-rate tercile bins per resolution (Mb/s).
        lo, hi = (12, 18) if r["h"] == 1080 else (5, 8)
        o = r["observed_mbps"]
        parts.append("low" if o < lo else ("mid" if o < hi else "high"))
    return "/".join(parts)


def compare(root, manifest, baseline, new, split="test", by=()):
    # group/(method)/(metric): accumulate squared error and pixel count
    sums = {}
    n = 0
    for line in manifest.read_text().splitlines():
        r = json.loads(line)
        if r["split"] != split:
            continue
        group = group_key(r, by)
        y = np.load(root/r["target"]).astype(np.float32)
        mask = np.broadcast_to(np.load(root/r["hud"])!=0,y.shape)
        mask_edge = edges(y)
        pairs = [(i,i+1) for i,k in enumerate(r["positions"][:-1]) if r["positions"][i+1]==k+1]
        for mode,parent in [("base",baseline),("new",new)]:
            # "identity" baseline = the decoded HEVC input itself (no restoration).
            x = np.load((root if str(parent) == "identity" else parent)/r["input"]).astype(np.float32)
            if x.shape != y.shape or not np.isfinite(x).all():
                raise ValueError(f"invalid {mode} for {r['input']}")
            d = x-y
            def add(stat, values):
                a = sums.setdefault((group,mode,stat),[0.,0])
                a[0] += float(np.square(values.astype(np.float64)).sum())
                a[1] += values.size
            add("Y",d)
            add("edge",d[mask_edge])
            add("HUD",d[mask])
            for i,j in pairs:
                add("flicker",(d[j]-d[i])[~mask[i]])
        n += 1
    if not n:
        raise ValueError("no test rows")
    for g in sorted({k[0] for k in sums}):
        def mse(m,s):
            val,num = sums[g,m,s]
            if not num:
                raise ValueError(f"missing {g}/{s} validation samples")
            return val/num
        y = test_psnr(mse("new","Y"))-test_psnr(mse("base","Y"))
        e = test_psnr(mse("new","edge"))-test_psnr(mse("base","edge"))
        h = math.sqrt(mse("new","HUD"))-math.sqrt(mse("base","HUD"))
        f = math.sqrt(mse("new","flicker"))-math.sqrt(mse("base","flicker"))
        p = y >= LIMITS["y_psnr_min_gain"] and e >= LIMITS["edge_psnr_min_gain"] and h <= .15 and f <= .10
        print(json.dumps(dict(group=g,y_psnr_base=test_psnr(mse("base","Y")),y_psnr_new=test_psnr(mse("new","Y")),
                              y_psnr_gain=y,edge_psnr_gain=e,hud_rmse_delta=h,
                              temporal_error_rms_delta=f,rows=sums[g,"new","Y"][1]//(8*int(g.split("/")[0])*(16*int(g.split("/")[0])//9)),passed=p)))
        if not p:
            failed.add(g)


def probes(checkpoint):
    import torch
    from model import CAP,HEADROOM,LumaResidual

    ckpt = torch.load(checkpoint,map_location="cpu",weights_only=True)
    m = LumaResidual(ckpt["kind"]).eval()
    m.load_state_dict(ckpt["model"])
    size = 256
    x = np.broadcast_to(np.linspace(64,200,size,dtype=np.float32)[None,:],(size,size)).copy()
    flat = np.full_like(x,84)
    def residual(v):
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(v))[None,None]
            return (m.native((t-64)/HEADROOM)[0,0]*HEADROOM).clamp(-CAP,CAP).numpy()
    r = residual(x)
    if not np.isfinite(r).all():
        raise ValueError("nonfinite residual")
    sl = np.s_[24:-24,24:-24]
    center = r[sl]
    worst = 0.
    for i,j in [(0,1),(1,0),(1,1)]:
        a = residual(np.roll(x,(i,j),(0,1)))
        a = np.roll(a,(-i,-j),(0,1))[sl]
        worst = max(worst,float(np.sqrt(np.mean((a-center)**2))))
    fr = residual(flat)[sl]
    phase = [float(fr[i::2,j::2].mean()) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    span = max(phase)-min(phase)
    z = (x+r)[sl]
    reverse_steps = int(np.sum(np.diff(z,axis=1)<-.1))
    p = worst <= .25 and span <= .25 and reverse_steps==0
    print(json.dumps(dict(phase_shift_rms=worst,flat_phase_span=span,ramp_reverse_steps=reverse_steps,passed=p)))
    if not p:
        failed.add("probes")


def predict(manifest, checkpoint, dest, device, split="test"):
    import torch
    from model import CAP, HEADROOM, LumaResidual

    state = torch.load(checkpoint,map_location="cpu",weights_only=True)
    model = LumaResidual(state["kind"]).to(device).eval()
    model.load_state_dict(state["model"])
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row["split"] != split:
            continue
        src = np.load(manifest.parent/row["input"],mmap_mode="r")
        predictions = []
        with torch.inference_mode():
            for frame in src:
                x = torch.from_numpy(frame.astype(np.float32))[None,None].to(device)
                delta = (model.native((x-64)/HEADROOM) * HEADROOM).clamp(-CAP,CAP)
                predictions.append((x + delta).clamp(64,940).cpu().numpy()[0,0])
        output = dest/row["input"]
        output.parent.mkdir(parents=True,exist_ok=True)
        np.save(output,np.stack(predictions))


def identity(manifest, checkpoint, device, split="val"):
    """Clean-input gate: residual RMS <=0.3 code and |r| p99 <=1 code outside HUD."""
    import torch
    from model import CAP, HEADROOM, LumaResidual

    state = torch.load(checkpoint,map_location="cpu",weights_only=True)
    model = LumaResidual(state["kind"]).to(device).eval()
    model.load_state_dict(state["model"])
    seen, out = set(), {}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row["split"] != split or row["target"] in seen:
            continue
        seen.add(row["target"])
        y = np.load(manifest.parent/row["target"],mmap_mode="r")
        m = np.load(manifest.parent/row["hud"]) == 0
        acc = out.setdefault(row["h"],[0.,0,[]])
        with torch.inference_mode():
            for frame in y:
                t = torch.from_numpy(frame.astype(np.float32))[None,None].to(device)
                r = (model.native((t-64)/HEADROOM)*HEADROOM).clamp(-CAP,CAP)[0,0].cpu().numpy()
                r = r[2:-2,2:-2][m[2:-2,2:-2]]
                acc[0] += float(np.square(r.astype(np.float64)).sum()); acc[1] += r.size
                acc[2].append(np.abs(r)[::97])
    for h,(sq,n,samples) in sorted(out.items()):
        rms = math.sqrt(sq/n)
        p99 = float(np.percentile(np.concatenate(samples),99))
        ok = rms <= .3 and p99 <= 1.
        print(json.dumps(dict(identity=h,residual_rms=rms,abs_p99=p99,frames=len(samples),passed=ok)))
        if not ok:
            failed.add(f"identity{h}")


failed = set()
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command",required=True)
    c = sub.add_parser("compare")
    c.add_argument("manifest",type=Path)
    c.add_argument("--baseline",type=Path,required=True)
    c.add_argument("--new",type=Path,required=True)
    c.add_argument("--split",default="test")
    c.add_argument("--by",default="",help="comma list: mode,clip,rate")
    s = sub.add_parser("probes")
    s.add_argument("checkpoint",type=Path)
    m = sub.add_parser("predict")
    m.add_argument("manifest",type=Path)
    m.add_argument("checkpoint",type=Path)
    m.add_argument("--out",type=Path,required=True)
    m.add_argument("--device",default="cuda")
    m.add_argument("--split",default="test")
    i = sub.add_parser("identity")
    i.add_argument("manifest",type=Path)
    i.add_argument("checkpoint",type=Path)
    i.add_argument("--device",default="cuda")
    i.add_argument("--split",default="val")
    args = p.parse_args()
    if args.command == "compare":
        compare(args.manifest.parent,args.manifest,args.baseline,args.new,args.split,
                tuple(x for x in args.by.split(",") if x))
    elif args.command == "identity":
        identity(args.manifest,args.checkpoint,args.device,args.split)
    elif args.command == "probes":
        probes(args.checkpoint)
    else:
        predict(args.manifest,args.checkpoint,args.out,args.device,args.split)
    if failed:
        raise SystemExit("gates failed: "+", ".join(sorted(failed)))
