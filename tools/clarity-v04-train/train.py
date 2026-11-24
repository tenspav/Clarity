"""Train a fixed low-capacity luma residual. Train-only augmentations stay in Python."""
import argparse
import functools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import CAP, COUNTS, HEADROOM, LumaResidual, SPECS

HALO = 16


@functools.lru_cache(maxsize=32)
def mmap(path):
    return np.load(path, mmap_mode="r")


class Patches(Dataset):
    def __init__(self, manifest, split, count, patch, seed, paired):
        rows = [json.loads(s) for s in manifest.read_text().splitlines()]
        self.bins = [[r for r in rows if r["split"] == split and r["h"] == h] for h in (1080,720)]
        self.bins = [b for b in self.bins if b]
        if not self.bins:
            raise ValueError(f"no {split} data")
        self.root, self.count, self.width = manifest.parent, count, patch+HALO*2
        self.seed, self.training, self.paired = seed, split=="train", paired

    def __len__(self):
        return self.count

    def __getitem__(self,i):
        rng = random.Random(random.getrandbits(64) if self.training else self.seed+i)
        row = rng.choice(self.bins[i%len(self.bins)])
        x = mmap(self.root / row["input"])
        y = mmap(self.root / row["target"])
        m = mmap(self.root / row["hud"])
        phase = ((i>>1)&1, (i>>2)&1)
        n = self.width
        h,w = m.shape
        if self.training and rng.random() < .25 and m.any():
            ys,xs = np.nonzero(m)
            k = rng.randrange(len(ys))
            sy = min(h-n-1,max(0,int(ys[k])-n//2+rng.randint(-n//4,n//4)))
            sx = min(w-n-1,max(0,int(xs[k])-n//2+rng.randint(-n//4,n//4)))
        else:
            sy,sx = rng.randrange(h-n),rng.randrange(w-n)
        sy,sx = (sy//2)*2+phase[0],(sx//2)*2+phase[1]
        crop = np.s_[sy:sy+n,sx:sx+n]
        possible_pairs = [(j,j+1) for j,t in enumerate(row["positions"][:-1])
                          if row["positions"][j+1] == t+1]
        if self.paired:
            k,j = rng.choice(possible_pairs)
        else:
            k = rng.randrange(len(x))
            j = k
        def patch(data, index):
            return torch.from_numpy(np.array(data[index][crop],dtype=np.float32))[None]
        return patch(x,k),patch(y,k),torch.from_numpy(np.array(m[crop],np.float32))[None],patch(x,j),patch(y,j)


def robust(error, eps=.5):
    return (error.square() + eps**2).sqrt() - eps


def interior(x):
    return x[...,HALO:-HALO,HALO:-HALO]


def loss_value(raw, x, y, mask, second=None):
    # FP32 code-step units for loss, caps and normalization.
    raw = raw.float()*HEADROOM
    z = (x + raw.clamp(-CAP,CAP)).clamp(64,940)
    bounded_error = interior(z-y)
    weight = 1 + 2*interior(F.max_pool2d(mask,3,stride=1,padding=1))
    pixel = (robust(bounded_error)*weight).mean()/weight.mean()
    err = interior(z-y)
    dx,dy = err[...,1:]-err[...,:-1], err[...,1:,:]-err[...,:-1,:]
    gx,gy = torch.maximum(weight[...,1:],weight[...,:-1]),torch.maximum(weight[...,1:,:],weight[...,:-1,:])
    edge = .5*((robust(dx)*gx).mean()/gx.mean()+(robust(dy)*gy).mean()/gy.mean())
    cap_penalty = (interior(raw.abs())-CAP).relu().mean()
    value = pixel+.20*edge+.02*cap_penalty
    if second is not None:
        raw2,x2,y2 = second
        z2 = (x2 + (raw2.float()*HEADROOM).clamp(-CAP,CAP)).clamp(64,940)
        # Compare errors only where clean luma changed little. No runtime flow dependency.
        valid = (interior((y2-y).abs()) <= 4).float()*(1-interior(mask))
        flicker = (robust(interior((z2-y2)-(z-y)))*valid).sum()/valid.sum().clamp_min(1.)
        value = value + .20*robust(interior(z2-y2)).mean() + .05*flicker
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest",type=Path)
    p.add_argument("kind",choices=SPECS)
    p.add_argument("--epochs",type=int,default=36)
    p.add_argument("--steps",type=int,default=750)
    p.add_argument("--patch",type=int,default=128)
    p.add_argument("--batch",type=int)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--device",default="cuda")
    p.add_argument("--temporal",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=2417)
    p.add_argument("--head-warm",type=int,default=0,
                   help="train only the zero-initialised last conv for this many steps (avoids dying ReLUs)")
    p.add_argument("--head-init",type=float,default=0.,help="std of last-conv weight init (0 = spec zero init)")
    p.add_argument("--init-from",type=Path,help="Packed16 only: function-preserving warm start from a Packed8 checkpoint")
    p.add_argument("--no-amp",action="store_true",help="FP32 training")
    p.add_argument("--lr",type=float,default=3e-4,help="peak learning rate")
    p.add_argument("--log",type=Path,help="JSONL loss curve (train mean, val per epoch)")
    a = p.parse_args()
    if a.patch % 2:
        p.error("--patch must be even")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    amp = dev.type == "cuda" and not a.no_amp
    if amp:
        torch.backends.cudnn.benchmark = True
    batch = a.batch or (16 if a.kind == "Packed16" else 24)
    model = LumaResidual(a.kind).to(dev)
    if a.init_from:
        src = torch.load(a.init_from,map_location="cpu",weights_only=True)
        assert a.kind == "Packed16" and src["kind"] == "Packed8"
        w = src["model"]
        n = model.net
        with torch.no_grad():
            n[0].weight[:8] = w["net.0.weight"]; n[0].bias[:8] = w["net.0.bias"]
            n[2].weight[:8, :8] = w["net.2.weight"]; n[2].weight[:8, 8:] = 0; n[2].bias[:8] = w["net.2.bias"]
            n[4].weight[:8] = 0; n[4].bias[:8] = 0
            for c in range(8):
                n[4].weight[c, c, 1, 1] = 1.  # identity on the Packed8 path (ReLU of ReLU output)
            n[6].weight.zero_(); n[6].weight[:, :8] = w["net.4.weight"]; n[6].bias.copy_(w["net.4.bias"])
        model = model.to(dev)
    if a.head_init > 0:
        torch.nn.init.normal_(model.net[-1].weight, std=a.head_init)
    head = list(model.net[-1].parameters())
    body = [q for q in model.parameters() if all(q is not h for h in head)]
    opt = torch.optim.AdamW([{"params":body},{"params":head}],lr=a.lr,betas=(.9,.99),weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda",enabled=amp)
    train = DataLoader(Patches(a.manifest,"train",a.steps*batch,a.patch,10,a.temporal),
                       batch_size=batch,drop_last=True,num_workers=a.workers,
                       pin_memory=amp,persistent_workers=a.workers>0)
    val = DataLoader(Patches(a.manifest,"val",batch*24,a.patch,501,False),
                     batch_size=batch,num_workers=0)
    total_steps = a.steps*a.epochs
    best = math.inf
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        start = time.perf_counter()
        model.train()
        run_loss = torch.zeros((),device=dev)
        for i,(x,y,mask,x2,y2) in enumerate(train):
            step = epoch*a.steps+i
            warm = min(500,max(1,total_steps//10))
            if step < warm:
                lr = a.lr*(step+1)/warm
            else:
                lr = a.lr/10+.5*(a.lr-a.lr/10)*(1+math.cos(math.pi*(step-warm)/max(1,total_steps-warm)))
            opt.param_groups[0]["lr"] = 0. if step < a.head_warm else lr
            opt.param_groups[1]["lr"] = lr
            x,y,mask,x2,y2 = [z.to(dev,non_blocking=True) for z in (x,y,mask,x2,y2)]
            if i%8 == 0:
                # Entire identity batch, including synthetic HUD where present.
                x = y
                x2 = y2
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type,dtype=torch.float16,enabled=amp):
                r = model.native((x-64)/HEADROOM)
                r2 = model.native((x2-64)/HEADROOM) if a.temporal and i%4==1 else None
            loss = loss_value(r,x,y,mask,(r2,x2,y2) if r2 is not None else None)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            run_loss += loss.detach()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt)
            scaler.update()
        model.eval()
        scores = []
        with torch.no_grad():
            for x,y,mask,_,_ in val:
                x,y,mask = (z.to(dev) for z in (x,y,mask))
                r = model.native((x-64)/HEADROOM)
                scores.append(loss_value(r,x,y,mask).item())
        score = float(np.mean(scores))
        train_loss = float(run_loss)/a.steps
        print(f"{a.kind} ({COUNTS[a.kind]} params) epoch {epoch+1}, train={train_loss:.5f}, val={score:.5f}, "
              f"lr={lr:.2e}, {time.perf_counter()-start:.1f}s",flush=True)
        if a.log:
            with a.log.open("a") as f:
                f.write(json.dumps(dict(kind=a.kind,epoch=epoch+1,train=train_loss,val=score,lr=lr,
                                        seconds=time.perf_counter()-start))+"\n")
        if score < best:
            best = score
            torch.save({"kind":a.kind,"model":model.state_dict(),"epoch":epoch+1,"val_loss":score},a.output)


if __name__ == "__main__":
    main()
