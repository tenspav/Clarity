"""Prepare aligned native H.265 and clean HDR luma arrays from three sources."""
import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 2417
FRAMES = 150
KEEP = [30, 31, 47, 66, 67, 92, 120, 121]
TAGS = ["-color_range", "tv", "-colorspace", "bt2020nc", "-color_trc",
        "smpte2084", "-color_primaries", "bt2020", "-chroma_sample_location", "left"]
FF = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
M1, M2 = 2610 / 16384, 2523 / 32
C1, C2, C3 = 3424 / 4096, 2413 / 128, 2392 / 128
KR, KB, KG = .2627, .0593, .6780
CHROMA = {"left", "center", "topleft", "top", "bottomleft", "bottom"}


def run(cmd):
    subprocess.run(cmd, check=True)


def probe(path, entries):
    return json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         entries, "-of", "json", str(path)]))


def source_info(path, chroma_override):
    data = probe(path, "format=duration:stream=codec_name,width,height,pix_fmt,"
                       "color_range,color_space,color_transfer,color_primaries,"
                       "chroma_location,avg_frame_rate")
    s = data["streams"][0]
    required = {"width": 3840, "height": 2160, "pix_fmt": "yuv420p10le",
                "color_range": "tv", "color_space": "bt2020nc",
                "color_transfer": "smpte2084", "color_primaries": "bt2020"}
    for k, v in required.items():
        if s.get(k) != v:
            raise ValueError(f"{path}: {k}={s.get(k)!r}, expected {v!r}. Inspect source before overriding.")
    n, d = map(int, s["avg_frame_rate"].split("/"))
    if d == 0 or abs(n / d - 30) > .05:
        raise ValueError(f"{path}: needs nominal 30 or 29.97 fps")
    chroma = chroma_override or s.get("chroma_location")
    if chroma not in CHROMA:
        raise ValueError(f"{path}: verify source chroma siting and pass --chroma-in")
    duration = float(data["format"]["duration"])
    if .90 * duration < 1800:
        raise ValueError(f"{path}: middle 90% has less than 30 minutes")
    return duration, chroma


def eotf(pq):
    t = np.power(np.clip(pq, 0, 1), 1 / M2)
    return np.power(np.maximum(t - C1, 0) / (C2 - C3*t), 1 / M1)


def inverse_eotf(linear):
    t = np.power(np.clip(linear, 0, 1), M1)
    return np.power((C1 + C2*t) / (1 + C3*t), M2)


def hud_sprite(rng, w, h, font_path, variant):
    """Whole sprite is in a small even-aligned ROI. Values are linear BT.2020."""
    sw, sh = (w // 5) & ~1, max(42, (h // 11) & ~1)
    sx = (rng.randrange(max(4, w//12), w-sw-max(4, w//12)) // 2) * 2
    sy = ((rng.choice([h // 10, h * 4 // 5])) // 2) * 2
    sy = min(sy, h-sh-4)
    ss = 3
    mask = Image.new("L", (sw*ss, sh*ss))
    draw = ImageDraw.Draw(mask)
    font = ImageFont.truetype(str(font_path), max(9, round(h * rng.uniform(.014, .030))) * ss)
    if variant:
        label = rng.choice(["EXIT 24   1.26 km", "NAV  014", "BRIDGE  7:42", "ARRIVAL  0.84 km"])
    else:
        label = rng.choice(["4:17   0.42 km", "N   034", "SUBTITLE  7", "R2  HOLD"])
    draw.text((7*ss, 3*ss), label, font=font, fill=255, stroke_width=0)
    draw.line((7*ss, 30*ss, (sw-12)*ss, 30*ss), fill=190, width=rng.choice([2, 3, 4]))
    a_text = np.asarray(mask.resize((sw, sh), Image.Resampling.BOX), np.float32) / 255
    a_panel = np.zeros((sh, sw), np.float32)
    if rng.random() < .7:
        a_panel[1:-3, 2:-3] = rng.uniform(.18, .55)
    fg = np.array(rng.choice([[1,1,1], [.2,.8,1], [1,.18,.35], [.7,1,.1]]),
                  np.float32) * (rng.uniform(80, 220) / 10000)
    panel = np.array([.00010, .00016, .00020], np.float32)
    return sx, sy, a_text, a_panel, fg, panel


def over_hud(y, u, v, sprite):
    sx, sy, a, p, fg, panel = sprite
    sh, sw = a.shape
    view = np.s_[sy:sy+sh, sx:sx+sw]
    gate = (a > 0) | (p > 0)

    # In luma centers: 4:2:0 Left has chroma at (2j, 2k+0.5).
    x, t = np.arange(sx, sx+sw) / 2, (np.arange(sy, sy+sh) - .5) / 2
    ix, iy = np.floor(x).astype(int), np.floor(t).astype(int)
    fx, fy = x-ix, t-iy
    def up(ch):
        xa, xb = np.clip(ix, 0, ch.shape[1]-1), np.clip(ix+1, 0, ch.shape[1]-1)
        ya, yb = np.clip(iy, 0, ch.shape[0]-1), np.clip(iy+1, 0, ch.shape[0]-1)
        r = ((1-fx)*ch[np.ix_(ya, xa)] + fx*ch[np.ix_(ya, xb)])
        b = ((1-fx)*ch[np.ix_(yb, xa)] + fx*ch[np.ix_(yb, xb)])
        return (((1-fy[:,None])*r + fy[:,None]*b) - 512) / 896
    yy, uu, vv = (y[view].astype(np.float32)-64)/876, up(u), up(v)
    rr, bb = yy+2*(1-KR)*vv, yy+2*(1-KB)*uu
    gg = (yy-KR*rr-KB*bb)/KG
    rgb = eotf(np.stack((rr, gg, bb), axis=-1))
    rgb = (1-p[...,None])*rgb + p[...,None]*panel
    rgb = (1-a[...,None])*rgb + a[...,None]*fg
    rr, gg, bb = np.moveaxis(inverse_eotf(rgb), -1, 0)
    yy = KR*rr + KG*gg + KB*bb
    y[view][gate] = np.clip(np.rint(64+876*yy[gate]), 64, 940).astype(np.uint16)

    for plane, vals in [(u, (bb-yy)/(2*(1-KB))), (v, (rr-yy)/(2*(1-KR)))]:
        # Left sample centers, simple 2-row area average. Leave unaffected UV samples alone.
        ds = .5*(vals[::2,::2]+vals[1::2,::2])
        mask = gate[::2,::2] | gate[1::2,::2]
        roi = plane[sy//2:(sy+sh)//2, sx//2:(sx+sw)//2]
        roi[mask] = np.clip(np.rint(512+896*ds[mask]), 64, 960).astype(np.uint16)


def clean_segment(source, start, chroma, w, h, dst):
    vf = (f"fps=30,zscale=w={w}:h={h}:filter=spline36:dither=none:"
          f"rangein=limited:range=limited:primariesin=2020:primaries=2020:"
          f"transferin=smpte2084:transfer=smpte2084:matrixin=2020_ncl:"
          f"matrix=2020_ncl:chromalin={chroma}:chromal=left,format=yuv420p10le")
    run(FF+["-threads", "4", "-ss", f"{start:.3f}", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-vf", vf, "-frames:v", str(FRAMES),
            "-f", "rawvideo", str(dst)])
    if dst.stat().st_size != FRAMES*w*h*3:
        raise ValueError(f"unexpected raw length for {source} at {start}")


def encoder(mode, mbps, gop, w, h):
    rate = f"{mbps}M"
    buf = f"{int(mbps*500)}k"
    if mode == "nvenc":
        return ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-pix_fmt", "p010le",
                "-preset", "p4", "-tune", "ll", "-rc", "cbr", "-b:v", rate,
                "-maxrate", rate, "-bufsize", buf, "-g", str(gop), "-bf", "0",
                "-rc-lookahead", "0", "-zerolatency", "1", "-spatial_aq", "1"]
    opts = (f"pools=4:frame-threads=2:keyint={gop}:min-keyint={gop}:"
            "open-gop=0:scenecut=0:bframes=0:rc-lookahead=0:repeat-headers=1:"
            "chromaloc=0:colorprim=bt2020:transfer=smpte2084:"
            "colormatrix=bt2020nc:range=limited")
    c = ["-c:v", "libx265", "-profile:v", "main10", "-preset", "veryfast",
         "-tune", "zerolatency", "-pix_fmt", "yuv420p10le"]
    if mode == "abr":
        c += ["-b:v", rate, "-maxrate", f"{int(mbps*1150)}k", "-bufsize", buf]
    elif mode.startswith("crf"):
        c += ["-crf", mode[3:]]
    return c + ["-x265-params", opts]


def encode_luma(raw, dst, w, h, mode, mbps, gop):
    enc = dst.with_suffix(".mkv")
    decoded = dst.with_suffix(".raw")
    run(FF+["-f","rawvideo","-pixel_format","yuv420p10le","-video_size",f"{w}x{h}",
            "-framerate","30","-i",str(raw),"-an"]
        + encoder(mode, mbps, gop, w, h) + TAGS + [str(enc)])
    meta = probe(enc, "stream=codec_name,profile,pix_fmt,has_b_frames,color_transfer,"
                      "color_range,color_space,color_primaries,chroma_location:"
                      "packet=size,pts_time,flags:frame=pict_type,best_effort_timestamp_time")
    # ffprobe may report packets and frames together when both are requested.
    packets = meta.get("packets") or [p for p in meta.get("packets_and_frames", []) if p.get("type") == "packet"]
    frames = meta.get("frames") or [p for p in meta.get("packets_and_frames", []) if p.get("type") == "frame"]
    si = meta["streams"][0]
    expected = {"profile": "Main 10", "color_range": "tv", "color_transfer": "smpte2084",
                "color_space": "bt2020nc", "color_primaries": "bt2020", "chroma_location": "left"}
    if any(si.get(k) != value for k, value in expected.items()):
        raise ValueError(f"unexpected encoded tags: {si}")
    if (len(frames) != FRAMES or len(packets) != FRAMES or int(si["has_b_frames"]) != 0
            or any(f["pict_type"] == "B" for f in frames)):
        raise ValueError(f"unexpected packet count, frame count or reordering: {enc}")
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in frames])
    if np.max(np.abs((pts-pts[0])-np.arange(FRAMES)/30)) > .002:
        raise ValueError(f"non-CFR or misaligned timestamps: {enc}")
    run(FF+["-i",str(enc),"-map","0:v:0","-vf","extractplanes=y",
            "-pix_fmt","gray10le","-f","rawvideo",str(decoded)])
    if decoded.stat().st_size != FRAMES*w*h*2:
        raise ValueError(f"decoded shape does not match: {enc}")
    x = np.memmap(decoded, dtype="<u2", mode="r", shape=(FRAMES,h,w))
    np.save(dst, np.clip(x[KEEP], 64, 940))
    del x
    decoded.unlink()
    enc.unlink()
    return {"mode": mode, "requested_mbps": None if mode.startswith("crf") else mbps,
            "observed_mbps": sum(int(p["size"]) for p in packets) * 8 / (FRAMES/30) / 1e6,
            "gop": gop, "pict_types": [f["pict_type"] for f in frames],
            "frame_bytes": [int(p["size"]) for p in packets]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs=3, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    ap.add_argument("--chroma-in", choices=sorted(CHROMA))
    ap.add_argument("--extra", choices=["crf","nvenc","none"], default="crf")
    ap.add_argument("--smoke", action="store_true", help="one 720p burst per source")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("--out must be a new directory")
    infos = [source_info(path, a.chroma_in) for path in a.videos]
    a.out.mkdir(parents=True)
    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory(dir=a.out) as temp, (a.out/"manifest.jsonl").open("w") as manifest:
        for vid, (path,(duration,chroma)) in enumerate(zip(a.videos, infos)):
            bin_width = .90*duration/30
            for block in range(1 if a.smoke else 30):
                # Pool A/B train, contiguous val windows at end, C held out.
                split = "test" if vid == 2 else ("train" if block < 24 else "val")
                if vid != 2 and block in (24,25):
                    continue
                if a.smoke and vid == 1:
                    split = "val"
                origin = .05*duration+block*bin_width + (bin_width-60)/2
                for burst in range(1 if a.smoke else 3):
                    start = origin + (0,20,40)[burst] + rng.uniform(3,12)
                    use_hud = burst == 0 or (not a.smoke and rng.random() < .15)
                    sprite_seed = rng.randrange(2**32)
                    for w,h in ([(1280,720)] if a.smoke else [(1920,1080),(1280,720)]):
                        name = f"v{vid}_b{block:02d}_s{burst}_{h}"
                        folder = a.out/name
                        folder.mkdir()
                        clean = Path(temp)/"clean.raw"
                        clean_segment(path,start,chroma,w,h,clean)
                        yuv = np.memmap(clean,dtype="<u2",mode="r+",shape=(FRAMES,3*w*h//2))
                        y = yuv[:,:w*h].reshape(FRAMES,h,w)
                        u = yuv[:,w*h:w*h*5//4].reshape(FRAMES,h//2,w//2)
                        v = yuv[:,w*h*5//4:].reshape(FRAMES,h//2,w//2)
                        np.clip(y,64,940,out=y)
                        np.clip(u,64,960,out=u)
                        np.clip(v,64,960,out=v)
                        mask = np.zeros((h,w),np.uint8)
                        if use_hud:
                            sprite = hud_sprite(random.Random(sprite_seed),w,h,a.font,vid==2)
                            sx,sy,at,pa,*_ = sprite
                            mask[sy:sy+at.shape[0],sx:sx+at.shape[1]] = ((at>0)|(pa>0))
                            for i in range(FRAMES):
                                over_hud(y[i],u[i],v[i],sprite)
                        np.save(folder/"target.npy",y[KEEP])
                        np.save(folder/"hud.npy",mask)
                        yuv.flush()
                        del y,u,v,yuv
                        mbps = rng.choice([10,12,15,18,22] if h==1080 else [4,5,7,8,10])
                        gop = rng.choice([30,60,120])
                        modes = ["abr"]
                        if a.extra != "none":
                            modes.append("crf"+str(rng.choice([20,24,28,32])) if a.extra=="crf" else "nvenc")
                        for mode in modes:
                            dest = folder/f"{mode}.npy"
                            stats = encode_luma(clean,dest,w,h,mode,mbps,gop)
                            row = dict(split=split, clip=vid, block=block, burst=burst, source=str(path),
                                       start_s=start, h=h, w=w, positions=KEEP,
                                       target=f"{name}/target.npy", hud=f"{name}/hud.npy",
                                       input=f"{name}/{mode}.npy", **stats)
                            manifest.write(json.dumps(row)+"\n")
                            manifest.flush()
                        clean.unlink()


if __name__ == "__main__":
    main()
