"""Clarity v0.4 dataset: aligned native H.265 and clean HDR luma arrays from N sources.

Differences from prepare.py (v1 spec):
  * any number of videos; strata scale with duration (--strata per video, or auto)
  * all videos supply train and val: per video the last ~10% of strata are val,
    preceded by one skipped guard stratum; intro/outro excluded (middle 90%)
  * 59.94/60 fps sources are normalised to 30 fps by the fps filter
  * one 4K decode per burst feeds both target resolutions (split + two zscale)
  * 122 frames encoded per burst (last kept index is 121; low-delay P, no
    lookahead, so later frames cannot influence kept ones)
  * bursts run in a process pool; all random draws happen up front, so the
    dataset is deterministic for a given seed and argument set
  * per-burst audit: pre-clip out-of-range luma fraction, adjacent-pair MAD,
    exact duplicate pairs
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import prepare as base  # noqa: E402  (reuses HUD, EOTF and encoder code)

SEED = 2417
FRAMES = 122
KEEP = base.KEEP
base.FRAMES = FRAMES
_encoder = base.encoder


X265_PRESET = "superfast"


def _quiet_encoder(mode, mbps, gop, w, h):
    c = _encoder(mode, mbps, gop, w, h)
    if "-x265-params" in c:
        c[-1] = "log-level=error:" + c[-1]
        c[c.index("-preset") + 1] = X265_PRESET
    return c


base.encoder = _quiet_encoder


def encode_luma_fast(raw, dst, w, h, mode, mbps, gop):
    """base.encode_luma with packet-level probing (no extra HEVC decode).

    Picture types come from packet key flags: with B frames off and no scenecut
    every I picture is a keyframe and every non-key picture is P (verified on
    x265 and hevc_nvenc output against ffprobe frame pict_type).
    """
    enc = dst.with_suffix(".mkv")
    decoded = dst.with_suffix(".raw")
    base.run(FF + ["-f", "rawvideo", "-pixel_format", "yuv420p10le", "-video_size", f"{w}x{h}",
                   "-framerate", "30", "-i", str(raw), "-an"]
             + base.encoder(mode, mbps, gop, w, h) + base.TAGS + [str(enc)])
    meta = base.probe(enc, "stream=codec_name,profile,pix_fmt,has_b_frames,color_transfer,"
                           "color_range,color_space,color_primaries,chroma_location:"
                           "packet=size,pts_time,flags")
    packets = meta["packets"]
    si = meta["streams"][0]
    expected = {"profile": "Main 10", "color_range": "tv", "color_transfer": "smpte2084",
                "color_space": "bt2020nc", "color_primaries": "bt2020", "chroma_location": "left"}
    if any(si.get(k) != value for k, value in expected.items()):
        raise ValueError(f"unexpected encoded tags: {si}")
    if len(packets) != FRAMES or int(si["has_b_frames"]) != 0:
        raise ValueError(f"unexpected packet count or reordering: {enc}")
    pts = np.array([float(p["pts_time"]) for p in packets])
    if np.max(np.abs((pts - pts[0]) - np.arange(FRAMES) / 30)) > .002:
        raise ValueError(f"non-CFR or misaligned timestamps: {enc}")
    base.run(FF + ["-i", str(enc), "-map", "0:v:0", "-vf", "extractplanes=y",
                   "-pix_fmt", "gray10le", "-f", "rawvideo", str(decoded)])
    if decoded.stat().st_size != FRAMES * w * h * 2:
        raise ValueError(f"decoded shape does not match: {enc}")
    x = np.memmap(decoded, dtype="<u2", mode="r", shape=(FRAMES, h, w))
    np.save(dst, np.clip(x[KEEP], 64, 940))
    del x
    decoded.unlink()
    enc.unlink()
    return {"mode": mode, "requested_mbps": None if mode.startswith("crf") else mbps,
            "observed_mbps": sum(int(p["size"]) for p in packets) * 8 / (FRAMES / 30) / 1e6,
            "gop": gop, "pict_types": ["I" if "K" in p["flags"] else "P" for p in packets],
            "frame_bytes": [int(p["size"]) for p in packets], "probe": "packets"}


ENCODE = base.encode_luma
FF = base.FF
RES = [(1920, 1080), (1280, 720)]


def source_info(path, chroma_override):
    data = base.probe(path, "format=duration:stream=codec_name,width,height,pix_fmt,"
                            "color_range,color_space,color_transfer,color_primaries,"
                            "chroma_location,avg_frame_rate")
    s = data["streams"][0]
    required = {"width": 3840, "height": 2160, "pix_fmt": "yuv420p10le",
                "color_range": "tv", "color_space": "bt2020nc",
                "color_transfer": "smpte2084", "color_primaries": "bt2020"}
    for k, v in required.items():
        if s.get(k) != v:
            raise ValueError(f"{path}: {k}={s.get(k)!r}, expected {v!r}")
    n, d = map(int, s["avg_frame_rate"].split("/"))
    fps = n / d
    if not (abs(fps - 30) < .05 or abs(fps - 60) < .1):
        raise ValueError(f"{path}: fps {fps}, expected 29.97/30 or 59.94/60")
    chroma = chroma_override or s.get("chroma_location")
    if chroma not in base.CHROMA:
        raise ValueError(f"{path}: chroma siting {chroma!r}; pass --chroma-in")
    return float(data["format"]["duration"]), chroma, fps, s.get("codec_name")


def clean_both(source, start, chroma, dsts, threads):
    """One decode, fps=30, then Spline36 to each target resolution."""
    z = ("zscale=w={w}:h={h}:filter=spline36:dither=none:"
         "rangein=limited:range=limited:primariesin=2020:primaries=2020:"
         "transferin=smpte2084:transfer=smpte2084:matrixin=2020_ncl:"
         "matrix=2020_ncl:chromalin={c}:chromal=left,format=yuv420p10le")
    graph = f"[0:v:0]fps=30,split={len(dsts)}" + "".join(f"[s{i}]" for i in range(len(dsts)))
    for i, (w, h, _) in enumerate(dsts):
        graph += f";[s{i}]" + z.format(w=w, h=h, c=chroma) + f"[o{i}]"
    cmd = FF + ["-threads", str(threads), "-ss", f"{start:.3f}", "-i", str(source),
                "-an", "-sn", "-filter_complex", graph]
    for i, (w, h, dst) in enumerate(dsts):
        cmd += ["-map", f"[o{i}]", "-frames:v", str(FRAMES), "-f", "rawvideo", str(dst)]
    base.run(cmd)
    for w, h, dst in dsts:
        if dst.stat().st_size != FRAMES * w * h * 3:
            raise ValueError(f"unexpected raw length for {source} at {start}: {dst}")


def do_burst(job):
    t0 = time.time()
    out, temp = Path(job["out"]), Path(job["temp"]) / job["key"]
    temp.mkdir(parents=True, exist_ok=True)
    try:
        raws = [(w, h, temp / f"clean{h}.raw") for w, h in RES[:job["nres"]]]
        clean_both(job["source"], job["start"], job["chroma"], raws, job["threads"])
        rows = []
        for w, h, clean in raws:
            name = f"{job['key']}_{h}"
            folder = out / name
            folder.mkdir()
            yuv = np.memmap(clean, dtype="<u2", mode="r+", shape=(FRAMES, 3 * w * h // 2))
            y = yuv[:, :w * h].reshape(FRAMES, h, w)
            u = yuv[:, w * h:w * h * 5 // 4].reshape(FRAMES, h // 2, w // 2)
            v = yuv[:, w * h * 5 // 4:].reshape(FRAMES, h // 2, w // 2)
            yk = np.asarray(y[KEEP])
            audit = dict(
                y_below_64=float((yk < 64).mean()), y_above_940=float((yk > 940).mean()),
                pair_mad=[float(np.abs(yk[i + 1].astype(np.int32) - yk[i]).mean())
                          for i in range(len(KEEP) - 1) if KEEP[i + 1] == KEEP[i] + 1],
                dup_pairs=[int(np.array_equal(yk[i + 1], yk[i]))
                           for i in range(len(KEEP) - 1) if KEEP[i + 1] == KEEP[i] + 1],
                y_mean=float(yk.mean()))
            np.clip(y, 64, 940, out=y)
            np.clip(u, 64, 960, out=u)
            np.clip(v, 64, 960, out=v)
            mask = np.zeros((h, w), np.uint8)
            if job["use_hud"]:
                sprite = base.hud_sprite(random.Random(job["sprite_seed"]), w, h, job["font"],
                                         job["split"] == "val")
                sx, sy, at, pa, *_ = sprite
                mask[sy:sy + at.shape[0], sx:sx + at.shape[1]] = ((at > 0) | (pa > 0))
                for i in range(FRAMES):
                    base.over_hud(y[i], u[i], v[i], sprite)
            np.save(folder / "target.npy", y[KEEP])
            np.save(folder / "hud.npy", mask)
            yuv.flush()
            del y, u, v, yuv
            enc = job["enc"][str(h)]
            for mode in enc["modes"]:
                stats = ENCODE(clean, folder / f"{mode}.npy", w, h, mode,
                                         enc["mbps_" + mode] if "mbps_" + mode in enc else enc["mbps"],
                                         enc["gop"])
                stats["encoder"] = "hevc_nvenc p4/ll CBR" if mode == "nvenc" else f"libx265 {X265_PRESET}"
                rows.append(dict(split=job["split"], clip=job["vid"], block=job["block"],
                                 burst=job["burst"], source=Path(job["source"]).name,
                                 start_s=job["start"], h=h, w=w, positions=KEEP,
                                 target=f"{name}/target.npy", hud=f"{name}/hud.npy",
                                 input=f"{name}/{mode}.npy", hud_on=bool(job["use_hud"]),
                                 **audit, **stats))
            clean.unlink()
        return job["key"], rows, time.time() - t0
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def plan(videos, infos, strata, smoke, extra, font, out, temp, threads, nres, vid_ids):
    jobs = []
    for vid, path, (duration, chroma, fps, _), n in zip(vid_ids, videos, infos, strata):
        rng = random.Random(SEED * 1000 + vid)  # per video: runs can be split by video
        n_val = max(1, round(.10 * (n - 1)))
        guard = n - n_val - 1
        bin_width = .90 * duration / n
        sub = bin_width / 3
        blocks = [0] if smoke else range(n)
        for block in blocks:
            if smoke:
                split = "val" if vid % 2 else "train"
            elif block == guard:
                continue
            else:
                split = "val" if block > guard else "train"
            origin = .05 * duration + block * bin_width
            for burst in range(1 if smoke else 3):
                # Burst windows tile the stratum; 5 s burst + >=1 s margin each side.
                start = origin + burst * sub + rng.uniform(1, max(1.5, sub - 6.2))
                use_hud = burst == 0 or (not smoke and rng.random() < .15)
                sprite_seed = rng.randrange(2 ** 32)
                enc = {}
                for w, h in RES[:nres]:
                    mbps = rng.choice([10, 12, 15, 18, 22] if h == 1080 else [4, 5, 7, 8, 10])
                    gop = rng.choice([30, 60, 120])
                    modes = ["abr"]
                    if extra != "none":
                        modes.append("crf" + str(rng.choice([20, 24, 28, 32])) if extra == "crf" else "nvenc")
                    enc[str(h)] = dict(mbps=mbps, gop=gop, modes=modes)
                    if "nvenc" in modes:  # independent rate draw widens coverage
                        enc[str(h)]["mbps_nvenc"] = rng.choice(
                            [10, 12, 15, 18, 22] if h == 1080 else [4, 5, 7, 8, 10])
                jobs.append(dict(key=f"v{vid}_b{block:03d}_s{burst}", vid=vid, block=block,
                                 burst=burst, split=split, source=str(path), start=start,
                                 chroma=chroma, use_hud=use_hud, sprite_seed=sprite_seed,
                                 enc=enc, font=str(font), out=str(out), temp=str(temp),
                                 threads=threads, nres=nres))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strata", type=int, nargs="*", help="per video; default 45, or 100 if >2 h")
    ap.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    ap.add_argument("--chroma-in", choices=sorted(base.CHROMA))
    ap.add_argument("--extra", choices=["crf", "nvenc", "none"], default="nvenc")
    ap.add_argument("--x265-preset", default="superfast")
    ap.add_argument("--packet-probe", action="store_true", help="skip ffprobe frame decode")
    ap.add_argument("--smoke", action="store_true", help="one burst per source")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=3, help="decoder threads per burst")
    ap.add_argument("--nres", type=int, default=2, choices=[1, 2])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--vid-ids", type=int, nargs="*", help="clip ids, default 0..N-1")
    a = ap.parse_args()
    global X265_PRESET, ENCODE
    X265_PRESET = a.x265_preset
    if a.packet_probe:
        ENCODE = encode_luma_fast
    infos = [source_info(p, a.chroma_in) for p in a.videos]
    strata = a.strata or [100 if d > 7200 else 45 for d, *_ in infos]
    if len(strata) != len(a.videos):
        raise SystemExit("--strata needs one value per video")
    if a.out.exists() and not a.resume:
        raise SystemExit("--out must be a new directory (or --resume)")
    a.out.mkdir(parents=True, exist_ok=True)
    temp = a.out / "_tmp"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    jobs = plan(a.videos, infos, strata, a.smoke, a.extra, a.font, a.out, temp, a.threads, a.nres,
                a.vid_ids or list(range(len(a.videos))))
    (a.out / "plan.json").write_text(json.dumps(dict(
        videos=[str(p.name) for p in a.videos], strata=strata, frames=FRAMES, keep=KEEP,
        infos=[dict(duration=d, chroma=c, fps=f, codec=k) for d, c, f, k in infos],
        jobs=len(jobs)), indent=1))
    done = set()
    mpath = a.out / "manifest.jsonl"
    if a.resume and mpath.exists():
        for line in mpath.read_text().splitlines():
            r = json.loads(line)
            done.add(r["input"].split("/")[0].rsplit("_", 1)[0])
    for j in jobs:  # remove partial folders from an interrupted run
        if j["key"] not in done:
            for w, h in RES:
                shutil.rmtree(a.out / f"{j['key']}_{h}", ignore_errors=True)
    todo = [j for j in jobs if j["key"] not in done]
    print(f"{len(jobs)} bursts planned, {len(todo)} to do, strata={strata}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool, mpath.open("a") as manifest:
        futs = [pool.submit(do_burst, j) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            key, rows, dt = f.result()
            for r in rows:
                manifest.write(json.dumps(r) + "\n")
            manifest.flush()
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] {key} {dt:.1f}s elapsed {el/60:.1f} min "
                  f"eta {el/k*(len(todo)-k)/60:.1f} min", flush=True)
    shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
