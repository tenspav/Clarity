"""Dataset smoke/stat checks for a prepare_v04 manifest (one or merged)."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

man = Path(sys.argv[1])
rows = [json.loads(l) for l in man.read_text().splitlines()]
root = man.parent
deep = "--deep" in sys.argv
by = collections.defaultdict(list)
for r in rows:
    by[(r["split"], r["h"], r["mode"][:3])].append(r)
print("rows", len(rows), "bursts", len({(r["clip"], r["block"], r["burst"]) for r in rows}),
      "scenes(unique timestamps)", 8 * len({(r["clip"], r["block"], r["burst"]) for r in rows}))
for k in sorted(by):
    v = by[k]
    obs = np.array([r["observed_mbps"] for r in v])
    req = [r["requested_mbps"] for r in v if r["requested_mbps"]]
    ratio = np.array([r["observed_mbps"] / r["requested_mbps"] for r in v if r["requested_mbps"]])
    print(k, "n", len(v), "obs Mb/s p10/p50/p90 %.2f/%.2f/%.2f" % tuple(np.percentile(obs, [10, 50, 90])),
          ("obs/req p50 %.3f min %.3f max %.3f" % (np.median(ratio), ratio.min(), ratio.max())) if len(ratio) else "",
          "gop", dict(collections.Counter(r["gop"] for r in v)))
bad = []
for r in rows:
    pt = r["pict_types"]
    exp = ["I" if i % r["gop"] == 0 else "P" for i in range(len(pt))]
    if pt != exp:
        bad.append(("picttypes", r["input"], "".join(pt)))
print("pict-type mismatches", len(bad), bad[:3])
for c in sorted({r["clip"] for r in rows}):
    v = [r for r in rows if r["clip"] == c and r["mode"] == "abr" and r["h"] == 720]
    mad = np.array([m for r in v for m in r["pair_mad"]])
    dup = sum(sum(r["dup_pairs"]) for r in v)
    print(f"clip {c}: bursts {len(v)} train {sum(r['split']=='train' for r in v)} val {sum(r['split']=='val' for r in v)} "
          f"blocks val {sorted({r['block'] for r in v if r['split']=='val'})} "
          f"pairMAD p50 {np.median(mad):.2f} p95 {np.percentile(mad,95):.2f} (>12: {np.mean(mad>12)*100:.1f}%) dup {dup} "
          f"clip<64 {np.mean([r['y_below_64'] for r in v])*100:.4f}% >940 {np.mean([r['y_above_940'] for r in v])*100:.4f}% "
          f"hud {np.mean([r['hud_on'] for r in v])*100:.0f}%")
if deep:
    import random
    sample = random.Random(0).sample(rows, min(40, len(rows)))
    ps = collections.defaultdict(list)
    worst = []
    for r in sample:
        x = np.load(root / r["input"], mmap_mode="r").astype(np.float32)
        y = np.load(root / r["target"], mmap_mode="r").astype(np.float32)
        assert x.shape == y.shape == (8, r["h"], r["w"]) and x.min() >= 64 and x.max() <= 940
        m = np.load(root / r["hud"])
        assert m.shape == (r["h"], r["w"])
        for k in range(8):
            e = {s: float(((np.roll(x[k], s, 1) - y[k])[:, 8:-8] ** 2).mean()) for s in (-1, 0, 1)}
            if not (e[0] < e[-1] and e[0] < e[1]):
                worst.append((r["input"], k, e))
        ps[(r["h"], r["mode"][:3])].append(20 * np.log10(876) - 10 * np.log10(((x - y) ** 2).mean()))
    for k, v in sorted(ps.items()):
        print("PSNR", k, "n", len(v), "p10/p50/p90 %.2f/%.2f/%.2f" % tuple(np.percentile(v, [10, 50, 90])))
    print("alignment failures (shift ±1 better than 0):", len(worst), worst[:3])
