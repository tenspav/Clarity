"""Clarity v0.3-beta luma stage (Stage 3 de-ringer + Stage 4 mid-band/flat cleanup) on manifest inputs.

NumPy port of the CPU reference (ClarityMath.dering / midbandDeltas, separate=True) with the shipped
v0.3 plan: Auto mode, home LAN, eligibility from the row's observed bits per pixel per frame. Float32
throughout (the shader uses half on differences, so expect sub-0.1-code deviations). Clamp-to-edge
borders. Output: same per-input path under --out, float32 code units (not re-quantized).
"""
import argparse
import json
from pathlib import Path

import numpy as np

DERING_T, DERING_STRENGTH = 8.0, 0.4
BPP_FULL, BPP_ZERO = 0.30, 0.70


def preset(h):
    # (gFine, gMid, cap, rangeT); LAN 1080 / 720.
    return (0.15, 1.0, 8.0, 16.0) if h >= 1000 else (0.0, 0.9, 5.0, 11.0)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def constrain(d, t):
    m = np.minimum(np.abs(d), np.maximum(t - np.abs(d), 0))
    return np.where(d < 0, -m, m)


def luma_v03(Y, bpp):
    elig = float(min(1, max(0, (BPP_ZERO - bpp) / (BPP_ZERO - BPP_FULL))))
    gF, gM, cap, T = preset(Y.shape[0])
    k = .5 + .5 * elig
    gF, gM = gF * k, gM * k
    flat_clean = elig
    strength = DERING_STRENGTH * elig
    P = np.pad(Y, 2, mode="edge")
    h, w = Y.shape

    def t(dy, dx):
        return P[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    N, S, W, E = t(-1, 0), t(1, 0), t(0, -1), t(0, 1)
    NW, NE, SW, SE = t(-1, -1), t(-1, 1), t(1, -1), t(1, 1)
    NN, SS, WW, EE = t(-2, 0), t(2, 0), t(0, -2), t(0, 2)
    # Stage 3
    yD = Y
    if strength > 0:
        gx, gy = np.abs(E - W), np.abs(S - N)
        wx = np.where(gx > gy + 4, .15, 1.0)
        wy = np.where((gy > gx + 4) & ~(gx > gy + 4), .15, 1.0)
        d = (wx * (constrain(E - Y, DERING_T) + constrain(W - Y, DERING_T))
             + wy * (constrain(N - Y, DERING_T) + constrain(S - Y, DERING_T))) / (2 * (wx + wy))
        md = np.maximum(np.maximum(np.abs(N - Y), np.abs(S - Y)), np.maximum(np.abs(E - Y), np.abs(W - Y)))
        guard = 1 - smoothstep(10, 24, md)
        lo4 = np.minimum(Y, np.minimum(np.minimum(N, S), np.minimum(E, W)))
        hi4 = np.maximum(Y, np.maximum(np.maximum(N, S), np.maximum(E, W)))
        yD = np.clip(Y + np.clip(strength * guard * d, -2, 2), lo4, hi4)
    # Stage 4 (v0.3 separate bands)
    a1 = [N, S, W, E]
    d1 = [NW, NE, SW, SE]
    lo = np.minimum.reduce([Y] + a1 + d1)
    hi = np.maximum.reduce([Y] + a1 + d1)
    span = hi - lo
    g = [((N + Y - NW - W) * .5, (W + Y - NW - N) * .5), ((NE + E - N - Y) * .5, (Y + E - N - NE) * .5),
         ((Y + S - W - SW) * .5, (SW + S - W - Y) * .5), ((E + SE - Y - S) * .5, (S + SE - Y - E) * .5)]
    jxx = sum(a * a for a, _ in g) * .25
    jyy = sum(b * b for _, b in g) * .25
    jxy = sum(a * b for a, b in g) * .25
    tr = jxx + jyy
    coh = np.minimum(1, np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1.024))

    def r(q):
        return np.clip(2 - np.abs(q - Y) / T, 0, 1)

    ra1 = [r(q) for q in a1]
    rd1 = [r(q) for q in d1]
    ra2 = [r(q) for q in (NN, SS, WW, EE)]
    vA1 = sum(rw * q for rw, q in zip(ra1, a1))
    vD1 = sum(rw * q for rw, q in zip(rd1, d1))
    vA2 = sum(rw * q for rw, q in zip(ra2, (NN, SS, WW, EE)))
    sA1, sD1, sA2 = sum(ra1), sum(rd1), sum(ra2)
    b1 = (4 * Y + 2 * vA1 + vD1) / (4 + 2 * sA1 + sD1)
    b2 = (4 * Y + 2 * vA1 + 2 * vD1 + vA2) / (4 + 2 * sA1 + 2 * sD1 + sA2)
    fine, mid = Y - b1, b1 - b2
    texture = smoothstep(3, 9, span)
    strong = coh * smoothstep(18, 42, span)
    spike = 1 - smoothstep(5, 14, np.abs(fine) - 2 * np.abs(mid))
    tone = smoothstep(110, 200, Y) * (1 - smoothstep(720, 840, Y))
    sv = np.where(((N > Y) & (S > Y)) | ((N < Y) & (S < Y)), np.minimum(np.abs(N - Y), np.abs(S - Y)), 0)
    sh = np.where(((W > Y) & (E > Y)) | ((W < Y) & (E < Y)), np.minimum(np.abs(W - Y), np.abs(E - Y)), 0)
    stroke = 1 - smoothstep(10, 24, np.maximum(sv, sh))
    gain = texture * (1 - strong) * tone * stroke * spike if (gF > 0 or gM > 0) else 0
    band = gF * fine + gM * mid
    shadow = 1 - smoothstep(150, 300, Y)
    cg = flat_clean * (1 - smoothstep(2 + 2 * shadow, 4 + 4 * shadow, span))
    dflat = np.clip(cg * (b2 - Y), -1.5, 1.5)
    out = yD + (1 - gain) * dflat + gain * np.clip(band, -cap, cap)
    return np.clip(np.clip(out, lo, hi), 64, 940).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    for line in a.manifest.read_text().splitlines():
        row = json.loads(line)
        if row["split"] != a.split:
            continue
        x = np.load(a.manifest.parent / row["input"]).astype(np.float32)
        bpp = row["observed_mbps"] * 1e6 / (30 * row["w"] * row["h"])
        out = np.stack([luma_v03(f, bpp) for f in x])
        dst = a.out / row["input"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, out)


if __name__ == "__main__":
    main()
