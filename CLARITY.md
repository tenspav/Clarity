# Clarity

Clarity is a compression cleanup and chroma reconstruction engine for Apple silicon, built for the FramePortal PS5 Remote Play app. It makes a 10-45 Mbps 10-bit HEVC HDR stream from the PS5 look closer to the console's own picture, with **no extra frame of latency** and a GPU budget of **1.50 ms p95** (v0.2; v0.1 had 1.0 ms) on a MacBook Air M2, with Motion FrameGen and MetalFX running.

**Current version: Clarity v0.2-beta.** Clarity is versioned separately from the app (like [Motion](https://github.com/tenspav/Motion)). Modes: **Auto / Off / On** (`ClarityMode`).

Engineering log and benchmark history: [CLARITY_PROGRESS.md](CLARITY_PROGRESS.md).

## In plain English

A game streams as compressed video. Two things get lost on the way:

- **Colour resolution.** The video carries colour at half the width and half the height of the brightness (4:2:0). A plain decoder blurs the colour back up, so coloured edges (red tail lights, a pink map route, coloured HUD text) bleed a pixel or two past the edge they belong to.
- **Fine detail at low bitrates.** When the network allows only 10-15 Mbps, the encoder leaves faint ripples ("ringing") next to sharp edges and noise in flat areas. A sharpening pass then makes that noise more visible.

Clarity fixes what it safely can, inside passes the renderer already runs:

1. **Colour that follows the edges (Stage 1).** When colour is rebuilt for a pixel, the four nearby colour samples are weighted by how similar their brightness is to this pixel's brightness. A pixel on the bright side of an edge takes its colour from the bright side. This also reads exactly where the colour samples sit on the grid (the stream says so), instead of assuming it.
2. **Sharpening that knows about compression (Stage 2).** At low bitrates the sharpening steps back on faint, noisy patches and keeps full strength on real edges. In HDR it sharpens brightness only, by scaling the colour as a whole, so bright highlights never turn into coloured fringes.
3. **A gentle de-ringer (Stage 3).** On low-bitrate sessions, tiny ripples next to edges are nudged toward their neighbours by at most 2 of 1023 steps. Real edges and texture are left alone. It is off on a clean, high-bitrate stream.
4. **Texture recovery and flat cleanup (Stage 4, v0.2).** Compression smears medium-sized texture (leaves, asphalt grain, brickwork) more than it smears edges. At the source resolution, before upscaling, Clarity lifts that mid-sized detail back where the picture is already textured, and leaves edges, thin lines, text, very dark and very bright areas alone. In flat areas (sky, walls, dark shadows) it does the opposite and eases small compression blocks and steps toward their neighbours. Because detail is restored here, the sharpening after upscaling runs at 70 % so the two do not stack.

What Clarity does not do: no film grain, no fixed 8×8 deblocking grid, no AI upscaling, no extra frame of delay.

## Settings

| Mode | Stage 1 (chroma) | Stage 2 (adaptive CAS) | Stage 3 (de-ringer) | Stage 4 (mid-band + flat clean) |
|---|---|---|---|---|
| **Auto** (default) | on | on, ×0.7, noise limiter scaled by received bitrate | any route, scaled by received bitrate | any route; gains ×(0.5 + 0.5 × eligibility), flat clean × eligibility |
| **On** | on | on, ×0.7, full noise limiter | full strength | full strength, home LAN and away |
| **Off** | plain bilinear (the pre-Clarity path) | plain CAS | off | off |

"WAN" is any route that is not the home LAN (over the internet). The route picks the Stage 4 preset (below); in v0.2 it no longer switches stages on or off.

**Auto is calibrated on the received bitrate.** The PS5 delivers about 45-65 % of the requested rate (10.8 Mbps average on a 45 Mbps 1080p request, 0.17 bpp/frame), so v0.2 moves the eligibility ramp up: full at ≤ 0.30 bpp/frame, zero at ≥ 0.70 (v0.1: 0.20 / 0.50). For 1080p30 that is ≤ 18.7 Mbps received for full strength and ≥ 43.5 Mbps for none, so every real 1080p session today runs at or near full strength.

## Architecture

```
PS5 HEVC 10-bit ──► VideoToolbox ──► x420 planes (Y 16-bit unorm, CbCr 16-bit unorm)
                                         │                     │
                                         │                     └──► Motion FrameGen reads these raw planes
                                         │                          (optical flow on untouched luma)
                                         ▼
   clarity_yuv_rgb (stream res, replaces yuv_rgb)     ◄── same pass, same dispatch, one write
     13 luma reads (centre, radius 1, radius 2) + 2 chroma gathers + 4 guide luma reads per pixel
     Stage 1  siting-correct, luma-guided chroma (guide = ORIGINAL luma)
     Stage 3  de-ringer on DISPLAY luma
     Stage 4  mid-band enhancer + flat cleanup on DISPLAY luma
     YCbCr ──► R'G'B' (PQ / BT.2020 signal kept)
                                         │
                                         ▼
   deblock (existing) ──► MetalFX spatial (existing)
                                         │
                                         ▼
   CAS (existing pass, fused into present at 1:1)
     Stage 2  clarity_cas: same 5 taps, compression-adaptive, HDR-safe, RCAS noise limiter
                                         │
                                         ▼
   present: PQ decode, EDR tone map, deband (existing)
```

- Real frames and Motion-made frames go through the same conversion pass, so the same plan (and de-ringer strength) applies to both.
- Clarity adds **no pass and no texture**. Stages 1, 3 and 4 replace the body of the existing conversion kernel. Stage 2 replaces the body of the existing CAS taps, with **zero new texture reads**.
- Stages 3 and 4 change only the luma that is displayed. The decoder's planes are read-only, so Motion's optical flow still sees the original luma.

### Source layout

| File | What |
|---|---|
| `Sources/Clarity/ClarityVersion.swift` | `ClarityVersion.current = "0.2-beta"` |
| `Sources/Clarity/ClarityCore.swift` | Mode, chroma siting, bitrate meter, mid-band presets, plan, `ClarityController`, CPU reference math |
| `Sources/Clarity/ClarityEngine.swift` | Metal source (`ClarityShaders.source`) and `ClarityEngine` |
| `Sources/Clarity/YUVCoefficients.swift` | YCbCr → R'G'B' matrix and range coefficients |
| `Sources/clarity-bench/main.swift` | Standalone GPU-cost bench with before / after PNGs |

The Metal source has its own include, `clarity_` names and function constants 40-46 (40 guided, 41 dering, 42 raw, 43 mid-band, 44 structure tensor (bench only), 45 linear FP16 output (bench only), 46 wide B2 (bench only)). It is prepended to the renderer's shader source, so both live in one library.

## Shader math

All thresholds work in **10-bit code steps**. For the 16-bit unorm textures that back x420, `code10 = sample × 65535 / 64`. (8-bit planes use `sample × 1020`.)

### Stage 1: siting-correct, luma-guided chroma

**Siting.** Clarity reads Core Video's `kCVImageBufferChromaLocationTopFieldKey`. In luma texel-index coordinates (luma sample k sits at k), chroma sample j sits at `2j + offset`:

| Siting | offset | Note |
|---|---|---|
| Left | (0, 0.5) | HEVC default (chroma_sample_loc_type 0); used when the attachment is missing |
| Center | (0.5, 0.5) | JPEG / MPEG-1 style |
| TopLeft | (0, 0) | BT.2020 / UHD masters often use it |
| Top, BottomLeft, Bottom | (0.5, 0), (0, 1), (0.5, 1) | |

The legacy kernel assumed Left. On a Left stream (the PS5) step (a) is the same picture; on Center or TopLeft content it removes a quarter-pixel colour shift.

**Footprint** of luma pixel `p`: `u = (p − offset) / 2`, `i = floor(u)`, `f = fract(u)`. Bilinear weights over the 2×2 chroma neighbours [00, 10, 01, 11]:

```
b = [(1−fx)(1−fy), fx(1−fy), (1−fx)fy, fx·fy]
```

**Guided weights.** For each neighbour j:

```
Cj = CbCr at j (point read; two gathers fetch the whole 2x2 footprint)
Lj = bilinear read of the ORIGINAL luma at 2j + offset
d  = (Ycentre − Lj) / sigmaY            sigmaY = 16 code steps
wj = bj / (1 + d²)
Cbase  = Σ bj Cj
Cguide = Σ wj Cj / Σ wj
```

**Confidence.** Guiding only helps where a luma edge and a chroma edge coincide. For each horizontal and vertical pair in the footprint:

```
score = smoothstep(4, 16, |ΔY|) × smoothstep(4, 16, max(|ΔCb|, |ΔCr|))
guideConfidence = max(score) × 0.65
Cout = mix(Cbase, Cguide, guideConfidence)
```

All weights are positive, so `Cout` always stays inside the range of its four neighbours: no overshoot, no halo. Stage 1 runs at every bitrate.

v0.2 adds a second cap for very confident edges, `cap = mix(guideMax, guideMaxHigh, smoothstep(0.75, 1, score))`. A/B at 0.85 and 1.0 lowered chroma PSNR on every clip (CLARITY_PROGRESS.md), so `guideMaxHigh` ships at 0.65 (= `guideMax`, the v0.1 picture). With the tile kernel and Left siting the guide luma comes from the threadgroup tile of the original luma, so Stage 1 still reads the original, not the Stage 3/4 output.

### Stage 2: compression-adaptive, HDR-safe CAS

Same 5 taps as the existing CAS (centre c, up, left, right, down), no new reads.

- **Gate.** From the 5 lumas: range `rY`, gradient ratio `max(|E−W|, |S−N|) / rY` (a step through the centre), and axis anisotropy `|lapX − lapY| / (lapX + lapY)` (a thin line). `coherent = saturate(1.5 × max(both) − 0.5)`. Low contrast: `lowC = 1 − smoothstep(0.05, 0.14, rY)`.

  ```
  gate = 1 − noise × lowC × (1 − 0.75 × coherent)        noise = bitrate eligibility (0..1)
  CAS weight × gate
  ```

  A clean stream (noise 0) gets exactly the plain CAS strength. A low-bitrate stream stops sharpening faint, incoherent patches and keeps coherent edges.
- **RCAS noise limiter (v0.2).** From the same 5 taps: `nz = saturate(|mean(N, W, E, S) − c| / rY)`, `gate × (1 − 0.5 × nz × limiter)`. A lone pixel that sticks out of its cross (a compression speck) is sharpened half as much.
- **Gain (v0.2).** While Stage 4 runs, the CAS amount is × 0.7 (`casGainWithMidband`), so detail is not boosted twice.
- **HDR / EDR safety.** The CAS result is reduced to a luma delta, bounded to ±0.06 of signal, clamped to the local luma min / max (no halo), and positive deltas taper to zero as the brightest channel approaches the highlight knee (the PQ level where the present pass starts its roll-off: 0.75 × EDR peak). The pixel is then scaled as a whole: `RGB × (Ynew / Y)`, capped so no channel passes 1.0. No channel is boosted on its own, so saturated HDR colours keep their hue.

### Stage 3: CDEF-inspired de-ringer (WAN)

4 extra point reads (N, S, E, W) of the display luma, `T = 8`:

```
constrain(d, T) = sign(d) · min(|d|, max(T − |d|, 0))
gx = |E − W|, gy = |S − N|
gx > gy + 4: wx = 0.15, wy = 1     gy > gx + 4: wx = 1, wy = 0.15     else 1, 1
D = [wx (c(E−Y) + c(W−Y)) + wy (c(N−Y) + c(S−Y))] / (2 (wx + wy))
edgeGuard = 1 − smoothstep(10, 24, max |neighbour − Y|)
Yout = clamp(Y + clamp(strength · edgeGuard · D, −2, 2), localMin, localMax)
```

The change is at most 2 code steps and never leaves the local min / max.

**Auto strength.** `ClarityBitrateMeter` keeps an exponential average (time constant 1.5 s) of encoded video bytes per frame as received (HEVC payload only, no audio, no FEC), as bits per pixel per frame:

```
eligibility = 1                 at ≤ 0.30 bpp/frame        (v0.1: 0.20)
            = (0.70 − bpp)/0.40 between
            = 0                 at ≥ 0.70                  (v0.1: 0.50)
strength = 0.4 × eligibility    (Auto: any route; On: 0.4)
```

For 1080p30: 0.30 bpp ≈ 18.7 Mbps, 0.70 bpp ≈ 43.5 Mbps received. The same eligibility drives Stage 2's `noise` and Stage 4's strength.

### Stage 4: source-resolution mid-band enhancer and flat cleanup (v0.2)

Runs on the display luma right after Stage 3, inside the same conversion pass, at stream resolution (before MetalFX). Taps around the centre Y: `A1` = N, S, W, E at radius 1, `D1` = the 4 diagonals, `A2` = NN, SS, WW, EE at radius 2. All maths is half precision on differences to the centre (`|d| ≤ 1023` codes is exact in half).

```
r(d)  = saturate(2 − |d| / T)                       range weight: 1 up to T, 0 from 2T
B1    = (4Y + 2 Σ r·A1 + Σ r·D1) / (4 + 2 Σ r_A1 + Σ r_D1)                  3x3, centre 4
B2    = (4Y + 2 Σ r·A1 + 2 Σ r·D1 + Σ r·A2) / (4 + 2 Σ r_A1 + 2 Σ r_D1 + Σ r_A2)   13 taps
fine  = Y − B1,  mid = B1 − B2
span  = max − min of the 3x3
coh   = structure-tensor coherence of the 3x3 (4 cells)

textureGate = smoothstep(3, 9, span)                       no boost on flat patches
strongEdge  = coh × smoothstep(18, 42, span)               no boost on clean, strong edges
spikeGate   = 1 − smoothstep(5, 14, |fine| − 2 |mid|)       no boost on isolated specks
toneGuard   = smoothstep(110, 200, Y) × (1 − smoothstep(720, 840, Y))   deep shadows and highlights
strokeGuard = 1 − smoothstep(10, 24, thin-line contrast)   1-px lines and text strokes (N/S or W/E both on one side)
hudGuard    = 0 inside a HUD rect (dilated by 2 px), else 1
gain  = textureGate (1 − strongEdge) spikeGate toneGuard strokeGuard hudGuard
delta = clamp(gain (gFine · fine + gMid · mid), ±cap)

flat clean: shadow = 1 − smoothstep(150, 300, Y)
            cg = flatClean × (1 − smoothstep(2 + 2·shadow, 4 + 4·shadow, span)) × hudGuard
            delta += clamp(cg × (B2 − Y), ±1.5)

Yout = clamp(Yin + delta, min3x3, max3x3)                  never leaves the original 3x3 range: no halo, no overshoot
```

Presets (`ClarityMidband`; Auto scales `gFine` and `gMid` by `0.5 + 0.5 × eligibility`, On uses them as is):

| Preset | gFine | gMid | cap | T |
|---|---|---|---|---|
| 1080p, home LAN | 0.15 | 1.0 | 8 | 16 |
| 1080p, WAN | 0 | 1.0 | 6 | 11 |
| 720p and below, any route | 0 | 0.9 | 5 | 11 |

Deviations from the v0.2 spec, all measured (CLARITY_PROGRESS.md › v0.2-beta):
- **Gains.** The spec's gains (gMid 0.55 / 0.30 / 0.25) moved luma by about 0.4 code RMS: no visible change. The shipped gains are about 2-3×; they restore the 5×5-band detail energy of the original-bitrate picture to the level of Off, without Off's noise, and still beat v0.1 on Y PSNR and flicker.
- **Range weight.** `saturate(2 − |d|/T)` (a linear ramp) instead of `1 − smoothstep(T, 2T, |d|)`: the same end points, cheaper in half precision.
- **Structure tensor** (anisotropic weights along edges): measured no gain (within ±0.02 dB and ±0.003 flicker on every clip), so it is not shipped. Function constant 44 keeps it for the bench.
- **25-tap [1,2,2,2,1]² B2**: more detail energy (1.005 vs 0.965 at 10 Mbps) but lower Y PSNR, more flicker and +0.2-0.3 ms. Not shipped; function constant 46 keeps it for the bench.
- **Tile vs direct reads**: a 16×16 threadgroup tile (20×20 with halo, guide luma read from the tile) was 0.12-0.16 ms faster than direct reads with the 25-tap B2, but with the shipped 13-tap B2 direct reads are 0.02-0.06 ms faster at p50, so direct reads ship (`clarity_yuv_rgb_tile` stays for the bench).

## HDR and chroma handling

- The work textures keep the stream's own signal (PQ / BT.2020 in HDR). Clarity runs before PQ decode (Stages 1, 3) and on the PQ-coded output of MetalFX (Stage 2), the same places as the code it replaces. No extra conversion.
- Luma weights follow the signal: BT.2020 for PQ, BT.709 for SDR.
- All Stage 1 and 3 decisions use original luma code values; only display luma is changed.
- Chroma never leaves the convex hull of its four neighbours.
- Stage 4 never leaves the original 3×3 luma range and skips deep shadows (< 110 codes) and highlights (> 840 codes), so it cannot push HDR highlights or crush blacks.

**Colour domain (v0.2 A/B).** The render chain upscales PQ-coded R'G'B' in rgb10a2 with MetalFX `.perceptual`. v0.2 measured the alternative: linear-light FP16 out of the conversion pass, MetalFX `.hdr`, CAS on linear, and only tone map plus deband left in present. Test: the original clip downscaled 2× in linear light, upscaled back by each chain, Y PSNR in PQ codes against the original (11 frames).

| Domain | Y PSNR all / edges / flat | Whole chain GPU time vs PQ |
|---|---|---|
| PQ R'G'B' rgb10a2, MetalFX `.perceptual` (shipped) | 39.67 / 33.49 / 39.63 dB | |
| Linear FP16, MetalFX `.hdr` | 39.95 / 33.89 / 39.93 dB | +1.0 to +1.2 ms p50 |

Linear is +0.3 to +0.4 dB, but the test favours it (the downscale itself was done in linear light) and it costs about 1 ms per frame (the conversion itself is only +0.02-0.06 ms p50; the rest is MetalFX `.hdr` on FP16 and the doubled bandwidth). That is most of the 1.50 ms budget for a gain below what is visible in the crops, so **the PQ `.perceptual` path stays**.

## GPU cost

MacBook Air M2, output 2560×1440 MetalFX, CAS 0.25. Per kernel: 10 back-to-back dispatches per command buffer, GPU time / 10 (stage-boundary timestamps include waits for other GPU work and overstate small kernels). "Total" is (convert − legacy convert) + (CAS − legacy CAS), paired per frame. Full tables: [CLARITY_PROGRESS.md](CLARITY_PROGRESS.md) › v0.2-beta.

| Clarity GPU time added, ms p50 / p95 | v0.1-beta | v0.2-beta |
|---|---|---|
| 1080p 18.6 Mbps, Motion FrameGen load | 0.27 / 0.59 | **0.88 / 1.23** |
| 1080p 10.1 Mbps, Motion FrameGen load | 0.25 / 0.60 | **0.87 / 1.20** |
| 1080p 18.6 Mbps, no Motion | 0.35 / 0.51 | 1.01 / 1.34 |
| 720p 4.7 Mbps WAN, Motion FrameGen load | 0.25 / 0.41 | **0.42 / 0.60** |

All under the 1.50 ms p95 gate. The conversion pass is most of it (Stage 4: about +0.6 ms at 1080p); the CAS limiter adds nothing measurable (+0.00 ms vs v0.1 CAS). Without Motion the GPU runs at a lower clock, so every kernel, legacy included, is slower; the gate is set with Motion running, as in a real session.

## Verification

This repo:

```
swift run -c release clarity-bench CLIP.mp4 [--count N] [--mode on|auto] [--bpp X] [--wan] [--png DIR --frame K]
```

It times the plain conversion + CAS against Clarity's (10 back-to-back dispatches per command buffer, at the source resolution) and writes before / after PNGs of one frame.

The quality numbers in [CLARITY_PROGRESS.md](CLARITY_PROGRESS.md) come from FramePortal's in-app bench (`--bench-clarity CLIP --v02`), which also runs Motion, MetalFX and the whole render chain. Its columns: Off, v0.1, v0.2, v0.2 + structure tensor, v0.2 at the spec's 13-tap gains, and an "alt" variant. Quality is Y PSNR against the original-bitrate decode over all, textured (3×3 span 6-48 codes) and flat pixels, the 5×5-band detail energy and its correlation with the reference on textured pixels, and the added frame-to-frame flicker.

- Self-check: the GPU kernel against a plain-read reference kernel (`Variant(perPixel: true)`) and against the CPU reference (`ClarityMath`) for all six sitings.
- Chroma quality: a 4:4:4 reference with known full chroma, made from the clip (960×540 and 480×270), re-subsampled with each siting and rebuilt. CbCr PSNR.
- De-ringer and pipeline quality: a low-bitrate re-encode against the original-bitrate decode.
