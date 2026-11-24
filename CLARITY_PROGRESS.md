# Clarity: progress log

The engineering log of the [Clarity engine](CLARITY.md), newest first. Every Clarity experiment adds an entry with its numbers.

Clarity has its own version number, separate from the FramePortal app it ships in (like [Motion](https://github.com/tenspav/Motion)). It keeps the `-beta` tag until a live A/B on real sessions confirms the offline results.

| Clarity | Shipped in FramePortal | Summary |
|---|---|---|
| **v0.3-beta** | v1.13.0 | FP16 coherence overflow fixed, CAS gain on the luma delta, flat cleanup separated from detail. Same GPU cost as v0.2. Last pure-Metal tuning pass |
| **v0.2-beta** | v1.11.0 | Source-resolution mid-band texture recovery + flat cleanup, RCAS noise limiter, CAS ×0.7, received-bitrate Auto, On = full strength on any route |
| **v0.1-beta** | v1.9.8 | Siting-correct luma-guided chroma, compression-adaptive HDR-safe CAS, WAN de-ringer |

How the numbers are measured (FramePortal's in-app `--bench-clarity` bench; [clarity-bench](README.md#benchmark-it-yourself) in this repo reproduces the timing):
- **Footage:** a 22-second 1080p30 HDR gameplay recording (HEVC Main10, PQ / BT.2020, chroma siting Left, 18.6 Mbps, 660 frames), plus two local re-encodes of it at 14.6 and 10.1 Mbps (HEVC Main10, VideoToolbox). The re-encodes are test copies only and are not in the repository.
- **Chroma PSNR:** a 4:4:4 reference with known full chroma is made from the clip, then re-subsampled with each siting and rebuilt by each method. Two sizes: 960×540 (chroma plane as is, the 2× reference) and 480×270 (chroma as sharp as luma, the harder case). CbCr PSNR in 10-bit codes over all pixels and over chroma-edge pixels. 22 frames per file.
- **De-ringer and pipeline:** the low-bitrate decode, run through each step, against the original-bitrate decode run through the same chain. Y PSNR over all pixels, a band near strong edges, and flat areas.
  - *Fidelity* compares to the original **without** CAS. Less sharpening of noise scores higher here.
  - *Visibility* compares to the original run through the **same** step, so it measures how much of the compression damage stays visible after the step.
- **GPU time:** MacBook Air M2, 1080p source, 2560×1440 MetalFX output, CAS 0.25, 60 FPS pacing, 3 runs × 120 frames.
  - Per-kernel time comes from encoder stage-boundary timestamps (`MTLCounterSampleBuffer`, 1 ns/tick on M2). Increments are paired per frame against the legacy kernel.
  - The whole chain (convert → deblock → MetalFX → CAS) is timed with command-buffer `gpuStartTime` / `gpuEndTime`. Legacy, Clarity 1+2 and Clarity 1+2+3 chains are shuffled per frame.
  - `--motion` adds a live Motion FrameGen load (Motion interpolates the same frames on the same GPU between chains).

## Current standing (v0.3-beta)

| | Off | v0.1-beta | v0.2-beta | v0.3-beta |
|---|---|---|---|---|
| Y PSNR vs original, 1080p 10.1 Mbps LAN (all / textured / flat) | 43.28 / 46.09 / 56.26 | 45.35 / 46.51 / 56.26 | 45.60 / 46.41 / 56.61 | **45.95 / 46.44 / 56.61** |
| Detail energy vs original (1.0 = same), 10.1 Mbps | 0.962 | 0.909 | 0.965 | **0.967** |
| Added flicker RMS, 10.1 Mbps (codes) | 9.01 | 8.17 | 8.14 | **8.03** |
| Clarity GPU time, 1080p + Motion, p50 / p95 | | 0.27 / 0.59 ms | 0.88 / 1.23 ms (gate 1.50) | same as v0.2 (paired +0.00 to +0.02 ms p50) |

**Verdict (v0.3):** a small, consistent step on the numbers at the same cost: Y PSNR +0.16 to +0.63 dB, flicker −0.05 to −0.15 codes and detail energy within ±0.003 on every clip. Almost all of it comes from the CAS gain now acting on the luma delta (0.7, which really sharpens less than v0.2's ×0.7 on the amount). The fidelity metric rewards less sharpening, so part of the PSNR gain is "less CAS" rather than "better CAS"; flicker and detail energy say it is not a loss. Band separation is neutral, and the headroom bound and lazy reads were measured and left off. The overflow fix matters on strong edges and is a correctness fix more than a visible one. No live session A/B yet. **Pure-Metal Clarity tuning is frozen from here.**

**Verdict (v0.2):** v0.2 wins back the fine texture that v0.1 lost (detail energy 0.91 → 0.97, Off's level) and keeps v0.1's noise reduction (flicker and overall PSNR slightly better than v0.1). On textured pixels alone PSNR is 0.03-0.12 dB below v0.1: it adds detail that is close to the original's, but not identical. The visible change is modest: crisper foliage and asphalt grain at 1080p, cleaner sky and flat walls, nothing else moves. Inside the 1.50 ms p95 gate in every final run with Motion.

---

## v0.4-beta experiment: learned ANE luma residual (not shipped)

A tiny 1× luma residual network on the Neural Engine that replaces Stages 3 and 4. Trained offline, exported to Core ML FP16, timed on the M2 through a zero-copy IOSurface bridge. **It fails the ship gate, so it is not part of the Clarity engine (this branch only).** Models: `models/clarity-v04/`. Training and validation code: `tools/clarity-v04-train/`. M2 bench: `tools/clarity-ane-bench/`.

**Ship gate (set before testing):** at least +0.3 dB Y PSNR over v0.3 on average, no clip worse, identity residual ≤0.3 code RMS on clean input, and at most 1.5 ms extra at 1080p p95 under load.

**Data.** Four 4K HDR gameplay recordings (AV1 10-bit, BT.2020 / PQ, limited range; two at 30 fps and two at 59.94 fps, normalised to 30 fps). The strata are scaled to duration: 45 strata per 42-60 min video and 100 on the 4 h drive. Each stratum has 3 bursts, and each burst keeps 8 frames, including 3 true adjacent pairs.
- Size: 693 bursts, **5,544 scenes**, 95 GB of uint16 arrays.
- Split per video: train is the middle 90 % of each video, val is the last ~10 % of strata, with one skipped guard stratum before val. Train has 627 bursts, val 66 per resolution and variant.
- Targets: Spline36 4K → 1080p and 720p in Y′CbCr (zscale), synthetic HUD on ~44 % of the bursts.
- Inputs are HEVC Main10 low-delay P (GOP 30/60/120) in two variants:
  - x265 superfast ABR at 10-22 Mb/s (1080p) and 4-10 Mb/s (720p);
  - NVENC p4/ll CBR at an independent rate draw (in place of the spec's x265 CRF variant, to fit the CPU budget).
- Checks: observed/requested rate p50 1.02-1.04, 0 picture-type mismatches, 0 duplicate pairs, 0 alignment failures, clipping below 0.003 %.

**Training.** AdamW with 500 warm-up steps to 3e-4, then cosine to 3e-5. 36 epochs × 750 steps, FP16 autocast, patch 128 plus a 16 px halo, identity batch every 8th step. Each model ran ~35 min on one GPU at ~14-20 steps/s (data-loader bound).
- **Packed16 never learned** from the spec's zero-initialised last layer: final weights ~0.001 and output 0.04 code. Lower LR, another seed, FP32 and a head-only warm-up did not fix it either.
- It was retrained from a function-preserving warm start out of the trained Packed8: Packed16 starts as exactly Packed8 (max diff 0.0), then trains 36 epochs.

| Model (params) | val loss ep 1 → 36 | Y PSNR vs no-op, 1080p / 720p | **vs v0.3 luma, 1080p / 720p** | edge PSNR vs v0.3 | flicker Δ vs v0.3 | identity RMS / p99 (gate ≤0.3 / ≤1) | phase probes |
|---|---|---|---|---|---|---|---|
| Packed8 (1,172) | 3.114 → 3.066 | +0.059 / +0.036 dB | **+0.10 / +0.03 dB** | +0.04 / +0.02 dB | −0.07 / −0.03 code | 0.51-0.57 / 1.9-2.1 code: fail | pass |
| Packed16 warm start (5,812) | 3.124 → 3.053 | +0.166 / +0.121 dB | **+0.21 / +0.11 dB** | — | — | 0.76-0.85 / 2.6-2.8 code: fail | pass |
| Unpacked8 (737) | 3.108 → 3.011 | +0.149 / +0.106 dB | **+0.19 / +0.10 dB** | +0.09 / +0.05 dB | −0.14 / −0.10 code | 0.87-1.03 / 3.3-4.0 code: fail | pass |

The v0.3 column is a NumPy port of `ClarityMath.dering` + `midbandDeltas` (v0.3 Auto plan, home LAN, received bpp per clip), not the Metal shader: `v03_luma.py`. The shader matches the CPU reference by `--selfcheck`. On this val set v0.3's luma stage alone scores −0.04 dB (1080p) and +0.01 dB (720p) against the decoded input. All gains are larger on the x265 variant than on NVENC (Packed8 at 1080p: +0.15 vs +0.05 dB over v0.3).

**M2 timing.** Pack, then prediction, through preallocated `OneComponent16Half` IOSurfaces (`MLMultiArray(pixelBuffer:)` + `outputBackings`, with the output surface identity checked). 1,000 warmed calls each. `MLComputePlan` (`.cpuAndNeuralEngine`) places every conv and ReLU on the ANE in all 8 packages. Core ML output matches PyTorch within 0.04-0.06 code RMS (max 0.33), and signed deltas survive, including the FP16 grayscale image path. "Under load" means a concurrent 256 MB GPU blit loop: the ANE shares memory bandwidth, so prediction is about 1.8× slower.

| Total pack + predict, p50 / p95 / p99 (ms) | 1080p idle | 1080p under load | 720p idle | 720p under load |
|---|---|---|---|---|
| Packed8 | 1.02 / 1.29 / 1.92 | 1.89 / **2.56** / 3.49 | 0.63 / 0.71 / 0.91 | 1.12 / 1.47 / 1.98 |
| Packed16 | 2.06 / 2.62 / 3.22 | 3.67 / **4.07** / 4.63 | 1.08 / 1.61 / 2.57 | 1.85 / 2.88 / 3.41 |
| Unpacked8 (tensor) | 2.75 / 3.12 / 3.50 | 4.95 / **5.76** / 7.74 | 1.36 / 1.54 / 1.92 | 2.74 / 3.74 / 4.79 |
| Unpacked8 (image) | 2.74 / 3.24 / 3.69 | 4.92 / 5.67 / 6.73 | 1.38 / 1.87 / 2.44 | 2.54 / 3.55 / 4.24 |

The GPU pack kernel itself is 0.03-0.18 ms p50. Under the "never wait" contract the residual is only usable when it is ready before the conversion pass that consumes the frame. Ready within 2 / 3 / 4 ms under load at 1080p: Packed8 66 / 98 / 99 %, Packed16 0 / 1 / 92 %, Unpacked8 0 / 0 / 0 %.

**Verdict: fail, not integrated.**
- **Quality:** the best average gain over v0.3 is +0.16 dB (Packed16; Unpacked8 +0.15 dB). The gate is +0.3 dB.
- **Identity:** every model changes clean input by 0.5-1.0 code RMS. The gate is ≤0.3. One identity batch in eight does not teach the net to leave clean frames alone.
- **Timing:** only Packed8 at 720p stays under 1.5 ms p95 under load. At 1080p Packed8 is 2.56 ms.
- **Not done:** the five-clip `clarity-bench` A/B against the real Metal v0.3 path. It needs the model wired into the bench, and the gate already fails on identity and timing.

Next, if resumed:
- add clean-input pairs at a higher rate, or a bitrate input, to fix identity;
- train Packed8 at 720p only;
- try W8A8 on A19 Pro for bandwidth;
- if one passes, run the real `clarity-bench` A/B.

---

## v0.3-beta: audit fixes, CAS delta gain, band separation (last pure-Metal pass)

Source: an independent audit of v0.2. One focused pass; after it the hand-tuned Metal stages are frozen (bug fixes only). Bench: FramePortal's in-app `--bench-clarity CLIP --v02` (v0.3 columns).

### What changed

- **FP16 overflow in the Stage 4 structure tensor (real).** The audit's claim checked numerically (float16 emulation over all 512 extreme 3×3 patterns): even with the 1/64 pre-scale, `(jxx − jyy)² + 4 jxy²` reaches 1,044,484 over the full code range and 561,750 on a legal 64 → 940 edge; half tops out at 65,504. 308 of 512 full-range patterns overflow, the first at a 512-code step (724 for a straight edge). Coherence then became infinite, so `1 − strongEdge` went to −∞ and the gain to ±∞ or NaN: on large mid-tone steps the delta jumped to ±cap instead of being gated off (the 3×3 clamp limited the damage). Fixed: half gradients, float tensor and coherence (`invTrace`, `length(float2(xx − yy, 2 xy)) × invTrace`, saturated). The stroke guard uses same-sign comparisons instead of products.
- **CAS gain on the luma delta.** New `ClarityCASParams.correctionGain` (struct 48 bytes) scales the bounded, range-clamped luma delta. The plan moves the 0.7 from the sharpness amount (`casGain`, now 1) to `correctionGain`. Sharpness 0 or gain 0 return the pixel unchanged. CPU reference: `ClarityMath.cas`.
- **Flat cleanup separated from detail** (function constant 47): `Yout = Yin + mix(dFlat, clamp(band, ±cap), w)`; v0.2 added `clamp(w · band, ±cap) + dFlat`. CPU reference: `ClarityMath.midbandDeltas`.
- **Instrumentation:** function constant 48 writes the de-ring, flat and detail deltas and `w`; `--deltas` reports them.
- **R'G'B' headroom bound** (function constant 49) and **lazy radius-2 reads** (50): built, measured, not shipped (below).
- `ClarityPlan.enabled` now includes `flatClean`. `ClarityEngine.convertPSO` is lock-guarded and the shipped variants compile on a background thread at start. A threadgroup-height knob for the sweep.
- **Static-HUD rects: not wired.** Motion's static-HUD state is a per-cell GPU buffer private to Motion, not rectangles; feeding `ClarityController.hudRects` from it would mean changing Motion. Still open.
- `ClarityVersion` 0.3-beta. `ClarityPlan.v02()` keeps the v0.2 plan for the A/B.

### Footage

The v0.2 set (1080p30 HDR original at 18.6 Mbps with 10.1 / 14.6 Mbps LAN and 7.5 Mbps WAN re-encodes; 720p 4.7 Mbps WAN against a 40 Mbps 720p reference), plus a 20 s section of the 1080p60 drive recording (asphalt, foliage, shadows, minimap and nav card) re-encoded at 9.3 Mbps against a 53 Mbps re-encode of the same section. Quality on 11 frames (10 on the drive), +1 each for flicker. Test copies only, not in the repository.

### correctionGain: 0.7 vs v0.2's apparent sharpening

`--cas-match` on the 10.1 Mbps clip (CAS 0.25, stream resolution), RMS of the CAS luma change: v0.2 (amount ×0.7, full delta) 2.68 codes; v0.3 with the full amount and the delta ×0.5 / 0.7 / 0.9 / 0.95 / 1.0: 1.40 / 1.96 / 2.52 / 2.66 / 2.80 codes. **0.95 reproduces v0.2's sharpening**; v0.2's ×0.7 on the amount was a ×0.96 on the delta.

### Quality A/B

Y PSNR all / textured / flat (dB) | detail energy, corr | added flicker RMS (codes). "v0.3 @0.95" is v0.3 with the delta gain matched to v0.2.

| Clip | v0.2 | v0.3 @0.95 | **v0.3 (shipped, 0.7)** |
|---|---|---|---|
| 1080p 10.1 Mbps LAN | 45.60 / 46.41 / 56.61, 0.965 / 0.901, 8.138 | 45.61 / 46.41 / 56.61, 0.965 / 0.901, 8.133 | **45.95 / 46.44 / 56.61, 0.967 / 0.902, 8.033** |
| 1080p 14.6 Mbps LAN | 47.56 / 48.74 / 57.85, 0.987 / 0.939, 6.217 | 47.58 / 48.74 / 57.85, 0.987 / 0.939, 6.211 | **48.19 / 48.77 / 57.85, 0.989 / 0.940, 6.066** |
| 1080p 7.5 Mbps WAN | 43.58 / 44.32 / 55.08, 0.889 / 0.853, 10.363 | 43.59 / 44.32 / 55.08, 0.889 / 0.853, 10.359 | **43.77 / 44.34 / 55.08, 0.890 / 0.854, 10.290** |
| 720p 4.7 Mbps WAN | 42.09 / 44.15 / 54.49, 0.950 / 0.850, 10.402 | 42.12 / 44.15 / 54.48, 0.950 / 0.850, 10.388 | **42.36 / 44.18 / 54.48, 0.949 / 0.851, 10.289** |
| 1080p60 drive, 9.3 Mbps LAN | 45.03 / 45.20 / 53.73, 0.930 / 0.871, 9.028 | 45.03 / 45.20 / 53.73, 0.930 / 0.871, 9.026 | **45.19 / 45.21 / 53.73, 0.933 / 0.872, 8.974** |

Off and v0.1 are unchanged from the v0.2 entry (the v0.2 column reproduces its numbers exactly). The drive clip: Off 43.81 / 45.04 / 53.53, 0.922, 9.457; v0.1 44.91 / 45.27 / 53.53, 0.876, 9.002.

Feature ablations (v0.3 with one feature flipped, same runs; 10.1 Mbps, 720p, drive):
- **Band separation off:** within ±0.01 dB, ±0.001 detail energy and ±0.001 flicker on every clip. Neutral. The parts (`--deltas`) show why: on textured pixels the applied flat cleanup drops from 0.065 to 0.053 codes RMS (drive 0.032 → 0.024); detail (0.76 codes RMS on textured) and de-ring are unchanged. Kept on: it costs nothing and flat cleanup no longer acts where detail is restored.
- **Headroom bound on:** identical numbers on every clip; `--selfcheck` finds 0-1 of the sampled pixels changed (the PQ clips have almost no in-gamut pixels near a channel limit that Stage 3/4 moves). **Off by default** (it costs time, below).
- **correctionGain:** 0.7 beats 0.95 on Y PSNR (+0.16 to +0.61 dB), flicker (−0.05 to −0.15) on every clip, with detail energy within 0.003 (+0.002 to +0.003, 720p −0.001); **0.7 ships.** Caveat: the fidelity metric compares against the original without CAS, so less sharpening scores higher there. The flicker and detail-energy columns, which do not favour softness, also do not get worse.

### Correctness (`--selfcheck`, final build)

- v0.2 checks as before: direct vs plain-read 0.0001 code, GPU vs CPU 0.005 code max (10.1 Mbps), 0.006 (720p), 0.005 (drive).
- v0.3: band separation GPU vs CPU 0.005 / 0.006 / 0.005 code; the deltas output against `ClarityMath.midbandDeltas` 0.006-0.010 code.
- Lazy vs eager radius-2 reads: 0.0000 on both 1080p clips, 0.0034 code on 720p (instruction order inside the fast-math kernel); lazy is not shipped.
- The all-half tensor (bench) measured 0.02-0.05 code GPU vs CPU, against 0.005 for float.

### GPU time

`--batch 10`, 3 runs × 150 frames per bench, output 2560×1440, CAS 0.25, Motion FrameGen load (Motion v0.5 default settings), ms p50 / p95. Each bench times v0.2 and v0.3 kernels interleaved per frame (rotating order), so "v0.3 − v0.2 convert" is paired. Each configuration was repeated 3 times, alternating with the main-branch build (whose v0.2 kernel is the original v0.2 source). The machine had background load (browser, Tailscale, about 1-2 cores); repeat 1 was quiet, repeats 2-3 were not.

v0.3 conversion vs the v0.2 conversion, paired, p50 (final build, three repeats):

| Clip | repeat 1 | 2 | 3 |
|---|---|---|---|
| 1080p 18.6 Mbps + Motion | +0.002 | +0.017 | +0.007 |
| 1080p 10.1 Mbps + Motion | +0.003 | +0.010 | +0.019 |
| 720p 4.7 Mbps WAN + Motion | +0.003 | −0.001 | −0.001 |

Clarity total ((convert − legacy) + (CAS − legacy CAS)), p50 / p95:

| Clip, repeat | main build (original v0.2) | v0.3 build: v0.2 | v0.3 build: **v0.3** |
|---|---|---|---|
| 1080p 18.6 Mbps, 1 (quiet) | 0.87 / 1.15 | 0.99 / 1.27 | **0.97 / 1.35** |
| 1080p 18.6 Mbps, 2 | 1.42 / 2.10 | 1.70 / 2.39 | 1.82 / 2.29 |
| 1080p 18.6 Mbps, 3 | 1.43 / 1.84 | 1.39 / 1.78 | 1.39 / 1.94 |
| 1080p 10.1 Mbps, 1 | 1.18 / 1.59 | 1.09 / 1.40 | **1.07 / 1.50** |
| 1080p 10.1 Mbps, 2 | 1.26 / 1.68 | 1.43 / 1.82 | 1.40 / 2.02 |
| 1080p 10.1 Mbps, 3 | 1.33 / 1.73 | 1.39 / 1.75 | 1.40 / 1.89 |
| 720p 4.7 Mbps WAN, 1 | 0.54 / 0.77 | 0.64 / 0.84 | **0.65 / 0.90** |
| 720p 4.7 Mbps WAN, 2 | 0.67 / 0.90 | 0.75 / 0.98 | 0.74 / 0.96 |
| 720p 4.7 Mbps WAN, 3 | 0.62 / 0.85 | 0.69 / 0.90 | 0.68 / 0.91 |

Spread between repeats: ±0.3 ms p50 and ±0.5 ms p95 at 1080p, set by the background load, far more than any v0.3 − v0.2 difference. The total's p95 is the p95 of a per-frame sum of two noisy paired differences, so it swings more than the convert-only pairing (the CAS kernel is identical in v0.2 and v0.3).

Speed items:
- **First build** (float tensor, band separation, headroom on, lazy reads on): +0.07 to +0.10 ms p50 over v0.2 at 1080p, +0.03 at 720p (3 repeats each). The two items below explain it, and both are off in the final build.
- **Lazy radius-2 reads:** paired +0.030 to +0.049 ms p50 at 1080p, +0.014 to +0.015 at 720p (3 repeats each), so slower than plain reads: a SIMD group waits for its slowest lane, and the extra 3×3 logic costs more than four cached reads. Off.
- **Headroom bound:** paired +0.01 to +0.08 ms p50 (12 runs). Off.
- **Float vs all-half tensor:** the half form was 0.006 to 0.014 ms p50 *slower* (4 runs). Float ships.
- **Band separation:** +0.001 to +0.005 ms p50. On.
- **Threadgroup shape** (`--tg-sweep`, 300 frames, Motion): 32 × 4 / × 8 / × 32 within 0.01 ms p50 for both v0.2 and v0.3 at 1080p and 720p. The maximum stays.

Gates:
- **v0.3 total ≤ v0.2 total:** met as parity (paired convert +0.00 to +0.02 ms p50). The audit's 0.7-0.8 ms target was not reached: neither the lazy reads nor the threadgroup shape speed up this kernel.
- **p95 ≤ 1.50 ms with Motion:** met in the quiet repeat at 1080p 10.1 Mbps (1.50) and at 720p; 1.35 ms at 1080p 18.6 Mbps quiet. Under background load both v0.2 and v0.3 went over the gate together (1.7-2.4 ms). It needs a re-run on a quiet machine before the live A/B.
- **No quality regression vs v0.2:** met (above).

### Still to do

- Re-measure GPU time on a quiet machine with Motion v0.5 (the gate runs here were under background load).
- Live A/B on real LAN and WAN sessions with a perf trace, before the beta tag comes off.
- Feed `ClarityController.hudRects` (needs an exported rectangle form of Motion's static-HUD mask).
- Pure-Metal tuning is frozen: the next quality step is the learned residual below.

---

## v0.2-beta: texture recovery, flat cleanup, received-bitrate Auto

Bench: FramePortal's `--bench-clarity CLIP --v02`.

### What changed

- **Stage 4 (new), in the conversion pass at source resolution:** an edge-aware mid-band enhancer (13-tap B2) with texture, strong-edge, spike, tone, stroke and HUD gates, plus a flat cleanup that eases 2-8 code blocks toward B2 (wider in shadows). It never leaves the original 3×3 range. Maths in CLARITY.md › Stage 4.
- **Stage 2:** RCAS-style 5-tap noise limiter (no new reads); CAS amount ×0.7 while Stage 4 runs.
- **Stage 3:** the de-ringer now runs on any route in Auto (scaled by eligibility), not WAN only.
- **Auto** recalibrated on the RECEIVED rate: full at ≤ 0.30 bpp/frame, none at ≥ 0.70 (v0.1: 0.20 / 0.50). The PS5 delivers 45-65 % of the requested rate (10.8 Mbps average on a 45 Mbps request = 0.17 bpp at 1080p30), so every real 1080p session is at full strength.
- **On** = full strength on the home LAN and away (v0.1 still scaled it by bitrate).
- `ClarityVersion` 0.2-beta. HUD guard rects exist in the plan (`ClarityController.hudRects`, max 2, dilated by 2 px), but nothing feeds them yet.

### Footage

- Original: 22 s 1080p30 HDR (HEVC Main10, PQ, Left siting, 18.6 Mbps, 660 frames). Local re-encodes at 14.6 and 10.1 Mbps (LAN) and 7.5 Mbps (WAN preset).
- 720p: a 40 Mbps 720p downscale as reference and a 4.7 Mbps 720p re-encode (WAN preset).
- 1080p60 drive (minimap, nav card, signage, car, foliage): a recorded session.
- Quality on every 60th frame (11 frames, +1 each for flicker). The re-encodes are test copies only, not in the repository.

### Step 1: colour-domain A/B (recorded first)

The original clip downscaled 2× in linear light, upscaled back through each chain, Y / RGB PSNR in PQ codes vs the original, 11 frames:

| Domain | Y all | Y edges | RGB all | Chain GPU vs PQ (with present), p50 |
|---|---|---|---|---|
| PQ R'G'B' rgb10a2, MetalFX `.perceptual` (today) | 39.67 | 33.49 | 39.63 | |
| Linear FP16, MetalFX `.hdr`, CAS on linear, present = tone map + deband | 39.95 | 33.89 | 39.93 | +0.97 to +1.21 ms |

- Linear is +0.28 to +0.40 dB, but the test is biased toward it (the 2× downscale was made in linear light).
- The linear conversion itself costs only +0.02 to +0.06 ms p50; the ~1 ms is MetalFX `.hdr` on FP16 plus the doubled bandwidth.
- **Decision: keep PQ `.perceptual`.** About 1 ms is two thirds of the whole budget, for a gain not visible in the crops (v0.2 vs v0.2-linear differ only on the edges of high-contrast wires and signs, at ×8 amplification).

### Step 2: mid-band enhancer A/Bs

Y PSNR all / textured / flat (dB) | 5×5-band detail energy vs the original, correlation | added flicker RMS (codes).

1080p 10.1 Mbps, LAN Auto (0.163 bpp):

| Config | Y PSNR | Detail energy, corr | Flicker |
|---|---|---|---|
| Off | 43.28 / 46.09 / 56.26 | 0.962, 0.889 | 9.013 |
| v0.1 | 45.35 / 46.51 / 56.26 | 0.909, 0.902 | 8.165 |
| **v0.2 (shipped: 13-tap, gFine 0.15, gMid 1.0, cap 8, T 16)** | **45.60 / 46.41 / 56.61** | **0.965, 0.901** | **8.138** |
| v0.2 + structure tensor | 45.60 / 46.42 / 56.63 | 0.963, 0.901 | 8.135 |
| v0.2 at spec gains (gMid 0.55, cap 5) | 45.64 / 46.49 / 56.69 | 0.935, 0.902 | 8.094 |
| 25-tap [1,2,2,2,1]² B2 (gFine 0.08, gMid 1.0, cap 6) | 45.52 / 46.24 / 56.38 | 1.005, 0.899 | 8.234 |

1080p 14.6 Mbps, LAN Auto (0.235 bpp):

| Config | Y PSNR | Detail energy, corr | Flicker |
|---|---|---|---|
| Off | 44.13 / 48.27 / 57.42 | 0.977, 0.928 | 7.400 |
| v0.1 | 47.05 / 48.86 / 57.42 | 0.928, 0.940 | 6.307 |
| **v0.2** | **47.56 / 48.74 / 57.85** | **0.987, 0.939** | **6.217** |
| v0.2 at spec gains | 47.61 / 48.85 / 57.92 | 0.956, 0.941 | 6.170 |
| 25-tap B2 | 47.44 / 48.48 / 57.59 | 1.027, 0.937 | 6.326 |

1080p 7.5 Mbps, WAN Auto (0.121 bpp; preset gMid 1.0, cap 6, T 11):

| Config | Y PSNR | Detail energy, corr | Flicker |
|---|---|---|---|
| Off | 42.11 / 44.02 / 54.77 | 0.934, 0.839 | 11.044 |
| v0.1 | 43.44 / 44.35 / 55.09 | 0.863, 0.853 | 10.393 |
| **v0.2** | **43.58 / 44.32 / 55.08** | **0.889, 0.853** | **10.363** |
| v0.2 at spec gains | 43.60 / 44.35 / 55.17 | 0.872, 0.854 | 10.341 |
| 25-tap B2 | 43.56 / 44.27 / 54.93 | 0.910, 0.852 | 10.403 |

720p 4.7 Mbps, WAN Auto (0.171 bpp; preset gMid 0.9, cap 5, T 11):

| Config | Y PSNR | Detail energy, corr | Flicker |
|---|---|---|---|
| Off | 39.85 / 43.57 / 54.10 | 1.042, 0.827 | 11.608 |
| v0.1 | 41.91 / 44.18 / 54.49 | 0.928, 0.849 | 10.442 |
| **v0.2** | **42.09 / 44.15 / 54.49** | **0.950, 0.850** | **10.402** |
| v0.2 at spec gains | 42.11 / 44.19 / 54.60 | 0.933, 0.850 | 10.372 |
| 25-tap B2 | 42.08 / 44.11 / 54.38 | 0.963, 0.849 | 10.428 |

Findings:
- **Spec gains are invisible.** gMid 0.55 / 0.30 / 0.25 moves luma by about 0.4 code RMS, and detail energy stays at v0.1's level. They score best on PSNR because they change least. The shipped gains are 2-3× higher and still beat v0.1 on overall PSNR, flat PSNR and flicker on every clip.
- **Structure tensor: rejected.** Within ±0.02 dB and ±0.003 flicker of the isotropic weights on every clip. Not worth the ALU.
- **25-tap B2 (the spec's "if still subtle" step): rejected.** More detail energy (+0.04), but −0.08 to −0.17 dB, more flicker, and +0.10 to +0.34 ms p50 compared with the 13-tap B2 at higher gains.
- **Tile vs direct reads:** with the 25-tap B2 the 16×16 tile was 0.12-0.16 ms faster; with the shipped 13-tap B2, direct reads are 0.02-0.06 ms faster at p50. Direct reads ship.
- **Range weight:** `saturate(2 − |d|/T)` replaces `1 − smoothstep(T, 2T, |d|)` (same end points). The half-precision rewrite of the stage cut the convert kernel from 2.16 ms to 1.19 ms in the first build.
- **Flat cleanup:** flat-pixel PSNR +0.35 to +0.43 dB over v0.1 at 10.1 and 14.6 Mbps; ±0 at the WAN rates (the de-ringer already ran there in v0.1).
- Correctness (`--selfcheck`): direct vs tile 0.0001 code, direct vs plain-read 0.0001, GPU vs the CPU reference (`ClarityMath.midband`) 0.005 code max. Luma change on the original clip: RMS 0.51, p99 1.76, max 4.27 codes.

### Guided chroma cap above 0.65 (rejected)

CbCr PSNR (Left siting, all / edges); the cap is reached as edge confidence goes from 0.75 to 1:

| Clip | 2× ref, cap 0.65 | 0.85 | 1.0 | 4× ref, cap 0.65 | 0.85 | 1.0 |
|---|---|---|---|---|---|---|
| 10.1 Mbps | **61.84 / 51.00** | 61.71 / 50.56 | 61.56 / 50.14 | **57.38 / 49.21** | 57.32 / 49.05 | 57.22 / 48.83 |
| 14.6 Mbps | **61.14 / 51.00** | 61.02 / 50.61 | 60.89 / 50.22 | **56.88 / 49.17** | 56.83 / 49.03 | 56.74 / 48.83 |
| 7.5 Mbps | **62.54 / 50.97** | 62.39 / 50.51 | 62.23 / 50.06 | **58.00 / 49.18** | 57.93 / 48.99 | 57.81 / 48.75 |
| 720p 4.7 Mbps | **60.30 / 49.84** | 60.16 / 49.47 | 60.00 / 49.09 | **56.14 / 48.24** | 56.09 / 48.11 | 55.99 / 47.91 |

Every step above 0.65 loses, most on chroma edges (−0.4 to −0.9 dB). `guideMaxHigh` stays in the code at 0.65 (= v0.1).

### Step 3: CAS limiter and gain

The 5-tap RCAS limiter costs nothing measurable vs v0.1's CAS (paired p50 −0.003 to +0.004 ms). Its effect is included in the v0.2 rows above (CAS ×0.7 plus the limiter). CAS ×0.7 keeps edges from being boosted twice with Stage 4; the crops show no visible softening of text, wires or sign borders.

### Step 4: Motion-flow gain stabilisation (not done)

The spec makes it conditional on Stage 4 shimmering. Added flicker is equal to or lower than v0.1 on every clip (above), so it was not built.

### GPU time

`--batch 10` (10 back-to-back dispatches per command buffer, GPU time / 10), 3 runs × 150 frames, output 2560×1440, CAS 0.25, ms p50 / p95. Totals are paired per frame: (convert − legacy convert) + (CAS − legacy CAS).

| Run | legacy convert | v0.2 convert | v0.2 CAS vs legacy CAS | **Clarity v0.2 total** | v0.1 total |
|---|---|---|---|---|---|
| 1080p 18.6 Mbps + Motion | 0.172 / 0.393 | 0.914 / 1.252 | 0.096 / 0.445 | **0.883 / 1.225** | 0.268 / 0.588 |
| 1080p 10.1 Mbps + Motion | 0.169 / 0.356 | 0.934 / 1.252 | 0.081 / 0.398 | **0.873 / 1.195** | 0.251 / 0.603 |
| 1080p 18.6 Mbps, no Motion | 0.171 / 0.237 | 1.069 / 1.364 | 0.127 / 0.225 | **1.007 / 1.336** | 0.348 / 0.508 |
| 720p 4.7 Mbps WAN + Motion | 0.057 / 0.120 | 0.394 / 0.532 | 0.071 / 0.238 | **0.419 / 0.596** | 0.253 / 0.412 |

Whole chain (convert → deblock → MetalFX → CAS, command buffer, shuffled per frame), p95(v0.2) − p95(legacy) / p50 delta: +0.84 / +0.83 ms (1080p + Motion), +0.87 / +0.81 ms (10.1 Mbps + Motion), +1.27 / +1.04 ms (no Motion), +0.70 / +0.40 ms (720p). Paired per-frame chain p95 deltas are 1.6-5.8 ms with Motion: they measure Motion's own work landing between chains, not Clarity.

Variants vs shipped, paired p50 / p95 (1080p + Motion): 25-tap B2 +0.28 / +0.62, 16×16 tile +0.04 / +0.42, linear FP16 output +0.03 / +0.41.

Notes:
- **Gate 1.50 ms p95 with Motion: pass** in every final run (worst 1.23 ms). One run of the same build made while crop images were being processed on the same machine measured 1.93 ms; a re-run on a quiet machine gave 1.20 ms. The machine is shared (browser, WindowServer), so single runs swing by ±0.3 ms.
- Without Motion the GPU runs at a lower clock and every kernel, legacy included, is slower. Totals there were 1.0-1.3 ms p95 in quiet runs, and up to 2.2 ms in one busy run before the switch to the 13-tap B2.
- Stage-boundary timestamps (v0.1's method) overstated the linear variant by 2.5 ms; batched dispatch is the method from v0.2 on.

### Visual notes

Crops of the 2560×1440 output, 1:1 and 2×, columns Off / v0.1 / v0.2 / (v0.2 − v0.1) ×8. Scratch files, not in the repository.

| Region | Verdict |
|---|---|
| Foliage (palms, bushes, 10.1 Mbps) | v0.1 is visibly softer than Off. v0.2 brings leaf contrast back to about Off's level without Off's grain. Modest but visible at 2×, hard to see at 1:1. |
| Asphalt (18.6 Mbps) | Slightly crisper grain than v0.1; the diff is texture-shaped, with no pattern. Subtle. On dark asphalt (10.1 Mbps) no visible change (tone guard). |
| Distant buildings, palms, sky | Same detail as Off. Sky and flat walls slightly cleaner (fewer faint blocks). |
| Car reflections, bright paint | Identical; the diff is noise-level only. No halo on the highlight. |
| HUD text (nav card, "1.23 km"), signage | Unchanged to the eye: no halo, no thickening. The ×8 diff shows small changes inside the nav card's arrow tile, because live HUD rects are empty. |
| Minimap (pink route, labels) | Unchanged. |
| Dark shadows and gradients | Slightly smoother blotches; most blocks remain. The tone guard (< 110 codes) keeps Stage 4 out of the darkest areas, and flat clean is limited to ±1.5 codes. |
| 720p WAN 4.7 Mbps | No difference visible to the eye; the numbers are slightly better than v0.1. |

Overall: a real but small step. The most visible change is that v0.2 no longer looks softer than Off on foliage, while keeping v0.1's cleaner flats.

### Rejected or deferred in v0.2

- Linear FP16 / MetalFX `.hdr` colour domain: +0.3-0.4 dB (biased test) for about 1 ms.
- Structure-tensor weights: no gain.
- 25-tap B2: more energy, lower PSNR, more flicker, more cost.
- Guided chroma cap above 0.65: lower chroma PSNR at every cap on every clip.
- 16×16 tile: slower than direct reads with the 13-tap B2.
- Motion-flow gain stabilisation (step 4): not needed (no added flicker).

### Still to do

- Live A/B on real LAN and WAN sessions with a perf trace, before the beta tag comes off.
- Feed `ClarityController.hudRects` from real HUD positions (minimap, nav card).
- Shadows: blocks below 110 codes are mostly left alone; a shadow-specific deblock would need its own gate.

## Roadmap

- **v0.3-beta**: last pure-Metal pass; live A/B, then drop the beta tag.
- **Clarity v1.0 Path B, after the trip:** a learned luma residual on the Neural Engine. `ClarityLuma1x-Packed8` is a CoreML model of about 1,172 parameters that runs on the ANE and predicts a small luma residual at source resolution from packed 8-pixel inputs, replacing the hand-tuned Stages 3 and 4. Targets: the same zero-latency slot, GPU time near v0.1's (the ANE does the work), trained and validated on PS5 captures at the received bitrates measured above.

---

## v0.1-beta: first engine

### GPU time

Whole chain, command buffer, ms:

| Run | Legacy p50 / p95 | Clarity 1+2 p50 / p95 | Clarity 1+2+3 p50 / p95 | Δ 1+2 p50 / p95 | Δ 1+2+3 p50 / p95 |
|---|---|---|---|---|---|
| 10.1 Mbps | 4.00 / 6.24 | 4.50 / 6.84 | 4.73 / 7.16 | +0.50 / +0.60 | +0.73 / +0.92 |
| 10.1 Mbps, Motion load | 3.30 / 9.69 | 3.91 / 9.73 | 3.96 / 9.02 | +0.60 / +0.05 | +0.66 / −0.66 |
| 18.6 Mbps | 3.15 / 4.22 | 3.51 / 4.56 | 3.72 / 4.56 | +0.36 / +0.33 | +0.57 / +0.34 |
| 18.6 Mbps, Motion load | 2.45 / 5.47 | 2.71 / 5.84 | 2.89 / 5.64 | +0.26 / +0.37 | +0.44 / +0.17 |

Per kernel, increments paired per frame, ms p50 / p95:

| Run | (a) siting | Stage 1 (b) | Stage 2 (c) | Stage 3 (d − b) | 1+2 | 1+2+3 |
|---|---|---|---|---|---|---|
| 10.1 Mbps | 0.01 / 0.50 | 0.67 / 1.60 | 0.18 / 0.79 | 0.50 / 1.75 | 0.82 / 1.98 | 1.36 / 2.85 |
| 10.1 Mbps, Motion load | 0.01 / 0.53 | 0.53 / 1.58 | 0.13 / 0.63 | 0.45 / 1.65 | 0.71 / 1.83 | 1.15 / 2.59 |
| 18.6 Mbps | 0.01 / 0.44 | 0.58 / 1.17 | 0.15 / 0.46 | 0.46 / 1.18 | 0.71 / 1.51 | 1.13 / 1.99 |
| 18.6 Mbps, Motion load | 0.05 / 0.27 | 0.39 / 0.84 | 0.08 / 0.39 | 0.37 / 0.89 | 0.49 / 1.07 | 0.82 / 1.51 |

Notes:
- The per-kernel p95 includes timestamp jitter: step (a) does the same work as legacy (p50 +0.01 ms) but still shows a p95 of +0.3 to +0.5 ms. On a shared GPU the stage-boundary p95 overstates the cost; the whole-chain delta is the number that reaches the frame time.
- Absolute chain times move between runs with background load (legacy p50 2.45 to 4.00 ms). Deltas are paired within a run.
- The gate is 1.0 ms p95 incremental. On the whole chain, Stages 1+2 pass in every run (worst +0.60 ms) and Stages 1+2+3 pass too (worst +0.92 ms, at 10.1 Mbps). On per-kernel p95, Stage 1 alone is over budget in the busiest runs. Stage 3 costs about 0.4-0.5 ms p50 for a small quality gain, so it stays gated (Auto: WAN only, below 0.50 bpp).
- Stage 1 is the main cost: four bilinear luma reads and two chroma gathers per pixel at stream resolution, about 2.5× the legacy kernel.

Rejected kernels (slower than the per-pixel kernel with gathers):
- **2×2 quad per thread:** one thread per 2×2 luma quad, sharing the chroma footprint. Fewer threads, but more registers and worse occupancy.
- **16×16 threadgroup tile:** chroma and luma staged in threadgroup memory. The barrier cost outweighed the saved reads.

### Correctness (`--selfcheck`)

- Production kernel vs plain-read reference kernel: max difference 0.0001 code.
- GPU vs CPU reference (`ClarityMath`): chroma 0.000 code, de-ringed luma 0.000 code, for all six sitings.

### Chroma reconstruction (CbCr PSNR dB, all / edges)

2× reference (960×540, chroma plane as is):

| Source siting / method | 10.1 Mbps | 14.6 Mbps | 18.6 Mbps |
|---|---|---|---|
| Left / legacy | 61.73 / 50.73 | 61.01 / 50.57 | 59.72 / 50.29 |
| Left / (a) siting bilinear | 61.73 / 50.73 | 61.01 / 50.57 | 59.72 / 50.29 |
| Left / (b) guided | **61.93 / 50.98** | **61.25 / 50.92** | **60.03 / 50.76** |
| Center / legacy | 60.57 / 49.01 | 59.93 / 48.95 | 58.72 / 48.83 |
| Center / (a) | 61.91 / 50.89 | 61.21 / 50.72 | 59.85 / 50.39 |
| Center / (b) | **62.18 / 51.20** | **61.53 / 51.15** | **60.25 / 50.93** |
| TopLeft / legacy | 59.52 / 48.10 | 58.91 / 48.10 | 57.80 / 48.06 |
| TopLeft / (a) | 61.46 / 50.41 | 60.72 / 50.28 | 59.51 / 50.09 |
| TopLeft / (b) | **61.58 / 50.63** | **60.87 / 50.57** | **59.72 / 50.50** |

4× reference (480×270, chroma as sharp as luma):

| Source siting / method | 10.1 Mbps | 14.6 Mbps | 18.6 Mbps |
|---|---|---|---|
| Left / legacy = (a) | 57.04 / 48.49 | 56.51 / 48.39 | 55.59 / 48.16 |
| Left / (b) guided | **57.51 / 49.13** | **57.00 / 49.07** | **56.10 / 48.87** |
| Center / legacy | 56.31 / 47.52 | 55.83 / 47.46 | 54.95 / 47.31 |
| Center / (b) | **57.79 / 49.34** | **57.27 / 49.27** | **56.34 / 49.04** |
| TopLeft / legacy | 55.34 / 46.75 | 54.90 / 46.73 | 54.12 / 46.65 |
| TopLeft / (b) | **57.22 / 49.08** | **56.72 / 49.00** | **55.85 / 48.81** |

- On the PS5's own Left siting, (a) is the same picture as legacy, as expected. Guided chroma adds +0.2 to +0.5 dB overall and +0.3 to +0.7 dB on chroma edges.
- On Center or TopLeft content, honouring the siting alone gains +1.1 to +2.0 dB, and guiding adds up to +0.6 dB more.

### De-ringer and pipeline (Y PSNR dB vs the 18.6 Mbps decode: all / edge band / flat)

| | 10.1 Mbps | 14.6 Mbps |
|---|---|---|
| Plan (Auto, WAN) | CAS noise 1.00, de-ring 0.40 | CAS noise 0.88, de-ring 0.35 |
| De-ringer change, RMS / max | 0.26 / 1.60 codes | 0.24 / 1.41 codes |
| Raw luma, de-ringer off | 45.83 / 43.67 / 56.36 | 48.50 / 46.34 / 57.63 |
| Raw luma, de-ringer plan | 45.85 / 43.67 / **56.71** | 48.52 / 46.36 / **58.02** |
| Fidelity, step L (legacy) | 42.36 / 41.14 / 55.02 | 43.50 / 42.83 / 56.29 |
| Fidelity, step b | 42.36 / 41.15 / 55.02 | 43.51 / 42.84 / 56.29 |
| Fidelity, step c | 43.84 / 42.23 / 55.02 | 45.82 / 44.80 / 56.29 |
| Fidelity, step d | **43.86 / 42.24 / 55.37** | **45.84 / 44.82 / 56.68** |
| Visibility, step L (legacy) | 41.71 / 40.35 / 55.02 | 44.08 / 42.64 / 56.29 |
| Visibility, step c | 43.25 / 41.99 / 55.02 | 45.73 / 44.56 / 56.29 |
| Visibility, step d | **43.32 / 42.06 / 55.74** | **45.81 / 44.64 / 57.08** |

- **Stage 2** is the largest measured gain: +1.5 dB at 10.1 Mbps and +1.7 dB at 14.6 Mbps in how much of the compression damage stays visible after sharpening. Part of the fidelity gain is simply less sharpening of noise on faint patches; the visibility metric controls for that.
- **Stage 3** is nearly neutral overall (+0.02 dB) and helps flat areas by +0.35 to +0.8 dB. It moves luma by 0.26 codes RMS and never more than 2 codes, so it cannot create artefacts, but it also does little.
- The original 18.6 Mbps file gets an Auto plan of CAS noise 0.67 and de-ring 0.27 (0.30 bpp/frame), with a de-ringer change of 0.17 codes RMS.

### Visual notes (crops at 1080p, columns legacy, a, b, c, d, original)

- HUD text ("1.35 km"), the pink minimap route, signage text, the red reel against a yellow field, sky with power lines, a road sign.
- At 1:1 and at 3× the differences between legacy and Clarity are small. The pink route and the red reel edge are marginally tighter with guided chroma; there is no colour bleed, halo, hue shift or new ringing in any crop.
- Stage 2 does not add visible softness on real edges (text, wires, sign borders keep their contrast).
- The gap to the original-bitrate picture is mostly lost texture (field, signage detail), which no post filter can restore.

### Still to do

- Live A/B on real WAN and LAN sessions, with a perf trace, before the beta tag comes off.
- Re-measure on a quiet GPU: the per-kernel p95 numbers above include jitter from other GPU work.
