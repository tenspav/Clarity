<p align="center">
  <img src="assets/icon.svg" width="160" alt="Clarity icon">
</p>

<h1 align="center">Clarity</h1>

<p align="center">
  <b>Cleaner, sharper game streams on Apple silicon: pure Metal, zero added frames of delay.</b><br>
  Fixes what video compression leaves behind, inside passes your renderer already has.
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/version-v1.0-2ea44f">
  <img alt="platform" src="https://img.shields.io/badge/platform-macOS%2026%20%7C%20iOS%2026-lightgrey">
  <img alt="engine" src="https://img.shields.io/badge/engine-pure%20Metal-orange">
  <img alt="neural engine" src="https://img.shields.io/badge/Neural%20Engine-0%25-blueviolet">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

---

## At a glance

| | Clarity v1.0 |
|---|---|
| ⚡ Conversion kernel (1080p in, 1440p out, M2) | **0.766 ms** p95 (v0.3: 0.891 ms, **−13 %**) |
| 🎮 Added to the full frame, full strength | **+0.76 ms** p50 / **+0.98 ms** p95 (budget 1.50 ms) |
| 🌊 Clean high-bitrate stream (25 Mbps, Auto) | **0 ms**: bypass, colour fix only |
| 🧠 Neural Engine / trained model | **0 %** / none: hand-designed, pure Metal |
| ⏱️ Extra frames of delay | **0**: no extra pass, no extra texture |
| 📱 Runs on | Apple silicon Mac (macOS 26), iPhone / iPad (iOS 26) |
| 🎞️ Input | Decoded YCbCr straight from the hardware decoder, 8-bit or 10-bit HDR |

> **v1.0 milestone (final pure-Metal release):** a smooth Auto strength driven by the bitrate actually received, a clean-stream bypass, one plan per generated frame pair (real and generated frames always match), and a two-pixel conversion kernel that is 0.12 ms faster. Every quality gate still passes (GPU vs CPU reference 0.0000 on all six chroma sitings; seam score 0.447 vs 0.448 in v0.3). Built for low-latency PS5 Remote Play in FramePortal; frame generation is its sister project [Motion](https://github.com/tenspav/Motion).

## Picture quality

Real 1080p HDR gameplay from a PS5 stream, compared with the original high-bitrate picture (higher is better):

| Stream | Plain decode | With Clarity | Flicker, plain → Clarity |
|---|---|---|---|
| 1080p, 15 Mbps | 44.1 dB | **48.2 dB** (+4.1) | 7.4 → **6.1** |
| 1080p, 10 Mbps | 43.3 dB | **46.0 dB** (+2.7) | 9.0 → **8.0** |
| 1080p, 7.5 Mbps (internet) | 42.1 dB | **43.8 dB** (+1.7) | 11.0 → **10.3** |
| 720p, 4.7 Mbps (internet) | 39.9 dB | **42.4 dB** (+2.5) | 11.6 → **10.3** |

> Quality was measured with v0.3's stages, which v1.0 keeps unchanged at full strength; v1.0's two-pixel kernel differs only by float rounding (at most 1.2e-6, a 10-bit step is 9.8e-4).

## Speed (MacBook Air M2)

GPU time in FramePortal's real render path: 1080p HDR source, MetalFX upscale to 1440p, Motion frame generation running, measured against the same path without Clarity.

| Kernel / path | p50 | p95 |
|---|---|---|
| Conversion, one pixel per thread (v0.3) | 0.879 ms | 0.891 ms |
| **Conversion, two pixels per thread (v1.0)** | **0.764 ms** | **0.766 ms** |
| Clarity sharpening (plain sharpening: 0.359 ms) | 0.460 ms | 0.474 ms |
| **Whole frame, full strength, added** | **+0.76 ms** | **+0.98 ms** |
| Whole frame, Auto at 25 Mbps (bypass) | **0 ms** | **0 ms** |

## What it fixes

| Problem | What Clarity does |
|---|---|
| 🎨 Colour bleeding past edges (tail lights, map routes, HUD text) | Rebuilds quarter-resolution colour so it follows the edges in the picture |
| 🍃 Texture lost at low bitrates (leaves, asphalt, brick) | Lifts mid-sized detail where the picture is already textured; leaves edges, text, shadows and highlights alone |
| 🧱 Ripples and blocks in flat areas (sky, walls) | Eases them toward their neighbours |
| ✨ Sharpening that amplifies noise | Steps back on noisy patches, full strength on real edges; never tints HDR highlights |

No film grain, no learned or invented detail: luma changes are clamped to each pixel's 3×3 neighbourhood, sharpening to the local luma range.

## Modes

| Mode | Behaviour |
|---|---|
| **Auto** (recommended) | Strength 0 to 1 from the bitrate actually received. At 1080p30: full at ≤ ~7.8 Mbps, easing to 0 at ~18.7 Mbps; warms up over the first second. At 0, a clean stream gets only the colour fix. |
| **On** | Full strength always. |
| **Off** | Plain conversion, for comparison. |

If your renderer has its own 8×8 RGB deblocking pass, keep it: Clarity does not replace it.

Full design: [CLARITY.md](CLARITY.md). Every experiment with its numbers: [CLARITY_PROGRESS.md](CLARITY_PROGRESS.md).

## Requirements

- Apple silicon Mac with macOS 26, or iPhone / iPad with iOS 26
- Swift 6 toolchain (Xcode 26)
- Frames as bi-planar YCbCr `CVPixelBuffer`s (`420v`, `420f`, `x420`, `xf20`), IOSurface-backed and Metal-compatible. Video decoders give you these.

## Use it

Add the package (`https://github.com/tenspav/Clarity`), or copy the four files in [`Sources/Clarity`](Sources/Clarity) into your project.

Clarity replaces two steps a video renderer usually already has: the YCbCr → RGB conversion, and the sharpening pass.

```swift
import Clarity
import Metal

let device = MTLCreateSystemDefaultDevice()!
let clarity = try ClarityEngine(device: device)
clarity.prewarm()                           // compile the shipped pipelines off the render thread
let control = ClarityController(mode: .auto)
control.begin(wan: false, width: 1920, height: 1080)

// Network thread, per received video access unit:
control.addVideo(bytes: auSize, at: CACurrentMediaTime())

// Per decoded frame `pb` (CVPixelBuffer):
control.setDimensions(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
let plan = control.plan                     // hold one plan for a frame and the frames generated from it
guard let planes = clarity.planes(pb) else { return }        // Y and CbCr textures, zero copy
let matrix = CVBufferCopyAttachment(pb, kCVImageBufferYCbCrMatrixKey, nil) as? String
let params = ClarityYUVParams(YUVCoefficients.make(matrix: matrix, fullRange: isFullRange, tenBit: planes.tenBit),
                              siting: ChromaSiting(buffer: pb), tenBit: planes.tenBit, plan: plan)

let enc = commandBuffer.makeComputeCommandEncoder()!
// 1. Conversion: R'G'B' (same transfer function as the source) into `rgb`, the size of the luma plane.
//    Use a ready pipeline on the render thread; fall back to the guided-only one while it compiles.
if let pso = clarity.readyPSO(ClarityEngine.variant(for: plan)) ?? clarity.readyPSO(ClarityEngine.guidedOnly) {
    clarity.encodeConvert(enc, y: planes.y, c: planes.c, dst: rgb, params: params, pso: pso)
}
// ... your RGB deblock and upscaler here, if any (separate textures) ...
// 2. Sharpening at the output size, unless Auto is at strength 0.
if !plan.casBypass {
    clarity.encodeCAS(enc, src: upscaled, dst: output,
                      params: ClarityCASParams(sharpness: 0.25 * plan.casGain, noise: plan.casNoise, pq: isHDR,
                                               limiter: plan.casLimiter, correctionGain: plan.correctionGain))
}
enc.endEncoding()
commandBuffer.addCompletedHandler { _ in withExtendedLifetime(planes.refs) {} }   // keep the planes alive until done
```

`wan: true` picks gentler presets for streams over the internet. If your renderer already has its own Metal library, compile `ClarityShaders.source` into it and pass that library to `ClarityEngine(device:library:)`. `ClarityController` wraps the mode, meter and plan in one thread-safe object.

## Benchmark it yourself

```bash
swift run -c release clarity-bench your-clip.mp4 --count 300 --png out --frame 60
```

This times a plain conversion plus a baseline sharpening pass (the same CAS kernel with Clarity's gating off) against Clarity's conversion and sharpening on any recording, at the source resolution, and writes before / after images of frame 60. It prints the device, OS and plan. Add `--mode auto --bpp 0.2` to try Auto at a given bitrate, `--wan` for the internet presets, and `--8bit` to decode as 8-bit. On a MacBook Air M2 with a 1080p HDR clip (300 frames, nothing else running) it reports about 0.72 / 0.91 ms added at full strength and about −0.2 ms at Auto strength 0 (no sharpening at all). It does not run an upscaler, Motion or a present pass, so these numbers are not the in-app ones above.

## Datasets & Credits

Clarity v1.0 is hand-designed (no trained model). Its shipped benchmarks use recordings of PS5 streams captured for testing; none of them are included here. The third-party datasets below are credited as prior research on gaming and HDR video quality and are planned for evaluating future versions. None of their footage is included in this repository; get it from the sources below, under the terms set by their authors.

| Dataset | Authors / source | Link | Used for |
|---|---|---|---|
| GamingHDRVideoSET | Nabajeet Barman et al. | [github.com/NabajeetBarman/GamingHDRVideoSET](https://github.com/NabajeetBarman/GamingHDRVideoSET) | HDR gameplay video: prior research, planned evaluation (not used in the v1.0 numbers) |
| KUGVD (Kingston University Gaming Video Dataset) | Nabajeet Barman et al. | [KUGVD download](https://kingston.box.com/v/KUGVD) · [paper](https://eprints.kingston.ac.uk/id/eprint/43406/1/Barman-N-43406-VoR.pdf) | Gameplay at several bitrates: prior research, planned evaluation (not used in the v1.0 numbers) |
| Netflix Open Content, Chimera | Netflix, hosted by Xiph.org | [media.xiph.org/video/derf/Chimera/](https://media.xiph.org/video/derf/Chimera/) | 4K 10-bit reference video: prior research, planned evaluation (not used in the v1.0 numbers) |

Thanks to the authors for making this footage available to researchers. If you use Clarity's results in published work, please cite their papers too.

This section is kept up to date whenever footage used for training or benchmarks is added or changed.

## License

MIT. See [LICENSE](LICENSE).
