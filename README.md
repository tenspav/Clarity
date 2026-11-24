# Clarity

Clarity makes compressed game streams look closer to the real thing. It cleans up what video compression leaves behind (smeared colour edges, faint ripples next to sharp lines, blocky flat areas) and brings back some of the fine texture the encoder threw away. It runs inside passes a video renderer already has (no extra pass, no extra texture, no Neural Engine) and adds **no extra frame of delay**.

It works on decoded YCbCr video frames as they come out of the hardware decoder, 8-bit or 10-bit HDR. It needs nothing beyond Apple's own frameworks (Metal and Core Video).

**Current version: v1.0** (final pure-Metal release). It was built for low-latency PS5 Remote Play streaming in the FramePortal app. Its sister project [Motion](https://github.com/tenspav/Motion) does the frame generation.

## What it fixes

- **Colour bleeding past edges.** Video stores colour at a quarter of the resolution of brightness. A plain decoder blurs it back up, so red tail lights, coloured map routes and HUD text bleed a pixel or two. Clarity rebuilds the colour so it follows the edges in the picture.
- **Texture lost at low bitrates.** Leaves, asphalt and brickwork turn to mush when the stream drops to 10-15 Mbps. Clarity lifts that mid-sized detail back where the picture is already textured, and leaves edges, text, deep shadows and bright highlights alone.
- **Ripples and blocks.** Faint ringing next to edges and small blocks in flat areas (sky, walls, shadows) are eased toward their neighbours.
- **Sharpening that makes noise worse.** Clarity's sharpening steps back on noisy patches and keeps full strength on real edges. In HDR it never tints bright highlights.

It adds no film grain and no learned or invented detail. Its luma changes are clamped to the range of each pixel's 3×3 neighbourhood and its sharpening to the local luma range, which keeps halos and overshoot small.

## How well it works

On real 1080p HDR gameplay recordings from a PS5 stream, compared with the original high-bitrate picture (higher is better):

| Stream | Picture quality, plain decode | With Clarity | Frame-to-frame flicker, plain → Clarity |
|---|---|---|---|
| 1080p, 10 Mbps | 43.3 dB | **46.0 dB** | 9.0 → **8.0** |
| 1080p, 15 Mbps | 44.1 dB | **48.2 dB** | 7.4 → **6.1** |
| 1080p, 7.5 Mbps (internet) | 42.1 dB | **43.8 dB** | 11.0 → **10.3** |
| 720p, 4.7 Mbps (internet) | 39.9 dB | **42.4 dB** | 11.6 → **10.3** |

GPU time Clarity adds in FramePortal's real render path (MacBook Air M2, 1080p HDR source, MetalFX upscale to 1440p, Motion frame generation running, median / 95th percentile, measured against the same path without Clarity):

| | Added time |
|---|---|
| 1080p, low or medium bitrate (full or partial strength) | **about 0.76 / 0.9-1.0 ms** |
| 1080p, clean high-bitrate stream (Auto strength 0) | **about 0** |

The quality table above was measured with v0.3's stages, which v1.0 keeps unchanged at full strength. What v1.0 adds is in [CLARITY_PROGRESS.md](CLARITY_PROGRESS.md) › v1.0: a smooth Auto strength, a clean-stream bypass, one plan per generated frame pair, and a faster conversion kernel.

The full design is in [CLARITY.md](CLARITY.md). Every experiment, with its numbers, is in [CLARITY_PROGRESS.md](CLARITY_PROGRESS.md).

## Modes

- **Auto** (recommended): one strength from 0 to 1, set by the bitrate actually received. At 1080p, full strength at or below about 7.8 Mbps (30 FPS source), easing smoothly to zero at about 18.7 Mbps; it warms up over the first second. At zero, a clean stream gets only the colour fix: no luma cleanup and no sharpening.
- **On**: full strength always.
- **Off**: the plain conversion, for comparison.

If your renderer has its own 8×8 RGB deblocking pass, keep it: Clarity v1.0 does not replace it (a fused source-luma alternative was built and measured, and did not beat it).

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
