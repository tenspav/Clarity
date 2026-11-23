# Clarity

Clarity makes compressed game streams look closer to the real thing. It cleans up what video compression leaves behind (smeared colour edges, faint ripples next to sharp lines, blocky flat areas) and brings back some of the fine texture the encoder threw away. It runs on the GPU in well under a millisecond per frame and adds **no extra frame of delay**.

It works on decoded YCbCr video frames as they come out of the hardware decoder, 8-bit or 10-bit HDR. It needs nothing beyond Apple's own frameworks (Metal and Core Video).

**Current version: v0.2-beta.** It was built for low-latency PS5 Remote Play streaming in the FramePortal app. Its sister project [Motion](https://github.com/tenspav/Motion) does the frame generation.

## What it fixes

- **Colour bleeding past edges.** Video stores colour at a quarter of the resolution of brightness. A plain decoder blurs it back up, so red tail lights, coloured map routes and HUD text bleed a pixel or two. Clarity rebuilds the colour so it follows the edges in the picture.
- **Texture lost at low bitrates.** Leaves, asphalt and brickwork turn to mush when the stream drops to 10-15 Mbps. Clarity lifts that mid-sized detail back where the picture is already textured, and leaves edges, text, deep shadows and bright highlights alone.
- **Ripples and blocks.** Faint ringing next to edges and small blocks in flat areas (sky, walls, shadows) are eased toward their neighbours.
- **Sharpening that makes noise worse.** Clarity's sharpening steps back on noisy patches and keeps full strength on real edges. In HDR it never tints bright highlights.

It never adds film grain, never invents detail with AI, and never pushes a pixel outside the range of its neighbours, so it cannot create halos.

## How well it works

On real 1080p HDR gameplay recordings from a PS5 stream, compared with the original high-bitrate picture (higher is better):

| Stream | Picture quality, plain decode | With Clarity | Frame-to-frame flicker, plain → Clarity |
|---|---|---|---|
| 1080p, 10 Mbps | 43.3 dB | **45.6 dB** | 9.0 → **8.1** |
| 1080p, 15 Mbps | 44.1 dB | **47.6 dB** | 7.4 → **6.2** |
| 1080p, 7.5 Mbps (internet) | 42.1 dB | **43.6 dB** | 11.0 → **10.4** |
| 720p, 4.7 Mbps (internet) | 39.9 dB | **42.1 dB** | 11.6 → **10.4** |

GPU time Clarity adds, MacBook Air M2 (median / 95th percentile):

| | Added time |
|---|---|
| 1080p, with Motion frame generation and upscaling to 1440p running | **0.88 / 1.23 ms** |
| 720p, same | **0.42 / 0.60 ms** |

The full design is in [CLARITY.md](CLARITY.md). Every experiment, with its numbers, is in [CLARITY_PROGRESS.md](CLARITY_PROGRESS.md).

## Modes

- **Auto** (recommended): strength follows the bitrate actually received. Low-bitrate streams get full strength; a clean, high-bitrate stream gets only the colour fix.
- **On**: full strength always.
- **Off**: the plain conversion, for comparison.

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
var meter = ClarityBitrateMeter()          // feeds Auto mode

// Per decoded frame `pb` (CVPixelBuffer) of `bytes` encoded size:
meter.add(frameBytes: bytes, at: CACurrentMediaTime())
let bpp = meter.bpp(width: width, height: height)
let plan = ClarityPlan.make(mode: .auto, wan: false, bpp: bpp, height: height)

guard let planes = clarity.planes(pb) else { return }        // Y and CbCr textures, zero copy
let params = ClarityYUVParams(YUVCoefficients.make(matrix: matrixAttachment, fullRange: false, tenBit: planes.tenBit),
                              siting: ChromaSiting(buffer: pb), tenBit: planes.tenBit, plan: plan)

let enc = commandBuffer.makeComputeCommandEncoder()!
// 1. Conversion: writes R'G'B' (same transfer function as the source) into `rgb`, the size of the luma plane.
try clarity.encodeConvert(enc, y: planes.y, c: planes.c, dst: rgb, params: params, variant: ClarityEngine.variant(for: plan))
// ... your upscaler here, if any ...
// 2. Sharpening, at the output size.
clarity.encodeCAS(enc, src: upscaled, dst: output,
                  params: ClarityCASParams(sharpness: 0.25 * plan.casGain, noise: plan.casNoise, pq: isHDR, limiter: plan.casLimiter))
enc.endEncoding()
// Keep `planes.refs` alive until the command buffer completes.
```

`wan: true` picks gentler presets for streams over the internet. If your renderer already has its own Metal library, compile `ClarityShaders.source` into it and pass that library to `ClarityEngine(device:library:)`. `ClarityController` wraps the mode, meter and plan in one thread-safe object.

## Benchmark it yourself

```bash
swift run -c release clarity-bench your-clip.mp4 --count 300 --png out --frame 60
```

This times the plain conversion and sharpening against Clarity's on any recording, and writes before / after images of frame 60. Add `--mode auto --bpp 0.2` to try Auto at a given bitrate, `--wan` for the internet presets, and `--8bit` to decode as 8-bit. On a MacBook Air M2 with a 1080p HDR clip it reports about 0.72 / 0.83 ms added (sharpening at 1080p here; the table above sharpens at 1440p).

## License

MIT. See [LICENSE](LICENSE).
