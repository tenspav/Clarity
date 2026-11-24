import Foundation
import AVFoundation
import CoreVideo
import ImageIO
import Metal
import UniformTypeIdentifiers
import Clarity

// clarity-bench: GPU cost of Clarity on a real recording, plus before/after PNGs.
//
// Per frame, two separate command buffers on the same decoded frame:
//   baseline   plain YCbCr -> R'G'B' conversion + plain CAS sharpen (what a renderer without Clarity does)
//   clarity    Clarity conversion (guided chroma, de-ringer, mid-band, flat cleanup) + Clarity CAS (limiter)
// Times: each kernel runs --batch (10) times back to back in one command buffer; GPU time / batch, at source resolution.
//
// usage: clarity-bench CLIP.mp4 [--count N] [--start S] [--mode on|auto] [--bpp X] [--wan] [--8bit]
//                              [--sharp S] [--batch B] [--png DIR] [--frame K]

let args = Array(CommandLine.arguments.dropFirst())
func opt(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }

guard let path = args.first, !path.hasPrefix("--") else {
    print("usage: clarity-bench CLIP.mp4 [--count N] [--start S] [--mode on|auto] [--bpp X] [--wan] [--8bit] [--sharp S] [--png DIR] [--frame K]")
    exit(1)
}
let count = Int(opt("--count") ?? "300") ?? 300
let start = Double(opt("--start") ?? "0") ?? 0
let mode: ClarityMode = opt("--mode") == "auto" ? .auto : .on
let bpp = opt("--bpp").flatMap(Double.init)
let wan = args.contains("--wan")
let sharp = Float(opt("--sharp") ?? "0.25") ?? 0.25
let pngDir = opt("--png")
let pngFrame = Int(opt("--frame") ?? "30") ?? 30
let pixelFormat = args.contains("--8bit") ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange

func percentile(_ v: [Double], _ p: Double) -> Double {
    guard !v.isEmpty else { return 0 }
    let s = v.sorted()
    return s[min(s.count - 1, Int(Double(s.count - 1) * p / 100 + 0.5))]
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { print("no Metal device"); exit(1) }
let engine = try ClarityEngine(device: device)

let asset = AVURLAsset(url: URL(fileURLWithPath: path))
guard let track = try await asset.loadTracks(withMediaType: .video).first, let reader = try? AVAssetReader(asset: asset) else {
    print("cannot open \(path)"); exit(1)
}
reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600), duration: .positiveInfinity)
let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
    kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    kCVPixelBufferMetalCompatibilityKey as String: true])
output.alwaysCopiesSampleData = false
reader.add(output)
guard reader.startReading() else { print("reader failed"); exit(1) }

/// Writes an 8-bit PNG preview of an rgb10a2Unorm R'G'B' texture. PQ frames get a simple SDR tone map.
func writePNG(_ t: MTLTexture, pq: Bool, to url: URL) {
    let w = t.width, h = t.height
    var packed = [UInt32](repeating: 0, count: w * h)
    t.getBytes(&packed, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    func pqDecode(_ e: Float) -> Float {
        let m1: Float = 2610.0 / 16384, m2: Float = 2523.0 / 4096 * 128
        let c1: Float = 3424.0 / 4096, c2: Float = 2413.0 / 4096 * 32, c3: Float = 2392.0 / 4096 * 32
        let p = powf(max(e, 0), 1 / m2)
        return powf(max(p - c1, 0) / (c2 - c3 * p), 1 / m1) * 10000 / 203
    }
    var px = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0..<(w * h) {
        for ch in 0..<3 {
            var v = Float((packed[i] >> UInt32(10 * ch)) & 1023) / 1023
            if pq { let l = pqDecode(v); v = powf(l / (1 + l), 1 / 2.2) }
            px[i * 4 + ch] = UInt8(min(255, max(0, v * 255 + 0.5)))
        }
    }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let img = ctx.makeImage(),
          let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
}

func gpuMs(_ cmd: MTLCommandBuffer) -> Double { (cmd.gpuEndTime - cmd.gpuStartTime) * 1000 }

let batch = max(1, Int(opt("--batch") ?? "10") ?? 10)
var msConvB: [Double] = [], msCasB: [Double] = [], msConv: [Double] = [], msCAS: [Double] = []
var W = 0, H = 0, isPQ = false
var plan = ClarityPlan.off
var tex: [MTLTexture] = []
var i = 0

/// One command buffer with `batch` back-to-back dispatches of the same kernel; returns GPU ms per dispatch.
func timed(_ body: (MTLComputeCommandEncoder) throws -> Void) async throws -> Double {
    let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
    for _ in 0..<batch { try body(enc) }
    enc.endEncoding(); cmd.commit()
    await cmd.completed()
    return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000 / Double(batch)
}

while i < count, let sample = output.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sample) {
    guard let pl = engine.planes(pb) else { print("unsupported pixel format"); exit(1) }
    if i == 0 {
        W = CVPixelBufferGetWidth(pb); H = CVPixelBufferGetHeight(pb)
        let tf = CVBufferCopyAttachment(pb, kCVImageBufferTransferFunctionKey, nil) as? String
        isPQ = tf == (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String)
        plan = ClarityPlan.make(mode: mode, wan: wan, bpp: bpp, height: H)
        // rgb10a2Unorm R'G'B', the format a renderer hands to MetalFX / the display pass.
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgb10a2Unorm, width: W, height: H, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .shared
        tex = (0..<4).map { _ in device.makeTexture(descriptor: d)! }
    }
    let plain = ClarityEngine.params(pb, deringStrength: 0)
    let cp = ClarityYUVParams(YUVCoefficients.make(matrix: CVBufferCopyAttachment(pb, kCVImageBufferYCbCrMatrixKey, nil) as? String,
                                                   fullRange: false, tenBit: pl.tenBit),
                              siting: ChromaSiting(buffer: pb), tenBit: pl.tenBit, plan: plan)
    let casPlain = ClarityCASParams(sharpness: sharp, noise: 0, pq: isPQ)
    let casClarity = ClarityCASParams(sharpness: sharp * plan.casGain, noise: plan.casNoise, pq: isPQ, limiter: plan.casLimiter,
                                      correctionGain: plan.correctionGain)

    // Baseline: plain conversion (no guided chroma, no luma stage) + plain CAS. Clarity: its conversion + its CAS.
    let t0 = try await timed { try engine.encodeConvert($0, y: pl.y, c: pl.c, dst: tex[0], params: plain, variant: ClarityEngine.Variant(guided: false)) }
    let t1 = try await timed { engine.encodeCAS($0, src: tex[0], dst: tex[1], params: casPlain) }
    let t2 = try await timed { try engine.encodeConvert($0, y: pl.y, c: pl.c, dst: tex[2], params: cp, variant: ClarityEngine.variant(for: plan)) }
    let t3 = try await timed { engine.encodeCAS($0, src: tex[2], dst: tex[3], params: casClarity) }
    withExtendedLifetime(pl.refs) {}

    if i >= 5 {          // the first frames include pipeline warm-up
        msConvB.append(t0); msCasB.append(t1); msConv.append(t2); msCAS.append(t3)
    }
    if let pngDir, i == pngFrame {
        let dir = URL(fileURLWithPath: pngDir)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        writePNG(tex[1], pq: isPQ, to: dir.appendingPathComponent("frame\(i)-off.png"))
        writePNG(tex[3], pq: isPQ, to: dir.appendingPathComponent("frame\(i)-clarity.png"))
    }
    i += 1
}

let name = URL(fileURLWithPath: path).lastPathComponent
print("=== \(name): \(i) frames, \(W)x\(H), \(isPQ ? "HDR (PQ)" : "SDR"), Clarity v\(ClarityVersion.current), mode \(mode), "
      + "bpp \(bpp.map { String(format: "%.2f", $0) } ?? "-"), \(wan ? "WAN" : "LAN") ===")
print("                         p50 ms   p95 ms")
func row(_ n: String, _ v: [Double]) {
    print(n.padding(toLength: 22, withPad: " ", startingAt: 0) + String(format: "%8.3f %8.3f", percentile(v, 50), percentile(v, 95)))
}
row("plain convert", msConvB)
row("plain CAS", msCasB)
row("Clarity convert", msConv)
row("Clarity CAS", msCAS)
let extra = (0..<msConv.count).map { (msConv[$0] - msConvB[$0]) + (msCAS[$0] - msCasB[$0]) }
row("Clarity extra cost", extra)
if let pngDir, pngFrame < i { print("PNGs: \(pngDir)/frame\(pngFrame)-off.png and -clarity.png") }
