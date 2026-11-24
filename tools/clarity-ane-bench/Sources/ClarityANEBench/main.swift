// Clarity v0.4 ANE bench: compute-plan placement and zero-copy IOSurface timing.
//
//   ClarityANEBench plan  MODEL.mlmodelc
//   ClarityANEBench time  MODEL.mlmodelc --w 1920 --h 1080 [--packed] [--image] [--iters 1000]
//                         [--frame codes.u16] [--dump delta.f16] [--load-gpu]
//
// time: x420 10-bit source -> Metal clarity_pack into the preallocated FP16 input IOSurface ->
// Core ML prediction into the preallocated output IOSurface (outputBackings). Each iteration is
// pack submit -> pack complete -> prediction return, measured separately.
import CoreML
import CoreVideo
import Foundation
import Metal

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}
func flag(_ name: String) -> Bool { CommandLine.arguments.contains(name) }

func config() -> MLModelConfiguration {
    let c = MLModelConfiguration()
    c.computeUnits = .cpuAndNeuralEngine
    return c
}

func pct(_ v: [Double], _ p: Double) -> Double {
    let s = v.sorted()
    return s[min(s.count - 1, Int((Double(s.count - 1) * p).rounded()))]
}

func runPlan(_ url: URL) async throws -> Bool {
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: config())
    guard case let .program(program) = plan.modelStructure, let main = program.functions["main"] else {
        print("not an ML Program"); return false
    }
    var counts: [String: Int] = [:]
    var bad: [String] = []
    var lines: [String] = []
    func name(_ d: MLComputeDevice) -> String {
        switch d {
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .neuralEngine: return "ANE"
        @unknown default: return "?"
        }
    }
    func walk(_ block: MLModelStructure.Program.Block) {
        for op in block.operations {
            if let u = plan.deviceUsage(for: op) {
                let p = name(u.preferred)
                counts[op.operatorName + "@" + p, default: 0] += 1
                let cost = plan.estimatedCost(of: op)?.weight ?? -1
                lines.append("  \(op.operatorName) -> \(p) supported=\(u.supported.map(name)) cost=\(String(format: "%.3f", cost))")
                if p != "ANE" { bad.append(op.operatorName + "@" + p) }
            }
            for b in op.blocks { walk(b) }
        }
    }
    walk(main.block)
    lines.forEach { print($0) }
    print("PLAN \(url.lastPathComponent): \(counts.sorted { $0.key < $1.key }.map { "\($0.key)×\($0.value)" }.joined(separator: " ")) ane_only=\(bad.isEmpty)")
    if !bad.isEmpty { print("NON-ANE compute ops: \(bad)") }
    return bad.isEmpty
}

func runTime(_ url: URL) throws {
    let w = Int(arg("--w") ?? "1920")!, h = Int(arg("--h") ?? "1080")!
    let packed = flag("--packed"), image = flag("--image")
    let iters = Int(arg("--iters") ?? "1000")!
    let model = try loadClarity(url)
    let slot = try ClaritySlot(width: w, height: h, packed: packed, imageIO: image)

    // x420 source with a real frame (or a ramp).
    var src: CVPixelBuffer?
    let attrs: [String: Any] = [kCVPixelBufferMetalCompatibilityKey as String: true,
                                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]]
    guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, attrs as CFDictionary, &src)
        == kCVReturnSuccess, let src else { fatalError("x420 alloc") }
    var codes = [UInt16](repeating: 0, count: w * h)
    if let f = arg("--frame") {
        let d = try Data(contentsOf: URL(fileURLWithPath: f))
        precondition(d.count == w * h * 2, "frame size")
        _ = codes.withUnsafeMutableBytes { d.copyBytes(to: $0) }
    } else {
        for y in 0..<h { for x in 0..<w { codes[y * w + x] = UInt16(64 + (x * 136 / w) + (y % 7)) } }
    }
    CVPixelBufferLockBaseAddress(src, [])
    let yb = CVPixelBufferGetBaseAddressOfPlane(src, 0)!, ys = CVPixelBufferGetBytesPerRowOfPlane(src, 0)
    for y in 0..<h {
        let row = yb.advanced(by: y * ys).assumingMemoryBound(to: UInt16.self)
        for x in 0..<w { row[x] = codes[y * w + x] << 6 }
    }
    let cb = CVPixelBufferGetBaseAddressOfPlane(src, 1)!, cs = CVPixelBufferGetBytesPerRowOfPlane(src, 1)
    for y in 0..<h / 2 {
        let row = cb.advanced(by: y * cs).assumingMemoryBound(to: UInt16.self)
        for x in 0..<w { row[x] = 512 << 6 }
    }
    CVPixelBufferUnlockBaseAddress(src, [])

    let dev = MTLCreateSystemDefaultDevice()!
    let queue = dev.makeCommandQueue()!
    let srcURL = Bundle.module.url(forResource: "ClarityBridge", withExtension: "metal")!
    let lib = try dev.makeLibrary(source: String(contentsOf: srcURL, encoding: .utf8), options: nil)
    let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "clarity_pack")!)
    var cache: CVMetalTextureCache?
    CVMetalTextureCacheCreate(nil, nil, dev, nil, &cache)
    var yImg: CVMetalTexture?
    CVMetalTextureCacheCreateTextureFromImage(nil, cache!, src, nil, .r16Unorm, w, h, 0, &yImg)
    let yTex = CVMetalTextureGetTexture(yImg!)!
    let (inImg, inTex) = try ClaritySlot.metalView(slot.input, cache: cache!, usage: .shaderWrite)
    _ = inImg
    var packedFlag: UInt32 = packed ? 1 : 0
    let gw = packed ? w / 2 : w, gh = packed ? h / 2 : h

    // Optional background GPU load (a busy renderer stand-in): a large blit loop on its own queue.
    var stopLoad = false
    if flag("--load-gpu") {
        let q2 = dev.makeCommandQueue()!
        let a = dev.makeBuffer(length: 64 << 20, options: .storageModePrivate)!
        let b = dev.makeBuffer(length: 64 << 20, options: .storageModePrivate)!
        Thread.detachNewThread {
            while !stopLoad {
                let c = q2.makeCommandBuffer()!
                let e = c.makeBlitCommandEncoder()!
                for _ in 0..<4 { e.copy(from: a, sourceOffset: 0, to: b, destinationOffset: 0, size: 64 << 20) }
                e.endEncoding(); c.commit(); c.waitUntilCompleted()
            }
        }
    }

    var pack: [Double] = [], gpuPack: [Double] = [], pred: [Double] = [], total: [Double] = []
    let warm = 50
    for i in 0..<(iters + warm) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let c = queue.makeCommandBuffer()!
        let e = c.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso)
        e.setTexture(yTex, index: 0)
        e.setTexture(inTex, index: 1)
        e.setBytes(&packedFlag, length: 4, index: 0)
        e.dispatchThreads(MTLSize(width: gw, height: gh, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        e.endEncoding()
        c.commit()
        c.waitUntilCompleted()
        let t1 = DispatchTime.now().uptimeNanoseconds
        _ = try slot.predict(using: model)
        let t2 = DispatchTime.now().uptimeNanoseconds
        if i >= warm {
            pack.append(Double(t1 - t0) / 1e6)
            gpuPack.append((c.gpuEndTime - c.gpuStartTime) * 1e3)
            pred.append(Double(t2 - t1) / 1e6)
            total.append(Double(t2 - t0) / 1e6)
        }
    }
    stopLoad = true
    func s(_ v: [Double]) -> String {
        String(format: "p50 %.3f p95 %.3f p99 %.3f max %.3f", pct(v, 0.5), pct(v, 0.95), pct(v, 0.99), v.max()!)
    }
    let tag = "\(url.lastPathComponent) \(w)x\(h)\(image ? " image" : "")\(flag("--load-gpu") ? " +gpuload" : "")"
    print("TIME \(tag) n=\(iters)")
    print("  pack wall  \(s(pack))")
    print("  pack gpu   \(s(gpuPack))")
    print("  predict    \(s(pred))")
    print("  total      \(s(total))")
    let thresholds = [2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 16.7]
    print("  ready-by   " + thresholds.map { t in
        String(format: "%.1fms:%.1f%%", t, 100 * Double(total.filter { $0 <= t }.count) / Double(total.count)) }.joined(separator: " "))
    print("  thermal \(ProcessInfo.processInfo.thermalState.rawValue) surfaces in=\(CVPixelBufferGetWidth(slot.input))x\(CVPixelBufferGetHeight(slot.input)) bpr=\(CVPixelBufferGetBytesPerRow(slot.input))")
    if let d = arg("--dump") {
        let out = slot.output
        CVPixelBufferLockBaseAddress(out, .readOnly)
        let base = CVPixelBufferGetBaseAddress(out)!, bpr = CVPixelBufferGetBytesPerRow(out)
        let ow = CVPixelBufferGetWidth(out), oh = CVPixelBufferGetHeight(out)
        var data = Data(capacity: ow * oh * 2)
        for y in 0..<oh { data.append(Data(bytes: base.advanced(by: y * bpr), count: ow * 2)) }
        CVPixelBufferUnlockBaseAddress(out, .readOnly)
        try data.write(to: URL(fileURLWithPath: d))
        print("  dumped \(ow)x\(oh) fp16 -> \(d)")
    }
}

let cmd = CommandLine.arguments.count > 2 ? CommandLine.arguments[1] : ""
let url = URL(fileURLWithPath: CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "")
do {
    switch cmd {
    case "plan": exit(try await runPlan(url) ? 0 : 2)
    case "time": try runTime(url)
    default: print("usage: ClarityANEBench plan|time MODEL.mlmodelc [options]"); exit(1)
    }
} catch {
    print("error: \(error)"); exit(1)
}
