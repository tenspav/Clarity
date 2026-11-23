import Foundation
import CoreVideo

// Clarity: compression cleanup and chroma reconstruction for the decoded stream (see CLARITY.md).
// This file is the pure, testable part: the user setting, chroma siting, the bitrate meter and the per-frame plan.
// The Metal side is ClarityEngine / ClarityShaders.

/// Settings › Advanced "Clarity": Auto / Off / On.
public enum ClarityMode: String, CaseIterable, Codable, Sendable {
    /// Every stage, with the luma cleanup and the mid-band strength keyed to the bitrate actually received.
    case auto
    /// The exact pre-Clarity path (bilinear chroma, plain CAS).
    case off
    /// Every stage at full strength on any route (home LAN and away) and at any bitrate.
    case on

    public static let defaultsKey = "clarityMode"
    public var label: String { rawValue.capitalized }
    public var caption: String {
        switch self {
        case .auto: return "Sharper color edges and texture; cleans up compression as much as the bitrate needs."
        case .off: return "Plain decoding, as before Clarity."
        case .on: return "Full-strength detail and cleanup, at home and away."
        }
    }
    /// Stored value, else Auto.
    public static func load(_ stored: String?) -> ClarityMode { stored.flatMap(ClarityMode.init(rawValue:)) ?? .auto }
}

/// Where the 4:2:0 chroma samples sit relative to the luma grid (Core Video's kCVImageBufferChromaLocationTopFieldKey).
public enum ChromaSiting: String, CaseIterable, Sendable {
    case left, center, topLeft, top, bottomLeft, bottom

    /// `location`: the attachment value ("Left", "Center", "TopLeft", ...). Missing or unknown: left, the
    /// HEVC / H.264 default (chroma_sample_loc_type 0), which is what the PS5 sends.
    public init(cvLocation location: String?) {
        self = ChromaSiting.allCases.first { location != nil && $0.cvLocation == location } ?? .left
    }

    /// Reads the top-field chroma location of a decoded buffer.
    public init(buffer pb: CVPixelBuffer) {
        self.init(cvLocation: CVBufferCopyAttachment(pb, kCVImageBufferChromaLocationTopFieldKey, nil) as? String)
    }

    /// Position of chroma sample (0, 0) in luma texel-index coordinates (luma sample k sits at k). Chroma sample j
    /// sits at 2j + offset.
    public var offset: SIMD2<Float> {
        switch self {
        case .left: return SIMD2(0, 0.5)
        case .center: return SIMD2(0.5, 0.5)
        case .topLeft: return SIMD2(0, 0)
        case .top: return SIMD2(0.5, 0)
        case .bottomLeft: return SIMD2(0, 1)
        case .bottom: return SIMD2(0.5, 1)
        }
    }

    public var cvLocation: String {
        switch self {
        case .left: return kCVImageBufferChromaLocation_Left as String
        case .center: return kCVImageBufferChromaLocation_Center as String
        case .topLeft: return kCVImageBufferChromaLocation_TopLeft as String
        case .top: return kCVImageBufferChromaLocation_Top as String
        case .bottomLeft: return kCVImageBufferChromaLocation_BottomLeft as String
        case .bottom: return kCVImageBufferChromaLocation_Bottom as String
        }
    }
}

/// Fixed tuning of Clarity (CLARITY.md › Shader math).
public enum ClarityTuning {
    /// Luma similarity scale of the guided chroma weights, in 10-bit code steps (spec range 12-20).
    public static let sigmaY: Float = 16
    /// Most the guided chroma may replace the plain bilinear one (confidence 1 x this).
    public static let guideMax: Float = 0.65
    /// v0.2: cap where the edge confidence is high (score 0.75 -> 1 ramps the cap from `guideMax` to this). Tried at 0.85
    /// and 1.0: chroma PSNR lower on every clip (CLARITY_PROGRESS.md), so it ships equal to `guideMax`.
    public static let guideMaxHigh: Float = 0.65
    /// v0.2: post-upscale CAS gain while the mid-band enhancer runs (no double boost).
    public static let casGainWithMidband: Float = 0.7
    /// De-ringer threshold, 10-bit code steps.
    public static let deringT: Float = 8
    /// De-ringer strength at full eligibility.
    public static let deringStrength: Float = 0.4
    /// Encoded video bits per pixel per frame, as RECEIVED (the PS5 delivers about 45-65 % of the requested rate:
    /// 10.8 Mbps average on a 45 Mbps 1080p request = 0.17 bpp): full eligibility at or below `bppFull`, none at or
    /// above `bppZero`. v0.1 used 0.20 / 0.50.
    public static let bppFull = 0.30
    public static let bppZero = 0.70
    /// Bitrate meter time constant (seconds).
    public static let smoothingSeconds = 1.5

    /// Normalised sample -> 10-bit code steps: 16-bit-unorm-backed x420 (10 bits in the top of 16) or 8-bit planes.
    public static func codeScale(tenBit: Bool) -> Float { tenBit ? 65535.0 / 64.0 : 255.0 * 4.0 }

    /// 1 at or below 0.30 bpp/frame, linear down to 0 at 0.70.
    public static func eligibility(bpp: Double) -> Float {
        guard bpp.isFinite else { return 0 }
        return Float(min(1, max(0, (bppZero - bpp) / (bppZero - bppFull))))
    }
}

/// Smoothed encoded video bits per pixel per frame. Feed it every received video access unit (HEVC payload only:
/// no audio, no FEC, no transport headers). Exponential average of bytes per frame with a time constant in seconds,
/// so a keyframe burst moves it a little and a lasting bitrate change moves it in about 1-2 s.
public struct ClarityBitrateMeter: Sendable {
    public let tau: Double
    private var bytesPerFrame: Double?
    private var lastT: Double?

    public init(tau: Double = ClarityTuning.smoothingSeconds) { self.tau = tau }

    public mutating func reset() { bytesPerFrame = nil; lastT = nil }

    /// `t`: arrival time in seconds (any monotonic clock).
    public mutating func add(frameBytes: Int, at t: Double) {
        let b = Double(max(0, frameBytes))
        guard let v = bytesPerFrame, let l = lastT else { bytesPerFrame = b; lastT = t; return }
        let dt = min(max(t - l, 0), 0.5)               // a stall does not wipe the history in one step
        let a = 1 - exp(-dt / max(tau, 1e-3))
        bytesPerFrame = v + a * (b - v)
        lastT = t
    }

    /// Bits per pixel per frame for a `width` x `height` stream; nil before the first frame.
    public func bpp(width: Int, height: Int) -> Double? {
        guard let v = bytesPerFrame, width > 0, height > 0 else { return nil }
        return v * 8 / Double(width * height)
    }
}

/// Clarity v0.2 mid-band enhancer settings (CLARITY.md › Mid-band). Code steps; 0 gains = off.
public struct ClarityMidband: Equatable, Sendable {
    /// Gain on the fine band (Y - B1) and on the mid band (B1 - B2).
    public var gFine: Float
    public var gMid: Float
    /// Largest change of one pixel, 10-bit code steps.
    public var cap: Float
    /// Range threshold T of the edge-aware weights: saturate(2 - |q - Y| / T) (1 up to T, 0 from 2T).
    public var rangeT: Float

    public init(gFine: Float, gMid: Float, cap: Float, rangeT: Float) {
        self.gFine = gFine; self.gMid = gMid; self.cap = cap; self.rangeT = rangeT
    }

    public static let off = ClarityMidband(gFine: 0, gMid: 0, cap: 0, rangeT: 16)
    /// 1080p on the home LAN. The spec's 13-tap B2 at the spec's gains (gMid 0.55) moved luma by 0.4 code RMS
    /// (invisible). These gains, on the same 13-tap B2, beat the 25-tap (5x5) B2 on Y PSNR and flicker on every clip
    /// and cost 0.2-0.3 ms less (CLARITY_PROGRESS.md › v0.2-beta).
    public static let lan1080 = ClarityMidband(gFine: 0.15, gMid: 1.0, cap: 8, rangeT: 16)
    /// 1080p over the internet (noisier: no fine band, lower threshold).
    public static let wan1080 = ClarityMidband(gFine: 0, gMid: 1.0, cap: 6, rangeT: 11)
    /// 720p and below, any route.
    public static let p720 = ClarityMidband(gFine: 0, gMid: 0.9, cap: 5, rangeT: 11)
    /// The 25-tap (5x5) B2 tuning that was tried first (bench A/B record only, with `wide`).
    public static let wideLan1080 = ClarityMidband(gFine: 0.08, gMid: 1.0, cap: 6, rangeT: 16)
    /// The spec's 13-tap presets (bench A/B record only).
    public static let specLan1080 = ClarityMidband(gFine: 0.08, gMid: 0.55, cap: 5, rangeT: 16)
    public static let specWan1080 = ClarityMidband(gFine: 0, gMid: 0.30, cap: 3, rangeT: 11)
    public static let spec720 = ClarityMidband(gFine: 0, gMid: 0.25, cap: 3, rangeT: 11)
    public static func specPreset(wan: Bool, height: Int) -> ClarityMidband {
        height < 1000 ? .spec720 : (wan ? .specWan1080 : .specLan1080)
    }

    public static func widePreset(wan: Bool, height: Int) -> ClarityMidband {
        height < 1000 ? ClarityMidband(gFine: 0, gMid: 0.6, cap: 4, rangeT: 11)
            : (wan ? ClarityMidband(gFine: 0, gMid: 0.8, cap: 5, rangeT: 11) : .wideLan1080)
    }

    public static func preset(wan: Bool, height: Int) -> ClarityMidband {
        height < 1000 ? .p720 : (wan ? .wan1080 : .lan1080)
    }

    public var active: Bool { gFine > 0 || gMid > 0 }

    /// Gains times `s` (0...1); cap and threshold kept.
    public func scaled(_ s: Float) -> ClarityMidband {
        let k = min(max(s, 0), 1)
        return ClarityMidband(gFine: gFine * k, gMid: gMid * k, cap: cap, rangeT: rangeT)
    }
}

/// What Clarity does on the next frames. Computed on the session side and handed to the renderer; the same plan
/// applies to real and Motion-generated frames (they share the conversion pass).
public struct ClarityPlan: Equatable, Sendable {
    /// Stage 1: siting-correct, luma-guided chroma reconstruction.
    public var chroma: Bool
    /// Stage 2: compression-adaptive, HDR-safe CAS (needs a sharpness preset other than Off to do anything).
    public var adaptiveCAS: Bool
    /// Stage 2 gating amount, 0 (clean stream: plain CAS strength) ... 1 (low bitrate: noisy flat patches are not sharpened).
    public var casNoise: Float
    /// Stage 3: de-ringer strength on display luma, 0 = off.
    public var deringStrength: Float
    /// v0.2: source-resolution mid-band enhancer (fused into the conversion pass).
    public var midband: ClarityMidband
    /// v0.2: flat-area / shadow block cleanup amount, 0 ... 1 (at most 1.5 code steps).
    public var flatClean: Float
    /// v0.2: multiplier on the post-upscale CAS amount (lower while the mid-band enhancer runs).
    public var casGain: Float
    /// v0.2: RCAS-style 5-tap noise limiter in the CAS.
    public var casLimiter: Bool
    /// v0.2: guided-chroma cap where the edge confidence is high (= ClarityTuning.guideMax: v0.1 behaviour).
    public var guideMaxHigh: Float
    /// HUD / minimap rectangles in stream pixels (x0, y0, x1, y1), at most 2: no mid-band change inside them.
    public var hudRects: [SIMD4<Float>]

    public static let off = ClarityPlan(chroma: false, adaptiveCAS: false, casNoise: 0, deringStrength: 0)

    public init(chroma: Bool, adaptiveCAS: Bool, casNoise: Float, deringStrength: Float, midband: ClarityMidband = .off,
                flatClean: Float = 0, casGain: Float = 1, casLimiter: Bool = false, guideMaxHigh: Float = ClarityTuning.guideMax,
                hudRects: [SIMD4<Float>] = []) {
        self.chroma = chroma; self.adaptiveCAS = adaptiveCAS; self.casNoise = casNoise; self.deringStrength = deringStrength
        self.midband = midband; self.flatClean = flatClean; self.casGain = casGain; self.casLimiter = casLimiter
        self.guideMaxHigh = guideMaxHigh; self.hudRects = hudRects
    }

    public var enabled: Bool { chroma || adaptiveCAS || deringStrength > 0 || midband.active }
    /// The conversion pass runs the luma stage (de-ringer, mid-band, flat cleanup).
    public var lumaStage: Bool { deringStrength > 0 || midband.active || flatClean > 0 }

    /// `wan`: the route crosses the internet (not the home LAN). `bpp`: smoothed RECEIVED bits per pixel per frame,
    /// nil while unknown. `height`: stream height (below 1080p the 720p mid-band preset applies).
    /// - Auto: every stage; luma cleanup (de-ringer, flat cleanup, CAS noise gate) scaled by the received-bitrate
    ///   eligibility on any route, mid-band at 50-100 % of its preset (100 % at or below 0.30 bpp).
    /// - On: every stage at full strength, home LAN and away, whatever the bitrate.
    public static func make(mode: ClarityMode, wan: Bool, bpp: Double?, height: Int = 1080) -> ClarityPlan {
        guard mode != .off else { return .off }
        let elig: Float = mode == .on ? 1 : (bpp.map(ClarityTuning.eligibility) ?? 0)
        let mb = ClarityMidband.preset(wan: wan, height: height).scaled(mode == .on ? 1 : 0.5 + 0.5 * elig)
        return ClarityPlan(chroma: true, adaptiveCAS: true, casNoise: elig, deringStrength: ClarityTuning.deringStrength * elig,
                           midband: mb, flatClean: elig, casGain: mb.active ? ClarityTuning.casGainWithMidband : 1,
                           casLimiter: true, guideMaxHigh: ClarityTuning.guideMaxHigh)
    }

    /// The v0.1-beta plan (bench A/B only): de-ringer on WAN (Auto) or any route (On), 0.20 / 0.50 bpp taper.
    public static func v01(mode: ClarityMode, wan: Bool, bpp: Double?) -> ClarityPlan {
        guard mode != .off else { return .off }
        func e(_ b: Double) -> Float { Float(min(1, max(0, (0.50 - b) / 0.30))) }
        let elig = bpp.map(e) ?? (mode == .on ? 1 : 0)
        let dering: Float = mode == .auto ? (wan ? ClarityTuning.deringStrength * elig : 0) : ClarityTuning.deringStrength * elig
        return ClarityPlan(chroma: true, adaptiveCAS: true, casNoise: bpp.map(e) ?? 0, deringStrength: dering)
    }

    /// One log line: "chroma on, CAS noise 0.42 x0.70, dering 0.17, mid 0.55/0.08 cap 5, clean 0.42".
    public var summary: String {
        guard enabled else { return "off" }
        return String(format: "chroma %@, CAS noise %.2f x%.2f, dering %.2f, mid %.2f/%.2f cap %.0f, clean %.2f",
                      chroma ? "on" : "off", adaptiveCAS ? casNoise : 0, casGain, deringStrength,
                      midband.gMid, midband.gFine, midband.cap, flatClean)
    }
}

/// Session glue, thread-safe: the network thread feeds video access unit sizes, the render queue asks for the plan
/// every frame (a lock and a few multiplies), main changes the mode live.
public final class ClarityController: @unchecked Sendable {
    private let lock = NSLock()
    private var meter = ClarityBitrateMeter()
    private var _mode: ClarityMode
    private var wan = false
    private var width = 1920, height = 1080

    public init(mode: ClarityMode = .auto) { _mode = mode }

    public var mode: ClarityMode {
        get { lock.withLock { _mode } }
        set { lock.withLock { _mode = newValue } }
    }

    /// Session start: route and stream size. Resets the bitrate history.
    public func begin(wan: Bool, width: Int, height: Int) {
        lock.withLock { self.wan = wan; self.width = width; self.height = height; meter.reset() }
    }

    /// One received video access unit (HEVC payload bytes). `t`: arrival, seconds, monotonic.
    public func addVideo(bytes: Int, at t: Double) { lock.withLock { meter.add(frameBytes: bytes, at: t) } }

    public var bpp: Double? { lock.withLock { meter.bpp(width: width, height: height) } }

    /// HUD / minimap rectangles (stream pixels) kept out of the mid-band enhancer; empty = none known.
    public var hudRects: [SIMD4<Float>] {
        get { lock.withLock { _hud } }
        set { lock.withLock { _hud = Array(newValue.prefix(2)) } }
    }
    private var _hud: [SIMD4<Float>] = []

    public var plan: ClarityPlan {
        lock.withLock {
            var p = ClarityPlan.make(mode: _mode, wan: wan, bpp: meter.bpp(width: width, height: height), height: height)
            if p.enabled { p.hudRects = _hud }
            return p
        }
    }
}

/// CPU reference of the shader math (tests and docs; the GPU kernels in ClarityShaders follow it 1:1).
/// All values in 10-bit code steps.
public enum ClarityMath {
    /// Footprint of luma pixel `p` (texel index): chroma base index and bilinear fraction.
    public static func footprint(p: SIMD2<Float>, siting: SIMD2<Float>) -> (i: SIMD2<Int>, f: SIMD2<Float>) {
        let u = (p - siting) * 0.5
        let fl = SIMD2<Float>(u.x.rounded(.down), u.y.rounded(.down))
        return (SIMD2(Int(fl.x), Int(fl.y)), u - fl)
    }

    /// Bilinear weights [00, 10, 01, 11].
    public static func bilinear(_ f: SIMD2<Float>) -> SIMD4<Float> {
        SIMD4((1 - f.x) * (1 - f.y), f.x * (1 - f.y), (1 - f.x) * f.y, f.x * f.y)
    }

    static func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
        let t = min(1, max(0, (x - a) / (b - a)))
        return t * t * (3 - 2 * t)
    }

    /// Guided chroma for one luma pixel. `c`: the 4 chroma neighbours [00, 10, 01, 11] (one channel), `l`: original
    /// luma at those chroma sites, `y`: centre luma, `cOther`: the other chroma channel (for the confidence term).
    public static func guidedChroma(y: Float, l: SIMD4<Float>, c: SIMD4<Float>, cOther: SIMD4<Float>, b: SIMD4<Float>,
                                    sigmaY: Float = ClarityTuning.sigmaY, guideMax: Float = ClarityTuning.guideMax,
                                    guideMaxHigh: Float? = nil) -> Float {
        let base = (b * c).sum()
        let d = (SIMD4(repeating: y) - l) / sigmaY
        let w = b / (1 + d * d)
        let guided = (w * c).sum() / max(w.sum(), 1e-6)
        // Pairs: (00,10), (01,11) horizontal; (00,01), (10,11) vertical.
        let pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
        var s: Float = 0
        for (a, bb) in pairs {
            let dy = abs(l[a] - l[bb])
            let dc = max(abs(c[a] - c[bb]), abs(cOther[a] - cOther[bb]))
            s = max(s, smoothstep(4, 16, dy) * smoothstep(4, 16, dc))
        }
        // v0.2: the cap ramps from guideMax to guideMaxHigh as the edge confidence goes from 0.75 to 1.
        let cap = guideMax + ((guideMaxHigh ?? guideMax) - guideMax) * smoothstep(0.75, 1, s)
        return base + (guided - base) * (s * cap)
    }

    /// constrain(d, T) = sign(d) * min(|d|, max(T - |d|, 0)).
    public static func constrain(_ d: Float, _ t: Float) -> Float {
        let m = min(abs(d), max(t - abs(d), 0))
        return d < 0 ? -m : m
    }

    /// De-ringer on display luma: centre `y` and its 4 neighbours.
    public static func dering(y: Float, n: Float, s: Float, e: Float, w: Float,
                              strength: Float, t: Float = ClarityTuning.deringT) -> Float {
        let gx = abs(e - w), gy = abs(s - n)
        var wx: Float = 1, wy: Float = 1
        if gx > gy + 4 { wx = 0.15; wy = 1 } else if gy > gx + 4 { wx = 1; wy = 0.15 }
        let d = (wx * (constrain(e - y, t) + constrain(w - y, t)) + wy * (constrain(n - y, t) + constrain(s - y, t))) / (2 * (wx + wy))
        let maxDiff = max(max(abs(n - y), abs(s - y)), max(abs(e - y), abs(w - y)))
        let guardW = 1 - smoothstep(10, 24, maxDiff)
        let lo = min(y, min(min(n, s), min(e, w))), hi = max(y, max(max(n, s), max(e, w)))
        return min(max(y + min(max(strength * guardW * d, -2), 2), lo), hi)
    }
}

extension ClarityMath {
    /// Inputs of the v0.2 mid-band enhancer, 10-bit code steps. Stencil: centre, 4 axial at r1 (N, S, W, E),
    /// 4 diagonal at r1 (NW, NE, SW, SE), 4 axial at r2 (NN, SS, WW, EE).
    public struct MidbandTaps {
        public var y: Float
        public var a1: SIMD4<Float>
        public var d1: SIMD4<Float>
        public var a2: SIMD4<Float>
        /// Wide B2 only: (+-2, +-1), (+-1, +-2), (+-2, +-2), each ordered (-,-), (+,-), (-,+), (+,+).
        public var wide: (k1: SIMD4<Float>, k2: SIMD4<Float>, k3: SIMD4<Float>)?
        public init(y: Float, a1: SIMD4<Float>, d1: SIMD4<Float>, a2: SIMD4<Float>,
                    wide: (k1: SIMD4<Float>, k2: SIMD4<Float>, k3: SIMD4<Float>)? = nil) {
            self.y = y; self.a1 = a1; self.d1 = d1; self.a2 = a2; self.wide = wide
        }
    }

    /// v0.2 mid-band enhancer + flat cleanup on display luma (CLARITY.md › Mid-band). `yIn`: luma after the
    /// de-ringer (or the centre itself); all decisions use the original taps. `hud`: pixel is inside a HUD rectangle.
    /// `tensor`: structure-tensor direction weights (A/B switch).
    public static func midband(yIn: Float, _ t: MidbandTaps, mb: ClarityMidband, flatClean: Float, hud: Bool = false,
                               tensor: Bool = false) -> Float {
        let Y = t.y, a1 = t.a1, d1 = t.d1, a2 = t.a2
        let lo = min(Y, min(a1.min(), d1.min())), hi = max(Y, max(a1.max(), d1.max()))
        let span = hi - lo
        let (N, S, W, E) = (a1.x, a1.y, a1.z, a1.w), (NW, NE, SW, SE) = (d1.x, d1.y, d1.z, d1.w)
        // Four 2x2 cells (a b / c d) of the central 3x3: g = 0.5 (b + d - a - c, c + d - a - b).
        let g = [SIMD2<Float>(N + Y - NW - W, W + Y - NW - N), SIMD2(NE + E - N - Y, Y + E - N - NE),
                 SIMD2(Y + S - W - SW, SW + S - W - Y), SIMD2(E + SE - Y - S, S + SE - Y - E)].map { $0 * 0.5 }
        var jxx: Float = 0, jyy: Float = 0, jxy: Float = 0
        for v in g { jxx += v.x * v.x; jyy += v.y * v.y; jxy += v.x * v.y }
        jxx *= 0.25; jyy *= 0.25; jxy *= 0.25
        let tr = jxx + jyy
        let coh = ((jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy).squareRoot() / (tr + 1.024)
        let T = mb.rangeT
        // Range weight r(d) = saturate(2 - |d| / T): 1 up to T, 0 from 2T.
        func r(_ q: SIMD4<Float>) -> SIMD4<Float> {
            let d = (q - SIMD4(repeating: Y)) / T
            return SIMD4(min(max(2 - abs(d.x), 0), 1), min(max(2 - abs(d.y), 0), 1), min(max(2 - abs(d.z), 0), 1), min(max(2 - abs(d.w), 0), 1))
        }
        var ra1 = r(a1), rd1 = r(d1), ra2 = r(a2)
        if tensor {
            let k = 0.65 * coh / (tr + 1.024)
            let dh = 1 - k * jxx, dv = 1 - k * jyy, dd = 1 - k * 0.5 * (tr + 2 * jxy), da = 1 - k * 0.5 * (tr - 2 * jxy)
            ra1 *= SIMD4(dv, dv, dh, dh); rd1 *= SIMD4(dd, da, da, dd); ra2 *= SIMD4(dv, dv, dh, dh)
        }
        let vA1 = (ra1 * a1).sum(), vD1 = (rd1 * d1).sum(), vA2 = (ra2 * a2).sum()
        let sA1 = ra1.sum(), sD1 = rd1.sum(), sA2 = ra2.sum()
        let b1 = (4 * Y + 2 * vA1 + vD1) / (4 + 2 * sA1 + sD1)
        let b2: Float
        if let k = t.wide {
            let r1 = r(k.k1), r2 = r(k.k2), r3 = r(k.k3)
            b2 = (4 * Y + 4 * vA1 + 4 * vD1 + 2 * vA2 + 2 * (r1 * k.k1).sum() + 2 * (r2 * k.k2).sum() + (r3 * k.k3).sum())
                / (4 + 4 * sA1 + 4 * sD1 + 2 * sA2 + 2 * r1.sum() + 2 * r2.sum() + r3.sum())
        } else {
            b2 = (4 * Y + 2 * vA1 + 2 * vD1 + vA2) / (4 + 2 * sA1 + 2 * sD1 + sA2)
        }
        let fine = Y - b1, mid = b1 - b2
        let textureGate = smoothstep(3, 9, span)
        let strongEdge = coh * smoothstep(18, 42, span)
        let spikeGate = 1 - smoothstep(5, 14, abs(fine) - 2 * abs(mid))
        let toneGuard = smoothstep(110, 200, Y) * (1 - smoothstep(720, 840, Y))
        let sv: Float = (N - Y) * (S - Y) > 0 ? min(abs(N - Y), abs(S - Y)) : 0
        let sh: Float = (W - Y) * (E - Y) > 0 ? min(abs(W - Y), abs(E - Y)) : 0
        let strokeGuard = 1 - smoothstep(10, 24, max(sv, sh))
        let hudGuard: Float = hud ? 0 : 1
        let gain = textureGate * (1 - strongEdge) * spikeGate * toneGuard * strokeGuard * hudGuard
        var delta = min(max(gain * (mb.gFine * fine + mb.gMid * mid), -mb.cap), mb.cap)
        let shadow = 1 - smoothstep(150, 300, Y)
        let cg = flatClean * (1 - smoothstep(2 + 2 * shadow, 4 + 4 * shadow, span)) * hudGuard
        delta += min(max(cg * (b2 - Y), -1.5), 1.5)
        return min(max(yIn + delta, lo), hi)
    }
}
