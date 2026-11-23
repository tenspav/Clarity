import Foundation
import Metal
import CoreVideo

/// Clarity GPU side (CLARITY.md). No extra pass and no extra frame of latency:
///   1. siting-correct, luma-guided chroma reconstruction  } fused into the stream-res YCbCr -> R'G'B' pass
///   3. de-ringer on display luma (raw planes untouched)    }   (`clarity_yuv_rgb`, replaces `yuv_rgb`)
///   4. v0.2 mid-band enhancer + flat cleanup (13 taps)     }   one luma stage on the same taps
///   2. compression-adaptive, HDR-safe CAS with an RCAS-style noise limiter -> same 5 taps as CAS (`clarity_cas`,
///      used by `clarity_cas_sharpen` and the fused present pass)
/// The source is self-contained (own include, `clarity_` prefixes, function constants 40-45) so it is prepended to
/// a host renderer's own shader source and shares its library.
public enum ClarityShaders {
    public static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // Variant switches (zero cost when off: compiled out).
    constant bool clarity_guided [[function_constant(40)]];
    constant bool clarity_dering [[function_constant(41)]];
    constant bool clarity_raw    [[function_constant(42)]];   // bench: write (Y, Cb, Cr) in 10-bit code steps
    constant bool clarity_mid    [[function_constant(43)]];   // v0.2: mid-band enhancer + flat cleanup
    constant bool clarity_tensor [[function_constant(44)]];   // v0.2 A/B: structure-tensor direction weights
    constant bool clarity_linear [[function_constant(45)]];   // bench A/B: write linear light (PQ EOTF here)
    constant bool clarity_wide   [[function_constant(46)]];   // v0.2: B2 = 25-tap [1,2,2,2,1]^2 (5x5) instead of 13 taps

    // 128 bytes. First 8 floats = the plain YCbCr -> RGB parameters (YUVCoefficients).
    struct ClarityYUVParams {
        float yOff; float yScale; float cScale; float rCr; float gCb; float gCr; float bCb; float cOff;
        float2 siting;        // chroma sample (0,0) in luma texel-index coordinates
        float codeScale;      // normalised sample -> 10-bit code steps
        float invSigmaY;      // 1 / sigmaY (code steps)
        float guideMax;       // 0.65
        float deringStrength; // 0 = off
        float deringT;        // 8
        float guideMaxHigh;   // v0.2: cap at high edge confidence (= guideMax: v0.1)
        float gFine; float gMid; float cap; float rangeT;   // v0.2 mid-band
        float flatClean; float pad0; float pad1; float pad2;
        float4 hud0; float4 hud1;                            // HUD rectangles (x0, y0, x1, y1), stream pixels
    };

    inline float clarity_constrain(float d, float t) { return sign(d) * min(fabs(d), max(t - fabs(d), 0.0)); }

    // Stage 1 core: 4-tap luma-guided chroma. C00..C11: chroma neighbours, L: original luma at their sites (code
    // steps), b: bilinear weights, Yc: centre luma. Positive weights only: the result stays inside the 4 neighbours.
    inline float2 clarity_guide(float2 C00, float2 C10, float2 C01, float2 C11, float4 L, float4 b, float Yc,
                                constant ClarityYUVParams &p) {
        float2 base = b.x * C00 + b.y * C10 + b.z * C01 + b.w * C11;
        float4 d = (Yc - L) * p.invSigmaY;
        float4 w = b / (1.0 + d * d);
        float2 guided = (w.x * C00 + w.y * C10 + w.z * C01 + w.w * C11) / max(w.x + w.y + w.z + w.w, 1e-6);
        // Confidence: a luma step AND a chroma step across the same footprint pair (code steps).
        float4 dy = fabs(float4(L.x - L.y, L.z - L.w, L.x - L.z, L.y - L.w));
        float2 e0 = fabs(C00 - C10), e1 = fabs(C01 - C11), e2 = fabs(C00 - C01), e3 = fabs(C10 - C11);
        float4 dc = float4(max(e0.x, e0.y), max(e1.x, e1.y), max(e2.x, e2.y), max(e3.x, e3.y)) * p.codeScale;
        float4 sc = smoothstep(4.0, 16.0, dy) * smoothstep(4.0, 16.0, dc);
        float s = max(max(sc.x, sc.y), max(sc.z, sc.w));
        float conf = s * mix(p.guideMax, p.guideMaxHigh, smoothstep(0.75, 1.0, s));
        return mix(base, guided, conf);
    }

    // Stage 3 core: conservative CDEF-style de-ringer (code steps). At most +-2 codes, inside the local min / max.
    inline float clarity_dering_core(float Y, float N, float S, float E, float W, constant ClarityYUVParams &p) {
        float gx = fabs(E - W), gy = fabs(S - N);
        float wx = 1.0, wy = 1.0;
        if (gx > gy + 4.0) { wx = 0.15; } else if (gy > gx + 4.0) { wy = 0.15; }
        float T = p.deringT;
        float D = (wx * (clarity_constrain(E - Y, T) + clarity_constrain(W - Y, T))
                 + wy * (clarity_constrain(N - Y, T) + clarity_constrain(S - Y, T))) / (2.0 * (wx + wy));
        float md = max(max(fabs(N - Y), fabs(S - Y)), max(fabs(E - Y), fabs(W - Y)));
        float guardW = 1.0 - smoothstep(10.0, 24.0, md);
        float lo = min(Y, min(min(N, S), min(E, W))), hi = max(Y, max(max(N, S), max(E, W)));
        return clamp(Y + clamp(p.deringStrength * guardW * D, -2.0, 2.0), lo, hi);
    }

    inline bool clarity_in(float2 q, float4 r) { return q.x >= r.x && q.y >= r.y && q.x < r.z && q.y < r.w; }

    // v0.2 mid-band enhancer + flat cleanup (code steps). Y: original centre; a1 = (N, S, W, E) at r1, d1 =
    // (NW, NE, SW, SE), a2 = (NN, SS, WW, EE) at r2. yIn: luma after the de-ringer. Two range-weighted blurs
    // B1 (centre 4, A1 2, D1 1) and B2; fine = Y - B1, mid = B1 - B2. Gated off on flat patches, strong coherent
    // edges, isolated spikes, deep shadows / highlights, thin strokes and HUD rects. Never leaves the 3x3 min / max.
    // B2: 13 taps (centre 4, A1 2, D1 2, A2 1), or with clarity_wide the 5x5 [1,2,2,2,1] x [1,2,2,2,1] blur; k1 =
    // (+-2, +-1), k2 = (+-1, +-2), k3 = (+-2, +-2), each ordered (-,-), (+,-), (-,+), (+,+).
    // Half precision on differences to the centre (|d| <= 1023 is exact; the tensor runs on d / 32 so its squares fit).
    // Range weight: r(d) = saturate(2 - |d| / T) (1 up to T, 0 from 2T).
    inline half4 clarity_rw(half4 d, half invT) { return saturate(2.0h - fabs(d) * invT); }

    inline float clarity_midband(float yIn, float Yf, float4 a1f, float4 d1f, float4 a2f, float4 k1f, float4 k2f, float4 k3f,
                                 float2 pos, constant ClarityYUVParams &p) {
        half4 a1 = half4(a1f - Yf), d1 = half4(d1f - Yf);
        half lo = min(0.0h, min(min(min(a1.x, a1.y), min(a1.z, a1.w)), min(min(d1.x, d1.y), min(d1.z, d1.w))));
        half hi = max(0.0h, max(max(max(a1.x, a1.y), max(a1.z, a1.w)), max(max(d1.x, d1.y), max(d1.z, d1.w))));
        half span = hi - lo;
        // Structure tensor over the four 2x2 cells (a b / c d) of the central 3x3: g = 0.5 (b + d - a - c, c + d - a - b).
        half4 A = a1 * 0.015625h, D = d1 * 0.015625h;   // 0.5 / 32
        half N = A.x, S = A.y, W = A.z, E = A.w, NW = D.x, NE = D.y, SW = D.z, SE = D.w;
        half4 gx = half4(N - NW - W, NE + E - N, S - W - SW, E + SE - S);
        half4 gy = half4(W - NW - N, E - N - NE, SW + S - W, S + SE - E);
        half jxx = dot(gx, gx) * 0.25h, jyy = dot(gy, gy) * 0.25h, jxy = dot(gx, gy) * 0.25h;
        half tr = jxx + jyy;
        half coh = sqrt((jxx - jyy) * (jxx - jyy) + 4.0h * jxy * jxy) / (tr + 0.001h);
        half invT = half(1.0 / p.rangeT);
        half4 ra1 = clarity_rw(a1, invT), rd1 = clarity_rw(d1, invT);
        half4 a2 = half4(a2f - Yf);
        half4 ra2 = clarity_rw(a2, invT);
        if (clarity_tensor) {
            half k = 0.65h * coh / (tr + 0.001h);
            half dh = 1.0h - k * jxx, dv = 1.0h - k * jyy, dd = 1.0h - k * 0.5h * (tr + 2.0h * jxy), da = 1.0h - k * 0.5h * (tr - 2.0h * jxy);
            ra1 *= half4(dv, dv, dh, dh); rd1 *= half4(dd, da, da, dd); ra2 *= half4(dv, dv, dh, dh);
        }
        const half4 one = half4(1.0h);
        half vA1 = dot(ra1, a1), vD1 = dot(rd1, d1), vA2 = dot(ra2, a2);
        half sA1 = dot(ra1, one), sD1 = dot(rd1, one), sA2 = dot(ra2, one);
        half B1 = (2.0h * vA1 + vD1) / (4.0h + 2.0h * sA1 + sD1);          // B1 - Y
        half B2;                                                           // B2 - Y
        if (clarity_wide) {
            half4 k1 = half4(k1f - Yf), k2 = half4(k2f - Yf), k3 = half4(k3f - Yf);
            half4 r1 = clarity_rw(k1, invT), r2 = clarity_rw(k2, invT), r3 = clarity_rw(k3, invT);
            B2 = (4.0h * (vA1 + vD1) + 2.0h * (vA2 + dot(r1, k1) + dot(r2, k2)) + dot(r3, k3))
               / (4.0h + 4.0h * (sA1 + sD1) + 2.0h * (sA2 + dot(r1, one) + dot(r2, one)) + dot(r3, one));
        } else {
            B2 = (2.0h * (vA1 + vD1) + vA2) / (4.0h + 2.0h * (sA1 + sD1) + sA2);
        }
        half fine = -B1, mid = B1 - B2;
        half textureGate = smoothstep(3.0h, 9.0h, span);
        half strongEdge = coh * smoothstep(18.0h, 42.0h, span);
        half spikeGate = 1.0h - smoothstep(5.0h, 14.0h, fabs(fine) - 2.0h * fabs(mid));
        half Y = half(Yf);
        half toneGuard = smoothstep(110.0h, 200.0h, Y) * (1.0h - smoothstep(720.0h, 840.0h, Y));
        half sv = a1.x * a1.y > 0.0h ? min(fabs(a1.x), fabs(a1.y)) : 0.0h;
        half sh = a1.z * a1.w > 0.0h ? min(fabs(a1.z), fabs(a1.w)) : 0.0h;
        half strokeGuard = 1.0h - smoothstep(10.0h, 24.0h, max(sv, sh));
        half hudGuard = (clarity_in(pos, p.hud0) || clarity_in(pos, p.hud1)) ? 0.0h : 1.0h;
        half gain = textureGate * (1.0h - strongEdge) * spikeGate * toneGuard * strokeGuard * hudGuard;
        half cap = half(p.cap);
        half delta = clamp(gain * (half(p.gFine) * fine + half(p.gMid) * mid), -cap, cap);
        // Flat cleanup: blocks and steps of a few codes in flat areas (wider in shadows) eased toward B2.
        half shadow = 1.0h - smoothstep(150.0h, 300.0h, Y);
        half cg = half(p.flatClean) * (1.0h - smoothstep(2.0h + 2.0h * shadow, 4.0h + 4.0h * shadow, span)) * hudGuard;
        delta += clamp(cg * B2, -1.5h, 1.5h);
        return clamp(yIn + float(delta), Yf + float(lo), Yf + float(hi));
    }

    // Luma stage on the taps (de-ringer, then mid-band on the original taps).
    inline float clarity_luma(float Yc, float4 a1, float4 d1, float4 a2, float4 k1, float4 k2, float4 k3, float2 pos,
                              constant ClarityYUVParams &p) {
        float yD = Yc;
        if (clarity_dering && p.deringStrength > 0.0) yD = clarity_dering_core(Yc, a1.x, a1.y, a1.w, a1.z, p);
        if (clarity_mid) yD = clarity_midband(yD, Yc, a1, d1, a2, k1, k2, k3, pos, p);
        return yD;
    }

    inline float3 clarity_pq_decode(float3 e) {
        float3 q = pow(max(e, 0.0), 1.0 / 78.84375);
        return pow(max(q - 0.8359375, 0.0) / (18.8515625 - 18.6875 * q), 1.0 / 0.1593017578125);
    }

    inline void clarity_out(texture2d<float, access::write> dst, uint2 at, float yDisp, float2 C, constant ClarityYUVParams &p) {
        if (clarity_raw) { dst.write(float4(yDisp * p.codeScale, C * p.codeScale, 1.0), at); return; }
        float y = (yDisp - p.yOff) * p.yScale;
        float2 c = (C - p.cOff) * p.cScale;
        float3 rgb = saturate(float3(y + p.rCr * c.y, y - p.gCb * c.x - p.gCr * c.y, y + p.bCb * c.x));
        if (clarity_linear) rgb = clarity_pq_decode(rgb) * (10000.0 / 203.0);   // BT.2020 linear, 1.0 = 203 nits
        dst.write(float4(rgb, 1.0), at);
    }

    inline float clarity_ld(texture2d<float> t, int2 q, int2 mx, float K) { return t.read(uint2(clamp(q, int2(0), mx))).r * K; }

    inline void clarity_wide_taps(texture2d<float> t, int2 c, int2 mx, float K, thread float4 &k1, thread float4 &k2, thread float4 &k3) {
        k1 = float4(clarity_ld(t, c + int2(-2, -1), mx, K), clarity_ld(t, c + int2(2, -1), mx, K),
                    clarity_ld(t, c + int2(-2, 1), mx, K), clarity_ld(t, c + int2(2, 1), mx, K));
        k2 = float4(clarity_ld(t, c + int2(-1, -2), mx, K), clarity_ld(t, c + int2(1, -2), mx, K),
                    clarity_ld(t, c + int2(-1, 2), mx, K), clarity_ld(t, c + int2(1, 2), mx, K));
        k3 = float4(clarity_ld(t, c + int2(-2, -2), mx, K), clarity_ld(t, c + int2(2, -2), mx, K),
                    clarity_ld(t, c + int2(-2, 2), mx, K), clarity_ld(t, c + int2(2, 2), mx, K));
    }

    // Stage 1 chroma for pixel gid: gathers of the 2x2 footprint, guided by the ORIGINAL luma at the chroma sites.
    // `tileL`: the tile kernel's original luma at the 4 chroma sites (Left siting), else NAN = sample the texture.
    inline float2 clarity_chroma(texture2d<float> yT, texture2d<float> cT, uint2 gid, float Yc, constant ClarityYUVParams &p,
                                 float4 tileL = float4(NAN)) {
        const float K = p.codeScale;
        constexpr sampler ps(coord::pixel, filter::linear, address::clamp_to_edge);
        // Footprint: u = (p - siting) / 2, chroma base fl, fraction f.
        float2 u = (float2(gid) - p.siting) * 0.5;
        float2 fl = floor(u);
        float2 f = u - fl;
        // Gather at the shared corner of texels fl and fl + 1 (edge-clamped like the reads it replaces).
        // Order: x = (i0, j1), y = (i1, j1), z = (i1, j0), w = (i0, j0).
        float4 gb = cT.gather(ps, fl + 1.0, int2(0), component::x);
        float4 gr = cT.gather(ps, fl + 1.0, int2(0), component::y);
        float2 C00 = float2(gb.w, gr.w), C10 = float2(gb.z, gr.z), C01 = float2(gb.x, gr.x), C11 = float2(gb.y, gr.y);
        float4 b = float4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
        float2 C = b.x * C00 + b.y * C10 + b.z * C01 + b.w * C11;
        if (clarity_guided) {
            // Original luma at each chroma site j: 2j + siting (texel index) = pixel coordinate + 0.5.
            int2 cmax = int2(cT.get_width(), cT.get_height()) - 1;
            float2 j0 = float2(clamp(int2(fl), int2(0), cmax)), j1 = float2(clamp(int2(fl) + 1, int2(0), cmax));
            float4 L = tileL;
            if (isnan(L.x)) {
                float2 s0 = j0 * 2.0 + p.siting + 0.5, s1 = j1 * 2.0 + p.siting + 0.5;
                L = float4(yT.sample(ps, s0).r, yT.sample(ps, float2(s1.x, s0.y)).r,
                           yT.sample(ps, float2(s0.x, s1.y)).r, yT.sample(ps, s1).r) * K;
            }
            C = clarity_guide(C00, C10, C01, C11, L, b, Yc, p);
        }
        return C;
    }

    // Replaces a plain yuv_rgb conversion: 420v / x420 -> R'G'B' at stream resolution (before PQ decode and MetalFX).
    // One thread per pixel, direct (cached) reads. Fetches: luma 1, chroma 2 gathers, guided +4 bilinear luma taps,
    // luma stage +4 (de-ringer) / +12 (mid-band) reads.
    kernel void clarity_yuv_rgb(texture2d<float> yT [[texture(0)]],
                                texture2d<float> cT [[texture(1)]],
                                texture2d<float, access::write> dst [[texture(2)]],
                                constant ClarityYUVParams &p [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]]) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        const float K = p.codeScale;
        float Yc = yT.read(gid).r * K;
        float2 C = clarity_chroma(yT, cT, gid, Yc, p);
        float yD = Yc;
        if (clarity_dering || clarity_mid) {
            int2 c = int2(gid), mx = int2(yT.get_width(), yT.get_height()) - 1;
            float4 a1 = float4(clarity_ld(yT, c + int2(0, -1), mx, K), clarity_ld(yT, c + int2(0, 1), mx, K),
                               clarity_ld(yT, c + int2(-1, 0), mx, K), clarity_ld(yT, c + int2(1, 0), mx, K));
            float4 d1 = float4(0.0), a2 = float4(0.0);
            if (clarity_mid) {
                d1 = float4(clarity_ld(yT, c + int2(-1, -1), mx, K), clarity_ld(yT, c + int2(1, -1), mx, K),
                            clarity_ld(yT, c + int2(-1, 1), mx, K), clarity_ld(yT, c + int2(1, 1), mx, K));
                a2 = float4(clarity_ld(yT, c + int2(0, -2), mx, K), clarity_ld(yT, c + int2(0, 2), mx, K),
                            clarity_ld(yT, c + int2(-2, 0), mx, K), clarity_ld(yT, c + int2(2, 0), mx, K));
            }
            float4 k1 = float4(0.0), k2 = float4(0.0), k3 = float4(0.0);
            if (clarity_mid && clarity_wide) clarity_wide_taps(yT, c, mx, K, k1, k2, k3);
            yD = clarity_luma(Yc, a1, d1, a2, k1, k2, k3, float2(gid), p);
        }
        clarity_out(dst, gid, yD / K, C, p);
    }

    // A/B form: the 13 luma taps from a 16x16 + 2 px halo threadgroup tile (20x20 FP16 code values = 800 B).
    // Dispatch: threadgroups of exactly 16x16. Chroma and its guide still use the texture (gathers / samples).
    kernel void clarity_yuv_rgb_tile(texture2d<float> yT [[texture(0)]],
                                     texture2d<float> cT [[texture(1)]],
                                     texture2d<float, access::write> dst [[texture(2)]],
                                     constant ClarityYUVParams &p [[buffer(0)]],
                                     uint2 gid [[thread_position_in_grid]],
                                     uint2 lid [[thread_position_in_threadgroup]],
                                     uint2 tg [[threadgroup_position_in_grid]]) {
        threadgroup half tile[20 * 20];
        const float K = p.codeScale;
        int2 mx = int2(yT.get_width(), yT.get_height()) - 1;
        int2 origin = int2(tg) * 16 - 2;
        for (uint i = lid.y * 16 + lid.x; i < 400; i += 256) {
            int2 q = origin + int2(i % 20, i / 20);
            tile[i] = half(clarity_ld(yT, q, mx, K));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        int b = int(lid.y + 2) * 20 + int(lid.x + 2);
        float Yc = float(tile[b]);
        float4 a1 = float4(tile[b - 20], tile[b + 20], tile[b - 1], tile[b + 1]);
        float4 d1 = float4(tile[b - 21], tile[b - 19], tile[b + 19], tile[b + 21]);
        float4 a2 = float4(tile[b - 40], tile[b + 40], tile[b - 2], tile[b + 2]);
        float4 k1 = float4(0.0), k2 = float4(0.0), k3 = float4(0.0);
        if (clarity_wide) {
            k1 = float4(tile[b - 22], tile[b - 18], tile[b + 18], tile[b + 22]);
            k2 = float4(tile[b - 41], tile[b - 39], tile[b + 39], tile[b + 41]);
            k3 = float4(tile[b - 42], tile[b - 38], tile[b + 38], tile[b + 42]);
        }
        // Left siting (the PS5's): chroma site j sits on luma column 2j, halfway between rows 2j and 2j + 1; all four
        // sites of the footprint are inside the tile, so the guide luma comes from it (no texture samples).
        float4 tileL = float4(NAN);
        if (clarity_guided && p.siting.x == 0.0 && p.siting.y == 0.5) {
            int2 cmax = int2(cT.get_width(), cT.get_height()) - 1;
            int2 fl = int2(floor((float2(gid) - p.siting) * 0.5));
            int2 j0 = clamp(fl, int2(0), cmax), j1 = clamp(fl + 1, int2(0), cmax);
            int c0 = 2 * j0.x - origin.x, c1 = 2 * j1.x - origin.x, r0 = (2 * j0.y - origin.y) * 20, r1 = (2 * j1.y - origin.y) * 20;
            tileL = 0.5 * float4(float(tile[r0 + c0]) + float(tile[r0 + 20 + c0]), float(tile[r0 + c1]) + float(tile[r0 + 20 + c1]),
                                 float(tile[r1 + c0]) + float(tile[r1 + 20 + c0]), float(tile[r1 + c1]) + float(tile[r1 + 20 + c1]));
        }
        float2 C = clarity_chroma(yT, cT, gid, Yc, p, tileL);
        float yD = clarity_luma(Yc, a1, d1, a2, k1, k2, k3, float2(gid), p);
        clarity_out(dst, gid, yD / K, C, p);
    }

    // Reference form of clarity_yuv_rgb: plain clamped reads instead of gathers (same math). Self-check only.
    kernel void clarity_yuv_rgb_px(texture2d<float> yT [[texture(0)]],
                                   texture2d<float, access::read> cT [[texture(1)]],
                                   texture2d<float, access::write> dst [[texture(2)]],
                                   constant ClarityYUVParams &p [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
        uint w = dst.get_width(), h = dst.get_height();
        if (gid.x >= w || gid.y >= h) return;
        const float K = p.codeScale;
        float Yc = yT.read(gid).r * K;
        float2 u = (float2(gid) - p.siting) * 0.5;
        float2 fl = floor(u);
        float2 f = u - fl;
        int2 cmax = int2(cT.get_width(), cT.get_height()) - 1;
        int2 j0 = clamp(int2(fl), int2(0), cmax), j1 = clamp(int2(fl) + 1, int2(0), cmax);
        float2 C00 = cT.read(uint2(j0)).rg, C10 = cT.read(uint2(j1.x, j0.y)).rg;
        float2 C01 = cT.read(uint2(j0.x, j1.y)).rg, C11 = cT.read(uint2(j1)).rg;
        float4 b = float4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
        float2 C = b.x * C00 + b.y * C10 + b.z * C01 + b.w * C11;
        if (clarity_guided) {
            constexpr sampler ls(coord::pixel, filter::linear, address::clamp_to_edge);
            float2 s0 = float2(j0) * 2.0 + p.siting + 0.5, s1 = float2(j1) * 2.0 + p.siting + 0.5;
            float4 L = float4(yT.sample(ls, s0).r, yT.sample(ls, float2(s1.x, s0.y)).r,
                              yT.sample(ls, float2(s0.x, s1.y)).r, yT.sample(ls, s1).r) * K;
            C = clarity_guide(C00, C10, C01, C11, L, b, Yc, p);
        }
        float yD = Yc;
        if (clarity_dering || clarity_mid) {
            int2 c = int2(gid), mx = int2(yT.get_width(), yT.get_height()) - 1;
            float4 a1 = float4(clarity_ld(yT, c + int2(0, -1), mx, K), clarity_ld(yT, c + int2(0, 1), mx, K),
                               clarity_ld(yT, c + int2(-1, 0), mx, K), clarity_ld(yT, c + int2(1, 0), mx, K));
            float4 d1 = float4(clarity_ld(yT, c + int2(-1, -1), mx, K), clarity_ld(yT, c + int2(1, -1), mx, K),
                               clarity_ld(yT, c + int2(-1, 1), mx, K), clarity_ld(yT, c + int2(1, 1), mx, K));
            float4 a2 = float4(clarity_ld(yT, c + int2(0, -2), mx, K), clarity_ld(yT, c + int2(0, 2), mx, K),
                               clarity_ld(yT, c + int2(-2, 0), mx, K), clarity_ld(yT, c + int2(2, 0), mx, K));
            float4 k1 = float4(0.0), k2 = float4(0.0), k3 = float4(0.0);
            if (clarity_wide) clarity_wide_taps(yT, c, mx, K, k1, k2, k3);
            yD = clarity_luma(Yc, a1, d1, a2, k1, k2, k3, float2(gid), p);
        }
        clarity_out(dst, gid, yD / K, C, p);
    }

    // Stage 2 parameters (16 bytes + float4).
    struct ClarityCASParams {
        float sharpness;   // CAS amount (SharpnessPreset x plan gain)
        float noise;       // 0 = clean stream (plain CAS strength) ... 1 = low bitrate (noisy flat patches gated)
        float knee;        // signal value where positive deltas start to taper (highlight limit)
        float limiter;     // v0.2: RCAS-style noise limiter, 0 = off, 1 = on
        float4 lumaW;      // luma weights of the working signal (BT.2020 for PQ, BT.709 for SDR)
    };

    // Stage 2: CAS on the same 5 taps (c centre, a up, b left, e right, g down), compression-adaptive and HDR-safe:
    // the sharpened luma delta is bounded, kept inside the local luma range, tapered near the highlight limit and
    // applied as one ratio to R'G'B' (hue kept, no channel is boosted on its own, none pushed past 1.0).
    inline float3 clarity_cas(float3 c, float3 a, float3 b, float3 e, float3 g, constant ClarityCASParams &p) {
        float3 mn = min(c, min(min(a, b), min(e, g)));
        float3 mxc = max(c, max(max(a, b), max(e, g)));
        float3 rng = mxc - mn;
        if (max(rng.r, max(rng.g, rng.b)) < 0.04) return c;   // flat: same skip as the plain CAS
        float3 lw = p.lumaW.rgb;
        float lc = dot(c, lw), lu = dot(a, lw), ll = dot(b, lw), lr = dot(e, lw), ld = dot(g, lw);
        float lmn = min(lc, min(min(lu, ll), min(lr, ld))), lmx = max(lc, max(max(lu, ll), max(lr, ld)));
        float rY = lmx - lmn;
        // Coherence: a step through the centre (one axis carries the whole range) or a thin line (one axis curved,
        // the other flat). Codec noise is neither (isotropic, small).
        float gr = max(fabs(lr - ll), fabs(ld - lu)) / (rY + 1e-4);
        float lx = fabs(lr + ll - 2.0 * lc), ly = fabs(ld + lu - 2.0 * lc);
        float an = fabs(lx - ly) / (lx + ly + 1e-4);
        float coh = saturate(max(gr, an) * 1.5 - 0.5);
        float lowC = 1.0 - smoothstep(0.05, 0.14, rY);
        float gate = 1.0 - p.noise * lowC * (1.0 - 0.75 * coh);
        // RCAS noise limiter: a centre that sticks out of its 4 neighbours by a large part of the local range is
        // noise-like (grain, mosquito dots): the lobe drops to half there.
        float nz = saturate(fabs(0.25 * (lu + ll + lr + ld) - lc) / (rY + 1e-4));
        gate *= 1.0 - 0.5 * nz * p.limiter;
        float3 amp = sqrt(saturate(min(mn, 1.0 - mxc) / max(mxc, 1e-4)));
        float3 wgt = amp * (-1.0 / mix(8.0, 5.0, saturate(p.sharpness))) * gate;
        float3 outc = ((a + b + e + g) * wgt + c) / (1.0 + 4.0 * wgt);
        if (lc < 0.002) return c;
        float yN = clamp(dot(outc, lw), lc - 0.06, lc + 0.06);       // bounded delta
        yN = clamp(yN, lmn, lmx);                                     // no overshoot / halo
        float m = max(c.r, max(c.g, c.b));
        if (yN > lc) yN = lc + (yN - lc) * (1.0 - smoothstep(p.knee, min(p.knee + 0.15, 1.0), m));
        float ratio = min(yN / lc, 1.0 / max(m, 1e-4));
        return c * ratio;
    }

    // Fused form for a texture (present pass, mode 0): replaces cas_at when Clarity is on.
    inline float3 clarity_cas_at(texture2d<float> src, uint2 p, uint2 mx, constant ClarityCASParams &prm) {
        uint2 l = uint2(p.x > 0 ? p.x - 1 : 0, p.y), r = uint2(min(p.x + 1, mx.x), p.y);
        uint2 u = uint2(p.x, p.y > 0 ? p.y - 1 : 0), d = uint2(p.x, min(p.y + 1, mx.y));
        return clarity_cas(src.read(p).rgb, src.read(u).rgb, src.read(l).rgb, src.read(r).rgb, src.read(d).rgb, prm);
    }

    // Separate pass form: replaces cas_sharpen when Clarity is on.
    kernel void clarity_cas_sharpen(texture2d<float, access::read> src [[texture(0)]],
                                    texture2d<float, access::write> dst [[texture(1)]],
                                    constant ClarityCASParams &prm [[buffer(0)]],
                                    uint2 gid [[thread_position_in_grid]]) {
        uint w = src.get_width(), h = src.get_height();
        if (gid.x >= w || gid.y >= h) return;
        uint2 l = uint2(gid.x > 0 ? gid.x - 1 : 0, gid.y), r = uint2(min(gid.x + 1, w - 1), gid.y);
        uint2 u = uint2(gid.x, gid.y > 0 ? gid.y - 1 : 0), d = uint2(gid.x, min(gid.y + 1, h - 1));
        float3 o = clarity_cas(src.read(gid).rgb, src.read(u).rgb, src.read(l).rgb, src.read(r).rgb, src.read(d).rgb, prm);
        dst.write(float4(o, 1.0), gid);
    }
    """
}

/// Swift mirror of `ClarityYUVParams` (128 bytes).
public struct ClarityYUVParams {
    public var yOff: Float, yScale: Float, cScale: Float, rCr: Float, gCb: Float, gCr: Float, bCb: Float, cOff: Float
    public var siting: SIMD2<Float>
    public var codeScale: Float, invSigmaY: Float, guideMax: Float, deringStrength: Float, deringT: Float, guideMaxHigh: Float
    public var gFine: Float = 0, gMid: Float = 0, cap: Float = 0, rangeT: Float = 16
    public var flatClean: Float = 0, pad0: Float = 0, pad1: Float = 0, pad2: Float = 0
    public var hud0 = SIMD4<Float>(repeating: 0), hud1 = SIMD4<Float>(repeating: 0)

    public init(_ k: YUVCoefficients, siting: ChromaSiting, tenBit: Bool, deringStrength: Float,
         sigmaY: Float = ClarityTuning.sigmaY, guideMax: Float = ClarityTuning.guideMax) {
        yOff = k.yOffset; yScale = k.yScale; cScale = k.cScale; rCr = k.rCr; gCb = k.gCb; gCr = k.gCr; bCb = k.bCb; cOff = k.cOffset
        self.siting = siting.offset
        codeScale = ClarityTuning.codeScale(tenBit: tenBit)
        invSigmaY = 1 / sigmaY
        self.guideMax = guideMax
        guideMaxHigh = guideMax
        self.deringStrength = deringStrength
        deringT = ClarityTuning.deringT
    }

    /// Conversion parameters for a plan: de-ringer, guided-chroma cap, mid-band, flat cleanup, HUD rectangles
    /// (dilated by the 2 px filter radius).
    public init(_ k: YUVCoefficients, siting: ChromaSiting, tenBit: Bool, plan: ClarityPlan) {
        self.init(k, siting: siting, tenBit: tenBit, deringStrength: plan.deringStrength)
        apply(plan)
    }

    public mutating func apply(_ plan: ClarityPlan) {
        deringStrength = plan.deringStrength
        guideMaxHigh = plan.guideMaxHigh
        gFine = plan.midband.gFine; gMid = plan.midband.gMid; cap = plan.midband.cap; rangeT = plan.midband.rangeT
        flatClean = plan.flatClean
        let r = plan.hudRects.map { $0 + SIMD4(-2, -2, 2, 2) }
        hud0 = r.count > 0 ? r[0] : .zero
        hud1 = r.count > 1 ? r[1] : .zero
    }
}

/// Swift mirror of `ClarityCASParams` (32 bytes).
public struct ClarityCASParams {
    public var sharpness: Float, noise: Float, knee: Float, limiter: Float = 0
    public var lumaW: SIMD4<Float>

    /// `pq`: the work texture holds PQ / BT.2020 (HDR) instead of SDR R'G'B'. `edrPeak`: live headroom (1 = SDR
    /// screen); the taper starts where the present pass starts its highlight roll-off. `limiter`: v0.2 RCAS noise limiter.
    public init(sharpness: Float, noise: Float, pq: Bool, edrPeak: Float = 1000 / 203, limiter: Bool = false) {
        self.sharpness = sharpness
        self.noise = noise
        self.limiter = limiter ? 1 : 0
        lumaW = pq ? SIMD4(0.2627, 0.6780, 0.0593, 0) : SIMD4(0.2126, 0.7152, 0.0722, 0)
        if pq {
            // to_edr rolls off from 0.75 * peak (linear, 1.0 = 203 nits): the PQ code of that level.
            let nits = Double(max(1, edrPeak)) * 0.75 * 203
            knee = Float(ClarityEngine.pqEncode(nits / 10000))
        } else {
            knee = 0.85
        }
    }
}

/// Owns the Clarity pipelines. Standalone (its own library), or pass a host library that already compiled
/// `ClarityShaders.source`.
public final class ClarityEngine {
    public struct Variant: Hashable {
        public var guided = true, dering = false, raw = false
        /// v0.2 mid-band enhancer + flat cleanup (luma stage).
        public var mid = false
        /// v0.2 A/B: structure-tensor direction weights in the mid-band blurs.
        public var tensor = false
        /// Bench A/B: write linear light (rgba16Float destination).
        public var linear = false
        /// v0.2 A/B: 13 luma taps from a threadgroup tile.
        public var tile = false
        /// v0.2: 25-tap (5x5) B2.
        public var wide = false
        /// Plain-read reference kernel (4 chroma reads, no gather) instead of the production kernel. Self-check only.
        public var perPixel = false

        public init(guided: Bool = true, dering: Bool = false, raw: Bool = false, mid: Bool = false, tensor: Bool = false,
                    linear: Bool = false, tile: Bool = false, wide: Bool = false, perPixel: Bool = false) {
            self.guided = guided; self.dering = dering; self.raw = raw; self.mid = mid; self.tensor = tensor
            self.linear = linear; self.tile = tile; self.wide = wide; self.perPixel = perPixel
        }
    }

    public let device: MTLDevice
    public let library: MTLLibrary
    public let casPSO: MTLComputePipelineState
    private var convertPSOs: [Variant: MTLComputePipelineState] = [:]
    private var cache: CVMetalTextureCache?

    public init(device: MTLDevice, library: MTLLibrary? = nil) throws {
        self.device = device
        if let library { self.library = library } else {
            let opts = MTLCompileOptions()
            opts.mathMode = .fast
            self.library = try device.makeLibrary(source: ClarityShaders.source, options: opts)
        }
        casPSO = try device.makeComputePipelineState(function: self.library.makeFunction(name: "clarity_cas_sharpen")!)
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
    }

    public func convertPSO(_ v: Variant) throws -> MTLComputePipelineState {
        if let p = convertPSOs[v] { return p }
        let cv = MTLFunctionConstantValues()
        var g = v.guided, d = v.dering, r = v.raw, m = v.mid, t = v.tensor, l = v.linear, wd = v.wide
        cv.setConstantValue(&g, type: .bool, index: 40)
        cv.setConstantValue(&d, type: .bool, index: 41)
        cv.setConstantValue(&r, type: .bool, index: 42)
        cv.setConstantValue(&m, type: .bool, index: 43)
        cv.setConstantValue(&t, type: .bool, index: 44)
        cv.setConstantValue(&l, type: .bool, index: 45)
        cv.setConstantValue(&wd, type: .bool, index: 46)
        let name = v.perPixel ? "clarity_yuv_rgb_px" : v.tile ? "clarity_yuv_rgb_tile" : "clarity_yuv_rgb"
        let fn = try library.makeFunction(name: name, constantValues: cv)
        let p = try device.makeComputePipelineState(function: fn)
        convertPSOs[v] = p
        return p
    }

    /// Plane textures of a 420v / 420f / x420 / xf20 buffer. Keep `refs` alive until the GPU is done.
    public func planes(_ pb: CVPixelBuffer) -> (y: MTLTexture, c: MTLTexture, tenBit: Bool, refs: [CVMetalTexture])? {
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        let tenBit = fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        guard tenBit || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
              let cache else { return nil }
        var yRef: CVMetalTexture?, cRef: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, nil, tenBit ? .r16Unorm : .r8Unorm,
                                                  CVPixelBufferGetWidthOfPlane(pb, 0), CVPixelBufferGetHeightOfPlane(pb, 0), 0, &yRef)
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, nil, tenBit ? .rg16Unorm : .rg8Unorm,
                                                  CVPixelBufferGetWidthOfPlane(pb, 1), CVPixelBufferGetHeightOfPlane(pb, 1), 1, &cRef)
        guard let yRef, let cRef, let y = CVMetalTextureGetTexture(yRef), let c = CVMetalTextureGetTexture(cRef) else { return nil }
        return (y, c, tenBit, [yRef, cRef])
    }

    /// Conversion parameters of a buffer: its matrix, range and chroma siting attachments, plus the plan.
    public static func params(_ pb: CVPixelBuffer, deringStrength: Float, siting: ChromaSiting? = nil) -> ClarityYUVParams {
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        let tenBit = fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        let full = fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        let matrix = CVBufferCopyAttachment(pb, kCVImageBufferYCbCrMatrixKey, nil) as? String
        let k = YUVCoefficients.make(matrix: matrix, fullRange: full, tenBit: tenBit)
        return ClarityYUVParams(k, siting: siting ?? ChromaSiting(buffer: pb), tenBit: tenBit, deringStrength: deringStrength)
    }

    /// The conversion variant a plan needs (production kernel).
    public static func variant(for plan: ClarityPlan) -> Variant {
        let mid = plan.midband.active || plan.flatClean > 0
        // Direct reads: with the 13-tap B2 they are 0.03-0.07 ms faster than the 16x16 tile (the tile only wins with the
        // 25-tap B2, which is not shipped). CLARITY_PROGRESS.md › v0.2-beta.
        return Variant(guided: plan.chroma, dering: plan.deringStrength > 0, mid: mid, tile: false, wide: false)
    }

    /// `dst` has the luma plane's size.
    public func encodeConvert(_ enc: MTLComputeCommandEncoder, y: MTLTexture, c: MTLTexture, dst: MTLTexture,
                       params: ClarityYUVParams, variant: Variant) throws {
        let pso = try convertPSO(variant)
        var prm = params
        enc.setComputePipelineState(pso)
        enc.setTexture(y, index: 0)
        enc.setTexture(c, index: 1)
        enc.setTexture(dst, index: 2)
        enc.setBytes(&prm, length: MemoryLayout<ClarityYUVParams>.stride, index: 0)
        if variant.tile && !variant.perPixel {
            enc.dispatchThreadgroups(MTLSize(width: (dst.width + 15) / 16, height: (dst.height + 15) / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        } else {
            Self.dispatch(enc, pso, dst)
        }
    }

    public func encodeCAS(_ enc: MTLComputeCommandEncoder, src: MTLTexture, dst: MTLTexture, params: ClarityCASParams) {
        var prm = params
        enc.setComputePipelineState(casPSO)
        enc.setTexture(src, index: 0)
        enc.setTexture(dst, index: 1)
        enc.setBytes(&prm, length: MemoryLayout<ClarityCASParams>.stride, index: 0)
        Self.dispatch(enc, casPSO, dst)
    }

    public static func dispatch(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ dst: MTLTexture) {
        let w = pso.threadExecutionWidth, h = max(1, pso.maxTotalThreadsPerThreadgroup / w)
        enc.dispatchThreads(MTLSize(width: dst.width, height: dst.height, depth: 1), threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    }

    /// SMPTE ST 2084 inverse EOTF (linear 1.0 = 10000 nits).
    public static func pqEncode(_ l: Double) -> Double {
        let p = pow(max(l, 0), 0.1593017578125)
        return pow((0.8359375 + 18.8515625 * p) / (1 + 18.6875 * p), 78.84375)
    }
}
