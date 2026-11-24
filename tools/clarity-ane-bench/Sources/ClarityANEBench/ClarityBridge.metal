#include <metal_stdlib>
using namespace metal;

// packed: one thread per 2x2 source cell, stacked NCHW plane storage in an r16Float IOSurface.
kernel void clarity_pack(texture2d<float, access::read> x420Y [[texture(0)]],
                         texture2d<half, access::write> mlInput [[texture(1)]],
                         constant uint &packed [[buffer(0)]],
                         uint2 p [[thread_position_in_grid]]) {
    uint width = x420Y.get_width() / (packed ? 2 : 1);
    uint height = x420Y.get_height() / (packed ? 2 : 1);
    if (p.x >= width || p.y >= height) return;
    if (!packed) {
        float code = x420Y.read(p).r * (65535.0f / 64.0f);
        mlInput.write(half4(half(saturate((code - 64.0f) / 876.0f))), p);
        return;
    }
    for (uint phase = 0; phase != 4; ++phase) {
        uint2 source = 2*p + uint2(phase & 1, phase >> 1);
        float code = x420Y.read(source).r * (65535.0f / 64.0f);
        float y = saturate((code - 64.0f) / 876.0f);
        mlInput.write(half4(half(y)), uint2(p.x, p.y + phase*height));
    }
}

// Point form for existing conversion. For Motion's subpixel sample, unpack and
// bilerp four source-lattice residuals, or materialize the unpacked scalar plane
// and sample it with the same coordinates as Y. Never filter across phase planes.
inline float clarity_delta_code(texture2d<half, access::read> residual,
                                 uint2 p, uint fullHeight, bool packed) {
    uint phase = ((p.y & 1) << 1) | (p.x & 1);
    uint2 q = packed ? uint2(p.x >> 1, (p.y >> 1) + phase*(fullHeight >> 1)) : p;
    return clamp(float(residual.read(q).r) * 876.0f, -4.0f, 4.0f);
}
