// bench-only companion for tools/roofline_m1.mm: pure uint4 stream at the
using namespace metal;

// Copy of q27_dot8_q4 from src/metal/q27_kernels.metal (bench-only clone;
// keep in sync when touching the B0b dot arms).
inline int q27_dot8_q4(uint packed, char4 x0, char4 x1) {
    int sum = (int(packed         & 15u) - 8) * x0.x;
    sum += (int((packed >>  4) & 15u) - 8) * x0.y;
    sum += (int((packed >>  8) & 15u) - 8) * x0.z;
    sum += (int((packed >> 12) & 15u) - 8) * x0.w;
    sum += (int((packed >> 16) & 15u) - 8) * x1.x;
    sum += (int((packed >> 20) & 15u) - 8) * x1.y;
    sum += (int((packed >> 24) & 15u) - 8) * x1.z;
    sum += (int((packed >> 28)      ) - 8) * x1.w;
    return sum;
}
// production matvec dispatch geometry (32 rows/threadgroup, 256 threads,
// lane-contiguous 16B reads), zero quant math. NOT engine code.
kernel void bench_stream(device const uchar *weights [[buffer(0)]],
                         device float *out           [[buffer(4)]],
                         constant uint2 &shape       [[buffer(5)]],
                         uint group [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]],
                         ushort sg  [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (group * 8 + (uint)sg) * 4;
    const uint rows = shape.x, cols = shape.y;
    const uint chunks = cols / 1024;
    uint acc = 0;
    for (uint r = 0; r < 4; r++) {
        const uint row = min(row0 + r, rows - 1);
        device const uint4 *w = (device const uint4 *)(weights + (ulong)row * (cols / 2));
        for (uint chunk = 0; chunk < chunks; chunk++) {
            const uint4 wp = w[chunk * 32 + lane];
            acc += wp.x + wp.y + wp.z + wp.w;
        }
    }
    if (lane == 0) out[row0] = float(acc);
}

// B0 decomposition arms (2026-09-23): production loop structure (chunk
// outer, xp once per chunk, 4-row inner loop with w[4]/sbase[4]) so each
// arm differs from q27_matvec_q4_quantized by exactly one resource class.
// bench_stream2 re-runs the pure stream in production's structure;
// bench_stream above (row-outer) was the Step A roofline number.

kernel void bench_stream2(device const uchar *weights [[buffer(0)]],
                          device float *out           [[buffer(4)]],
                          constant uint2 &shape       [[buffer(5)]],
                          uint group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg  [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (group * 8 + (uint)sg) * 4;
    const uint rows = shape.x, cols = shape.y;
    device const uint4 *w[4];
    for (uint r = 0; r < 4; r++)
        w[r] = (device const uint4 *)(weights + (ulong)min(row0 + r, rows - 1) * (cols / 2));
    uint acc = 0;
    const uint chunks = cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        for (uint r = 0; r < 4; r++) {
            const uint4 wp = w[r][idx];
            acc += wp.x + wp.y + wp.z + wp.w;
        }
    }
    if (lane == 0) out[row0] = float(acc);
}

// A1 = bench_stream2 + production's x loads (2x int4 per lane per chunk,
// identical indices to the production kernel). Consumption trivial: measures
// the load-slot and L2 cost of the x redundancy only.
kernel void bench_stream_x(device const uchar *weights [[buffer(0)]],
                           device const char *x         [[buffer(2)]],
                           device float *out            [[buffer(4)]],
                           constant uint2 &shape        [[buffer(5)]],
                           uint group [[threadgroup_position_in_grid]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort sg  [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (group * 8 + (uint)sg) * 4;
    const uint rows = shape.x, cols = shape.y;
    device const uint4 *w[4];
    for (uint r = 0; r < 4; r++)
        w[r] = (device const uint4 *)(weights + (ulong)min(row0 + r, rows - 1) * (cols / 2));
    device const int4 *x16 = (device const int4 *)x;
    uint acc = 0;
    const uint chunks = cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const int4 xp0 = x16[idx * 2];
        const int4 xp1 = x16[idx * 2 + 1];
        for (uint r = 0; r < 4; r++) {
            const uint4 wp = w[r][idx];
            acc += wp.x + wp.y + wp.z + wp.w + (uint)(xp0.x ^ xp1.x);
        }
    }
    if (lane == 0) out[row0] = float(acc);
}

// A2 = A1 + both scale streams with production indexing and one fp32
// multiply chain per row: everything except nibble unpacking and the
// eight 8-bit dot products.
kernel void bench_stream_sc(device const uchar *weights      [[buffer(0)]],
                            device const half *weight_scales [[buffer(1)]],
                            device const char *x             [[buffer(2)]],
                            device const float *x_scales     [[buffer(3)]],
                            device float *out                [[buffer(4)]],
                            constant uint2 &shape            [[buffer(5)]],
                            uint group [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort sg  [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (group * 8 + (uint)sg) * 4;
    const uint rows = shape.x, cols = shape.y;
    const uint sgroups = cols / 64;
    device const uint4 *w[4];
    ulong sbase[4];
    for (uint r = 0; r < 4; r++) {
        w[r] = (device const uint4 *)(weights + (ulong)min(row0 + r, rows - 1) * (cols / 2));
        sbase[r] = (ulong)min(row0 + r, rows - 1) * sgroups;
    }
    device const int4 *x16 = (device const int4 *)x;
    float4 acc = 0.0f;
    const uint chunks = cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const int4 xp0 = x16[idx * 2];
        const int4 xp1 = x16[idx * 2 + 1];
        const uint c = chunk * 1024 + lane * 32;
        const float xs = x_scales[c / 32];
        const int raw = xp0.x + xp0.y + xp0.z + xp0.w + xp1.x + xp1.y + xp1.z + xp1.w;
        for (uint r = 0; r < 4; r++) {
            const uint4 wp = w[r][idx];
            const float dot = float(wp.x + wp.y + wp.z + wp.w) * 0.001f + float(raw);
            acc[r] += dot * float(weight_scales[sbase[r] + c / 64]) * xs;
        }
    }
    for (uint r = 0; r < 4; r++)
        if (lane == r && row0 + r < rows) out[row0 + r] = acc.x + acc.y + acc.z + acc.w;
}

// B0b: dot-count scaling arms. dot2 runs the real nibble dot on HALF the
// weight words (other words consumed by a cheap add), dot4 clones the
// production arithmetic fully. If time scales linearly with dot word count
// the kernel is ALU-bound; a convex jump between dot2 and dot4 says the
// register/scheduler cliff, not raw op count, is the wall.
kernel void bench_sc_dot2(device const uchar *weights      [[buffer(0)]],
                          device const half *weight_scales [[buffer(1)]],
                          device const char *x             [[buffer(2)]],
                          device const float *x_scales     [[buffer(3)]],
                          device float *out                [[buffer(4)]],
                          constant uint2 &shape            [[buffer(5)]],
                          uint group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg  [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (group * 8 + (uint)sg) * 4;
    const uint rows = shape.x, cols = shape.y;
    const uint sgroups = cols / 64;
    device const uint4 *w[4];
    ulong sbase[4];
    for (uint r = 0; r < 4; r++) {
        w[r] = (device const uint4 *)(weights + (ulong)min(row0 + r, rows - 1) * (cols / 2));
        sbase[r] = (ulong)min(row0 + r, rows - 1) * sgroups;
    }
    device const int4 *x16 = (device const int4 *)x;
    float4 acc = 0.0f;
    const uint chunks = cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const int4 xp0 = x16[idx * 2];
        const int4 xp1 = x16[idx * 2 + 1];
        const uint c = chunk * 1024 + lane * 32;
        const float xs = x_scales[c / 32];
        for (uint r = 0; r < 4; r++) {
            const uint4 wp = w[r][idx];
            const int dot0 = q27_dot8_q4(wp.x, as_type<char4>(xp0.x), as_type<char4>(xp0.y)) +
                             q27_dot8_q4(wp.y, as_type<char4>(xp0.z), as_type<char4>(xp0.w));
            // words z/w consumed without nibble math
            const int dot = dot0 + wp.z + wp.w;
            acc[r] += float(dot) * float(weight_scales[sbase[r] + c / 64]) * xs;
        }
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = tot;
    }
}

kernel void bench_sc_dot4(device const uchar *weights      [[buffer(0)]],
                          device const half *weight_scales [[buffer(1)]],
                          device const char *x             [[buffer(2)]],
                          device const float *x_scales     [[buffer(3)]],
                          device float *out                [[buffer(4)]],
                          constant uint2 &shape            [[buffer(5)]],
                          uint group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg  [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (group * 8 + (uint)sg) * 4;
    const uint rows = shape.x, cols = shape.y;
    const uint sgroups = cols / 64;
    device const uint4 *w[4];
    ulong sbase[4];
    for (uint r = 0; r < 4; r++) {
        w[r] = (device const uint4 *)(weights + (ulong)min(row0 + r, rows - 1) * (cols / 2));
        sbase[r] = (ulong)min(row0 + r, rows - 1) * sgroups;
    }
    device const int4 *x16 = (device const int4 *)x;
    float4 acc = 0.0f;
    const uint chunks = cols / 1024;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const int4 xp0 = x16[idx * 2];
        const int4 xp1 = x16[idx * 2 + 1];
        const uint c = chunk * 1024 + lane * 32;
        const float xs = x_scales[c / 32];
        for (uint r = 0; r < 4; r++) {
            const uint4 wp = w[r][idx];
            const int dot0 = q27_dot8_q4(wp.x, as_type<char4>(xp0.x), as_type<char4>(xp0.y)) +
                             q27_dot8_q4(wp.y, as_type<char4>(xp0.z), as_type<char4>(xp0.w));
            const int dot1 = q27_dot8_q4(wp.z, as_type<char4>(xp1.x), as_type<char4>(xp1.y)) +
                             q27_dot8_q4(wp.w, as_type<char4>(xp1.z), as_type<char4>(xp1.w));
            acc[r] += float(dot0 + dot1) * float(weight_scales[sbase[r] + c / 64]) * xs;
        }
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = tot;
    }
}
