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

// B2a-diag: same x/scale/dot operations as bench_sc_dot4 but the nibble
// math runs on CONSTANT words instead of the loaded ones (loaded words are
// consumed by a single add so they stay live). Ops/insts are identical to
// dot4; only the load->math dependency is cut. dot4-level time => the wall
// is raw op count/scheduling of the dot itself; stream-level time => it is
// the load-consumer dependency, and restructuring (prefetch/ILP) wins.
kernel void bench_sc_constdot(device const uchar *weights      [[buffer(0)]],
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
    const uint cw0 = 0x12345678u, cw1 = 0x9abcdef0u, cw2 = 0x0f1e2d3cu, cw3 = 0x76543210u;
    const uint chunks = cols / 1024;
    uint64_t lsum = 0;
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const int4 xp0 = x16[idx * 2];
        const int4 xp1 = x16[idx * 2 + 1];
        const uint c = chunk * 1024 + lane * 32;
        const float xs = x_scales[c / 32];
        for (uint r = 0; r < 4; r++) {
            const uint4 wp = w[r][idx];
            lsum += uint64_t(wp.x) + wp.y + wp.z + wp.w;    // keep loads live
            const int dot0 = q27_dot8_q4(cw0, as_type<char4>(xp0.x), as_type<char4>(xp0.y)) +
                             q27_dot8_q4(cw1, as_type<char4>(xp0.z), as_type<char4>(xp0.w));
            const int dot1 = q27_dot8_q4(cw2, as_type<char4>(xp1.x), as_type<char4>(xp1.y)) +
                             q27_dot8_q4(cw3, as_type<char4>(xp1.z), as_type<char4>(xp1.w));
            acc[r] += float(dot0 + dot1) * float(weight_scales[sbase[r] + c / 64]) * xs;
        }
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = tot + float(lsum) * 1e-9f;
    }
}

// B2a: exact fp16 magic-number nibble dot. as_half(n<<10) == n/1024 exactly
// (n in [0,15], normal range); products with half(x) (|x|<=127 exact in
// half) are <= 15*127/1024 < 2, needing <= 11 mantissa bits -> exact half
// values; accumulation happens in fp32 on values dot/1024 with |sum| < 32
// and ulp(fp32) headroom, and the final float(dot) is (sum*1024)/1024 — the
// SAME exact integer as the production dot, then the same float*ws*xs
// chain. -8 folds into Sx once per lane-chunk: sum((c-8)x) = sum(cx) - 8Sx.
// Extract: two columns per shift+mask+or (half2 with 0x6400 base, minus
// 1032 bias vector per word-pair multiply) — see bit-pairing comment.
inline float2 q27_half2_nib2(uint shifted_masked) {   // {1024+na, 1024+nb} low halfs
    return float2(as_type<half2>(shifted_masked | 0x64006400u)) - float2(1032.0h);
}

kernel void bench_sc_halfdot4(device const uchar *weights      [[buffer(0)]],
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
        // x as float columns xcol[32] (column k of the lane's 32-col slice).
        float xcol[32];
        xcol[0]  = float(as_type<char4>(xp0.x).x); xcol[1]  = float(as_type<char4>(xp0.x).y);
        xcol[2]  = float(as_type<char4>(xp0.x).z); xcol[3]  = float(as_type<char4>(xp0.x).w);
        xcol[4]  = float(as_type<char4>(xp0.y).x); xcol[5]  = float(as_type<char4>(xp0.y).y);
        xcol[6]  = float(as_type<char4>(xp0.y).z); xcol[7]  = float(as_type<char4>(xp0.y).w);
        xcol[8]  = float(as_type<char4>(xp0.z).x); xcol[9]  = float(as_type<char4>(xp0.z).y);
        xcol[10] = float(as_type<char4>(xp0.z).z); xcol[11] = float(as_type<char4>(xp0.z).w);
        xcol[12] = float(as_type<char4>(xp0.w).x); xcol[13] = float(as_type<char4>(xp0.w).y);
        xcol[14] = float(as_type<char4>(xp0.w).z); xcol[15] = float(as_type<char4>(xp0.w).w);
        xcol[16] = float(as_type<char4>(xp1.x).x); xcol[17] = float(as_type<char4>(xp1.x).y);
        xcol[18] = float(as_type<char4>(xp1.x).z); xcol[19] = float(as_type<char4>(xp1.x).w);
        xcol[20] = float(as_type<char4>(xp1.y).x); xcol[21] = float(as_type<char4>(xp1.y).y);
        xcol[22] = float(as_type<char4>(xp1.y).z); xcol[23] = float(as_type<char4>(xp1.y).w);
        xcol[24] = float(as_type<char4>(xp1.z).x); xcol[25] = float(as_type<char4>(xp1.z).y);
        xcol[26] = float(as_type<char4>(xp1.z).z); xcol[27] = float(as_type<char4>(xp1.z).w);
        xcol[28] = float(as_type<char4>(xp1.w).x); xcol[29] = float(as_type<char4>(xp1.w).y);
        xcol[30] = float(as_type<char4>(xp1.w).z); xcol[31] = float(as_type<char4>(xp1.w).w);
        for (uint r = 0; r < 4; r++) {
            const uint4 wp = w[r][idx];
            const uint wds[4] = {wp.x, wp.y, wp.z, wp.w};
            float dot = 0.0f;
            #pragma unroll
            for (uint k = 0; k < 4; k++) {
                const uint wv = wds[k];
                // Column bit order in the word: col0=bits0-3, col1=4-7,
                // ..., col7=28-31. Extraction (wv>>4s)&0x000F000F yields the
                // half2 pair {col s, col s+4}: shifts 0..3 give pairs
                // {0,4},{1,5},{2,6},{3,7}; matched to xcol below.
                const float2 nA0 = q27_half2_nib2((wv      ) & 0x000F000Fu); // cols 0,4
                const float2 nA1 = q27_half2_nib2((wv >>  4) & 0x000F000Fu); // cols 1,5
                const float2 nA2 = q27_half2_nib2((wv >>  8) & 0x000F000Fu); // cols 2,6
                const float2 nA3 = q27_half2_nib2((wv >> 12) & 0x000F000Fu); // cols 3,7
                const uint base = k * 8;
                dot += nA0.x * xcol[base + 0] + nA0.y * xcol[base + 4]
                     + nA1.x * xcol[base + 1] + nA1.y * xcol[base + 5]
                     + nA2.x * xcol[base + 2] + nA2.y * xcol[base + 6]
                     + nA3.x * xcol[base + 3] + nA3.y * xcol[base + 7];
            }
            // q27_half2_nib2 returns (n-8) exactly (0x6400|n = 1024+n,
            // minus 1032), so dot == production's sum((c-8)x), exact.
            acc[r] += dot * float(weight_scales[sbase[r] + c / 64]) * xs;
        }
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = tot;
    }
}

// B2b: halfdot4 + 2-stage software pipeline (constdot proved the wall is the
// load->math dependency, not op throughput). The next chunk's weight words
// and x words are loaded before the current chunk's math runs, so the
// DRAM-latency wait overlaps the unpack/multiply chain. Math is identical
// to bench_sc_halfdot4 (bit-identity gate applies).
kernel void bench_sc_halfdot4p(device const uchar *weights      [[buffer(0)]],
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
    // prime the pipeline
    int4 xp0 = x16[lane * 2], xp1 = x16[lane * 2 + 1];
    uint4 wp[4];
    for (uint r = 0; r < 4; r++) wp[r] = w[r][lane];
    for (uint chunk = 0; chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const uint nidx = idx + 32;
        // prefetch next chunk (predicated store into regs; the extra load
        // past the end is clamped by chunks being >=1: guard with nidx)
        int4 nxp0 = xp0, nxp1 = xp1;
        uint4 nwp[4];
        if (chunk + 1 < chunks) {
            nxp0 = x16[nidx * 2]; nxp1 = x16[nidx * 2 + 1];
            for (uint r = 0; r < 4; r++) nwp[r] = w[r][nidx];
        }
        const uint c = chunk * 1024 + lane * 32;
        const float xs = x_scales[c / 32];
        float xcol[32];
        xcol[0]  = float(as_type<char4>(xp0.x).x); xcol[1]  = float(as_type<char4>(xp0.x).y);
        xcol[2]  = float(as_type<char4>(xp0.x).z); xcol[3]  = float(as_type<char4>(xp0.x).w);
        xcol[4]  = float(as_type<char4>(xp0.y).x); xcol[5]  = float(as_type<char4>(xp0.y).y);
        xcol[6]  = float(as_type<char4>(xp0.y).z); xcol[7]  = float(as_type<char4>(xp0.y).w);
        xcol[8]  = float(as_type<char4>(xp0.z).x); xcol[9]  = float(as_type<char4>(xp0.z).y);
        xcol[10] = float(as_type<char4>(xp0.z).z); xcol[11] = float(as_type<char4>(xp0.z).w);
        xcol[12] = float(as_type<char4>(xp0.w).x); xcol[13] = float(as_type<char4>(xp0.w).y);
        xcol[14] = float(as_type<char4>(xp0.w).z); xcol[15] = float(as_type<char4>(xp0.w).w);
        xcol[16] = float(as_type<char4>(xp1.x).x); xcol[17] = float(as_type<char4>(xp1.x).y);
        xcol[18] = float(as_type<char4>(xp1.x).z); xcol[19] = float(as_type<char4>(xp1.x).w);
        xcol[20] = float(as_type<char4>(xp1.y).x); xcol[21] = float(as_type<char4>(xp1.y).y);
        xcol[22] = float(as_type<char4>(xp1.y).z); xcol[23] = float(as_type<char4>(xp1.y).w);
        xcol[24] = float(as_type<char4>(xp1.z).x); xcol[25] = float(as_type<char4>(xp1.z).y);
        xcol[26] = float(as_type<char4>(xp1.z).z); xcol[27] = float(as_type<char4>(xp1.z).w);
        xcol[28] = float(as_type<char4>(xp1.w).x); xcol[29] = float(as_type<char4>(xp1.w).y);
        xcol[30] = float(as_type<char4>(xp1.w).z); xcol[31] = float(as_type<char4>(xp1.w).w);
        for (uint r = 0; r < 4; r++) {
            const uint wds[4] = {wp[r].x, wp[r].y, wp[r].z, wp[r].w};
            float dot = 0.0f;
            #pragma unroll
            for (uint k = 0; k < 4; k++) {
                const uint wv = wds[k];
                const float2 nA0 = q27_half2_nib2((wv     ) & 0x000F000Fu);
                const float2 nA1 = q27_half2_nib2((wv >> 4) & 0x000F000Fu);
                const float2 nA2 = q27_half2_nib2((wv >> 8) & 0x000F000Fu);
                const float2 nA3 = q27_half2_nib2((wv >>12) & 0x000F000Fu);
                const uint base = k * 8;
                dot += nA0.x * xcol[base + 0] + nA0.y * xcol[base + 4]
                     + nA1.x * xcol[base + 1] + nA1.y * xcol[base + 5]
                     + nA2.x * xcol[base + 2] + nA2.y * xcol[base + 6]
                     + nA3.x * xcol[base + 3] + nA3.y * xcol[base + 7];
            }
            acc[r] += dot * float(weight_scales[sbase[r] + c / 64]) * xs;
        }
        xp0 = nxp0; xp1 = nxp1;
        for (uint r = 0; r < 4; r++) wp[r] = nwp[r];
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = tot;
    }
}


// B2b-alt: independent 2-chunk unroll of the halfdot math (bit-identical to
// bench_sc_halfdot4; two in-flight load streams, no double-buffer regs).
kernel void bench_sc_halfdot4u2(device const uchar *weights      [[buffer(0)]],
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
    for (uint chunk = 0; chunk + 1 < chunks; chunk += 2) {
        #pragma unroll
        for (uint h = 0; h < 2; h++) {
            const uint idx = (chunk + h) * 32 + lane;
            const int4 xp0 = x16[idx * 2];
            const int4 xp1 = x16[idx * 2 + 1];
            const uint c = (chunk + h) * 1024 + lane * 32;
            const float xs = x_scales[c / 32];
            float xcol[32];
            xcol[0]  = float(as_type<char4>(xp0.x).x); xcol[1]  = float(as_type<char4>(xp0.x).y);
            xcol[2]  = float(as_type<char4>(xp0.x).z); xcol[3]  = float(as_type<char4>(xp0.x).w);
            xcol[4]  = float(as_type<char4>(xp0.y).x); xcol[5]  = float(as_type<char4>(xp0.y).y);
            xcol[6]  = float(as_type<char4>(xp0.y).z); xcol[7]  = float(as_type<char4>(xp0.y).w);
            xcol[8]  = float(as_type<char4>(xp0.z).x); xcol[9]  = float(as_type<char4>(xp0.z).y);
            xcol[10] = float(as_type<char4>(xp0.z).z); xcol[11] = float(as_type<char4>(xp0.z).w);
            xcol[12] = float(as_type<char4>(xp0.w).x); xcol[13] = float(as_type<char4>(xp0.w).y);
            xcol[14] = float(as_type<char4>(xp0.w).z); xcol[15] = float(as_type<char4>(xp0.w).w);
            xcol[16] = float(as_type<char4>(xp1.x).x); xcol[17] = float(as_type<char4>(xp1.x).y);
            xcol[18] = float(as_type<char4>(xp1.x).z); xcol[19] = float(as_type<char4>(xp1.x).w);
            xcol[20] = float(as_type<char4>(xp1.y).x); xcol[21] = float(as_type<char4>(xp1.y).y);
            xcol[22] = float(as_type<char4>(xp1.y).z); xcol[23] = float(as_type<char4>(xp1.y).w);
            xcol[24] = float(as_type<char4>(xp1.z).x); xcol[25] = float(as_type<char4>(xp1.z).y);
            xcol[26] = float(as_type<char4>(xp1.z).z); xcol[27] = float(as_type<char4>(xp1.z).w);
            xcol[28] = float(as_type<char4>(xp1.w).x); xcol[29] = float(as_type<char4>(xp1.w).y);
            xcol[30] = float(as_type<char4>(xp1.w).z); xcol[31] = float(as_type<char4>(xp1.w).w);
            for (uint r = 0; r < 4; r++) {
                const uint wds[4] = {w[r][idx].x, w[r][idx].y, w[r][idx].z, w[r][idx].w};
                float dot = 0.0f;
                #pragma unroll
                for (uint k = 0; k < 4; k++) {
                    const uint wv = wds[k];
                    const float2 nA0 = q27_half2_nib2((wv     ) & 0x000F000Fu);
                    const float2 nA1 = q27_half2_nib2((wv >> 4) & 0x000F000Fu);
                    const float2 nA2 = q27_half2_nib2((wv >> 8) & 0x000F000Fu);
                    const float2 nA3 = q27_half2_nib2((wv >>12) & 0x000F000Fu);
                    const uint base = k * 8;
                    dot += nA0.x * xcol[base + 0] + nA0.y * xcol[base + 4]
                         + nA1.x * xcol[base + 1] + nA1.y * xcol[base + 5]
                         + nA2.x * xcol[base + 2] + nA2.y * xcol[base + 6]
                         + nA3.x * xcol[base + 3] + nA3.y * xcol[base + 7];
                }
                acc[r] += dot * float(weight_scales[sbase[r] + c / 64]) * xs;
            }
        }
    }
    for (uint chunk = chunks - (chunks & 1); chunk < chunks; chunk++) {
        const uint idx = chunk * 32 + lane;
        const int4 xp0 = x16[idx * 2];
        const int4 xp1 = x16[idx * 2 + 1];
        const uint c = chunk * 1024 + lane * 32;
        const float xs = x_scales[c / 32];
        float xcol[32];
        xcol[0]  = float(as_type<char4>(xp0.x).x); xcol[1]  = float(as_type<char4>(xp0.x).y);
        xcol[2]  = float(as_type<char4>(xp0.x).z); xcol[3]  = float(as_type<char4>(xp0.x).w);
        xcol[4]  = float(as_type<char4>(xp0.y).x); xcol[5]  = float(as_type<char4>(xp0.y).y);
        xcol[6]  = float(as_type<char4>(xp0.y).z); xcol[7]  = float(as_type<char4>(xp0.y).w);
        xcol[8]  = float(as_type<char4>(xp0.z).x); xcol[9]  = float(as_type<char4>(xp0.z).y);
        xcol[10] = float(as_type<char4>(xp0.z).z); xcol[11] = float(as_type<char4>(xp0.z).w);
        xcol[12] = float(as_type<char4>(xp0.w).x); xcol[13] = float(as_type<char4>(xp0.w).y);
        xcol[14] = float(as_type<char4>(xp0.w).z); xcol[15] = float(as_type<char4>(xp0.w).w);
        xcol[16] = float(as_type<char4>(xp1.x).x); xcol[17] = float(as_type<char4>(xp1.x).y);
        xcol[18] = float(as_type<char4>(xp1.x).z); xcol[19] = float(as_type<char4>(xp1.x).w);
        xcol[20] = float(as_type<char4>(xp1.y).x); xcol[21] = float(as_type<char4>(xp1.y).y);
        xcol[22] = float(as_type<char4>(xp1.y).z); xcol[23] = float(as_type<char4>(xp1.y).w);
        xcol[24] = float(as_type<char4>(xp1.z).x); xcol[25] = float(as_type<char4>(xp1.z).y);
        xcol[26] = float(as_type<char4>(xp1.z).z); xcol[27] = float(as_type<char4>(xp1.z).w);
        xcol[28] = float(as_type<char4>(xp1.w).x); xcol[29] = float(as_type<char4>(xp1.w).y);
        xcol[30] = float(as_type<char4>(xp1.w).z); xcol[31] = float(as_type<char4>(xp1.w).w);
        for (uint r = 0; r < 4; r++) {
            const uint wds[4] = {w[r][idx].x, w[r][idx].y, w[r][idx].z, w[r][idx].w};
            float dot = 0.0f;
            #pragma unroll
            for (uint k = 0; k < 4; k++) {
                const uint wv = wds[k];
                const float2 nA0 = q27_half2_nib2((wv     ) & 0x000F000Fu);
                const float2 nA1 = q27_half2_nib2((wv >> 4) & 0x000F000Fu);
                const float2 nA2 = q27_half2_nib2((wv >> 8) & 0x000F000Fu);
                const float2 nA3 = q27_half2_nib2((wv >>12) & 0x000F000Fu);
                const uint base = k * 8;
                dot += nA0.x * xcol[base + 0] + nA0.y * xcol[base + 4]
                     + nA1.x * xcol[base + 1] + nA1.y * xcol[base + 5]
                     + nA2.x * xcol[base + 2] + nA2.y * xcol[base + 6]
                     + nA3.x * xcol[base + 3] + nA3.y * xcol[base + 7];
            }
            acc[r] += dot * float(weight_scales[sbase[r] + c / 64]) * xs;
        }
    }
    for (uint r = 0; r < 4; r++) {
        const float tot = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) out[row0 + r] = tot;
    }
}
