// M1 Max prefill GEMM roofline arms (Phase 2B Step A, bench-only).
// Production q27_matmul_q4_mm_h is timed from the engine source by the
// harness; these arms decompose its time:
//   pf_stream   : same grid, raw weight-byte reads only (achievable stream
//                 at this parallelism; y-groups re-read like production)
//   pf_lut      : + LUT half2 gather into smem (dequant staging, no MMA)
//   pf_mma_peak : + full MMA from pre-staged smem (no device reads) -> MMA
//                 ceiling for this tile geometry
// GB/s in the harness is unique weight bytes / time; multiply by
// ceil(x_rows/16) for the device-traffic view.
#include <metal_stdlib>
using namespace metal;

struct MatmulArgs { uint rows; uint cols; uint x_rows; uint simdgroups; };

kernel void pf_stream(device const uchar *weights [[buffer(0)]],
                      device float *out [[buffer(4)]],
                      constant MatmulArgs &args [[buffer(5)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    const uint row0 = group.x * 32;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 2);
    float s = 0.0f;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        const uint2 wp = *(device const uint2 *)(wsrc + (c0 + wcb) / 2);
        s += float(wp.x) + float(wp.y);
    }
    if (s == 123456.0f) out[row0 + wrow] = s;   // keep, never true
}

constant half2 pf_luttab[256] = {};   // placeholder zero-filled LUT (timing only)

kernel void pf_lut(device const uchar *weights [[buffer(0)]],
                   device float *out [[buffer(4)]],
                   constant MatmulArgs &args [[buffer(5)]],
                   uint2 group [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    const uint row0 = group.x * 32;
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 2);
    float s = 0.0f;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        const uint2 wp = *(device const uint2 *)(wsrc + (c0 + wcb) / 2);
        threadgroup half2 *dst = (threadgroup half2 *)(Wt + wrow * 64 + wcb);
        dst[0] = pf_luttab[wp.x & 0xffu]; dst[1] = pf_luttab[(wp.x >> 8) & 0xffu];
        dst[2] = pf_luttab[(wp.x >> 16) & 0xffu]; dst[3] = pf_luttab[wp.x >> 24];
        dst[4] = pf_luttab[wp.y & 0xffu]; dst[5] = pf_luttab[(wp.y >> 8) & 0xffu];
        dst[6] = pf_luttab[(wp.y >> 16) & 0xffu]; dst[7] = pf_luttab[wp.y >> 24];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s += float(Wt[wrow * 64 + (tid % 4) * 16]);
    }
    if (s == 123456.0f) out[row0 + wrow] = s;
}

// MMA ceiling: same grid/threads, constant smem operands, pure tensor-core
// math; final simdgroup_store keeps accumulators live (no DCE). No feedback
// chain -> measures practical peak for this tile shape.
kernel void pf_mma_peak(device float *out [[buffer(4)]],
                        constant MatmulArgs &args [[buffer(5)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group.x * 32;
    if (row0 >= args.rows) return;
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5h));
    simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8, 8>(half(0.25h));
    const uint iters = args.cols / 64;   // 4 muls per 64-col step == production per group
    for (uint i = 0; i < iters; i++) {
        simdgroup_multiply(acc0, a0, b0);
        simdgroup_multiply(acc1, a0, b0);
        simdgroup_multiply(acc2, a0, b0);
        simdgroup_multiply(acc3, a0, b0);
    }
    threadgroup float st[4][64];
    simdgroup_store(acc0, *(threadgroup float (*)[64])&st[0]);
    simdgroup_store(acc1, *(threadgroup float (*)[64])&st[1]);
    simdgroup_store(acc2, *(threadgroup float (*)[64])&st[2]);
    simdgroup_store(acc3, *(threadgroup float (*)[64])&st[3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint k = 0; k < 4; k++) s += st[k][lane];
    if (s == 999999.0f) out[row0] = s;
}
