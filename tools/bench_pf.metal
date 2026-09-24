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

// --- production-copy arms (B1 x-reuse / B2 double-buffer), from q27_kernels.metal @ 4929153 ---
constant half2 q27_q4_half2_lut[256] = {
    half2(-8.0h, -8.0h),
    half2(-7.0h, -8.0h),
    half2(-6.0h, -8.0h),
    half2(-5.0h, -8.0h),
    half2(-4.0h, -8.0h),
    half2(-3.0h, -8.0h),
    half2(-2.0h, -8.0h),
    half2(-1.0h, -8.0h),
    half2(0.0h, -8.0h),
    half2(1.0h, -8.0h),
    half2(2.0h, -8.0h),
    half2(3.0h, -8.0h),
    half2(4.0h, -8.0h),
    half2(5.0h, -8.0h),
    half2(6.0h, -8.0h),
    half2(7.0h, -8.0h),
    half2(-8.0h, -7.0h),
    half2(-7.0h, -7.0h),
    half2(-6.0h, -7.0h),
    half2(-5.0h, -7.0h),
    half2(-4.0h, -7.0h),
    half2(-3.0h, -7.0h),
    half2(-2.0h, -7.0h),
    half2(-1.0h, -7.0h),
    half2(0.0h, -7.0h),
    half2(1.0h, -7.0h),
    half2(2.0h, -7.0h),
    half2(3.0h, -7.0h),
    half2(4.0h, -7.0h),
    half2(5.0h, -7.0h),
    half2(6.0h, -7.0h),
    half2(7.0h, -7.0h),
    half2(-8.0h, -6.0h),
    half2(-7.0h, -6.0h),
    half2(-6.0h, -6.0h),
    half2(-5.0h, -6.0h),
    half2(-4.0h, -6.0h),
    half2(-3.0h, -6.0h),
    half2(-2.0h, -6.0h),
    half2(-1.0h, -6.0h),
    half2(0.0h, -6.0h),
    half2(1.0h, -6.0h),
    half2(2.0h, -6.0h),
    half2(3.0h, -6.0h),
    half2(4.0h, -6.0h),
    half2(5.0h, -6.0h),
    half2(6.0h, -6.0h),
    half2(7.0h, -6.0h),
    half2(-8.0h, -5.0h),
    half2(-7.0h, -5.0h),
    half2(-6.0h, -5.0h),
    half2(-5.0h, -5.0h),
    half2(-4.0h, -5.0h),
    half2(-3.0h, -5.0h),
    half2(-2.0h, -5.0h),
    half2(-1.0h, -5.0h),
    half2(0.0h, -5.0h),
    half2(1.0h, -5.0h),
    half2(2.0h, -5.0h),
    half2(3.0h, -5.0h),
    half2(4.0h, -5.0h),
    half2(5.0h, -5.0h),
    half2(6.0h, -5.0h),
    half2(7.0h, -5.0h),
    half2(-8.0h, -4.0h),
    half2(-7.0h, -4.0h),
    half2(-6.0h, -4.0h),
    half2(-5.0h, -4.0h),
    half2(-4.0h, -4.0h),
    half2(-3.0h, -4.0h),
    half2(-2.0h, -4.0h),
    half2(-1.0h, -4.0h),
    half2(0.0h, -4.0h),
    half2(1.0h, -4.0h),
    half2(2.0h, -4.0h),
    half2(3.0h, -4.0h),
    half2(4.0h, -4.0h),
    half2(5.0h, -4.0h),
    half2(6.0h, -4.0h),
    half2(7.0h, -4.0h),
    half2(-8.0h, -3.0h),
    half2(-7.0h, -3.0h),
    half2(-6.0h, -3.0h),
    half2(-5.0h, -3.0h),
    half2(-4.0h, -3.0h),
    half2(-3.0h, -3.0h),
    half2(-2.0h, -3.0h),
    half2(-1.0h, -3.0h),
    half2(0.0h, -3.0h),
    half2(1.0h, -3.0h),
    half2(2.0h, -3.0h),
    half2(3.0h, -3.0h),
    half2(4.0h, -3.0h),
    half2(5.0h, -3.0h),
    half2(6.0h, -3.0h),
    half2(7.0h, -3.0h),
    half2(-8.0h, -2.0h),
    half2(-7.0h, -2.0h),
    half2(-6.0h, -2.0h),
    half2(-5.0h, -2.0h),
    half2(-4.0h, -2.0h),
    half2(-3.0h, -2.0h),
    half2(-2.0h, -2.0h),
    half2(-1.0h, -2.0h),
    half2(0.0h, -2.0h),
    half2(1.0h, -2.0h),
    half2(2.0h, -2.0h),
    half2(3.0h, -2.0h),
    half2(4.0h, -2.0h),
    half2(5.0h, -2.0h),
    half2(6.0h, -2.0h),
    half2(7.0h, -2.0h),
    half2(-8.0h, -1.0h),
    half2(-7.0h, -1.0h),
    half2(-6.0h, -1.0h),
    half2(-5.0h, -1.0h),
    half2(-4.0h, -1.0h),
    half2(-3.0h, -1.0h),
    half2(-2.0h, -1.0h),
    half2(-1.0h, -1.0h),
    half2(0.0h, -1.0h),
    half2(1.0h, -1.0h),
    half2(2.0h, -1.0h),
    half2(3.0h, -1.0h),
    half2(4.0h, -1.0h),
    half2(5.0h, -1.0h),
    half2(6.0h, -1.0h),
    half2(7.0h, -1.0h),
    half2(-8.0h, 0.0h),
    half2(-7.0h, 0.0h),
    half2(-6.0h, 0.0h),
    half2(-5.0h, 0.0h),
    half2(-4.0h, 0.0h),
    half2(-3.0h, 0.0h),
    half2(-2.0h, 0.0h),
    half2(-1.0h, 0.0h),
    half2(0.0h, 0.0h),
    half2(1.0h, 0.0h),
    half2(2.0h, 0.0h),
    half2(3.0h, 0.0h),
    half2(4.0h, 0.0h),
    half2(5.0h, 0.0h),
    half2(6.0h, 0.0h),
    half2(7.0h, 0.0h),
    half2(-8.0h, 1.0h),
    half2(-7.0h, 1.0h),
    half2(-6.0h, 1.0h),
    half2(-5.0h, 1.0h),
    half2(-4.0h, 1.0h),
    half2(-3.0h, 1.0h),
    half2(-2.0h, 1.0h),
    half2(-1.0h, 1.0h),
    half2(0.0h, 1.0h),
    half2(1.0h, 1.0h),
    half2(2.0h, 1.0h),
    half2(3.0h, 1.0h),
    half2(4.0h, 1.0h),
    half2(5.0h, 1.0h),
    half2(6.0h, 1.0h),
    half2(7.0h, 1.0h),
    half2(-8.0h, 2.0h),
    half2(-7.0h, 2.0h),
    half2(-6.0h, 2.0h),
    half2(-5.0h, 2.0h),
    half2(-4.0h, 2.0h),
    half2(-3.0h, 2.0h),
    half2(-2.0h, 2.0h),
    half2(-1.0h, 2.0h),
    half2(0.0h, 2.0h),
    half2(1.0h, 2.0h),
    half2(2.0h, 2.0h),
    half2(3.0h, 2.0h),
    half2(4.0h, 2.0h),
    half2(5.0h, 2.0h),
    half2(6.0h, 2.0h),
    half2(7.0h, 2.0h),
    half2(-8.0h, 3.0h),
    half2(-7.0h, 3.0h),
    half2(-6.0h, 3.0h),
    half2(-5.0h, 3.0h),
    half2(-4.0h, 3.0h),
    half2(-3.0h, 3.0h),
    half2(-2.0h, 3.0h),
    half2(-1.0h, 3.0h),
    half2(0.0h, 3.0h),
    half2(1.0h, 3.0h),
    half2(2.0h, 3.0h),
    half2(3.0h, 3.0h),
    half2(4.0h, 3.0h),
    half2(5.0h, 3.0h),
    half2(6.0h, 3.0h),
    half2(7.0h, 3.0h),
    half2(-8.0h, 4.0h),
    half2(-7.0h, 4.0h),
    half2(-6.0h, 4.0h),
    half2(-5.0h, 4.0h),
    half2(-4.0h, 4.0h),
    half2(-3.0h, 4.0h),
    half2(-2.0h, 4.0h),
    half2(-1.0h, 4.0h),
    half2(0.0h, 4.0h),
    half2(1.0h, 4.0h),
    half2(2.0h, 4.0h),
    half2(3.0h, 4.0h),
    half2(4.0h, 4.0h),
    half2(5.0h, 4.0h),
    half2(6.0h, 4.0h),
    half2(7.0h, 4.0h),
    half2(-8.0h, 5.0h),
    half2(-7.0h, 5.0h),
    half2(-6.0h, 5.0h),
    half2(-5.0h, 5.0h),
    half2(-4.0h, 5.0h),
    half2(-3.0h, 5.0h),
    half2(-2.0h, 5.0h),
    half2(-1.0h, 5.0h),
    half2(0.0h, 5.0h),
    half2(1.0h, 5.0h),
    half2(2.0h, 5.0h),
    half2(3.0h, 5.0h),
    half2(4.0h, 5.0h),
    half2(5.0h, 5.0h),
    half2(6.0h, 5.0h),
    half2(7.0h, 5.0h),
    half2(-8.0h, 6.0h),
    half2(-7.0h, 6.0h),
    half2(-6.0h, 6.0h),
    half2(-5.0h, 6.0h),
    half2(-4.0h, 6.0h),
    half2(-3.0h, 6.0h),
    half2(-2.0h, 6.0h),
    half2(-1.0h, 6.0h),
    half2(0.0h, 6.0h),
    half2(1.0h, 6.0h),
    half2(2.0h, 6.0h),
    half2(3.0h, 6.0h),
    half2(4.0h, 6.0h),
    half2(5.0h, 6.0h),
    half2(6.0h, 6.0h),
    half2(7.0h, 6.0h),
    half2(-8.0h, 7.0h),
    half2(-7.0h, 7.0h),
    half2(-6.0h, 7.0h),
    half2(-5.0h, 7.0h),
    half2(-4.0h, 7.0h),
    half2(-3.0h, 7.0h),
    half2(-2.0h, 7.0h),
    half2(-1.0h, 7.0h),
    half2(0.0h, 7.0h),
    half2(1.0h, 7.0h),
    half2(2.0h, 7.0h),
    half2(3.0h, 7.0h),
    half2(4.0h, 7.0h),
    half2(5.0h, 7.0h),
    half2(6.0h, 7.0h),
    half2(7.0h, 7.0h),
};
kernel void pf_b1_wide(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    if (row0 >= args.rows) return;
    // B1 (m1max 2B): six 16-token windows share ONE staged weight tile
    // (grid.y==1): per-chunk device weight traffic drops 6x. Barrier count
    // per token ~unchanged; racc widens to 6x float4 (24 regs/lane).

    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 2);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 64);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 64);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 raccw[6];
    #pragma unroll
    for (uint w = 0; w < 6; w++) raccw[w] = float4(0.0f);
    threadgroup float *sc = Sc + sg * 256;
    const uint tokL = lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            // 16 columns = 8 bytes per thread: one byte-LUT gather per
            // nibble pair, low nibble first (matches the float kernel's
            // LSB-first order). wcb is a multiple of 16 -> half2-aligned.
            const uint2 wp = *(device const uint2 *)(wsrc + (c0 + wcb) / 2);
            threadgroup half2 *dst = (threadgroup half2 *)(Wt + wrow * 64 + wcb);
            dst[0] = q27_q4_half2_lut[wp.x         & 0xffu];
            dst[1] = q27_q4_half2_lut[(wp.x >>  8) & 0xffu];
            dst[2] = q27_q4_half2_lut[(wp.x >> 16) & 0xffu];
            dst[3] = q27_q4_half2_lut[wp.x >> 24         ];
            dst[4] = q27_q4_half2_lut[wp.y         & 0xffu];
            dst[5] = q27_q4_half2_lut[(wp.y >>  8) & 0xffu];
            dst[6] = q27_q4_half2_lut[(wp.y >> 16) & 0xffu];
            dst[7] = q27_q4_half2_lut[wp.y >> 24         ];
        }
        #pragma unroll
        for (uint win = 0; win < 6; win++) {
        const uint tw = win * 16;
        {
            const uint xtok = tw + xloc;
            device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            // Raw int8 values: exact in half. The per-token 32-group scale
            // folds at the flush below; invalid token slots stage clamped
            // real values whose outputs are never stored.
            dst[0 * 16] = half(xa.x); dst[1 * 16] = half(xa.y);
            dst[2 * 16] = half(xa.z); dst[3 * 16] = half(xa.w);
            dst[4 * 16] = half(xb.x); dst[5 * 16] = half(xb.y);
            dst[6 * 16] = half(xb.z); dst[7 * 16] = half(xb.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint tokA = tw + tokL, tokB = tw + 8 + tokL;
        const float wsA = float(weight_scales[wsrowA + c0 / 64]);
        const float wsB = float(weight_scales[wsrowB + c0 / 64]);
        // The two 32-K sub-slabs (activation-scale groups) accumulate into
        // separate tile pairs so both fold in ONE barrier region per staged 64.
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            raccw[win] += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            raccw[win] += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }   // window loop
    }
    #pragma unroll
    for (uint w = 0; w < 6; w++) {
        const uint tokAw = w * 16 + tokL, tokBw = w * 16 + 8 + tokL;
        if (rowA < args.rows && tokAw < args.x_rows) out[(ulong)tokAw * args.rows + rowA] = raccw[w].x;
        if (rowB < args.rows && tokAw < args.x_rows) out[(ulong)tokAw * args.rows + rowB] = raccw[w].y;
        if (rowA < args.rows && tokBw < args.x_rows) out[(ulong)tokBw * args.rows + rowA] = raccw[w].z;
        if (rowB < args.rows && tokBw < args.x_rows) out[(ulong)tokBw * args.rows + rowB] = raccw[w].w;
    }
}

kernel void pf_b2_dbuf(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    // B2 (m1max 2B): two staged buffers; step i+1's device loads are issued
    // right after step i becomes visible, so DRAM/L2 latency overlaps the
    // MMA+flush work of step i. Same tiles and flush math as production.
    threadgroup half Wt[2][32 * 64];
    threadgroup half Xt[2][64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 2);
    (void)0;
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 64);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 64);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    const uint nsteps = args.cols / 64;
    // preload step 0
    uint buf = 0;
    {
        const uint c0 = 0;
        {
            // 16 columns = 8 bytes per thread: one byte-LUT gather per
            // nibble pair, low nibble first (matches the float kernel's
            // LSB-first order). wcb is a multiple of 16 -> half2-aligned.
            const uint2 wp = *(device const uint2 *)(wsrc + (c0 + wcb) / 2);
            threadgroup half2 *dst = (threadgroup half2 *)(Wt[buf] + wrow * 64 + wcb);
            dst[0] = q27_q4_half2_lut[wp.x         & 0xffu];
            dst[1] = q27_q4_half2_lut[(wp.x >>  8) & 0xffu];
            dst[2] = q27_q4_half2_lut[(wp.x >> 16) & 0xffu];
            dst[3] = q27_q4_half2_lut[wp.x >> 24         ];
            dst[4] = q27_q4_half2_lut[wp.y         & 0xffu];
            dst[5] = q27_q4_half2_lut[(wp.y >>  8) & 0xffu];
            dst[6] = q27_q4_half2_lut[(wp.y >> 16) & 0xffu];
            dst[7] = q27_q4_half2_lut[wp.y >> 24         ];
        }
        {
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt[buf] + xcb * 16 + xloc;
            // Raw int8 values: exact in half. The per-token 32-group scale
            // folds at the flush below; invalid token slots stage clamped
            // real values whose outputs are never stored.
            dst[0 * 16] = half(xa.x); dst[1 * 16] = half(xa.y);
            dst[2 * 16] = half(xa.z); dst[3 * 16] = half(xa.w);
            dst[4 * 16] = half(xb.x); dst[5 * 16] = half(xb.y);
            dst[6 * 16] = half(xb.z); dst[7 * 16] = half(xb.w);
        }
        // publish step0 staging, then enter the pipelined loop
        for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nbuf = buf ^ 1;
        if (c0 + 64 < args.cols) {
            const uint c1 = c0 + 64;
            const uint2 wpn = *(device const uint2 *)(wsrc + (c1 + wcb) / 2);
            threadgroup half2 *dn = (threadgroup half2 *)(Wt[nbuf] + wrow * 64 + wcb);
            dn[0] = q27_q4_half2_lut[wpn.x & 0xffu]; dn[1] = q27_q4_half2_lut[(wpn.x >> 8) & 0xffu];
            dn[2] = q27_q4_half2_lut[(wpn.x >> 16) & 0xffu]; dn[3] = q27_q4_half2_lut[wpn.x >> 24];
            dn[4] = q27_q4_half2_lut[wpn.y & 0xffu]; dn[5] = q27_q4_half2_lut[(wpn.y >> 8) & 0xffu];
            dn[6] = q27_q4_half2_lut[(wpn.y >> 16) & 0xffu]; dn[7] = q27_q4_half2_lut[wpn.y >> 24];
            const char4 xan = *(device const char4 *)(xsrc + c1 + xcb);
            const char4 xbn = *(device const char4 *)(xsrc + c1 + xcb + 4);
            threadgroup half *dx = Xt[nbuf] + xcb * 16 + xloc;
            dx[0*16]=half(xan.x); dx[1*16]=half(xan.y); dx[2*16]=half(xan.z); dx[3*16]=half(xan.w);
            dx[4*16]=half(xbn.x); dx[5*16]=half(xbn.y); dx[6*16]=half(xbn.z); dx[7*16]=half(xbn.w);
        }
        const float wsA = float(weight_scales[wsrowA + c0 / 64]);
        const float wsB = float(weight_scales[wsrowB + c0 / 64]);
        // The two 32-K sub-slabs (activation-scale groups) accumulate into
        // separate tile pairs so both fold in ONE barrier region per staged 64.
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt[buf] + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt[buf] + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt[buf] + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt[buf] + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt[buf] + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt[buf] + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        buf = nbuf;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}


// --- C4: production minus 3/4 of flush work (attribution arm; numbers intentionally wrong) ---
kernel void pf_c4_flushless(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 2);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 64);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 64);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            // 16 columns = 8 bytes per thread: one byte-LUT gather per
            // nibble pair, low nibble first (matches the float kernel's
            // LSB-first order). wcb is a multiple of 16 -> half2-aligned.
            const uint2 wp = *(device const uint2 *)(wsrc + (c0 + wcb) / 2);
            threadgroup half2 *dst = (threadgroup half2 *)(Wt + wrow * 64 + wcb);
            dst[0] = q27_q4_half2_lut[wp.x         & 0xffu];
            dst[1] = q27_q4_half2_lut[(wp.x >>  8) & 0xffu];
            dst[2] = q27_q4_half2_lut[(wp.x >> 16) & 0xffu];
            dst[3] = q27_q4_half2_lut[wp.x >> 24         ];
            dst[4] = q27_q4_half2_lut[wp.y         & 0xffu];
            dst[5] = q27_q4_half2_lut[(wp.y >>  8) & 0xffu];
            dst[6] = q27_q4_half2_lut[(wp.y >> 16) & 0xffu];
            dst[7] = q27_q4_half2_lut[wp.y >> 24         ];
        }
        {
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            // Raw int8 values: exact in half. The per-token 32-group scale
            // folds at the flush below; invalid token slots stage clamped
            // real values whose outputs are never stored.
            dst[0 * 16] = half(xa.x); dst[1 * 16] = half(xa.y);
            dst[2 * 16] = half(xa.z); dst[3 * 16] = half(xa.w);
            dst[4 * 16] = half(xb.x); dst[5 * 16] = half(xb.y);
            dst[6 * 16] = half(xb.z); dst[7 * 16] = half(xb.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float wsA = float(weight_scales[wsrowA + c0 / 64]);
        const float wsB = float(weight_scales[wsrowB + c0 / 64]);
        // The two 32-K sub-slabs (activation-scale groups) accumulate into
        // separate tile pairs so both fold in ONE barrier region per staged 64.
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
        simdgroup_store(acc0, sc, 8);
        simdgroup_store(acc1, sc + 64, 8);
        simdgroup_store(acc2, sc + 128, 8);
        simdgroup_store(acc3, sc + 192, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (((c0 / 64) & 3) == 3) {   // flush once per 4 steps (MATH WRONG: cost attribution only)
        {
            const ulong xrow_a = (ulong)min(tokA, args.x_rows - 1) * (args.cols / 32);
            const ulong xrow_b = (ulong)min(tokB, args.x_rows - 1) * (args.cols / 32);
            const float xsA0 = x_scales[xrow_a + c0 / 32],     xsB0 = x_scales[xrow_b + c0 / 32];
            const float xsA1 = x_scales[xrow_a + c0 / 32 + 1], xsB1 = x_scales[xrow_b + c0 / 32 + 1];
            racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]) *
                    float4(wsA * xsA0, wsB * xsA0, wsA * xsB0, wsB * xsB0);
            racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]) *
                    float4(wsA * xsA1, wsB * xsA1, wsA * xsB1, wsB * xsB1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}


// --- C5: scales folded into staged tiles at fp16; tensor acc runs all K; single flush (math: margin-class) ---
kernel void pf_c5_prescale(
        device const uchar *weights [[buffer(0)]], device const half *weight_scales [[buffer(1)]],
        device const char *x [[buffer(2)]], device const float *x_scales [[buffer(3)]],
        device float *out [[buffer(4)]], constant MatmulArgs &args [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[32 * 64];
    threadgroup half Xt[64 * 16];
    threadgroup float Sc[4 * 256];
    const uint row0 = group.x * 32;
    const uint tok0 = group.y * 16;   // 16-token tile (wide-chunk grid)
    if (row0 >= args.rows) return;
    const uint rlast = args.rows - 1;
    const uint wrow = tid / 4, wcb = (tid % 4) * 16;
    device const uchar *wsrc = weights + (ulong)min(row0 + wrow, rlast) * (args.cols / 2);
    const uint xloc = tid % 16, xcb = (tid / 16) * 8;   // Xt column is tile-local
    const uint xtok = tok0 + xloc;                       // device rows are global
    device const char *xsrc = x + (ulong)min(xtok, args.x_rows - 1) * args.cols;
    const uint rowA = row0 + sg * 8 + lane / 8, rowB = rowA + 4;
    const ulong wsrowA = (ulong)min(rowA, rlast) * (args.cols / 64);
    const ulong wsrowB = (ulong)min(rowB, rlast) * (args.cols / 64);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float4 racc = 0.0f;
    threadgroup float *sc = Sc + sg * 256;
    const uint tokA = tok0 + lane % 8, tokB = tok0 + 8 + lane % 8;
    for (uint c0 = 0; c0 < args.cols; c0 += 64) {
        {
            // 16 columns = 8 bytes per thread: one byte-LUT gather per
            // nibble pair, low nibble first (matches the float kernel's
            // LSB-first order). wcb is a multiple of 16 -> half2-aligned.
            const uint2 wp = *(device const uint2 *)(wsrc + (c0 + wcb) / 2);
            threadgroup half2 *dst = (threadgroup half2 *)(Wt + wrow * 64 + wcb);
            const half wsh = half(weight_scales[(ulong)min(row0 + wrow, rlast) * (args.cols / 64) + c0 / 64]);
            dst[0] = q27_q4_half2_lut[wp.x         & 0xffu] * wsh;
            dst[1] = q27_q4_half2_lut[(wp.x >>  8) & 0xffu] * wsh;
            dst[2] = q27_q4_half2_lut[(wp.x >> 16) & 0xffu] * wsh;
            dst[3] = q27_q4_half2_lut[wp.x >> 24         ];
            dst[4] = q27_q4_half2_lut[wp.y         & 0xffu] * wsh;
            dst[5] = q27_q4_half2_lut[(wp.y >>  8) & 0xffu] * wsh;
            dst[6] = q27_q4_half2_lut[(wp.y >> 16) & 0xffu] * wsh;
            dst[7] = q27_q4_half2_lut[wp.y >> 24         ];
        }
        {
            const char4 xa = *(device const char4 *)(xsrc + c0 + xcb);
            const char4 xb = *(device const char4 *)(xsrc + c0 + xcb + 4);
            threadgroup half *dst = Xt + xcb * 16 + xloc;
            const half xsh = half(x_scales[(ulong)min(xtok, args.x_rows - 1) * (args.cols / 32) + (c0 + xcb) / 32]);
            // Raw int8 values: exact in half. The per-token 32-group scale
            // folds at the flush below; invalid token slots stage clamped
            // real values whose outputs are never stored.
            dst[0 * 16] = half(xa.x) * xsh; dst[1 * 16] = half(xa.y) * xsh;
            dst[2 * 16] = half(xa.z) * xsh; dst[3 * 16] = half(xa.w) * xsh;
            dst[4 * 16] = half(xb.x) * xsh; dst[5 * 16] = half(xb.y) * xsh;
            dst[6 * 16] = half(xb.z) * xsh; dst[7 * 16] = half(xb.w) * xsh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The two 32-K sub-slabs (activation-scale groups) accumulate into
        // separate tile pairs so both fold in ONE barrier region per staged 64.
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc0, a, b, acc0);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc1, a, b, acc1);
        }
        for (uint k8 = 32; k8 < 64; k8 += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, Wt + (uint)sg * 8 * 64 + k8, 64);
            simdgroup_load(b, Xt + k8 * 16, 16);
            simdgroup_multiply_accumulate(acc2, a, b, acc2);
            simdgroup_load(b, Xt + k8 * 16 + 8, 16);
            simdgroup_multiply_accumulate(acc3, a, b, acc3);
        }
               threadgroup_barrier(mem_flags::mem_threadgroup);   // staging reusable
        }

    // single end-of-kernel flush (scales already folded at staging)
    simdgroup_store(acc0, sc, 8);
    simdgroup_store(acc1, sc + 64, 8);
    simdgroup_store(acc2, sc + 128, 8);
    simdgroup_store(acc3, sc + 192, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    racc += float4(sc[lane], sc[lane + 32], sc[lane + 64], sc[lane + 96]);
    racc += float4(sc[lane + 128], sc[lane + 160], sc[lane + 192], sc[lane + 224]);
    if (rowA < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowA] = racc.x;
    if (rowB < args.rows && tokA < args.x_rows) out[(ulong)tokA * args.rows + rowB] = racc.y;
    if (rowA < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowA] = racc.z;
    if (rowB < args.rows && tokB < args.x_rows) out[(ulong)tokB * args.rows + rowB] = racc.w;
}

