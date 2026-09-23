// bench-only companion for tools/roofline_m1.mm: pure uint4 stream at the
using namespace metal;
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
