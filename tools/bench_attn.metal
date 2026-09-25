// Attention decode roofline arms (M1 Max study, m1max STATUS). Production
// shapes from qwen36-27b-mtp-q4s: q_heads=24 kv_heads=4 head_dim=256 gqa=6,
// 17 attention layers of 65. Grids mirror the engine exactly. Attribution
// arms (noexp/nostage) deliberately change math for cost partitioning only;
// stream/empty bound any fix. Production kernels are timed from the engine
// source by the harness (compile the real shaders too), never copied.
#include <metal_stdlib>
using namespace metal;

struct AttentionGqaArgs {
    uint q_stride, seq_len, q_heads, kv_heads, head_dim, block, n_blocks;
    float scale;
};

kernel void attn_empty(device float *sink [[buffer(0)]],
                       constant AttentionGqaArgs &args [[buffer(4)]],
                       uint2 group [[threadgroup_position_in_grid]]) {
    if (group.x == 0xFFFFFFFFu && group.y == 0xFFFFFFFFu) sink[0] = 1.0f;
}

// Same grid/threads as q27_attention_f16_gqa; reads exactly the unique KV
// bytes for its (kvh, block) tile once — parallelism-limited BW reference.
kernel void attn_stream(device const half *kc [[buffer(1)]],
                        device const half *vc [[buffer(2)]],
                        device float *partials [[buffer(3)]],
                        constant AttentionGqaArgs &args [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    if (kvh >= args.kv_heads || blk >= args.n_blocks) return;
    const uint p0 = blk * args.block, p1 = min(p0 + args.block, args.seq_len);
    const uint tid = (uint)sg * 32 + lane, threads = 192;
    float s = 0.0f;
    for (uint p = p0; p < p1; p++) {
        device const half2 *k2 = (device const half2 *)(kc + ((ulong)p * args.kv_heads + kvh) * args.head_dim);
        device const half2 *v2 = (device const half2 *)(vc + ((ulong)p * args.kv_heads + kvh) * args.head_dim);
        for (uint i = tid; i < args.head_dim / 2; i += threads)
            s += float(k2[i].x) + float(k2[i].y) + float(v2[i].x) + float(v2[i].y);
    }
    s = simd_sum(s);
    if (lane == 0) partials[(ulong)kvh * args.n_blocks + blk] += s;
}

// Attribution: production gqa decode body without the two SFU exp calls
// (correction/weight replaced by cheap substitutes; loads/stores identical).
kernel void attn_noexp(device const float *q [[buffer(0)]],
                       device const half *kc [[buffer(1)]],
                       device const half *vc [[buffer(2)]],
                       device float *partials [[buffer(3)]],
                       constant AttentionGqaArgs &args [[buffer(4)]],
                       uint2 group [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * args.head_dim; idx += threads) {
            const uint r = idx / args.head_dim, d = idx % args.head_dim;
            const ulong row = ((ulong)(t0 + r) * args.kv_heads + kvh) * args.head_dim;
            Kt[r][d] = float(kc[row + d]);
            Vt[r][d] = float(vc[row + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            float partial = 0.0f;
            for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float correction = score > m ? 0.5f : 0.999f;   // no SFU
            const float weight = score * 0.001f;                  // no SFU
            l = l * correction + weight;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * correction + weight * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) ph[2 + d] = acc[i];
}

// Attribution: no threadgroup staging — every query head reads its own K/V
// rows straight from device (6x redundant device reads, zero smem/barriers).
kernel void attn_nostage(device const float *q [[buffer(0)]],
                         device const half *kc [[buffer(1)]],
                         device const half *vc [[buffer(2)]],
                         device float *partials [[buffer(3)]],
                         constant AttentionGqaArgs &args [[buffer(4)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         ushort lane [[thread_index_in_simdgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint p = p0; p < p1; p++) {
        device const half *kh = kc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
        device const half *vh = vc + ((ulong)p * args.kv_heads + kvh) * args.head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * float(kh[d]);
        const float score = simd_sum(partial) * args.scale;
        const float m_new = max(m, score);
        const float correction = exp(m - m_new);
        const float weight = exp(score - m_new);
        l = l * correction + weight;
        for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
            acc[i] = acc[i] * correction + weight * float(vh[d]);
        m = m_new;
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) ph[2 + d] = acc[i];
}

// Attribution: production gqa body with the online-softmax serial dependency
// removed (m fixed to 0: no running max/l chain across row groups). Loads,
// stores and dots identical — isolates the row-group dependency chain.
kernel void attn_nosm(device const float *q [[buffer(0)]],
                      device const half *kc [[buffer(1)]],
                      device const half *vc [[buffer(2)]],
                      device float *partials [[buffer(3)]],
                      constant AttentionGqaArgs &args [[buffer(4)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      ushort lane [[thread_index_in_simdgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * args.head_dim; idx += threads) {
            const uint r = idx / args.head_dim, d = idx % args.head_dim;
            const ulong row = ((ulong)(t0 + r) * args.kv_heads + kvh) * args.head_dim;
            Kt[r][d] = float(kc[row + d]);
            Vt[r][d] = float(vc[row + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; r++) {
            float partial = 0.0f;
            for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float weight = exp(score);          // independent of chain
            l += weight;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] += weight * Vt[r][d];
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = 0.0f; ph[1] = l; }
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) ph[2 + d] = acc[i];
}

// Attribution: two rows in flight per iteration (2 parallel score/weight
// chains; the acc rescale stays serial via one shared correction).
kernel void attn_w2row(device const float *q [[buffer(0)]],
                       device const half *kc [[buffer(1)]],
                       device const half *vc [[buffer(2)]],
                       device float *partials [[buffer(3)]],
                       constant AttentionGqaArgs &args [[buffer(4)]],
                       uint2 group [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * args.head_dim; idx += threads) {
            const uint r = idx / args.head_dim, d = idx % args.head_dim;
            const ulong row = ((ulong)(t0 + r) * args.kv_heads + kvh) * args.head_dim;
            Kt[r][d] = float(kc[row + d]);
            Vt[r][d] = float(vc[row + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = 0;
        for (; r + 1 < rows; r += 2) {
            float pa = 0.0f, pb = 0.0f;
            for (uint d = lane; d < args.head_dim; d += 32) {
                pa += qh_ptr[d] * Kt[r][d];
                pb += qh_ptr[d] * Kt[r + 1][d];
            }
            const float sa = simd_sum(pa) * args.scale;
            const float sb = simd_sum(pb) * args.scale;
            const float m_new = max(m, max(sa, sb));
            const float corr = exp(m - m_new);
            const float wa = exp(sa - m_new), wb = exp(sb - m_new);
            l = l * corr + wa + wb;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * corr + wa * Vt[r][d] + wb * Vt[r + 1][d];
            m = m_new;
        }
        if (r < rows) {
            float partial = 0.0f;
            for (uint d = lane; d < args.head_dim; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float corr = exp(m - m_new);
            const float w = exp(score - m_new);
            l = l * corr + w;
            for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++)
                acc[i] = acc[i] * corr + w * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < args.head_dim; d += 32, i++) ph[2 + d] = acc[i];
}

// --- turbo3 (8-bit KV) arms: decode's real path -------------------------
constant float turbo_centroids[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f };
inline float turbo_dequant(device const uchar *block, uint j) {
    const uint low = (block[2 + (j >> 2)] >> ((j & 3) * 2)) & 3;
    const uint high = (block[34 + (j >> 3)] >> (j & 7)) & 1;
    return turbo_centroids[low | (high << 2)] * float(*(device const half *)block);
}

// Candidate (w2row for turbo3_gqa): two KV rows staged/processed per
// iteration with a shared correction. Mathematically equivalent online
// softmax, different accumulation order -> golden margin gate, not digest.
kernel void attn_t3_w2row(device const float *q [[buffer(0)]],
                          device const uchar *kc [[buffer(1)]],
                          device const uchar *vc [[buffer(2)]],
                          device float *partials [[buffer(3)]],
                          constant AttentionGqaArgs &args [[buffer(4)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * 256; idx += threads) {
            const uint r = idx >> 8, d = idx & 255;
            device const uchar *kb = kc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            device const uchar *vb = vc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            Kt[r][d] = turbo_dequant(kb + (d >> 7) * 50, d & 127);
            Vt[r][d] = turbo_dequant(vb + (d >> 7) * 50, d & 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = 0;
        for (; r + 1 < rows; r += 2) {
            float pa = 0.0f, pb = 0.0f;
            for (uint d = lane; d < 256; d += 32) {
                pa += qh_ptr[d] * Kt[r][d];
                pb += qh_ptr[d] * Kt[r + 1][d];
            }
            const float sa = simd_sum(pa) * args.scale;
            const float sb = simd_sum(pb) * args.scale;
            const float m_new = max(m, max(sa, sb));
            const float corr = exp(m - m_new);
            const float wa = exp(sa - m_new), wb = exp(sb - m_new);
            l = l * corr + wa + wb;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * corr + wa * Vt[r][d] + wb * Vt[r + 1][d];
            m = m_new;
        }
        if (r < rows) {
            float partial = 0.0f;
            for (uint d = lane; d < 256; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float corr = exp(m - m_new);
            const float w = exp(score - m_new);
            l = l * corr + w;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * corr + w * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[i];
}

// D2 #2 alignment probe: pure turbo3 read floor at the production 50B
// chunk stride vs a 64B-padded stride. Same payload bytes read; only the
// alignment/coalescing pattern differs.
kernel void attn_t3_stream50(device const uchar *kc [[buffer(1)]],
                             device const uchar *vc [[buffer(2)]],
                             device float *partials [[buffer(3)]],
                             constant AttentionGqaArgs &args [[buffer(4)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    if (kvh >= args.kv_heads || blk >= args.n_blocks) return;
    const uint p0 = blk * args.block, p1 = min(p0 + args.block, args.seq_len);
    const uint tid = (uint)sg * 32 + lane, threads = args.q_heads / args.kv_heads * 32;
    float s = 0.0f;
    for (uint t = p0; t < p1; t++) {
        device const uchar *kb = kc + ((ulong)t * args.kv_heads + kvh) * 2 * 50;
        device const uchar *vb = vc + ((ulong)t * args.kv_heads + kvh) * 2 * 50;
        for (uint d = tid; d < 256; d += threads) {
            s += turbo_dequant(kb + (d >> 7) * 50, d & 127);
            s += turbo_dequant(vb + (d >> 7) * 50, d & 127);
        }
    }
    s = simd_sum(s);
    if (lane == 0) partials[(ulong)kvh * args.n_blocks + blk] += s;
}
kernel void attn_t3_stream64(device const uchar *kc [[buffer(1)]],
                             device const uchar *vc [[buffer(2)]],
                             device float *partials [[buffer(3)]],
                             constant AttentionGqaArgs &args [[buffer(4)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    if (kvh >= args.kv_heads || blk >= args.n_blocks) return;
    const uint p0 = blk * args.block, p1 = min(p0 + args.block, args.seq_len);
    const uint tid = (uint)sg * 32 + lane, threads = args.q_heads / args.kv_heads * 32;
    float s = 0.0f;
    for (uint t = p0; t < p1; t++) {
        device const uchar *kb = kc + ((ulong)t * args.kv_heads + kvh) * 2 * 64;
        device const uchar *vb = vc + ((ulong)t * args.kv_heads + kvh) * 2 * 64;
        for (uint d = tid; d < 256; d += threads) {
            s += turbo_dequant(kb + (d >> 7) * 64, d & 127);
            s += turbo_dequant(vb + (d >> 7) * 64, d & 127);
        }
    }
    s = simd_sum(s);
    if (lane == 0) partials[(ulong)kvh * args.n_blocks + blk] += s;
}

kernel void attn_t3_w2row_pad(device const float *q [[buffer(0)]],
                          device const uchar *kc [[buffer(1)]],
                          device const uchar *vc [[buffer(2)]],
                          device float *partials [[buffer(3)]],
                          constant AttentionGqaArgs &args [[buffer(4)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * 256; idx += threads) {
            const uint r = idx >> 8, d = idx & 255;
            device const uchar *kb = kc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 64;
            device const uchar *vb = vc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 64;
            Kt[r][d] = turbo_dequant(kb + (d >> 7) * 64, d & 127);
            Vt[r][d] = turbo_dequant(vb + (d >> 7) * 64, d & 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = 0;
        for (; r + 1 < rows; r += 2) {
            float pa = 0.0f, pb = 0.0f;
            for (uint d = lane; d < 256; d += 32) {
                pa += qh_ptr[d] * Kt[r][d];
                pb += qh_ptr[d] * Kt[r + 1][d];
            }
            const float sa = simd_sum(pa) * args.scale;
            const float sb = simd_sum(pb) * args.scale;
            const float m_new = max(m, max(sa, sb));
            const float corr = exp(m - m_new);
            const float wa = exp(sa - m_new), wb = exp(sb - m_new);
            l = l * corr + wa + wb;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * corr + wa * Vt[r][d] + wb * Vt[r + 1][d];
            m = m_new;
        }
        if (r < rows) {
            float partial = 0.0f;
            for (uint d = lane; d < 256; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float corr = exp(m - m_new);
            const float w = exp(score - m_new);
            l = l * corr + w;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * corr + w * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[i];
}
kernel void attn_t3_w4row(device const float *q [[buffer(0)]],
                          device const uchar *kc [[buffer(1)]],
                          device const uchar *vc [[buffer(2)]],
                          device float *partials [[buffer(3)]],
                          constant AttentionGqaArgs &args [[buffer(4)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint kvh = group.x, blk = group.y;
    const uint gqa = args.q_heads / args.kv_heads;
    if (kvh >= args.kv_heads || blk >= args.n_blocks || sg >= gqa) return;
    const uint p0 = blk * args.block;
    const uint p1 = min(p0 + args.block, args.seq_len);
    const uint qh = kvh * gqa + sg;
    device const float *qh_ptr = q + (ulong)qh * args.q_stride;
    const uint tid = (uint)sg * 32 + lane, threads = gqa * 32;
    threadgroup float Kt[8][256], Vt[8][256];
    float acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint t0 = p0; t0 < p1; t0 += 8) {
        const uint rows = min(8u, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rows * 256; idx += threads) {
            const uint r = idx >> 8, d = idx & 255;
            device const uchar *kb = kc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            device const uchar *vb = vc + ((ulong)(t0 + r) * args.kv_heads + kvh) * 2 * 50;
            Kt[r][d] = turbo_dequant(kb + (d >> 7) * 50, d & 127);
            Vt[r][d] = turbo_dequant(vb + (d >> 7) * 50, d & 127);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = 0;
        for (; r + 3 < rows; r += 4) {
            float pa = 0.0f, pb = 0.0f, pc = 0.0f, pd = 0.0f;
            for (uint d = lane; d < 256; d += 32) {
                pa += qh_ptr[d] * Kt[r][d];
                pb += qh_ptr[d] * Kt[r + 1][d];
                pc += qh_ptr[d] * Kt[r + 2][d];
                pd += qh_ptr[d] * Kt[r + 3][d];
            }
            const float sa = simd_sum(pa) * args.scale;
            const float sb = simd_sum(pb) * args.scale;
            const float sc = simd_sum(pc) * args.scale;
            const float sd = simd_sum(pd) * args.scale;
            const float m_new = max(max(m, sa), max(max(sb, sc), sd));
            const float corr = exp(m - m_new);
            const float wa = exp(sa - m_new), wb = exp(sb - m_new);
            const float wc = exp(sc - m_new), wd = exp(sd - m_new);
            l = l * corr + wa + wb + wc + wd;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * corr + wa * Vt[r][d] + wb * Vt[r + 1][d]
                                   + wc * Vt[r + 2][d] + wd * Vt[r + 3][d];
            m = m_new;
        }
        for (; r + 1 < rows; r += 2) {
            float pa = 0.0f, pb = 0.0f;
            for (uint d = lane; d < 256; d += 32) {
                pa += qh_ptr[d] * Kt[r][d];
                pb += qh_ptr[d] * Kt[r + 1][d];
            }
            const float sa = simd_sum(pa) * args.scale;
            const float sb = simd_sum(pb) * args.scale;
            const float m_new = max(m, max(sa, sb));
            const float corr = exp(m - m_new);
            const float wa = exp(sa - m_new), wb = exp(sb - m_new);
            l = l * corr + wa + wb;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * corr + wa * Vt[r][d] + wb * Vt[r + 1][d];
            m = m_new;
        }
        if (r < rows) {
            float partial = 0.0f;
            for (uint d = lane; d < 256; d += 32) partial += qh_ptr[d] * Kt[r][d];
            const float score = simd_sum(partial) * args.scale;
            const float m_new = max(m, score);
            const float corr = exp(m - m_new);
            const float w = exp(score - m_new);
            l = l * corr + w;
            for (uint d = lane, i = 0; d < 256; d += 32, i++)
                acc[i] = acc[i] * corr + w * Vt[r][d];
            m = m_new;
        }
    }
    device float *ph = partials + ((ulong)qh * args.n_blocks + blk) * 258;
    if (lane == 0) { ph[0] = m; ph[1] = l; }
    for (uint d = lane, i = 0; d < 256; d += 32, i++) ph[2 + d] = acc[i];
}