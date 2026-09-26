// M1 Max attention decode roofline/diagnosis harness (m1max study,
// 2026-09-24). Compiles the ENGINE source q27_kernels.metal (production
// q27_attention_f16, q27_attention_f16_gqa, q27_attention_gqa_merge timed
// as-is) plus attribution arms from tools/bench_attn.metal, at the real
// decode shapes of qwen36-27b-mtp-q4s (q_heads=24 kv_heads=4 head_dim=256,
// 17 attention layers/65). One model decode step reads KV once per layer:
// seq*4KB unique bytes/layer. Reports us/dispatch (best-of-5) so tg share
// = 17 * us, and effective GB/s vs the 356 GB/s DRAM ceiling.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct AttentionGqaArgs {
    uint32_t q_stride, seq_len, q_heads, kv_heads, head_dim, block, n_blocks;
    float scale;
};
struct AttentionArgs {   // non-gqa decode kernel
    uint32_t q_stride, seq_len, q_heads, kv_heads, head_dim; float scale;
};

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib, NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    NSError* err = nil;
    auto p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) fprintf(stderr, "pso %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String);
    return p;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string out_path = argc > 1 ? argv[1] : "/tmp/attn_roof.jsonl";
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        fprintf(stderr, "device=%s\n", dev.name.UTF8String);
        NSError* err = nil;
        NSString* src_path = [NSString stringWithUTF8String:"src/metal/q27_kernels.metal"];
        NSString* src = [NSString stringWithContentsOfFile:src_path encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "src: %s\n", err.localizedDescription.UTF8String); return 1; }
        auto prod_lib = [dev newLibraryWithSource:src
            options:[MTLCompileOptions new] error:&err];
        if (!prod_lib) { fprintf(stderr, "prod lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        NSString* bench_path = [NSString stringWithUTF8String:"tools/bench_attn.metal"];
        NSString* bsrc = [NSString stringWithContentsOfFile:bench_path encoding:NSUTF8StringEncoding error:&err];
        auto bench_lib = [dev newLibraryWithSource:bsrc options:[MTLCompileOptions new] error:&err];
        if (!bench_lib) { fprintf(stderr, "bench lib: %s\n", err.localizedDescription.UTF8String); return 1; }

        const uint32_t QH = 24, KVH = 4, HD = 256, GQA = 6, QS = 256;
        const uint32_t MAXSEQ = 65536;
        auto buf = [&](size_t bytes) { return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; };
        auto Q = buf((size_t)QH * QS * 4), KC = buf((size_t)MAXSEQ * KVH * HD * 2),
             VC = buf((size_t)MAXSEQ * KVH * HD * 2), PART = buf((size_t)QH * 512 * 258 * 4),   // nb stride up to 512 blocks @64K
             O = buf((size_t)QH * HD * 4),
             KCP = buf((size_t)MAXSEQ * KVH * 2 * 64), VCP = buf((size_t)MAXSEQ * KVH * 2 * 64);
        uint16_t* k = (uint16_t*)KC.contents; uint16_t* v = (uint16_t*)VC.contents;
        uint64_t st = 0x2437F11D2646C071ull;
        auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
        for (size_t i = 0; i < (size_t)MAXSEQ * KVH * HD; i++) {
            uint64_t r = rnd();
            k[i] = (uint16_t)(0x2C00u | (r & 0x3FFu));      // ~2^-13..2^-13*2 sign bits
            v[i] = (uint16_t)((r >> 16) & 0xFFFFu) & 0x7FFFu;// finite halves
            if ((r >> 15 & 3) == 0) v[i] |= 0x8000u;
        }
        float* qf = (float*)Q.contents;
        for (uint32_t i = 0; i < QH * QS; i++) qf[i] = ((rnd() & 0xFFFF) / 65536.0f - 0.5f);
        {   // D2 #2: fill padded buffers (50B payload + 14B pad per chunk)
            uint8_t* kp = (uint8_t*)KCP.contents; uint8_t* vp = (uint8_t*)VCP.contents;
            for (uint64_t t = 0; t < MAXSEQ; t++) for (uint h = 0; h < KVH; h++) {
                uint8_t* dk = kp + (t * KVH + h) * 2 * 64;
                uint8_t* dv = vp + (t * KVH + h) * 2 * 64;
                for (int b = 0; b < 100; b++) dk[b] = (uint8_t)rnd();
                for (int b = 0; b < 100; b++) dv[b] = (uint8_t)rnd();
            }
        }

        auto q = [dev newCommandQueue];
        std::vector<id<MTLComputePipelineState>> psos;
        std::vector<std::string> pnames;
        auto collect = [&](id<MTLLibrary> l, const char* name) {
            auto p = make_pso(dev, l, [NSString stringWithUTF8String:name]);
            if (p) { psos.push_back(p); pnames.push_back(name); }
            else fprintf(stderr, "missing %s\n", name);
        };
        collect(prod_lib, "q27_attention_turbo3_gqa");
        collect(prod_lib, "q27_attention_turbo3");
        collect(prod_lib, "q27_attention_f16");
        collect(prod_lib, "q27_attention_f16_gqa");  // actual f16 decode baseline for #3
        collect(bench_lib, "attn_empty");
        collect(bench_lib, "attn_stream");
        collect(bench_lib, "attn_noexp");
        collect(bench_lib, "attn_nostage");
        collect(bench_lib, "attn_nosm");
        collect(bench_lib, "attn_w2row");
        collect(bench_lib, "attn_t3_w2row");
        collect(bench_lib, "attn_t3_stream50");
        collect(bench_lib, "attn_t3_stream64");
        collect(bench_lib, "attn_t3_w2row_pad");
        collect(bench_lib, "attn_t3_w4row");
        collect(bench_lib, "attn_f16_w4row");

        FILE* out = fopen(out_path.c_str(), "w");
        fprintf(stderr, "%-22s %6s %6s %9s %9s %9s\n", "kernel", "seq", "block", "us", "us/token", "KV GB/s");
        uint32_t seqs[] = {512, 2048, 7168, 16384, 32768, 65536};
        uint32_t blocks[] = {256, 1024, 128};
        for (uint32_t si = 0; si < sizeof(seqs)/sizeof(*seqs); si++) {
            for (uint32_t bi = 0; bi < sizeof(blocks)/sizeof(*blocks); bi++) {
                for (size_t pi = 0; pi < psos.size(); pi++) {
                    auto pso = psos[pi];
                    NSString* n = [NSString stringWithUTF8String:pnames[pi].c_str()];
                    BOOL gqa_arm = [n hasPrefix:@"q27_attention_turbo3_gqa"] || [n hasPrefix:@"attn_"];
                    BOOL plain = [n isEqualToString:@"q27_attention_turbo3"] || [n isEqualToString:@"q27_attention_f16"];
                    BOOL merge = [n isEqualToString:@"q27_attention_gqa_merge"];
                    if (merge) continue;                     // timed with parent
                    if (plain) continue;   // turbo3-only run: engine routes >=1280 to gqa; plain f16 path timed in earlier runs  // engine routes >=1280 to gqa
                    uint32_t block = gqa_arm ? blocks[bi] : 1024;
                    if (plain && bi > 0) continue;
                    uint32_t nb = 1 + (seqs[si] - 1) / block;
                    const BOOL pad = [n containsString:@"_pad"] || [n containsString:@"stream64"];
                    AttentionGqaArgs ga{QS, seqs[si], QH, KVH, HD, block, nb, 0.0625f};
                    AttentionArgs pa{QS, seqs[si], QH, KVH, HD, 0.0625f};
                    NSUInteger gx = plain ? QH : KVH, gy = plain ? 1 : nb;
                    NSUInteger thr = plain ? 256 : GQA * 32;
                    float best = 1e30f;
                    @autoreleasepool {   // warmup (alloc + first-run shader setup)
                        auto cb = [q commandBuffer];
                        auto ce = [cb computeCommandEncoder];
                        [ce setComputePipelineState:pso];
                        [ce setBuffer:Q offset:0 atIndex:0];
                        [ce setBuffer:pad ? KCP : KC offset:0 atIndex:1];
                        [ce setBuffer:pad ? VCP : VC offset:0 atIndex:2];
                        [ce setBuffer:PART offset:0 atIndex:3];
                        [ce setBytes:&ga length:sizeof(ga) atIndex:4];
                        [ce dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
                        [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }
                    for (int rep = 0; rep < 5; rep++) {
                        @autoreleasepool {
                            auto cb = [q commandBuffer];
                            auto ce = [cb computeCommandEncoder];
                            [ce setComputePipelineState:pso];
                            [ce setBuffer:Q offset:0 atIndex:0];
                            [ce setBuffer:KC offset:0 atIndex:1];
                            [ce setBuffer:VC offset:0 atIndex:2];
                            [ce setBuffer:PART offset:0 atIndex:3];
                            if (plain) [ce setBytes:&pa length:sizeof(pa) atIndex:4];
                            else       [ce setBytes:&ga length:sizeof(ga) atIndex:4];
                            [ce dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                                threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
                            [ce endEncoding];
                            [cb commit];
                            [cb waitUntilCompleted];
                        }
                        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
                        @autoreleasepool {
                            auto cb = [q commandBuffer];
                            auto ce = [cb computeCommandEncoder];
                            [ce setComputePipelineState:pso];
                            [ce setBuffer:Q offset:0 atIndex:0];
                            [ce setBuffer:pad ? VCP : VC offset:0 atIndex:2];
                            [ce setBuffer:pad ? KCP : KC offset:0 atIndex:1];
                            [ce setBuffer:PART offset:0 atIndex:3];
                            if (plain) [ce setBytes:&pa length:sizeof(pa) atIndex:4];
                            else       [ce setBytes:&ga length:sizeof(ga) atIndex:4];
                            [ce dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                                threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
                            // 8 back-to-back dispatches: divide out the fixed
                            // encoder/commit/wait cost to expose device time.
                            for (int d = 0; d < 7; d++) {
                                [ce setComputePipelineState:pso];
                                if (plain) [ce setBytes:&pa length:sizeof(pa) atIndex:4];
                                else       [ce setBytes:&ga length:sizeof(ga) atIndex:4];
                                [ce dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                                    threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
                            }
                            [ce endEncoding];
                            [cb commit];
                            [cb waitUntilCompleted];
                        }
                        float us = (float)(CFAbsoluteTimeGetCurrent() - t0) * 1e6f / 8.0f;
                        if (us < best) best = us;
                    }
                    double kvbytes = ([n containsString:@"turbo3"]||[n containsString:@"t3"]) ? (double)seqs[si] * KVH * 2 * (pad?64:50) : (double)seqs[si] * KVH * HD * 2 * 2;
                    fprintf(out, "{\"kernel\":\"%s\",\"seq\":%u,\"block\":%u,\"us\":%.3f,\"kvgbs\":%.1f}\n",
                            n.UTF8String, seqs[si], block, best, kvbytes / (best * 1e-6));
                    fprintf(stderr, "%-22s %6u %6u %9.1f %9.2f %9.1f\n", n.UTF8String, seqs[si],
                            block, best, best / seqs[si] * 1000.0, kvbytes / (best * 1e-6));
                }
            }
        }
        // Numeric check: production turbo3_gqa partials vs attn_t3_w2row
        // partials (same random KV/q), max abs rel-diff on m/l/acc fields.
        {
            auto prod = make_pso(dev, prod_lib, @"q27_attention_turbo3_gqa");
            auto arm  = make_pso(dev, bench_lib, @"attn_t3_w2row");
            std::vector<float> p1((size_t)QH * 64 * 258);
            for (uint32_t si = 0; si < sizeof(seqs)/sizeof(*seqs); si++) {
                const uint32_t seq = seqs[si], block = 256;
                uint32_t nb = 1 + (seq - 1) / block;
                AttentionGqaArgs ga{QS, seq, QH, KVH, HD, block, nb, 0.0625f};
                auto run = [&](id<MTLComputePipelineState> p) {
                    memset(PART.contents, 0, (size_t)QH * 64 * 258 * 4);
                    @autoreleasepool {
                        auto cb = [q commandBuffer];
                        auto ce = [cb computeCommandEncoder];
                        [ce setComputePipelineState:p];
                        [ce setBuffer:Q offset:0 atIndex:0];
                        [ce setBuffer:KC offset:0 atIndex:1];
                        [ce setBuffer:VC offset:0 atIndex:2];
                        [ce setBuffer:PART offset:0 atIndex:3];
                        [ce setBytes:&ga length:sizeof(ga) atIndex:4];
                        [ce dispatchThreadgroups:MTLSizeMake(KVH, nb, 1)
                            threadsPerThreadgroup:MTLSizeMake(GQA * 32, 1, 1)];
                        [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    }
                };
                run(prod);
                memcpy(p1.data(), PART.contents, (size_t)QH * nb * 258 * 4);
                run(arm);
                float* a = p1.data(); float* b = (float*)PART.contents;
                double worst = 0.0; size_t at = 0;
                for (uint32_t qh = 0; qh < QH; qh++)
                    for (uint32_t k = 0; k < nb; k++) {
                        float* pa = a + ((size_t)qh * nb + k) * 258;   // p1 used tight layout? prod wrote with same nb layout
                        float* pb = b + ((size_t)qh * nb + k) * 258;
                        if (pa[1] == 0 && pb[1] == 0) continue;
                        for (uint32_t d = 0; d < 258; d++) {
                            double scale = fabs(pa[d]) + 1e-6;
                            double rel = fabs(pa[d] - pb[d]) / scale;
                            if (rel > worst) { worst = rel; at = d; }
                        }
                    }
                fprintf(out, "{\"numcheck\":\"attn_t3_w2row\",\"seq\":%u,\"max_rel\":%.3e}\n", seq, worst);
                fprintf(stderr, "numcheck seq %u max_rel=%.2e (field %zu)\n", seq, worst, at);
            }
        }

        fclose(out);
    }
    return 0;
}
