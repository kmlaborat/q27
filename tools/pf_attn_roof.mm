// pf_attn_roof.mm — D Step 1c: prefill attention (turbo3 causal t2) roofline
// at the MARGINAL 64K chunk: tokens=512, base_len=65024, block=1024 — the
// engine's actual last-chunk shape. Production source compiled as-is;
// attribution arms (bench_pf_attn_arms.txt, generated from the engine body)
// probe: stream floor / no-exp / no-staging. Pairwise: t2 vs t4 (engine
// kernels), producer-only vs producer+merge_rows (engine dispatch order).
// Same 8-back-to-back timing trick as attn_roof.mm; clean serial runs only.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct AttentionGqaCausalArgs { uint32_t q_stride, q_row_stride, base_len, q_heads, kv_heads, head_dim, block, n_blocks_max, tokens; float scale; };

static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, const char* n) {
    id<MTLFunction> f = [l newFunctionWithName:[NSString stringWithUTF8String:n]];
    if (!f) return nil;
    NSError* e = nil; auto p = [d newComputePipelineStateWithFunction:f error:&e];
    if (!p) fprintf(stderr, "pso %s: %s\n", n, e.localizedDescription.UTF8String);
    return p;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string outp = argc > 1 ? argv[1] : "bench/m1max/pf_attn_roof_64k.jsonl";
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        NSString* esrc = [NSString stringWithContentsOfFile:@"src/metal/q27_kernels.metal" encoding:NSUTF8StringEncoding error:&err];
        NSString* armstr = [NSString stringWithContentsOfFile:@"tools/bench_pf_attn_arms.txt" encoding:NSUTF8StringEncoding error:&err];
        std::string arms = armstr ? armstr.UTF8String : "";
        // single concatenated library: engine defs (incl turbo_dequant,
        // AttentionGqaCausalArgs) + bench arms
        NSString* all = [NSString stringWithFormat:@"%@\n%@", esrc, [NSString stringWithUTF8String:arms.c_str()]];
        auto lib = [dev newLibraryWithSource:all options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        auto q = [dev newCommandQueue];

        uint32_t QH = 24, KVH = 4, HD = 256, TOK = 512, BLK = 1024;
        for (int a = 2; a < argc; a++) {
            if (!strcmp(argv[a], "--tok128")) TOK = 128;
            if (!strcmp(argv[a], "--blk256")) BLK = 256;   // THIS machine's family default
            if (!strcmp(argv[a], "--blk128")) BLK = 128; }
        const uint32_t BASE = 65536 - TOK;
        const uint32_t NB = 1 + (BASE + TOK - 2) / BLK;
        auto buf = [&](size_t b){ return [dev newBufferWithLength:b options:MTLResourceStorageModeShared]; };
        auto Q  = buf((size_t)TOK * QH * HD * 4);
        auto KC = buf((size_t)65536 * KVH * 2 * 50);
        auto VC = buf((size_t)65536 * KVH * 2 * 50);
        auto P  = buf((size_t)TOK * QH * NB * 258 * 4);
        auto O  = buf((size_t)TOK * QH * HD * 4);
        uint64_t st = 0x9E3779B92437F11Dull;
        auto rnd = [&](){ st ^= st<<13; st ^= st>>7; st ^= st<<17; return st; };
        uint8_t* kc = (uint8_t*)KC.contents; uint8_t* vc = (uint8_t*)VC.contents;
        for (size_t i = 0; i < (size_t)65536 * KVH * 2 * 50; i++) { kc[i] = (uint8_t)rnd(); vc[i] = (uint8_t)rnd(); }
        float* qf = (float*)Q.contents;
        for (size_t i = 0; i < (size_t)TOK * QH * HD; i++) qf[i] = ((rnd() & 0xFFFF) / 65536.0f - 0.5f);
        AttentionGqaCausalArgs args{ HD, QH * HD, BASE, QH, KVH, HD, BLK, NB, TOK, 0.0625f };
        fprintf(stderr, "shape: tok=%u base=%u blk=%u nb=%u partials=%.0fMB\n", TOK, BASE, BLK, NB, P.length/(1024.0*1024));

        const char* names[] = { "q27_attention_turbo3_causal_gqa_t2",
                                "q27_attention_turbo3_causal_gqa_t4",
                                "pfa_stream", "pfa_noexp", "pfa_nostage", "pfa_nomax", "pfa_nosum" };
        const uint32_t zfac[] = { 2, 4, 2, 2, 2, 2, 2 };   // grid.z = tokens/TF per kernel
        FILE* out = fopen(outp.c_str(), "w");
        fprintf(stderr, "%-38s %10s %10s\n", "case", "us", "GB/s(dev)");
        for (int i = 0; i < 7; i++) {   // arms + 2 attribution probes (m1max D1c)
            auto ps = pso(dev, lib, names[i]);
            if (!ps) { fprintf(stderr, "missing %s\n", names[i]); continue; }
            uint32_t TF = zfac[i];
            float best = 1e30f;
            for (int rep = 0; rep < 5; rep++) {
                @autoreleasepool {
                    auto cb = [q commandBuffer]; auto ce = [cb computeCommandEncoder];
                    [ce setComputePipelineState:ps];
                    [ce setBuffer:Q offset:0 atIndex:0];
                    [ce setBuffer:KC offset:0 atIndex:1];
                    [ce setBuffer:VC offset:0 atIndex:2];
                    [ce setBuffer:P offset:0 atIndex:3];
                    [ce setBytes:&args length:sizeof(args) atIndex:4];
                    [ce dispatchThreadgroups:MTLSizeMake(KVH, NB, (TOK + TF - 1) / TF) threadsPerThreadgroup:MTLSizeMake(QH / KVH * 32, 1, 1)];
                    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
                @autoreleasepool {
                    auto cb = [q commandBuffer]; auto ce = [cb computeCommandEncoder];
                    for (int d = 0; d < 8; d++) {
                        [ce setComputePipelineState:ps];
                        [ce setBuffer:Q offset:0 atIndex:0];
                        [ce setBuffer:KC offset:0 atIndex:1];
                        [ce setBuffer:VC offset:0 atIndex:2];
                        [ce setBuffer:P offset:0 atIndex:3];
                        [ce setBytes:&args length:sizeof(args) atIndex:4];
                        [ce dispatchThreadgroups:MTLSizeMake(KVH, NB, (TOK + TF - 1) / TF) threadsPerThreadgroup:MTLSizeMake(QH / KVH * 32, 1, 1)];
                    }
                    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                float us = (float)(CFAbsoluteTimeGetCurrent() - t0) * 1e6f / 8.0f;
                if (us < best) best = us;
            }
            // device bytes: each (blk,kvh) tile reads its block rows (TOK/TF
            // groups share via L1/L2; unique per (blk,kvh) = block_rows*200B)
            double uniq = (double)NB * KVH * BLK * 2 * 50 * 0.5;   // triangle avg half-block
            fprintf(out, "{\"kernel\":\"%s\",\"us\":%.1f}\n", names[i], best);
            fprintf(stderr, "%-38s %10.1f %10.1f\n", names[i], best, uniq / (best * 1e-6) / 1e9);
            fflush(out);
        }
        // producer + merge_rows (full engine attention call, t2 route)
        {
            auto prod = pso(dev, lib, "q27_attention_turbo3_causal_gqa_t2");
            auto mrg  = pso(dev, lib, "q27_attention_gqa_merge_rows");
            float best = 1e30f;
            for (int rep = 0; rep < 5; rep++) {
                CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
                @autoreleasepool {
                    auto cb = [q commandBuffer]; auto ce = [cb computeCommandEncoder];
                    for (int d = 0; d < 8; d++) {
                        [ce setComputePipelineState:prod];
                        [ce setBuffer:Q offset:0 atIndex:0];
                        [ce setBuffer:KC offset:0 atIndex:1];
                        [ce setBuffer:VC offset:0 atIndex:2];
                        [ce setBuffer:P offset:0 atIndex:3];
                        [ce setBytes:&args length:sizeof(args) atIndex:4];
                        [ce dispatchThreadgroups:MTLSizeMake(KVH, NB, TOK / 2) threadsPerThreadgroup:MTLSizeMake(QH / KVH * 32, 1, 1)];
                        id<MTLResource> res[] = { P };
                        [ce memoryBarrierWithResources:res count:1];
                        [ce setComputePipelineState:mrg];
                        [ce setBuffer:P offset:0 atIndex:0];
                        [ce setBuffer:O offset:0 atIndex:1];
                        [ce setBytes:&args length:sizeof(args) atIndex:2];
                        [ce dispatchThreadgroups:MTLSizeMake(QH, TOK, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                float us = (float)(CFAbsoluteTimeGetCurrent() - t0) * 1e6f / 8.0f;
                if (us < best) best = us;
            }
            fprintf(out, "{\"kernel\":\"t2_plus_merge_rows\",\"us\":%.1f}\n", best);
            fprintf(stderr, "%-38s %10.1f\n", "t2_plus_merge_rows", best);
        }
        fclose(out);
    }
    return 0;
}
