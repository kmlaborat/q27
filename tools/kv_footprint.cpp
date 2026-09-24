// kv_footprint.cpp — D Step 0 item 1: MEASURED engine KV math (constants and
// formulas as implemented in metal_engine.cpp), not chat arithmetic.
// Prints per-token bytes, KV GiB at 16K/32K/64K, and the engine's own
// reservation figures from serving_reservation_bytes() on this machine.
#include "metal_engine.h"
#include <cstdio>
#include <cstdint>

int main(int argc, char** argv) {
    using q27::MetalEngine;
    const char* model = argc > 1 ? argv[1] : "models/qwen36-27b-mtp-q4s.q27";
    // constants replicated from metal_engine.h (private); verified against the
    // header at the time of writing: N_LAYER=64 N_KV=4 HEAD_DIM=256
    const uint64_t N_KV = 4, HD = 256, NL = 64;
    printf("constants: N_LAYER=%llu N_KV=%llu HEAD_DIM=%llu\n", NL, N_KV, HD);
    for (bool turbo3 : {false, true}) {
        const unsigned long long row = turbo3 ? (unsigned long long)(N_KV * 2 * 50)
                                              : (unsigned long long)(N_KV * HD * 2);
        // engine allocates, PER attn layer, k_cache and v_cache of ctx*row bytes
        const unsigned attn17 = 17;  // 16 attn layers + MTP attn block (has_mtp_)
        const unsigned long long tok = row * 2ull * attn17;
        printf("\n== %s KV == cache_row=%llu B/token/layer -> %.2f KB/token (x2 K+V, x17)\n",
               turbo3 ? "turbo3 (8-bit)" : "fp16", row, tok / 1024.0);
        for (uint32_t ctx : {16384u, 32768u, 65536u})
            printf("  ctx %6u: KV cache only = %7.2f GiB\n", ctx, tok * ctx / (1024.0*1024*1024));
    }
    printf("\n--- engine's own serving_reservation_bytes (incl gqa scratch, side fp16 cells, snapshots, fixed state) ---\n");
    try {
        auto shared = MetalEngine::open_shared(model);
        printf("cache_budget (engine-reserved, this machine) = %.2f GiB\n",
               shared->cache_budget / (1024.0*1024*1024));
        for (bool turbo3 : {false, true})
            for (uint32_t ctx : {16384u, 32768u, 65536u}) {
                uint64_t r = MetalEngine::serving_reservation_bytes(*shared, ctx, turbo3, 0);
                printf("  %s ctx %6u: reservation = %8.2f GiB  %s budget\n",
                       turbo3 ? "turbo3" : "fp16  ", ctx, r / (1024.0*1024*1024),
                       r > shared->cache_budget ? "OVER" : "within");
            }
    } catch (const std::exception& e) {
        printf("  (live reservation unavailable: %s)\n", e.what());
    }
    return 0;
}
