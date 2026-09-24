// M1 Max prefill GEMM roofline harness (Phase 2B Step A, bench-only).
// Times production q27_matmul_q4_mm_h (compiled from engine source) and
// decomposition arms from tools/bench_pf.metal at the REAL prefill chunk
// geometry (x_rows=96, grid=(rows/32, ceil(96/16)), 128 threads).
// GB/s = unique weight bytes / time (multiply by ceil(x_rows/16)=6 for the
// device-traffic view; L2 may rescue part of the y-group re-reads).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
struct MatmulArgs { uint rows; uint cols; uint x_rows; uint simdgroups; };
static id<MTLComputePipelineState> pso(id<MTLDevice> d, id<MTLLibrary> l, NSString* n) {
    auto f = [l newFunctionWithName:n]; if (!f) return nil; NSError* e = nil;
    auto p = [d newComputePipelineStateWithFunction:f error:&e];
    if (!p) fprintf(stderr, "pso %s: %s\n", n.UTF8String, e.localizedDescription.UTF8String);
    return p;
}
int main(int argc, char** argv) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    NSString* srcp = [NSString stringWithUTF8String:"src/metal/q27_kernels.metal"];
    NSString* src = [NSString stringWithContentsOfFile:srcp encoding:NSUTF8StringEncoding error:&err];
    auto plib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if (!plib) { fprintf(stderr, "prod: %s\n", err.localizedDescription.UTF8String); return 1; }
    NSString* bsrc = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:"tools/bench_pf.metal"] encoding:NSUTF8StringEncoding error:&err];
    auto blib = [dev newLibraryWithSource:bsrc options:[MTLCompileOptions new] error:&err];
    if (!blib) { fprintf(stderr, "bench: %s\n", err.localizedDescription.UTF8String); return 1; }
    const uint64_t MAXW = (uint64_t)17408 * 5120 / 2;
    auto W = [dev newBufferWithLength:MAXW * 8 options:MTLResourceStorageModeShared];
    auto X = [dev newBufferWithLength:(size_t)96 * 17408 options:MTLResourceStorageModeShared];
    auto XS = [dev newBufferWithLength:(size_t)17408 * 4 options:MTLResourceStorageModeShared];
    auto WS = [dev newBufferWithLength:(size_t)17408 * 8192 options:MTLResourceStorageModeShared];
    auto O = [dev newBufferWithLength:(size_t)17408 * 96 * 4 options:MTLResourceStorageModeShared];
    uint8_t* w = (uint8_t*)W.contents; for (uint64_t i = 0; i < MAXW * 4; i++) w[i] = (uint8_t)(i * 7 + 3);
    int8_t* x = (int8_t*)X.contents; for (size_t i = 0; i < (size_t)96 * 17408; i++) x[i] = (int8_t)((i % 251) - 125);
    auto q = [dev newCommandQueue];
    struct Sh { const char* name; uint rows, cols; };
    Sh shapes[] = {{"ffn_up", 17408, 5120}, {"attn_o", 5120, 5120}, {"ffn_down", 5120, 17408}};
    const uint XROWS = 96;
    fprintf(stderr, "%-14s %-9s %9s %9s %9s\n", "kernel", "shape", "us/call", "uniqGB/s", "devGB/s");
    FILE* out = fopen(argc > 1 ? argv[1] : "/tmp/pf_roof.jsonl", "w");
    for (auto& s : shapes) {
        MatmulArgs a{s.rows, s.cols, XROWS, 4};
        const double wbytes = (double)s.rows * s.cols / 2;
        const uint ygroups = (XROWS + 15) / 16;
        for (int ai = 0; ai < 4; ai++) {
            const char* nm[] = {"q27_matmul_q4_mm_h", "pf_stream", "pf_lut", "pf_mma_peak"};
            auto p = pso(dev, ai == 0 ? plib : blib, [NSString stringWithUTF8String:nm[ai]]);
            if (!p) { fprintf(stderr, "missing %s\n", nm[ai]); continue; }
            float best = 1e30f;
            for (int rep = 0; rep < 5; rep++) {
                CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
                @autoreleasepool {
                    auto cb = [q commandBuffer]; auto ce = [cb computeCommandEncoder];
                    [ce setComputePipelineState:p];
                    [ce setBuffer:W offset:0 atIndex:0];
                    [ce setBuffer:WS offset:0 atIndex:1];
                    [ce setBuffer:X offset:0 atIndex:2];
                    [ce setBuffer:XS offset:0 atIndex:3];
                    [ce setBuffer:O offset:0 atIndex:4];
                    [ce setBytes:&a length:sizeof(a) atIndex:5];
                    [ce dispatchThreadgroups:MTLSizeMake((s.rows + 31) / 32, ygroups, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                float us = (float)(CFAbsoluteTimeGetCurrent() - t0) * 1e6f;
                if (us < best) best = us;
            }
            double ugb = wbytes / (best * 1e-6) / 1e9, dgb = ugb * ygroups;
            fprintf(stderr, "%-14s %-9s %9.0f %9.1f %9.1f\n", nm[ai], s.name, best, ugb, dgb);
            fprintf(out, "{\"kernel\":\"%s\",\"shape\":\"%s\",\"us\":%.0f,\"uniqgbs\":%.2f,\"devgbs\":%.2f}\n", nm[ai], s.name, best, ugb, dgb);
        }
    }
    fclose(out);
    return 0;
} }
