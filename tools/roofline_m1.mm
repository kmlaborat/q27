// M1 Max matvec roofline + arm A/B harness (Step A of the matvec plan,
// 2026-09-23). Bench-only: loads the repo's real q27_kernels.metal source,
// plus a bench-only pure-stream kernel, and times three things on the
// engine's real layer shapes (read live from the .q27 file):
//
//   stream : grid = ceil(rows/32) x 256 threads, every lane reads 16B per
//            chunk from the weight range — same access geometry as the
//            production kernel, zero quant math. The machine's realistic
//            read roofline for this dispatch shape.
//   prod   : q27_matvec_q4_quantized exactly as the engine dispatches it
//            (32 rows/group, 256 threads).
//   r2     : q27_matvec_q4_quantized_r2 via the probe geometry (16 rows/
//            group). Retained M4-era comparison arm — its M4 promotion
//            numbers were measured on Apple9 and the round doc
//            (2026-07-17-q4-rewrite-round.md) is missing from the repo,
//            so this machine gets its own measurement.
//
// Reports us/dispatch and GB/s (weight bytes / us). No engine code involved.
//
// build: see header comment of bench command in bench/m1max/logs; standalone
// Metal + loader only.
// usage: roofline_m1 model.q27 [--layer pattern] [--iters N] [--out jsonl]

#include "loader.h"

#import <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

struct MatvecArgsBench { uint32_t rows; uint32_t cols; };

int main(int argc, char** argv) try {
    std::string path = argc > 1 ? argv[1] : "models/qwen36-27b-mtp-q4s.q27";
    std::string out_path = "bench/m1max/roofline_m1.jsonl";
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--out") out_path = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    // Repo kernel source (engine's own copy), plus the bench stream kernel.
    // Match the engine's compile flags (mathMode Safe) so measured PSOs are
    // the same binaries the engine builds.
    MTLCompileOptions* copts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) copts.mathMode = MTLMathModeSafe;
    NSData* repo = [NSData dataWithContentsOfFile:@"src/metal/q27_kernels.metal"];
    id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:(const char*)repo.bytes]
                                    options:copts error:&err];
    if (!lib) { fprintf(stderr, "repo kernels: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLLibrary> streamlib = [device newLibraryWithSource:[NSString stringWithContentsOfFile:@"tools/bench_stream.metal" encoding:NSUTF8StringEncoding error:&err] options:copts error:&err];
    if (!streamlib) { fprintf(stderr, "stream: %s\n", err.localizedDescription.UTF8String); return 1; }
        struct PSO { id<MTLComputePipelineState> p; NSString* name; };
    std::vector<PSO> psos;
    id<MTLFunction> sf = [streamlib newFunctionWithName:@"bench_stream"];
    psos.push_back({[device newComputePipelineStateWithFunction:sf error:&err], @"bench_stream"});
    for (int i = 0; i < 2; i++) {
        NSString* fn = i ? @"q27_matvec_q4_quantized_r2" : @"q27_matvec_q4_quantized";
        id<MTLFunction> f = [lib newFunctionWithName:fn];
        if (!f) { fprintf(stderr, "missing %s\n", fn.UTF8String); continue; }
        psos.push_back({[device newComputePipelineStateWithFunction:f error:&err], fn});
        fprintf(stderr, "%s maxTotal=%lu\n", fn.UTF8String,
                (unsigned long)psos.back().p.maxTotalThreadsPerThreadgroup);
    }

    q27::Model model = q27::Model::open(path);
    auto qbuf = [&](NSString* n){ return [device newCommandQueueWithMaxCommandBufferCount:1]; };

    std::ofstream out(out_path);
    // All 2-D Q4 weight tensors, deduped by shape; weight bytes = rows*cols/2.
    struct Shape { uint64_t rows, cols; size_t count; std::string name; };
    std::vector<Shape> shapes;
    for (auto& t : model.tensors) {
        if (t.rows() < 1 || t.cols() < 1 || t.rows() * t.cols() < 2500000) continue;
        bool found = false;
        for (auto& s : shapes)
            if (s.rows == t.rows() && s.cols == t.cols()) { found = true; s.count++; break; }
        if (!found) shapes.push_back({t.rows(), t.cols(), 1, t.name});
    }
    std::sort(shapes.begin(), shapes.end(), [](const Shape&a, const Shape&b){
        return a.rows * a.cols > b.rows * b.cols; });

    auto now = []{ return std::chrono::steady_clock::now(); };
    for (auto& s : shapes) {
        const uint64_t wbytes = s.rows * s.cols / 2;
        const uint32_t chunks = (uint32_t)(s.cols / 1024);
        if (s.cols % 64) continue;
        MTLCompileOptions* opts = nil; (void)chunks;
        id<MTLBuffer> W = [device newBufferWithLength:wbytes
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> S = [device newBufferWithLength:s.rows * (s.cols / 64) * 2
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> X = [device newBufferWithLength:s.cols options:MTLResourceStorageModeShared];
        id<MTLBuffer> XS = [device newBufferWithLength:(s.cols / 32) * 4
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> O = [device newBufferWithLength:s.rows * 4
                                              options:MTLResourceStorageModeShared];
        auto q = qbuf(nil);
        @autoreleasepool {
            auto enc = [q commandBuffer];
            for (auto& ps : psos) {
                const NSUInteger rpg = [ps.name isEqualToString:@"bench_stream"] ||
                        [ps.name isEqualToString:@"q27_matvec_q4_quantized"] ? 32 : 16;
                auto ce = [enc computeCommandEncoder];
                [ce setComputePipelineState:ps.p];
                [ce setBuffer:W offset:0 atIndex:0];
                [ce setBuffer:S offset:0 atIndex:1];
                [ce setBuffer:X offset:0 atIndex:2];
                [ce setBuffer:XS offset:0 atIndex:3];
                [ce setBuffer:O offset:0 atIndex:4];
                MatvecArgsBench a{(uint32_t)s.rows, (uint32_t)s.cols};
                [ce setBytes:&a length:sizeof(a) atIndex:5];
                [ce dispatchThreadgroups:MTLSizeMake((s.rows + rpg - 1) / rpg, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [ce endEncoding];
            }
            [enc commit]; [enc waitUntilCompleted];
        }
        // Timed run per arm.
        for (auto& ps : psos) {
            const NSUInteger rpg = [ps.name isEqualToString:@"bench_stream"] ||
                    [ps.name isEqualToString:@"q27_matvec_q4_quantized"] ? 32 : 16;
            int iters = std::max(3, (int)(300000000 / (double)wbytes * 30));  // >=30 passes
            iters = std::min(iters, 60);
            double best = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                auto t0 = now();
                @autoreleasepool {
                    auto cb = [q commandBuffer];
                    for (int i = 0; i < iters; i++) {
                        auto ce = [cb computeCommandEncoder];
                        [ce setComputePipelineState:ps.p];
                        [ce setBuffer:W offset:0 atIndex:0];
                        [ce setBuffer:S offset:0 atIndex:1];
                        [ce setBuffer:X offset:0 atIndex:2];
                        [ce setBuffer:XS offset:0 atIndex:3];
                        [ce setBuffer:O offset:0 atIndex:4];
                        MatvecArgsBench a{(uint32_t)s.rows, (uint32_t)s.cols};
                        [ce setBytes:&a length:sizeof(a) atIndex:5];
                        [ce dispatchThreadgroups:MTLSizeMake((s.rows + rpg - 1) / rpg, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [ce endEncoding];
                    }
                    [cb commit]; [cb waitUntilCompleted];
                }
                double us = std::chrono::duration<double, std::micro>(now() - t0).count() / iters;
                best = std::min(best, us);
            }
            const double gbs = wbytes / best / 1000.0;
            fprintf(stderr, "%-28s %5zu×%-6zu %9.1f us  %7.1f GB/s\n",
                    ps.name.UTF8String, s.rows, s.cols, best, gbs);
            out << "{\"kernel\":\"" << ps.name.UTF8String << "\",\"rows\":" << s.rows
                << ",\"cols\":" << s.cols << ",\"count\":" << s.count
                << ",\"us\":" << best << ",\"gbs\":" << gbs << "}\n";
        }
    }
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
