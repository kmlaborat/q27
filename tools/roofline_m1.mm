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
    for (NSString* sn : @[ @"bench_stream", @"bench_stream2", @"bench_stream_x", @"bench_stream_sc", @"bench_sc_dot2", @"bench_sc_dot4", @"bench_sc_constdot", @"bench_sc_halfdot4", @"bench_sc_halfdot4p", @"bench_sc_halfdot4u2" ]) {
        id<MTLFunction> sf = [streamlib newFunctionWithName:sn];
        if (!sf) { fprintf(stderr, "missing stream fn %s\n", sn.UTF8String); continue; }
        auto pso = [device newComputePipelineStateWithFunction:sf error:&err];
        psos.push_back({pso, sn});
        fprintf(stderr, "%s maxTotal=%lu\n", sn.UTF8String, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
    }
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

        // Randomize inputs for the bit-identity digest check (W: random
        // nibbles; S: fixed safe half 0.25 = 0x3400 patterns; x: random
        // signed bytes). All shapes here are multiples of 32 rows, so the
        // clamped-row race cannot perturb outputs.
        {
            uint64_t st = 0x9E3779B97F4A7C15ull ^ (s.rows << 32) ^ s.cols;
            auto xs64 = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
            uint8_t* wd = (uint8_t*)W.contents;
            for (uint64_t i = 0; i < wbytes; i += 8) *(uint64_t*)(wd + i) = xs64();
            uint16_t* sd = (uint16_t*)S.contents;
            for (uint64_t i = 0; i < s.rows * (s.cols / 64); i++)
                sd[i] = uint16_t(0x3C00u | ((xs64() >> 3) & 0x3Fu));   // 0.25..0.25*1.046..
            int8_t* xd = (int8_t*)X.contents;
            for (uint64_t i = 0; i < s.cols; i += 8) {
                uint64_t v = xs64();
                for (int b = 0; b < 8; b++) xd[i + b] = int8_t((v >> (b * 8)) & 0xff);
            }
            float* xsd = (float*)XS.contents;
            for (uint64_t i = 0; i < s.cols / 32; i++) xsd[i] = 1.0f + float(i % 7) * 0.01f;
        }
        // Bit-identity digest check: production kernel vs bench arms.
        {
            auto run_one = [&](id<MTLComputePipelineState> pso, NSUInteger rpg, float* dst) {
                @autoreleasepool {
                    auto cb = [q commandBuffer];
                    auto ce = [cb computeCommandEncoder];
                    [ce setComputePipelineState:pso];
                    [ce setBuffer:W offset:0 atIndex:0];
                    [ce setBuffer:S offset:0 atIndex:1];
                    [ce setBuffer:X offset:0 atIndex:2];
                    [ce setBuffer:XS offset:0 atIndex:3];
                    [ce setBuffer:O offset:0 atIndex:4];
                    MatvecArgsBench a{(uint32_t)s.rows, (uint32_t)s.cols};
                    [ce setBytes:&a length:sizeof(a) atIndex:5];
                    [ce dispatchThreadgroups:MTLSizeMake((s.rows + rpg - 1) / rpg, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
                }
                memcpy(dst, O.contents, s.rows * 4);
            };
            std::vector<float> ref(s.rows);
            for (auto& ps : psos) {
                if (![ps.name hasPrefix:@"bench_sc_"] && ![ps.name isEqualToString:@"bench_stream_sc"]) continue;
                if ([ps.name isEqualToString:@"bench_sc_constdot"]) continue;  // not comparable by design
                run_one(ps.p, [ps.name isEqualToString:@"q27_matvec_q4_quantized_r2"] ? 16 : 32, ref.data());
                std::vector<float> refprod(ref);
                (void)refprod; break;
            }
            // capture production reference explicitly
            for (auto& ps : psos)
                if ([ps.name isEqualToString:@"q27_matvec_q4_quantized"]) {
                    run_one(ps.p, 32, ref.data());
                    for (auto& ps2 : psos) {
                        NSString* n = ps2.name;
                        if (![n isEqualToString:@"bench_sc_dot4"] && ![n isEqualToString:@"bench_sc_halfdot4"] && ![n isEqualToString:@"bench_sc_halfdot4p"] && ![n isEqualToString:@"bench_sc_halfdot4u2"]) continue;
                        std::vector<float> got(s.rows);
                        run_one(ps2.p, 32, got.data());
                        size_t bad = 0;
                        for (uint64_t i = 0; i < s.rows; i++)
                            if (memcmp(&ref[i], &got[i], 4) != 0) bad++;
                        fprintf(stderr, "BITCHECK %-20s %5zux%-6zu mismatches=%zu %s\n",
                                n.UTF8String, s.rows, s.cols, bad, bad ? "FAIL" : "OK");
                        out << std::string("{\"bitcheck\":\"") + n.UTF8String + std::string("\", \"rows\":") + std::to_string((unsigned long long)s.rows) + std::string(", \"cols\":") + std::to_string((unsigned long long)s.cols) + std::string(", \"mismatches\":") + std::to_string(bad) + "}";
                    }
                    break;
                }
        }
        @autoreleasepool {
            auto enc = [q commandBuffer];
            for (auto& ps : psos) {
                const NSUInteger rpg = [ps.name isEqualToString:@"q27_matvec_q4_quantized_r2"] ? 16 : 32;
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
            const NSUInteger rpg = [ps.name isEqualToString:@"q27_matvec_q4_quantized_r2"] ? 16 : 32;
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
