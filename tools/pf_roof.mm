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
#include <cmath>
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
        for (int ai = 0; ai < 8; ai++) {
            const char* nm[] = {"q27_matmul_q4_mm_h", "pf_stream", "pf_lut", "pf_mma_peak", "pf_b1_wide", "pf_b2_dbuf", "pf_c4_flushless", "pf_c5_prescale"};
            auto p = pso(dev, ai == 0 ? plib : blib, [NSString stringWithUTF8String:nm[ai]]);
            if (!p) { fprintf(stderr, "missing %s\n", nm[ai]); continue; }
            const uint yg = (ai == 4) ? 1 : ygroups;   // B1 reuses six token-windows per tile
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
                    [ce dispatchThreadgroups:MTLSizeMake((s.rows + 31) / 32, yg, 1)
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
    // --- bit-identity gate: B1/B2 must reproduce production mm_h exactly
    // (same tiles, same k-order, same flush cadence — only staging differs) ---
    {
        const uint r = 128, c = 128, xr = 96;
        MatmulArgs a{r, c, xr, 4};
        auto W2=[dev newBufferWithLength:(size_t)r*(c/2) options:MTLResourceStorageModeShared];
        auto X2=[dev newBufferWithLength:(size_t)xr*c options:MTLResourceStorageModeShared];
        auto O1=[dev newBufferWithLength:(size_t)xr*r*2 options:MTLResourceStorageModeShared];
        auto O2=[dev newBufferWithLength:(size_t)xr*r*2 options:MTLResourceStorageModeShared];
        auto O3=[dev newBufferWithLength:(size_t)xr*r*2 options:MTLResourceStorageModeShared];
        uint8_t* wp=(uint8_t*)W2.contents; for(size_t i=0;i<W2.length;i++) wp[i]=(uint8_t)(i*131+7);
        int8_t* xp=(int8_t*)X2.contents; for(size_t i=0;i<X2.length;i++) xp[i]=(int8_t)((i%199)-99);
        const uint names2[]={4,5}; const char* n2[]={"pf_b1_wide","pf_b2_dbuf"};
        id<MTLBuffer> Ob[]={O1,O2,O3}; id<MTLBuffer> Wb[]={W2,W2,W2}, Xb[]={X2,X2,X2};
        id<MTLComputePipelineState> ps[3]={pso(dev,plib,@"q27_matmul_q4_mm_h"),pso(dev,blib,@"pf_b1_wide"),pso(dev,blib,@"pf_b2_dbuf")};
        uint yg3[]={yg3[0]}; (void)yg3;
        uint ygs[3]={(xr+15)/16, 1, (xr+15)/16};
        for(int i=0;i<3;i++){
            @autoreleasepool{
            auto cb=[q commandBuffer]; auto ce=[cb computeCommandEncoder];
            [ce setComputePipelineState:ps[i]];
            [ce setBuffer:Wb[i] offset:0 atIndex:0];
            [ce setBuffer:Xb[i] offset:0 atIndex:1];
            [ce setBuffer:XS offset:0 atIndex:2];
            [ce setBuffer:WS offset:0 atIndex:3];
            [ce setBuffer:Ob[i] offset:0 atIndex:4];
            [ce setBytes:&a length:sizeof(a) atIndex:5];
            [ce dispatchThreadgroups:MTLSizeMake(r/32,ygs[i],1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [ce endEncoding];[cb commit];[cb waitUntilCompleted];
            }
        }
        auto*o1=(uint16_t*)O1.contents; auto*o2=(uint16_t*)O2.contents; auto*o3=(uint16_t*)O3.contents;
        (void)0;
        size_t n=(size_t)xr*r; long m1=0,m2=0; double md1=0,md2=0;
        for(size_t i=0;i<n;i++){ if(memcmp(o1+i,o2+i,2)) {m1++; float f1,f2; memcpy(&f1,o1+i,2); memcpy(&f2,o2+i,2); double d=fabs((double)f2-f1)/(fabs(f1)+1e-2); if(d>md1)md1=d;} if(memcmp(o1+i,o3+i,2)){m2++; float f1,f2; memcpy(&f1,o1+i,2); memcpy(&f2,o3+i,2); double d=fabs((double)f2-f1)/(fabs(f1)+1e-2); if(d>md2)md2=d;} }
        fprintf(stderr,"bitcheck B1 mismatches %ld (maxrel %.5f) | B2 mismatches %ld (maxrel %.5f)\n",m1,md1,m2,md2);
    }
    return 0;
} }
