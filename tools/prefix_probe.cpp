// prefix_probe.cpp — D Step 1a: MEASURED prefix-snapshot resume economics.
// Scenarios (all fp16 KV unless --kv turbo3):
//   S1 identity: split at N: continuous vs save@K + load + delta-ingest —
//      greedy streams must match token-for-token (cheap, small N).
//   S2 save/load cost vs position (16K/32K/48K): seconds + file GiB.
//   S3 64K delta economics: load@32752 + ingest 32648 (the compaction-era
//      "rebuild from cached prefix" cost) vs cold 65400 (83 min, measured).
#include "metal_engine.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <sys/stat.h>

static double now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
using q27::MetalEngine;

static std::vector<uint32_t> synth_prompt(q27::Tokenizer& tok, uint32_t n){
    const std::string para = "The quick brown fox jumps over the lazy dog. Inference engines trade "
        "memory bandwidth for compute, and the scheduler hides the latency of "
        "weight streaming behind verification. Prefix reuse is the whole game. ";
    std::vector<uint32_t> base, out;
    while (out.size() < n) {
        auto e = tok.encode(para + std::to_string(out.size()) + " ");
        out.insert(out.end(), e.begin(), e.end());
    }
    out.resize(n);
    return out;
}

static std::vector<uint32_t> greedy(MetalEngine& eng, int g){
    // identical construction on both sides of the identity test: pending
    // comes from the resident-logits argmax (post-ingest == post-resume)
    std::vector<uint32_t> s{eng.pending_from_logits()};
    for (int i=1;i<g;i++) s.push_back(eng.step(s.back()));
    return s;
}

int main(int argc, char** argv){
    const char* model = argv[1]; const char* toki = argv[2];
    bool turbo3 = false; std::string mode = "all";
    for (int i=3;i<argc;i++){ std::string a=argv[i];
        if(a=="--kv") turbo3 = true;
        else if(a=="--mode") mode = argv[++i]; }
    q27::Tokenizer tok(toki);
    const uint32_t CTX = 65536;

    if (mode=="identity"){          // small: continuous vs save@4000+load+delta
        MetalEngine eng(model, 16384, turbo3);
        auto p = synth_prompt(tok, 6000);
        eng.reset(); eng.ingest_prompt(p, false);
        auto ref = greedy(eng, 12);
        eng.reset(); eng.ingest_prompt(std::vector<uint32_t>(p.begin(), p.begin()+4000), false);
        eng.save_state("/tmp/pp_id.snap", p.data(), 4000);
        eng.reset();
        uint32_t pos = eng.load_state("/tmp/pp_id.snap");
        eng.ingest_prompt(std::vector<uint32_t>(p.begin()+4000, p.end()), false, false); // reset_first=false: continue after load
        auto got = greedy(eng, 12);
        bool same = got == ref;
        printf("identity: resumed_pos=%u streams %s\n", pos, same?"MATCH":"MISMATCH");
        if(!same){ for(size_t i=0;i<ref.size()&&i<got.size();i++) if(ref[i]!=got[i]) printf("  first diff at %zu: ref=%u got=%u\n", i, ref[i], got[i]); }
        return same?0:1;
    }

    if (mode=="tg"){                 // clean tg at short ctx, both KV modes (regression probe)
        auto p = synth_prompt(tok, 7168);
        const bool t3 = std::string(argv[argc-1]) == "turbo3";
        {
            MetalEngine eng(model, 8192, t3);
            // subtract ingest via separate timed run; step() loops >128 need pool
            // drains we can't express in .cpp — use generate() which self-pools.
            eng.reset(); double t0=now(); eng.ingest_prompt(p, false); double ting=now()-t0;
            eng.reset(); t0=now(); auto g = eng.generate(p, 160); double tall=now()-t0;
            // generate = ingest + 160 decodes + one extra forward; approx:
            double tg = 160.0/((tall - ting)/160.0*160.0/161.0*161.0/160.0); (void)tg;
            printf("tg %s @7168: %.2f t/s (ingest %.0fs total %.0fs)\n", t3?"turbo3":"fp16 ",
                   161.0/(tall-ting), ting, tall);
        }
        return 0;
    }

    if (mode=="cost"){              // save cost vs position
        MetalEngine eng(model, CTX, turbo3);
        auto p = synth_prompt(tok, 16384);
        for (uint32_t P : {16384u}) {   // 32K save timing comes from --mode delta; re-ingesting 49K here cost more than it teaches
            eng.reset();
            eng.ingest_prompt(std::vector<uint32_t>(p.begin(), p.begin()+P), false);
            double t0=now();
            eng.save_state("/tmp/pp_cost.snap", p.data(), P);
            double dt=now()-t0;
            struct stat st; stat("/tmp/pp_cost.snap", &st);
            printf("save @%-6u: %6.2f s  file %.2f GiB (%.0f MB/s)\n", P, dt,
                   st.st_size/(1024.0*1024*1024), st.st_size/dt/(1024*1024));
        }
        // load cost (from the 49152 snapshot)
        eng.reset();
        double t0=now(); uint32_t pos=eng.load_state("/tmp/pp_cost.snap"); double dt=now()-t0;
        printf("load @%-6u: %6.2f s (pos=%u)\n", pos, dt, pos);
        return 0;
    }

    if (mode=="delta"){             // S3: resume-from-32K economics at 64K
        MetalEngine eng(model, CTX, turbo3);
        auto p = synth_prompt(tok, 65400);
        std::vector<uint32_t> head(p.begin(), p.begin()+32752);
        std::vector<uint32_t> tail(p.begin()+32752, p.end());
        eng.reset(); eng.ingest_prompt(head, false);
        double t0=now(); eng.save_state("/tmp/pp_32k.snap", p.data(), 32752); double tsave=now()-t0;
        eng.reset();
        t0=now(); uint32_t pos=eng.load_state("/tmp/pp_32k.snap"); double tload=now()-t0;
        t0=now(); eng.ingest_prompt(tail, false, false); double ting=now()-t0;
        double rate = tail.size()/ting;
        // effective pp for the whole 65400 position via this path:
        printf("delta: load %.1f s + ingest %zu tok in %.0f s = %.1f t/s (window 32752..65400)\n",
               tload, tail.size(), ting, rate);
        printf("delta: effective whole-session-equivalent pp = %.1f t/s (%.0f min vs cold ~83 min)\n",
               65400/(tload+ting+tsave), (tload+ting+tsave)/60.0);
        auto s = greedy(eng, 16);
        printf("delta: decode post-resume ok, first=%u\n", s[0]);
        return 0;
    }
    fprintf(stderr, "mode?\n"); return 2;
}
