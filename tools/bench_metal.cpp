// M1 Max tuning rig: llama-bench-style pp/tg measurement for the Metal engine.
// Separate timing for prefill (ingest_prompt) and decode (step loop), at
// several context lengths, median over reps. Also records load time,
// in-task memory (internal + wired) and recommendedMaxWorkingSetSize.
//
// build: see bench/m1max/build_bench.sh (kept out of the main Makefile until
// the numbers stabilize).
//
// usage: bench_metal model.q27 tokenizer.tok [--gen N] [--reps N]
//          [--ctx N] [--kv fp16|turbo3] [--seq a,b,c ...] [--mode greedy|suffix]
//          [--width W] [--min-match M] [--out file.jsonl]
//
// --seq: prompt token lengths to test (default 128,512,1024,2048,4096,8192).
// Each row: load once, then per (seq): reset -> timed ingest -> timed decode.

#include "../src/metal/metal_engine.h"
#include "../src/metal/metal_backend.h"
#include "../src/tokenizer.h"
#include "../src/suffixdraft.h"

#include <stdexcept>
#include <sys/sysctl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mach/mach.h>
#include <string>
#include <thread>
#include <cstdlib>
#include <vector>
#include <algorithm>

static double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Mem { uint64_t internal = 0, compressed = 0; };
static Mem task_memory() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    Mem m;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        m.internal = info.internal;
        m.compressed = info.compressed;
    }
    return m;
}

static uint64_t sysctl_u64(const char* name) {
    uint64_t v = 0; size_t sz = sizeof(v);
    sysctlbyname(name, &v, &sz, nullptr, 0);
    return v;
}

int main(int argc, char** argv) try {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.q27 tokenizer.tok [--gen N] [--reps N] [--ctx N] [--kv fp16|turbo3] [--seq 512,4096] [--mode greedy|suffix] [--width W] [--min-match M] [--out file.jsonl]\n", argv[0]);
        return 2;
    }
    uint32_t gen = 128, ctx = 8192, width = 4, min_match = q27::MetalEngine::SUFFIX_MIN_MATCH;
    int reps = 3;
    bool turbo3_kv = false;
    bool suffix_mode = false;
    std::string prompt_file;
    bool chunked_prefill = true;
    std::vector<uint32_t> seqs = {128, 512, 1024, 2048, 4096};
    std::string out_path;

    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* n) { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", n); exit(2);} return std::string(argv[++i]); };
        if (a == "--gen") gen = (uint32_t)std::stoul(need("--gen"));
        else if (a == "--reps") reps = std::stoi(need("--reps"));
        else if (a == "--ctx") ctx = (uint32_t)std::stoul(need("--ctx"));
        else if (a == "--kv") { auto m = need("--kv"); if (m == "turbo3") turbo3_kv = true; else if (m != "fp16") throw std::runtime_error("--kv fp16|turbo3"); }
        else if (a == "--seq") {
            auto s = need("--seq");
            seqs.clear();
            size_t p = 0;
            while (p < s.size()) {
                size_t c = s.find(',', p);
                if (c == std::string::npos) c = s.size();
                seqs.push_back((uint32_t)std::stoul(s.substr(p, c - p)));
                p = c + 1;
            }
        }
        else if (a == "--prefill") { auto m = need("--prefill"); if (m != "chunk" && m != "serial") throw std::runtime_error("--prefill chunk|serial"); chunked_prefill = (m == "chunk"); }
        else if (a == "--mode") { auto m = need("--mode"); suffix_mode = (m == "suffix"); if (m != "greedy" && m != "suffix") throw std::runtime_error("--mode greedy|suffix"); }
        else if (a == "--width") width = (uint32_t)std::stoul(need("--width"));
        else if (a == "--min-match") min_match = (uint32_t)std::stoul(need("--min-match"));
        else if (a == "--prompt-file") prompt_file = need("--prompt-file");
        else if (a == "--out") out_path = need("--out");
        else throw std::runtime_error("unknown arg " + a);
    }

    const auto t_load0 = now_s();
    q27::Tokenizer tokenizer(argv[2]);
    q27::MetalEngine engine(argv[1], ctx, turbo3_kv);
    engine.set_chunked_prefill(chunked_prefill);
    const double load_s = now_s() - t_load0;

    const uint64_t rws = engine.backend().recommended_working_set_size();
    fprintf(stderr, "device=%s recommendedMaxWorkingSetSize=%.2f GiB load=%.2f s\n",
            engine.backend().name().c_str(), rws / 1073741824.0, load_s);

    // --prompt-file: tokenize a real-text corpus instead of the synthetic
    // repeat paragraph (m1max Phase 2A: speculation gains are corpus-bound;
    // code/diff/prose samples measure what real workloads would see).
    // Corpus prompt: repeat a real English paragraph until length N tokens.
    const std::string para =
        "The quick brown fox jumps over the lazy dog. Inference engines trade "
        "memory bandwidth for compute, and the scheduler hides the latency of "
        "weight streaming behind verification. Prefix reuse is the whole game. ";
    std::vector<int> base;
    if (!prompt_file.empty()) {
        std::FILE* pf = std::fopen(prompt_file.c_str(), "rb");
        if (!pf) { fprintf(stderr, "prompt-file: cannot open %s\n", prompt_file.c_str()); return 1; }
        std::fseek(pf, 0, SEEK_END); long sz = std::ftell(pf); std::rewind(pf);
        std::string txt((size_t)sz, '\0');
        if (std::fread(&txt[0], 1, (size_t)sz, pf) != (size_t)sz) { std::fclose(pf); return 1; }
        std::fclose(pf);
        base = tokenizer.encode(txt);
        fprintf(stderr, "prompt-file: %zu tokens from %s\n", base.size(), prompt_file.c_str());
    } else
    for (int n = 0; n < 4000 && base.size() < 262144; n++) {   // m1max Phase D: reach 64K+ prompts (was 200x~36 ~= 7.2K)
        auto enc = tokenizer.encode(para + std::to_string(n) + " ");
        base.insert(base.end(), enc.begin(), enc.end());
    }

    std::ofstream out(out_path.empty() ? "/dev/stdout" : out_path);
    std::vector<double> med_pp, med_tg;
    for (uint32_t seq : seqs) {
        if (seq + gen + 8 > ctx) { fprintf(stderr, "skip seq=%u (ctx %u)\n", seq, ctx); continue; }
        std::vector<uint32_t> prompt(base.begin(), base.begin() + seq);
        std::vector<double> pp, tg;
        uint64_t digest = 0;
        uint64_t stats_rounds=0, stats_accepted=0, stats_burst=0, stats_fallback=0;
        for (int r = 0; r < reps; r++) {
            engine.reset();
            const double t0 = now_s();
            engine.ingest_prompt(prompt, false);
            const double t1 = now_s();
            uint32_t tok = engine.step(prompt.back());  // first decode token
            (void)tok;
            const double t2 = now_s();
            // Speculation-equivalence digest (m1max Phase 2A): committed id
            // stream must match the greedy stream token-for-token across any
            // width/min_match config. FNV-1a over exactly gen-1 ids.
            std::vector<uint32_t> dstream;
            if (suffix_mode) {
                // Mirror generate_suffix(): drafter appends happen inside suffix_step.
                std::vector<int> history(prompt.begin(), prompt.end());
                history.push_back((int)tok);
                q27::SuffixDraft drafter;
                drafter.reset(history);
                uint32_t pending = tok;
                uint32_t produced = 1;
                std::vector<uint32_t> committed;
                while (produced < gen - 1) {
                    pending = engine.suffix_step(drafter, pending, gen - produced,
                                                 UINT32_MAX, width, min_match, committed);
                    dstream.insert(dstream.end(), committed.begin(), committed.end());
                    produced += (uint32_t)committed.size();
                }
                if (dstream.empty()) dstream.push_back(tok);
                const auto ss = engine.last_suffix_stats();
                stats_burst += ss.burst_rounds; stats_fallback += ss.fallback_rounds;
                while (dstream.size() < gen - 1 && dstream.back() != pending)
                    dstream.push_back(pending);
            } else {
                dstream.push_back(tok);
                for (uint32_t g = 1; g < gen; g++) { tok = engine.step(tok); dstream.push_back(tok); }
                dstream.pop_back();  // same length basis as suffix: gen-1 ids
            }
            if (dstream.size() > gen - 1) dstream.resize(gen - 1);
            const auto sp = engine.last_spec_stats();
            stats_rounds += sp.rounds; stats_accepted += sp.accepted;
            if (getenv("Q27_BENCH_DUMPIDS")) {
                fprintf(stderr, "ids %zu:", dstream.size());
                for (uint32_t id : dstream) fprintf(stderr, " %u", id);
                fprintf(stderr, "\n");
            }
            digest = 1469598103934665603ull;
            for (uint32_t id : dstream) { digest ^= id; digest *= 1099511628211ull; }
            const double t3 = now_s();
            pp.push_back(seq / (t1 - t0));
            tg.push_back((gen - 1) / (t3 - t2));
        }
        std::sort(pp.begin(), pp.end()); std::sort(tg.begin(), tg.end());
        const double mpp = pp[pp.size() / 2], mtg = tg[tg.size() / 2];
        Mem m = task_memory();
        fprintf(stderr, "seq=%-5u gen=%-4u pp=%8.1f t/s  tg=%6.2f t/s  mem internal=%.2f GiB\n",
                seq, gen, mpp, mtg, m.internal / 1073741824.0);
        out << "{\"seq\":" << seq << ",\"gen\":" << gen << ",\"reps\":" << reps
            << ",\"mode\":\"" << (suffix_mode ? "suffix" : "greedy") << "\""
            << ",\"width\":" << width << ",\"min_match\":" << min_match
            << ",\"rounds\":" << stats_rounds << ",\"accepted\":" << stats_accepted
            << ",\"burst\":" << stats_burst << ",\"fallback\":" << stats_fallback
            << ",\"stream_digest\":\"0x" << std::hex << digest << std::dec
            << "\"" ",\"ctx\":" << ctx << ",\"kv\":\"" << (turbo3_kv ? "turbo3" : "fp16") << "\""
            << ",\"pp_med\":" << mpp << ",\"tg_med\":" << mtg
            << ",\"pp_all\":[";
        for (size_t k = 0; k < pp.size(); k++) out << (k ? "," : "") << pp[k];
        out << "],\"tg_all\":[";
        for (size_t k = 0; k < tg.size(); k++) out << (k ? "," : "") << tg[k];
        out << "],\"load_s\":" << load_s << ",\"rws\":" << rws
            << ",\"mem_internal\":" << m.internal << ",\"mem_compressed\":" << m.compressed
            << ",\"hw_memsize\":" << sysctl_u64("hw.memsize") << "}\n";
    }
    return 0;
} catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
