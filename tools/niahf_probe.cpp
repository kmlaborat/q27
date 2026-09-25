// niahf_probe.cpp — D decision gate for turbo3 KV: "needle in a haystack"
// at 64K. Deep-context PRECISE RECALL (numbers, identifiers, dates) is the
// failure mode PPL/golden margins cannot see: int8 KV noise could cost a
// pinpoint retrieval disproportionately. One run = ~65K tokens of filler
// with three needles at ~2%/37%/86% depth + pinned query tail, greedy
// decode, substring scoring. fp16 vs turbo3 on the SAME prompt.
#include "metal_engine.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static double now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
using q27::MetalEngine;

static const char* kParas[] = {
  "Inference engines trade memory bandwidth for compute. The scheduler hides weight streaming latency behind verification passes, and the verifier accepts or rejects speculative tokens against the greedy reference.",
  "A storage system chooses between log-structured merges and inplace updates; write amplification, read amplification and space amplification form the trilemma that no design escapes, only reallocates.",
  "The compiler lowers source through several intermediate representations; each lowering preserves semantics while shrinking the abstraction until the register allocator can color its interference graph.",
  "Distributed consensus proceeds in rounds; a proposer gathers a quorum of promises before committing, and any recovery procedure must re-validate the durable log against the chosen ballot to avoid divergence.",
  "Networked storage exposes leases so clients cache read-only shards; revocation must fence old holders before a new primary accepts writes, otherwise stale writes surface as silent corruption.",
  "The planner enumerates join orders with dynamic programming over subsets and prunes branches whose estimated cost exceeds the incumbent by more than a safety factor tuned on samples.",
  "Garbage collection pauses scale with live set size, not heap size; a generational collector exploits the hypothesis that most objects die young while remembered sets bound cross-generation references.",
  "A tokenizer merges subword units by score; byte-pair encoding grows the vocabulary greedily from the most frequent adjacent pair, which is why rare identifiers fragment in surprising ways.",
  "The page cache maps files into the unified address space; the GPU shares the same physical pages, and memory pressure reclaims clean file-backed pages before dirty anonymous ones.",
  "Reproducible research demands pinned dependencies, seeded randomness and hardware disclosure, because performance claims drift between silicon generations in ways no single microbenchmark predicts.",
};
struct Needle { const char* text; double depth; const char* probes[3]; };
static const Needle kNeedles[] = {
  {"Inventory note: bay 7 contains exactly 412 units of FALCON-91-DELTA coolant.", 0.02, {"412", "FALCON", "DELTA"}},
  {"Reminder while refactoring: in ledger.go the function reconcileLedger sets retryBudget = 17 before the loop.", 0.37, {"retryBudget", "17", "reconcile"}},
  {"Reviewer sign-off for release 3.9.1 came from tanaka on 2026-08-14 after the flaky test fix.", 0.86, {"tanaka", "08-14", "3.9.1"}},
};
static const char* kQuery =
  "\nQuestion (answer with exact values from the notes above):\n"
  "1. How many units of coolant are in bay 7, and what is the coolant code?\n"
  "2. What value does reconcileLedger assign to retryBudget?\n"
  "3. Who signed off release 3.9.1 and on what date?\nAnswer:";

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr,"usage: niahf_probe model.tok.tok --; argv: model tok [fp16|turbo3] [out.jsonl]\n"); return 2; }
    const bool t3 = argc > 3 && std::string(argv[3]) == "turbo3";
    const char* outp = argc > 4 ? argv[4] : "/tmp/niahf.jsonl";
    q27::Tokenizer tok(argv[2]);
    const uint32_t TARGET = 65100, GEN = 96;

    auto enc = [&](const std::string& s) { auto e = tok.encode(s); return std::vector<uint32_t>(e.begin(), e.end()); };
    std::vector<uint32_t> p; int seg = 0;
    auto fill = [&](size_t want) {
        while (p.size() < want) {
            auto t = enc(std::string(kParas[seg % 10]) + " (seg " + std::to_string(seg) + ") ");
            p.insert(p.end(), t.begin(), t.end()); seg++;
        }
    };
    for (const auto& nd : kNeedles) {
        fill(size_t(TARGET * nd.depth));
        auto t = enc(std::string(" ") + nd.text + " ");
        p.insert(p.end(), t.begin(), t.end());
    }
    fill(TARGET);
    auto q = enc(kQuery);
    p.insert(p.end(), q.begin(), q.end());
    fprintf(stderr, "prompt tokens: %zu (mode %s)\n", p.size(), t3 ? "turbo3" : "fp16");

    MetalEngine eng(argv[1], 65536, t3);
    double t0 = now();
    eng.reset();
    eng.ingest_prompt(p, false);
    double t_ing = now() - t0;
    std::vector<uint32_t> s{eng.pending_from_logits()};
    for (uint32_t i = 1; i < GEN; i++) s.push_back(eng.step(s.back()));
    std::string ans = tok.decode(std::vector<int>(s.begin(), s.end()));
    fprintf(stdout, "ANSWER(%s): %s\n", t3 ? "turbo3" : "fp16", ans.c_str());
    int hits = 0, total = 0;
    for (const auto& nd : kNeedles)
        for (const char* pr : nd.probes) {
            total++;
            bool hit = ans.find(pr) != std::string::npos;
            if (hit) hits++;
            fprintf(stdout, "probe %-12s %s\n", pr, hit ? "HIT" : "MISS");
        }
    FILE* f = fopen(outp, "a");
    fprintf(f, "{\"mode\":\"%s\",\"prompt\":%zu,\"pp\":%.2f,\"hits\":%d,\"total\":%d,\"ans\":\"",
            t3 ? "turbo3" : "fp16", p.size(), p.size() / t_ing, hits, total);
    for (char c : ans) if (c == '\n') fputs("\\n", f); else fputc(c, f);
    fprintf(f, "\"}\n"); fclose(f);
    return 0;
}
