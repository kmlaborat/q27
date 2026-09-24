// Phase 2A margin probe (m1max): characterize the single serial-vs-suffix
// token flip found at seq512 of the synthetic bench corpus (25th generated
// token, "13" vs "12"). Replays the greedy stream up to the flip position
// with the SAME prompt construction as bench_metal and dumps the top-2 logit
// margin of the step that produced the divergent token. Gate basis: the
// established margin threshold (root margin <= 0.5 => near-tie class, same
// as golden p29/step51), applied to the chunked-verify vs serial kernel-shape
// class per the user's Phase 2A rule.
#include "metal/metal_engine.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: p2_margin model.q27 tok.tok ids_file [--ctx N]\n"); return 2; }
    uint32_t ctx = 8192;
    for (int i = 4; i < argc; i++) if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = strtoul(argv[++i], nullptr, 10);
    q27::Tokenizer tokenizer(argv[2]);
    // Same corpus construction as bench_metal (must stay in sync: the flip
    // ids were produced by that prompt at seq=512).
    const std::string para =
        "The quick brown fox jumps over the lazy dog. Inference engines trade "
        "memory bandwidth for compute, and the scheduler hides the latency of "
        "weight streaming behind verification. Prefix reuse is the whole game. ";
    std::vector<int> base;
    for (int n = 0; n < 200 && base.size() < 65536; n++) {
        auto enc = tokenizer.encode(para + std::to_string(n) + " ");
        base.insert(base.end(), enc.begin(), enc.end());
    }
    std::vector<int> ids;
    if (FILE* f = fopen(argv[3], "r")) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::string buf((size_t)sz, '\0'); fread(&buf[0], 1, (size_t)sz, f); fclose(f);
        for (char* t = strtok(&buf[0] + buf.find(':') + 1, " \n"); t; t = strtok(nullptr, " \n"))
            ids.push_back(atoi(t));
    } else { fprintf(stderr, "cannot read ids\n"); return 1; }
    if (ids.size() < 30) { fprintf(stderr, "ids too short\n"); return 1; }

    q27::MetalEngine engine(argv[1], ctx, false);
    engine.set_chunked_prefill(true);
    const uint32_t seq = 512;
    std::vector<uint32_t> prompt(base.begin(), base.begin() + seq);
    engine.reset();
    engine.ingest_prompt(prompt, false);
    uint32_t tok = engine.step(prompt.back());          // id[0]
    fprintf(stderr, "ids[0..5]: %u %u %u %u %u / replay t0=%u\n", ids[0],ids[1],ids[2],ids[3],ids[4], tok);
    const size_t flip = 25;                              // divergence index (0-based)
    for (size_t i = 1; i <= flip; i++) {
        tok = engine.step(tok);
        
    }
    // Logits now describe the position whose greedy id was ids[flip-1] fed..
    // capture the distribution for the NEXT token (the flip row).
    auto lg = engine.read_logits();
    const size_t V = lg.size();
    size_t t1 = 0, t2 = 1;
    if (lg[1] > lg[0]) { t1 = 1; t2 = 0; }
    for (size_t v = 2; v < V; v++) {
        if (lg[v] > lg[t1]) { t2 = t1; t1 = v; }
        else if (lg[v] > lg[t2]) t2 = v;
    }
    printf("flip row: top1 id=%zu  margin(top1-top2)=%.4f  top2 id=%zu ; spec picked id=17 => %s\n",
           t1, lg[t1] - lg[t2], t2, (t2 == 17 ? "SPEC==TOP2 (near-tie flip)" : "NOT top2 (systematic!)"));
    return 0;
}
