// m1max #4: prefill stage-share inventory.
//
// Drives MetalEngine::pp_profile_chunk at several positions to attribute
// prefill wall time to: embedding, rmsnorm x2, attention (16 layers),
// GDN (48 layers), FFN (64 layers), residual adds x2. Attention cost
// grows with position (causal), everything else is position-independent;
// profile at low/mid/high positions and average the attention term to
// extrapolate a pp t/s estimate for a target length.
//
// Usage: pp_share MODEL TOK --kv turbo3|fp16 [--ctx N] [--reps N]
//        [--positions 0,3584,7104] [--prompt-file F]
#include "../src/metal/metal_engine.h"
#include "../src/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: pp_share MODEL TOK [opts]\n"); return 1; }
    uint32_t ctx = 8192, reps = 3;
    bool turbo3 = true;
    std::string positions = "0,3584,7104";
    std::string corpus;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* f) { if (i + 1 >= argc) throw std::runtime_error(f); return std::string(argv[++i]); };
        if (a == "--ctx") ctx = (uint32_t)std::stoul(need("--ctx"));
        else if (a == "--kv") { auto k = need("--kv"); turbo3 = (k == "turbo3"); }
        else if (a == "--reps") reps = (uint32_t)std::stoul(need("--reps"));
        else if (a == "--positions") positions = need("--positions");
        else if (a == "--prompt-file") {
            std::FILE* f = std::fopen(need("--prompt-file").c_str(), "rb");
            if (!f) throw std::runtime_error("no prompt file");
            std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::rewind(f);
            corpus.resize((size_t)sz);
            if (std::fread(&corpus[0], 1, (size_t)sz, f) != (size_t)sz) throw std::runtime_error("read fail");
            std::fclose(f);
        } else throw std::runtime_error("unknown arg " + a);
    }
    q27::Tokenizer tokenizer(argv[2]);
    q27::MetalEngine engine(argv[1], ctx, turbo3);
    engine.set_chunked_prefill(true);
    std::vector<uint32_t> toks;
    {
        std::vector<int> raw;
        if (!corpus.empty()) raw = tokenizer.encode(corpus);
        else {
            std::string s;
            for (int n = 0; n < 3000; n++) s += "The quick brown fox jumps over the lazy dog number " + std::to_string(n) + ". ";
            raw = tokenizer.encode(s);
        }
        toks.assign(raw.begin(), raw.end());
    }
    const uint32_t CH = engine.prefill_chunk_max();
    std::vector<uint32_t> poss; uint32_t p = 0;
    for (char c : positions) { if (c == ',') { poss.push_back(p); p = 0; } else p = p * 10 + (uint32_t)(c - '0'); }
    poss.push_back(p);
    const char* names[] = {"emb", "norm1", "attn", "gdn", "add1", "norm2", "ffn", "add2"};
    printf("kv=%s ctx=%u chunk=%u reps=%u\n", turbo3 ? "turbo3" : "fp16", ctx, CH, reps);
    std::vector<double> attn_by_pos;
    std::array<double, 8> last{};
    for (uint32_t P : poss) {
        engine.reset();
        while (engine.position() < P)
            engine.prefill_chunk(toks.data() + engine.position(), std::min(CH, P - engine.position()));
        std::vector<std::array<double, 8>> rs;
        double empty_commit = 0;
        for (uint32_t r = 0; r < reps; r++) {
            auto s = engine.pp_profile_chunk(toks.data() + P, CH);
            rs.push_back({s.emb, s.norm1, s.attn, s.gdn, s.add1, s.norm2, s.ffn, s.add2});
            empty_commit = s.empty_commit;
        }
        std::array<double, 8> med{};
        for (int k = 0; k < 8; k++) {
            std::vector<double> v;
            for (auto& a : rs) v.push_back(a[k]);
            std::sort(v.begin(), v.end());
            med[k] = v[v.size() / 2];
        }
        double total = 0; for (double x : med) total += x;
        printf("pos=%u empty_commit=%.3f ms  total=%.2f ms/chunk (%.2f ms/tok)\n",
               P, empty_commit, total, total / CH);
        for (int k = 0; k < 8; k++)
            printf("  %-6s %8.3f ms  %5.1f%%  %6.3f ms/tok\n",
                   names[k], med[k], 100.0 * med[k] / total, med[k] / CH);
        attn_by_pos.push_back(med[2]);
        last = med;
    }
    // Extrapolate: attention averaged over positions, rest from last profile.
    double attn_avg = 0; for (double x : attn_by_pos) attn_avg += x; attn_avg /= attn_by_pos.size();
    double fixed = 0;
    for (int k = 0; k < 8; k++) if (k != 2) fixed += last[k];
    double per_tok = (fixed + attn_avg) / CH;
    printf("EXTRAP: attn_avg=%.3f ms  fixed=%.3f ms  -> %.3f ms/tok = %.1f t/s (vs measured pp)\n",
           attn_avg, fixed, per_tok, 1000.0 / per_tok);
    return 0;
}
