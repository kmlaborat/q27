// M1 Max correctness baseline rig (phase 3 of the tuning instructions):
//
//   golden: greedy token trajectories + top-k logit fingerprints for a fixed
//           prompt set. Digest over all trajectories == the machine canonical
//           for this build; re-run after every kernel change and diff.
//   ppl:    teacher-forced NLL over a raw-text corpus (wikitext-2 test),
//           windowed through MetalEngine::teacher_force_logits_wide.
//
// usage: golden_metal model.q27 tok.tok golden [--prompts f] [--gen N]
//        [--topk K] [--reps R] --out f.jsonl [--print-digest]
//        golden_metal model.q27 tok.tok ppl --text corpus.txt [--tokens N]
//        [--window W] --out f.json
//
// Greedy here is the bare step() loop (argmax), no MTP/speculation: the
// canonical gate already covers spec-vs-greedy equality separately.

#include "../src/metal/metal_engine.h"
#include "../src/tokenizer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// 32 fixed prompts: code, prose, math, Japanese, systems, JSON, repetition,
// long-range reference. Kept inline so the corpus has no external file drift.
static const char* kPrompts[] = {
    "Explain in two sentences what a CPU cache is and why it helps.",
    "def fib(n):\n    if n < 2: return n\n    return ",
    "The largest prime below 100 that is one more than a square is",
    "Write a haiku about RAM running out at 3 a.m.",
    "次の文を日本語で三文に要約してください。The quick brown fox jumps over the lazy dog. Dog.",
    "Translate to English: 今日は良い天気ですね。明日も",
    "1, 1, 2, 3, 5, 8, 13,",
    "In C, the difference between memcpy and memmove is that",
    "{\"servers\": [{\"host\": \"alpha\", \"port\": 8080}, {\"host\":",
    "The A/C set to 24 degrees. Later, the room felt cold, so the user raised the set point to",
    "Once upon a time, in a village at the edge of a desert, there lived a girl named",
    "SELECT a.id, b.name FROM users a JOIN orders b ON a.id = b.user_id WHERE b.total >",
    "Rust's borrow checker rejects the following code because",
    "Rank these by latency: L1 cache, NVMe SSD, DRAM, register file, 10Gb Ethernet.",
    "A train leaves Kyoto at 9:12 traveling at 60 km/h. Another leaves at 9:40 at 80 km/h. The second catches the first at",
    "The mitochondria is the",
    "git rebase -i HEAD~3 opens",
    "Explain the difference between a mutex and a semaphore with one example each.",
    "Repeat exactly: memory bandwidth is the whole game. memory bandwidth is the whole game. memory bandwidth is the",
    "curl -s localhost:8080/v1/messages -d '{\"model\":",
    "class Matrix:\n    def __init__(self, rows):\n        self.rows = rows\n\n    def transpose(self):\n        return",
    "Why does the Metal backend on Apple Silicon bind at memory bandwidth rather than FLOPs? Because",
    "Summarize: 'Space invaders landed. They asked for the leader. The leader turned out to be a cat.' Summary:",
    "The MD5 of the empty string is d41d8cd98f00b204e9800998ecf8427e; the MD5 of 'hello' is",
    "Quant à Paris, il",
    "What is 27 * 43? Show the multiplication digit by digit.",
    "The term 'KV cache' in LLM inference refers to",
    "Fix this bug: for (int i = 0; i <= n; i++) arr[i] = 0; when n equals the array length,",
    "List the first ten files you would expect in a C compiler repo:",
    "If all Bloops are Razzies and all Razzies are Lazzies, then all Bloops are definitely",
    "Describe the taste of water.",
    "Attention is all you",
};

static std::string fnv_md5_like(const std::vector<uint32_t>& stream) {
    // FNV-1a 64 over the id stream; stable across runs, enough as a digest.
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t v : stream) {
        for (int b = 0; b < 4; b++) { h ^= (v >> (b * 8)) & 0xff; h *= 1099511628211ULL; }
    }
    char buf[17];
    snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
    return buf;
}

int main(int argc, char** argv) try {
    if (argc < 3) { fprintf(stderr, "usage: golden_metal model.q27 tok.tok golden|ppl [options] --out file\n"); return 2; }
    const std::string mode = argv[3];
    std::string out_path = "/dev/stdout";
    std::string text_path, prompt_path;
    uint32_t gen = 96, topk = 16, ctx = 4096, ppl_tokens = 16384, window = 192;
    int reps = 2;

    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* n) { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", n); exit(2);} return std::string(argv[++i]); };
        if (a == "--out") out_path = need("--out");
        else if (a == "--text") text_path = need("--text");
        else if (a == "--prompts") prompt_path = need("--prompts");
        else if (a == "--gen") gen = (uint32_t)std::stoul(need("--gen"));
        else if (a == "--topk") topk = (uint32_t)std::stoul(need("--topk"));
        else if (a == "--ctx") ctx = (uint32_t)std::stoul(need("--ctx"));
        else if (a == "--reps") reps = std::stoi(need("--reps"));
        else if (a == "--tokens") ppl_tokens = (uint32_t)std::stoul(need("--tokens"));
        else if (a == "--window") window = (uint32_t)std::stoul(need("--window"));
        else throw std::runtime_error("unknown arg " + a);
    }

    q27::Tokenizer tokenizer(argv[2]);
    q27::MetalEngine engine(argv[1], ctx, false);

    if (mode == "golden") {
        std::vector<std::string> prompts;
        if (!prompt_path.empty()) {
            std::ifstream pf(prompt_path);
            if (!pf) throw std::runtime_error("cannot open " + prompt_path);
            std::string line;
            while (std::getline(pf, line)) if (!line.empty()) prompts.push_back(line);
        } else {
            for (const char* p : kPrompts) prompts.push_back(p);
        }
        std::ofstream out(out_path);
        std::vector<uint32_t> digest_stream;
        for (int rep = 0; rep < reps; rep++) {
            for (size_t p = 0; p < prompts.size(); p++) {
                engine.reset();
                auto enc = tokenizer.encode(prompts[p]);
                std::vector<uint32_t> prompt(enc.begin(), enc.end());
                engine.ingest_prompt(prompt, false);
                uint32_t tok = engine.step(prompt.back());
                std::vector<uint32_t> ids;
                // top-k fingerprint of the first decode position.
                auto logits = engine.read_logits();
                std::vector<uint32_t> order(logits.size());
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                                  [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
                ids.push_back(tok);
                for (uint32_t g = 1; g < gen; g++) ids.push_back(engine.step(ids.back()));
                std::string h = fnv_md5_like(ids);
                digest_stream.insert(digest_stream.end(), ids.begin(), ids.end());
                out << "{\"rep\":" << rep << ",\"prompt\":" << p
                    << ",\"n\":" << ids.size() << ",\"digest\":\"" << h << "\",\"ids\":[";
                for (size_t k = 0; k < ids.size(); k++) out << (k ? "," : "") << ids[k];
                out << "],\"top\":[";
                for (uint32_t k = 0; k < topk; k++)
                    out << (k ? "," : "") << "{\"id\":" << order[k] << ",\"v\":" << logits[order[k]] << "}";
                out << "]}\n";
                fprintf(stderr, "rep%d p%-3zu %s\n", rep, p, h.c_str());
            }
        }
        fprintf(stderr, "DIGEST rep-independent stream (%zu ids): %s\n",
                digest_stream.size() / reps, fnv_md5_like(digest_stream).c_str());
        fprintf(stderr, "DIGEST full (all reps): %s\n", fnv_md5_like(digest_stream).c_str());
        return 0;
    }

    if (mode == "ppl") {
        if (text_path.empty()) throw std::runtime_error("ppl needs --text");
        std::ifstream tf(text_path);
        std::stringstream ss; ss << tf.rdbuf();
        std::string text = ss.str();
        auto enc = tokenizer.encode(text);
        std::vector<uint32_t> toks(enc.begin(), enc.end());
        if (toks.size() > ppl_tokens) toks.resize(ppl_tokens);
        if (toks.size() < window + 2) throw std::runtime_error("corpus too short");
        double nll = 0.0;
        uint64_t counted = 0;
        std::vector<float> logits;
        for (uint32_t base = 0; base + 1 < (uint32_t)toks.size(); base += window) {
            const uint32_t count = std::min(window, (uint32_t)toks.size() - base - 1);
            engine.reset();
            logits.clear();
            // Row j of the wide logits scores the prediction of token base+j+1.
            engine.teacher_force_logits_wide(&toks[base], count, logits);
            const size_t vocab = logits.size() / count;
            if (vocab * count != logits.size() || !vocab) throw std::runtime_error("logits shape unexpected");
            for (uint32_t j = 0; j + 1 < count; j++) {
                const float* row = &logits[(size_t)j * vocab];
                float maxv = row[0];
                for (size_t v = 1; v < vocab; v++) maxv = std::max(maxv, row[v]);
                double sum = 0.0;
                for (size_t v = 0; v < vocab; v++) sum += std::exp(row[v] - maxv);
                const float target = row[toks[base + j + 1]];
                nll += -(std::log(std::exp(target - maxv) / sum));
                counted++;
            }
            fprintf(stderr, "ppl %u/%u nll=%.4f ppl=%.4f\n", base, (uint32_t)toks.size(),
                    nll / counted, std::exp(nll / counted));
        }
        std::ofstream out(out_path);
        out << "{\"tokens\":" << toks.size() << ",\"scored\":" << counted
            << ",\"nll\":" << nll / counted << ",\"ppl\":" << std::exp(nll / counted)
            << ",\"window\":" << window << "}\n";
        return 0;
    }
    throw std::runtime_error("mode golden|ppl");
} catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
