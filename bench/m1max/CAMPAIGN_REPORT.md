# q27 Metal on M1 Max 32GB — Full Campaign Report

Scope: tuning and verification of the q27 Metal engine (Qwen3.6-27B-MTP, q4s
tier) on a single Apple M1 Max 32GB Mac Studio, from the first DeltaNet
occupancy discovery (2026-09-23) to campaign close (2026-09-26).
Machine: Mac13,1, M1 Max 32GB, 400 GB/s theoretical,
`recommendedMaxWorkingSetSize` 26.0 GiB, macOS 26.6.2.
Repo state: 32 commits, engine diff +462 lines across 4 files
(`src/metal/q27_kernels.metal`, `metal_backend.mm`, `metal_engine.cpp/.h`);
~6.7K lines of bench harness and records under `bench/m1max/`.
All numbers below are measured on this machine unless stated otherwise.

---

## 1. Origin: the DeltaNet 448 occupancy discovery

The engine's GDN (gated delta net) kernels required 512 threads per
threadgroup. On this Apple7 GPU the measured occupancy ceiling was **448
threads**, so every generation attempt died at pipeline creation. Root cause:
register pressure from the kernel's `saved[32]` frame, not a hard hardware
ceiling (threadgroup memory was only 3.5 KB, and >1024 threadgroups existed).

Fix: `q27_delta_step256` / `q27_delta_chunk256` — two physical tiles loop
over the original four virtual 32-row tiles, with op order and reduction
order mirrored to stay bit-exact. Dispatch became occupancy-adaptive (512
where allowed, 256 otherwise). Same pass introduced device-aware GQA
defaults (`gqa_threshold=1280`, `gqa_block=256` on the Apple7-not-Apple8
family), fixing +16–33% tg on long context. Committed as `c12bfc6`.

**Transferable lesson:** threadgroup-size assumptions are generational. A
kernel that works on Apple8+ can be dead-on-arrival on Apple7; probe
`maxTotalThreadsPerThreadgroup` and ship fallbacks.

## 2. Baseline and profiling infrastructure

Before touching kernels, three tools were built (all still in-tree):

- `tools/bench_metal.cpp` — pp/tg split bench with medians, memory and
  working-set logging.
- `tools/golden_metal.cpp` — greedy golden corpus with top-16 logit
  fingerprints and PPL; the correctness gate for every later change.
- `Q27_METAL_PROFILE=1` — per-kernel GPU-time table inside the engine
  (Instruments' Metal System Trace was less useful: per-kernel breakdown
  was not obtained).

Baseline decode profile: `q27_matvec_q4_quantized` = 65.5% of decode GPU
time at ~50% bandwidth efficiency; `q27_attention_f16` ran at ~3% of
bandwidth at 1024 tokens. These two became the first targets.

**Gate rule calibrated before any kernel work:** a change must keep golden
digests identical, or stay inside calibrated top-k logit margins, plus PPL
parity. This rule governed every promotion below.

## 3. Decode GEMV: the magic-number f16 dot (+26% tg)

Roofline work (Step A/B) decomposed a 356→191 GB/s drop and identified
the dequant-dot path as the limiter. Rewriting the dot product to use
f16 magic-number bit tricks (`q27_matvec_q4_quantized_h`) measured
+38–40% kernel time and **+26% tg, bit-identical output**. Promoted to
default behind an env gate; rollback is a single env var.

## 4. Prefill GEMM: promoting the dormant half-staging kernel (+47% pp)

A pre-existing but non-default kernel (`mm_h`, barrier-staged fp16 tiles +
tensor cores) was gated on this machine and promoted by flipping one
default (`gemm_half_q4 = true`). **pp +47%.** The kernel itself was
already in the tree — the win was a default-flag change validated by
measurement.

## 5. Two measurement traps (recorded because they recurred)

- **Profiler artifact ④:** a "36 ms/token GPU idle" reading turned out to
  be profiler overhead, not engine behavior. Profile-of-profile lies.
- **GPU contention pollution:** running a bench while another harness held
  the GPU produced a fake "big loss" (turbo3's first pp number) and fake
  OOMs. Rule adopted: one GPU workload at a time, serial chains only.
- **Stale-binary trap (bitten twice):** an ad-hoc-built binary "verified"
  a change it did not contain. Fixed structurally: every campaign tool now
  has a Makefile target listing the full engine source set (including
  `.metal` sources) as prerequisites. Ad-hoc one-liner builds are banned
  in the reproduce docs.

## 6. Speculation: corpus-bound acceptance (Phase 2A)

Suffix-draft speculation swept on a synthetic repeated-paragraph corpus
showed +120–140% tg. The same configs on four real-workload corpora
(source code, git diffs, English prose, Japanese instructions) measured
**~0 acceptance, ±1% tg** (one config, mm4, actively degraded).
Conclusion: speculation benchmarks are corpus-bound; never promote from
synthetic-corpus numbers.

## 7. 64K context on 32GB: capacity is fine, compute is the wall (Phase D)

- Measured KV footprint: fp16 68.0 KB/token, turbo3 13.3 KB/token.
  64K fp16 fits: peak RSS == reservation, zero swaps.
- Prefix-snapshot resume saves ~30% of re-ingest.
- Cold 64K prefill: ~83 minutes at 14.3 t/s. The wall is compute, not
  memory.

## 8. Decode attention: w2row → w4row (+49% cumulative at 64K)

Decode attention at 64K consumed ~66% of the step and ran at ~0.9% of
DRAM roof. Two chain-shortening variants:

- **w2row** (2 rows/tile): kernel −34%; engine tg **+28% @64K, +10%
  @7168**. Promoted to default (`Q27_METAL_ATT=row` rolls back).
  Notably, w2row had earlier been dismissed at 2048 as "shadowed"
  (+0.3% tg) — the dismissal was an artifact of measuring where
  attention share was only ~10%.
- **w4row** (4 rows/tile): kernel −23.5% vs w2 at every depth; engine tg
  **+15.8% @64K** (6.90/6.89 reproducible), +4.8% @7168, golden
  digest-identical. Promoted to default; fallback ladder w4→w2→row all
  verified same-binary.

Cumulative decode at 64K: **4.64 → 6.90 t/s (+49%)** vs the row route.

An alignment probe closed a side hypothesis cheaply: padding turbo3's 50B
KV chunks to 64B cost +1.3% stream time for +28% bytes — the odd stride
was never the problem.

## 9. Serving quality: turbo3 passes the long-context gate

NIAHF (needle-in-haystack-fusion) at 64K with three needles and a fixed
greedy query: **turbo3 == fp16, 9/9 identical** (same hits, same misses;
the two misses were format artifacts present in both modes). Independently
consistent with upstream's 355K-token needle results on other hardware.
Serving recommendation: `--kv turbo3` with w4 attention — 4x smaller KV,
no measurable quality loss.

## 10. fp16 path: measured, not assumed (#3)

The 4-row tiling was ported to the fp16 KV path
(`q27_attention_f16_gqa_w4`, opt-in) to make "turbo3 wins" a tested
conclusion:

| tg t/s | fp16 row | fp16 w4 | turbo3 w4 |
|---|---|---|---|
| @7168 | 11.52 | 13.35 | 13.45 |
| @64K | 4.51 | 6.59 | 6.90 |

fp16 w4 is −47% vs fp16 row-route and nearly closes the gap to turbo3.
Why: w4 removes the dequant latency that dominated turbo3, and fp16 then
runs near its bandwidth roof (~110 GB/s) while turbo3 remains
dequant-latency-bound (~12 GB/s). **The quantization bandwidth advantage
is largely cancelled once streaming stops being the bottleneck; turbo3's
remaining case is the 4x KV memory saving, not speed.** fp16 default
unchanged; golden fp16 row-vs-w4 digest-identical 64/64.

## 11. MTP at long context: the hypothesis inverted (Step 2)

Hypothesis: MTP chunked-verify bundles KV reads and might favor fp16
regardless of acceptance. Code-level check first: with `match < min_match`
the verify batch never dispatches — the round falls back to a serial step.
"Independent of acceptance" is structurally false.

Measured 2x2 (all with w4 attention; MTP = suffix w4/mm12):

| tg t/s | MTP off | MTP on | delta |
|---|---|---|---|
| turbo3 @64K | 6.94 | 5.84 | **−15.8%** |
| fp16 @64K | 6.61 | 5.52 | **−16.5%** |
| turbo3 @7168 | 13.53 | 12.45 | −8.0% |
| fp16 @7168 | 13.35 | 13.07 | −2.1% |

Forced-fallback check (min-match 999, verified burst=0/fallback=756):
drafter overhead alone is −0.7..−0.8% tg in both KV modes.

**The law that came out of it: wasted-lane cost ≈ lanes × KV-depth.** The
penalty tracks attention share: ±1% @2048 → −2..−8% @7168 → −16% @64K.
Long context amplifies speculation waste. Combined with §6, the
long-context rule for real-workload agents is: speculative decode
defaults-off is strongly correct. Do not enable MTP on general-purpose
priors.

## 12. Prefill stage inventory: no cheap lever remains (#4)

A per-stage profiler (engine `pp_profile_chunk` + `tools/pp_share.cpp`,
commit-overhead subtracted) attributed 7168 prefill:

| Stage | ms/tok | Share | Wall class |
|---|---|---|---|
| FFN | 9.67 | ~43% | dequant-ALU-bound (~18x off weight-stream roof) |
| attention | 7.3 | ~33% | row-serial SIMT structure |
| GDN | 3.93 | ~18% | sequential recurrence |
| norm+add | 1.1 | ~5% | nothing |
| embedding | ~0 | — | — |

Because FFN is dequant-ALU-bound (dequant work is fixed per weight byte,
independent of M), raising `PREFILL_CHUNK_MAX` cannot help — negated
structurally before measuring. All three share leaders are already
classified rewrite-class walls.

## 13. Final state

Shipped defaults on this machine: turbo3 KV + w4 decode attention +
`mm_h` prefill GEMV/GEMM + MTP off. Rollback ladder (env-only):
`Q27_METAL_ATT=row|w2`, `Q27_METAL_GEMM_HALF_Q4=0`, `--kv fp16`.

Cumulative vs campaign-start baseline: **64K decode +49%, 7168 decode
+15.5%, short-context pp +47%, decode GEMV +26%.**

Proven do-not-touch: MTP-on at long context (−16%), 50B→64B KV padding
(no effect), fp16 as serving default (loses with 4x memory cost).

Remaining walls, all rewrite-class and explicitly out of scope as separate
projects: prefill FFN dequant rewrite, prefill attention Flash-style
rewrite, GDN recurrence restructuring.

## 14. Method notes worth porting to other projects

1. **Build the correctness gate before the first kernel edit** (golden
   digests + margins + PPL). Every promotion in this campaign was decided
   by it; nothing shipped on vibes.
2. **Roofline before rewrite.** Every "slow" claim was converted to "X%
   of a named roof", which made rewrite-vs-tune decisions mechanical.
3. **Re-measure at the depth where the share dominates.** The single most
   expensive mistake avoided: w2row was wrongly dismissed at 2048 and
   turned out to be the biggest win at 64K.
4. **Real corpora only** for anything acceptance-dependent.
5. **Serial GPU access.** Parallel harness runs produced fake losses and
   fake OOMs.
6. **Structural guards over memory.** Stale-binary and silent-skip traps
   were each hit twice before being closed in the build system, not in
   notes.
7. **Negative results are deliverables.** Seven negative results and two
   rewrite-class closures are recorded in-tree with evidence, which is
   what makes the "stop here" decision defensible.

---

Records: `bench/m1max/STATUS.md` (FINAL STATE section is the restart
point), `bench/m1max/UPSTREAM_DRAFT.md` (six-section upstream sharing
draft), raw data in `bench/m1max/*.jsonl` and `bench/m1max/step2/`.
