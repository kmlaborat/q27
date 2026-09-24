# q27 Metal — M1 Max 32GB Tuning: Session Status & Resume Notes

Date: 2026-09-23 (JST) · Machine: Mac Studio Mac13,1, Apple M1 Max, 32GB unified
memory, 400 GB/s theoretical bandwidth, `recommendedMaxWorkingSetSize` = 26.0 GiB,
GPU max clock 1296 MHz, macOS 26.6.2 (25G83), Xcode installed at
/Applications/Xcode.app (Metal Toolchain component downloaded 2026-09-23).

Mission: tune the q27 Metal backend for this specific machine, working from the
user's instruction doc (phases: DeltaNet fix → baseline → profiling → correctness
harness → kernel work). Model under test: Qwen3.6-27B-MTP **q4s** tier
(`models/qwen36-27b-mtp-q4s.q27`, md5 `7e5454e0c0ded717136ad3e42634ba25`, 15.46 GB,
tokenizer `models/qwen36-27b-mtp.tok` md5 `bb95b3ca7647ce1cc061c141789e7102`).

## TL;DR — where things stand

1. **DeltaNet occupancy bug: FIXED and verified.** The 512-thread gate in
   `metal_backend.mm` killed all generation on Apple7 GPUs (measured occupancy
   448). New 256-thread kernels (`q27_delta_step256`, `q27_delta_chunk256`) with
   occupancy-adaptive dispatch. All Metal tests pass on this machine through the
   256 fallback. Cross-verification of 256 vs 512 numerics needs an Apple8+ machine.
2. **Decode-route config: FIXED to device-aware defaults, +16–33% tg on long
   context.** `gqa_threshold=1280 / gqa_block=256` on Apple7-not-Apple8 family
   (env vars still override). Verified golden-corpus invariance (32/32 identical)
   and PPL parity.
3. **Correctness harness live**: `tools/golden_metal.cpp` (32-prompt greedy
   golden + top-16 logit fingerprints + PPL). Deterministic across runs.
4. **Main targets identified, not yet touched**: `q27_matvec_q4_quantized`
   (65.5% of decode GPU time, ~50% bandwidth efficiency) and
   `q27_attention_f16` (runs at ~3% of bandwidth at 1024 tokens). Kernel
   modification of either was explicitly gated on the correctness harness —
   which is now done.
5. Committed as `4aefdfd` (2026-09-23, repo-local identity `user <user@example.invalid>`;
   amend `--author` if a different identity is wanted). `models/` added to
   `.gitignore` (14 GB weights stay out of git).

## Reproduce the environment

```bash
cd /Users/user/projects/q27
make build/q27-metal build/q27-metal-server build/test-metal-ops build/test-metal-backend
c++ -O2 -std=c++17 -Wall -Wextra -Werror -fobjc-arc -pthread -I src/metal \
  tools/bench_metal.cpp src/metal/metal_engine.cpp src/metal/metal_backend.mm \
  src/loader.cpp src/tokenizer.cpp -framework Foundation -framework Metal -o build/bench_metal
# same one-liner builds tools/golden_metal.cpp -> build/golden_metal
```

Key engine env vars: `Q27_METAL_DIAG=1` (pipeline occupancy log),
`Q27_METAL_PROFILE=1` (per-kernel GPU-time table — the primary profiling tool;
Instruments' Metal System Trace works but its Shader Timeline is off by default
and per-kernel breakdown was not obtained), `Q27_METAL_GQA_THRESHOLD`,
`Q27_METAL_GQA_BLOCK`.

## Code changes made (uncommitted, `git diff` = 182 insertions)

### `src/metal/q27_kernels.metal`
- Added `q27_delta_step256` and `q27_delta_chunk256`: two physical tiles loop
  over the original kernel's four virtual 32-row tiles; per-virtual-tile op
  order and `part[0..3]` reduction order mirror the 512-thread originals.
  `src*decay` recomputed in the update pass (bit-exact same product).

### `src/metal/metal_backend.mm`
- `make_pipeline`: `Q27_METAL_DIAG` occupancy logging + hypothesis comment
  (measured 448 on M1 Max; register pressure from the `saved[32]` frame is the
  cause; threadgroup memory 3.5 KB rules out limit (a); 1024+ threadgroups
  exist so not a hard ceiling (c)).
- Pipeline members `delta256`, `delta_chunked256` added and wired; both delta
  dispatch sites pick 512 where occupancy allows, 256 otherwise. The old
  hard `maxTotalThreadsPerThreadgroup < 512` throw is removed (shape check kept).
- Device-aware GQA defaults (before env overrides, env still wins):
  ```objc
  if ([impl_->device supportsFamily:MTLGPUFamilyApple7] &&
      ![impl_->device supportsFamily:MTLGPUFamilyApple8]) {
      impl_->gqa_threshold = 1280;
      if (!getenv("Q27_METAL_GQA_BLOCK") || !*getenv("Q27_METAL_GQA_BLOCK"))
          impl_->gqa_block = 256;
  }
  ```

## Numbers

### Baselines (q4s, fp16 KV, ctx 8192, gen 128, reps 3 medians, `bench_metal`)
Old code default route (threshold 2048 / block 1024), two independent runs,
agreement ~1%: `baseline_chunk_fp16.jsonl`, `baseline_chunk_fp16_run2.jsonl`.

| seq | pp t/s | tg t/s (old default) | tg t/s (new family default) |
|---|--:|--:|--:|
| 128 | 44.4 | 12.13 | 12.1 (t2) |
| 512 | 45.3 | 11.5 | 11.35 (t2) |
| 2048 | 43.4 | 8.0 | **10.64** (+33%) |
| 4096 | 38.7 | 8.0 | **10.6** (b256) |
| 7168 | 32.6 | 8.0 | **9.25** (+16%) |

Model load ~8 s; process `internal` 0.15–0.20 GiB (weights are file-backed
mmap, 14.4 GiB); no memory pressure (swap ~1 GB).

### Crossover grid (`grid_m1.jsonl` / `grid_m1_runner.log`, 20 trials, all OK, per-trial timeout script `tools/grid_m1.sh`)

| seq | t2 | b256 | b1024 | b512 |
|---|--:|--:|--:|--:|
| 2048 | 10.00 | **10.58** | 7.92 | 9.59 |
| 3072 | 9.14 | **10.52** | 7.91 | 9.70 |
| 4096 | 8.39 | **10.48** | 7.90 | 9.54 |
| 6144 | 7.18 | **9.45** | 7.92 | 9.30 |
| 7168 | 6.73 | **9.30** | 7.90 | 9.34 |

Fine mesh: t2@512=11.44, t2@1024=11.13, t2@1536=10.37 vs b256@512=10.96,
b256@1024=10.94, b256@1536=10.93 → crossover ≈ 1280, block 256 wins above.

### Decode kernel profile (Q27_METAL_PROFILE, seq 512, gen 600)
```
q27_matvec_q4_quantized  65.5%  194 µs/call  (seq-independent: pure weight streaming)
q27_matmul_q4_mm         16.4%  (spec/MTP verify path)
q27_attention_f16         7.0%  (see slope below)
q27_delta_step (GDN)      1.7%  (256 fallback performs fine)
```
`matvec_q4` costs ~2× the weight-bandwidth floor (14.4 GB/token → 36 ms floor;
measured ~71–84 ms/token across runs) → ~50% achieved bandwidth.

attention_f16 avg vs seq: 119 µs (32) / 247 (256) / 478 (512) / 685 (1024) →
~119 µs dispatch floor + ~0.57 µs/token slope; at 1024 the kernel reads KV at
roughly **3% of peak bandwidth**. The long-context tg decline is mostly this
slope. Biggest target after matvec.

### powermetrics (user-run `sudo powermetrics --samplers gpu_power,cpu_power -i 1000`,
log at `bench/m1max/powermetrics_log.txt`, window 19:18–19:30 overlapping run2)
GPU power avg 27.6 W / peak 32.9 W, residency 100% throughout, frequency pinned
1296 MHz. GPU never idle → not CPU-dispatch-bound, not power-capped; stalls are
on-device memory stalls. Note: `--samplers memory` does not exist on ASi
(Intel-only sampler).

## Correctness harness (phase 3, complete)

- `tools/golden_metal.cpp golden`: 32 fixed inline prompts (code/prose/math/
  Japanese/JSON/repetition/long-range), greedy 96 tokens via bare `step()`
  (no MTP), top-16 logit fingerprint of first decode position, FNV-1a digest
  over the id stream. **Deterministic: two runs identical.**
  - Old-default build, reps=2 stream digest: `c30b2e89d450d563`
  - b256 forced (th=1 blk=256), reps=1 digest: `38dbb265ba2bf2d6` — b256 is
    NOT bit-identical to the t2 route: 9/32 prompts diverge (first-token flip
    1/32, top-k logit deltas ≤ 0.16). The "bit-identical" comment near
    `metal_backend.mm:2281` is conditional; treat such claims as golden-testable.
  - Family-default build (new): 32/32 prompts identical to old default (all
    golden prompts < 1280 tokens stay on t2).
  - GOTCHA: digest covers all reps — always compare at equal `--reps`, or diff
    per-prompt (this caused one false alarm on 09-23).
- `tools/golden_metal.cpp ppl`: teacher-forced NLL, `--window 96` is the engine
  cap on `teacher_force_logits_wide`. Recorded value **14.4023** (wikitext-2
  test first 16384 tokens, corpus `bench/m1max/wikitext2_test.txt` pulled from
  HF datasets-server Salesforce/wikitext wikitext-2-raw-v1 test; the Salesforce
  S3 zip is dead — PermanentRedirect/403). b256 route: **14.4014** (parity).
  Absolute value is not comparable to README ~7.9 (those use longer context;
  w96 context truncation inflates PPL). It is a self-consistent regression metric.
- Existing gates still pass: `test-metal-ops` (incl. delta_step vs CPU 2e-3,
  chunk↔step recurrence 1e-5), `test-metal-backend`, `test_metal_stream`.
  `tools/metal_canonical_gate.sh` has no canonical for metal-m1 (only metal-m4);
  deriving one is a documented next candidate (CANON_ARCH override exists).

## Golden gate rule (agreed 2026-09-23, calibrated on real data)

Question from user: is the b256-vs-t2 divergence (9/32) harmless rounding or
systematic argmax change? Study: `--step-margins` added to golden_metal
(per-step top1/top2 dump); `margins_default.jsonl` vs `margins_b256.jsonl`.

Findings on identical contexts (23 non-branching prompts, 2208 steps):
- |t1 logit drift| between routes: median 0.035, p99 0.44, **max 2.15** — so
  the noise is NOT LSB-level; it is fp accumulation-order noise amplified
  through layers into ~5-8% of confident logit magnitudes. But at confident
  steps (margin 7-10) it never flips anything.
- All 9 sequence branches were rooted at steps whose min(top1-top2 margin)
  over both routes was **0.003–0.079** — deep tie band. No branch was ever
  rooted at a confident margin. Verdict: tie-band flips + cascade; quality
  parity confirmed by PPL (Δ0.0009).

Gate for reordering-class changes (anything matvec does): `tools/golden_check.py
BASELINE.jsonl CANDIDATE.jsonl [--ppl-a --ppl-b]` — PASS iff every prompt's
FIRST divergence (root) has min-margin <= tie_band (default 0.5, ~2x the
observed noise p99; observed roots <= 0.08), and |Δppl| <= 0.02. Post-root
steps are cascade and not evidence. Same-build determinism stays strict by
digest. A root at margin > band = treat as bug, stop.
Sanity: the checker passes default-vs-b256 at 0.5 and fails at 0.001.

## Lessons / false alarms on record

1. **"blk=512 hang" was my tooling artifact**: several trials batched inside one
   bash call, its cumulative timeout killed the running 512 trial → looked like
   a kernel hang. Two later reproductions (grid + exact original env) passed
   normally. Rule: one process per trial, per-trial timeout — implemented in
   `tools/grid_m1.sh`.
2. Digest-vs-reps trap (above).
3. macOS has no `timeout`; use background+kill loops (`gtimeout` absent).
4. `brew install --cask xcode` is gone; Xcode now via App Store / `xcodes`.
   After install, Metal Toolchain is a separate component:
   `xcodebuild -downloadComponent MetalToolchain` (~840 MB, fast).
5. xctrace works headless without sudo; per-kernel shader breakdown needs the
   Shader Timeline knob (template default off). Engine's own
   `Q27_METAL_PROFILE` is the practical tool.
6. `anchoredit_apply` tool failed with bare exit 1 on this machine; built-in
   edit works.

## Step A result — matvec roofline on M1 Max (2026-09-23, `bench/m1max/roofline_m1.jsonl`)

Harness `tools/roofline_m1.mm` (+ `tools/bench_stream.metal`): loads the
engine's own kernel source with the same compile flags, dispatches the
production kernel, the retained r2 arm, and a bench-only pure-uint4-read
kernel at the identical dispatch geometry, over the q4s file's real tensor
shapes (read live from the .q27 header), best-of-3 timed reps.

| regime | stream roofline | production q4 | r2 arm |
|---|--:|--:|--:|
| DRAM-bound (636 MB, 248320x5120) | **355.8 GB/s (89% of 400 peak)** | 191.3 | 154.2 |
| L2-resident (26–45 MB shapes) | ~560–590 GB/s | 172–183 | 141–150 |

Findings:
- The machine can stream 356 GB/s with pure reads in production's dispatch
  geometry — the DRAM roofline is 89% of theoretical. Headroom is real.
- Production q4 plateaus at 173–191 GB/s **regardless of whether its data
  fits L2** — the cap is inside the kernel (register pressure at maxTotal=448,
  spills and/or too few loads in flight), not the memory system.
- **r2 is worse than production on M1 Max (143–154 vs 173–191)** — the
  M4-era arm ranking does not transfer (consistent with the missing round
  doc; recorded as upstream-report evidence).
- Small shapes (3 MB) are ramp-bound (prod 86 GB/s at 30 µs); irrelevant to
  the aggregate.
- Projection if prod reaches ~80% of roofline (285 GB/s): matvec (65.5% of
  decode) x0.65 time -> tg 11.5 -> ~15 (+30%); full roofline would be ~+45%.

## Step B0 result — dot decomposition of the 356->191 GB/s drop (2026-09-23,
`bench/m1max/roofline_m1_b0.jsonl` + `_b0b.jsonl`)

Bench-only arms, production loop structure and dispatch, each adding one
resource class to the pure stream (all shapes; DRAM-bound 248320x5120 and
L2-resident 17408x5120 columns shown):

| arm (cumulative) | DRAM GB/s | L2 GB/s | step cost |
|---|--:|--:|--|
| bench_stream2 (weights only) | 359 | 615 | — |
| + x loads (prod indices) | 359 | 559 | 0% / -9% |
| + scales (read + fp multiply) | 337 | 502 | -6% / -10% |
| + HALF the nibble dots (`bench_sc_dot2`) | **332** | 321 | -1.5% / -36% |
| + full dots = production (`bench_sc_dot4` == prod to noise) | **191** | 182 | **-42% / -43%** |

Findings:
- x redundancy is NOT the problem (+0–8%; invisible at DRAM shapes).
- scale streams cost ~5–10%.
- **the nibble-dot shift/mask chain is the wall**: it alone takes 337 -> 191
  (DRAM) and the dot2 control arm shows a kernel with only HALF the dot
  arithmetic sits at 332 — essentially back to the roofline. The kernel is
  ALU/dependency-bound inside the dot, not bandwidth-bound.
- `bench_sc_dot4` reproduces production to noise (191.2 vs 191.1) — the
  decomposition is self-consistent.

Implication (see report): a cheaper EXACT nibble dot (fp16 magic-number
unpack: half(0x6400|n)-1032 == n-8 exactly; products exact in half, sum
exact in fp32 up to 32512) could target the dot2 arm's 332 GB/s while
keeping bit-identical math. B1 (x broadcast) alone: ~+5%, below the user's
+25% solo threshold; fold into B2 only if B2 needs the load slots.

## Step B2a/B2c — bit-identical fp16-dot rewrite: +38-40% (2026-09-23,
`bench/m1max/roofline_m1_b2a/b2b/b2c.jsonl`)

New bench arms in `tools/bench_stream.metal`, all compared BITWISE against
the production kernel on random inputs (random nibbles, nontrivial scales,
random signed x) inside `tools/roofline_m1.mm`:

- `bench_sc_constdot` (full dot4 math on CONSTANT words): 338 GB/s — the
  dot math is FREE when it doesn't consume loads. The wall is the
  load->shift-chain dependency, not instruction throughput.
- `bench_sc_halfdot4` (fp16 magic-number exact dot, `0x6400|n = 1024+n,
  -1032` bias fold): **264 GB/s DRAM shape, +38-40% over production,
  mismatches=0 on every shape** — bit-identical confirmed in-bench before
  any timing conclusion, per the user's standing rule.
- `bench_sc_halfdot4p` (manual double-buffer pipeline): WORSE (occupancy
  640->448) — negative result.
- `bench_sc_halfdot4u2` (independent 2-chunk unroll): no gain over half4
  (within noise) — negative result.

Status: half4 at 264 vs constdot ceiling 338: the short-chain extraction
recovers about half the dependency penalty; two standard ILP fixes came up
empty, further gains would need a deeper reformulation (direct-float-mantissa
extraction or tensor-core routing) with uncertain payoff. Decision point for
the user: integrate halfdot4 as-is at +38% (bit-identical gate), or keep
bench-mining toward ~338.

## Step B2 INTEGRATED — fp16-dot matvec promoted to default (2026-09-24)

Engine: `q27_matvec_q4_quantized_h` in q27_kernels.metal; backend selects it
by default (`Q27_METAL_Q4_ARM=r0` restores the previous int8-dot kernel).
Gate results (fresh binaries — note the earlier "verification" ran on a stale
binary without the kernel and was redone; lesson: the metal source is
embedded at compile time, rebuild golden/bench/ops after kernel edits):
- Golden digests bit-identical to pre-change baselines: 64 prompts x 2 reps,
  0 mismatches (both baseline pairs + arm-vs-arm invariance).
- test-metal-ops green both arms.
- tg @ seq512: 11.67 -> 15.04-15.06 (+29.0%); seq4096: 10.48 -> 13.59 (+30%).
- PPL: unchanged by construction (bit-identical logits).
Bench-side ceiling analysis (constdot 338) unchanged; deeper reformulation
(mantissa-direct / simdgroup_matrix) left as the low-priority B tail.

## Attention decode diagnosis (2026-09-24, `bench/m1max/attn_roof_*.jsonl`)

Harness `tools/attn_roof.mm` + `tools/bench_attn.metal` (bench-only; engine
kernels timed as-compiled). IMPORTANT: decode's real path is turbo3
(8-bit KV, 400B/token/layer), NOT attention_f16 — f16 only when KV=fp16
mode is forced. Real model: 24 q-heads / 4 kv / head_dim 256, 17 attn
layers of 65. tg unit is tokens/s (higher=better; reconciles with matvec
roofline: 1000/13.5 = 74ms/step ~= 14.2GB/264GB/s matvec + ~12ms attention
+ GDN/misc).

Findings (per dispatch, best-of-5, x8-amortized):
- exp/SFU cost: ZERO (noexp == prod). Online-softmax max/l chain alone:
  ZERO (nosm == prod). Dispatch floor (empty, same grids): ~30us.
- The wall is staging+dot structure: same-bytes stream arm is ~114us vs
  prod 660-730us (b256). nostage (device-direct, 6x redundant bytes) is
  WORSE -> keep smem staging, fix the per-row ILP instead.
- attn_w2row (two KV rows in flight per iteration; pair-rescaled online
  softmax, accumulation order changes -> margin-gate class) = -30..-33%
  at b256 across all seq. Candidate for a bench->engine integration.
- Grid levers on the HARNESS mislead: b128 looked -42% there but on
  device tg it is WORSE (-6.3% at 2048); b1024 loses badly as grid says.
  Device A/B (tg, seq2048, gen64): plain-only 12.48 | b256 13.81 | b1024
  9.53 (t/s) -> shipped b256 default confirmed best; grid_m1 stands.
Attention share at seq4096 ~ 12ms/74ms step; w2row upper bound ~ +5% tg
at long seq. Priority: medium (matvec done; this is the next real slice).

## w2row engine integration: NEGATIVE at tg level (2026-09-24)

`q27_attention_turbo3_gqa_w2` (Q27_METAL_ATT=w2, default off): 2-row/tile
pair-rescale online softmax. Kernel-level -33..-35% vs turbo3_gqa at every
seq (bench numcheck: partials agree to ~1e-6). Gates: ops green both arms;
golden 64 prompts x {base,w2} x 2 runs with --step-margins: 0 branches,
max-root-margin 0.0000 -> PASS. BUT device tg A/B (gen64, 2 runs best-of):
seq512 +0.3%, 2048 +0.3%, 4096 -0.0%, 7168 +0.2% — NO tg gain anywhere.
Interpretation: decode attention dispatches are absorbed alongside the
DRAM-bound matvec streams (GPU work saturates on matvec; attention overlaps
or queues invisibly), so attention kernel wins don't surface in tg while
matvec rules. Kernel stays opt-in; attention micro-opt is CLOSED as a tg
lever. Corollary: prior "attention_f16 0.57us/token slope = next target"
read is superseded — the target must still be bandwidth-side (matmul_q4
prefill study, KV compression) not latency-side kernels.

## Gate runs: dormant q4_mm_h (prefill half-staging GEMM) — 2026-09-24

All numeric gates PASS for Q27_METAL_GEMM_HALF_Q4=1 (q27_matmul_q4_mm_h):
- PPL w96/16K: 14.4023 -> 14.4088, |delta|=0.0065 <= 0.02 PASS
  (ppl_h4_default.jsonl / ppl_h4_halfq4.jsonl).
- golden --step-margins x {2 baselines x 2 arm runs}: one branch (p29 step51)
  root margin 0.0548 <= 0.5 PASS (golden_m1_h4_margins_*.jsonl).
- test-metal-ops green both arms.
Performance, quiet alternating best-of-3 (pp-only runs, idle machine):
pp 42.56 -> 62.5 t/s = 1.47x (stdev ~0.05); kernel-level (Q27_METAL_PROFILE,
same 8096 calls): 5319us -> 3386us avg = 1.57x. tg unchanged (decode uses
matvec). NOTE: the code gate says "ship pending valid >=1.7x quiet run" and
cites QUIET_BENCH_EVIDENCE.md, which is NOT in the repo (origin unclear,
likely another machine). M1 Max quiet numbers land at 1.47x pp / 1.57x
kernel: big win but BELOW the recorded 1.7x ship-line -> promotion was a
user decision. DECISION (user, 2026-09-24): PROMOTED to default. The legacy
">=1.7x quiet" line is NOT adopted as this repo's bar: its evidence file is
missing and it was likely a kernel-basis number measured on other hardware.
Precedent recorded for future ship-lines that arrive without reproducible
evidence: re-derive the gate on THIS machine from the project's own battery
(PPL + golden step-margins + ops + quiet alternating A/B), and document the
substitution here rather than silently inheriting or rejecting a number.
Caveat kept on file: 1.7x may once have encoded intent beyond speed
(compile time, margin headroom, maintenance); if such constraints ever
matter, revisit under this entry. Rollback: Q27_METAL_GEMM_HALF_Q4=0.

## SSM/GDN shadow (2026-09-24): decode profile diff gen96-gen8 at seq2048:
all GDN/SSM kernels sum to 3.0ms of 113ms wall = 2.7% <= noise floor;
closed as tg lever without noop runs. Full decode shares recorded in
STATUS history; GPU busy 76.5 vs wall 112.9 => ~36ms/token idle pool
(possible next campaign; explains w2row absorption).

## ④ CLOSED: the "36ms/token GPU idle" was a profiler artifact (2026-09-24)

Recompute with clean numbers: unprofiled tg at seq2048 = 74.5ms/token; the
Q27_METAL_PROFILE run measured 112.9ms/token tg. Difference 38.4ms over
1252 sampled encoders/token = ~31ns/encoder of timestamp-sampling overhead
— exactly the pool I had filed as "idle". There is NO idle pool: greedy
decode is GPU-saturated (busy ~= wall). All prior shares from profile logs
are directionally fine but their wall/busy GAP must not be interpreted.
Also corrected: bench_metal default --mode greedy is a plain serial step
loop (the printed width/min_match fields are display-only); suffix/MTP
paths are separate modes. Q27_SUFFIX_TRACE=1 now emits per-round
verify/read/commit phase walls (output-only instrumentation, this commit).
Consequence for the attention puzzle (route swap moves tg 9% but the w2
kernel -33% moves 0.3%): dispatch STRUCTURE (route, merge count, encoder
serialization), not kernel microseconds, is what surfaces in wall time.
If attention is chased again, chase structure. Remaining decode levers
after all closures: the matvec busy stream itself (53.5ms, 72%): the old
B-tail (264->338 GB/s ceiling, ~+10% tg upper bound, uncertain) is the
only sizeable unexploited item on record.

## Phase 2B Step A: prefill GEMM roofline decomposition (2026-09-24, `phase2b_stepA.jsonl`)

Prefill shares (profiled, +2.3% profiler inflation confirmed vs clean pp;
phase2b_ppcurve.jsonl clean curve 68.6/62.5/55.1/47.3 t/s @512/2048/4096/7168):
matmul_q4_mm_h 62-82% (per-call flat ~3.4ms), causal_gqa_t2 grows
8.5%->32.5% from seq2048->7168 (O(n^2), D-phase early signal), delta_chunk ~2%.

Step A arms (tools/pf_roof.mm + tools/bench_pf.metal, real ffn shapes,
production grid rows/32 x ceil(96/16), 128 threads; GB/s = unique weight
bytes/us; devGB/s = x6 device traffic from the y-group re-read pattern):
  stream   58 uniq / 348 dev  -> memory delivery IS at the DRAM roof (~98%)
  +LUT     33 uniq            -> dequant staging alone costs +74% vs stream
  prod mm_h 9.1 uniq (55 dev) -> 3.7x SLOWER than even the LUT arm
  mma_peak ~12x faster than prod -> tensor cores are NOT the constraint
Waterfall: 768us stream -> 1337 +LUT -> 4903 prod. The bulk (3.7x) is NOT
memory and NOT MMA throughput: it is the serialize pattern (per-64-col
barrier staging<->MMA, scale-flush every step, small x-tile forcing 6x
re-reads that don't bite until ALU costs drop). Headroom framing: reaching
the LUT arm's 33 uniq = ~3.6x GEMM time = ~2x pp (GEMM is 62-82%); past
that needs killing the 6x re-reads (wide x-tile) to approach 58 uniq.

## Phase D Step 0 DONE: 64K on 32GB — fits, no swap, flat footprint, compute is the wall

Step 0.2 (overrun behavior): engine rejects cleanly BEFORE allocation —
"requested KV cache ... exceeds the configured cache budget; use --kv
turbo3, raise --budget-mb, or reduce --ctx" at ctx 229440+/262144. No crash,
no swap death: budget check is the governor (cache_budget =
recommendedMaxWorkingSetSize/2 = 13.0 GiB).

Step 0.3 (bench_metal --ctx 65536 fp16 --seq 16320,32760,65400 --gen 16,
`bench/m1max/d_step0_rss.jsonl`; synthetic-repeated paragraph — fine for
memory/scaling, not for absolute real-text pp):

| ctx | pp t/s | tg t/s | peak RSS (process) | swaps |
|--:|--:|--:|--:|--:|
| 16320 | 33.6 | 9.58 | 4.8 GiB (flat) | 0 |
| 32760 | 22.1 | 6.85 | 4.8 GiB (flat) | 0 |
| 65400 | **13.1** | **4.52** | 5.17 GiB | 0 |

- 64K works. Peak footprint 5.17 GiB == reservation 5.15 GiB + process: the
  reservation math is honest, no hidden transient blowup (chunked prefill
  stays inside the formula's envelope).
- vmmap stayed 4.8G across 16K->32K->64K re-ingests in ONE process
  (reset->ingest->decode x3): **compaction-style re-ingest does not leak** —
  KV release tracks. (Prefix-cache/server path untested; bench-level reset
  loop is the proxy.)
- pp curve collapses with context: 62.5 (short) -> 47.3 (7.2K) -> 33.6
  (16K) -> 22.1 (32K) -> 13.1 (64K): worse than the naive O(n^2) attention
  extrapolation; a 64K cold ingest = ~83 minutes. tg halves too
  (14.7 -> 4.52). Capacity NOT the constraint (2.5x headroom at fp16,
  6x+ at turbo3); **prefill attention compute is the wall**.
- bench_metal synthetic prompt cap raised (200 -> 4000 repeats) to reach 64K.

## Phase D Step 0.1: MEASURED KV footprint (`tools/kv_footprint.cpp` -> build/kv_footprint)

Engine-own math on this machine (constants N_LAYER=64, N_KV=4, HEAD_DIM=256;
17 attn layers incl MTP block; turbo3 cache_row = N_KV*2*50):

| mode | B/token (all layers, K+V) | 64K KV-only | 64K full reservation* |
|---|--:|--:|--:|
| fp16 (CLI default) | **68.0 KB** | 4.25 GiB | **5.15 GiB** |
| turbo3 (8-bit, `--kv turbo3`) | **13.3 KB** | 0.83 GiB | 1.73 GiB |

*reservation = serving_reservation_bytes(): KV + gqa_partial scratch + side
fp16 cells + fixed state. Engine cache_budget = recommendedMaxWorkingSetSize
/2 = 13.0 GiB — **64K fits with 2.5x headroom even at fp16** (weights 14.4
GiB + 5.15 + process ~0.2 ~= 19.8 GiB < 26 GiB recommendedMaxWorkingSetSize
< 32 GB). KV capacity is NOT the 64K constraint; compute (O(n^2) attention)
is. Chat arithmetic corrected: user's 34 KB/token was int8-basis (x1 byte);
fp16 is exactly 2x = 68 KB; earlier "~160KB ceiling" phrasing was wrong.
Remaining unknown for Step 0: transient prefill activations at 64K (chunked
prefill bounds them by CHUNK_MAX, not fully in the formula) -> measure RSS
empirically when running 64K (Step 0 item 3).

## Phase 2B Step B: the four cost hypotheses — ALL NEGATIVE (closed, `phase2b_stepB.jsonl`)

| arm | targets (independently) | ffn_up us vs prod 4833 | verdict |
|---|---|--:|---|
| pf_b1_wide | 6x weight re-read: one staged tile serves 6 token windows (bit-identical, harness-verified) | 7078 (-46%) | NEG — reuse is L2-served anyway; wider racc regs cost occupancy |
| pf_b2_dbuf | load->compute serialization: double-buffered staging (bit-identical) | 5728 (-19%) | NEG — smem doubling cut threadgroups/SM; latency was not exposed |
| pf_c4_flushless | flush cadence /4 (math wrong by design, attribution only) | 4584 (-5%) | flush scale-fold ~= 300us total |
| pf_c5_prescale | flush eliminated entirely (scales fp16-folded at staging; tensor acc across full K; margin-class math) | 4387 (-9%) | scale/flush machinery ~= 450us |

What survives the four falsifications: the per-step COUPLING of
threadgroup-staging writes with tensor-core smem operand reads. Remove
either side and the same shape runs at pf_lut 1335us or mma_peak 414us;
keep both coupled in ANY arrangement measured here and it sits 4.4-5.7ms.
Likely the Apple P-tile shared LSU/tensor fabric (32KB smem tiles per step
per threadgroup, banked operand reads) — a hardware trait, not scheduling.
Remaining fix families (both beyond current budget, PARKED with this
attribution): smem-free dot-style prefill GEMM (x kept L1/L2-hot, weights
streamed once; needs token-blocking with either 12x weight re-stream or
96-accumulator register games), or P-tile-shaped mma primitives.
Lesson for Apple-silicon GEMM: components can each measure ~free while
their coupling is the wall — probe pairwise, not unarily. pp +47% (mm_h)
stands as the shipped prefill win; prefill micro-optimization is NOT
recommended as the next session's first move.

## Phase 2A: speculation config sweep — CLOSED with no config change (2026-09-24)

Synthetic bench corpus (same paragraph x200 + counters) inflates suffix-draft
acceptance to the point of absurdity: w32 showed +140%/+120% tg there,
while on four REAL-workload corpora (bench/m1max/phase2a/corpora/: Metal
source, git diffs, engineering prose, Japanese instructions) acceptance is
~0 tokens/token and every config sits within +-1% (noise) — except mm4,
which degrades -7..-18% (low-quality bursts). Current default w4/mm12 is
NOT a stale M4 misfit: it is simply near-neutral on real text, like every
other config. Lesson: speculative-decode tg numbers are corpus-bound; the
synthetic corpus must never be the basis for spec-config promotion.
Gate event: one serial-vs-suffix token flip at seq512 idx25 (synthetic
corpus) proven near-tie: logit margin 0.0077 with spec picking exact top2
(tools/p2_margin.cpp). Existing root-margin<=0.5 gate covers the
chunked-verify-vs-serial kernel-shape class; range formally extended.
Tools added: bench_metal --prompt-file + stream_digest + rounds/accepted/
burst/fallback stats; p2_margin probe. All phase2a data committed.

## Session close state (2026-09-24) — everything above resolved or closed

Shipped (all gates passed, docs in this directory):
- decode tg +26% (lower bound): fp16 magic-number matvec dot default
  (rollback Q27_METAL_Q4_ARM=r0); bit-identical, golden 64x2 x3 runs.
- prefill pp +47%: q27_matmul_q4_mm_h default (rollback
  Q27_METAL_GEMM_HALF_Q4=0); gate-substitution precedent recorded above.
- GQA defaults family-aware (b256/threshold 1280 on Apple7), grid_m1 data.
Negative results on record (do not retry without new evidence): matvec B1
(x-broadcast), manual double-buffer (halfdot4p), 2-chunk unroll (u2),
b128 grid (device-worse), w2row attention (tg-flat), SSM/GDN kernels
(shadow, 2.7% ceiling), "GPU idle pool" (profiler artifact — never read
busy/wall gap from Q27_METAL_PROFILE logs).
Open but user-closed as "uncertain payoff, medium-large cost":
- matvec B tail: 264->338 GB/s ceiling needs deep reformulation
  (mantissa-direct / simdgroup_matrix); ~+10% tg upper bound.
- prefill post-h4 share unprofiled (needs a harness that does NOT sample
  per-encoder, or reads gaps only from clean runs).
If resumed: user agreed prefill inventory first (momentum side), B tail
second. Optional: share the M4-assumptions-dont-transfer corpus
(DeltaNet thresholds, r2 arm, 1.7x gate, attention route data) upstream
with signalnine as an issue — user suggestion, not started.
Verification checklist for any kernel edit: rebuild tools -> strings | grep
<kernel_name> -> ops -> golden(margins) -> clean tg/pp A/B (best-of-2;
differences <2.5% are unmeasurable by this rig).

## File map (new stuff this session)

```
bench/m1max/
  STATUS.md                      this file
  INSTRUCTIONS.md                user's tuning instructions (source of truth) + amends
  baseline_chunk_fp16.jsonl      run 1 (old default route)
  baseline_chunk_fp16_run2.jsonl run 2 (powermetrics-overlapped)
  grid_m1.jsonl                  20-trial route x seq grid
  confirm_b256.jsonl / confirm_family_default.jsonl
  golden_m1_run1.jsonl / golden_m1_run2.jsonl   (old default, reps=2, deterministic)
  golden_m1_b256.jsonl           (forced b256, diverges 9/32 — expected)
  golden_m1_family_default.jsonl (new default, 32/32 invariant)
  ppl_m1_q4s.json / ppl_m1_q4s_b256.json
  wikitext2_test.txt             corpus (1.29 MB text, ~330K tokens)
  logs/                          build, download, grid, profile, repro logs
tools/
  bench_metal.cpp                pp/tg split bench, medians, memory+rws logging
  golden_metal.cpp               golden + ppl harness (+ --step-margins)
  golden_check.py                reordering-class gate (root-margin + PPL)
  roofline_m1.mm                 matvec roofline/arm harness (Step A)
  bench_stream.metal             bench-only stream kernel (Step A)
  grid_m1.sh                     per-trial-timeout grid runner
src/metal/
  metal_backend.mm               delta fallback wiring, diag logging, family GQA defaults
  q27_kernels.metal              q27_delta_step256, q27_delta_chunk256
  powermetrics_log.txt           (user-collected, 46K samples)
models/                          q4s weights + tokenizer (downloaded, checksummed)
```
