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

## Pending / next (in order)

1. **matvec_q4_quantized analysis** (approved direction): register/threadgroup
   stats via now-installed Metal Toolchain (`xcrun metal -c … && xcrun
   metal-objdump -d` or air stats), kernel read-pattern review on M1 Max, then
   a modification proposal. Gate: golden digest + PPL before/after, plus
   test-metal-ops.
2. attention_f16 slope inefficiency (~3% of bandwidth) — first target after
   matvec; ~60% of decode time at 7K context is attention.
3. Decide: commit current work (2 src files + tools + bench/m1max/), and/or
   derive metal-m1 canonical for `metal_canonical_gate.sh` (old-route digest
   `c30b2e89d450d563` and family-default 32/32 invariance are on disk to start from).
4. Deferred by agreement: turbo3 3-bit KV evaluation on M1; MTP/continuous-batch
   Metal-side effectiveness measurement (`matmul_q4_mm` 16.4% path); 512-vs-256
   delta cross-check (needs Apple8+ or CUDA machine); PPL with longer context.
5. For powermetrics runs alongside benches, user starts
   `sudo powermetrics --samplers gpu_power,cpu_power -i 1000` manually.

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
  golden_metal.cpp               golden + ppl harness
  grid_m1.sh                     per-trial-timeout grid runner
src/metal/
  metal_backend.mm               delta fallback wiring, diag logging, family GQA defaults
  q27_kernels.metal              q27_delta_step256, q27_delta_chunk256
  powermetrics_log.txt           (user-collected, 46K samples)
models/                          q4s weights + tokenizer (downloaded, checksummed)
```
