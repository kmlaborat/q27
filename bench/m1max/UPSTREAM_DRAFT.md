# Upstream sharing draft (ready to file)

Status: DRAFT — written from the m1max tuning campaign (bench/m1max/STATUS.md,
commits 7668547..HEAD). Nothing here is filed yet; each item is self-contained
and can be posted independently. Numbers are from a single M1 Max 32GB
(macOS 26.3) running this repo at the cited commits. Verify on your own
hardware before quoting.

Suggested venue: one umbrella discussion ("M-series generational assumptions
in Metal tuning constants") with the items below as sections, rather than
six separate issues.

---

## 1. Decode GEMV picks f16-hfma2 on M1 Max; magic-number f16 dot wins by ~26%

The Metal decode path is tuned (on M4) assuming half2-fmul2/hfma2 fp16 math
performs per lane. On M1 Max the fp32 dot product compiled from magic-number
f16 unpack (`dot(b, w16) + 8*rowsum`) beats the f16 path: tg 11.65 -> 14.7
t/s (+26%), bit-identical output (golden step margins unchanged over 3 runs,
64x2). The win comes from shorter dependency chains per lane, not from raw
ALU peak, so it does not conflict with the M4 measurements — the two parts
sit in different latency regimes. Suggested direction: pick the dot variant
by device generation (or benchmark at load, as the existing arm gates
already allow).

Rollback gate in-tree: `Q27_METAL_Q4_ARM=r0`.

## 2. Prefill Q4 GEMM: barrier-staged fp16 tiles + tensor cores give +47% pp on M1 Max

`q27_matmul_q4_mm_h` (32-row x 16-token tiles, 64-column staging steps,
byte-LUT dequant into threadgroup memory, simdgroup_matrix accumulate with
per-step scale flush) became the default prefill GEMM here: pp 42.6 -> 62.5
t/s (+47%) on M1 Max. The promotion passed numeric gates (ops green, PPL,
step margins) but landed below the recorded 1.7x kernel gate — that gate's
original measurement predates this machine and appears to be from different
silicon; the gate substitution is documented in bench/m1max/STATUS.md.
Rollback gate in-tree: `Q27_METAL_GEMM_HALF_Q4=0`.

## 3. A measured ceiling for attention kernels on Apple silicon tensor units

While rooflining `attention_f16_causal_gqa_t2` on M1 Max we built probe
arms (stream-only, no-stage, no-exp variants across block sizes) that
separate KV-bandwidth, operand staging, and exponential/SFU costs; see
bench/m1max/attn_roof_m1*.jsonl for the per-shape numbers. Several shipped
kernel constants (threadgroup shapes chosen for M4 tensor throughput) sit
far from what M1-class parts can do; the probes give concrete per-shape
numbers if a load-time autotune is ever considered.

## 4. Staging writes and tensor-core smem reads couple into a hard wall on P-tile parts

We spent a full attribution pass on why the prefill GEMM sits at ~9 uniq
GB/s when the same kernel minus MMA runs at 33 GB/s and minus staging runs
at tensor peak: removing either side individually is nearly free, and every
combined arrangement we built (single/double-buffered staging, wide tiles,
flush-eliminated scales — all bit-identical or tolerance-checked) lands in
a 4.4-5.7 ms band. The coupling itself (shared LSU/tensor fabric
arbitration?) appears to be the constraint on M1 Max. Upstream may want to
know because kernel designs that "look" memory-bound or MMA-bound in
isolated micro-benchmarks will mispredict on these parts; we could not find
a writeup of this behaviour anywhere.

## 5. Speculative-decode acceptance benchmarks are corpus-bound

Config sweeps on a synthetic repeated-paragraph corpus showed +120..140% tg
gains; the same configs on four real-workload corpora (source files, git
diffs, technical prose, Japanese instructions) measured ~0 acceptance and
+-1% tg across all settings (one config, mm4, degraded -7..-18% on real
text). If any current defaults were promoted from synthetic-corpus numbers,
they are probably neither good nor harmful — but the methodology deserves a
note in the repo so future tuning starts from real-text corpora.

---

## 6. Long-context re-evaluation: attention tile-row wins, and speculation waste
   scales with KV depth

Two findings from pushing this machine to 64K context that revise earlier
short-context reads:

a) **Decode attention tile rows.** The 2-row/tile variant measured −33%
kernel time but +0.3% tg at 2048 — dismissed as shadowed. At 64K, where
attention is ~66% of the decode step, the same technique (extended to
4-row tiles) measured +28% then +15.8% tg on top (cumulative +49% vs the
row route), golden digest-identical. Lesson for the repo: a kernel win that
looks "shadowed" at short context can be the dominant win at long context —
re-measure attention changes at the depth where attention share dominates,
not at the depth that's convenient to run.

b) **Speculation waste amplifies with context.** With forced-fallback
(acceptance≈0) MTP costs −0.8% tg (drafter overhead). With burst rounds
firing at moderate acceptance, MTP-on costs −16% tg at 64K in both fp16
and turbo3 KV paths, and the penalty tracks attention share (±1% @2048 →
−2..−8% @7168 → −16% @64K): every wasted lane re-reads the full KV.
Combined with section 5 (acceptance is corpus-bound), the practical rule
for long-context agents is that speculative decode defaults-off is not
merely safe but strongly correct on real workloads.

c) **fp16-vs-quantized attention head-to-head at w4.** After porting the
4-row tiling to the fp16 KV path (−47% kernel vs fp16 row-route), fp16
nearly closes the gap to turbo3 (+0.7% @7168 / +4.7% @64K remaining for
turbo3) because w4 removes the dequant latency that dominated and fp16
then runs near its bandwidth roof. The quantization bandwidth advantage is
largely cancelled once streaming is no longer the bottleneck; turbo3's
remaining case is the 4x KV memory saving, not speed.

---

### Reproduction pointers (all in-tree)

- decode A/B: `tools/bench_metal.cpp` arms; `tools/grid_m1.sh`
- prefill gate/rollback + margins: `tools/golden_metal.cpp`,
  `bench/m1max/golden_m1_h4_margins_*.jsonl`
- attention roofline probes: `tools/attn_roof.mm` (tile-row arms:
  `attn_w2row`, `attn_f16_w4row`, `attn_t3_w4row`, stride probes
  `attn_t3_stream50/64`)
- prefill GEMM waterfall + attribution arms: `tools/pf_roof.mm`,
  `tools/bench_pf.metal` (bit-identity harness included)
- speculation corpus harness: `bench/m1max/phase2a/`
- long-context 2x2 (MTP x KV) + forced-fallback check: `bench/m1max/step2/`
- pp stage attribution: engine `pp_profile_chunk` + `tools/pp_share.cpp`
- long-context quality gate: `tools/niahf_probe.cpp` (`d_niahf.jsonl`)
