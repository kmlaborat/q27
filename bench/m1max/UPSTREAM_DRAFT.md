# Upstream sharing draft (ready to file)

Status: DRAFT — written from the m1max tuning campaign (bench/m1max/STATUS.md,
commits 7668547..HEAD). Nothing here is filed yet; each item is self-contained
and can be posted independently. Numbers are from a single M1 Max 64GB
(macOS 26.3) running this repo at the cited commits. Verify on your own
hardware before quoting.

Suggested venue: one umbrella discussion ("M-series generational assumptions
in Metal tuning constants") with the items below as sections, rather than
five separate issues.

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

### Reproduction pointers (all in-tree)

- decode A/B: `tools/bench_metal.cpp` arms; `tools/grid_m1.sh`
- prefill gate/rollback + margins: `tools/golden_metal.cpp`,
  `bench/m1max/golden_m1_h4_margins_*.jsonl`
- attention roofline probes: `tools/attn_roof.mm`
- prefill GEMM waterfall + attribution arms: `tools/pf_roof.mm`,
  `tools/bench_pf.metal` (bit-identity harness included)
- speculation corpus harness: `bench/m1max/phase2a/`
