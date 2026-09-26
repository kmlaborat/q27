#!/bin/bash
# Phase D Step 2: MTP x KV 2x2. All combos run with Q27_METAL_ATT=w4
# (attention w4row in BOTH KV modes; distinct from MTP draft width).
# MTP on  = --mode suffix --width 4 --min-match 12  (w4/mm12)
# MTP off = --mode greedy (serial step only)
M=models/qwen36-27b-mtp-q4s.q27; T=models/qwen36-27b-mtp.tok
C64=bench/m1max/step2_corpus64k.txt
D=bench/m1max/step2
mkdir -p $D
run64 () { # $1=kv $2=mode-label $3=extra-args...  (rep passed via $REP)
  local kv=$1 lab=$2; shift 2
  # STRUCTURAL: rep number MUST be in the filename — without it the skip
  # check saw rep1's files and silently skipped rep2 (happened in Step 2;
  # same class as the stale-binary trap: silent, structural, harmless only
  # by luck).
  [ -s $D/64k_${lab}_rep${REP}.jsonl ] && { echo "skip 64k_${lab}_rep${REP}" >> $D/progress.txt; return; }
  Q27_METAL_ATT=w4 ./build/bench_metal $M $T --kv $kv --ctx 65536 --seq 65400 --gen 32 --reps 1 "$@" --prompt-file $C64 --out $D/64k_${lab}_rep${REP}.jsonl > $D/64k_${lab}_rep${REP}.log 2>&1
}
run7k () {
  local kv=$1 lab=$2; shift 2
  [ -s $D/7168_${lab}_rep${REP}.jsonl ] && { echo "skip 7168_${lab}_rep${REP}" >> $D/progress.txt; return; }
  Q27_METAL_ATT=w4 ./build/bench_metal $M $T --kv $kv --ctx 8192 --seq 7168 --gen 128 --reps 2 "$@" --prompt-file $C64 --out $D/7168_${lab}_rep${REP}.jsonl > $D/7168_${lab}_rep${REP}.log 2>&1
}
for REP in 1 2; do
  run64 turbo3 t3_mtpon  --mode suffix --width 4 --min-match 12
  run64 turbo3 t3_mtpoff --mode greedy
  run64 fp16   f16_mtpon  --mode suffix --width 4 --min-match 12
  run64 fp16   f16_mtpoff --mode greedy
  echo "rep${REP}-64k-done" >> $D/progress.txt
done
for REP in 1 2; do
  run7k turbo3 t3_mtpon  --mode suffix --width 4 --min-match 12
  run7k turbo3 t3_mtpoff --mode greedy
  run7k fp16   f16_mtpon  --mode suffix --width 4 --min-match 12
  run7k fp16   f16_mtpoff --mode greedy
done
echo ALL-DONE >> $D/progress.txt
