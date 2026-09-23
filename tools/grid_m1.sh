#!/bin/bash
# Crossover grid for the M1 Max tuning pass: decode-route choice
# (t2 vs blocked GQA) x context length, with per-trial timeouts so a
# hanging kernel (blk=512 at long seq) can't stall the sweep. Each trial
# loads the model fresh (root is chosen by env at backend init), ingests
# --seq tokens, then times gen 32 decode tokens; tg was reproducible to
# ~1% across reps in the baseline runs, so reps=1 keeps the sweep ~1h.
#
# timeout formula: 60s margin + ingest at 20 t/s worst case + decode at
# 2 t/s worst case. A killed trial is recorded as HANG and the sweep moves on.
#
# usage: tools/grid_m1.sh [out.jsonl]   (run from repo root)

set -u
cd "$(dirname "$0")/.."
OUT="${1:-bench/m1max/grid_m1.jsonl}"
MODEL=models/qwen36-27b-mtp-q4s.q27
TOK=models/qwen36-27b-mtp.tok
GEN=32
mkdir -p bench/m1max
: > "$OUT"

# name:TH:BLOCK pairs; TH=0 keeps the t2 route at any seq.
ROUTES="t2:0:1024 b256:1:256 b1024:1:1024 b512:1:512"
SEQS="2048 3072 4096 6144 7168"

for route in $ROUTES; do
  name="${route%%:*}"; rest="${route#*:}"; th="${rest%%:*}"; blk="${rest#*:}"
  for seq in $SEQS; do
    tmo=$((60 + seq/20 + GEN/2))
    log="bench/m1max/logs/grid_${name}_${seq}.log"
    Q27_METAL_GQA_THRESHOLD="$th" Q27_METAL_GQA_BLOCK="$blk" \
      ./build/bench_metal "$MODEL" "$TOK" --seq "$seq" --gen "$GEN" --reps 1 \
      --ctx 8192 --out /dev/null > "$log" 2>&1 &
    pid=$!
    waited=0; status=OK
    while kill -0 $pid 2>/dev/null; do
      sleep 5; waited=$((waited+5))
      if [ $waited -ge $tmo ]; then
        kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
        status=HANG; break
      fi
    done
    if [ "$status" = OK ] && ! wait $pid; then status=FAIL; fi
    line=$(grep -oE "seq=[0-9]+.*tg=[ ]*[0-9.]+ t/s" "$log" 2>/dev/null | head -1)
    tg=$(echo "$line" | grep -oE "tg=[ ]*[0-9.]+" | grep -oE "[0-9.]+")
    pp=$(echo "$line" | grep -oE "pp=[ ]*[0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$tg" ] && { tg=null; pp=null; [ "$status" = OK ] && status=NODATA; }
    printf '{"route":"%s","th":%s,"blk":%s,"seq":%s,"gen":%d,"tg":%s,"pp":%s,"status":"%s","timeout":%d}\n' \
      "$name" "$th" "$blk" "$seq" "$GEN" "$tg" "$pp" "$status" "$tmo" >> "$OUT"
    echo "$(date +%H:%M:%S) $name seq=$seq tg=$tg pp=$pp $status"
  done
done
echo DONE >> "$OUT"
