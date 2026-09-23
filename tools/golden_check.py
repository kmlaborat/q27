#!/usr/bin/env python3
"""Golden comparison gate for the M1 Max tuning pass.

usage: golden_check.py BASELINE.jsonl CANDIDATE.jsonl [--tie-band 0.5]
                      [--max-ppl-delta 0.02 --ppl-a pplA.json --ppl-b pplB.json]

Both jsonl files must come from tools/golden_metal.cpp golden runs with the
same --gen and --reps and --step-margins enabled.

Verdicts, encoding the rule agreed 2026-09-23 after the b256/t2 study
(margins_default.jsonl / margins_b256.jsonl):
  * Reordering-class changes (anything altering fp accumulation order)
    produce bounded noise: measured identical-context |t1 logit| drift up to
    2.15 at confident steps, p99 0.44; every sequence branch was rooted at a
    step whose top1-top2 margin (min over the two routes) was <= 0.0785,
    deep inside the tie band. No branch was ever rooted at a confident margin.
  * PASS requires: every divergence point has min(margin_base, margin_cand)
    <= tie_band (a flip at a confident margin is treated as a bug, not
    noise); and |ppl delta| <= max-ppl-delta when PPL files are given.
  * Determinism (same build, same route) is checked by digest equality and
    is strict; this script is for the reordering-class comparison only.
"""
import argparse, json, sys

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline"); ap.add_argument("candidate")
    ap.add_argument("--tie-band", type=float, default=0.5)
    ap.add_argument("--max-ppl-delta", type=float, default=0.02)
    ap.add_argument("--ppl-a"); ap.add_argument("--ppl-b")
    a = ap.parse_args()

    base = {(r["prompt"], r.get("rep", 0)): r for r in load(a.baseline)}
    cand = {(r["prompt"], r.get("rep", 0)): r for r in load(a.candidate)}
    if set(base) != set(cand):
        print("FAIL: prompt/rep sets differ"); sys.exit(1)

    branches, confident_branches = [], []
    nsteps = 0
    for k, rb in sorted(base.items()):
        rc = cand[k]
        ib, ic = rb["ids"], rc["ids"]
        mb, mc = rb.get("margins"), rc.get("margins")
        if mb is None or mc is None:
            print("FAIL: --step-margins output required"); sys.exit(1)
        nsteps += min(len(ib), len(ic))
        # Only the FIRST divergence of each prompt is a decision point: past
        # it the two sequences are legitimately different contexts and their
        # per-step comparisons are cascade, not evidence. The root trigger
        # must be a tie-band flip; anything else is a bug.
        first = next((i for i in range(min(len(ib), len(ic))) if ib[i] != ic[i]), None)
        if first is None:
            continue
        margin = min(mb[first][1] - mb[first][2], mc[first][1] - mc[first][2])
        branches.append((k[0], first, margin))
        if margin > a.tie_band:
            confident_branches.append((k[0], first, margin))
    print(f"steps={nsteps} prompts={len(base)} branches={len(branches)} "
          f"max-root-margin={max((m for _,_,m in branches), default=0.0):.4f}")
    for p, i, m in branches:
        print(f"  branch p{p} first-diff step {i} root margin {m:.4f}")
    ok = not confident_branches
    for p, i, m in confident_branches:
        print(f"  CONFIDENT BRANCH (bug?) p{p} step {i} margin {m:.4f} > tie band {a.tie_band}")
    if a.ppl_a and a.ppl_b:
        pa = json.load(open(a.ppl_a))["ppl"]; pb = json.load(open(a.ppl_b))["ppl"]
        d = abs(pa - pb)
        print(f"ppl: baseline={pa:.4f} candidate={pb:.4f} delta={d:.4f} (max {a.max_ppl_delta})")
        ok = ok and d <= a.max_ppl_delta
    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
