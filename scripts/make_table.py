#!/usr/bin/env python3
"""Regenerates the table in the arXiv abstract (arXiv:2609.30328v1) from run outputs.

  python3 scripts/make_table.py --c0 DIR_WITH_C0 --c1 DIR_WITH_C1 [--judge 8b|qwen3-14b|both] [--seed 1234]

Columns: direct accuracy (C0), MARCH ungated accuracy (C1), share of comparisons kept by the
gate, MARCH gated accuracy, gain. Also prints the two measurements the paper builds on:
claim invariance (both candidates got identical proposer questions) and checker agreement.

Rules (identical for every number):
  * A cell is used only if it has all 2 x tier-size rows (both orderings of every pair).
  * Accuracy is strict: pred == label; ties and unparsed verdicts count as wrong.
  * The gate withholds a comparison when the proposer asked the same questions about both
    solutions. It reads nothing else.
"""
import argparse, collections, glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v1_pipeline", "src"))
import march_verbatim as MV

TIER = {("claude_3.7_sonnet","easy"):162,("claude_3.7_sonnet","medium"):92,("claude_3.7_sonnet","hard"):71,
        ("gemini_2.5_pro","easy"):64,("gemini_2.5_pro","medium"):57,("gemini_2.5_pro","hard"):135,
        ("qwen3_235b","easy"):46,("qwen3_235b","medium"):59,("qwen3_235b","hard"):113,("hef","all"):142}
PAT = re.compile(r"(c\d_\w+?)__(?:codejudgebench__codegen|humanevalfix__codegen)__(.+?)__(easy|medium|hard|all)(?:__(qwen3-14b))?(?:__s(\d+))?$")

def load(root, cond, seed):
    out = {}
    for d in glob.glob(os.path.join(root, f"{cond}__*")):
        m = PAT.match(os.path.basename(d))
        if not m or (m.group(5) or "1234") != seed: continue
        gen, tier, judge = m.group(2), m.group(3), m.group(4) or "8b"
        rows = {}
        for f in glob.glob(d + "/shard_*.jsonl"):
            for l in open(f):
                r = json.loads(l); rows[(r["uid"], r["reverse"])] = r
        out[(gen, tier, judge)] = rows
    return out

def complete(cells):
    return {k: v for k, v in cells.items() if len(v) == 2 * TIER[(k[0], k[1])]}

ap = argparse.ArgumentParser()
ap.add_argument("--c0", required=True); ap.add_argument("--c1", required=True)
ap.add_argument("--judge", default="both"); ap.add_argument("--hef", action="store_true", help="also report HumanEvalFix, separately from the CodeJudgeBench table"); ap.add_argument("--seed", default="1234")
a = ap.parse_args()
c0 = complete(load(a.c0, "c0_direct", a.seed)); c1 = complete(load(a.c1, "c1_verbatim", a.seed))
allk = sorted(k for k in c1 if k in c0 and (a.judge == "both" or k[2] == a.judge))
keys = [k for k in allk if k[0] != "hef"]; hefk = [k for k in allk if k[0] == "hef"]
if not keys: sys.exit("no cell has both C0 and C1 complete")
ident = lambda r: sorted(q["question"] for q in r["pos_qa_pairs"]) == sorted(q["question"] for q in r["neg_qa_pairs"])
acc = lambda rs: 100 * sum(r["pred"] == r["label"] for r in rs) / len(rs) if rs else float("nan")
print(f"cells used ({len(keys)}): " + "; ".join(f"{g}/{t}/{j}" for g, t, j in keys))
print(f"\n{'':8}{'direct':>8}{'ungated':>9}{'kept':>8}{'gated':>8}{'gain':>8}{'inv.':>8}{'chk agr':>9}{'rows':>7}")
for tier in ("easy", "medium", "hard", "all"):
    ks = [k for k in keys if tier == "all" or k[1] == tier]
    if not ks: continue
    d = [r for k in ks for r in c0[k].values()]; m = [r for k in ks for r in c1[k].values()]
    kept = [r for r in m if not ident(r)]
    inv = 100 * (len(m) - len(kept)) / len(m)
    agree = tot = 0
    for r in m:
        for s in ("pos", "neg"):
            qa, ck = r[s + "_qa_pairs"], r[s + "_checker"]
            for c in ck:
                t = c.split("</think>")[-1] if "</think>" in c else c
                t = t if t.strip().startswith("1.") else "1." + t.strip()
                info = MV.extract_detailed_info(t)
                for i, q in enumerate(qa):
                    tot += 1; agree += (i + 1) in info and MV.check_ans_eq(info[i + 1]["answer"], q["answer"])
    print(f"{tier:8}{acc(d):8.1f}{acc(m):9.1f}{100*len(kept)/len(m):7.1f}%{acc(kept):8.1f}{acc(kept)-acc(m):+8.1f}{inv:7.1f}%{100*agree/tot if tot else float('nan'):8.1f}%{len(m):7d}")

# paired exact test, direct vs MARCH, over all comparisons in the table (two-sided McNemar)
import math
from fractions import Fraction
def mcnemar(ks, tag):
    n10 = n01 = 0
    for k in ks:
        for key, r in c1[k].items():
            d = c0[k][key]; a_ = d["pred"] == d["label"]; b_ = r["pred"] == r["label"]
            n10 += a_ and not b_; n01 += b_ and not a_
    n = n10 + n01; kk = min(n10, n01)
    p = min(1.0, float(2 * Fraction(sum(math.comb(n, i) for i in range(kk + 1)), 2 ** n))) if n else 1.0
    print(f"{tag}: direct right and MARCH wrong {n10}, MARCH right and direct wrong {n01}, exact McNemar p = {p:.2e}")
print(); mcnemar(keys, "all CodeJudgeBench comparisons")
if a.hef:
    if not hefk: print("\nHumanEvalFix: no cell has both conditions complete")
    for k in hefk:
        m = list(c1[k].values()); d = list(c0[k].values())
        print(f"\nHumanEvalFix ({k[2]}): direct {acc(d):.1f}  MARCH {acc(m):.1f}  tie {100*sum(bool(r.get('tie')) for r in m)/len(m):.1f}%  rows {len(m)}")
        mcnemar([k], f"  HumanEvalFix {k[2]}")
