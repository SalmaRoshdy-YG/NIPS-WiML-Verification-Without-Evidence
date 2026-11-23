# Replication package: *When Is a Multi-Agent Code Judge Actually Grounded?*

WiML poster abstract: Aly, Assaf, Kobti, arXiv:2609.30328 (v1). This repo has the scripts that produce the
numbers in that abstract.

## What the paper does
It runs MARCH (Li et al., ACL 2026) unmodified as a pairwise code judge on CodeJudgeBench (two solutions to
one problem, one correct, the judge picks it) and compares it with the same Qwen3 model judging directly.
MARCH runs three agents: a **Solver** forms an assessment of each solution, a **Proposer** turns it into
checkable claims, and a **Checker** answers each claim from the problem and the solution without seeing the
Solver's reasoning. The solution with fewer failed claims wins; equal scores are a tie. Two label-free
measurements come from the pipeline's own logs: **claim invariance** (the share of comparisons where the
Proposer asked the same questions about both solutions) and **checker agreement**. A **gate** withholds a
comparison when the questions are identical.

## Results in the paper
Qwen3-8B and Qwen3-14B judges, CodeJudgeBench (CodeGen split), accuracy over all comparisons.

| Difficulty | Direct | MARCH ungated | Kept after gate | MARCH gated | Gain |
|---|---|---|---|---|---|
| Easy | 94.3% | 28.7% | 62.5% | 43.5% | +14.9 |
| Medium | 75.0% | 21.2% | 54.3% | 35.4% | +14.2 |
| Hard | 45.1% | 13.5% | 37.9% | 28.9% | +15.4 |
| All | 73.4% | 20.7% | 50.6% | 36.9% | +16.2 |

Also reported: the pipeline reaches 4.4% where the direct judge reaches 43.7% (p = 1.1e-20); the proposer
asked the same questions about both solutions in 76.6% of comparisons; the checker agreed with 80.7% of the
claims it verified.

> **Note.** These numbers come from the pipeline as it ran for v1. A later revision of how the Solver's
> output reaches the claim generator does not reproduce the gate's gain (+0.0 instead of +16.2, with the gate
> keeping 99.7% of comparisons). A revised paper will report those results.

## Layout
```
pipeline/
  src/run_condition.py     one (condition, generator, tier, judge) cell, sharded
  src/march_verbatim.py    MARCH's published prompts and comparator, unmodified
  src/selftest.py          checks the three-call structure; no GPU needed
  slurm/xfer_cell.sh       one shard per Slurm array task
  SHA256.txt               checksums of the runner and MARCH modules
scripts/
  submit_runs.sh           submits the direct-judge (c0) and MARCH (c1) runs
  make_table.py            rebuilds the table, invariance, checker agreement and the paired test
data/hef_pairs.jsonl       the 142 HumanEvalFix pairs
environment/README.md      modules, package versions, models, data, sampling settings
```

## Reproduce
Run on the GPU login node (`trig-login01`); see `environment/README.md` for the environment.
```bash
SCOPE=full bash scripts/submit_runs.sh c0      # direct judge: 9 CodeJudgeBench cells + HumanEvalFix, both judges
SCOPE=full bash scripts/submit_runs.sh c1      # MARCH, same cells
python3 scripts/make_table.py --c0 $STUDY/march-gens-c0 --c1 $STUDY/march-gens-c1 --hef
```
Drop `SCOPE=full` for a 24-shard pilot (gemini, three tiers, Qwen3-8B). `make_table.py` prints the table
above, claim invariance, checker agreement, and an exact McNemar test of direct against MARCH. A cell counts
only when it has every row (both orderings of every pair). Accuracy is strict: ties and unparsed verdicts
count as wrong. Check the code first with `(cd pipeline/src && python3 selftest.py)`.

## What "reproduced" means
Generation is sampled (temperature 0.6, seed 1234) and the abstract ran on another cluster with another
transformers version (5.14.1; the environment file lists 5.17.0). Expect agreement within run-to-run noise,
not identical digits.

## Status
The scripts are tested (self-test passes; the table script runs on existing outputs). The numbers above are
the paper's; the rerun that regenerates them has not been done yet.
