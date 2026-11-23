# Replication package: *When Is a Multi-Agent Code Judge Actually Grounded?*

Paper: Aly, Assaf, Kobti. WiML poster abstract, arXiv:2609.30328 (v1, 23 Sep 2026).
This package contains the code and results for that paper only.

**Read `ERRATA.md` first.** The pipeline that produced the v1 abstract had a defect. This package
reruns that pipeline on purpose so the v1 numbers can be regenerated and compared with the
corrected pipeline. It does not claim the v1 conclusions hold.

## What the paper does
It runs MARCH (Li et al., ACL 2026) unmodified as a pairwise code judge on CodeJudgeBench
(two solutions to one problem, one correct, the judge picks the correct one) and compares it with
the same Qwen3 model judging directly. It then measures two things from logs the pipeline already
writes: (1) claim invariance, the share of comparisons where the proposer asked the same questions
about both solutions, and (2) checker agreement. It uses (1) as a gate that withholds comparisons.

## Layout
```
ERRATA.md                what in v1 holds, what does not, and why
v1_pipeline/             the pipeline as it ran for v1, defect included
  src/run_condition.py     the runner (identical to the pre-fix backup; SHA256.txt)
  src/march_verbatim.py    MARCH's published prompts and comparator, unmodified
  src/march_code.py ...    supporting modules the runner imports
  slurm/xfer_cell.sh       one shard per Slurm array task
data/
  hef_pairs.jsonl          the 142 HumanEvalFix pairs (327 KB)
scripts/
  submit_v1_repro.sh       reruns C1 with the v1 pipeline (run on trig-login01)
  make_table.py            regenerates the abstract's table from any run directory
results/corrected/       the paper's table and per-cell paired tests from the corrected pipeline
results/v1_rerun/        filled once the v1 rerun finishes
```

## The defect, in one paragraph
In `v1_pipeline/src/run_condition.py` line 268, the Solver's output was passed through
`strip_think`, which keeps only the text after `</think>`. For Qwen3 that is about 30 characters
("FINAL: CORRECT"), so the claim generator saw almost none of the Solver's assessment. The helper
is not in MARCH's code; it was added in the port. The fix keeps the full assessment with only the
tags removed.

## Reproduce
1. Submit the v1 rerun (pilot = gemini, three tiers, Qwen3-8B, 24 shards):
   `bash scripts/submit_v1_repro.sh`   (must be run on `trig-login01`)
   `SCOPE=full bash scripts/submit_v1_repro.sh` runs all nine CodeJudgeBench cells and
   HumanEvalFix (`data/hef_pairs.jsonl`) with both judges, 148 shards in total.
2. When the shards finish, build the table:
   `python3 scripts/make_table.py --c0 <dir with C0 outputs> --c1 <march-gens-v1repro>`
   Add `--hef` to also report HumanEvalFix. The table ends with the exact paired McNemar test.
   C0 (the direct judge) has no defect, so the existing C0 outputs are reused.
3. To see the corrected behaviour, point `--c1` at a directory produced by the current runner.

## What "reproduced" means here
Generation is sampled (temperature 0.6, seed 1234), and v1 ran on a different cluster with
transformers 5.14.1 (this cluster has 5.17.0). Expect agreement within run-to-run noise, not
identical digits. Seed-to-seed spread for C0 and C1 in the corrected grid is under 3 points.

## Raw generations
Not in this repo (about 825 MB of JSONL). Archive them (for example on Zenodo) and link the DOI here.
Every table can be rebuilt from them with `scripts/make_table.py`.

## Status
| Item | State |
|---|---|
| v1 pipeline snapshot | done, checksummed |
| Table script | done, tested on corrected data |
| Corrected-pipeline results | done, in `results/corrected/` |
| v1 rerun | **not yet run** |
| v1 vs rerun comparison | pending the rerun |
