# Errata for the WiML poster abstract (arXiv:2609.30328v1)

The v1 pipeline discarded the Solver's reasoning before claim generation (see README, "The
defect"). Every v1 result that depends on claim generation is affected. Status is from the
corrected pipeline, seed 1234, the 12 cells where both direct and MARCH runs are complete.

| v1 claim | v1 value | Corrected | Status |
|---|---|---|---|
| Direct judge accuracy (all tiers) | 73.4% | 74.3% | holds (no claim generator involved) |
| MARCH far below the direct judge | 20.7% vs 73.4% | 17.4% vs 74.3% | **holds** |
| Easier problems help the direct judge, not MARCH | +49 pts vs under 1 | +48 vs about +3 | holds |
| Proposer asked the same questions about both solutions | 76.6% | 0.3% (0.7% over 311 items in the earlier audit) | **does not hold** |
| Checker agreed with the claims it verified | 80.7% | 93.6% | **does not hold** |
| Gate keeps about half of comparisons | 50.6% | 99.7% | **does not hold** |
| Gate raises accuracy | 20.7% to 36.9% (+16.2) | 17.4% to 17.4% (+0.0) | **does not hold** |
| MARCH declares a tie on 78 to 95% of comparisons | 78 to 95% | 56 to 72% | direction holds, cause differs |

Not checked here: the abstract's p = 1.1e-20 and its 4.4% accuracy cell, which come from
cells outside the 12 used above. The numbers above come from different cell sets
(12 cells corrected vs the paper's 54), so read them as direction, not a like-for-like match.

What survives: MARCH, ported unmodified, judges far worse than the same model asked directly, and
a larger judge does not close the gap. What does not: the explanation (identical proposer
questions) and the repair (the gate). The v1 text should be revised before this package is cited.
