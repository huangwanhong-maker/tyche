# 0036. Ranks of Two Samples

- **Status:** Accepted, provisionally (2026-11-23).
- **Date:** 2026-11-23
- **Decided by:** the infrastructure's economics domain, whose `diversification` tests whether the relatedness density of a region around an activity predicts that the region takes it up (Hidalgo et al. 2018's out-of-sample test), which needs the area under the curve of the densities of entries against those of non-entries.

## 1. Context

The area under the receiver operating characteristic curve is Mann and Whitney's U over the pairs of a positive and a negative. tyche compared two groups by their means (decision 0027) and had no rank statistic.

## 2. Decision

`testing/include/tyche/testing/ranks.hpp`: `mann_whitney(x, y)` → `RankSum{u, auc, statistic, p_value, n1, n2}`: U by midranks in O(n log n), the AUC U/(n₁n₂), and the normal test with the variance corrected for ties, without a continuity correction; NaNs left out.

## 3. Alternatives Considered

- **A continuity correction.** It changes the statistic by half a unit of U; scipy's asymptotic method offers both, and the uncorrected form is the AUC's natural companion. A permutation test (permutation.hpp) is exact for small samples.

## 4. Consequences

`check_tyche_ranks`, 7 checks: U, the AUC and the tie-corrected z worked by hand; U equal to the count of pairs on 200 random samples with many ties; perfect separation and alike samples; the size under the null (4.25% at 5% on 2,000 runs); NaNs; refusals. tyche 1826.
