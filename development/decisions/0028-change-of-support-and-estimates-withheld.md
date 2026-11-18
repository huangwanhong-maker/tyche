# 0028. Change of Support, and Estimates Withheld Where They Cannot Be Trusted

- **Status:** Accepted, provisionally (2026-11-18)
- **Date:** 2026-11-18
- **Decided by:** Claude, for the infrastructure's cities across scales (its ADR 0128), at the user's request that an index a dataset does not hold be estimated by any suitable model and, where it cannot be estimated well, taken up when a model is applied; and at the survey made for it (`research/literature/reports/Indexes of cities across scales.md`, Part II).

## 1. Context

An index known on one set of zones (census areas) is wanted on another (city blocks), and an index observed at some units is wanted at others. tyche had Gaussian conditioning, regression and its cross-validation, but no change of support between partitions and no estimator that says where it should not be believed. The survey's rankings put first a refusal that says why, then volume-preserving change of support, then the area of applicability with cross-validation matched to the prediction task, and conformal intervals within it (Part II, recommendations 1, 2, 5 and 10). It also found that proxies validated by random cross-validation look much better than they are (Part II §3).

## 2. Decision

1. **`spatial/support.hpp`**: areal weighting of extensive and intensive quantities; dasymetric carriage on ancillary amounts, a source without one carried by measure and counted; the densities of ancillary classes by non-negative least squares (the regression form of intelligent dasymetric mapping); pycnophylactic interpolation on a grid of zones. Every method keeps the sources' totals where the targets cover them.
2. **`spatial/prediction.hpp`**: a linear estimator (least squares or ridge on standardised covariates, of the logarithm if asked) judged by blocked cross-validation (one block of units held out at a time), its intervals conformal from the errors on the blocks held out, the dissimilarity index of Meyer and Pebesma and its threshold (the upper whisker of the training units' indices against the other blocks), and every estimate withheld with its reason where the estimator does not beat the mean on blocks held out by more than a declared floor, where the target lies outside the area of applicability, or where its interval is wider than a declared share of it.
3. **A withheld estimate is not a missing value**: it is reported as withheld, with the reason, for a person to declare.

## 3. Alternatives Considered

- **Random forests and gradient boosting** as the learner, as Meyer and Pebesma use: a learner without a reference implementation here; the area of applicability and the blocked judgement are what matter, and they wrap any learner. The linear estimator is the first.
- **Area-to-point kriging** for change of support: coherent and with variances, but it needs a covariance model the callers do not yet state; left for when one does.
- **Fay–Herriot extensions** for estimates from surveys: they are peira's (beside its EBLUP).
- **Random folds**: they overstate skill on spatially dependent data (Roberts et al. 2017), so they are not offered.

## 4. Consequences

- The infrastructure's API offers both through `science/scales.hpp` (`carry`, `estimate`); the Urban Studio's plug-in *Cities across scales* uses the dasymetric carriage to share a census's withheld floor space among its blank cells.
- Two check programs, 33 checks; tyche has 1706.
- Not built: area-to-point kriging; conformal intervals localised in space; NNDM folds; temporal disaggregation (Denton, Chow–Lin); BYM2 areal models.
