# 0006. Change Points, Penalised Regression, Non-Negative Least Squares and Non-Negative Factorisation

- **Status:** Accepted, provisionally (univariate change points with a sum for joint series; the gaussian and binomial families for penalised regression; dense matrices throughout)
- **Date:** 2026-11-01
- **Decided by:** the infrastructure's genome analyses of its ADR 0064 (genomes that depart from their reference, their description, and expression predicted) need methods that name no object in nature: segmenting read depth, allele fractions and tumour copy number; predicting expression from genotypes and from sequence features with sparse linear models; refitting and extracting mutational signatures. Claude wrote the methods and this record at the user's request ("implement traditional and advanced genome analysis algorithms, especially abnormal detection, expression prediction"); the numerics they stand on that calx lacked (the assignment problem, multinomial variates) went to calx (its decision 0014).

## 1. Context

`time_series` listed change points as planned; `regression` listed regularisation as planned; `multivariate` was a planned row. The genome analyses need:

- **Segmentation** of a sequence of observations (read depth in bins, B-allele fractions, log ratios) into stretches of constant law: exactly, by a penalised likelihood, for the counts and Gaussian measurements of sequencing, with minimum segment lengths; and by circular binary segmentation, the method of array CGH and its successors, with its permutation reference distribution.
- **Penalised regression** as glmnet computes it, since PrediXcan's weights are glmnet's elastic net with cross-validation, and models from many sequence features need the lasso.
- **Non-negative least squares** (the exposures of known signatures) and **non-negative matrix factorisation** with the stability analysis by which mutational signatures are extracted.

## 2. Decision

### 2.1 `time_series/changepoint.hpp`

- **Costs.** `SegmentCost` for the models `mean` (known or robustly estimated σ², precision weights), `variance` (known mean), `mean_variance` and `poisson` (with exposures), each from compensated prefix sums in O(1) a segment; `SumCost` for several series cut at once; any type with `size()` and `operator()(begin, end)` is a cost (`SegmentCostFunction`).
- **Penalties.** `Penalty{bic, mbic, aic, hannan_quinn, manual, none}` → `PenaltyTerms{per_changepoint, per_segment, n}`; MBIC is Zhang and Siegmund's for a change in mean (3 log n per change, Σ log(ℓ/n) over the segments), with (d + 2) log n and weight d for d parameters (a stated generalisation).
- **Exact searches.** `optimal_partitioning` and `pelt`, which return the same segmentation:
  - a candidate found dominated at t is removed only from t + ℓ_min on, which keeps PELT exact with a minimum segment length;
  - the pruning margin K = 10⁻⁹(1 + |F(t)|) absorbs the rounding of the cost inequality, so that no candidate tying the optimum is pruned, and both take the earliest of tied last change points.
- **Approximate searches.** `binary_segmentation` (gains above β, greatest first, to `max_changepoints`) and `wild_binary_segmentation` (Fryzlewicz's random intervals, the WBS.NONREC solution path, then a threshold or the penalised choice along the path).
- **Circular binary segmentation** (`circular_binary_segmentation`, `cbs_statistic`), as Olshen et al. (2004) define it:
  - the maximal statistic by an exact branch and bound over pairs of index ranges;
  - permutation p-values (1 + R)/(1 + B) from calx streams keyed by the segment's bounds, so that the result does not depend on the order of the recursion;
  - Besag and Clifford's curtailment, which keeps the full test's decision;
  - an early-rejection boundary after Venkatraman and Olshen (2007), but derived here: b_i the least b at which a hypergeometric draw shows fewer than i exceedances with probability ≤ η 2^{−i} when the full record just fails to reject. Its error given any record is below η, and its added size under the null is computed exactly in the checks;
  - pruning by the sum of squares (the fewest change points, chosen by dynamic programming, with SS_j/SS_K − 1 < γ) and undoing by the means' separation.
- **Plain structs.** `Segment{begin, end, mean, variance, cost}` and `Segmentation{changepoints, segments, cost, penalty, objective, variance, scores, p_values, evaluations}`.

### 2.2 `regression/penalized.hpp`

- **Fits.** `elastic_net_path`, `elastic_net`, `lasso` and `ridge` (in closed form by calx's QR of the augmented design), as Friedman, Hastie and Tibshirani (2010) define them:
  - the gaussian and binomial families, the second by IRLS around the coordinate descent (with step halving, which glmnet omits);
  - weights normalised to sum to 1, offsets, and penalty factors rescaled to sum to p as glmnet rescales them;
  - standardisation by the weighted population standard deviation, with the coefficients reported on X's scale;
  - λ_max, and glmnet's default path (100 values down to 10⁻⁴ or 10⁻² of λ_max);
  - active-set sweeps and warm starts, then an exact finish: with the active set and its signs fixed, the stationarity conditions are linear and are solved by calx's QR, the result kept if no sign changes and every inactive coordinate satisfies its condition. If a sign would change, an active-set step goes as far as the first coordinate reaching 0 and the sweeps go on. Coordinate descent alone converges linearly, slowly on correlated columns, and its small last steps left errors of 4 × 10⁻³ against scikit-learn where p > N; with the finish, fits agree to rounding. The finish is skipped above 400 active coordinates or when its system is singular (the lasso with more active coordinates than N).
- **Verification and use.** `penalized_kkt` measures a fit's KKT violation; `predict` gives the linear predictor or the probabilities.
- **Cross-validation.** `cross_validate`:
  - balanced deterministic folds from a calx stream, or folds given;
  - the same λ in every fold;
  - squared error, deviance or misclassification;
  - cvm, cvsd, λ_min and λ_1se by the one-standard-error rule.

### 2.3 `regression/nnls.hpp`

`nnls(A, b)` → `NnlsResult{x, dual, residual_norm, iterations, converged}` by Lawson and Hanson's algorithm (ch. 23), its least-squares subproblems by calx's QR; `nnls_columns(A, B)`.

### 2.4 `multivariate/nmf.hpp`, the module's first content

- **Factorisation.** `nmf`, `nmf_from`, `nmf_objective` and `normalise_factors`: Lee and Seung's multiplicative updates for the Frobenius norm and the generalised Kullback–Leibler divergence, with seeded restarts.
- **The stability of a rank.** `nmf_stability` and `nmf_rank_stability`:
  - replicates of V resampled (multinomially within columns, Alexandrov et al. 2013; or Poisson, Islam et al. 2022) and optionally normalised (log2, 100X);
  - their signatures clustered with one of each replicate in each cluster, by the assignment of greatest cosine similarity, until the average silhouette settles;
  - each cluster's stability as its mean silhouette under the cosine distance, the consensus signatures, exposures by NNLS, and the reconstruction error.
- **Helpers.** `cosine_similarity` and `silhouette`.

### 2.4a `multivariate/binomial_mixture.hpp`

Written in the same round for the genomics domain's clones of cancer cell fractions: `fit_binomial_mixture(k, n, c, options)` → `BinomialMixtureFit`, a finite mixture P(kᵢ) = Σ_j π_j Bin(kᵢ; nᵢ, cᵢθ_j) with known scales cᵢ. It is fitted by expectation–maximisation (Dempster, Laird and Rubin 1977): the M-step for θ_j is the root of a strictly concave score by calx's Brent, in closed form for equal scales, with θ ≤ θ_max. Quantile and seeded random starts are tried and the best likelihood kept. `select_binomial_mixture(k, n, c, K_max, options)` chooses K by BIC = −2ℓ + (2K − 1) log N. It is checked by `check_tyche_binomial_mixture` (12 checks).

### 2.5 Departures from the brief

- **Weights in CBS** and the hybrid tail approximation of Venkatraman and Olshen (2007) are not written. The approximation's formula could not be read from an open source and checked, and the exact search with curtailment and early rejection is fast enough (§4).
- **The CBS early-rejection boundary** is this record's construction with a proved error bound, not Venkatraman and Olshen's numbers.
- **The clustering of NMF replicates** starts from the replicate of least objective rather than random centroids, for determinism.
- **SigProfilerExtractor's Gaussian-mixture normalisation and its Wilcoxon choice of rank** are left to the caller and planned (they need `testing`'s rank tests).

## 3. Alternatives Considered

- **Change points in a domain (genomics).** Rejected: segmentation names no object in nature, and the domains of the economy and the street need it as much (CLAUDE.md of the infrastructure, §2.1).
- **CBS by DNAcopy's code.** Rejected: DNAcopy is GPL-licensed, and tyche is MIT. The method is written from the papers. DNAcopy itself could not be run here (no R), so no fixture comes from it; the checks rest on brute force, invariants and simulated truth.
- **The assignment problem in tyche or morphe.** Rejected: an optimisation over permutations with real costs is domain-free numerics, so calx's `optimisation`; morphe keeps exact combinatorics.
- **Coordinate descent with covariance updates** (Friedman et al. 2010, §2.2). Deferred: the naive updates serve the sizes of the first uses (thousands of observations, thousands of features) and are the ones IRLS needs.

## 4. Consequences

- **`check_tyche_changepoint`.**
  - Costs: each model's −2 log-likelihood in closed form, and the cost inequality.
  - The robust variance.
  - Brute force over every segmentation of 300 short series (all models, BIC, MBIC and manual penalties, minimum lengths 1–3): optimal partitioning and PELT reach the least objective, with the same segmentation.
  - PELT equal to optimal partitioning on 200 series of up to 600, with fewer cost evaluations.
  - ruptures 1.1.10's PELT (l2, normal) and binary segmentation breakpoints on 40 cases (cross-check).
  - Planted changes recovered: means, variances, both, Poisson rates with exposures, joint series, WBS on short middle segments.
  - CBS: the branch-and-bound maximum and threshold decisions equal to brute force on 400 series; the pooled t; the boundary's levels and its exact added size; the null size of the first split; planted arcs, pruning, undoing and determinism.
  - The speed for n = 10⁵.
- **`check_tyche_penalized`.**
  - Closed forms: the orthonormal design's soft threshold, ridge's normal equations, λ = 0 against `ols` and `glm`.
  - λ_max, and the KKT conditions along twelve paths.
  - scikit-learn 1.9.1's Lasso, ElasticNet and LogisticRegression (cross-check).
  - Cross-validation's folds, its error recomputed by hand, and a sparse truth selected at λ_1se.
- **`check_tyche_nnls`.** KKT on 200 random problems; brute force over supports; least squares when unconstrained; scipy 1.18.1's `nnls` (cross-check).
- **`check_tyche_binomial_mixture`** (12): the binomial mixture's EM and its choice of K (the module's README).
- **`check_tyche_nmf`.** Monotone objectives (Lee and Seung's theorems); one update against the formulas; a planted separable factorisation recovered; silhouettes by hand; the stability of planted signatures by rank.
- **Fixtures.** `time_series/tests/make_changepoint_fixtures.py` (ruptures 1.1.10, numpy 2.5.4) and `regression/tests/make_penalized_fixtures.py` (scikit-learn 1.9.1, scipy 1.18.1).
- **Dependencies.** `regression` now uses calx `random`; `multivariate` uses `regression` and calx `linalg`, `optimisation`, `random`, `solvers` and `special_functions` (the last two for the binomial mixture).
- **Conventions.** Stated in the glossary, §3 and §4: costs on the −2 log-likelihood scale, penalties per change point, half-open segments, the elastic net's scaling, and the stability's distance.
