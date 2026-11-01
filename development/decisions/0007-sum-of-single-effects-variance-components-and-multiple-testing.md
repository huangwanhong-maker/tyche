# 0007. The Sum of Single Effects, Variance Components, Smoothing Splines and Multiple Testing

- **Status:** Accepted, provisionally (dense matrices; one Gaussian residual; SuSiE's variational posterior as susieR 0.14 computes it)
- **Date:** 2026-11-01
- **Decided by:** the infrastructure's evolution domain's module `association` (its ADR 0064: genotype to trait and to expression) needs statistics that name no object in nature: Bayesian variable selection summarised by credible sets (fine-mapping), restricted maximum likelihood of variance components and Haseman–Elston regression (heritability), the beta approximation of permutation minima and Storey's q-values (eQTL mapping). Claude wrote the methods and this record at the user's request ("implement traditional and advanced genome analysis algorithms, especially … expression prediction"); penalised regression, which PrediXcan also needs, is decision 0006's.

## 1. Context

`regression` held least squares and generalised linear models; mixed models and nonparametric regression were planned. `testing` was a planned module ("parametric and permutation tests; multiple comparisons; power"). The genetic analyses need, in general form:

- regression with a sparse, multi-effect posterior whose uncertainty is reported as sets of columns (SuSiE), on individual data and on sufficient or summary statistics;
- variance components with known kernels, by REML (exact for one kernel, AI-REML for several) and by a method of moments;
- the false discovery rate of many tests, and the gene-level adjustment of a minimum over correlated tests by permutations.

Working on them showed a defect of `probability`: Student's upper tail switched formulas at t² = ν and lost every digit of tails below about 10⁻¹⁶ for large ν (at ν = 3000, t = 10, the tail is 1.75·10⁻²³), the regime of every association scan.

## 2. Decision

### 2.1 `regression/susie.hpp`

- `susie(X, y, options, pool)` on individual data, `susie_suff_stat(XᵀX, Xᵀy, yᵀy, n, options)`, `susie_rss(z, R, n, options)` → `SusieFit` (α, μ, μ² of each effect, the effects' log Bayes factors and per-column ones, prior and residual variances, the ELBO by iteration, PIPs, coefficients, credible sets with their purity).
- IBSS on the sufficient statistics (one core); prior variances by Brent's maximisation on log σ₀² ∈ [−30, 15] or the EM step, with the null check; the convergence test, defaults and summaries are susieR 0.14's, including its guard √ε on the prior weights (an option; 0 gives the paper's formulas).
- SuSiE-RSS with PVE-adjusted z-scores (Zou et al. 2022).

### 2.2 `regression/variance_components.hpp`

- `reml_spectral(X, y, K)`: one kernel by its eigendecomposition, ℓ_R(h²) profiled in σ² and maximised exactly (a grid, then Brent); AI standard errors; the profile kept for drawing.
- `reml(X, y, kernels, options)`: AI-REML with GCTA's start and first EM step, step-halving and the non-negativity constraint.
- `haseman_elston(K, y, X)`: the slope of products on the kernel; its model-based standard error by the delta method on the Gaussian quadratic form (the standardisation's variance included); the jackknife's (conservative here, documented) and Visscher's null approximation.

### 2.3 `regression/smoothing_spline.hpp`

- The cubic smoothing spline by Reinsch's equations, given λ or the equivalent degrees of freedom; it serves Storey's smoother of π̂₀ and is nonparametric regression's first content.

### 2.4 `testing` (new module)

- `multiple.hpp`: Bonferroni, Holm, Benjamini–Hochberg; Storey's π̂₀ by a fixed λ, the smoother (qvalue's default) or the bootstrap's closed form; q-values.
- `permutation.hpp`: (b + 1)/(B + 1); the correlation's p-value on ν degrees of freedom; the beta law of permutation minima by maximum likelihood (Newton with the trigamma Hessian); FastQTL's effective degrees of freedom.

### 2.5 A correction to `probability`

- `StudentT::upper_half` uses ½ I_x(ν/2, ½) by the direct continued fraction wherever calx's incomplete beta would (the tail keeps its relative accuracy), and the complement with y = t²/(ν + t²) formed exactly otherwise.

### 2.6 Departures from the references

- **susieR's purity** is estimated on a random 100 members of a large set; here every pair is used, so results for sets above 100 can differ.
- **A constant column** gets log-weight 0 in susieR (its ŝ² is infinite); here its Bayes factor is 1 and its weight π_j, as the model has it.
- **R's smooth.spline** integrates the penalty with 0.3330 for 1/3 (its Fortran `sgram`), so qvalue's smoother differs from the exact spline by about 2·10⁻⁴ relative in the fit and 10⁻⁷ in π̂₀; this code is the exact spline (held to an independent B-spline solution).
- **The Haseman–Elston jackknife** overestimates the spread by about 1.6 at h² = 0.4 (the slope is nearly a degenerate U-statistic); the model-based error is the one to read.

## 3. Alternatives Considered

- **SuSiE, REML and q-values in the evolution domain.** Rejected: they name no object in nature (CLAUDE.md of the infrastructure, §2.1 "reuse before writing").
- **Matching susieR exactly by its quirks** (purity subsampling; constant columns). Rejected: the first is random, the second wrong; both stated.
- **R's smooth.spline reproduced to the digit** by its B-spline basis and 0.3330. Rejected: that would copy a defect; the exact spline is kept and the difference stated.
- **The local FDR of qvalue** (a density estimate). Deferred.

## 4. Consequences

- `check_tyche_multiple` (28): Bonferroni, Holm and BH equal R's p.adjust; qvalue 2.38.0's π̂₀(λ), smoothed π̂₀, q-values by its three methods; the spline against an independent B-spline solution to 10⁻⁹ and against R to R's own defect; limits of λ; FDR control by simulation.
- `check_tyche_permutation` (17): the permutation p-value's size; the correlation's p-value and Student's tail against SciPy to 10⁻¹⁷³; the beta fit against SciPy and Beta(1, m) for independent minima; the effective degrees of freedom against tensorQTL's formula by SciPy; the beta approximation against 20 000 direct permutations and its uniformity under the null.
- `check_tyche_susie` (62): five fits against susieR 0.14.2 (α, μ, μ², variances, ELBO by iteration, PIPs, log Bayes factors, credible sets with purity and coverage, coefficients); the SER against the enumeration of its models; the ELBO's monotonicity; coverage and PIP calibration over 300 simulated regions.
- `check_tyche_variance_components` (23): one and two kernels against R's regress 1.3.22 (components, fixed effects, ℓ_R by R's algebra, standard errors); AI-REML reaching the spectral optimum; recovery of a simulated h² over 60 samples by REML and Haseman–Elston, with calibrated standard errors.
- The infrastructure's evolution domain's `association` uses all of them.
