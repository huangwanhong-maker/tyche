# testing: Multiple Comparisons, Permutation Tests, Comparisons of Groups and Adaptive Testing

*Status: draft (2026-11-01): the family-wise error rate and the false discovery rate, Storey's q-values, permutation p-values and the beta approximation of permutation minima ([decision 0007](../development/decisions/0007-sum-of-single-effects-variance-components-and-multiple-testing.md)).* Comparisons of two groups, the bootstrap of clusters, online control of the false discovery rate, e-values and the reusable holdout since 2026-11-18 ([decision 0027](../development/decisions/0027-comparing-groups-and-testing-what-a-search-finds.md)). Parametric tests as objects, the local false discovery rate, power, e-processes and asynchronous online testing are planned.

## 1. Concepts

Many tests at once need an error rate for the family. The family-wise rate (the chance of any false rejection) is kept by Bonferroni's and Holm's adjustments under any dependence; the false discovery rate (the expected share of false rejections among rejections) by Benjamini and Hochberg's step-up, which is much more powerful when many hypotheses are false. Storey's q-values sharpen it by estimating the share π₀ of true nulls from the flat right part of the p-values' histogram: π̂₀(λ) = #{p ≥ λ}/(m(1 − λ)), λ chosen by a smoothing spline over a grid (R qvalue's default) or by the bootstrap's mean squared error.

A family of correlated tests (the variants near a gene) is summarised by its least p-value, whose null law depends on the correlation. Permutations give it exactly, at a resolution of 1/B; FastQTL fits a beta law to the permutation minima and reads the observed minimum's adjusted p from it, resolved far below 1/B, after learning the effective degrees of freedom on which the minima are Beta(1, b).

**Comparing groups.** A weighted mean's spread uses Kish's effective size (Σw)²/Σw², so that equal weights give the textbook s/√n. Two means are compared by Welch's test with Satterthwaite's degrees of freedom; a share against a level by the score test with Wilson's interval; two shares by Newcombe's hybrid score interval, the difference's bounds built from each share's Wilson interval.

**Testing what a search finds.** When hypotheses arrive one at a time (a search admitting what it finds), each must be decided before the next and never revised. Online procedures spend a level α_t on the t-th, rejecting when its p-value is at most α_t, and let later levels depend on the rejections so far: LOND, LORD++ and SAFFRON control the FDR under independence (LOND also under positive dependence), α-investing the marginal FDR, and e-LOND, on e-values, under any dependence. An e-value has expectation at most 1 under the null; by Markov's inequality P(E ≥ 1/α) ≤ α, and e-values combine by averaging under any dependence. A part of the data kept aside and consulted adaptively is fitted like training data unless it is asked through a mechanism such as Thresholdout, which answers the training value unless the two differ by more than a noisy threshold, then the holdout's with noise, at a unit of a budget.

## 2. Content

`multiple.hpp`:

| Name | Content |
|---|---|
| `bonferroni(p)`, `holm(p)`, `benjamini_hochberg(p)` | adjusted p-values, R's p.adjust |
| `pi0_estimate(p, Pi0Options{method (fixed, smoother, bootstrap), lambda, smooth_df, smooth_log})` → `Pi0{pi0, lambda, by_lambda, smoothed}` | Storey's π̂₀, as R qvalue's pi0est |
| `qvalues(p, options, pi0)` → `QValues{q, pi0}` | Storey's q-values (π₀ = 1: Benjamini–Hochberg) |

`permutation.hpp`:

| Name | Content |
|---|---|
| `permutation_p_value(b, B)` | (b + 1)/(B + 1) |
| `correlation_p_value(r², ν)` | the two-sided p of t = √(ν r²/(1 − r²)), I_{1−r²}(ν/2, ½) |
| `fit_beta(p)` → `BetaFit{a, b, log_likelihood, moment_a, moment_b, iterations, converged, cdf(p)}` | the beta law by maximum likelihood (Newton, trigamma Hessian) |
| `effective_df(r², ν₀)` | FastQTL's effective degrees of freedom (the moment estimate of a equal to 1) |

The smoothing spline the smoother needs is `regression/smoothing_spline.hpp`.

`two_sample.hpp`: `weighted_mean(x, w)` → `WeightedMean{mean, sd, se, total, effective, rows}`; `welch(a, b)`, `against_level(a, level)`, `share_against_level(k, n, level)`, `share_difference(k₁, n₁, k₂, n₂)` → `Comparison{difference, se, statistic, df, p_value, low, high}`.

`bootstrap.hpp`: `bootstrap(units, statistic(multiplicity), resamples, stream, confidence, clusters)` → `Bootstrap{values, mean, se, low, high, failed}`.

`online.hpp`: `Spending::lord()`, `Spending::power(s)`; `Online({rule, alpha, initial, lambda, payout})` with `next_level()`, `test(value)` → `OnlineDecision{level, rejected, wealth, candidate}`, `run(values)`; rules `lond`, `lord++`, `saffron`, `alpha-investing`, `e-lond`.

`evalues.hpp`: `calibrate(p, κ)`, `calibrate_mixture(p)`, `e_to_p(e)`, `e_mean`, `e_product`, `e_bh(e, α)`.

`holdout.hpp`: `laplace(stream, b)`; `Thresholdout({threshold, sigma, budget}, stream)` with `ask(train, holdout)` → `HoldoutAnswer{value, overfit, spent}`, `saved()` and resumption.

## 3. Checks

- `check_tyche_multiple` (28): Bonferroni, Holm and BH against R 4.4.3's p.adjust on 1000 seeded p-values; R qvalue 2.38.0's π̂₀(λ), its smoothed π̂₀ and q-values (to 10⁻⁶, the effect of R's spline penalty weight), the bootstrap's and λ = ½'s exactly; the smoothing spline against an independent exact B-spline solution (10⁻⁹) and against R's smooth.spline (whose `sgram` uses 0.3330 for 1/3: the B-spline solution with that weight is R's to 4·10⁻¹¹); the spline's limits (df → 2, n) and lines reproduced; BH's FDR π₀q and q-values' FDR q by simulation.
- `check_tyche_permutation` (17): the permutation p-value's exact size; the correlation's p and Student's tail against SciPy 1.18.1 down to 10⁻¹⁷³; the beta fit against scipy.stats.beta.fit to 10⁻¹³ and Beta(1, 50) for minima of independent uniforms; the effective df against tensorQTL's df_cost solved by SciPy; the beta approximation against 20 000 direct permutations (8 signal strengths) and uniform under the null.

- `check_tyche_two_sample`: the weighted mean, Welch's test and the one-sample test against SciPy; Newcombe's interval, the pooled z and the score test against statsmodels; the bootstrap; refusals.
- `check_tyche_adaptive` (52): the spending sequences' normalisers against mpmath's ζ(1.6) and the sum of Javanmard and Montanari's sequence (partial sums and the exact tail); each rule's levels by hand; the FDR of each rule at α = 0.1 by 2000 replications (e-LOND on equicorrelated tests), α-investing's marginal FDR, SAFFRON and LORD++ more powerful than LOND; the calibrators against mpmath; e-BH by hand and as BH on 1/e; Thresholdout without noise, its budget and resumption, the Laplace draws, and Dwork et al.'s demonstration (the plain holdout rates a classifier of noise at 0.629, Thresholdout at 0.516).

## 4. Dependencies

`probability` (the beta, Student's t, the normal), `regression` (the smoothing spline); calx `special_functions` (incomplete beta, digamma, trigamma) and `solvers` (Brent's roots).

## 5. References

- Benjamini, Y. and Hochberg, Y. (1995), Controlling the false discovery rate, *JRSS B* 57, 289–300.
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. and Roth, A. (2015), The reusable holdout, *Science* 349, 636–638.
- Foster, D. P. and Stine, R. A. (2008), α-investing, *JRSS B* 70, 429–444.
- Javanmard, A. and Montanari, A. (2018), Online rules for control of false discovery rate and false discovery exceedance, *Ann. Statist.* 46, 526–554.
- Kish, L. (1965), *Survey Sampling*, Wiley.
- Newcombe, R. G. (1998), Interval estimation for the difference between independent proportions, *Statist. Med.* 17, 873–890.
- Ramdas, A., Yang, F., Wainwright, M. J. and Jordan, M. I. (2017), Online control of the false discovery rate with decaying memory, *NeurIPS*.
- Ramdas, A., Zrnic, T., Wainwright, M. and Jordan, M. I. (2018), SAFFRON, *ICML*, PMLR 80, 4286–4294.
- Vovk, V. and Wang, R. (2021), E-values: calibration, combination and applications, *Ann. Statist.* 49, 1736–1754.
- Wang, R. and Ramdas, A. (2022), False discovery rate control with e-values, *JRSS B* 84, 822–852.
- Welch, B. L. (1947), The generalization of Student's problem when several different population variances are involved, *Biometrika* 34, 28–35.
- Xu, Z. and Ramdas, A. (2024), Online multiple testing with e-values, *AISTATS*, PMLR 238, 3997–4005.
- Holm, S. (1979), A simple sequentially rejective multiple test procedure, *Scand. J. Statist.* 6, 65–70.
- Ongen, H., Buil, A., Brown, A. A., Dermitzakis, E. T. and Delaneau, O. (2016), Fast and efficient QTL mapper for thousands of molecular phenotypes, *Bioinformatics* 32, 1479–1485.
- Phipson, B. and Smyth, G. K. (2010), Permutation p-values should never be zero, *Stat. Appl. Genet. Mol. Biol.* 9, art. 39.
- Storey, J. D. (2002), A direct approach to false discovery rates, *JRSS B* 64, 479–498.
- Storey, J. D. and Tibshirani, R. (2003), Statistical significance for genomewide studies, *PNAS* 100, 9440–9445.
- Storey, J. D., Taylor, J. E. and Siegmund, D. (2004), Strong control, conservative point estimation and simultaneous conservative consistency of false discovery rates, *JRSS B* 66, 187–205.
- Taylor-Weiner, A. et al. (2019), Scaling computational genomics to millions of individuals with GPUs, *Genome Biology* 20, 228 (tensorQTL).

## 6. Open Questions

1. The local false discovery rate (qvalue's lfdr) needs a density estimate on the probit scale.
2. A smoothing spline in O(n) (banded Reinsch) for long grids.
