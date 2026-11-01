# testing: Multiple Comparisons and Permutation Tests

*Status: draft (2026-11-01): the family-wise error rate and the false discovery rate, Storey's q-values, permutation p-values and the beta approximation of permutation minima ([decision 0007](../development/decisions/0007-sum-of-single-effects-variance-components-and-multiple-testing.md)).* Parametric tests as objects, the local false discovery rate, and power are planned.

## 1. Concepts

Many tests at once need an error rate for the family. The family-wise rate (the chance of any false rejection) is kept by Bonferroni's and Holm's adjustments under any dependence; the false discovery rate (the expected share of false rejections among rejections) by Benjamini and Hochberg's step-up, which is much more powerful when many hypotheses are false. Storey's q-values sharpen it by estimating the share π₀ of true nulls from the flat right part of the p-values' histogram: π̂₀(λ) = #{p ≥ λ}/(m(1 − λ)), λ chosen by a smoothing spline over a grid (R qvalue's default) or by the bootstrap's mean squared error.

A family of correlated tests (the variants near a gene) is summarised by its least p-value, whose null law depends on the correlation. Permutations give it exactly, at a resolution of 1/B; FastQTL fits a beta law to the permutation minima and reads the observed minimum's adjusted p from it, resolved far below 1/B, after learning the effective degrees of freedom on which the minima are Beta(1, b).

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

## 3. Checks

- `check_tyche_multiple` (28): Bonferroni, Holm and BH against R 4.4.3's p.adjust on 1000 seeded p-values; R qvalue 2.38.0's π̂₀(λ), its smoothed π̂₀ and q-values (to 10⁻⁶, the effect of R's spline penalty weight), the bootstrap's and λ = ½'s exactly; the smoothing spline against an independent exact B-spline solution (10⁻⁹) and against R's smooth.spline (whose `sgram` uses 0.3330 for 1/3: the B-spline solution with that weight is R's to 4·10⁻¹¹); the spline's limits (df → 2, n) and lines reproduced; BH's FDR π₀q and q-values' FDR q by simulation.
- `check_tyche_permutation` (17): the permutation p-value's exact size; the correlation's p and Student's tail against SciPy 1.18.1 down to 10⁻¹⁷³; the beta fit against scipy.stats.beta.fit to 10⁻¹³ and Beta(1, 50) for minima of independent uniforms; the effective df against tensorQTL's df_cost solved by SciPy; the beta approximation against 20 000 direct permutations (8 signal strengths) and uniform under the null.

## 4. Dependencies

`probability` (the beta, Student's t, the normal), `regression` (the smoothing spline); calx `special_functions` (incomplete beta, digamma, trigamma) and `solvers` (Brent's roots).

## 5. References

- Benjamini, Y. and Hochberg, Y. (1995), Controlling the false discovery rate, *JRSS B* 57, 289–300.
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
