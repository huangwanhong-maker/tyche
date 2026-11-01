# regression: Linear and Generalised Linear Models

*Status: draft (2026-10-24): least squares with robust covariances and Wald tests; generalised linear models by IRLS ([decision 0003](../development/decisions/0003-the-first-modules.md)); (2026-11-01) ridge, the lasso and the elastic net along paths with cross-validation, and non-negative least squares ([decision 0006](../development/decisions/0006-change-points-penalties-and-factorisations.md)); the sum of single effects, variance components and smoothing splines ([decision 0007](../development/decisions/0007-sum-of-single-effects-variance-components-and-multiple-testing.md)).* Mixed models, quantile regression, nonparametric and semiparametric regression, penalised Poisson and multinomial models, more GLM families (gamma, negative binomial, inverse Gaussian) and links (complementary log-log), and two-way clustering are planned.

## 1. Concepts

A linear model explains a response y by regressors X through E(y | X) = Xβ. Least squares estimates β; its uncertainty depends on what is assumed of the errors. If they are independent with one variance, the classical covariance σ²(XᵀX)⁻¹ is right, and the t and F statistics are exact under normality. If their variance differs from one observation to the next, White's sandwich (XᵀX)⁻¹(Σ eᵢ²xᵢxᵢᵀ)(XᵀX)⁻¹ is consistent whatever the form of the heteroskedasticity, and its refinements HC1–HC3 correct its downward bias in small samples. If errors are correlated within groups (firms, countries, years), the cluster-robust covariance sums the scores within each group first, and its inference rests on the number of groups, not of observations.

A generalised linear model replaces the identity by a link: g(E(y | X)) = Xβ, with the variance of y a function of its mean (binomial, Poisson). Its maximum likelihood estimate is found by iteratively reweighted least squares: each step is a weighted regression of a linearised response. The Poisson model with a log link has a remarkable robustness: its estimator is consistent whenever the mean is exp(Xβ), whatever the distribution of y, which makes it, with a robust covariance, the estimator of gravity equations of trade (PPML).

Numerically, every fit here goes through Householder QR with column pivoting, never the normal equations; columns are scaled to unit norm first, so that the rank decision does not depend on units.

When there are many regressors, or more than observations, least squares overfits or is not defined, and a penalty on the coefficients trades a little bias for much less variance. Ridge's squared penalty shrinks correlated coefficients together; the lasso's absolute penalty sets many exactly to zero, selecting variables; the elastic net mixes the two, selecting groups of correlated variables together. Each is solved fastest by cyclic coordinate descent, one coefficient at a time by soft thresholding, along a decreasing path of penalties from the least at which all coefficients vanish, each solution warm-starting the next; the same holds for logistic regression inside iteratively reweighted least squares. The penalty's strength is chosen by cross-validation, and the one-standard-error rule takes the sparsest model whose error is within a standard error of the best. A different constraint, non-negativity, makes least squares a problem of which coefficients are active: Lawson and Hanson's algorithm moves coordinates in and out of the free set until the duals certify the optimum.

## 2. Content

`linear.hpp` (namespace `tyche::regression`):

| Name | Content |
|---|---|
| `design_matrix(columns, intercept)` | a `Rectangular` n × k from column vectors, with a leading column of ones when asked |
| `ols(X, y, OlsOptions)` → `LinearFit` | OLS or WLS (`weights`); `covariance`: `classical`, `hc0`, `hc1`, `hc2`, `hc3`, `cluster` (with `clusters`); `confidence`; `drop_collinear`; `rank_tolerance` |
| `LinearFit` | `n, k, rank, df_residual, df_inference, has_intercept, aliased, clusters`; `beta, se, t, p, ci_lower, ci_upper, covariance, xtx_inverse`; `fitted, residuals, leverage`; `rss, tss, r_squared, adj_r_squared, sigma`; `f_statistic, f_df1, f_df2, f_p_value`; `log_likelihood, aic, bic` |
| `wald(fit, R, r)` → `WaldTest` | H₀: Rβ = r with the fit's covariance: `f_statistic, chi2_statistic, df1, df2, f_p_value, chi2_p_value` |
| `predict(fit, X_new)` → `Prediction` | `mean, se, lower, upper` (the confidence interval of the mean) |

`glm.hpp`:

| Name | Content |
|---|---|
| `glm(X, y, GlmOptions)` → `GlmFit` | `family`: `gaussian`, `binomial`, `poisson`; `link`: `canonical`, `identity`, `logit`, `probit`, `log`; `offset`; `weights` (binomial: trials, y a proportion); `covariance`: `model`, `hc0`, `hc1`, `cluster`; `max_iterations, tolerance, beta_tolerance, drop_collinear` |
| `GlmFit` | `beta, se, z, p, ci_lower, ci_upper, covariance, xtx_inverse`; `fitted` (μ̂), `linear_predictor`, `residuals`, `pearson_residuals`, `deviance_residuals`, `working_weights`; `deviance, null_deviance, pearson_chi2, dispersion, log_likelihood, null_log_likelihood, aic, bic, pseudo_r_squared`; `iterations, converged, uses_t, df_inference` |

`penalized.hpp` ([decision 0006](../development/decisions/0006-change-points-penalties-and-factorisations.md)):

| Name | Content |
|---|---|
| `elastic_net_path(X, y, PenalizedOptions)` → `PenalizedPath` | glmnet's path: `PenalizedOptions{family (gaussian, binomial), alpha, lambda, lambda_count, lambda_min_ratio, standardize, intercept, weights, penalty_factor, offset, tolerance, max_passes}`; `PenalizedPath{family, alpha, lambda, fits, lambda_max, converged}` |
| `elastic_net(X, y, λ, o)`, `lasso(X, y, λ, o)`, `ridge(X, y, λ, o)` → `PenalizedFit` | one λ (ridge in closed form); `PenalizedFit{family, alpha, lambda, intercept, beta, df, deviance, null_deviance, deviance_ratio, objective, passes, converged}` |
| `predict(fit, X, offset, response)` | the linear predictor, or the probabilities |
| `penalized_kkt(X, y, fit, o)` | the largest violation of the KKT conditions, relative to λ |
| `cross_validate(X, y, o, CvOptions{folds, seed, fold, measure})` → `CrossValidation` | `path, mean_error, standard_error, index_min, index_1se, lambda_min, lambda_1se, fold, measure`; `CvMeasure{automatic, mse, deviance, misclassification}`; `make_folds(n, K, seed)` |

`nnls.hpp`:

| Name | Content |
|---|---|
| `nnls(A, b, NnlsOptions{max_iterations, tolerance})` → `NnlsResult` | min ‖Ax − b‖ subject to x ≥ 0 by Lawson and Hanson: `x, dual, residual_norm, iterations, converged` |
| `nnls_columns(A, B)` | one NNLS per column of B |

`susie.hpp` ([decision 0007](../development/decisions/0007-sum-of-single-effects-variance-components-and-multiple-testing.md)): the sum of single effects (Wang, Sarkar, Carbonetto and Stephens 2020), as susieR 0.14 computes it.

| Name | Content |
|---|---|
| `susie(X, y, SusieOptions, pool)` → `SusieFit` | IBSS on individual data (centred, standardised); `SusieOptions{effects, scaled_prior_variance, prior_weights, residual_variance, estimate_residual_variance, estimate_prior_variance (fixed, optimise, em), null_threshold, prior_weight_guard, prior_tolerance, tolerance, max_iterations, standardize, intercept, coverage, min_abs_corr}` |
| `susie_suff_stat(XᵀX, Xᵀy, yᵀy, n, o)`, `susie_rss(z, R, n, o)` | the same from sufficient statistics, and from z-scores and their correlation (PVE-adjusted, Zou et al. 2022) |
| `SusieFit` | `alpha, mu, mu2, lbf_variable` (L × p), `lbf, prior_variance, kl` (L), `residual_variance, elbo, iterations, converged, pip, coef, intercept, fitted, center, scale`, `sets` (`CredibleSet{effect, variables, coverage, min/mean/median_abs_corr, log_bayes_factor}`) |

`variance_components.hpp`: y = Xβ + Σ u_k + e, u_k ~ N(0, σ_k² K_k).

| Name | Content |
|---|---|
| `reml_spectral(X, y, K)` → `VarianceComponents` | one kernel: exact REML over h² by one eigendecomposition; `sigma2, se, covariance (AI⁻¹), h2, h2_se, beta, beta_se, log_likelihood, profile_h2, profile_log_likelihood` |
| `reml(X, y, kernels, RemlOptions{max_iterations, tolerance, constrain})` | AI-REML for several kernels, GCTA's start and first EM step |
| `haseman_elston(K, y, X)` → `HasemanElston` | the slope of y_i y_j on K_ij: `h2, intercept, se` (model-based), `se_jackknife` (conservative), `se_null`, `pairs` |

`smoothing_spline.hpp`:

| Name | Content |
|---|---|
| `smoothing_spline(t, y, df, weights)`, `smoothing_spline_lambda(t, y, λ, weights)` → `SmoothingSpline` | the natural cubic smoothing spline by Reinsch's equations: `knots, fitted, second, leverage, lambda, df`, and its value at any x |

A minimal use:

```cpp
#include <tyche/regression/linear.hpp>
using namespace tyche::regression;
const Rectangular X = design_matrix({x1, x2});          // 1, x1, x2
OlsOptions o;  o.covariance = Covariance::cluster;  o.clusters = firm;
const LinearFit f = ols(X, y, o);                         // f.beta, f.se, f.p, f.r_squared …
```

## 3. Checks

- `check_tyche_linear` (47): NIST's linear least-squares reference datasets, coefficients, standard errors, residual standard deviation, R² and F as log relative errors: Norris 12.3, Pontius 12.4, NoInt1 and NoInt2 14.8 and 15 (uncentred R²), Longley 11.3, Wampler1 9.6 (an exact fit), Wampler4 7.7, Filip 7.7 digits; Petersen's (2009) test panel: OLS, White (HC1) and firm- and year-clustered standard errors equal to his published values (0.0284, 0.0286; 0.0284, 0.0284; 0.0670, 0.0506; 0.0234, 0.0334); HC0 of a simple regression in closed form, HC0–HC3 of a mean, singleton clusters equal to HC1; a Wald F equal to t² and to the ANOVA F; integer weights equal to replication; collinear columns refused or aliased; heteroskedastic simulations within four HC3 standard errors and the coverage of 95% intervals over 400 replications; predictions.
- `check_tyche_glm` (31): Dobson's Poisson example as R prints it (β and standard errors in closed form for the independence model, deviance 5.1291, null deviance 10.5814, AIC 56.761) and its score equations; Bliss's beetles with the logit and probit links against Dobson and Barnett's printed estimates, standard errors and deviances, and against statsmodels to 10⁻⁸ (cross-check); the gaussian family equal to `ols` on Norris; PPML recovering a gravity equation under heteroskedastic multiplicative errors where least squares on logarithms is off by more than six standard errors (Santos Silva and Tenreyro 2006); offsets; singleton clusters equal to HC1.
- `check_tyche_penalized`: on an orthonormal design the lasso and elastic net equal the soft threshold S(xⱼᵀy/N, λα)/(1 + λ(1 − α)) (Friedman et al. 2010, eq. 5); ridge's closed form equals the normal equations' solution (weighted, standardised or not) and the coordinate descent at α = 0; at λ = 0 the lasso is `ols` and the binomial lasso `glm`; λ_max zeroes every penalised coefficient and 0.999 λ_max does not; the KKT conditions hold along twelve paths (gaussian and binomial, α = 1, 0.5, 0.05, weights, an unpenalised column, p > N); scikit-learn 1.9.1's Lasso, ElasticNet (with sample weights, on X and on X standardised) and LogisticRegression (saga, elastic net) on 52 fits (cross-check); cross-validation's balanced folds, determinism, cvm(λ_min) recomputed from ten refits by hand, the one-standard-error rule, a sparse truth (5 of 40 columns) selected at λ_1se, binomial deviance and misclassification.
- `check_tyche_susie` (62): susie, its EM variant, the single-effect regression, susie_suff_stat and susie_rss against susieR 0.14.2 on seeded genotype-like data (α, μ, μ², σ₀², σ², the ELBO at every iteration, PIPs, log Bayes factors, credible sets with purity and coverage, coefficients, fitted values); the SER against the enumeration of its one-column models; the ELBO never falls; over 300 simulated regions the 95% sets cover an effect (99.6%) and the PIPs are calibrated.
- `check_tyche_variance_components` (23): one kernel (spectral and AI-REML) and two kernels (AI-REML) against R's regress 1.3.22: components to 10⁻⁶–10⁻⁴, fixed effects, ℓ_R against R's evaluation, standard errors within 5%; the estimate the maximum of its profile; over 60 simulated samples REML and Haseman–Elston recover h² = 0.4 and their standard errors match their spread.
- `check_tyche_nnls`: the KKT conditions on 200 random problems, over- and underdetermined; the least residual over every non-negative support (brute force, 300 problems with n ≤ 7); least squares when the unconstrained solution is positive; scipy 1.18.1's `nnls` solutions and residual norms (cross-check); `nnls_columns`.

## 4. Dependencies

`probability` (t, F, χ², the normal); calx `linalg` (`Rectangular`, `qr`, `cholesky`), `special_functions` and `random` (the folds of cross-validation).

## 5. References

- Cameron, A. C. and Miller, D. L. (2015), A practitioner's guide to cluster-robust inference, *J. Human Resources* 50, 317–372.
- Dobson, A. J. (1990), *An Introduction to Generalized Linear Models*, Chapman and Hall; Dobson, A. J. and Barnett, A. G. (2008), 3rd ed., CRC.
- Friedman, J., Hastie, T. and Tibshirani, R. (2010), Regularization paths for generalized linear models via coordinate descent, *J. Stat. Softw.* 33(1), 1–22.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, §3.4 and §7.10.
- Hoerl, A. E. and Kennard, R. W. (1970), Ridge regression: biased estimation for nonorthogonal problems, *Technometrics* 12, 55–67.
- Lawson, C. L. and Hanson, R. J. (1974), *Solving Least Squares Problems*, Prentice-Hall, ch. 23.
- Tibshirani, R. (1996), Regression shrinkage and selection via the lasso, *JRSS B* 58, 267–288.
- Zou, H. and Hastie, T. (2005), Regularization and variable selection via the elastic net, *JRSS B* 67, 301–320.
- Golub, G. H. and Van Loan, C. F. (2013), *Matrix Computations*, 4th ed., Johns Hopkins, ch. 5.
- Greene, W. H. (2012), *Econometric Analysis*, 7th ed., Pearson.
- Long, J. S. and Ervin, L. H. (2000), Using heteroscedasticity consistent standard errors in the linear regression model, *Amer. Statist.* 54, 217–224.
- MacKinnon, J. G. and White, H. (1985), Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties, *J. Econometrics* 29, 305–325.
- McCullagh, P. and Nelder, J. A. (1989), *Generalized Linear Models*, 2nd ed., Chapman and Hall.
- NIST/ITL, *Statistical Reference Datasets: Linear Least Squares Regression*, https://www.itl.nist.gov/div898/strd/lls/lls.shtml (retrieved 2026-10-24).
- Petersen, M. A. (2009), Estimating standard errors in finance panel data sets: comparing approaches, *Rev. Financial Studies* 22, 435–480.
- Santos Silva, J. M. C. and Tenreyro, S. (2006), The log of gravity, *Rev. Econ. Stat.* 88, 641–658.
- White, H. (1980), A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity, *Econometrica* 48, 817–838.

## 6. Open Questions

1. Two-way clustering (Cameron, Gelbach and Miller 2011), wild cluster bootstrap for few clusters.
2. Iterative refinement of least-squares solutions for ill-conditioned problems such as Wampler4, where the residual is large and the error grows like κ²ε.
3. High-dimensional fixed effects (absorbed by alternating projections) for gravity equations; their place, tyche or nomos (CLAUDE.md §8, open decision 1).
4. Penalised regression: covariance updates for N ≫ p, strong rules for discarding columns (Tibshirani et al. 2012), sparse designs, the Poisson, multinomial and Cox families, and the group lasso.
