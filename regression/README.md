# regression: Linear and Generalised Linear Models

*Status: draft (2026-10-24): least squares with robust covariances and Wald tests; generalised linear models by IRLS ([decision 0003](../development/decisions/0003-the-first-modules.md)).* Mixed models, quantile regression, nonparametric and semiparametric regression, ridge and lasso, more GLM families (gamma, negative binomial, inverse Gaussian) and links (complementary log-log), and two-way clustering are planned.

## 1. Concepts

A linear model explains a response y by regressors X through E(y | X) = Xβ. Least squares estimates β; its uncertainty depends on what is assumed of the errors. If they are independent with one variance, the classical covariance σ²(XᵀX)⁻¹ is right, and the t and F statistics are exact under normality. If their variance differs from one observation to the next, White's sandwich (XᵀX)⁻¹(Σ eᵢ²xᵢxᵢᵀ)(XᵀX)⁻¹ is consistent whatever the form of the heteroskedasticity, and its refinements HC1–HC3 correct its downward bias in small samples. If errors are correlated within groups (firms, countries, years), the cluster-robust covariance sums the scores within each group first, and its inference rests on the number of groups, not of observations.

A generalised linear model replaces the identity by a link: g(E(y | X)) = Xβ, with the variance of y a function of its mean (binomial, Poisson). Its maximum likelihood estimate is found by iteratively reweighted least squares: each step is a weighted regression of a linearised response. The Poisson model with a log link has a remarkable robustness: its estimator is consistent whenever the mean is exp(Xβ), whatever the distribution of y, which makes it, with a robust covariance, the estimator of gravity equations of trade (PPML).

Numerically, every fit here goes through Householder QR with column pivoting, never the normal equations; columns are scaled to unit norm first, so that the rank decision does not depend on units.

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

## 4. Dependencies

`probability` (t, F, χ², the normal); calx `linalg` (`Rectangular`, `qr`, `cholesky`) and `special_functions`.

## 5. References

- Cameron, A. C. and Miller, D. L. (2015), A practitioner's guide to cluster-robust inference, *J. Human Resources* 50, 317–372.
- Dobson, A. J. (1990), *An Introduction to Generalized Linear Models*, Chapman and Hall; Dobson, A. J. and Barnett, A. G. (2008), 3rd ed., CRC.
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
