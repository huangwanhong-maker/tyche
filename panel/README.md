# panel: Linear Models of Panel Data

*Status: draft (2026-11-08), moved from nomos's `econometrics` ([decision 0014](../development/decisions/0014-panels-and-the-sandwich.md)).* Built on tyche's `regression`: every least-squares fit is `ols` (pivoted QR, never the normal equations).

## 1. Concepts

- **Absorbed fixed effects.** The effects of one or more factors (units, periods, origins, destinations) are removed by the theorem of Frisch, Waugh and Lovell: the coefficients on X of a regression on X and the factors' indicators are those of the regression of the residualised outcome on the residualised X, with the same residuals and the same sandwich. The residualising is done without forming the indicators, by alternating weighted projections (Guimarães and Portugal 2010; Gaure 2013), exact in one sweep for one factor. The parameters absorbed are L for one factor and L₁ + L₂ − (connected components) for two (Abowd, Creecy and Kramarz 2002).
- **Panel data.** The within estimator treats unit (and period) effects as parameters; its classical standard errors count the absorbed parameters in the degrees of freedom, and its cluster-robust ones count only those not nested in the clusters (Cameron and Miller 2015), as Stata's `xtreg` and `reghdfe` do. The between estimator regresses unit means. Random effects is feasible GLS on quasi-demeaned data with Swamy and Arora's (1972) variance components; in a balanced panel it is a matrix-weighted average of within and between (Maddala 1971). Hausman's (1978) test compares fixed and random effects.

## 2. Content

| Header | Content |
|---|---|
| `absorb.hpp` | `Factor{level, levels}`, `factor(labels)`; `AbsorbOptions{tolerance, max_sweeps}`; `demean(columns, factors, w, options)`; `components(a, b)`, `occurring(f)`, `absorbed_rank(factors)`, `nested(f, clusters)`; `recover_effects(d, factors, w, options)` |
| `panel.hpp` | `PanelEffects {unit, two_way}`, `PanelCovariance {classical, cluster}`, `PanelOptions`; `within(X, y, unit, period, options)` → `WithinFit{fit, unit_effects, period_effects, unit_labels, period_labels, absorbed, absorbed_not_nested, sweeps}`; `between(X, y, unit, ols_options)` → `tyche::regression::LinearFit`; `random_effects(X, y, unit, ols_options)` → `RandomEffectsFit{fit, sigma_u, sigma_e, rho, theta}`; `hausman(fe, re)` → `HausmanTest{statistic, df, p_value, positive_definite}` |

The within estimator's `fit` is a `tyche::regression::LinearFit` whose covariance, degrees of freedom and inference have been corrected for the absorbed parameters, so `wald` applies to it; its `r_squared` is the within R².

## 3. Checks

`tests/check_panel.cpp` (21):
- **One-way within** against the dummy-variable regression: the coefficients, classical errors, residuals and effects; clustered errors in the exact ratio √((n − k − N)/(n − k)).
- **Two-way, on an unbalanced panel:**
  - the coefficients, classical errors and fitted values, and the effects rebuilding them;
  - N + T − 1 parameters absorbed, T − 1 of them not nested in unit clusters;
  - the truth recovered, and a time-invariant regressor absorbed.
- **Random effects** equal to Maddala's weighted average, exactly; Swamy and Arora's components near the truth.
- **Hausman's test** accepting and rejecting.

## 4. Dependencies

tyche `regression` and `probability`; calx `linalg`.

## 5. References

- Abowd, J. M., Creecy, R. H. and Kramarz, F. (2002), Computing person and firm effects using linked longitudinal employer-employee data, US Census Bureau, Technical Paper 2002-06.
- Baltagi, B. H. (2013), *Econometric Analysis of Panel Data*, 5th ed., Wiley.
- Cameron, A. C. and Miller, D. L. (2015), A practitioner's guide to cluster-robust inference, *J. Hum. Resour.* 50, 317–372.
- Correia, S. (2016), Linear models with high-dimensional fixed effects: an efficient and feasible estimator, working paper (reghdfe).
- Frisch, R. and Waugh, F. V. (1933), Partial time regressions as compared with individual trends, *Econometrica* 1, 387–401; Lovell, M. C. (1963), Seasonal adjustment of economic time series and multiple regression analysis, *J. Am. Stat. Assoc.* 58, 993–1010.
- Gaure, S. (2013), OLS with multiple high dimensional category variables, *Comput. Stat. Data Anal.* 66, 8–18.
- Guimarães, P. and Portugal, P. (2010), A simple feasible procedure to fit models with high-dimensional fixed effects, *Stata J.* 10, 628–649.
- Hausman, J. A. (1978), Specification tests in econometrics, *Econometrica* 46, 1251–1271.
- Maddala, G. S. (1971), The use of variance components models in pooling cross section and time series data, *Econometrica* 39, 341–358.
- Swamy, P. A. V. B. and Arora, S. S. (1972), The exact finite sample properties of the estimators of coefficients in the error components regression models, *Econometrica* 40, 261–275.
- Wooldridge, J. M. (2010), *Econometric Analysis of Cross Section and Panel Data*, 2nd ed., MIT Press.
