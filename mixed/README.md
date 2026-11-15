# Mixed

Status: **draft** (2026-11-15). Module `tyche::mixed`, [decision 0025](../development/decisions/0025-mixed-and-latent-variable-models.md).

## 1. Concepts

Linear mixed models: observations in groups, each group with random effects of its own, an unstructured covariance among them, by restricted or full maximum likelihood (Laird and Ware 1982).

- **Estimation.** The covariance is written σ²ΛΛᵀ. β and σ² are profiled out, and the relative Cholesky factor Λ is searched by Nelder–Mead (Bates et al. 2015). Woodbury's identity keeps every matrix q × q.
- **Prediction.** Groups' effects are predicted by their BLUPs, and the fixed effects carry standard errors. With a random intercept alone, the intraclass correlation is reported.

## 2. Content

- `mixed.hpp`: `Data`, `Options`, `Fit`, `criterion`, `fit`.

## 3. Checks

`check_tyche_mixed` (14):
- in the balanced one-way layout:
  - REML gives the ANOVA estimators, and ML its variant;
  - the intercept is the grand mean with its standard error;
  - the BLUPs are shrunk means;
- with random slopes in unequal groups:
  - components and fixed effects are recovered;
  - no step of θ lowers the criterion;
  - the profiled criterion equals the restricted likelihood computed from the whole covariance matrix.

## 4. Not Yet

Nested and crossed random effects; generalised linear mixed models (Laplace, adaptive quadrature); tests of variance components; Kenward–Roger degrees of freedom.

## 5. References

- Laird, N. M. and Ware, J. H. (1982), Random-effects models for longitudinal data, *Biometrics* 38.
- Bates, D., Mächler, M., Bolker, B. and Walker, S. (2015), Fitting linear mixed-effects models using lme4, *Journal of Statistical Software* 67(1).
- Harville, D. A. (1977), Maximum likelihood approaches to variance component estimation, *JASA* 72.
- Searle, S. R., Casella, G. and McCulloch, C. E. (1992), *Variance Components*, Wiley, ch. 3–4.
