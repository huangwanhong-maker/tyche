# Latent

Status: **draft** (2026-11-15). Module `tyche::latent`, [decision 0025](../development/decisions/0025-mixed-and-latent-variable-models.md).

## 1. Concepts

- **Factor analysis.** Factor analysis by maximum likelihood, Σ = ΛΛᵀ + Ψ, fitted by Rubin and Thayer's EM (Lawley and Maxwell 1971). Its discrepancy never rises. Heywood cases are reported, and the likelihood-ratio test carries Bartlett's correction. Varimax rotation with Kaiser's normalisation leaves the fit unchanged.
- **Latent classes.** Latent class analysis of binary items by EM from several seeded starts, with posteriors and BIC (Lazarsfeld and Henry 1968; Goodman 1974).

## 2. Content

- `latent.hpp`:
  - factor analysis: `FactorFit`, `discrepancy`, `factor_analysis`, `varimax_criterion`, `varimax`;
  - latent classes: `ClassFit`, `latent_classes`.

## 3. Checks

`check_tyche_latent` (18):
- a covariance two factors generate exactly, recovered with a discrepancy of zero and the EM monotone;
- varimax finding a simple structure a rotation hid, ΛΛᵀ unchanged;
- on a sample, the true number of factors not rejected and one fewer rejected;
- one latent class equal to independence;
- three classes recovered from 3,000 simulated persons, with monotone EM, posteriors summing to one and BIC preferring three.

## 4. Not Yet

Oblique rotations; factor scores; confirmatory factor analysis and structural equation models; latent classes with polytomous items and covariates; mixtures of continuous variables.

## 5. References

- Lawley, D. N. and Maxwell, A. E. (1971), *Factor Analysis as a Statistical Method*, 2nd ed.
- Rubin, D. B. and Thayer, D. T. (1982), EM algorithms for ML factor analysis, *Psychometrika* 47.
- Kaiser, H. F. (1958), The varimax criterion for analytic rotation in factor analysis, *Psychometrika* 23.
- Goodman, L. A. (1974), Exploratory latent structure analysis using both identifiable and unidentifiable models, *Biometrika* 61.
