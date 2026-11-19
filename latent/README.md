# Latent

Status: **draft** (2026-11-15). Module `tyche::latent`, [decision 0025](../development/decisions/0025-mixed-and-latent-variable-models.md); ideal points from votes since 2026-11-19, [decision 0030](../development/decisions/0030-ideal-points-from-votes.md).

## 1. Concepts

- **Factor analysis.** Factor analysis by maximum likelihood, Σ = ΛΛᵀ + Ψ, fitted by Rubin and Thayer's EM (Lawley and Maxwell 1971). Its discrepancy never rises. Heywood cases are reported, and the likelihood-ratio test carries Bartlett's correction. Varimax rotation with Kaiser's normalisation leaves the fit unchanged.
- **Latent classes.** Latent class analysis of binary items by EM from several seeded starts, with posteriors and BIC (Lazarsfeld and Henry 1968; Goodman 1974).
- **Ideal points.** Members placed on one line from their votes by the probit model of roll calls, α_j + β_j x_i plus a normal error (Clinton, Jackman and Rivers 2004). Its posterior mode is found by EM (Imai, Lo and Olmsted 2016), accelerated by SQUAREM with the log posterior kept from falling (Varadhan and Roland 2008). Reflection is fixed by an anchor, and the far tail is computed by Laplace's continued fraction. Intervals come from a parametric bootstrap, each replicate put on the fit's location and scale (Lewis and Poole 2004).

## 2. Content

- `latent.hpp`:
  - factor analysis: `FactorFit`, `discrepancy`, `factor_analysis`, `varimax_criterion`, `varimax`;
  - latent classes: `ClassFit`, `latent_classes`.
- `ideal_points.hpp`: `Votes`, `IdealPointOptions`, `IdealPoints`, `log_posterior`, `ideal_points`, `standardised`, `bootstrap`.

## 3. Checks

`check_tyche_latent` (18):
- a covariance two factors generate exactly, recovered with a discrepancy of zero and the EM monotone;
- varimax finding a simple structure a rotation hid, ΛΛᵀ unchanged;
- on a sample, the true number of factors not rejected and one fewer rejected;
- one latent class equal to independence;
- three classes recovered from 3,000 simulated persons, with monotone EM, posteriors summing to one and BIC preferring three.

`check_tyche_ideal_points` (15):
- recovery on 120 simulated members and 400 items with a fifth of the votes missing (r = 0.996), the line classifying as well as the true one;
- EM monotone, and the posterior mode reached again by BFGS with the analytic gradient;
- reflection and standardisation invariant; the far tail continuous and finite;
- the bootstrap: more votes narrower, a member with a fifth of its votes √5 wider, the truth covered for 30 of 40 at 90%.

## 4. Not Yet

Oblique rotations; factor scores; ideal points in several dimensions, from ordinal votes, or moving over time; confirmatory factor analysis and structural equation models; latent classes with polytomous items and covariates; mixtures of continuous variables.

## 5. References

- Lawley, D. N. and Maxwell, A. E. (1971), *Factor Analysis as a Statistical Method*, 2nd ed.
- Rubin, D. B. and Thayer, D. T. (1982), EM algorithms for ML factor analysis, *Psychometrika* 47.
- Kaiser, H. F. (1958), The varimax criterion for analytic rotation in factor analysis, *Psychometrika* 23.
- Clinton, J., Jackman, S. and Rivers, D. (2004), The statistical analysis of roll call data, *American Political Science Review* 98(2).
- Imai, K., Lo, J. and Olmsted, J. (2016), Fast estimation of ideal points with massive data, *American Political Science Review* 110(4).
- Varadhan, R. and Roland, C. (2008), Simple and globally convergent methods for accelerating the convergence of any EM algorithm, *Scandinavian Journal of Statistics* 35.
- Lewis, J. B. and Poole, K. T. (2004), Measuring bias and uncertainty in ideal point estimates via the parametric bootstrap, *Political Analysis* 12(2).
- Poole, K. T. and Rosenthal, H. (1997), *Congress: A Political-Economic History of Roll Call Voting*, Oxford University Press.
- Goodman, L. A. (1974), Exploratory latent structure analysis using both identifiable and unidentifiable models, *Biometrika* 61.
