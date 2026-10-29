# spatial: Random Fields and Their Statistics

*Status: draft (2026-10-29): Gaussian conditioning, extended one observation at a time ([decision 0004](../development/decisions/0004-gaussian-conditioning.md)).* Variograms and their fitting, kriging with an unknown mean (ordinary and universal), point patterns, areal models and statistics on the sphere are planned.

## 1. Concepts

A random field is known in part when some functionals of it have been observed: its values at points, components of its gradient, averages over regions. If the field is Gaussian with a known mean and covariance and the observations are those functionals plus independent Gaussian noise, the law of every other functional given the observations is Gaussian too, with the mean and variance of the conditioning formulas (Rasmussen and Williams 2006, eqs. 2.23–2.24). Under three names they are one computation: simple kriging in geostatistics (Cressie 1993, §3.4.1), Gaussian-process regression in machine learning, and the analysis step of optimal interpolation in data assimilation (Kalnay 2003, §5.4).

The posterior mean is the best estimate in mean square; the posterior variance says how well the observations determine each functional. When the model is the truth, the errors over their posterior standard deviations are standard normal, and the innovations, whitened by the Cholesky factor of their covariance, are independent standard normals: the model can be tested against the observations it was conditioned on.

The module names no field. The covariances come from the caller: physis `analysis` gives those of a random potential field from its spectrum, and any kernel will do.

## 2. Content

`conditioning.hpp`:

| Name | Content |
|---|---|
| `GaussianConditioning::observe(c, variance, noise, innovation)` | one more observation, its covariances with the earlier ones: the Cholesky factor grown by a row, O(n²) |
| `predict(k, prior_mean, prior_variance)` | the posterior mean and variance of a query, O(n²) |
| `mean(k, prior_mean)`, `weights()` | the posterior mean alone by the weights C⁻¹ d, O(n) a query |
| `covariance(k, k′, prior)` | the posterior covariance of two queries |
| `whiten(k)` | L⁻¹ k, from which the means and covariances of several queries follow at one substitution each |
| `whitened()`, `log_likelihood()` | the whitened innovations L⁻¹ d; log p(y) |

## 3. Checks

`check_tyche_spatial` (13): posterior means, variances, covariances, weights and the log likelihood against the batch formulas by one dense factorisation, and the same from whitened covariances; the same posterior and likelihood whatever the order of the observations; one observation in closed form; an observation without noise fixing the query; and calibration over 2000 fields drawn from the prior: held-out errors of mean square 1, ±1.96 sd covering 95 %, whitened innovations of mean square 1.

## 4. Dependencies

None at run time (std::span and std::vector); its checks use calx `linalg` (Cholesky) and `random`.

## 5. References

- Cressie, N. A. C. (1993), *Statistics for Spatial Data*, rev. ed., Wiley.
- Higham, N. J. (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM.
- Kalnay, E. (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, Cambridge University Press.
- Rasmussen, C. E. and Williams, C. K. I. (2006), *Gaussian Processes for Machine Learning*, MIT Press.

## 6. Open Questions

1. An unknown mean (ordinary and universal kriging), and the estimation of a covariance's parameters by the marginal likelihood, which `log_likelihood` already gives.
2. Observations removed as well as added (a sliding window), by the downdating of the Cholesky factor.
3. Many observations: sparse and low-rank approximations, and the place of `assimilation`'s Kalman filters beside them.
