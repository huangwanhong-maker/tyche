# spatial: Random Fields and Their Statistics

*Status: draft (2026-10-29): Gaussian conditioning, extended one observation at a time ([decision 0004](../development/decisions/0004-gaussian-conditioning.md)); (2026-10-31) the co-location quotient of marked points and Clark and Evans's ratio ([decision 0005](../development/decisions/0005-spectral-inference-harmonics-and-point-patterns.md)).* Variograms and their fitting, kriging with an unknown mean (ordinary and universal), Ripley's K and pair correlations, areal models and statistics on the sphere are planned.

## 1. Concepts

A random field is known in part when some functionals of it have been observed: its values at points, components of its gradient, averages over regions. If the field is Gaussian with a known mean and covariance and the observations are those functionals plus independent Gaussian noise, the law of every other functional given the observations is Gaussian too, with the mean and variance of the conditioning formulas (Rasmussen and Williams 2006, eqs. 2.23–2.24). Under three names they are one computation: simple kriging in geostatistics (Cressie 1993, §3.4.1), Gaussian-process regression in machine learning, and the analysis step of optimal interpolation in data assimilation (Kalnay 2003, §5.4).

The posterior mean is the best estimate in mean square; the posterior variance says how well the observations determine each functional. When the model is the truth, the errors over their posterior standard deviations are standard normal, and the innovations, whitened by the Cholesky factor of their covariance, are independent standard normals: the model can be tested against the observations it was conditioned on.

The module names no field. The covariances come from the caller: physis `analysis` gives those of a random potential field from its spectrum, and any kernel will do.

A point pattern is a set of places where events or things are, possibly marked by kind. Its null is complete spatial randomness (a Poisson process), against which Clark and Evans's ratio of the mean nearest-neighbour distance to its expectation says whether the points cluster (R < 1) or keep apart (R > 1); near the edge of a bounded region the expectation must be corrected. For marked points the question is relative: whether kind B gathers about kind A more than B's share of all points would have it, which the co-location quotient measures, and which permuting the marks over the fixed places tests. Permuting only within strata (parts of the region that share a terrain) keeps each stratum's composition, so that what remains is gathering beyond the terrain.

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

`point_patterns.hpp`:

| Name | Content |
|---|---|
| `colocation(x, y, mark, marks, stratum, radius, permutations, seed)` | `Colocation{marks, clq, null_mean, null_sd, p_upper, p_lower, count, permutations}`: Leslie and Kronenfeld's CLQ_{A→B} by nearest neighbours (ties shared) or within a radius, with a permutation null of the marks, within strata when given |
| `clark_evans(x, y, x0, x1, y0, y1)` | `ClarkEvans{ratio, expected, observed, z, p}`: R = r̄ / E r̄ with Donnelly's edge correction for the rectangle |

## 3. Checks

`check_tyche_spatial` (13): posterior means, variances, covariances, weights and the log likelihood against the batch formulas by one dense factorisation, and the same from whitened covariances; the same posterior and likelihood whatever the order of the observations; one observation in closed form; an observation without noise fixing the query; and calibration over 2000 fields drawn from the prior: held-out errors of mean square 1, ±1.96 sd covering 95 %, whitened innovations of mean square 1.

`check_tyche_point_patterns` (24): the co-location quotient by hand (six points, a tie shared; within a radius; translated); Σ_B CLQ_{A→B} N'_B/(N − 1) = 1; under random labelling the null's mean 1 and the upper p-value's size 0.05 (400 labellings); attraction found; a pattern whose halves differ in composition significant against random labelling and not against the stratified null; Clark and Evans on a square lattice (R = 2 uncorrected, Donnelly's in closed form) and a hexagonal one (2.1491), on Poisson points (mean R = 1, size 0.05 over 1000 patterns) and on a Thomas process.

## 4. Dependencies

calx `random` (the permutations of `point_patterns`); conditioning needs nothing at run time (std::span and std::vector); the checks use calx `linalg` (Cholesky) and `random`.

## 5. References

- Besag, J. and Diggle, P. J. (1977), Simple Monte Carlo tests for spatial pattern, *Appl. Statist.* 26, 327–333.
- Clark, P. J. and Evans, F. C. (1954), Distance to nearest neighbor as a measure of spatial relationships in populations, *Ecology* 35, 445–453.
- Cressie, N. A. C. (1993), *Statistics for Spatial Data*, rev. ed., Wiley.
- Donnelly, K. P. (1978), Simulations to determine the variance and edge effect of total nearest neighbour distance, in I. Hodder (ed.), *Simulation Studies in Archaeology*, Cambridge University Press, 91–95.
- Higham, N. J. (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM.
- Kalnay, E. (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, Cambridge University Press.
- Leslie, T. F. and Kronenfeld, B. J. (2011), The colocation quotient: a new measure of spatial association between categorical subsets of points, *Geogr. Anal.* 43, 306–326.
- Rasmussen, C. E. and Williams, C. K. I. (2006), *Gaussian Processes for Machine Learning*, MIT Press.

## 6. Open Questions

1. An unknown mean (ordinary and universal kriging), and the estimation of a covariance's parameters by the marginal likelihood, which `log_likelihood` already gives.
2. Observations removed as well as added (a sliding window), by the downdating of the Cholesky factor.
3. Many observations: sparse and low-rank approximations, and the place of `assimilation`'s Kalman filters beside them.
4. Point patterns in regions other than rectangles (edge corrections of Ripley's K), on networks (holdings along streets, where distance is along the street), and the local co-location quotient of each point.
