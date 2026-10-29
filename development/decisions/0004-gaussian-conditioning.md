# 0004. Gaussian Conditioning, the First Content of `spatial`

- **Status:** Accepted, provisionally (a known mean; observations added, not removed; dense factors)
- **Date:** 2026-10-29
- **Decided by:** the user asked the infrastructure for embodied agents whose world's fields are "gaussed/assimilated" while the agents "provide us new info"; the first such mission, a rover mapping the Moon's crustal field in Cosmopoiesis, needed the posterior of a Gaussian field given its readings. Claude wrote the module and this record, as part of the infrastructure's [ADR 0054](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/development/decisions/0054-bodies-dynamics-and-knowledge.md), beside the work of tyche's founding (the infrastructure's ADR 0048, in progress).

## 1. Context

`spatial` was a planned row of the README ("geostatistics and kriging"). A mission learning a field from a body's readings conditions a Gaussian law on noisy linear functionals, one reading at a time, and asks the posterior at many points. It also asks whether its own model is consistent with what it has read.

## 2. Decision

`spatial/conditioning.hpp`: `GaussianConditioning` (the module's [README](../../spatial/README.md)).
- **The computation.** The Cholesky factor of the observations' covariance, grown by a row each observation, O(n²) each: the batch factorisation, the same to rounding.
- **What it answers:**
  - posterior means, variances and covariances of queries;
  - the weights C⁻¹ d, for means at O(n) a query;
  - `whiten`, L⁻¹ k, from which several queries' joint posterior follows at one substitution each;
  - the whitened innovations, a test of the model against its own data;
  - the log marginal likelihood.
- **What it names.** No field: the caller supplies the covariances (physis `analysis` gives a random potential field's).

## 3. Alternatives Considered

- **In physis** beside the covariance. Rejected: conditioning is statistics, and tyche stands on physis, not the reverse.
- **A Kalman filter in `assimilation`.** For a static field observed linearly, the filter's analysis step is this conditioning. `assimilation` will hold the dynamics between observations when a field evolves.
- **Low-rank or sparse approximations.** Not yet: the first mission has some thousand observations.

## 4. Consequences

- `check_tyche_spatial` (13): the batch formulas; order invariance; closed forms; calibration over 2000 seeded fields.
- Open: an unknown mean (ordinary and universal kriging), hyperparameters by the marginal likelihood, downdating, many observations.
