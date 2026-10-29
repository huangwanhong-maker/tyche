# tyche

*Probability, statistics and inference.* The name comes from the Greek *τύχη*, "chance".

## 1. Purpose

tyche implements the mathematics of chance: probability distributions as objects, the description of samples, and the inference that goes from data to models (regression, time series, and in time estimation, testing, Bayesian computation, learning). It names no object in nature: a regression is a regression of y on X, whatever y measures.

It stands on its three sibling libraries:

- [calx](https://github.com/huangwanhong-maker/calx), for the numerical methods it computes with: special functions, linear algebra (QR, Cholesky, eigenvalues), optimisation, root finding and random streams;
- [morphe](https://github.com/huangwanhong-maker/morphe), for exact and combinatorial structures (graphs for causal models, words for sequences);
- [physis](https://github.com/huangwanhong-maker/physis), for dynamical systems, whose statistics (filtering, assimilation, the uncertainty of models) are tyche's.

tyche is one of the foundation libraries of [The Science Infrastructures](https://github.com/huangwanhong-maker/The-Science-Infrastructures). It stands on its own.

## 2. Modules

| Module | Content | Status |
|---|---|---|
| [`probability`](probability/) | Distributions as objects: densities, distribution and survival functions, quantiles, moments; p-values; transformations, mixtures and copulas; limit theorems | draft: normal, Student, χ², F, gamma, beta, Poisson, binomial; p-values |
| `information` | Entropy, mutual information, divergences; the information geometry of statistical models | planned |
| `processes` | Markov chains, point processes (Poisson, renewal, Hawkes), Gaussian processes and random fields, diffusions, random graphs | planned |
| [`descriptive`](descriptive/) | Summaries, moments, quantiles, weighted statistics, covariance and correlation; robust statistics, empirical distributions, kernel density estimation | draft: moments, the nine sample quantiles, weighted statistics, covariance and correlation matrices, summaries |
| `estimation` | Likelihood, M-estimators, the method of moments; the bootstrap; confidence sets | planned |
| `testing` | Parametric and permutation tests; multiple comparisons; power | planned |
| [`regression`](regression/) | Linear, generalised linear and mixed models; robust covariances; nonparametric regression; regularisation | draft: OLS and WLS by pivoted QR, classical, HC0–HC3 and cluster-robust covariances, Wald tests; GLMs by IRLS (gaussian, binomial logit and probit, Poisson with offsets: PPML) |
| `multivariate` | Principal components, factor analysis, canonical correlation; clustering and mixtures | planned |
| [`time_series`](time_series/) | Autoregressive and moving-average models; unit roots and cointegration; filters; vector autoregressions; spectral analysis; state-space models | draft: ACF, PACF, Ljung–Box; AR, ARIMA by CSS, forecasts; ADF with MacKinnon's p-values and critical values, KPSS, Engle–Granger; Hodrick–Prescott; VAR, order selection, stability, impulse responses, forecasts |
| [`spatial`](spatial/) | Geostatistics and kriging; point patterns; areal models; statistics on the sphere | draft: Gaussian conditioning (simple kriging), extended one observation at a time, with whitened innovations and the marginal likelihood |
| `bayes` | Conjugate models, MCMC (Metropolis–Hastings, Gibbs, Hamiltonian), sequential Monte Carlo, variational inference | planned |
| `assimilation` | Kalman filters, smoothers, variational and particle methods | planned |
| `uncertainty` | Propagation of uncertainty, sensitivity analysis, surrogates, calibration | planned |
| `causal` | Causal graphs, potential outcomes, identification and estimation of effects | planned |
| `design` | Design of experiments; survey sampling and weighting | planned |
| `learning` | Kernels, trees and ensembles, neural networks; validation | planned |

The order of the modules is that of the infrastructure's [map of the mathematics](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/docs/mathematics.md), §6: `probability` stands alone; `descriptive` uses it; `regression` uses `probability`; `time_series` uses `regression`, `descriptive` and `probability` (decision 0003).

## 3. Principles

- **Chance, not nature.** tyche knows distributions, samples and models; the domains give them meaning.
- **Stable numerics.** Least squares by orthogonal factorisations, never by normal equations; tails computed directly, never as one minus a probability near one; sums compensated.
- **Published references.** Every method is checked against certified or published values (NIST's Statistical Reference Datasets, textbooks' worked examples, published critical values) and closed forms, and against recovery of known parameters from seeded simulations; cross-checks against statsmodels are labelled as such and never the only reference.
- **Plain interfaces.** Functions take `std::span<const double>`, `std::vector<double>` and calx's matrices, and return plain structs, so that engines answering JSON can call them directly.

## 4. Relation to Sibling Libraries

tyche depends on calx, morphe and physis; none of them depends on tyche. The methods of the quantitative social sciences (econometrics, measurement, games and social choice) are a library above tyche, nomos.

- **Inside the infrastructure**, tyche uses the copies under `foundations/`.
- **Built on its own**, it would fetch them (not yet written).

## 5. Status

**Draft (2026-10-24).** The first four modules are written and checked ([decision 0003](development/decisions/0003-the-first-modules.md)): `probability`, `descriptive`, `regression` and `time_series`; the rest is planned.

Inside The Science Infrastructures, build and check with CMake 3.20 or later (calx, morphe and physis are taken from `../calx`, `../morphe` and `../physis`):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 6. License

tyche is released under the MIT License. See [`LICENSE`](LICENSE).
