# 0015. Spatial Regression

- **Status:** Accepted, provisionally (2026-11-09).
- **Decided by:** the infrastructure's ADR 0088 (the urban priorities' stage 4, demand), at the user's word ("continue the next feature").

## 1. Decision

`spatial` gains `regression.hpp`, models in which an outcome depends on its neighbours':
- **the spatial lag model** y = ρWy + Xβ + ε and **the spatial error model** y = Xβ + u, u = λWu + ε, by maximum likelihood (Ord 1975; Anselin 1988):
  - the likelihood is concentrated in the spatial parameter;
  - the Jacobian is Σ ln|1 − ρω| over the eigenvalues of W, complex where W is not symmetric;
  - the parameter is found by Brent's method;
  - the covariance is the inverse of the full information matrix of (β, the parameter, σ²);
- **the Lagrange multiplier tests**, plain and robust (Anselin, Bera, Florax and Yoon 1996), and SARMA;
- **the lag model's impacts**, direct, indirect and total (LeSage and Pace 2009).

## 2. Rationale

Hedonic prices, crime, growth and disease all carry spatial dependence. The models name no object in nature and estimate within a model, so they are tyche's, beside Moran's I.

## 3. Verification

`check_tyche_spatial_regression` (14) holds the models to spreg 1.9.1 (`spatial/tests/fixtures/make_regression_fixtures.py`):
- **Columbus** (Anselin's 49 neighbourhoods, queen contiguity):
  - the lag and error models' coefficients, parameters and log-likelihoods to 10⁻⁸, and their standard errors to 10⁻⁵;
  - the five LM statistics to 10⁻¹²;
  - the impacts against numpy's traces of (I − ρW)⁻¹.
- **Sixty points on five nearest neighbours,** an unsymmetric W with complex eigenvalues.
- **Recovery** of ρ = 0.6 on a lattice.

tyche's checks rise from 807 to 821.

## 4. Consequences

Dense matrices (eigenvalues, the inverse of I − ρW) bound n to a few thousand. Sparse log-determinants (Chebyshev, Barry and Pace) can follow if larger records ask.
