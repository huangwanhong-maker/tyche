# CLAUDE.md

This file guides Claude Code in the tyche repository.

tyche is one of the foundation libraries of [The Science Infrastructures](https://github.com/huangwanhong-maker/The-Science-Infrastructures), which includes it as the git submodule `foundations/tyche`. It is a project in its own right and must make sense without the infrastructure.

## 1. Purpose and Scope

tyche implements the mathematics of chance (Mathematics Subject Classification 60, 62 and the statistical parts of 68 and 91):

- probability: distributions as objects, transformations, limit theorems;
- information: entropy, divergences, information geometry;
- stochastic processes and their statistics;
- descriptive statistics;
- estimation and testing;
- regression: linear, generalised linear and mixed models;
- multivariate analysis;
- time series;
- spatial statistics;
- Bayesian inference;
- data assimilation and the uncertainty of models;
- causal inference;
- the design of experiments and surveys;
- statistical learning.

It builds on three sibling libraries:

- **calx**, for numerical methods: special functions, linear algebra, optimisation, roots, random streams;
- **morphe**, for exact and combinatorial structures;
- **physis**, for dynamical systems and physical models whose statistics are tyche's.

Out of scope:

- **Particular datasets and their meaning.** These belong to the domains of The Science Infrastructures. tyche's fixtures are reference data for checks, not knowledge.
- **The methods of the social sciences as such** (measurement, surveys, econometric designs, games and social choice). These belong to nomos, which is built on tyche.
- **Domain-free numerics.** A method that names no statistical object (least squares, a minimiser, an eigenvalue) belongs to calx; tyche uses it.

## 2. Invariants

1. **Dependencies.** tyche depends only on calx, morphe and physis among project libraries, and never on the infrastructure's domains, edges or works.
2. **No particular data.** Datasets are inputs; reference data live only in `tests/fixtures/`, with their source and retrieval date.
3. **Stable numerics.** Least squares by orthogonal factorisations; tail probabilities computed directly; sums compensated where cancellation threatens.
4. **Cited sources.** Every implemented estimator, test or distribution cites its source, with equation, table or section numbers, and states its accuracy and range of validity.
5. **Conventions stated once.** Signs (of MA polynomials), divisors (of variances and autocovariances), the counting of parameters in information criteria and the types of sample quantiles are stated in `docs/glossary.md` and followed everywhere.

## 3. Design Philosophy

tyche shares its governing principle with its sibling repositories: **modular, extensible, revisable**.

- **Status vocabulary:**
  - `planned`: named, no content yet;
  - `seed`: an idea, possibly wrong;
  - `draft`: being worked out;
  - `stable`: tested and relied upon;
  - `deprecated`: superseded, with a pointer to its replacement.
- **References before cross-checks.** A method is checked against certified or published values and closed forms, and by the recovery of known parameters from seeded simulations; agreement with another package (statsmodels, R) is a cross-check, labelled as such, never the only reference.
- **Plain interfaces.** `std::span<const double>`, `std::vector<double>`, `calx::linalg::Rectangular` (design matrices) and `calx::linalg::Matrix` (covariances) in; plain structs out.

## 4. Repository Layout

```
tyche/
├── probability/    distributions, p-values
├── descriptive/    moments, quantiles, weighted statistics, covariance
├── regression/     least squares and robust covariances; generalised linear models
├── time_series/    autocorrelation; AR and ARIMA; unit roots and cointegration; filters; VAR
├── spatial/        Gaussian conditioning (simple kriging), one observation at a time
├── docs/           glossary of conventions
└── development/    decisions · ideas · release notes
```

Each module directory is created when its first content is written; until then it is a `planned` row in the README. Each written module has `CMakeLists.txt` (the target `tyche::<module>`), `include/tyche/<module>/`, `tests/` (checks on calx's `verification`, with `tests/fixtures/` when they read reference data) and a `README.md`.

## 5. Working Procedures

### 5.1 Module Contract

Each module has:

- a `README.md` giving its concepts first, then its content, its checks, dependencies, references and open questions;
- public headers in `include/tyche/<module>/`, namespace `tyche::<module>`;
- the implementation in `src/` when it is compiled (header-only so far);
- checks in `tests/`, registered by `tyche_add_check`, which tells them where their fixtures are (`TYCHE_FIXTURES`).

### 5.2 Decisions and Ideas

- Decisions are numbered records in `development/decisions/`. Records are never rewritten; a later record supersedes an earlier one.
- Ideas are files named `development/ideas/YYYY-MM-DD-slug.md`, with status `seed`, `promoted` or `abandoned`.
- **Release notes.** At each commit, add to `development/release-notes/YYYY-MM-DD.md` (one file per day, one short section per commit) a concise summary in plain prose of what is new, what one can now do and what works better, not a list of changes (the git log keeps those); not between commits. This convention is shared with the sibling repositories.

### 5.3 Change Discipline

- Before restructuring or changing an invariant (§2), propose the change and wait for the user's agreement.
- Keep each change to one concept.

### 5.4 Shared Conventions

calx, morphe, physis, tyche and the infrastructure keep these conventions identical:

- writing style;
- status vocabulary;
- decision, idea and release-note conventions.

Change them in every repository of [The Science Infrastructures](https://github.com/huangwanhong-maker/The-Science-Infrastructures) together.

### 5.5 Working Inside the Infrastructure

When you edit tyche inside the infrastructure's `foundations/tyche`:

1. commit in tyche first;
2. then commit the updated pointer in the infrastructure.

When a change needs calx, morphe or physis to change as well, commit them in dependency order: calx and morphe first, then physis, then tyche.

## 6. Statistical and Mathematical Conventions

- **Notation follows the references cited**, with departures stated in `docs/glossary.md`: the MA polynomial θ(B) = 1 + θ₁B + … (R's and statsmodels' sign, opposite to Box and Jenkins's); variances with divisor n − ddof (ddof = 1 by default), autocovariances with divisor n; sample quantiles of Hyndman and Fan's type 7 by default; AIC = −2ℓ + 2k with k the coefficients estimated (σ² counted only where stated).
- **Accuracy is stated.** Each docstring states the accuracy of the method, its range of validity, and whether its inference is exact or asymptotic.
- **Rigour.** Write notes with the precision of a mathematics text: hypotheses, definitions distinguished from results, proofs given or cited.
- **Writing style.**
  - Headings are declarative, numbered and academic, never questions.
  - Concepts come before structure.
  - Label asides with *Rationale.*, *Remark.* or *Caveat.*
  - Use English throughout, with British spelling.

## 7. Technology

- **Language: C++20,** header-only targets `tyche::<module>` while the modules are templates and inline functions, following ADR 0003 of The Science Infrastructures.
- **Build:** CMake 3.20 or later; checks are programs on calx's `verification`, registered with CTest (`tyche_add_check`, a wrapper of `calx_add_check`); on the development machine the compiler is GCC in WSL.
- **Dependency resolution.** Inside the infrastructure, tyche uses the targets of `foundations/calx`, `foundations/morphe` and `foundations/physis`, adding them when built alone. Never nest submodules.
- **Licences.** tyche is MIT-licensed. Check every dependency before adopting it. Reference data keep their own terms (NIST's are in the public domain).

## 8. Open Decisions

1. **The boundary with nomos.** Econometric estimators that are general statistics (instrumental variables, panel estimators) could live in tyche's `regression` or in nomos; provisionally nomos, using tyche's least squares and covariances.
2. **Exact ARMA likelihood.** ARIMA is estimated by conditional sum of squares; the exact Gaussian likelihood needs a state-space form and the Kalman filter (`assimilation`, or `time_series` with physis's dynamics).
3. **Random variates.** calx samples the standard distributions; whether tyche's distribution objects gain `sample(stream)` members, delegating to calx, is open.
4. **Python bindings** (nanobind), shared with the siblings.

## 9. Current State

As of 2026-10-24, the first four modules are written and checked ([decision 0003](development/decisions/0003-the-first-modules.md)): `probability` (eight distributions and p-values), `descriptive` (moments, quantiles, weighted statistics, covariance and correlation, summaries), `regression` (OLS and WLS by pivoted QR with classical, HC0–HC3 and cluster-robust covariances and Wald tests; GLMs by IRLS for the gaussian, binomial (logit, probit) and Poisson families, with offsets and robust covariances) and `time_series` (autocorrelation, partial autocorrelation and Ljung–Box; AR by least squares and Yule–Walker; ARIMA by conditional sum of squares with forecasts; the ADF test with MacKinnon's p-values and critical values, KPSS, Engle–Granger; the Hodrick–Prescott filter; VARs with order selection, stability, impulse responses and forecasts). The methods they needed that name no statistical object (rectangular matrices, QR with pivoting, Cholesky and band factorisations, general eigenvalues, Nelder–Mead, BFGS, the simplex method) went to calx (calx decision 0008). In all, 8 check programs. On 2026-10-29 `spatial` gained Gaussian conditioning, extended one observation at a time, for the infrastructure's rover mission ([decision 0004](development/decisions/0004-gaussian-conditioning.md)): a ninth check program. The repository is published at `https://github.com/huangwanhong-maker/tyche`.
