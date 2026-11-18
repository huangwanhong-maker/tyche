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
- statistical learning;
- statistical decision theory: decisions under uncertainty, in one step (under risk, ignorance and ambiguity, with decision trees and the value of information; decision 0024) and sequential and partially observed (POMDPs and their variants; decision 0009).

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
├── classes/        diversity of a law and its partition among units; richness, coverage, rarefaction and extrapolation from samples
├── compositions/   parts of a whole: Aitchison's geometry of the simplex, balances, the centre and variation, zeros, compositions in linear models
├── networks/       relational data: censuses, clustering, components and cores; assortativity and homophily; brokerage; QAP, MRQAP, conditional uniform graphs
├── regression/     least squares and robust covariances; generalised linear models; penalised regression; NNLS
├── mixed/          linear mixed models: random intercepts and slopes by profiled REML or ML, BLUPs
├── latent/         factor analysis by maximum likelihood, varimax; latent class analysis
├── multivariate/   non-negative matrix factorisation and the stability of its rank; dissimilarities, principal coordinates, PERMANOVA, Mantel
├── time_series/    autocorrelation; AR and ARIMA; unit roots and cointegration; filters; VAR; spectra; change points
├── spatial/        Gaussian conditioning (simple kriging), one observation at a time; change of support; estimates withheld outside their area of applicability
├── bayes/          conjugate models; Markov chain Monte Carlo (adaptive Metropolis, HMC) and its diagnostics
├── uncertainty/    designs; propagation of uncertainty; Morris's screening and Sobol' indices
├── graphical/      Bayesian networks over discrete variables: factors, exact elimination, likelihood ratios, the most probable explanation, belief propagation, EM
├── beliefs/        Dempster–Shafer belief functions: mass functions, transforms, rules of combination, discounting, the pignistic law, Choquet expectations
├── decision/       decisions under uncertainty: in one step (risk, ignorance, ambiguity), trees, the value of information; POMDPs and their kin
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

As of 2026-10-24, the first four modules are written and checked ([decision 0003](development/decisions/0003-the-first-modules.md)): `probability` (eight distributions and p-values), `descriptive` (moments, quantiles, weighted statistics, covariance and correlation, summaries), `regression` (OLS and WLS by pivoted QR with classical, HC0–HC3 and cluster-robust covariances and Wald tests; GLMs by IRLS for the gaussian, binomial (logit, probit) and Poisson families, with offsets and robust covariances) and `time_series` (autocorrelation, partial autocorrelation and Ljung–Box; AR by least squares and Yule–Walker; ARIMA by conditional sum of squares with forecasts; the ADF test with MacKinnon's p-values and critical values, KPSS, Engle–Granger; the Hodrick–Prescott filter; VARs with order selection, stability, impulse responses and forecasts). The methods they needed that name no statistical object (rectangular matrices, QR with pivoting, Cholesky and band factorisations, general eigenvalues, Nelder–Mead, BFGS, the simplex method) went to calx (calx decision 0008). In all, 8 check programs. On 2026-10-29 `spatial` gained Gaussian conditioning, extended one observation at a time, for the infrastructure's rover mission ([decision 0004](development/decisions/0004-gaussian-conditioning.md)): a ninth check program. On 2026-10-31, for the analysis of streets ([decision 0005](development/decisions/0005-spectral-inference-harmonics-and-point-patterns.md)): spectral inference (intervals, the levels of coherence and bicoherence, the red-noise background, surrogates, aliasing), harmonic analysis at given frequencies with the Rayleigh criterion and coloured errors, and the Lomb–Scargle periodogram with Baluev's false-alarm probability in `time_series`; co-location quotients with stratified permutation nulls and Clark and Evans's test in `spatial`. In all, 12 check programs. On 2026-11-01, for the infrastructure's genome analyses ([decision 0006](development/decisions/0006-change-points-penalties-and-factorisations.md)): change points in `time_series` (Gaussian, variance and Poisson costs; BIC and MBIC; optimal partitioning and PELT, exact and equal; binary and wild binary segmentation; circular binary segmentation with permutation p-values); ridge, the lasso and the elastic net along glmnet's path with cross-validation, and non-negative least squares, in `regression`; non-negative matrix factorisation and the stability of its rank, the first content of `multivariate`. The assignment problem and multinomial variates they need went to calx (its decision 0014). On 2026-11-14, for the infrastructure's foundation of evidence, tekmerion ([decision 0023](development/decisions/0023-graphical-models-and-belief-functions.md)): `graphical`, Bayesian networks over discrete variables (factors; networks validated with reasons; exact inference by variable elimination with hard and virtual evidence, P(e) kept in logarithms, joints, marginals, likelihood ratios and the most probable explanation; belief propagation, exact on polytrees and reported approximate when loopy; EM for the tables from incomplete records, with Dirichlet pseudo-counts), and `beliefs`, Dempster–Shafer belief functions (mass functions, belief, plausibility and commonality with the Möbius transforms, which are morphe's; Dempster's rule with its conflict, the conjunctive, Yager's and disjunctive rules; discounting and conditioning; the pignistic law, the credal set's bounds and Choquet expectations): four check programs, 138 checks. On 2026-11-15, at the infrastructure's question of decision theory and of evidence-based policy ([decision 0024](development/decisions/0024-decisions-in-one-step-and-the-value-of-information.md)): `decision` gained decisions in one step (a table of acts against states; expected utility, dominance and admissibility; maximin, maximax, Hurwicz, Laplace and minimax regret; credal sets with Γ-maximin, Γ-maximax, α-maxmin, interval dominance, maximality and E-admissibility by a linear programme), decision trees folded back with their strategies, the value of perfect and sample information by preposterior analysis and for a normal net benefit, and lotteries with stochastic dominance and certainty equivalents: two check programs, 48 checks. On 2026-11-15, for the layers beneath policy making ([decision 0025](development/decisions/0025-mixed-and-latent-variable-models.md)): `mixed`, linear mixed models by profiled REML or ML, and `latent`, factor analysis and latent classes: two check programs, 32 checks. On 2026-11-16, for decisions on evidence in space and time ([decision 0026](development/decisions/0026-information-bought-under-a-budget-and-intervals-for-a-proportion.md)): `decision/budget.hpp`, the tests worth running before separate decisions within a budget, as a knapsack of their EVSIs on calx's; and `probability/proportions.hpp`, Wilson's and Clopper and Pearson's intervals: two check programs, 29 checks; tyche 1597. On 2026-11-18, for the infrastructure's search for needs ([decision 0027](development/decisions/0027-comparing-groups-and-testing-what-a-search-finds.md)): `testing` gained comparisons of two groups (Welch, a share against a level, Newcombe's interval for two shares), the bootstrap of clusters, online control of the false discovery rate (LOND, LORD++, SAFFRON, α-investing, e-LOND), e-values (calibrators, e-BH) and the reusable holdout (Thresholdout); two check programs; tyche 1673. On 2026-11-18, for the infrastructure's cities across scales ([decision 0028](development/decisions/0028-change-of-support-and-estimates-withheld.md)): `spatial` gained change of support between partitions (areal weighting, dasymetric carriage, the densities of ancillary classes by NNLS, Tobler's pycnophylactic interpolation) and an estimator that says where it should not be believed (blocked cross-validation, conformal intervals from it, Meyer and Pebesma's area of applicability, each estimate withheld with its reason); two check programs, 33 checks; tyche 1706. On 2026-11-18, for the infrastructure's risk management ([decision 0029](development/decisions/0029-layers-of-a-loss-and-basis-risk.md)): `descriptive` gained layers of a loss (a year's expected layer loss and its spread, per year or per event as compound Poisson, towers split exactly, the layer read from a published curve) and the basis risk of a payout on an index, with the linear payout fitted on quantiles; one check program, 16 checks; tyche 1722. The repository is published at `https://github.com/huangwanhong-maker/tyche`.
