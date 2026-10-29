# 0002. Scope, Boundaries and Dependencies

- **Status:** Accepted
- **Date:** 2026-10-24
- **Decided by:** the user, who approved a foundation library for inference and chose its name, *tyche* (Greek *τύχη*, chance), closing open decision 3 of the infrastructure's CLAUDE.md as far as the inference foundation goes; its place had been designed in the infrastructure's [map of the mathematics](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/docs/mathematics.md), §6, under the provisional name *inference*. Claude drew the boundaries; the infrastructure records the library in its [ADR 0048](https://github.com/huangwanhong-maker/The-Science-Infrastructures/tree/main/development/decisions).

## 1. Context

The sciences share their statistics as they share their mathematics: a regression, a test of a unit root or a Kalman filter is the same in economics, meteorology and neuroscience. The infrastructure's foundations held numbers (calx), structures (morphe) and the continuum (physis), but no home for chance. The economics domain and the social-science foundation nomos need regressions, likelihoods and time series now; neuroscience will need point processes and tests of maps.

## 2. Decision

1. **Scope.** tyche covers probability, information, stochastic processes, descriptive statistics, estimation, testing, regression, multivariate analysis, time series, spatial statistics, Bayesian inference, data assimilation, the uncertainty of models, causal inference, design, and statistical learning: Mathematics Subject Classification 60 and 62, with the statistical parts of 68 and 91. Its modules are those of the map, §6, in its order.
2. **Dependencies.** tyche depends on calx, morphe and physis, and on nothing else of the project. physis never depends on tyche: a stochastic differential equation is a system of physis, integrated by calx, whose statistics are tyche's.
3. **Boundaries.**
   - A method that names no statistical object (least squares, a minimiser, an eigenvalue, a linear programme) belongs to calx, even when tyche is its first user; tyche uses it.
   - Exact and combinatorial structures (graphs, words, groups) belong to morphe.
   - The methods of the social sciences as such (measurement, surveys, econometric designs, games, social choice) belong to nomos, a library above tyche.
   - Data and their meaning belong to the domains; tyche keeps only reference data for its checks.
4. **No particular data.** Every dataset is an input.
5. **Language and build.** C++20, header-only targets `tyche::<module>`, checks on calx's `verification`, as the siblings (the infrastructure's ADR 0003 and ADR 0035).

## 3. Alternatives Considered

- **Statistics inside calx.** Rejected: calx knows numbers and arrays, not distributions, samples or models; its invariant 1 keeps it free of any scientific meaning, and statistics carries the meaning of chance.
- **Statistics inside each domain.** Rejected by the infrastructure's principle that a general method is written once, in a foundation.
- **One library for statistics and the social sciences.** Rejected: measurement, games and social choice are not statistics, and the methods of statistics serve every science; nomos sits above tyche.
- **The name *inference*.** The map's provisional name; the user preferred a name of the family of calx, morphe and physis.

## 4. Consequences

- A change that spans libraries is committed in dependency order: calx and morphe first, then physis, then tyche, then nomos, then the infrastructure.
- The domains and nomos apply tyche's methods to their data.
