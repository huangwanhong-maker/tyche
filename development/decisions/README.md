# Decision Log

Each record captures one decision that shapes the structure, interfaces or tooling of tyche.

- Records are numbered and never rewritten.
- When a decision changes, a new record supersedes the old one, and the old record's status is updated to point to it.

## 1. Index

| No. | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-scope-boundaries-and-dependencies.md) | Scope, boundaries and dependencies | Accepted |
| [0003](0003-the-first-modules.md) | The first modules: probability, descriptive statistics, regression and time series | Accepted, provisionally |
| [0004](0004-gaussian-conditioning.md) | Gaussian conditioning, the first content of `spatial` | Accepted, provisionally |
| [0005](0005-spectral-inference-harmonics-and-point-patterns.md) | Spectral inference, harmonic analysis, Lomb–Scargle, and point patterns of marks | Accepted, provisionally |
| [0006](0006-change-points-penalties-and-factorisations.md) | Change points, penalised regression, non-negative least squares and non-negative factorisation | Accepted, provisionally |
| [0007](0007-sum-of-single-effects-variance-components-and-multiple-testing.md) | The sum of single effects, variance components, smoothing splines and multiple testing | Accepted, provisionally |
| [0008](0008-the-learning-module.md) | The learning module: networks, backpropagation and Adam; rank correlation | Accepted, provisionally |
| [0009](0009-decisions-under-partial-observation.md) | Decisions under partial observation: POMDPs and their variants | Accepted, provisionally |
| [0010](0010-markov-chains-observed-at-discrete-times.md) | Markov chains observed at discrete times: the panel likelihood | Accepted, provisionally |
| [0011](0011-hidden-markov-models.md) | Hidden Markov models with Gaussian emissions | Accepted, provisionally |
| [0012](0012-exceedance.md) | Annual losses and exceedance curves | Accepted, provisionally |
| [0013](0013-spatial-autocorrelation.md) | Spatial autocorrelation: Moran's I, global and local | Accepted, provisionally |
| [0014](0014-panels-and-the-sandwich.md) | Panels and the sandwich, moved from nomos | Accepted, provisionally |
| [0015](0015-spatial-regression.md) | Spatial regression: lag and error models, LM tests, impacts | Accepted, provisionally |
| [0016](0016-information-through-a-record.md) | Information through a record: entropy, the chain rule of relative entropy as what a record shows and hides, the rate at which jump processes part | Accepted, provisionally |

## 2. Template

```markdown
# NNNN. Title

- **Status:** Proposed | Accepted | Superseded by NNNN
- **Date:** YYYY-MM-DD
- **Decided by:** …

## 1. Context
## 2. Decision
## 3. Alternatives Considered
## 4. Consequences
```
