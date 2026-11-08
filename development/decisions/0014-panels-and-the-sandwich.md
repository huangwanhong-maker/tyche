# 0014. Panels and the Sandwich

- **Status:** Accepted, provisionally (2026-11-08).
- **Decided by:** the infrastructure's ADR 0082, stage 2 ("panel fixed effects to tyche"), on the user's go-ahead; the module's form, Claude.

## 1. Decision

- **A new module, `panel`** (`tyche::panel`), receives from nomos's `econometrics`, unchanged but for its namespace:
  - `absorb.hpp`: the absorption of one or two factors' fixed effects by alternating projections, their absorbed rank, and the recovery of the effects;
  - `panel.hpp`: the within estimator, one-way and two-way on unbalanced panels, with classical and cluster-robust covariances counted as Stata's xtreg and reghdfe count them; the between estimator; random effects by Swamy and Arora; the Hausman test.
- **`regression` gains `sandwich.hpp`.** The pieces of robust covariances that estimators share: columns joined, the meat of a sandwich (heteroskedasticity-robust or by clusters), the sandwich, a solve of a positive (semi-)definite system.

These are inference within a model, which is tyche's (ADR 0048). Panels were in nomos because economics needed them first.

## 2. Verification

nomos's panel checks moved with the code, unchanged: within against the dummy-variable regression (one-way and two-way, unbalanced), the cluster counting, random effects against Maddala's weighted average, Swamy and Arora's components, Hausman. 21 checks. The sandwich is exercised by peira's instruments and nomos's Poisson and multinomial checks.

## 3. Consequences

- nomos's remaining econometrics (marginal effects, the multinomial logit, Poisson with absorbed effects) reads `absorb.hpp` and `sandwich.hpp` from here (nomos's decision 0007); peira's quasi-experiments read `panel.hpp` (peira's decision 0004).
