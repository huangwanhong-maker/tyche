# 0025. Linear Mixed Models and Latent-Variable Models

- **Status:** Accepted, provisionally (2026-11-15).
- **Date:** 2026-11-15
- **Decided by:** the user, who placed quantitative analysis among the layers beneath policy making (the infrastructure's ADR 0116, its fourth strand); and Claude.

## 1. Context

Social data are grouped (pupils in schools, people in places), and many quantities of interest are latent. tyche held variance components with known kernels, but neither mixed models with random slopes nor factor or latent-class models.

## 2. Decision

- `mixed`: linear mixed models with an unstructured covariance of random effects, by REML or ML, profiled, with BLUPs.
- `latent`: maximum-likelihood factor analysis with varimax, and latent class analysis.

Both are general statistical models and so tyche's. The scales and items of the social sciences remain nomos's (its decision 0013).

## 3. Consequences

`check_tyche_mixed` (14), `check_tyche_latent` (18).
