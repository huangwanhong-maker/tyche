# 0022. Uncertainty and Sensitivity: Propagation, Morris's Screening and Sobol' Indices

- **Status:** Accepted, provisionally (2026-11-13).
- **Date:** 2026-11-13
- **Decided by:** the infrastructure's programme of the handbook *Material and Ecological Conditions of Life* (§7: SALib; §7.2: parameter uncertainty and sensitivity), at the user's request to implement its methods. `uncertainty` had been planned since decision 0003.

## 1. Context

Every model of the domains has declared parameters: lifetimes, intensities, behavioural coefficients, flows. Which of them its answer depends on, and how far its answer is uncertain when they are, are general questions of chance. They name no model.

## 2. Decision

**`uncertainty/uncertainty.hpp`.** A model is any f: ℝᵈ → ℝ; each input is given by its law's quantile function, so any of tyche's laws can stand for it. The inputs are independent. It holds:

- `monte_carlo` and `latin_hypercube` designs (McKay, Beckman and Conover 1979);
- `propagate`: the output's sample, mean, standard deviation and quantiles;
- `delta_method`: the first-order law ∇fᵀΣ∇f (GUM, JCGM 100:2008, eq. 13);
- `morris`: elementary effects on a grid of levels inside (0, 1), giving μ, μ* and σ (Morris 1991; Campolongo et al. 2007);
- `sobol`: first-order and total indices from two designs and their hybrids (Saltelli et al. 2010; Jansen 1999), with bootstrap intervals.

## 3. Alternatives Considered

- **Sobol's low-discrepancy sequences.** Deferred until their direction numbers are taken from Joe and Kuo's published tables, checked. Pseudo-random designs give the same estimators at a slower rate.
- **Dependent inputs.** The indices in this form need independence; Shapley effects, which do not, are planned (nomos holds the Shapley value).

## 4. Consequences

Fifteen checks:

- a Latin hypercube meets every stratum once;
- a linear model's propagated mean and spread, and the delta method's, exact for it;
- the delta method on a product against its closed form, and the Monte Carlo spread against the exact one, which the first order misses;
- Morris's effects of a linear model, exact;
- Sobol' indices of a linear model, and of Ishigami's function against its closed form, the bootstrap interval holding the truth.
