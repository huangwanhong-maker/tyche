# 0010. Markov Chains Observed at Discrete Times: the Panel Likelihood

- **Status:** Accepted, provisionally (the first content of `processes`; point processes, Gaussian processes, diffusions and random graphs planned)
- **Date:** 2026-11-04
- **Decided by:**
  - **The need.** The infrastructure's urban-science domain fits the deterioration of the City of Melbourne's road surfaces from one rating each, at its age ([ADR 0071](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/development/decisions/0071-agents-and-decisions-for-urban-planning.md)). That is a Markov chain observed at discrete times, which names no road.
  - **What Claude did.** It wrote the header and this record, under the user's request for POMDPs and agents in urban planning (decision 0009).

## 1. Context

The README planned `processes` for "Markov chains, point processes (Poisson, renewal, Hawkes), Gaussian processes and random fields, diffusions, random graphs".

**What existed.** physis's `statistical_mechanics` holds master equations and the transient law of a chain whose rates are given (`evolve`), and its chains on classified sets fit their rates from transitions (physis decision 0020). Neither fits a parametrised generator to states seen at irregular intervals, the panel data of Kalbfleisch and Lawless (1985), of which current-status data are the special case of one observation a unit.

## 2. Decision

**`processes/panel_markov.hpp`:**
- `PanelObservation` (from, to, interval, weight) and the log-likelihood Σ w log [exp(QΔ)]_{from,to} of a generator Q(θ) given as a function.
- Each needed row of exp(QΔ) is computed by uniformisation, to mean + 10√mean + 20 terms; calx's expm takes over when ΛΔ > 500.
- Misclassification: a matrix E(j | k) of recording j in state k, the hidden Markov model of panel data (Jackson et al. 2003).
- `fit_panel`: BFGS on numerical gradients (calx), with a gradient tolerance scaled by the data's weight, parameters held within ±30, a start where an observation is impossible refused with the reason, and Laplace's errors from the Hessian by differences.
- `sequential_generator`: stages in order, the last absorbing; its rates exp(θ_k).

## 3. Alternatives Considered

- **In physis's `statistical_mechanics`, beside the master equation.** Rejected: fitting a model to data is inference, which is tyche's. The chain's law, which physis holds, is what the likelihood evaluates.
- **calx's expm for every observation.** Rejected after its check. Ages are continuous, so a fit needed thousands of exponentials an evaluation; the rows by uniformisation cost O(K n²).
- **Stopping uniformisation when the weights sum to within 10⁻¹⁵ of 1.** Rejected after its check: the sum's rounding can keep it above that for ever, and the loop then ran to its cap. The number of terms is now fixed by the Poisson tail.
- **An absolute gradient tolerance.** Rejected: −ℓ of thousands of observations cannot be resolved to 10⁻⁶ by central differences, and BFGS then never stopped.
- **Ratings read without error.** Rejected after the Hoddle Grid's kerbs: some rated in class 4 on the day they were laid, impossible for a chain that starts in class 5. The misclassification the planning model declares for an inspection is the one the fit uses.
- **Parameters unbounded.** Rejected: a line search reached log-rates near 670, where expm overflows into finite but meaningless numbers it could accept.

## 4. Consequences

- **Checks.** `check_tyche_panel_markov` (8):
  - two states in closed form;
  - the four rates of a sequential chain recovered from 6000 units of current-status data simulated by holding times;
  - Laplace's errors within 40 % of the spread over 15 replicates of 4000 units;
  - misclassification: one observation by hand, a start made impossible without it refused, the rates recovered with it;
  - uniformisation's rows equal calx's expm to 1.2 × 10⁻¹⁵;
  - the refusals.
- **Its first use.** urban-science's `maintenance` fits the deterioration of kerbs, channels, footways and carriageways.
- **Planned.**
  - Observation that depends on the state (the inspection paradox), and covariates of the rates;
  - Poisson, renewal and Hawkes processes;
  - Gaussian processes beside `spatial`'s conditioning.
