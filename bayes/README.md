# bayes: Bayesian Computation

*Status: draft (2026-11-13; [decision 0021](../development/decisions/0021-bayesian-computation.md)).* Conjugate models, and Markov chain Monte Carlo with its diagnostics.

## 1. Concepts

A posterior is the prior times the likelihood, normalised. When the prior is conjugate to the likelihood the posterior stays in the prior's family, and everything (its moments, the predictive law of a new observation, the marginal likelihood that compares models) is in closed form.

Otherwise the posterior is known only up to its constant, and a Markov chain whose stationary law it is draws from it. The draws are trusted only as far as the diagnostics allow:

- R̂ compares the spread within chains with the spread between them;
- the effective sample size counts how many independent draws the correlated ones are worth.

## 2. Content

| Header | What it holds |
|---|---|
| `conjugate.hpp` | `beta_binomial`, `gamma_poisson` (with exposures), `normal_known_variance`, `normal_inverse_gamma` (the mean's Student t, the predictive), `dirichlet_multinomial`: posteriors, predictives, log marginal likelihoods |
| `mcmc.hpp` | `Target` (a log density, its gradient optional); `metropolis` (adaptive in the warm-up, Haario et al. 2001); `hmc` (dual averaging, Hoffman and Gelman 2014; a diagonal mass matrix; a jittered path); `diagnose` (split R̂, effective sample size, Monte Carlo error) |
| `bayes.hpp` | both |

## 3. Checks

- `check_conjugate.cpp` (11): the conjugate models against closed forms, integrals and predictives.
- `check_mcmc.cpp` (14): the samplers on normal laws and a conjugate posterior within their errors; the effective sample size of AR(1); R̂ on chains that do not mix.

## 4. Dependencies and References

- **Depends on:** calx (`linalg`, `random`, `special_functions`) and tyche's `probability`.
- **References:**
  - Gelman, A. et al. (2013), *Bayesian Data Analysis*, 3rd ed.;
  - Hoffman, M. D. and Gelman, A. (2014), *JMLR* 15, 1593–1623;
  - Neal, R. M. (2011), MCMC using Hamiltonian dynamics, *Handbook of MCMC*, ch. 5;
  - Vehtari, A. et al. (2021), *Bayesian Analysis* 16, 667–718.
- **Open:**
  - the No-U-Turn sampler;
  - rank-normalised R̂;
  - sequential Monte Carlo;
  - variational inference.
