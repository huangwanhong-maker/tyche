# 0021. Bayesian Computation: Conjugate Models and Markov Chain Monte Carlo

- **Status:** Accepted, provisionally (2026-11-13).
- **Date:** 2026-11-13
- **Decided by:** the infrastructure's ADR 0106 and its programme of the handbook *Material and Ecological Conditions of Life* (§7.2: Bayesian modelling and the kinds of uncertainty), at the user's request to implement the handbook's methods. The agent-based and dynamical models of the domains, and kinesis's templates, are calibrated to records. A posterior is the honest form of that calibration, and `bayes` had been planned since decision 0003.

## 1. Context

tyche held likelihoods within its estimators (regression, time series, HMMs), and Gaussian conditioning in `spatial`. It had no general way to draw from a posterior known up to its constant, and no diagnostics of such draws. PyMC and Stan do this for Python and R; the infrastructure's C++ engines need it in C++.

## 2. Decision

- **`bayes/conjugate.hpp`**, conjugate models in closed form, with predictive laws and marginal likelihoods:
  - beta–binomial;
  - gamma–Poisson with exposures;
  - the normal mean with a known variance;
  - normal–inverse-gamma, the mean's posterior a Student t;
  - Dirichlet–multinomial.
- **`bayes/mcmc.hpp`**, samplers of a `Target`: a log density on ℝᵈ, with its gradient if the caller has one, else central differences.
  - `metropolis`: random-walk Metropolis whose proposal covariance is learnt in the warm-up (Haario et al. 2001), scaled toward an acceptance of 0.234, and fixed afterwards.
  - `hmc`: Hamiltonian Monte Carlo. Its diagonal mass matrix comes from the warm-up's middle window, its step size from dual averaging toward 0.8 (Hoffman and Gelman 2014), and its path is of random length about π/2; divergences are counted.
  - `diagnose`: split R̂, the effective sample size by Geyer's initial monotone sequence over the chains combined, and the Monte Carlo error of each mean.
- **Several chains.** Each runs from its own stream, so results repeat exactly for a seed.

## 3. Alternatives Considered

- **The No-U-Turn sampler.** Deferred: HMC with a jittered path and adapted mass serves the domains' targets of moderate dimension. NUTS replaces the path's length when a target asks for it.
- **Stan or PyMC behind the interface.** Rejected for the C++ engines, which compose without a Python runtime. Python works use PyMC directly, as the infrastructure's rules allow.
- **Sequential Monte Carlo and variational inference.** Remain planned.

## 4. Consequences

Twenty-five checks:

- the conjugate posteriors by hand;
- the beta–binomial's marginal likelihoods summing to one;
- the gamma–Poisson's against an integral over the rate;
- the normal–inverse-gamma's against its predictive Student t, singly and in sequence;
- the Dirichlet's as a product of predictives;
- Metropolis on a normal law, and HMC on a correlated five-dimensional normal, within their Monte Carlo errors, with R̂ below 1.01 and HMC's acceptance near 0.8;
- both samplers against a conjugate posterior;
- the effective sample size of an AR(1) series;
- R̂ flagging chains that do not mix.
