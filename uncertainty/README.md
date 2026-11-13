# uncertainty: Propagation and Sensitivity

*Status: draft (2026-11-13; [decision 0022](../development/decisions/0022-uncertainty-and-sensitivity.md)).* How a model's output follows from uncertain inputs, and which inputs it answers to most.

## 1. Concepts

A model f maps inputs to an output. Each input is uncertain, with a law given by its quantile function. Two questions follow:

- **Propagation** asks for the output's law: by sampling the inputs (Monte Carlo, or a Latin hypercube that meets every stratum), or to first order by the gradient (the delta method, exact for an affine f).
- **Sensitivity** asks which inputs it depends on:
  - Morris's elementary effects screen many inputs cheaply;
  - Sobol' indices split the output's variance among the inputs, first order (an input alone) and total (an input with all its interactions).

## 2. Content

`uncertainty.hpp`:

- the designs `monte_carlo` and `latin_hypercube`;
- `propagate` and `delta_method`;
- `morris` (μ, μ*, σ);
- `sobol` (first-order and total indices, with bootstrap intervals).

## 3. Checks

`check_uncertainty.cpp` (15):

- Latin hypercube strata;
- a linear model, exactly, by every method;
- the delta method on a product against its closed form and against Monte Carlo;
- Morris on a linear model;
- Sobol' on a linear model and on Ishigami's function against its closed form.

## 4. Dependencies and References

- **Depends on:** calx (`linalg`, `random`).
- **References:**
  - McKay, Beckman and Conover (1979), *Technometrics* 21, 239–245;
  - Morris (1991), *Technometrics* 33, 161–174;
  - Campolongo, Cariboni and Saltelli (2007), *Environ. Model. Softw.* 22, 1509–1518;
  - Saltelli et al. (2010), *Comput. Phys. Commun.* 181, 259–270;
  - Jansen (1999), *Comput. Phys. Commun.* 117, 35–43;
  - JCGM 100:2008 (GUM).
- **Open:**
  - Sobol's low-discrepancy sequences;
  - Shapley effects for dependent inputs;
  - surrogates;
  - calibration.
