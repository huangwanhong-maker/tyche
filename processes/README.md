# processes: Stochastic Processes and Their Statistics

*Status: draft (2026-11-04): continuous-time Markov chains observed at discrete times, their panel likelihood and its maximum ([decision 0010](../development/decisions/0010-markov-chains-observed-at-discrete-times.md)).* Point processes, Gaussian processes, diffusions and random graphs are planned.

## 1. Concepts

A Markov chain in continuous time is often seen only now and then: a surface rated once at its age, a patient at a visit, a building at a census. Between two sightings the chain moved unseen, so the likelihood of what was seen is the transition probability of the interval, [exp(QΔ)]_{ij}, for a generator Q(θ) with parameters (Kalbfleisch and Lawless 1985). The parameters are the caller's. Log-rates keep the rates positive; a sequential chain has one rate per stage.

Each observation needs one row of exp(QΔ). Uniformisation writes it as a Poisson mixture of the powers of the stochastic matrix M = I + Q/Λ. The sum has no cancellation and is cut where the Poisson tail is negligible.

## 2. Content

| Header | Content |
|---|---|
| `panel_markov.hpp` | `PanelObservation{from, to, interval, weight}`; `Misclassification` (E(j \| k), the hidden Markov model of panel data; Jackson et al. 2003); `transition_row` (uniformisation; expm above ΛΔ = 500); `panel_log_likelihood`; `fit_panel` (BFGS within log-parameters of ±30, a start where an observation is impossible refused; Laplace's errors) → `PanelFit`; `sequential_generator` |

## 3. Checks

`check_tyche_panel_markov` (8):
- two states in closed form;
- a sequential chain's four rates recovered from simulated current-status data, by holding times;
- Laplace's errors against the spread of 15 replicates;
- misclassification: one observation's likelihood by hand; a new unit recorded worse than new refused without it, and the rates recovered with it;
- uniformisation against calx's expm;
- the refusals.

## 4. Dependencies

calx `linalg` (expm), `optimisation` (BFGS, Hessians by differences).

## 5. References

- Kalbfleisch, J. D. and Lawless, J. F. (1985), The analysis of panel data under a Markov assumption, *J. Amer. Statist. Assoc.* 80, 863–871.
- Jackson, C. H., Sharples, L. D., Thompson, S. G., Duffy, S. W. and Couto, E. (2003), Multistate Markov models for disease progression with classification error, *J. Roy. Statist. Soc. D* 52, 193–209.
- Jensen, A. (1953), Markoff chains as an aid in the study of Markoff processes, *Skand. Aktuarietidskr.* 36, 87–91; Grassmann, W. K. (1977), Transient solutions in Markovian queueing systems, *Comput. Oper. Res.* 4, 47–53.
- Nocedal, J. and Wright, S. J. (2006), *Numerical Optimization*, 2nd ed., Springer.

## 6. Open Questions

1. **Informative observation.** If the time of a sighting depends on the state (worse surfaces inspected sooner), the likelihood above is biased.
2. **Covariates.** Rates that depend on a unit's covariates (design life, material, traffic) would fit by a generator with a regression inside.
