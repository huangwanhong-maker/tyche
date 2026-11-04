# processes: Stochastic Processes and Their Statistics

*Status: draft (2026-11-05): continuous-time Markov chains observed at discrete times, their panel likelihood and its maximum ([decision 0010](../development/decisions/0010-markov-chains-observed-at-discrete-times.md)); hidden Markov models with Gaussian emissions ([decision 0011](../development/decisions/0011-hidden-markov-models.md)).* Point processes, Gaussian processes, diffusions and random graphs are planned.

## 1. Concepts

A Markov chain in continuous time is often seen only now and then: a surface rated once at its age, a patient at a visit, a building at a census. Between two sightings the chain moved unseen, so the likelihood of what was seen is the transition probability of the interval, [exp(QΔ)]_{ij}, for a generator Q(θ) with parameters (Kalbfleisch and Lawless 1985). The parameters are the caller's. Log-rates keep the rates positive; a sequential chain has one rate per stage.

Each observation needs one row of exp(QΔ). Uniformisation writes it as a Poisson mixture of the powers of the stochastic matrix M = I + Q/Λ. The sum has no cancellation and is cut where the Poisson tail is negligible.

A chain may also be hidden altogether, seen only through what it emits: in its state k an observation y ~ N(μ_k, σ_k²). Bayes's rule carries the law of the state given the observations so far (the filter); the backward pass gives it given the whole sequence (the smoother); Viterbi's recursion gives the most probable path. Expectation–maximisation fits π, A, μ and σ to many sequences at once, which share the model and keep their own paths. With one σ for all states the states differ by their means alone, Hamilton's (1989) reading of the business cycle.

## 2. Content

| Header | Content |
|---|---|
| `hidden_markov.hpp` | `GaussianHmm{initial, transition, mean, sd}`; `decode(M, y)` → `Decoded{filtered, smoothed, path, log_likelihood}` (the scaled forward filter, the backward smoother, Viterbi; NaN for a missing observation); `fit_hmm(sequences, K, starts, max_iterations, tolerance, seed, shared_sd)` → `HmmFit{model, log_likelihood, iterations, converged, trace}` (Baum–Welch over many sequences, the best of several starts from the pooled quantiles, σ held above 10⁻³ of the pooled deviation, the states ordered by their means) |
| `panel_markov.hpp` | `PanelObservation{from, to, interval, weight}`; `Misclassification` (E(j \| k), the hidden Markov model of panel data; Jackson et al. 2003); `transition_row` (uniformisation; expm above ΛΔ = 500); `panel_log_likelihood`; `fit_panel` (BFGS within log-parameters of ±30, a start where an observation is impossible refused; Laplace's errors) → `PanelFit`; `sequential_generator` |

## 3. Checks

`check_tyche_panel_markov` (8):
- two states in closed form;
- a sequential chain's four rates recovered from simulated current-status data, by holding times;
- Laplace's errors against the spread of 15 replicates;
- misclassification: one observation's likelihood by hand; a new unit recorded worse than new refused without it, and the rates recovered with it;
- uniformisation against calx's expm;
- the refusals.

`check_tyche_hidden_markov` (7):
- the log-likelihood and every smoothed marginal against the sum over all 729 paths of three states and six observations, one missing;
- the most probable path against brute force;
- the last filtered law is the last smoothed one, and every law sums to 1;
- a two-state model recovered from 300 sequences of 60 steps, the likelihood never falling along EM;
- with a shared deviation, three means and their one σ recovered.

## 4. Dependencies

calx `linalg` (expm), `optimisation` (BFGS, Hessians by differences), `random` (the starts' draws).

## 5. References

- Kalbfleisch, J. D. and Lawless, J. F. (1985), The analysis of panel data under a Markov assumption, *J. Amer. Statist. Assoc.* 80, 863–871.
- Jackson, C. H., Sharples, L. D., Thompson, S. G., Duffy, S. W. and Couto, E. (2003), Multistate Markov models for disease progression with classification error, *J. Roy. Statist. Soc. D* 52, 193–209.
- Jensen, A. (1953), Markoff chains as an aid in the study of Markoff processes, *Skand. Aktuarietidskr.* 36, 87–91; Grassmann, W. K. (1977), Transient solutions in Markovian queueing systems, *Comput. Oper. Res.* 4, 47–53.
- Nocedal, J. and Wright, S. J. (2006), *Numerical Optimization*, 2nd ed., Springer.
- Baum, L. E., Petrie, T., Soules, G. and Weiss, N. (1970), A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains, *Ann. Math. Statist.* 41, 164–171.
- Rabiner, L. R. (1989), A tutorial on hidden Markov models and selected applications in speech recognition, *Proc. IEEE* 77, 257–286.
- Hamilton, J. D. (1989), A new approach to the economic analysis of nonstationary time series and the business cycle, *Econometrica* 57, 357–384.

## 6. Open Questions

1. **Informative observation.** If the time of a sighting depends on the state (worse surfaces inspected sooner), the likelihood above is biased.
2. **Covariates.** Rates that depend on a unit's covariates (design life, material, traffic) would fit by a generator with a regression inside.
