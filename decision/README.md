# decision: Decisions Under Partial Observation

*Status: draft (2026-11-04).* Finite POMDPs and their bounds, exact and point-based solvers, online planning by Monte Carlo tree search, mixed observability, components joined by a budget, constraints on expected costs, and decentralised POMDPs ([decision 0009](../development/decisions/0009-decisions-under-partial-observation.md)). SARSOP, DESPOT, factored solvers with vectors by visible state, Bayes-adaptive POMDPs and stochastic games are planned.

## 1. Concepts

A decision maker who cannot see the state acts on a *belief*, the law of the state given what it has done and seen; Bayes's rule carries the belief forward, and the belief is all it needs (Åström 1965; Smallwood and Sondik 1973). A POMDP is therefore a Markov decision process on the simplex Δ(S). Its value over a finite horizon is the upper envelope of finitely many linear functions, the α-vectors, each the value of a conditional plan; its infinite-horizon questions are undecidable in general (Madani, Hanks and Condon 2003), so the solvers here answer with **bounds**: a lower bound that is the value of a plan one can carry out, and, where the solver can, an upper bound. A planning answer is a policy with an interval for its value.

Every variant is the same tuple restricted, extended or multiplied:

- **mixed observability** sees part of the state;
- **weakly coupled components** are many POMDPs joined only by a budget;
- **constrained POMDPs** add costs with budgets;
- **decentralised POMDPs** give each agent its own action and observation.

Nothing names an object in nature. A state, an action and an observation are numbered, and the domains give them meaning: an asset's condition, a district's uses, a crowd on a street.

## 2. Content

| Header | Content |
|---|---|
| `pomdp.hpp` | `Pomdp` (dense T, Z, R, γ, b₀) and `validate`; `predict`, `observation_law`, `update` (Bayes's rule), `expected_reward`; the MDP beneath: `solve_mdp` (ε-optimal, Puterman 1994, Theorem 6.3.1), `solve_mdp_horizon` |
| `alpha.hpp` | `AlphaVector`, `ValueFunction`; the bounds `qmdp`, `fast_informed_bound` (above V*) and `blind_lower_bound` (below; Hauskrecht 2000); `prune` (Lark's filter, its witness programmes on calx's simplex, posed in units of the vectors' differences); `prune_dominated`; `exact_backup` by incremental pruning (Cassandra, Littman and Zhang 1997), `exact_value_iteration`; `point_backup` |
| `point_based.hpp` | `lookahead` (the exact search of the belief tree, to a depth, on any leaf value); `pbvi` (Pineau, Gordon and Thrun 2003); `perseus` (Spaan and Vlassis 2005), stopping on the Bellman residual on its beliefs; `UpperBound` (the sawtooth, pruned of redundant points) and `hsvi` (Smith and Simmons 2004, 2005) |
| `simulate.hpp` | `evaluate`: the Monte Carlo value of a policy of beliefs, with its standard error and truncation bound, each episode on its own stream |
| `online.hpp` | generative models (`Step<State>`, any state type, observations as keys); `Pomcp` (Silver and Veness 2010): UCB1 over histories, particle beliefs refilled by rejection, rollouts or a leaf value (`set_leaf`); `TabularModel` |
| `structured.hpp` | `Momdp` with `to_pomdp` and `MomdpBelief` (Ong et al. 2010); `Component`s joined by a budget: `lagrangian_bound` (Adelman and Mersereau 2008), `coupled_actions` (a ranked policy within the budget), `joint_pomdp` (the exact product, for checks) |
| `constrained.hpp` | `PlanVector` (a plan's reward and cost vectors), its `point_backup` for R − λC (Kim et al. 2011); `reachable_beliefs`; `scalarised_finite` (exact over a horizon); `scalarised_infinite` (Perseus's backups, then a finite-state `Controller` evaluated exactly); `lagrangian_dual`: the mixture of two plans that spends the budget, and the dual bound |
| `multiagent.hpp` | `DecPomdp`, `PolicyTree` (breadth-first), `value` of a joint policy, `all_trees`, `exhaustive` (two agents), `best_response` (exact, on beliefs over the state and the others' nodes), `jesp` (Nair et al. 2003) with restarts |

## 3. Checks

- **`check_tyche_pomdp` (26).**
  - Bayes's rule on the tiger in closed form, and the refusals.
  - Exact value iteration equals the lookahead on the belief tree at 300 beliefs, every horizon to 5: the tiger, discounted and not, and two random POMDPs.
  - Pruning keeps only vectors with witnesses and changes no value.
  - The tiger at discount 0.75 (pomdp.org's `tiger.aaai`): HSVI's interval of width 10⁻⁶ contains the published V*(½) = 1.933439, and PBVI reaches it.
  - QMDP ≥ FIB ≥ V* ≥ blind, with V* bracketed by the lookahead of depth 4 on each bound.
  - PBVI's, Perseus's and HSVI's bounds lie on their sides of V*, and HSVI closes at b₀.
  - HSVI's policy, simulated, earns its interval.
- **`check_tyche_online` (3).** POMCP listens first. Its particles after one listen hold Bayes's 0.85. Played online with the blind bound as leaf value, it earns the optimal interval.
- **`check_tyche_multiagent` (11).** Dec-Tiger's published optima −4.000000 and 5.190812 at horizons 2 and 3 (Oliehoek et al. 2013). A best response is the best of all 2187 trees. JESP finds the optima at horizons 2 and 3; at horizon 4 it reaches an equilibrium below the optimum 4.802755.
- **`check_tyche_structured` (8).**
  - A MOMDP's beliefs as a POMDP's equal its own.
  - Three assets under one repair budget: the Lagrangian bound lies above the 27-state product's lower bound and below the budget ignored; the ranked policy keeps the budget, and its simulated value lies within the interval.
- **`check_tyche_constrained` (9).**
  - The tiger with a cost for listening, over three steps: backups at every reachable belief give the best of all 2187 trees for every λ; the dual's mixture attains the linear programme's optimum over mixtures of trees, with strong duality.
  - At discount 0.95: the controllers' exact values agree with their simulation, and the mixture lies below the dual bound.

## 4. Dependencies

calx `linalg`, `optimisation` (the simplex, for pruning and the checks' programmes), `random` (streams and categorical draws).

## 5. References

- Åström, K. J. (1965), Optimal control of Markov processes with incomplete state information, *J. Math. Anal. Appl.* 10, 174–205.
- Smallwood, R. D. and Sondik, E. J. (1973), The optimal control of partially observable Markov processes over a finite horizon, *Oper. Res.* 21, 1071–1088.
- Kaelbling, L. P., Littman, M. L. and Cassandra, A. R. (1998), Planning and acting in partially observable stochastic domains, *Artif. Intell.* 101, 99–134.
- Madani, O., Hanks, S. and Condon, A. (2003), On the undecidability of probabilistic planning and related stochastic optimization problems, *Artif. Intell.* 147, 5–34.
- Puterman, M. L. (1994), *Markov Decision Processes*, Wiley.
- Cassandra, A. R., Littman, M. L. and Zhang, N. L. (1997), Incremental pruning, *UAI-97*, 54–61.
- Hauskrecht, M. (2000), Value-function approximations for partially observable Markov decision processes, *JAIR* 13, 33–94.
- Pineau, J., Gordon, G. and Thrun, S. (2003), Point-based value iteration, *IJCAI-03*, 1025–1031.
- Spaan, M. T. J. and Vlassis, N. (2005), Perseus, *JAIR* 24, 195–220.
- Smith, T. and Simmons, R. (2004), Heuristic search value iteration for POMDPs, *UAI-04*, 520–527; (2005), Point-based POMDP algorithms: improved analysis and implementation, *UAI-05*, 542–549.
- Silver, D. and Veness, J. (2010), Monte-Carlo planning in large POMDPs, *NIPS 23*, 2164–2172.
- Ong, S. C. W., Png, S. W., Hsu, D. and Lee, W. S. (2010), Planning under uncertainty for robotic tasks with mixed observability, *IJRR* 29, 1053–1068.
- Adelman, D. and Mersereau, A. J. (2008), Relaxations of weakly coupled stochastic dynamic programs, *Oper. Res.* 56, 712–727.
- Altman, E. (1999), *Constrained Markov Decision Processes*, Chapman & Hall/CRC.
- Kim, D., Lee, J., Kim, K.-E. and Poupart, P. (2011), Point-based value iteration for constrained POMDPs, *IJCAI-11*, 1968–1973.
- Poupart, P., Malhotra, A., Pei, P., Kim, K.-E., Goh, B. and Bowling, M. (2015), Approximate linear programming for constrained partially observable Markov decision processes, *AAAI* 29, 3342–3348.
- Bernstein, D. S., Givan, R., Immerman, N. and Zilberstein, S. (2002), The complexity of decentralized control of Markov decision processes, *Math. Oper. Res.* 27, 819–840.
- Nair, R., Tambe, M., Yokoo, M., Pynadath, D. and Marsella, S. (2003), Taming decentralized POMDPs, *IJCAI-03*, 705–710.
- Oliehoek, F. A., Spaan, M. T. J., Amato, C. and Whiteson, S. (2013), Incremental clustering and expansion for faster optimal planning in Dec-POMDPs, *JAIR* 46, 449–509.

## 6. Open Questions

1. **Exact value iteration grows.** On the tiger at γ = 0.95 it holds 76 vectors at horizon 25, and a minute of programmes by 31. It serves short horizons and checks. Point-based solvers serve the rest.
2. **HSVI's upper bound converges slowly.** On a random POMDP of 4 states its sawtooth needed 65,791 points to close to 10⁻³ (600 s) before pruning. SARSOP's pruning and sampling, and better corner values, would help.
3. **POMCP needs a leaf value.** With uniformly random rollouts on the tiger its estimates are too noisy to learn from (−17 ± 7 at 2000 simulations a step, against V* = 19.37). With the blind bound as leaf value it approaches V*. Domains should give their own rollout policies or leaf values.
4. **Constrained policies are controllers.** Acting greedily on a constrained solver's vectors does not spend what the vectors claim: on the tiger it listened 10.3 discounted times against a plan of 2. The executed policy is therefore a finite-state controller, evaluated exactly.
5. **Planned.** DESPOT; SARSOP; vectors by visible state for MOMDPs; factored models by dynamic Bayesian networks; CC-POMCP; Bayes-adaptive POMDPs; ρ-POMDPs; stochastic games (in nomos, on its games).
