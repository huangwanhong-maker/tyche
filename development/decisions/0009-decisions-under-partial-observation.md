# 0009. Decisions Under Partial Observation: POMDPs and Their Variants

- **Status:** Accepted, provisionally (dense tables; SARSOP, DESPOT, factored solvers, Bayes-adaptive POMDPs and stochastic games planned)
- **Date:** 2026-11-04
- **Decided by:**
  - **The request.** The user asked for "two important module for modern urban planning. One is ABM, another one is POMDP (many variant)", reusable and kept apart from the applications. Asked where they belong, they chose agent-based models in nomos's `agents` and POMDPs in a new tyche module `decision`, with all four families offered: exact and point-based, online Monte Carlo, factored and mixed observability, multi-agent and constrained.
  - **What Claude did.** It wrote the module and this record. The design is the infrastructure's [ADR 0071](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/development/decisions/0071-agents-and-decisions-for-urban-planning.md) and its note [development/ideas/2026-11-04-agents-and-decisions-for-urban-planning.md](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/development/ideas/2026-11-04-agents-and-decisions-for-urban-planning.md).

## 1. Context

**What was needed.** The infrastructure's urban-science domain plans the maintenance of assets whose condition is seen only when inspected. It also plans street regimes from counts at a few sensors, and interventions on a district's uses seen only at each census. Each is a decision under partial observation. The robotics domain and Cosmopoiesis's missions plan so too.

**Why tyche.** A belief is a law updated by Bayes's rule. Acting on it is statistical decision theory (MSC 62C), which is tyche's, with the sequential problems of MSC 90C40 beside it.

**What was there.** Nothing in the foundations held a POMDP. calx held the simplex method and random streams; physis held master equations and chains on classified sets (its decision 0020), whose contraction of a record is the forward filter a belief is.

## 2. Decision

**A new module, `decision`**, header-only, on calx's `linalg`, `optimisation` and `random`. Its README lists the content; in brief:

1. **The tuple and its bounds.**
   - Dense finite POMDPs, Bayes's rule, and the MDP beneath, solved to ε-optimality.
   - QMDP and the fast informed bound above V*, blind policies below it.
2. **Exact value iteration.**
   - Incremental pruning, with Lark's filter on calx's simplex: each candidate is tested against the vectors kept, not against every other candidate.
   - The witness programmes are posed in units of the vectors' differences and with positive right-hand sides. Near convergence those differences fall below the simplex's pivot tolerance, and with zero right-hand sides every pivot is degenerate.
3. **Point-based solvers.**
   - PBVI and Perseus. Perseus stops on the Bellman residual on its beliefs, not on a stage's gain, since a stage can end with nothing raised.
   - HSVI with the sawtooth upper bound, pruned of redundant points whenever their number doubles.
   - The exact lookahead on the belief tree, for checks and small horizons.
4. **Online planning.** POMCP on generative models of any state type. Particles are refilled by rejection. It takes a leaf value in place of a rollout.
5. **Structure.**
   - MOMDPs, written exactly as POMDPs.
   - Components joined by a budget: the Lagrangian upper bound, a ranked policy within the budget, and the exact product for checks.
6. **Constraints.**
   - Plans carrying a reward vector and a cost vector, backed up for R − λC.
   - Exact over a finite horizon, by backups at every reachable belief.
   - Over an infinite horizon, a finite-state controller on the final plans, evaluated exactly.
   - The Lagrangian dual by bisection, with the mixture of two plans that spends the budget.
7. **Several decision makers.**
   - Dec-POMDPs, with policies as trees.
   - Exact values, exhaustive search, exact best responses, and JESP with restarts.

The scope in tyche's CLAUDE.md gains *statistical decision theory: decisions under uncertainty, sequential and partially observed*. The boundary with nomos is drawn where a decision has several rewards: a stochastic game's equilibria are nomos's (its `games`), a cooperative Dec-POMDP's optimum is tyche's.

## 3. Alternatives Considered

- **POMDPs in nomos, beside its games.** Rejected: robotics and the missions would then declare the social sciences' library to plan, and a single decision maker's problem is not a social one.
- **POMDPs in physis's planned `control`.** Rejected for finite POMDPs: they are discrete and statistical, while physis's control is of dynamical systems. Linear–Gaussian control under partial observation (LQG) stays planned there.
- **Binding an existing library (APPL/SARSOP, AI-Toolbox, the MADP toolbox).** Rejected: they are GPL. They may serve as external oracles in fixture scripts, never linked. POMDPs.jl and pomdp_py are MIT, and stay references.
- **Greedy execution of constrained vectors.** Rejected after its check. On the tiger with a budget of 2 discounted listens, the greedy policy listened 10.3 times. The controller's exact values agree with simulation to within its standard error.
- **Random rollouts as POMCP's only leaf estimate.** Kept as the default, with a leaf value offered beside it. On the tiger the rollouts' noise swamped the differences between actions: −17 ± 7 at 2000 simulations a step, worse with more. The blind bound as leaf value approaches V*.

## 4. Consequences

- **Checks.** Five programs, 57 checks; the module's README lists them. They include:
  - the published tiger value at discount 0.75 (1.933439);
  - Dec-Tiger's published optima at horizons 2 and 3 (−4.000000 and 5.190812);
  - the linear programme over mixtures of all 2187 trees against the constrained solver, with strong duality;
  - throughout, the exact lookahead on the belief tree, which shares no code with the solvers.
- **calx.** It gained `random/categorical.hpp` (a categorical draw by inversion, Walker's alias table by Vose's construction), which this module and nomos's agents both need.
- **calx's simplex** reports "iteration limit" when its phase I meets a column of negative reduced cost with no positive entry. With an objective bounded below that can only be numerical, but the status misleads. This is noted for calx; the witness programmes here avoid it by their form.
- **Planned.**
  - DESPOT and POMCPOW;
  - SARSOP;
  - vectors by visible state for MOMDPs;
  - factored models by dynamic Bayesian networks;
  - CC-POMCP;
  - ρ-POMDPs and Bayes-adaptive POMDPs.
