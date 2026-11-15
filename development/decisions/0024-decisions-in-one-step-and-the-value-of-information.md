# 0024. Decisions in One Step, Decision Trees and the Value of Information

- **Status:** Accepted, provisionally (2026-11-15).
- **Date:** 2026-11-15
- **Decided by:** the user, who asked of the infrastructure "What about decision theory", and then that evidence-based policy making be placed among decision theory's uses and built; and Claude, who placed the theory itself here (tyche's scope already names statistical decision theory) and its use for policy in a domain of the infrastructure.

## 1. Context

`decision` held decisions taken step after step on a state partly seen: POMDPs and their kin (decision 0009). It had no account of the single decision, which every sequential one is made of and which most uses need first: a table of acts against states, chosen among when the probabilities are known, unknown, or known only within a set; a tree of decisions and chances folded back; and what it is worth to learn more before choosing. The evidence foundation tekmerion had written the value of information for a normal net benefit for its own use; it is general, and belongs here.

## 2. Decision

Four headers join `decision`:

1. `choice.hpp`: a decision table; expected utility; dominance and admissibility; under ignorance, maximin, maximax, Hurwicz's criterion, Laplace's and minimax regret; credal sets as convex hulls of finitely many probabilities, built from bounds; lower and upper expectations; Γ-maximin, Γ-maximax and α-maxmin; interval dominance, maximality and E-admissibility, the last by a linear programme on calx's simplex.
2. `trees.hpp`: decision trees, folding back, strategies listed and evaluated one by one, and the tree of deciding after a test.
3. `value.hpp`: EVPI and EVSI by preposterior analysis on a table, and for a normal net benefit (moved from tekmerion, which now uses these).
4. `risk.hpp`: lotteries, first- and second-order stochastic dominance, certainty equivalents under constant absolute and relative risk aversion.

A choice is a set: every act within a stated tie of the best is chosen. Criteria under ignorance and ambiguity are offered side by side, each named; none is the library's.

## 3. Alternatives Considered

- **A foundation of decision theory.** The theory is the mathematics of choosing under uncertainty, tyche's by its scope; a foundation apart would split it from the POMDPs it underlies.
- **nomos.** Games, social choice and appraisal are decisions among agents or by institutions; the single decision maker's theory is beneath them, and nomos stands on tyche.
- **One criterion of choice under ambiguity.** Rejected: Γ-maximin, E-admissibility and maximality answer different questions (Troffaes 2007), and which is right is a decision maker's stance, not a theorem.

## 4. Consequences

- **Checks:** `check_tyche_choice` (24) and `check_tyche_trees` (24). They hold: a table on which five criteria under ignorance choose five different acts; Ellsberg's pattern under maxmin expected utility and its impossibility under any single probability; Troffaes's inclusions on 400 random problems, E-admissibility's programme held to 24,000 sampled probabilities; folding back against every strategy on a wildcatter's tree and on random trees; Good's theorem and 0 ≤ EVSI ≤ EVPI on 300 random problems, EVSI computed a second way by a tree; the normal formulas against quadrature; stochastic dominance against the utilities that characterise it.
- **Planned:** influence diagrams evaluated on `graphical`'s networks; Choquet expected utility on `beliefs`; rank-dependent and prospect-theoretic choice (descriptive, and so nomos's); the value of information for POMDPs.
