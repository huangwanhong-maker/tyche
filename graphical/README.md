# graphical: Bayesian Networks over Discrete Variables

*Status: draft (2026-11-14; [decision 0023](../development/decisions/0023-graphical-models-and-belief-functions.md)).* Joint laws of many discrete variables given by a directed acyclic graph and a conditional table for each variable; exact inference by elimination, belief propagation, and the tables learnt from incomplete records.

## 1. Concepts

A Bayesian network writes the joint law of variables X₁, …, X_n as a product of local laws, P(x) = Π_i P(x_i | x_{Pa_i}), one for each variable given its parents in a directed acyclic graph. The graph says which variables are conditionally independent of which (Pearl 1988); the tables say how strongly each depends on its parents.

Inference asks for P(Q | e), the law of some variables given evidence about others. Evidence is hard (a variable is in a known state) or virtual (a likelihood λ over a variable's states, from an observation outside the network). Three ways to answer:

- **Elimination** sums the variables out of the product one at a time. It is exact, and its cost is exponential in the induced width of the order, which min-fill or min-degree keeps small.
- **Belief propagation** passes messages on the factor graph. It is exact on a tree, as the graph of a polytree's tables is; on a graph with cycles it is an approximation, and says so.
- **Max-product elimination** gives the most probable full assignment consistent with the evidence.

The weight of evidence for a hypothesis H: X ∈ S is the likelihood ratio P(e | H) / P(e | ¬H): the posterior odds are the ratio times the prior odds.

When the tables are unknown, EM learns them from records in which some values are missing: the E-step completes each record in expectation, exactly, by elimination; the M-step counts. Dirichlet pseudo-counts turn the maximum likelihood into a maximum a posteriori.

## 2. Content

| Header | What it holds |
|---|---|
| `common.hpp` | `Refusal` (an `std::invalid_argument` with the reason); `Evidence` (hard `observed`, virtual `likelihood`) |
| `factor.hpp` | `Factor` (a scope, its cardinalities, a row-major table); `factor`, `constant`, `product`, `sum_out`, `max_out`, `reduce`, `marginalise`, `reorder`, `total`, `normalise` |
| `network.hpp` | `Variable`, `Node`, `Network` (with `index`, `state`, `arcs`, `probability`); `network` (validated: names, parents, acyclicity by morphe's topological order, rows summing to one), `with_tables`, `evidence` by names, `sample` (ancestral) |
| `elimination.hpp` | `Heuristic` (min-fill, min-degree), `EliminationOptions`, `Plan` and `plan` (order and induced width); `sum_product` on any list of factors; `evidence_probability`, `log_evidence_probability`, `joint_with_evidence`, `joint`, `marginal`, `marginals`; `likelihood_ratio` (one state or a set); `most_probable_explanation` |
| `propagation.hpp` | `PropagationOptions` (rounds, tolerance, damping), `Propagation` (marginals, `tree`, `converged`, `exact`, `iterations`, `last_change`); `propagate` on a network or on any factor graph |
| `learning.hpp` | `Record` (a state or nothing per variable); `EmOptions` (pseudo-counts, tolerance, rounds), `EmResult` (the network learnt, the objective at every step, the expected counts); `em`, `log_likelihood` |
| `graphical.hpp` | all of them |

Conventions (the layout of tables, the order of a product's scope, ties in the most probable explanation, the objective of EM) are in the [glossary](../docs/glossary.md), §8.

## 3. Checks

- `check_bayesian_networks.cpp` (49): the factor operations entry by entry; the refusals of a network (a cycle, named; a row not summing to one, named by its parents' states; an unknown parent; a table of the wrong size; a negative entry; a repeated name; an unknown state); the Asia network (Lauritzen and Spiegelhalter 1988), its marginals with and without evidence, joints, likelihood ratios with the odds identity, virtual evidence and the most probable explanation against enumeration of its joint law, written in the test without the library's factors; impossible evidence refused; min-fill's width of 2 on Asia; log P(e) on a chain of 1100 variables, where P(e) underflows; thirty seeded random networks against enumeration.
- `check_propagation.cpp` (23): belief propagation equal to elimination to 10⁻¹² on forty seeded random polytrees, with and without evidence, reported exact; a pairwise chain given as a factor graph; on the diamond A → B, A → C, B → D, C → D and on Asia, loopy propagation converged and close to the exact marginals given evidence (within 0.011 and 0.016), reported approximate, damping reaching the same fixed point; evidence that cuts the cycle making the result exact; a run stopped early; impossible evidence refused.
- `check_network_em.cpp` (16): ancestral sampling against the exact marginals; on complete records the counts' proportions, and with pseudo-counts (N + a)/(N_row + k a); with a fifth of 20 000 records' values missing at random, the log-likelihood never decreasing, the tables learnt explaining the records at least as well as the true ones and within 1.9 standard errors of them, the objective equal to the records' log-likelihood, the expected counts adding up, the E-step by families equal to the E-step by a record's missing part; with pseudo-counts the log-posterior never decreasing; malformed and impossible records refused.

## 4. Dependencies and References

- **Depends on:** morphe (`combinatorics`: the topological order and ancestors of acyclic relations) and calx (`random`, for sampling).
- **References:**
  - Pearl, J. (1988), *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann;
  - Lauritzen, S. L. and Spiegelhalter, D. J. (1988), Local computations with probabilities on graphical structures and their application to expert systems, *J. R. Statist. Soc. B* 50, 157–224;
  - Zhang, N. L. and Poole, D. (1994), A simple approach to Bayesian network computations, *Proc. 10th Canadian Conference on Artificial Intelligence*, 171–178;
  - Dechter, R. (1996), Bucket elimination: a unifying framework for probabilistic inference, *Proc. UAI* 12, 211–219;
  - Kjærulff, U. (1990), *Triangulation of graphs: algorithms giving small total state space*, Aalborg University, report R 90-09;
  - Shachter, R. D. (1986), Evaluating influence diagrams, *Operations Research* 34, 871–882;
  - Dawid, A. P. (1992), Applications of a general propagation algorithm for probabilistic expert systems, *Statistics and Computing* 2, 25–36;
  - Kschischang, F. R., Frey, B. J. and Loeliger, H.-A. (2001), Factor graphs and the sum-product algorithm, *IEEE Trans. Inf. Theory* 47, 498–519;
  - Murphy, K., Weiss, Y. and Jordan, M. I. (1999), Loopy belief propagation for approximate inference: an empirical study, *Proc. UAI* 15, 467–475;
  - Yedidia, J. S., Freeman, W. T. and Weiss, Y. (2005), Constructing free-energy approximations and generalized belief propagation algorithms, *IEEE Trans. Inf. Theory* 51, 2282–2312;
  - Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977), Maximum likelihood from incomplete data via the EM algorithm, *J. R. Statist. Soc. B* 39, 1–38;
  - Lauritzen, S. L. (1995), The EM algorithm for graphical association models with missing data, *Computational Statistics & Data Analysis* 19, 191–201;
  - Heckerman, D. (1995), *A tutorial on learning with Bayesian networks*, Microsoft Research, MSR-TR-95-06;
  - Koller, D. and Friedman, N. (2009), *Probabilistic Graphical Models*, MIT Press: ch. 9 (variable elimination), ch. 11 (loopy propagation), ch. 13 (MAP inference), ch. 19 (partially observed data).
- **Open:**
  - junction trees, for every marginal from one pass (now one elimination per variable);
  - the Bethe approximation of log P(e) from loopy propagation;
  - continuous and conditional Gaussian variables;
  - structure learning (scores, constraint-based search) and the d-separation of a network's graph, which peira's `graphs` holds for causal graphs;
  - logarithmic factors for networks whose factors underflow within one table.
