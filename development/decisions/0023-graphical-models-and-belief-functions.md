# 0023. Graphical Models and Belief Functions

- **Status:** Accepted, provisionally (2026-11-14).
- **Date:** 2026-11-14
- **Decided by:** the infrastructure's new foundation for evidence, *tekmerion*, being written beside this change, which weighs testimony and records for the history domain's statements and for any science that reasons from evidence. It needs two kinds of reasoning under uncertainty that name no object in nature: probabilistic networks of discrete variables, in which a body of evidence is weighed by its likelihood ratio, and belief functions, in which evidence may support a set of answers without choosing among them. Both are mathematics of chance, and so tyche's.

## 1. Context

tyche held distributions, estimators and decisions, but no joint law of many discrete variables given by a graph, and no calculus of evidence other than probability. The evidence foundation will ask: given what was observed, how probable is each answer; by how much does an observation favour one answer over the rest; which full account of the case is most probable; how should tables be learnt from incomplete records; and, where a source commits only to a set of answers, what it implies, what it leaves possible, and how independent sources combine. pgmpy and pyAgrum answer the first questions in Python, and pyds or the R package `ibelief` the last; the infrastructure's C++ engines need them in C++.

## 2. Decision

**A module `graphical`** (namespace `tyche::graphical`), on morphe's `combinatorics` (the topological order of an acyclic relation, and ancestors) and calx's `random`:

- **factors** over ordered sets of discrete variables, row-major, with product, summing and maximising out, reduction by evidence, marginalisation, reordering and normalisation;
- **networks**, validated with reasons: names, parents, acyclicity (the cycle named), tables of the right size whose rows sum to one within a stated tolerance (the row named by its parents' states);
- **exact inference by variable elimination** (Zhang and Poole 1994) with min-fill or min-degree orders, barren variables pruned, factors rescaled with their logarithms kept apart so that P(e) may lie below the smallest double: P(e), P(Q, e), P(Q | e) (refused when P(e) = 0), every marginal, the likelihood ratio P(e | X ∈ S) / P(e | X ∉ S) with the prior and posterior, and the most probable explanation by max-product elimination; hard and virtual evidence;
- **belief propagation** on any factor graph or a network's: exact on forests (polytrees), loopy with damping and a limit on rounds otherwise, reporting whether it converged, its last change, and whether the result is exact;
- **EM** for the tables from records with missing values (Dempster, Laird and Rubin 1977; Lauritzen 1995), the E-step exact by elimination, optional Dirichlet pseudo-counts for the maximum a posteriori, the objective recorded at every step.

**A module `beliefs`** (namespace `tyche::beliefs`), on morphe's `combinatorics` (the zeta and Möbius transforms on the lattice of subsets, which morphe already held):

- **mass functions** on frames of up to 30 elements, focal sets as bit masks, validated; normal or, where asked, with m(∅) > 0 (the transferable belief model);
- **belief, plausibility, commonality, implicability** at a set and on every subset (frames of up to 20), and the Möbius transforms back, refusing a table that is no belief function;
- **combination**: Dempster's rule with the conflict reported and total conflict refused, the unnormalised conjunctive rule, Yager's rule, the disjunctive rule; discounting by reliability, normalisation, Dempster's conditioning;
- **decision**: the pignistic transform, the bounds [Bel(A), Pl(A)] on P(A) over the credal set, the lower and upper expectations of a function (the Choquet integrals), membership of the credal set, the contour; Bayesian, consonant, vacuous and normal mass functions recognised.

The conventions that a choice was needed for (the layout of tables, the order of a product's scope, ties, the objective of EM, the treatment of m(∅) and of the conflict) are stated in `docs/glossary.md`, §8 and §9.

## 3. Alternatives Considered

- **The evidence foundation holding these methods.** Rejected by the infrastructure's rule of reuse: networks and belief functions serve diagnosis, reliability, ecology and genomics as much as the weighing of testimony. tekmerion will hold what is particular to evidence (sources, their grades and dependence) on top of them.
- **Junction trees** (Lauritzen and Spiegelhalter 1988) for exact inference. Deferred: elimination answers every query exactly; a junction tree gives all marginals in one pass and becomes worth its complexity when networks grow. The open question is recorded in the module.
- **Borrowing pgmpy, pyAgrum or a belief-function library.** The Python packages cannot be called from the C++ engines; pyAgrum's aGrUM core is C++, but adopting it would need its licence checked and would make its interfaces tyche's. The methods are short and written here from the papers.
- **Belief functions on product frames and valuation networks** (Shenoy and Shafer). Deferred until a domain needs evidence spread over several variables at once.
- **A threshold for total conflict.** Rejected: K = 1 is decided by whether any focal sets meet, so near-total conflict is combined and reported, never silently refused.

## 4. Consequences

Four check programs, 138 checks, all passing:

- `check_tyche_bayesian_networks` (49): factors by hand; the refusals; the Asia network (Lauritzen and Spiegelhalter 1988) and thirty seeded random networks against enumeration of the joint law written in the test without the library's factors; impossible evidence refused; log P(e) where P(e) underflows;
- `check_tyche_propagation` (23): propagation equal to elimination on forty random polytrees to 10⁻¹²; loopy propagation on a diamond and on Asia close to exact and reported approximate; evidence that cuts a cycle making it exact;
- `check_tyche_network_em` (16): the counts' proportions on complete records; with missing values the objective never decreasing, the tables learnt within a few standard errors of the truth; the refusals;
- `check_tyche_belief_functions` (50): the identities of belief and plausibility, the Möbius transforms, Dempster's rule against its definition and its algebra, Bayes's product as a special case, Zadeh's example, discounting, conditioning, consonance, the pignistic law, and the Choquet expectations against a linear programme over the credal set (calx's simplex method) and against the laws of all orderings of the frame.

One fact found while checking is recorded in the checks: with no evidence, loopy propagation on a Bayesian network gives the exact marginals, since every message toward a parent is uniform; loopy error appears only with evidence below a cycle.

The evidence foundation can now weigh evidence by likelihood ratios in networks and combine sources as belief functions, without numerical code of its own.
