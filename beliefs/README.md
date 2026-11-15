# beliefs: Dempster–Shafer Belief Functions

*Status: draft (2026-11-14; [decision 0023](../development/decisions/0023-graphical-models-and-belief-functions.md)).* Evidence that commits belief to sets of answers rather than to single ones, on a finite frame: belief and plausibility, their combination, and the probabilities and expectations they bound.

## 1. Concepts

A frame Ω holds the possible answers to one question. A body of evidence is a mass function m: it gives m(A) to a set A of answers, which the evidence supports without saying which of A's answers is true. A probability is the special case in which every mass sits on a single answer (a Bayesian mass function); ignorance is m(Ω) = 1.

From m come two measures of A:

- **belief** Bel(A), the mass of the sets inside A: what the evidence implies;
- **plausibility** Pl(A), the mass of the sets meeting A: what it leaves possible.

They bound every probability compatible with the evidence: Bel(A) ≤ P(A) ≤ Pl(A), and the set of such P, the credal set, gives lower and upper expectations of any function (Choquet integrals).

Evidence from independent sources combines by intersecting what each supports. Dempster's rule renormalises away the conflict K, the mass both sources together give to nothing; when K is large this can mislead (Zadeh's example), so the conflict is always reported, and the unnormalised rule of the transferable belief model, Yager's rule and the disjunctive rule are offered beside it. A source trusted only in part is discounted toward ignorance before it is combined. To decide, the pignistic transform shares each focal set's mass equally among its elements.

## 2. Content

| Header | What it holds |
|---|---|
| `common.hpp` | `Set` (a subset as a bit mask), `max_frame` (30), `dense_frame` (20), `full`, `singleton`, `cardinality`, `complement`, `within`; `Frame` with named elements and `frame`; `Refusal` |
| `mass.hpp` | `Focal`, `MassOptions` (tolerance, `allow_empty`), `Mass` (m(A), m(∅)); `mass` (validated), `vacuous`, `categorical`, `simple_support`, `bayesian`; `belief`, `plausibility`, `commonality`, `implicability`; `is_normal`, `is_bayesian`, `is_consonant`, `is_vacuous` |
| `transforms.hpp` | on every subset: `to_vector`, `belief_function`, `plausibility_function`, `commonality_function`; back by the Möbius transforms: `from_vector`, `from_belief`, `from_plausibility`, `from_commonality` |
| `combination.hpp` | `Combined` (the mass function and the conflict K); `conjunctive`, `dempster` (of two or of many), `yager`, `disjunctive`; `normalise`, `discount`, `condition` |
| `decision.hpp` | `Interval`; `probability_bounds`, `lower_expectation`, `upper_expectation`, `in_credal_set`, `pignistic`, `contour` |
| `beliefs.hpp` | all of them |

Conventions (the masks, the treatment of m(∅), where the conflict goes) are in the [glossary](../docs/glossary.md), §9.

## 3. Checks

`check_belief_functions.cpp` (50):

- the refusals of malformed mass functions (masses not summing to one, negative, outside the frame, m(∅) > 0 unless allowed; frames too large) and of a table that is no belief function, the set named;
- on sixty seeded random mass functions over a frame of five: Bel ≤ Pl and Bel(A) + Pl(Ω ∖ A) = 1; the point and tabulated functions alike; the commonality by its definition; the Möbius transforms inverting exactly;
- Dempster's rule against its definition summed over all pairs of subsets, commutative and associative, of three at once equal to twice; the conjunctive rule multiplying commonalities and the disjunctive rule implicabilities;
- on Bayesian masses, Dempster's rule as Bayes's normalised product;
- Zadeh's example: its conflict 1 − 0.01², Dempster's rule giving full belief to what both sources thought nearly impossible, Yager's rule giving it 10⁻⁴; total conflict refused;
- discounting at α = 1 (the identity), α = 0 (the vacuous function) and α = 0.7; conditioning against the formula of conditional belief; a consonant function's plausibility from its contour;
- the pignistic law summing to one and lying in the credal set;
- the Choquet lower and upper expectations against a linear programme over the credal set (calx's simplex method) and against the laws induced by every ordering of the frame (Shapley 1971), and [Bel(A), Pl(A)] as the range of P(A) over those laws.

## 4. Dependencies and References

- **Depends on:** morphe (`combinatorics`: the zeta and Möbius transforms on the lattice of subsets). The checks also use calx's `optimisation` and `random`.
- **References:**
  - Dempster, A. P. (1967), Upper and lower probabilities induced by a multivalued mapping, *Ann. Math. Statist.* 38, 325–339;
  - Shafer, G. (1976), *A Mathematical Theory of Evidence*, Princeton University Press;
  - Smets, P. and Kennes, R. (1994), The transferable belief model, *Artificial Intelligence* 66, 191–234;
  - Smets, P. (1993), Belief functions: the disjunctive rule of combination and the generalized Bayesian theorem, *Int. J. Approx. Reasoning* 9, 1–35;
  - Yager, R. R. (1987), On the Dempster–Shafer framework and new combination rules, *Information Sciences* 41, 93–137;
  - Zadeh, L. A. (1984), Review of *A Mathematical Theory of Evidence*, *AI Magazine* 5(3), 81–83;
  - Kennes, R. (1992), Computational aspects of the Möbius transformation of graphs, *IEEE Trans. Systems, Man and Cybernetics* 22, 201–223;
  - Shapley, L. S. (1971), Cores of convex games, *Int. J. Game Theory* 1, 11–26;
  - Chateauneuf, A. and Jaffray, J.-Y. (1989), Some characterizations of lower probabilities and other monotone capacities through the use of Möbius inversion, *Math. Social Sci.* 17, 263–283.
- **Open:**
  - other rules of combination (Dubois and Prade's, the cautious rule for dependent sources, PCR5);
  - contextual discounting, and the generalised Bayesian theorem;
  - belief functions on product frames (marginalisation and vacuous extension), and valuation networks that propagate them as `graphical` propagates probabilities;
  - measures of conflict and of non-specificity.
