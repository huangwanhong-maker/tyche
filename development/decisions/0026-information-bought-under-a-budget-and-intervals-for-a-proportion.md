# 0026. Information Bought under a Budget, and Intervals for a Proportion

- **Status:** Accepted, provisionally (2026-11-16)
- **Date:** 2026-11-16
- **Decided by:** Claude, at the infrastructure's request for decisions on evidence in space and time (its ADR 0123): which places to look at first, and the channel of a kind of report counted.

## 1. Context

Two needs came from an archive of evidence about many places.

- **Which tests to run.** Each place has its own decision (treat it or leave it) and a test that could be run before it, at a cost. The budget does not run to them all. `value.hpp` gives one test's EVSI for one decision; nothing chose the tests together.
- **How good a kind of report is.** Its hit and false-alarm rates are counted against a reference: k of n. An estimate needs an honest interval, also at k = 0 and k = n.

## 2. Decision

1. `decision/budget.hpp`: `Informable` (a decision with its table, prior, test and the test's cost) and `buy(problems, budget, charged)` → `Purchase` (each test's analysis and worth; the tests bought; what they gain and spend; the decisions' worth without tests; a bound and whether it is exact).
   - For separate decisions (states independent, utilities adding), the worth of running a set of tests is the sum of their EVSIs, since the preposterior analysis of the product problem factorises. The best set is then a 0/1 knapsack, on calx's (its decision 0018), with its certificate.
   - With `charged`, a cost in the utilities' units is subtracted from what the test gains, and a test not worth its cost is never run.
2. `probability/proportions.hpp`: `wilson_interval(k, n, confidence)` (Wilson 1927: the score test inverted) and `clopper_pearson_interval(k, n, confidence)` (Clopper and Pearson 1934: the two tails inverted exactly, by quantiles of beta laws).

## 3. Alternatives Considered

- **Tests on decisions that share states.** Their worths do not add: two tests of one state are partly redundant. That is sequential, and tyche's POMDP modules hold it; `buy` refuses it by construction, each problem carrying its own table.
- **The Wald interval, p̂ ± z √(p̂(1 − p̂)/n).** It has no width at k = 0 and covers badly near the ends (Brown, Cai and DasGupta 2001).
- **Only one interval.** Wilson's covers about the nominal level on average; Clopper and Pearson's at least the nominal level everywhere, and is wider for it. Both are given, and said apart.

## 4. Consequences

- `check_tyche_budget` (15): the sum of EVSIs held by forming the product problem outright on 200 random pairs; three sites by hand, where only the doubtful one is worth inspecting; every subset of 100 problems up to ten tests, costs charged in half; equal costs.
- `check_tyche_proportions` (14): Wilson's bounds as roots of the score equation; the exact bounds as the points where each tail holds α/2; closed forms at none and all; seven of twenty by hand; exact coverage over a grid of true proportions, Clopper–Pearson's never under the level.
- tyche 1597 checks.
