# census: Populations With a Genealogy, and the Price Decomposition

*Status: draft (2026-11-23; [decision 0035](../development/decisions/0035-census-records-and-the-price-decomposition.md)).*

## 1. Concepts

A **census** is a population at one time: members, each with a weight (copies, capacity, employment, output, a budget share) and a character (a productivity, a unit cost, a fitness, any measured property). A **record** joins two censuses by a genealogy: each member after names its parents before with transmission shares, and the part of it that descends from no one **arrives**, split by its sources, so that its links and arrivals sum to one. A member before with no descendant **exits**, with its reason; each member after is made by an **operator**. Firms in an industry and candidate solutions in a search are records of the same form, and so are lineages in any evolving population.

The change of the character's share-weighted mean is decomposed exactly by **Price's equation** (Price 1970, 1995): over the members that continue, W̄ ΔZ_C = Cov_σ(w, z) + E_σ(w Δz), selection and transmission. With entry and exit the decomposition needs a **reference** against which entrants and exiters are judged, and the reference carries meaning (Melitz and Polanec 2015):

| Form | Entrants judged against | Exiters judged against | Source |
|---|---|---|---|
| `survivors` | the survivors after | the survivors before | Price; Melitz and Polanec's dynamic Olley–Pakes reference |
| `incumbent` | the mean before | the mean before | Foster, Haltiwanger and Krizan; Holm 2014, eq. 4 |
| `average` | the mean of the two means | the same | Griliches and Regev |
| `declared` | a reference a caller declares | the same | |

Every form is an identity whatever the data. The terms are an **accounting**, never causes: selection is a causal claim only when what makes the covariance is supplied (Hodgson and Knudsen 2006; Frank 2012).

## 2. Content

`include/tyche/census/price.hpp`, templated on the number type (double, or morphe's exact `Rational`):

- `Census`, `Link`, `Arrival`, `Record`, `validate`: a record checked and refused with its fault (shares not summing to one, negative weights, a parent outside the census, an ancestor of zero weight with descendants);
- `decompose(record, form, reference)` → `Decomposition`: selection, transmission (within and cross in the incumbent and declared forms), entry and exit; in the survivors' form Price's fitness form (W̄, the covariance, the expectation) and selection split over nested groups (the multilevel Price equation: between groups at each declared level, then within the finest); transmission credited to operators pair by pair, entry to sources, exit to reasons;
- `second_moment`: the change of the character's variance, from the decomposition of the mean square and the shift of the mean;
- `compose(first, second)`: two steps joined into one span, transmission shares multiplied along lineages, arrivals carried with their sources;
- `by_identity`: a record from two panels keyed by identity, continuing, entering and exiting members found by their ids.

## 3. Checks

`check_tyche_price` (30): a panel of three firms worked by hand in every form; Fisher's theorem (the trait the fitness, copied exactly: Δē = Var(e)/ē); the replicator step; neutral selection; Melitz and Polanec's closed form (every character growing by γ: entry credited 0, s_N γ/2 and s_N γ by the survivors', average and incumbent forms); two levels by hand; a search's outside proposal credited to its source; 200 random records with several parents, partial arrivals, zero weights and two levels (every form an identity in exact rationals, selection by level summing, Price's fitness form exact, the credits summing, the second moment's identity, doubles within 10⁻¹² of rationals); composed spans whose totals add over steps while their selection terms do not (Holm 2014, §4), and a step composed with an identity changing no term; refusals.

## 4. Dependencies

None beyond the standard library; the checks use morphe's `arithmetic` and calx's `random`.

## 5. References

- G. R. Price, "Selection and covariance", *Nature* 227 (1970) 520–521; "The nature of selection", *J. Theor. Biol.* 175 (1995) 389–396.
- S. A. Frank, "Natural selection. IV. The Price equation", *J. Evol. Biol.* 25 (2012) 1002–1019.
- G. M. Hodgson and T. Knudsen, "The nature and units of social selection", *J. Evol. Econ.* 16 (2006) 477–489, appendix.
- J. R. Holm, "The significance of selection and learning for productivity growth", *J. Evol. Econ.* 24 (2014) 1013–1040.
- M. J. Melitz and S. Polanec, "Dynamic Olley–Pakes productivity decomposition with entry and exit", *RAND J. Econ.* 46 (2015) 362–375.
- L. Foster, J. Haltiwanger and C. J. Krizan, "Aggregate productivity growth", in *New Developments in Productivity Analysis*, Chicago 2001, 303–372.
- Z. Griliches and H. Regev, "Firm productivity in Israeli industry 1979–1988", *J. Econometrics* 65 (1995) 175–203.

## 6. Open Questions

- Characters as vectors (the decomposition is linear in the character, so each component is decomposed alone for now).
- Causal credit for entrants, by counterfactual runs or a Shapley value over sources; the accounting credit is exact, not causal.
- Baldwin and Gu's reference (entrants against exiters), and decompositions in levels against logs, which the economics domain's productivity decompositions will state.
