# networks: Relational Data

*Status: draft (2026-11-13, [decision 0019](../development/decisions/0019-relational-data.md)).* The statistics of who is tied to whom: the description of a network, the mixing of ties by attributes, brokerage, and inference that respects the dependence of relational data. Written for the infrastructure's sociology domain (personal and social networks) and its bridge of social provision (who cares for whom, who helps which household), and general: the actors may be persons, households, firms, places or genes.

## 1. Concepts

**A network is a set of ties.** n actors and the ties among them, undirected (a symmetric relation: kin, co-residence) or directed (a sender and a receiver: naming a friend, giving help), each with an optional positive weight; node attributes (a category, a number) are passed beside the network to the functions that use them. Loops are refused, a tie given twice is one tie with its weights summed, and the underlying graph of a directed network joins two actors when either names the other (Wasserman and Faust 1994; Newman 2018).

**Description.** Density, reciprocity (and Garlaschelli and Loffredo's ρ, reciprocity against what the density alone would give), the dyad census (mutual, asymmetric, null) and the triad census of the sixteen isomorphism classes of directed triads, in which the theories of balance, ranking and transitivity are tested (Holland and Leinhardt 1976); local clustering and transitivity, the share of two-paths that close; weak and strong components; k-cores; degree distributions.

**Mixing and homophily.** Whether ties join actors alike: Newman's assortativity by a category (from the mixing matrix), by a number or by degree, the Pearson correlation of the attribute at the two ends of the ties; Krackhardt and Stern's E–I index of external against internal ties, with the index of the complete network on the same groups as the reference its group sizes impose; inbreeding homophily, the excess of a group's ties to itself over its share of the population.

**Brokerage.** Burt's structural holes: an actor whose contacts are not tied to one another brokers between them. Constraint measures how much an actor's relations are invested, directly and through each other, in each contact; effective size counts the contacts less their redundancy (Burt 1992).

**Inference with relational dependence.** The cells of a dyadic matrix share actors, so they are not independent observations. The quadratic assignment procedure tests a correlation or a regression over cells against relabellings of the actors, which keep every dependence within rows and columns: QAP correlation, and MRQAP by Y-permutation or by double semi-partialling (Dekker, Krackhardt and Snijders 2007), with p-values reproducible by seed or exact over all n! relabellings of small networks. Conditional uniform graph tests set any statistic against networks drawn uniformly with the observed number of ties, or with the observed degrees by double-edge swaps.

## 2. Content

All in namespace `tyche::networks`; `networks.hpp` includes every header.

`network.hpp`:

| Name | Content |
|---|---|
| `Tie` | `std::array<std::size_t, 2>`, (tail, head) |
| `network(n, ties, directed, weights = {})` → `Network` | `nodes`, `directed`, `weighted`, `ties` (unique, sorted), `weights`, `out`, `in`, `out_weight`, `in_weight`; `size()`, `tied(i, j)`, `weight(i, j)` |
| `from_adjacency(A, directed)`, `adjacency(g)` | from and to a square `Matrix` of weights |
| `underlying(g)`, `neighbourhoods(g)` | the undirected network of either direction (weights summed); sorted neighbours in it |
| `induced(g, nodes)`, `ego_network(g, ego, &members)` | subnetworks, renumbered |
| `degrees(g)` → `Degrees` | `out`, `in`, `total`, `neighbours`, `out_strength`, `in_strength`, `strength` |

`descriptive.hpp`:

| Name | Content |
|---|---|
| `density(g)` | m/(n(n − 1)), or 2m/(n(n − 1)) undirected |
| `reciprocity(g)` → `Reciprocity` | `arc` (2M/m), `dyadic` (M/(M + A)), `garlaschelli_loffredo` (ρ), `density`; directed only |
| `dyad_census(g)` → `DyadCensus` | `mutual`, `asymmetric`, `null` |
| `triad_census(g)` → `TriadCensus` | `std::array<std::uint64_t, 16>` in the order of `triad_names` (003, 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201, 120D, 120U, 120C, 210, 300) |
| `triad_type(g, a, b, c)`, `triad_type(arc)` | the class of one triad |
| `triangles(g)`, `local_clustering(g)`, `average_clustering(g)`, `transitivity(g)` | on the underlying graph |
| `weak_components(g)`, `strong_components(g)` → `Components` | `label`, `sizes`, `count()`, `largest()` |
| `core_numbers(g)`, `k_core(g, k)` | on the underlying graph |
| `degree_distribution(degrees)` → `DegreeDistribution` | `share` (p_k), `at_least`, `mean`, `variance` |

`mixing.hpp`:

| Name | Content |
|---|---|
| `mixing_matrix(g, category, K = 0, weighted = false)` | e_ab, K × K `Rectangular` summing to 1 |
| `categorical_assortativity(g, category, weighted)` | (Σ e_aa − Σ a_a b_a)/(1 − Σ a_a b_a) |
| `numeric_assortativity(g, value, weighted)` | Pearson's correlation at the ends of the ties |
| `degree_assortativity(g, DegreeEnds = out_in, strength = false)` | by degree; `DegreeEnds{out_in, out_out, in_in, in_out}` for directed networks |
| `ei_index(g, category, weighted)` → `EiIndex` | `index`, `external`, `internal`, `group` (per category), `complete` (the index of the complete network on the groups) |
| `inbreeding_homophily(g, category, weighted)` → `Homophily` | `members`, `share` (w_g), `same` (H_g), `inbreeding` ((H_g − w_g)/(1 − w_g)) |

`brokerage.hpp`:

| Name | Content |
|---|---|
| `brokerage(g)` → `Brokerage` | `contacts`, `constraint`, `effective_size`, `efficiency`; NaN for an isolated node |

`qap.hpp`:

| Name | Content |
|---|---|
| `qap_correlation(A, B, QapOptions)` → `QapCorrelation` | `r`, `p_greater`, `p_less`, `p_two_sided`, `permutations`, `exact`; `QapOptions{permutations = 999, seed, exact, symmetric}` |
| `mrqap(Y, {X₁, …}, MrqapOptions)` → `Mrqap` | `fit` (a `regression::LinearFit` over the cells), `p_two_sided`, `p_greater`, `p_less` per coefficient (NaN for the intercept), `cells`, `permutations`, `exact`, `method`; `MrqapOptions{method = double_semi_partialling or y_permutation, intercept, permutations, seed, exact, symmetric}` |
| `dsp_statistic(Y, X, column, relabelling, intercept, symmetric)` | the double semi-partialling t at one relabelling |

`null_models.hpp`:

| Name | Content |
|---|---|
| `uniform_network(n, m, directed, stream)` | uniform on the networks of n nodes and m ties |
| `degree_preserving(g, attempts, stream, &accepted)` | double-edge swaps keeping every degree |
| `cug_test(g, statistic, CugOptions)` → `CugTest` | `observed`, `values`, `mean`, `sd`, `p_greater`, `p_less`, `draws`, `swaps`, `conditioning`; `CugOptions{conditioning = ties or degrees, draws = 1000, seed, swaps_per_tie = 10}` |

A minimal use, a personal network and a test of homophily:

```cpp
#include <tyche/networks/networks.hpp>
namespace nw = tyche::networks;
const auto g = nw::network(n, ties, /*directed*/ true);
const auto census = nw::triad_census(g);                                  // the sixteen classes
const double r = nw::categorical_assortativity(g, gender);                // homophily by a category
const auto b = nw::brokerage(g);                                          // b.constraint[i]
const auto test = nw::cug_test(g, [&](const nw::Network& h) { return nw::categorical_assortativity(h, gender); },
                               {.conditioning = nw::Conditioning::degrees});
```

## 3. Checks

`check_tyche_network_descriptive` (41): construction (duplicate ties merged, loops and zero weights refused, adjacency round trip, the underlying network, ego networks, degrees); closed forms on a triangle, a star, K₆, directed 3- and 5-cycles, a directed path and a hand-worked reciprocity; the triad classifier on all 64 labelled triads, whose classes have the sizes 3!/|Aut| and are invariant under relabelling, and on the example of each class; Batagelj and Mrvar's census equal to the classification of every triple of 30 random networks, both censuses summing to C(n, 2) and C(n, 3); triangles, transitivity, strong and weak components and core numbers against brute force; cross-checks against networkx 3.7 (`tests/fixtures/make_networkx_fixtures.py`: Zachary's karate club and two seeded random networks; transitivity, clustering, density, reciprocity, triangles, core numbers, the triad census, components).

`check_tyche_mixing` (28): degree assortativity equal to Newman's (2002) form over edges and to the Pearson correlation of degrees at both ends; +1 on disjoint cliques, −1 on a star and K₂,₅; categorical assortativity +1 and −1 on the same constructions; directed degree assortativity by hand; the E–I index, group indices, complete-network reference and inbreeding homophily worked by hand; Burt's constraint 1/k at a star's centre and 9/8 in a triangle, effective sizes k and 1, Borgatti's n − 2t/n on 20 random networks; cross-checks against networkx's assortativities, constraint and effective size (directed and weighted among them).

`check_tyche_qap` (32): QAP correlation equal to Pearson's over the cells, its exact p-values equal to a count over all 120 relabellings of a 5-node matrix, approached by 20,000 random ones, reproducible by seed, the same from the pairs of symmetric data, and near uniform under independence; MRQAP coefficients as least squares over the cells, the double semi-partialling statistic at the identity equal to the full t, exact p-values (Y-permutation and double semi-partialling) equal to brute-force counts, a real effect found by both procedures, the false-positive rate of an irrelevant correlated matrix near its level; G(n, m) with exactly m ties and uniform over the 20 graphs of three ties on four nodes; swaps keeping every degree, reaching all 70 two-regular graphs on six nodes and drawing two triangles with probability 1/7; CUG tests reproducible by seed, placing a ring lattice's transitivity above every draw under both conditionings and averaging ρ to 0 over random digraphs.

## 4. Dependencies

`regression` (least squares for MRQAP); calx `linalg` (`Matrix`, `Rectangular`) and `random` (`Stream`); morphe `combinatorics` (the spanning forest of the weak components).

## 5. References

- Batagelj, V. and Mrvar, A. (2001), A subquadratic triad census algorithm for large sparse networks with small maximum degree, *Social Networks* 23, 237–243.
- Batagelj, V. and Zaveršnik, M. (2011), Fast algorithms for determining (generalized) core groups in social networks, *Adv. Data Anal. Classif.* 5, 129–145.
- Besag, J. and Clifford, P. (1989), Generalized Monte Carlo significance tests, *Biometrika* 76, 633–642.
- Borgatti, S. P. (1997), Structural holes: unpacking Burt's redundancy measures, *Connections* 20, 35–38.
- Burt, R. S. (1992), *Structural Holes: The Social Structure of Competition*, Harvard University Press.
- Butts, C. T. (2008), Social network analysis with sna, *J. Stat. Softw.* 24(6).
- Coleman, J. S. (1958), Relational analysis: the study of social organizations with survey data, *Hum. Organ.* 17, 28–36.
- Currarini, S., Jackson, M. O. and Pin, P. (2009), An economic model of friendship: homophily, minorities, and segregation, *Econometrica* 77, 1003–1045.
- Davis, J. A. and Leinhardt, S. (1972), The structure of positive interpersonal relations in small groups, in *Sociological Theories in Progress* 2, 218–251.
- Dekker, D., Krackhardt, D. and Snijders, T. A. B. (2007), Sensitivity of MRQAP tests to collinearity and autocorrelation conditions, *Psychometrika* 72, 563–581.
- Garlaschelli, D. and Loffredo, M. I. (2004), Patterns of link reciprocity in directed networks, *Phys. Rev. Lett.* 93, 268701.
- Holland, P. W. and Leinhardt, S. (1970), A method for detecting structure in sociometric data, *Amer. J. Sociol.* 76, 492–513.
- Holland, P. W. and Leinhardt, S. (1976), Local structure in social networks, *Sociol. Methodol.* 7, 1–45.
- Hubert, L. J. and Schultz, J. (1976), Quadratic assignment as a general data analysis strategy, *Br. J. Math. Statist. Psychol.* 29, 190–241.
- Krackhardt, D. (1987), QAP partialling as a test of spuriousness, *Social Networks* 9, 171–186.
- Krackhardt, D. (1988), Predicting with networks: nonparametric multiple regression analysis of dyadic data, *Social Networks* 10, 359–381.
- Krackhardt, D. and Stern, R. N. (1988), Informal networks and organizational crises: an experimental simulation, *Soc. Psychol. Q.* 51, 123–140.
- Maslov, S. and Sneppen, K. (2002), Specificity and stability in topology of protein networks, *Science* 296, 910–913.
- Milo, R., Kashtan, N., Itzkovitz, S., Newman, M. E. J. and Alon, U. (2003), On the uniform generation of random graphs with prescribed degree sequences, arXiv cond-mat/0312028.
- Newman, M. E. J. (2002), Assortative mixing in networks, *Phys. Rev. Lett.* 89, 208701.
- Newman, M. E. J. (2003), Mixing patterns in networks, *Phys. Rev. E* 67, 026126.
- Newman, M. E. J. (2018), *Networks*, 2nd ed., Oxford University Press.
- Phipson, B. and Smyth, G. K. (2010), Permutation p-values should never be zero, *Stat. Appl. Genet. Mol. Biol.* 9, 39.
- Rao, A. R., Jana, R. and Bandyopadhyay, S. (1996), A Markov chain Monte Carlo method for generating random (0, 1)-matrices with given marginals, *Sankhyā A* 58, 225–242.
- Seidman, S. B. (1983), Network structure and minimum degree, *Social Networks* 5, 269–287.
- Tarjan, R. E. (1972), Depth-first search and linear graph algorithms, *SIAM J. Comput.* 1, 146–160.
- Taylor, R. (1981), Constrained switchings in graphs, in *Combinatorial Mathematics VIII*, Lecture Notes in Math. 884, 314–336.
- Wasserman, S. and Faust, K. (1994), *Social Network Analysis: Methods and Applications*, Cambridge University Press.
- Watts, D. J. and Strogatz, S. H. (1998), Collective dynamics of 'small-world' networks, *Nature* 393, 440–442.

## 6. Open Questions

1. **Models of networks:** exponential random graph models (and their MCMC MLE), stochastic block models, latent space models, and stochastic actor-oriented models of network change; the triad census's expectations under the U|MAN null (Holland and Leinhardt's conditional distributions) in closed form.
2. **Personal-network data:** compositional and structural measures across many ego networks, and multilevel models of alters within egos.
3. **Combinatorial pieces that belong to morphe:** strong components and core numbers are written here because morphe was being changed by another session; they name no statistical object and may move to morphe's `combinatorics` (weak components already come from there).
4. **Directed degree-preserving draws:** triangle reversals to make the chain irreducible, and a diagnostic of mixing for the number of swaps.
5. **Weighted measures:** weighted clustering (Onnela et al., Barrat et al.), strength distributions, and generalised cores.
