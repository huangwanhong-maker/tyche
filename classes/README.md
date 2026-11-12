# classes: Classes in a Population Sampled

*Status: draft (2026-11-12, [decision 0017](../development/decisions/0017-classes-and-communities.md)).* How many classes a population holds, how evenly they are filled, how a diversity divides among units, and what a smaller or a larger sample would find. Written for the infrastructure's ecology domain (the street trees of a city by species), and general: the classes may be species, words of a vocabulary, k-mers of a library, kinds of business, the events of a canon.

## 1. Concepts

**Diversity as an effective number.** A law p on classes is worth ^qD(p) = (Σ p_i^q)^{1/(1−q)} equally common classes (Hill 1973): the number of classes at q = 0, the exponential of Shannon's entropy at q = 1, the reciprocal of Simpson's concentration at q = 2, the reciprocal of the largest share at q = ∞. The order q says how much the rare classes count; ^qD falls with q, and the profile over q describes a law's evenness at a glance. When classes resemble one another (by a similarity matrix Z, from traits, a phylogeny or a taxonomy), Leinster and Cobbold's ^qD^Z counts a law spread over similar classes as less diverse; with Z = I it is Hill's number.

**Partition among units.** When units (plots, blocks, years) each hold some classes, the pooled diversity gamma is the diversity of a typical unit, alpha, times beta, the effective number of wholly distinct units: 1 when the units are alike, N when they share nothing (Jost 2007). Jost's alpha keeps alpha and beta independent for every q only when the units are weighted equally; with unequal weights only for q = 1. Units of unequal size can instead be weighted by size with Chao, Chiu and Jost's (2014) alpha, which counts their difference in size as part of beta. Beta is normalised to overlaps in [0, 1], of Sørensen's and of Jaccard's type, which for q = 0 and two units are Sørensen's and Jaccard's indices and for q = 2 Morisita and Horn's.

**What a sample does not see.** A sample of individuals (abundance data) or of sampling units (incidence data) misses the rarest classes. The singletons and doubletons (f₁, f₂) carry the information about them: Chao1, S_obs + ((n − 1)/n) f₁²/(2f₂), is a lower bound for the richness that is sharp when the rarest classes are alike; Ĉ_n = 1 − (f₁/n)·x, x = (n − 1)f₁/((n − 1)f₁ + 2f₂), estimates the sample's coverage, the share of the population's individuals that belong to classes seen. Chao, Wang and Jost's estimator of entropy and the unbiased estimate of Σ p² give the asymptotes of orders 1 and 2.

**Rarefaction and extrapolation.** Diversity grows with the size of a sample, so samples are compared at a common size or, better, a common coverage (Chao and Jost 2012). Below the sample's size, the expected Hill numbers of a subsample follow exactly from the hypergeometric law; above it, Chao et al. (2014) extrapolate orders 0, 1 and 2, and the two curves meet at the sample. Coverage along the same curve gives the size at which two samples are equally complete. Bootstrap intervals come from Chao et al.'s bootstrap assemblage, which gives the undetected classes their estimated share.

## 2. Content

All in namespace `tyche::classes`; `classes.hpp` includes every header.

`counts.hpp`:

| Name | Content |
|---|---|
| `abundances(counts)` → `Abundances` | `counts` (X_i > 0), `n`, `f` (f_k), `observed()`, `frequency(k)`, `singletons()`, `doubletons()`; counts as doubles, non-negative integers, zeros dropped |
| `incidences(frequencies, T)`, `incidences(detections)` → `Incidences` | `units` (T), `counts` (Y_i > 0), `total` (U), `Q` (Q_k), `uniques()`, `duplicates()`; from incidence frequencies, or a table of units by classes (any positive entry a detection) |

`diversity.hpp`:

| Name | Content |
|---|---|
| `hill(p, q)`, `renyi(p, q)` | ^qD and H_q = ln ^qD of a law, q ≥ 0 or ∞ |
| `hill_profile(p, orders)` | ^qD at each order |
| `law_of(counts)` | X_i/n |
| `similarity_diversity(p, Z, q)`, `similarity_profile(p, Z, orders)` | Leinster and Cobbold's ^qD^Z (Z in [0, 1], Z_ii = 1) |
| `ordinariness(p, Z)`, `similarity_from_distances(D, u)` | (Zp)_i; Z = e^{−u d} |

`partition.hpp`:

| Name | Content |
|---|---|
| `partition(table, q, Weighting)` → `Partition` | `Weighting::equal` (Jost's alpha, any q) or `Weighting::size` (Chao, Chiu and Jost's alpha, any q); rows are units, columns classes |
| `partition(table, q, weights)` | Jost's alpha with given weights: any q if equal, otherwise q = 1 only (refused for other q) |
| `Partition` | `q`, `weighting`, `given_weights`, `units`, `weights`, `pooled`, `unit_diversity`, `gamma`, `alpha`, `beta`, `overlap` |
| `overlap(beta, N, q)` → `Overlap` | `sorensen` (C_qN), `jaccard` (U_qN), `sorensen_dissimilarity`, `jaccard_dissimilarity` |

`richness.hpp` (abundance data):

| Name | Content |
|---|---|
| `chao1(sample, level, Chao1Form)` → `RichnessEstimate` | `estimate`, `observed`, `undetected`, `variance`, `se`, `lower`, `upper`, `level`, `lognormal`; `Chao1Form::classic` (iNEXT's) or `bias_corrected` (Chao 2005) |
| `undetected(sample, form)`, `detection_ratio(sample)` | f̂₀; x = n f̂₀/(n f̂₀ + f₁) |
| `coverage(sample)`, `turing_coverage(sample)` | Ĉ_n (Chao and Jost 2012, eq. 4a); 1 − f₁/n |
| `ace(sample, threshold = 10)` → `AceEstimate` | `estimate`, `rare`, `abundant`, `rare_individuals`, `coverage_rare`, `cv_squared`, `defined` |

`asymptotic.hpp`:

| Name | Content |
|---|---|
| `entropy_estimate(sample)` | Chao, Wang and Jost's Ĥ, nats |
| `observed_entropy(sample)`, `concentration_estimate(sample)` | the plug-in entropy; Σ X(X − 1)/(n(n − 1)) |
| `asymptotic_diversity(sample, level)` → `Asymptotic` | `observed[3]` and `estimate[3]` for q = 0, 1, 2; `entropy`, `entropy_observed`, `coverage`, `richness` (Chao1 with its interval) |

`rarefaction.hpp`:

| Name | Content |
|---|---|
| `expected_frequencies(sample, m)` | f̂_k(m), k = 0, …, m |
| `interpolated(sample, q, m)` | the rarefied Hill number, 1 ≤ m ≤ n, any finite q |
| `extrapolated(sample, q, m)` | q ∈ {0, 1, 2}, m ≥ n |
| `estimate_at(sample, q, m)` | either, by m |
| `coverage_at(sample, m)`, `size_at_coverage(sample, C)` | coverage along the curve, and its inverse |
| `bootstrap_assemblage(sample)` | the law sampled by the bootstrap |
| `curve(sample, CurveOptions)` → `std::vector<CurvePoint>` | the curves with coverage and bootstrap intervals; `CurveOptions{orders, sizes, endpoint, knots, replicates, level, seed}`; `CurvePoint{size, method, order, estimate, se, lower, upper, coverage, coverage_se, coverage_lower, coverage_upper}`, `Method{interpolated, observed, extrapolated}` |
| `default_sizes(n, endpoint, knots)` | about `knots` sizes from 1 to 2n, n among them |

`incidence.hpp`:

| Name | Content |
|---|---|
| `chao2(sample, level, form)` → `RichnessEstimate` | Chao2 with Chao1's variance and interval, A = (T − 1)/T |
| `undetected`, `detection_ratio`, `coverage` (on `Incidences`) | Q̂₀, x, Ĉ_T |
| `richness_at(sample, t)`, `coverage_at(sample, t)` | interpolated (t ≤ T) and extrapolated richness and coverage along the number of units |
| `bootstrap_detection(sample)` | the detection probabilities of the incidence bootstrap assemblage |
| `richness_curve(sample, IncidenceCurveOptions)` | the richness curve with coverage and bootstrap intervals |

A minimal use:

```cpp
#include <tyche/classes/classes.hpp>
namespace cl = tyche::classes;
const auto trees = cl::abundances(counts_by_species);          // one count per species
const auto chao = cl::chao1(trees);                             // estimate, lower, upper
cl::CurveOptions o;                                             // q = 0, 1, 2; 40 sizes to 2n; 50 replicates
const auto points = cl::curve(trees, o);
const double m = cl::size_at_coverage(trees, 0.95);             // compare two cities at equal coverage
```

## 3. Checks

`check_tyche_diversity` (43): Hill numbers of a uniform law (D at every order), of a dyadic law worked by hand at q = 0, 1, 2, 3, ∞, their limits at q = 1 and ∞ and their monotonicity in q on 100 random laws; Rényi entropy as ln ^qD and Shannon's entropy; Leinster and Cobbold's diversity equal to Hill's when Z = I, to the groups' Hill number for block similarities, to 1 for Z of ones, non-increasing in q, never above Hill's; the partition: beta = 1 for units of one law (equal weights) or of one law and size (size weights), beta = N/^qD(w) for one law at sizes w weighted by size, beta = N for units sharing no class under both weightings, 1 ≤ beta ≤ N on 80 random tables, Jost's q = 1 beta with unequal weights equal to exp I(unit; class); the overlaps equal to Sørensen's, Jaccard's, Whittaker's, Horn's and Morisita and Horn's indices worked by hand for two units, and C_qN = U_{(2−q)N}; refusals.

`check_tyche_richness` (50): Chao1 by hand in both forms and with f₂ = 0; its variance against Chao's closed form, the f₂ = 0 form, and the delta method with derivatives by central differences; the interval without singletons; ACE and the coverage by hand; the identity Σ_k (1/k) C(n − X, k)/C(n − 1, k) = Σ_{k=X}^{n−1} 1/k for n ≤ 40; Chao, Wang and Jost's estimator against its printed formula in long double on both numerical branches; seeded samples: at n = 40 000 Chao1, ACE, Ĥ and the q = 2 asymptote within 1%, 1%, 0.1% and 0.5% of the truth; at n = 1500 Ĉ_n's mean error 0.0002 against the sample's coverage, Ĥ's bias a fifth or less of the plug-in's, Chao1 between S_obs and S under heterogeneity; for equally common classes Chao1 within 2% and its 95% interval covering S; cross-checks against scikit-bio 0.7.4 (Hill numbers, Rényi entropies, ACE, Chao1 up to the factor (n − 1)/n, Good's coverage; `tests/fixtures/make_skbio_fixtures.py`).

`check_tyche_rarefaction` (55): the interpolated Hill numbers of orders 0, ½, 1, 2, 3 and the expected frequency counts equal to the mean over every subsample of every size, enumerated individual by individual (n = 7 and 13); Ŝ_m, Ĉ_m and Σ X(X − 1)/(n(n − 1)) exactly unbiased, by enumerating every multinomial sample of 7 from a law on 4 classes; at m = n both formulas return the sample's value, the extrapolation is continuous there and the increments on either side agree; the extrapolations tend to the asymptotic estimates; coverage 0 at 0, Simpson's at 1, Turing's at n − 1, Ĉ_n at n, never falling, and inverted by `size_at_coverage`; the bootstrap assemblage a law with the undetected holding 1 − Ĉ_n; curves repeatable by seed; seeded samples from a known law: interpolation within 0.5% and extrapolation to 2n within 4% (q = 0) and 1% (q = 1, 2) of what samples of those sizes find, bootstrap intervals missing in at most 15% of 120 samples; a sample of 82 000 runs.

`check_tyche_incidence` (22): Chao2, its variance and Ĉ_T by hand; the interpolated richness equal to the mean over every set of t units (T = 5 and 9); continuity at T and the limit Chao2; monotone richness and coverage; the bootstrap assemblage keeping the expected incidences; seeded Bernoulli samples: rarefied richness unbiased, extrapolation to 2T within 3%, Ĉ_T's mean error 0.001, Chao2 within 2% at T = 400.

## 4. Dependencies

`probability` (the normal quantile of the intervals); calx `linalg` (`Rectangular`, `Matrix`), `random` (`Stream`, `multinomial`) and `special_functions` (log Γ, ψ).

## 5. References

- Chao, A. (1987), Estimating the population size for capture–recapture data with unequal catchability, *Biometrics* 43, 783–791.
- Chao, A. (2005), Species estimation and applications, in *Encyclopedia of Statistical Sciences*, 2nd ed., Wiley.
- Chao, A. and Lee, S.-M. (1992), Estimating the number of classes via sample coverage, *J. Amer. Statist. Assoc.* 87, 210–217.
- Chao, A., Ma, M.-C. and Yang, M. C. K. (1993), Stopping rules and estimation for recapture debugging with unequal failure rates, *Biometrika* 80, 193–201.
- Chao, A. and Jost, L. (2012), Coverage-based rarefaction and extrapolation: standardizing samples by completeness rather than size, *Ecology* 93, 2533–2547.
- Chao, A., Wang, Y. T. and Jost, L. (2013), Entropy and the species accumulation curve: a novel entropy estimator via discovery rates of new species, *Methods Ecol. Evol.* 4, 1091–1100.
- Chao, A., Gotelli, N. J., Hsieh, T. C., Sander, E. L., Ma, K. H., Colwell, R. K. and Ellison, A. M. (2014), Rarefaction and extrapolation with Hill numbers: a framework for sampling and estimation in species diversity studies, *Ecol. Monogr.* 84, 45–67.
- Chao, A., Chiu, C.-H. and Jost, L. (2014), Unifying species diversity, phylogenetic diversity, functional diversity, and related similarity and differentiation measures through Hill numbers, *Annu. Rev. Ecol. Evol. Syst.* 45, 297–324.
- Chiu, C.-H., Jost, L. and Chao, A. (2014), Phylogenetic beta diversity, similarity, and differentiation measures based on Hill numbers, *Ecol. Monogr.* 84, 21–44.
- Colwell, R. K., Mao, C. X. and Chang, J. (2004), Interpolating, extrapolating, and comparing incidence-based species accumulation curves, *Ecology* 85, 2717–2727.
- Colwell, R. K., Chao, A., Gotelli, N. J., Lin, S.-Y., Mao, C. X., Chazdon, R. L. and Longino, J. T. (2012), Models and estimators linking individual-based and sample-based rarefaction, extrapolation and comparison of assemblages, *J. Plant Ecol.* 5, 3–21.
- Good, I. J. (1953), The population frequencies of species and the estimation of population parameters, *Biometrika* 40, 237–264.
- Hill, M. O. (1973), Diversity and evenness: a unifying notation and its consequences, *Ecology* 54, 427–432.
- Hurlbert, S. H. (1971), The nonconcept of species diversity: a critique and alternative parameters, *Ecology* 52, 577–586.
- Jost, L. (2006), Entropy and diversity, *Oikos* 113, 363–375.
- Jost, L. (2007), Partitioning diversity into independent alpha and beta components, *Ecology* 88, 2427–2439.
- Leinster, T. and Cobbold, C. A. (2012), Measuring diversity: the importance of species similarity, *Ecology* 93, 477–489.
- Rényi, A. (1961), On measures of entropy and information, *Proc. 4th Berkeley Symp. Math. Statist. Probab.* 1, 547–561.
- Shen, T.-J., Chao, A. and Lin, C.-F. (2003), Predicting the number of new species in further taxonomic sampling, *Ecology* 84, 798–804.

## 6. Open Questions

1. Incidence-based diversity of orders 1 and 2 (Chao et al. 2014), and Hill numbers of any order estimated from a sample (Chao and Jost 2015, *Methods Ecol. Evol.* 6, 873–882).
2. ICE and the variance of ACE; Chao1's variance with the covariance of S_obs as EstimateS computes it for the bias-corrected form.
3. Phylogenetic and functional diversity estimated from samples (Chao, Chiu and Jost 2014 survey them), beyond Leinster and Cobbold's similarity of a known law; the partition of ^qD^Z.
4. Comparisons of two samples (the shared classes, Chao's estimator of shared richness), and the estimation of beta from samples.
