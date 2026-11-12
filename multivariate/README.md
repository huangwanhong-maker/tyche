# multivariate: Factorisations, Components, Clusters and Communities

*Status: draft (2026-11-01): non-negative matrix factorisation by Lee and Seung's multiplicative updates, and the stability of a factorisation's rank by resampling and constrained clustering ([decision 0006](../development/decisions/0006-change-points-penalties-and-factorisations.md)).* Mixtures of binomials with known scales by EM, their order by BIC (for ADR 0064's clones). Since 2026-11-12, communities: dissimilarities between samples of counts, principal coordinates, PERMANOVA, the test of dispersions and the Mantel test ([decision 0017](../development/decisions/0017-classes-and-communities.md)). Principal components, factor analysis, canonical correlation, k-means and other mixtures are planned.

## 1. Concepts

A non-negative matrix V (counts of m kinds of event in n samples) is often a superposition: each sample's counts the sum of a few processes' contributions, each process with its own distribution over the kinds. Then V ≈ WH with W (m × k) the processes' distributions and H (k × n) their activities, all entries non-negative. Non-negativity is what makes the parts interpretable: they can only add, never cancel, so a factor is a pattern that appears, not a contrast.

The factorisation minimises a divergence between V and WH: the squared Frobenius distance, or the generalised Kullback–Leibler divergence, which is the negative log-likelihood of Poisson counts with means WH. Lee and Seung's multiplicative updates multiply each entry by a ratio of non-negative terms, so the factors stay non-negative, and each objective never increases along its own updates; but the problem is not convex, and the updates find local minima. The factorisation is also unique at most up to a permutation and a positive rescaling of its k components, and only when the factors are sufficiently sparse.

How many components the data determine is a question of stability. If V is resampled within its noise (each count Poisson about itself, or each sample's counts multinomially redrawn with its total kept) and factorised again and again, the components of a well-determined rank come back the same each time; those of too high a rank wander. Clustering all the replicates' components, one of each replicate in each cluster, and measuring how tight and separated the clusters are (by silhouettes under the cosine distance) gives each rank its stability; the reconstruction error gives its fit. This is how mutational signatures are extracted from cancer genomes (Alexandrov et al. 2013; Islam et al. 2022), and it applies to any non-negative data.

Samples of counts over the same classes (plots by species, blocks by kinds of business) are compared by a dissimilarity: Bray–Curtis for abundances, Jaccard and Sørensen for presence, Hellinger and chord distances for compositions that should be Euclidean (Legendre and Legendre 2012, ch. 7). A dissimilarity matrix is drawn as points by principal coordinates: the double-centred matrix −½ J d² J is the Gram matrix of points whose distances are d exactly when d is Euclidean, and otherwise has negative eigenvalues, which Lingoes's or Cailliez's constant removes (Gower 1966). Whether groups of samples differ is then a question about the matrix alone: PERMANOVA's pseudo-F compares the sums of squared dissimilarities within and among groups, which for Euclidean distances is the ANOVA F, and its law under exchangeable labels comes from relabelling (Anderson 2001); since F also responds to differences in spread, the test of dispersions compares the distances to the groups' centroids (Anderson 2006). The Mantel test asks whether two matrices among the same objects agree, by the correlation of their entries and its law under relabelling one matrix's objects (Mantel 1967).

## 2. Content

`nmf.hpp` (namespace `tyche::multivariate`):

| Name | Content |
|---|---|
| `nmf(V, k, NmfOptions)` → `Nmf` | the factorisation of rank k from seeded random starts (the best of `restarts`); `NmfOptions{loss, max_iterations, min_iterations, check_every, tolerance, restarts, seed, normalise}`, `NmfLoss{frobenius, kullback_leibler}` |
| `nmf_from(V, W0, H0, NmfOptions)` → `Nmf` | the updates from given factors |
| `Nmf` | `W` (m × k), `H` (k × n), `objective`, `iterations`, `converged`, `restart`, `restart_objectives`, `trace` (the objective at each check) |
| `nmf_objective(V, W, H, loss)` | ‖V − WH‖²_F or D(V‖WH) |
| `normalise_factors(W, H)` | W's columns to sum 1, H's rows rescaled (WH unchanged) |
| `nmf_stability(V, k, StabilityOptions)` → `RankStability` | R resampled factorisations, clustered by calx's assignment, with silhouettes: `signatures` (W̄), `exposures` (H̄ by NNLS), `stability` and `spread` per cluster, `average_stability`, `minimum_stability`, `frobenius_error`, `relative_error`, `mean_objective`, `stable`, `rounds`, `cluster` |
| `nmf_rank_stability(V, k_min, k_max, StabilityOptions)` | the same for each rank |
| `StabilityOptions` | `replicates`, `resampling` (`none`, `multinomial`, `poisson`), `normalisation` (`none`, `log2`, `hundred_x`), `nmf`, `seed`, `max_rounds`, `average_threshold` (0.80), `minimum_threshold` (0.20) |
| `cosine_similarity(a, b)`, `silhouette(D, labels, far)` | the similarity of two vectors; Rousseeuw's silhouette widths from a distance matrix |

`binomial_mixture.hpp` (written for the genomics domain's clones of cancer cell fractions, ADR 0064):

| Name | Content |
|---|---|
| `fit_binomial_mixture(k, n, c, BinomialMixtureOptions)` → `BinomialMixtureFit` | a finite mixture P(kᵢ) = Σ_j π_j Bin(kᵢ; nᵢ, cᵢθ_j) with known scales cᵢ, by EM (the M-step for θ_j the root of a strictly concave score by calx's Brent, closed form for equal scales), θ ≤ θ_max; quantile and seeded random starts, the best likelihood kept; `weights`, `theta` (decreasing), `responsibility`, `assignment`, `log_likelihood`, `bic` = −2ℓ + (2K − 1) log N, `aic`, `iterations`, `converged`, `monotone` |
| `select_binomial_mixture(k, n, c, K_max, options)` → `BinomialMixtureSelection` | the fits for K = 1, …, K_max and the index of the least BIC |

A minimal use:

```cpp
#include <tyche/multivariate/nmf.hpp>
namespace mv = tyche::multivariate;
mv::StabilityOptions o;                       // 100 Poisson-resampled replicates, KL updates
const auto ranks = mv::nmf_rank_stability(V, 1, 10, o);
for (const auto& r : ranks) { /* r.rank, r.average_stability, r.minimum_stability, r.relative_error */ }
```

`dissimilarity.hpp`, `pcoa.hpp`, `permanova.hpp`, `mantel.hpp` (communities, decision 0017):

| Name | Content |
|---|---|
| `dissimilarity(x, y, Dissimilarity)` | `bray_curtis`, `jaccard`, `sorensen`, `hellinger`, `chord`, `euclidean` between two samples |
| `dissimilarities(table, kind)` → `Matrix` | between the rows of a table of samples by classes |
| `pcoa(d, Correction)` → `PCoA` | `eigenvalues` (descending, negative ones kept), `coordinates` (n × k, the positive axes), `explained`, `negative` (their sum), `correction`, `constant`, `distances`; `Correction{none, lingoes, cailliez}` |
| `gower_centred(d, power)`, `cailliez_constant(d)` | J(−½dᵖ)J; Cailliez's constant |
| `permanova(d, group, B, seed)`, `permanova_exact(d, group)` → `Permanova` | `n`, `groups`, `df_among`, `df_within`, `ss_total`, `ss_within`, `ss_among`, `F`, `r_squared`, `p_value`, `permutations`, `exact` |
| `dispersion(d, group, B, seed)` → `Dispersion`; `centroid_distances(d, group)` | PERMDISP: `distances` to the group centroids, `labels`, `group_mean`, `F`, `p_value` |
| `mantel(A, B, permutations, seed)`, `mantel_exact(A, B)` → `Mantel` | `r`, `p_value` (one-sided), `permutations`, `exact` |

## 3. Checks

`check_tyche_nmf`:

| Check | What it verifies |
|---|---|
| monotone objectives | Lee and Seung's Theorems 1 and 2: ‖V − WH‖²_F and D(V‖WH) never rise over 12 random problems × 2000 updates, checked at every update |
| one update | each rule equals eqs. 4 and 5 written out entry by entry, to 10⁻¹⁴ |
| planted factorisation | a separable V = WH (30 × 60, rank 4) recovered by both objectives: the objective near 0, W up to permutation (matched by calx's assignment) with cosines above 0.999, W's columns summing to 1 |
| normalisation | WH unchanged |
| silhouettes | four points on a line against Rousseeuw's definitions by hand |
| stability | Poisson counts of three planted signatures: at rank 3 tight clusters (average stability > 0.9), the consensus signatures matching the planted ones (cosine > 0.99), called stable; rank 5 less stable; the relative error falling to rank 3 and little after; clusters ordered, one signature of each replicate in each; multinomial resampling (keeping the columns' totals) giving the same signatures; determinism |

`check_tyche_binomial_mixture`:

| Check | What it verifies |
|---|---|
| one component | θ = Σk/(cΣn) exactly with equal scales; with unequal scales the score vanishes at θ̂ (to 6 · 10⁻¹³) |
| three components | 900 counts of three components with scales of a tumour's copy numbers: three chosen by BIC, θ recovered within 0.03 (measured 0.013), weights within 0.04, 90% of observations assigned to their own; every EM run's likelihood non-decreasing; θ in (0, 1]; the same fit for the same seed |

`check_tyche_communities` (46):

| Check | What it verifies |
|---|---|
| dissimilarities | each worked by hand on two samples; empty samples; symmetry, positivity and a zero diagonal on 25 samples; the triangle inequality on every triple for Jaccard, Hellinger, chord and Euclidean; Bray–Curtis breaking it on (1, 0), (1, 1), (0, 1); Sørensen = Bray–Curtis on presence; Hellinger ≤ √2 |
| principal coordinates | Euclidean distances of 12 points in 3 dimensions: three axes, the distances reproduced, the configuration's centred Gram matrix reproduced (the configuration up to rotation), shares summing to one; Bray–Curtis: negative eigenvalues, removed by Lingoes's constant (−λ_min) and by Cailliez's, which is the least (d + 0.98c is still not Euclidean) |
| PERMANOVA | F = the one-way ANOVA F for Euclidean distances in one dimension and the trace F in three; SS_W and R²; the exact p-value against an independent enumeration of the 20 and 90 relabellings of two small designs; 20 000 random relabellings within 0.012 of it; repeatability; separated groups found |
| PERMDISP | for Euclidean distances, the distances to the group means; spreads 0.5 and 3 told apart; for Bray–Curtis, the distances over every principal axis with the imaginary ones subtracted |
| Mantel | r = Pearson's on the 21 pairs; the exact p-value against an independent enumeration of 7! relabellings; 20 000 random relabellings within 0.01 of it; a matrix against itself |

## 4. Dependencies

`regression` (`nnls_columns`, for the exposures); calx `linalg`, `optimisation` (`assignment`, calx decision 0014) and `random` (`Stream`, `multinomial`); for the mixture, calx `solvers` (Brent) and `special_functions` (log Γ). For communities, calx `linalg` (`symmetric_eigen`, the general `eigenvalues` for Cailliez's constant) and `random`.

## 5. References

- Alexandrov, L. B., Nik-Zainal, S., Wedge, D. C., Campbell, P. J. and Stratton, M. R. (2013), Deciphering signatures of mutational processes operative in human cancer, *Cell Reports* 3, 246–259.
- Islam, S. M. A., Díaz-Gay, M., Wu, Y. et al. (2022), Uncovering novel mutational signatures by de novo extraction with SigProfilerExtractor, *Cell Genomics* 2, 100179.
- Lee, D. D. and Seung, H. S. (2001), Algorithms for non-negative matrix factorization, *Advances in Neural Information Processing Systems* 13, 556–562.
- Rousseeuw, P. J. (1987), Silhouettes: a graphical aid to the interpretation and validation of cluster analysis, *J. Comput. Appl. Math.* 20, 53–65.
- Brunet, J.-P., Tamayo, P., Golub, T. R. and Mesirov, J. P. (2004), Metagenes and molecular pattern discovery using matrix factorization, *PNAS* 101, 4164–4169.
- Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977), Maximum likelihood from incomplete data via the EM algorithm, *JRSS B* 39, 1–38; McLachlan, G. and Peel, D. (2000), *Finite Mixture Models*, Wiley, §2.8.
- Keribin, C. (2000), Consistent estimation of the order of mixture models, *Sankhyā A* 62, 49–66.
- Roth, A. et al. (2014), PyClone: statistical inference of clonal population structure in cancer, *Nature Methods* 11, 396–398.

- Anderson, M. J. (2001), A new method for non-parametric multivariate analysis of variance, *Austral Ecol.* 26, 32–46.
- Anderson, M. J. (2006), Distance-based tests for homogeneity of multivariate dispersions, *Biometrics* 62, 245–253.
- Anderson, M. J. and Walsh, D. C. I. (2013), PERMANOVA, ANOSIM, and the Mantel test in the face of heterogeneous dispersions, *Ecol. Monogr.* 83, 557–574.
- Cailliez, F. (1983), The analytical solution of the additive constant problem, *Psychometrika* 48, 305–308.
- Gower, J. C. (1966), Some distance properties of latent root and vector methods used in multivariate analysis, *Biometrika* 53, 325–338.
- Legendre, P. and Anderson, M. J. (1999), Distance-based redundancy analysis: testing multispecies responses in multifactorial ecological experiments, *Ecol. Monogr.* 69, 1–24.
- Legendre, P. and Legendre, L. (2012), *Numerical Ecology*, 3rd English ed., Elsevier.
- Lingoes, J. C. (1971), Some boundary conditions for a monotone analysis of symmetric matrices, *Psychometrika* 36, 195–203.
- Mantel, N. (1967), The detection of disease clustering and a generalized regression approach, *Cancer Res.* 27, 209–220.
- McArdle, B. H. and Anderson, M. J. (2001), Fitting multivariate models to community data: a comment on distance-based redundancy analysis, *Ecology* 82, 290–297.
- Phipson, B. and Smyth, G. K. (2010), Permutation p-values should never be zero, *Stat. Appl. Genet. Mol. Biol.* 9, art. 39.
- Torgerson, W. S. (1952), Multidimensional scaling: I. Theory and method, *Psychometrika* 17, 401–419.

## 6. Open Questions

1. SigProfilerExtractor's normalisation by a Gaussian mixture of the samples' totals, and its initialisation by NNDSVD.
2. Its choice of rank by Wilcoxon tests of the samples' reconstruction errors (needs rank tests in `testing`).
3. Faster solvers (alternating non-negative least squares, hierarchical ALS) for large catalogues; parallel replicates on calx's scheduler.
4. PERMANOVA of several factors and their interactions (McArdle and Anderson 2001) and with strata; distance-based redundancy analysis; non-metric multidimensional scaling; the partial Mantel test and Mantel correlograms.
