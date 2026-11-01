# multivariate: Factorisations, Components and Clusters

*Status: draft (2026-11-01): non-negative matrix factorisation by Lee and Seung's multiplicative updates, and the stability of a factorisation's rank by resampling and constrained clustering ([decision 0006](../development/decisions/0006-change-points-penalties-and-factorisations.md)).* Mixtures of binomials with known scales by EM, their order by BIC (for ADR 0064's clones). Principal components, factor analysis, canonical correlation, k-means and other mixtures are planned.

## 1. Concepts

A non-negative matrix V (counts of m kinds of event in n samples) is often a superposition: each sample's counts the sum of a few processes' contributions, each process with its own distribution over the kinds. Then V ≈ WH with W (m × k) the processes' distributions and H (k × n) their activities, all entries non-negative. Non-negativity is what makes the parts interpretable: they can only add, never cancel, so a factor is a pattern that appears, not a contrast.

The factorisation minimises a divergence between V and WH: the squared Frobenius distance, or the generalised Kullback–Leibler divergence, which is the negative log-likelihood of Poisson counts with means WH. Lee and Seung's multiplicative updates multiply each entry by a ratio of non-negative terms, so the factors stay non-negative, and each objective never increases along its own updates; but the problem is not convex, and the updates find local minima. The factorisation is also unique at most up to a permutation and a positive rescaling of its k components, and only when the factors are sufficiently sparse.

How many components the data determine is a question of stability. If V is resampled within its noise (each count Poisson about itself, or each sample's counts multinomially redrawn with its total kept) and factorised again and again, the components of a well-determined rank come back the same each time; those of too high a rank wander. Clustering all the replicates' components, one of each replicate in each cluster, and measuring how tight and separated the clusters are (by silhouettes under the cosine distance) gives each rank its stability; the reconstruction error gives its fit. This is how mutational signatures are extracted from cancer genomes (Alexandrov et al. 2013; Islam et al. 2022), and it applies to any non-negative data.

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

## 4. Dependencies

`regression` (`nnls_columns`, for the exposures); calx `linalg`, `optimisation` (`assignment`, calx decision 0014) and `random` (`Stream`, `multinomial`); for the mixture, calx `solvers` (Brent) and `special_functions` (log Γ).

## 5. References

- Alexandrov, L. B., Nik-Zainal, S., Wedge, D. C., Campbell, P. J. and Stratton, M. R. (2013), Deciphering signatures of mutational processes operative in human cancer, *Cell Reports* 3, 246–259.
- Islam, S. M. A., Díaz-Gay, M., Wu, Y. et al. (2022), Uncovering novel mutational signatures by de novo extraction with SigProfilerExtractor, *Cell Genomics* 2, 100179.
- Lee, D. D. and Seung, H. S. (2001), Algorithms for non-negative matrix factorization, *Advances in Neural Information Processing Systems* 13, 556–562.
- Rousseeuw, P. J. (1987), Silhouettes: a graphical aid to the interpretation and validation of cluster analysis, *J. Comput. Appl. Math.* 20, 53–65.
- Brunet, J.-P., Tamayo, P., Golub, T. R. and Mesirov, J. P. (2004), Metagenes and molecular pattern discovery using matrix factorization, *PNAS* 101, 4164–4169.
- Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977), Maximum likelihood from incomplete data via the EM algorithm, *JRSS B* 39, 1–38; McLachlan, G. and Peel, D. (2000), *Finite Mixture Models*, Wiley, §2.8.
- Keribin, C. (2000), Consistent estimation of the order of mixture models, *Sankhyā A* 62, 49–66.
- Roth, A. et al. (2014), PyClone: statistical inference of clonal population structure in cancer, *Nature Methods* 11, 396–398.

## 6. Open Questions

1. SigProfilerExtractor's normalisation by a Gaussian mixture of the samples' totals, and its initialisation by NNDSVD.
2. Its choice of rank by Wilcoxon tests of the samples' reconstruction errors (needs rank tests in `testing`).
3. Faster solvers (alternating non-negative least squares, hierarchical ALS) for large catalogues; parallel replicates on calx's scheduler.
