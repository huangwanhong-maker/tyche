# Glossary of Conventions

*Status: draft (2026-10-24).* The conventions tyche follows wherever a choice exists, stated once (CLAUDE.md, invariant 5). Each module's headers cite the source of its formulas; this page fixes only what the sources disagree on.

## 1. Samples and Moments

| Quantity | Convention | Code |
|---|---|---|
| Sample variance | Σ(x − x̄)² / (n − ddof), ddof = 1 by default (unbiased); ddof = 0 gives the second central moment | `descriptive::variance(x, ddof)` |
| Central moment mᵣ | Σ(x − x̄)ʳ / n | `descriptive::central_moment` |
| Skewness | g₁ = m₃ / m₂^{3/2}; G₁ = g₁ √(n(n − 1))/(n − 2) when bias-corrected (Joanes and Gill 1998) | `descriptive::skewness` |
| Kurtosis | excess g₂ = m₄ / m₂² − 3; G₂ when bias-corrected | `descriptive::kurtosis` |
| Sample quantile | Hyndman and Fan (1996) type 7 by default (R's and NumPy's), types 1–9 by parameter | `descriptive::quantile(x, p, type)` |
| Autocovariance | γ̂(h) = Σ_{t=1}^{n−h}(x_t − x̄)(x_{t+h} − x̄) / n, divisor n at every lag | `time_series::autocovariance` |

## 2. Distributions

| Quantity | Convention |
|---|---|
| Gamma | shape a and scale θ (mean aθ), not rate |
| χ²_k | Gamma(k/2, 2) |
| Student's t, F | standard forms (no location or scale) |
| Poisson, binomial cdf | P(X ≤ k); survival P(X > k) |
| Discrete quantile | the least k with P(X ≤ k) ≥ p |
| upper_quantile(q) | the x with P(X > x) = q, computed from the upper tail |
| Two-sided p-value | 2 min(P(X ≤ x), P(X ≥ x)), capped at 1 |

## 3. Regression

| Quantity | Convention |
|---|---|
| Design matrix | `calx::linalg::Rectangular`, n × k, one row per observation; an intercept is a column of ones the caller includes (`design_matrix(columns, true)`) |
| Intercept detected | some column constant and non-zero; R² is then centred, otherwise uncentred (NIST, R) |
| Weights | WLS weights wᵢ ∝ 1/Var εᵢ; residuals reported unweighted, sums of squares weighted |
| HC1, cluster factor | HC1 = HC0 · n/(n − r); cluster CR1 = G/(G − 1) · (n − 1)/(n − r) (Stata, statsmodels) |
| t degrees of freedom | n − r; G − 1 with clusters |
| AIC, BIC (regressions, GLMs) | −2ℓ + 2r and −2ℓ + r log n with r the coefficients estimated (statsmodels); R adds one parameter for σ² in Gaussian models |
| GLM inference | t on n − r degrees of freedom for the gaussian family (φ estimated), normal otherwise; t on G − 1 with clusters |
| Binomial response | the proportion of successes, with the numbers of trials as prior weights |
| Elastic-net objective | ½ Σ vᵢ(yᵢ − β₀ − x̃ᵢᵀβ)² + λ Σ γⱼ[½(1 − α)βⱼ² + α\|βⱼ\|] with Σ v = 1 (glmnet's; scikit-learn's ElasticNet is λ = its alpha, α = its l1_ratio, unstandardised); the binomial loss −Σ vᵢ[yᵢηᵢ − log(1 + e^ηᵢ)] (scikit-learn's LogisticRegression: λ = 1/(C N)) |
| Standardisation | columns centred at their weighted means (with an intercept) and divided by their weighted standard deviations, divisor Σ v = 1 (N); the penalty falls on the standardised coefficients, which are reported on X's scale |
| Penalty factors | γⱼ ≥ 0, rescaled to sum to the number of columns (glmnet); γⱼ = 0 leaves a column unpenalised |
| Cross-validation error | cvm the fold-weighted mean of the folds' mean losses; cvsd = √(Σ_k W_k(cv_k − cvm)²/ΣW_k/(K − 1)); λ_1se the largest λ with cvm ≤ cvm(λ_min) + cvsd(λ_min) |
| NMF objectives | ‖V − WH‖²_F (no factor ½) and D(V‖WH) = Σ[V log(V/WH) − V + WH]; W's columns normalised to sum to 1 |
| Rank stability | the mean silhouette of a cluster of replicate signatures under the cosine distance 1 − cos; the average over clusters and the least |

## 4. Time Series

| Quantity | Convention |
|---|---|
| AR polynomial | φ(B) = 1 − φ₁B − … − φ_pB^p (Box and Jenkins) |
| MA polynomial | θ(B) = 1 + θ₁B + … + θ_qB^q (R, statsmodels; Box and Jenkins write the opposite sign) |
| ARIMA constant | μ, the mean of the d-th difference (a drift when d = 1) |
| ARIMA AIC | −2ℓ + 2(k + 1), σ² counted, conditional likelihood |
| ADF regression | Δy_t on y_{t−1}, Δy_{t−1..t−k}, then deterministic terms; trend counted from 1 in the regression sample |
| ADF lag choice | every lag fitted on the common sample t = k_max + 2, …, n; the chosen one refitted on its own |
| Critical values | MacKinnon (2010) with T the observations of the test regression (ADF) or n − 1 (Engle–Granger) |
| VAR information criteria | Lütkepohl's: the mK² slope coefficients counted, not the intercepts (statsmodels counts them; the argmin is the same) |
| Impulse responses | Θ_i(a, b): variable a at horizon i after a one-standard-deviation orthogonal shock b, P the lower Cholesky factor of Σ̂_u (the ordering of the variables matters) |
| Spectral densities | one-sided, as calx's `signal` and scipy: a bin 0 < f < fs/2 holds the power of ±f (c = 2), the bins at 0 and fs/2 their own (c = 1); every estimate integrates to the variance (divisor n); white noise of variance σ² lies at 2σ²/fs |
| Cross spectrum | S_xy = mean of conj(X) Y: its phase is arg Y − arg X, so y lagging x by τ has arg S_xy = −2πfτ |
| Harmonic phase | a constituent A cos(2πf(t − t₀) − φ), φ = atan2(b, a) ∈ [0, 2π) for a cos + b sin: greatest φ/(2πf) after t₀ (the phase lag of tidal analysis) |
| Red-noise background | the AR(1) with the series' lag-one autocorrelation and variance (divisor n) |
| Segments and change points | segments half-open [τ_k, τ_{k+1}), 0-based; a change point τ is the first observation of a new segment |
| Segment cost | −2 times the maximised log-likelihood of the segment, up to a sum over observations independent of the segmentation (the mean model with σ² fixed: Σ w(y − ȳ_w)²/σ²) |
| Change-point penalty | β per change point on the same −2 log-likelihood scale: BIC (SIC) (d + 1) log n, AIC 2(d + 1), Hannan–Quinn 2(d + 1) log log n, MBIC (d + 2) log n plus d Σ log(ℓ_k/n), d the parameters of a segment |
| Robust noise variance | (1.4826 MAD of (y_{i+1} − yᵢ)/√2)², or with precisions (y_{i+1} − yᵢ)/√(1/wᵢ + 1/w_{i+1}) |
| CBS statistic and p-value | Z = (S̃_j − S̃_i)/√(k(m − k)/m) on the centred partial sums, t = Z/√((TSS − Z²)/(m − 2)); p = (1 + R)/(1 + B) |

## 5. Classes and Communities

| Quantity | Convention | Code |
|---|---|---|
| Abundance sample | X_i the count of class i, n = Σ X_i, f_k = #{i : X_i = k}, S_obs = Σ_{k≥1} f_k; zeros dropped | `classes::Abundances{counts, n, f}` |
| Incidence sample | T units, Y_i the units holding class i, U = Σ Y_i, Q_k = #{i : Y_i = k} | `classes::Incidences{units, counts, total, Q}` |
| Hill number | ^qD = (Σ p_i^q)^{1/(1−q)} over p_i > 0; q = 1 the limit exp H, q = ∞ the limit 1/max p; Rényi entropy H_q = ln ^qD in nats | `classes::hill`, `classes::renyi` |
| Similarity matrix | Z_ij ∈ [0, 1], Z_ii = 1, the similarity of class j to class i; (Zp)_i the ordinariness | `classes::similarity_diversity` |
| Partition weights | `equal`: w_j = 1/N, Jost's alpha; `size`: w_j = z_j+/z_++, alpha = ^qD(z/z_++)/N (Chao, Chiu and Jost); given unequal weights: Jost's alpha, q = 1 only | `classes::Weighting` |
| Overlaps | C_qN Sørensen-type, U_qN Jaccard-type, both 1 − ln β/ln N at q = 1; dissimilarities 1 − C_qN, 1 − U_qN | `classes::Overlap` |
| Chao1's factor and forms | A = (n − 1)/n (T for incidence: (T − 1)/T); classic: A f₁²/(2f₂) if f₂ > 0, else A f₁(f₁ − 1)/2 (iNEXT's); bias-corrected: A f₁(f₁ − 1)/(2(f₂ + 1)) for every f₂ (Chao 2005; scikit-bio's omits A) | `classes::Chao1Form` |
| Chao1's variance | the delta method with multinomial covariances among Ŝ cells, Σ d_k² f_k + T − (D − T)²/Ŝ; without anything undetected, the Poisson variance of S_obs conditioned on n (U for incidence) | `classes::chao1`, `classes::chao2` |
| Detection ratio | x = n f̂₀/(n f̂₀ + f₁) of the classic f̂₀: (n − 1)f₁/((n − 1)f₁ + 2f₂), or (n − 1)(f₁ − 1)/((n − 1)(f₁ − 1) + 2) when f₂ = 0, 0 when f₁ = 0; 1 − x is the A of Chao, Wang and Jost's entropy | `classes::detection_ratio` |
| Sample coverage | Ĉ_n = 1 − (f₁/n) x (Chao and Jost 2012, eq. 4a when f₂ > 0); Turing's 1 − f₁/n separately | `classes::coverage`, `classes::turing_coverage` |
| Rarefied Hill numbers | the Hill function of the subsamples' expected frequency counts f̂_k(m): exp of the mean entropy (q = 1), reciprocal of the mean concentration (q = 2), mean richness (q = 0); not the mean of the subsamples' Hill numbers | `classes::interpolated` |
| Bootstrap intervals | B samples of n from the bootstrap assemblage; se the replicates' standard deviation, divisor B − 1; interval estimate ± z se, coverage's clipped to [0, 1]; replicate r on the stream identity (0x636c6173, r, 0) (incidence: (·, r, 1)) | `classes::curve`, `classes::richness_curve` |
| Default sizes | ⌊knots/2⌋ sizes evenly from 1 to n − 1, n, ⌊knots/2⌋ from n + 1 to the endpoint (2n), rounded and unique | `classes::default_sizes` |
| Dissimilarities | Bray–Curtis and Jaccard of two empty samples 0, of one empty sample and another 1; Hellinger and chord refused for an empty sample; Euclidean on any real values | `multivariate::Dissimilarity` |
| Principal coordinates | eigenvalues of J(−½d²)J descending, those within 10⁻¹⁰ of the largest in magnitude set to 0; coordinates v_k √λ_k for λ_k > 0, each axis signed so that its largest entry is positive; corrections only when a negative eigenvalue exists | `multivariate::pcoa` |
| Permutation p-values | (b + 1)/(B + 1) for B random relabellings; exact: the share of all relabellings (the observed among them); ties F* ≥ F(1 − 10⁻¹⁰), r* ≥ r − 10⁻¹²; Mantel one-sided for a positive association | `multivariate::permanova`, `multivariate::mantel` |
| PERMDISP distances | z_i² = (1/n_g)Σ_j d_ij² − (1/(2n_g²))Σ_{j,k} d_jk² within the group; its absolute value where imaginary axes make it negative (vegan's betadisper); residuals z_i − z̄_g permuted | `multivariate::dispersion` |

## 6. Compositions

| Quantity | Convention | Code |
|---|---|---|
| Composition | positive parts, any units; only ratios carry information; every result closed to a stated `total` (default 1) | `compositions::closure` |
| Logarithms and log-ratios | natural; clr(x)_i = ln(x_i/g(x)), g the geometric mean; alr(x)_i = ln(x_i/x_k), the denominator k the last part by default and left out of the coordinates | `compositions::clr`, `compositions::alr` |
| Inner product | ⟨x, y⟩_a = Σ clr_i(x) clr_i(y) = (1/(2D)) Σ_ij ln(x_i/x_j) ln(y_i/y_j) | `compositions::inner` |
| Contrast matrix | Ψ, (D − 1) × D, rows the clr's of the basis: ilr(x) = Ψ ln x, ilr⁻¹(z) = C(exp(Ψᵀz)), clr = Ψᵀ ilr | `compositions::Basis::contrast` |
| Sequential binary partition | D − 1 rows of +1, −1, 0, each splitting a group made by the rows before it (the first row marks every part); written in that order | `compositions::basis` |
| Orientation of a balance | √(rs/(r + s)) ln(g(x₊)/g(x₋)): positive when the parts marked +1 are larger in geometric mean; contrast entries √(s/(r(r + s))) on +, −√(r/(s(r + s))) on − | `compositions::balance` |
| Default basis | pivot coordinates of the parts in order (Hron, Filzmoser and Thompson 2012): row j marks part j +1 and the parts after it −1; `pivot_basis(D, l)` puts part l first and the others in order | `compositions::pivot_basis` |
| Centre | the closure of the parts' geometric means (not the mean of closed rows) | `compositions::centre` |
| Variation matrix, total variance | t_ij = var(ln(x_i/x_j)) with divisor n − ddof, ddof = 1 by default (Pawlowsky-Glahn et al. write divisor n); totvar = (1/(2D)) Σ t_ij; metric standard deviation √(totvar/(D − 1)) | `compositions::variation_matrix`, `compositions::total_variance` |
| Zero replacement | multiplicative: δ_j for a zero part, (1 − Σδ/c) x_j otherwise, c the total (default the composition's own sum); δ_j = 0.65 × detection limit by default, or one δ; essential zeros refused | `compositions::replace_zeros` |
| Model's reference | `CompositionPredictor::centre`, the sample's centre closed to its mean total; reallocations are in the reference's units | `compositions::fit_predictor` |
| One for one | Δ moved from `from` to `to`, the other parts held | `compositions::one_for_one` |
| One for remaining | Δ added to one part and taken from the others in proportion to their sizes (Δ < 0 gives to them) | `compositions::one_for_remaining` |

## 7. Relational Data

| Quantity | Convention | Code |
|---|---|---|
| Network | nodes 0, …, n − 1; ties as (tail, head); undirected ties stored as (min, max); loops refused; a tie given twice is one tie, its weights summed; weights positive, 1 when none are given | `networks::network` |
| Directed and undirected | directed: i → j and j → i are two ties, mutual when both exist; undirected: one tie, read as mutual by the censuses | `networks::Network::directed` |
| Underlying graph | i — j when i → j or j → i, weight the sum; clustering, transitivity, cores, neighbourhoods and brokerage use it | `networks::underlying` |
| Degrees | `out`, `in`; `total` = out + in when directed (a mutual pair twice), the degree when undirected; `neighbours` distinct in the underlying graph | `networks::degrees` |
| Density | m/(n(n − 1)) directed, 2m/(n(n − 1)) undirected | `networks::density` |
| Reciprocity | arc r = 2M/m; dyadic M/(M + A); Garlaschelli and Loffredo's ρ = (r − ā)/(1 − ā), ā the density | `networks::reciprocity` |
| Triad classes | the order 003, 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201, 120D, 120U, 120C, 210, 300 (Batagelj and Mrvar's, networkx's); D: the lone node sends (021D out-star, 120D) or the asymmetric tie runs into the mutual pair (111D: A↔B, C→B) | `networks::triad_names` |
| Local clustering | 2t_i/(k_i(k_i − 1)) on the underlying graph, 0 when k_i < 2; the average over every node | `networks::local_clustering` |
| Core numbers | on the underlying graph, by distinct neighbours (networkx counts in + out degree for directed graphs) | `networks::core_numbers` |
| Mixing pairs | each directed tie once as (tail, head); each undirected tie twice; optionally weighted | `networks::mixing_matrix` |
| Directed degree assortativity | the tail's out-degree against the head's in-degree by default (`DegreeEnds::out_in`) | `networks::degree_assortativity` |
| E–I index | over ties, each once; a group's E_g and I_g over the ties incident to it; the reference the complete network on the same groups | `networks::ei_index` |
| Inbreeding homophily | H_g from the ties g's members send (each undirected tie from both ends) | `networks::inbreeding_homophily` |
| Burt's measures | z_ij = w_ij + w_ji over the underlying neighbours; NaN for an isolated node | `networks::brokerage` |
| QAP cells | the off-diagonal cells, every ordered pair, or the pairs i < j with `symmetric`; the diagonal never | `networks::qap_correlation`, `networks::mrqap` |
| Permutation and Monte Carlo p-values | (b + 1)/(B + 1) for B random relabellings or draws, the observed counted; exact: b/n! over all relabellings; two-sided \|stat*\| ≥ \|stat\|; ties within 10⁻¹² (r, CUG) or 10⁻¹⁰(1 + \|t\|) (t) | `networks::QapCorrelation`, `networks::CugTest` |
| Streams | QAP correlation (seed, (0x6e657477, 1, 0)); MRQAP (·, (·, 2, 0)); CUG draw r (·, (·, 5, r)) | `networks::qap_correlation`, `networks::cug_test` |
| Degree-preserving draws | each draw restarts from the observed network and attempts `swaps_per_tie` × m swaps; weights travel with the tails | `networks::degree_preserving` |
