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
