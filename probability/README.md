# probability: Distributions as Objects

*Status: draft (2026-10-24): eight distributions and p-values ([decision 0003](../development/decisions/0003-the-first-modules.md)).* Further distributions (lognormal, Weibull, negative binomial, multinomial, multivariate normal, Dirichlet), transformations, mixtures, copulas and characteristic functions are planned.

## 1. Concepts

A distribution is known through its distribution function F(x) = P(X ≤ x), its density or mass function, its survival function S(x) = P(X > x) = 1 − F(x), and its quantile function F⁻¹(p). In statistics the interesting values are in the tails: a p-value is a tail probability, a critical value an upper quantile. In floating point 1 − F(x) loses every digit once F(x) is within ε of 1, so the survival function is computed in its own right (from the complementary incomplete gamma or beta function, or erfc), and the upper quantile is found from the survival function. Most distributions of statistics reduce to two special functions: the regularised incomplete gamma function (χ², gamma, Poisson) and the incomplete beta function (t, F, beta, binomial), which calx provides.

## 2. Content

`distributions.hpp`: each distribution is a struct of its parameters with `pdf(x)` or `pmf(k)`, `log_pdf`/`log_pmf`, `cdf`, `survival`, `quantile(p)`, `upper_quantile(q)`, `mean()`, `variance()`.

| Distribution | Parameters | Distribution function | Quantile |
|---|---|---|---|
| `Normal{mu, sigma}` | μ, σ > 0 | ½ erfc(−z/√2) | Wichura's AS 241 (relative 10⁻¹⁶); `Normal::standard_quantile(p)` |
| `StudentT{nu}` | ν > 0 (∞ allowed) | incomplete beta (A&S 26.5.27), two forms for accuracy near 0 and in the tails | closed forms at ν = 1, 2; Brent's method on [0, b] otherwise |
| `ChiSquared{k}` | k > 0 | P(k/2, x/2), Q(k/2, x/2) | calx's inverse of P; the upper tail by root finding in log x |
| `Gamma{shape, scale}` | a, θ > 0 | P(a, x/θ) | as χ² |
| `FisherF{d1, d2}` | d₁, d₂ > 0 (d₂ = ∞ allowed) | incomplete beta (A&S 26.6.2) | root finding in log x |
| `Beta{a, b}` | a, b > 0 | I_x(a, b) | root finding in logit x |
| `Poisson{lambda}` | λ > 0 | Q(k + 1, λ) | the least k with F(k) ≥ p |
| `Binomial{n, p}` | n ≥ 0, 0 ≤ p ≤ 1 | I_{1−p}(n − k, k + 1) | the least k with F(k) ≥ p |

`p_values.hpp`: `upper_tail(d, x)`, `lower_tail(d, x)`, `two_sided(d, x)` (equal-tailed), `two_sided_normal(z)`, `two_sided_t(t, nu)`.

## 3. Checks

`check_tyche_probability` (66): Φ at 1, 2, 3 (A&S Table 26.1) and in the far tails (Φ(−10), P(Z > 30)) to 10⁻¹⁴ relative; AS 241 against Wichura's published test values and z at 0.975 and 0.9999 (at the exact double inputs), the round trip Φ(Φ⁻¹(p)) over 10⁻¹² < p < 1; t, χ² and F critical values against Fisher and Yates's tables and 40-digit values (mpmath), fractional degrees of freedom; tails to 10⁻¹⁷ (χ²₁₀ above 100) with full relative accuracy; closed forms (Cauchy and t₂ quantiles, χ²₂ as an exponential, F(1, ν) = t², Beta(a, 1), beta and binomial sums, Poisson sums); discrete quantiles as least k; quantile round trips for t, F, beta and gamma; p-values; refusal of invalid parameters.

## 4. Dependencies

calx `special_functions` (log Γ, the incomplete gamma and beta functions, the inverse of P) and `solvers` (Brent's root finder).

## 5. References

- Abramowitz, M. and Stegun, I. A. (1964), *Handbook of Mathematical Functions*, NBS, ch. 26.
- Fisher, R. A. and Yates, F. (1963), *Statistical Tables for Biological, Agricultural and Medical Research*, 6th ed., Oliver and Boyd.
- Johnson, N. L., Kotz, S. and Balakrishnan, N. (1994, 1995), *Continuous Univariate Distributions*, 2nd ed., 2 vols., Wiley.
- Lehmann, E. L. and Romano, J. P. (2005), *Testing Statistical Hypotheses*, 3rd ed., Springer.
- Wichura, M. J. (1988), Algorithm AS 241: the percentage points of the normal distribution, *Appl. Statist.* 37, 477–484.

## 6. Open Questions

1. Sampling: `sample(stream)` members delegating to calx's random streams (CLAUDE.md §8, open decision 3).
2. Accuracy of the incomplete beta function for very large parameters (above 10⁴), where calx's continued fraction slows; an asymptotic expansion (DiDonato and Morris 1992) would be calx's.
