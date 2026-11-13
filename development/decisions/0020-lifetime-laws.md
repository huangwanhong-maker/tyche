# 0020. Lifetime Laws: the Exponential, Weibull and Lognormal Distributions

- **Status:** Accepted, provisionally (2026-11-13).
- **Date:** 2026-11-13
- **Decided by:** the infrastructure's ADR 0106, which founds its industrial-ecology domain. Its dynamic stock models read the survival of goods in use (buildings, vehicles, pipes, appliances) from declared lifetime laws, and the Weibull, lognormal and exponential laws are the ones the literature fits (Müller 2006; Miatto et al. 2017; Pauliuk and Heeren 2020).

## 1. Context

`probability` held the normal, Student, χ², gamma, beta, F, Poisson and binomial laws. Survival analysis (peira) estimates lifetimes from records, but a declared law needs its survival function S(x) = P(X > x) at any age, to the relative accuracy of the tail.

## 2. Decision

`probability/distributions.hpp` gains three laws, each with the members of the others (pdf, log_pdf, cdf, survival, quantile, upper_quantile, mean, variance):

- **`Exponential{rate}`:** S(x) = e^{−λx}. Its cdf is by expm1, so that F(10⁻¹²) keeps its digits.
- **`Weibull{shape, scale}`:** S(x) = exp(−(x/λ)^k), with `Weibull::with_mean(mean, shape)`.
- **`LogNormal{mu, sigma}`:** log X ~ N(μ, σ²), through the normal's erfc and AS 241, with `LogNormal::with_moments(mean, sd)` from the mean and standard deviation of X itself (Johnson, Kotz and Balakrishnan 1994, 14.8).

Every function is in closed form; the accuracy is that of exp, log, expm1, log1p and erfc.

## 3. Alternatives Considered

- **In the domain.** Rejected: the laws are distributions. Reliability, hydrology and survival all use them.
- **Parameterising the lognormal by the moments of X only.** Rejected: μ and σ of log X are the law's own parameters. `with_moments` gives the other reading, which ODYM's users state.

## 4. Consequences

Twelve checks:

- the Weibull law of shape 1 is the exponential;
- the exponential's tail, and its cdf near zero, keep their relative digits;
- the Weibull law's mean, variance and median, and the lognormal's, are held to closed forms;
- `with_mean` and `with_moments` give back their moments;
- F(F⁻¹(p)) = p and S(S⁻¹(q)) = q from 10⁻⁹ to 1 − 10⁻⁶.
