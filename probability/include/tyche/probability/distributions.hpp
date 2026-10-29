// The standard distributions as objects: densities, distribution functions, survival functions
// and quantiles of the normal, Student, χ², Fisher–Snedecor, gamma, beta, Poisson and binomial
// distributions.
//
// Each distribution is a plain struct of its parameters with member functions:
//   pdf(x) / pmf(k), log_pdf / log_pmf, cdf(x) = P(X ≤ x), survival(x) = P(X > x),
//   quantile(p) = F⁻¹(p), upper_quantile(q) = the x with P(X > x) = q, mean(), variance().
// The survival function is computed directly, never as 1 − cdf, so that the far upper tail keeps
// its relative accuracy (a p-value of 10⁻³⁰ is returned as such); upper_quantile does the same for
// critical values. Parameters are checked at each call (std::invalid_argument).
//
// Sources and accuracy:
// - Normal: Φ(z) = ½ erfc(−z/√2) with the C library's erfc (relative accuracy near ε in both
//   tails until underflow); Φ⁻¹ by Wichura's algorithm AS 241 (Wichura 1988, Appl. Statist. 37,
//   477–484, PPND16: rational approximations in three regions, relative accuracy about 10⁻¹⁶).
// - Student's t: F(t) = 1 − ½ I_{ν/(ν+t²)}(ν/2, ½) for t > 0 and ½ + ½ I_{t²/(ν+t²)}(½, ν/2)
//   near 0 (Abramowitz and Stegun 1964, 26.7.1 and 26.5.27; Johnson, Kotz and Balakrishnan 1995,
//   Continuous Univariate Distributions 2, ch. 28), with calx's incomplete beta function.
// - χ²_k: P(k/2, x/2) and Q(k/2, x/2), the regularised incomplete gamma functions (A&S 26.4.19);
//   gamma(shape a, scale θ): P(a, x/θ), Q(a, x/θ).
// - F(d₁, d₂): I_{d₁x/(d₁x+d₂)}(d₁/2, d₂/2) and its complement I_{d₂/(d₂+d₁x)}(d₂/2, d₁/2)
//   (A&S 26.6.2).
// - Beta(a, b): I_x(a, b).
// - Poisson(λ): P(X ≤ k) = Q(k + 1, λ), P(X > k) = P(k + 1, λ) (A&S 26.4.21);
//   binomial(n, p): P(X ≤ k) = I_{1−p}(n − k, k + 1), P(X > k) = I_p(k + 1, n − k) (A&S 26.5.24).
// The incomplete gamma and beta functions are calx's (relative error about 10⁻¹⁴ for moderate
// parameters, below 10⁴), and so bound the accuracy of everything here except the normal.
//
// Quantiles. The normal's is AS 241; the χ² and gamma lower quantiles use calx's inverse of
// P(a, x) (Halley's method, relative 10⁻¹³). The rest are found by safeguarded root finding
// (calx's Brent method) on the distribution or survival function, after bracketing: on the line
// for t, in log x for the positive distributions and in logit x for the beta, so that the
// result has relative accuracy near 10⁻¹⁴ wherever the distribution function does. The discrete
// quantile is the least k with F(k) ≥ p, found by stepping from the normal approximation.

#pragma once

#include <calx/solvers/roots.hpp>
#include <calx/special_functions/gamma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace tyche::probability {

inline constexpr double infinity = std::numeric_limits<double>::infinity();

namespace detail {

namespace sf = calx::special_functions;

inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::probability: ") + what);
}
inline void require_probability(double p) { require(p >= 0 && p <= 1, "a probability in [0, 1]"); }

// The root of an increasing function g on (0, ∞), found in u = log x.
template <class G>
inline double root_on_half_line(G&& g, double x0) {
  auto h = [&](double u) { return g(std::exp(u)); };
  const double u0 = std::log(x0);
  double a = u0 - 1, b = u0 + 1;
  for (int i = 0; i < 200 && h(a) > 0; i++) a = u0 - (u0 - a) * 2;
  for (int i = 0; i < 200 && h(b) < 0; i++) b = u0 + (b - u0) * 2;
  return std::exp(calx::solvers::brent(h, a, b, 1e-15, 300).x);
}

// The root of an increasing function g on (0, 1), found in u = logit x.
template <class G>
inline double root_on_unit_interval(G&& g) {
  auto x_of = [](double u) { return u >= 0 ? 1 / (1 + std::exp(-u)) : std::exp(u) / (1 + std::exp(u)); };
  auto h = [&](double u) { return g(x_of(u)); };
  double a = -1, b = 1;
  for (int i = 0; i < 200 && h(a) > 0; i++) a *= 2;
  for (int i = 0; i < 200 && h(b) < 0; i++) b *= 2;
  return x_of(calx::solvers::brent(h, a, b, 1e-15, 300).x);
}

}  // namespace detail

// ---- the normal distribution N(μ, σ²) ----
struct Normal {
  double mu = 0, sigma = 1;

  void validate() const { detail::require(std::isfinite(mu) && sigma > 0 && std::isfinite(sigma), "Normal needs a finite μ and σ > 0"); }
  double log_pdf(double x) const {
    validate();
    const double z = (x - mu) / sigma;
    return -0.5 * z * z - std::log(sigma) - 0.5 * std::log(2 * std::numbers::pi);
  }
  double pdf(double x) const { return std::exp(log_pdf(x)); }
  double cdf(double x) const {
    validate();
    return 0.5 * std::erfc(-(x - mu) / (sigma * std::numbers::sqrt2));
  }
  double survival(double x) const {
    validate();
    return 0.5 * std::erfc((x - mu) / (sigma * std::numbers::sqrt2));
  }
  // The standard normal quantile, Wichura's AS 241 (PPND16).
  static double standard_quantile(double p) {
    detail::require_probability(p);
    if (p == 0) return -infinity;
    if (p == 1) return infinity;
    const double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
      const double r = 0.180625 - q * q;
      return q * (((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r + 6.7265770927008700853e4) * r + 4.5921953931549871457e4) * r +
                     1.3731693765509461125e4) * r + 1.9715909503065514427e3) * r + 1.3314166789178437745e2) * r + 3.3871328727963666080e0) /
             (((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r + 3.9307895800092710610e4) * r + 2.1213794301586595867e4) * r +
                 5.3941960214247511077e3) * r + 6.8718700749205790830e2) * r + 4.2313330701600911252e1) * r + 1.0);
    }
    double r = q < 0 ? p : 1 - p;
    r = std::sqrt(-std::log(r));
    double z;
    if (r <= 5) {
      r -= 1.6;
      z = (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r + 2.41780725177450611770e-1) * r + 1.27045825245236838258e0) * r +
              3.64784832476320460504e0) * r + 5.76949722146069140550e0) * r + 4.63033784615654529590e0) * r + 1.42343711074968357734e0) /
          (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r + 1.51986665636164571966e-2) * r + 1.48103976427480074590e-1) * r +
              6.89767334985100004550e-1) * r + 1.67638483018380384940e0) * r + 2.05319162663775882187e0) * r + 1.0);
    } else {
      r -= 5;
      z = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 1.24266094738807843860e-3) * r + 2.65321895265761230930e-2) * r +
              2.96560571828504891230e-1) * r + 1.78482653991729133580e0) * r + 5.46378491116411436990e0) * r + 6.65790464350110377720e0) /
          (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r + 1.84631831751005468180e-5) * r + 7.86869131145613259100e-4) * r +
              1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r + 5.99832206555887937690e-1) * r + 1.0);
    }
    return q < 0 ? -z : z;
  }
  double quantile(double p) const {
    validate();
    return mu + sigma * standard_quantile(p);
  }
  double upper_quantile(double q) const {
    validate();
    return mu - sigma * standard_quantile(q);
  }
  double mean() const { return mu; }
  double variance() const { return sigma * sigma; }
};

// ---- Student's t with ν > 0 degrees of freedom (standard: location 0, scale 1) ----
struct StudentT {
  double nu = 1;

  void validate() const { detail::require(nu > 0, "StudentT needs ν > 0"); }
  double log_pdf(double t) const {
    validate();
    if (std::isinf(nu)) return Normal{}.log_pdf(t);
    return detail::sf::log_gamma((nu + 1) / 2) - detail::sf::log_gamma(nu / 2) - 0.5 * std::log(nu * std::numbers::pi) - (nu + 1) / 2 * std::log1p(t * t / nu);
  }
  double pdf(double t) const { return std::exp(log_pdf(t)); }
  // P(T > t) for t ≥ 0, accurate in both regimes.
  double upper_half(double t) const {
    const double t2 = t * t;
    if (t2 < nu) return 0.5 - 0.5 * detail::sf::beta_i(0.5, nu / 2, t2 / (nu + t2));
    return 0.5 * detail::sf::beta_i(nu / 2, 0.5, nu / (nu + t2));
  }
  double cdf(double t) const {
    validate();
    if (std::isnan(t)) return t;
    if (std::isinf(nu)) return Normal{}.cdf(t);
    return t >= 0 ? 1 - upper_half(t) : upper_half(-t);
  }
  double survival(double t) const {
    validate();
    if (std::isnan(t)) return t;
    if (std::isinf(nu)) return Normal{}.survival(t);
    return t >= 0 ? upper_half(t) : 1 - upper_half(-t);
  }
  double upper_quantile(double q) const {
    validate();
    detail::require_probability(q);
    if (q == 0) return infinity;
    if (q == 1) return -infinity;
    if (q == 0.5) return 0;
    if (q > 0.5) return -upper_quantile(1 - q);
    if (std::isinf(nu)) return Normal{}.upper_quantile(q);
    if (nu == 1) return std::tan(std::numbers::pi * (0.5 - q));  // Cauchy
    if (nu == 2) {                                               // closed form (Hill 1970)
      const double p = 1 - q;
      return (2 * p - 1) / std::sqrt(2 * p * q);
    }
    // g(t) = q − P(T > t) increases on [0, ∞) from q − ½ < 0; bracket [0, b] and solve.
    auto g = [&](double t) { return q - upper_half(t); };
    double b = std::max(2.0, 2 * Normal{}.upper_quantile(q));
    for (int i = 0; i < 1100 && g(b) < 0; i++) b *= 2;
    return calx::solvers::brent(g, 0.0, b, 1e-15, 300).x;
  }
  double quantile(double p) const { return -upper_quantile(p); }
  double mean() const { return nu > 1 ? 0.0 : std::numeric_limits<double>::quiet_NaN(); }
  double variance() const { return nu > 2 ? nu / (nu - 2) : nu > 1 ? infinity : std::numeric_limits<double>::quiet_NaN(); }
};

// ---- the gamma distribution with shape a > 0 and scale θ > 0 ----
struct Gamma {
  double shape = 1, scale = 1;

  void validate() const { detail::require(shape > 0 && scale > 0 && std::isfinite(shape) && std::isfinite(scale), "Gamma needs shape > 0 and scale > 0"); }
  double log_pdf(double x) const {
    validate();
    if (x < 0) return -infinity;
    if (x == 0) return shape < 1 ? infinity : shape == 1 ? -std::log(scale) : -infinity;
    return (shape - 1) * std::log(x) - x / scale - shape * std::log(scale) - detail::sf::log_gamma(shape);
  }
  double pdf(double x) const { return std::exp(log_pdf(x)); }
  double cdf(double x) const {
    validate();
    return x <= 0 ? 0.0 : std::isinf(x) ? 1.0 : detail::sf::gamma_p(shape, x / scale);
  }
  double survival(double x) const {
    validate();
    return x <= 0 ? 1.0 : std::isinf(x) ? 0.0 : detail::sf::gamma_q(shape, x / scale);
  }
  double quantile(double p) const {
    validate();
    detail::require_probability(p);
    if (p == 0) return 0;
    if (p == 1) return infinity;
    if (p > 0.999) return upper_quantile(1 - p);
    return scale * detail::sf::gamma_p_inverse(shape, p);
  }
  double upper_quantile(double q) const {
    validate();
    detail::require_probability(q);
    if (q == 0) return infinity;
    if (q == 1) return 0;
    if (q > 0.001) return scale * detail::sf::gamma_p_inverse(shape, 1 - q);
    const double x0 = scale * detail::sf::gamma_p_inverse(shape, 0.999);
    return detail::root_on_half_line([&](double x) { return q - survival(x); }, x0);
  }
  double mean() const { return shape * scale; }
  double variance() const { return shape * scale * scale; }
};

// ---- χ² with k > 0 degrees of freedom: Gamma(k/2, 2) ----
struct ChiSquared {
  double k = 1;

  void validate() const { detail::require(k > 0 && std::isfinite(k), "ChiSquared needs k > 0"); }
  Gamma as_gamma() const {
    validate();
    return Gamma{k / 2, 2};
  }
  double log_pdf(double x) const { return as_gamma().log_pdf(x); }
  double pdf(double x) const { return as_gamma().pdf(x); }
  double cdf(double x) const { return as_gamma().cdf(x); }
  double survival(double x) const { return as_gamma().survival(x); }
  double quantile(double p) const { return as_gamma().quantile(p); }
  double upper_quantile(double q) const { return as_gamma().upper_quantile(q); }
  double mean() const { return k; }
  double variance() const { return 2 * k; }
};

// ---- the beta distribution with a, b > 0 ----
struct Beta {
  double a = 1, b = 1;

  void validate() const { detail::require(a > 0 && b > 0 && std::isfinite(a) && std::isfinite(b), "Beta needs a > 0 and b > 0"); }
  double log_pdf(double x) const {
    validate();
    if (x < 0 || x > 1) return -infinity;
    if (x == 0) return a < 1 ? infinity : a == 1 ? -detail::sf::log_beta(a, b) : -infinity;
    if (x == 1) return b < 1 ? infinity : b == 1 ? -detail::sf::log_beta(a, b) : -infinity;
    return (a - 1) * std::log(x) + (b - 1) * std::log1p(-x) - detail::sf::log_beta(a, b);
  }
  double pdf(double x) const { return std::exp(log_pdf(x)); }
  double cdf(double x) const {
    validate();
    return x <= 0 ? 0.0 : x >= 1 ? 1.0 : detail::sf::beta_i(a, b, x);
  }
  double survival(double x) const {
    validate();
    return x <= 0 ? 1.0 : x >= 1 ? 0.0 : detail::sf::beta_i(b, a, 1 - x);
  }
  double quantile(double p) const {
    validate();
    detail::require_probability(p);
    if (p == 0) return 0;
    if (p == 1) return 1;
    if (p <= 0.5) return detail::root_on_unit_interval([&](double x) { return cdf(x) - p; });
    const double q = 1 - p;
    return detail::root_on_unit_interval([&](double x) { return q - survival(x); });
  }
  double upper_quantile(double q) const {
    validate();
    detail::require_probability(q);
    if (q == 0) return 1;
    if (q == 1) return 0;
    if (q >= 0.5) return quantile(1 - q);
    return detail::root_on_unit_interval([&](double x) { return q - survival(x); });
  }
  double mean() const { return a / (a + b); }
  double variance() const { return a * b / ((a + b) * (a + b) * (a + b + 1)); }
};

// ---- the F distribution with (d₁, d₂) degrees of freedom ----
struct FisherF {
  double d1 = 1, d2 = 1;

  void validate() const { detail::require(d1 > 0 && d2 > 0, "FisherF needs d1 > 0 and d2 > 0"); }
  double log_pdf(double x) const {
    validate();
    if (x < 0) return -infinity;
    if (x == 0) return d1 < 2 ? infinity : d1 == 2 ? 0.0 : -infinity;
    if (std::isinf(d2)) return ChiSquared{d1}.log_pdf(d1 * x) + std::log(d1);
    return 0.5 * (d1 * std::log(d1) + d2 * std::log(d2)) + (d1 / 2 - 1) * std::log(x) - (d1 + d2) / 2 * std::log(d2 + d1 * x) -
           detail::sf::log_beta(d1 / 2, d2 / 2);
  }
  double pdf(double x) const { return std::exp(log_pdf(x)); }
  double cdf(double x) const {
    validate();
    if (x <= 0) return 0;
    if (std::isinf(x)) return 1;
    if (std::isinf(d2)) return ChiSquared{d1}.cdf(d1 * x);
    return detail::sf::beta_i(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2));
  }
  double survival(double x) const {
    validate();
    if (x <= 0) return 1;
    if (std::isinf(x)) return 0;
    if (std::isinf(d2)) return ChiSquared{d1}.survival(d1 * x);
    return detail::sf::beta_i(d2 / 2, d1 / 2, d2 / (d2 + d1 * x));
  }
  double quantile(double p) const {
    validate();
    detail::require_probability(p);
    if (p == 0) return 0;
    if (p == 1) return infinity;
    if (p <= 0.5) return detail::root_on_half_line([&](double x) { return cdf(x) - p; }, 1.0);
    return upper_quantile(1 - p);
  }
  double upper_quantile(double q) const {
    validate();
    detail::require_probability(q);
    if (q == 0) return infinity;
    if (q == 1) return 0;
    if (q > 0.5) return quantile(1 - q);
    return detail::root_on_half_line([&](double x) { return q - survival(x); }, 1.0);
  }
  double mean() const { return d2 > 2 ? d2 / (d2 - 2) : std::numeric_limits<double>::quiet_NaN(); }
  double variance() const {
    if (!(d2 > 4)) return d2 > 2 ? infinity : std::numeric_limits<double>::quiet_NaN();
    return 2 * d2 * d2 * (d1 + d2 - 2) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4));
  }
};

// ---- the Poisson distribution with mean λ > 0 ----
struct Poisson {
  double lambda = 1;

  void validate() const { detail::require(lambda > 0 && std::isfinite(lambda), "Poisson needs λ > 0"); }
  double log_pmf(std::int64_t k) const {
    validate();
    if (k < 0) return -infinity;
    const double kd = static_cast<double>(k);
    return kd * std::log(lambda) - lambda - detail::sf::log_gamma(kd + 1);
  }
  double pmf(std::int64_t k) const { return std::exp(log_pmf(k)); }
  double cdf(std::int64_t k) const {
    validate();
    return k < 0 ? 0.0 : detail::sf::gamma_q(static_cast<double>(k) + 1, lambda);
  }
  double survival(std::int64_t k) const {
    validate();
    return k < 0 ? 1.0 : detail::sf::gamma_p(static_cast<double>(k) + 1, lambda);
  }
  // The least k with P(X ≤ k) ≥ p.
  std::int64_t quantile(double p) const {
    validate();
    detail::require_probability(p);
    if (p == 1) return std::numeric_limits<std::int64_t>::max();
    double z = p <= 0 ? -10 : Normal::standard_quantile(std::clamp(p, 1e-300, 1 - 1e-16));
    auto k = static_cast<std::int64_t>(std::max(0.0, std::floor(lambda + std::sqrt(lambda) * z)));
    while (cdf(k) < p) k++;
    while (k > 0 && cdf(k - 1) >= p) k--;
    return k;
  }
  double mean() const { return lambda; }
  double variance() const { return lambda; }
};

// ---- the binomial distribution: n trials of probability p ----
struct Binomial {
  std::int64_t n = 1;
  double p = 0.5;

  void validate() const { detail::require(n >= 0 && p >= 0 && p <= 1, "Binomial needs n ≥ 0 and 0 ≤ p ≤ 1"); }
  double log_pmf(std::int64_t k) const {
    validate();
    if (k < 0 || k > n) return -infinity;
    const double kd = static_cast<double>(k), nd = static_cast<double>(n);
    if (p == 0) return k == 0 ? 0.0 : -infinity;
    if (p == 1) return k == n ? 0.0 : -infinity;
    return detail::sf::log_choose(nd, kd) + kd * std::log(p) + (nd - kd) * std::log1p(-p);
  }
  double pmf(std::int64_t k) const { return std::exp(log_pmf(k)); }
  double cdf(std::int64_t k) const {
    validate();
    if (k < 0) return 0;
    if (k >= n) return 1;
    if (p == 0) return 1;
    if (p == 1) return 0;
    return detail::sf::beta_i(static_cast<double>(n - k), static_cast<double>(k) + 1, 1 - p);
  }
  double survival(std::int64_t k) const {
    validate();
    if (k < 0) return 1;
    if (k >= n) return 0;
    if (p == 0) return 0;
    if (p == 1) return 1;
    return detail::sf::beta_i(static_cast<double>(k) + 1, static_cast<double>(n - k), p);
  }
  std::int64_t quantile(double q) const {
    validate();
    detail::require_probability(q);
    const double nd = static_cast<double>(n);
    const double z = q <= 0 ? -10 : q >= 1 ? 10 : Normal::standard_quantile(q);
    auto k = static_cast<std::int64_t>(std::clamp(std::floor(nd * p + std::sqrt(nd * p * (1 - p)) * z), 0.0, nd));
    while (k < n && cdf(k) < q) k++;
    while (k > 0 && cdf(k - 1) >= q) k--;
    return k;
  }
  double mean() const { return static_cast<double>(n) * p; }
  double variance() const { return static_cast<double>(n) * p * (1 - p); }
};

}  // namespace tyche::probability
