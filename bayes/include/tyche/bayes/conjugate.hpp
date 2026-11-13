// Conjugate models: priors whose posteriors stay in their family, in closed form, with their predictive laws and marginal
// likelihoods (decision 0021; Gelman et al. 2013, Bayesian Data Analysis, 3rd ed., ch. 2–3; Bernardo and Smith 1994, Bayesian
// Theory, App. A).
//
//   beta_binomial(a, b, k, n)            Beta(a, b) prior on a probability, k successes in n trials: the posterior Beta(a + k,
//                                        b + n − k), the predictive probability of the next success, and the marginal likelihood
//                                        p(k | n) = C(n, k) B(a + k, b + n − k) / B(a, b)                         (BDA3 eq. 2.8)
//   gamma_poisson(α, β, y, t)            Gamma(shape α, rate β) prior on a rate, counts y_i over exposures t_i: the posterior
//                                        Gamma(α + Σy, β + Σt), and log p(y) = Σ[y_i log t_i − log y_i!] + α log β − log Γ(α)
//                                        + log Γ(α + Σy) − (α + Σy) log(β + Σt)                                  (BDA3 §2.6)
//   normal_known_variance(μ₀, τ₀, x, σ)  N(μ₀, τ₀²) prior on a mean, data of known σ: posterior precision 1/τ₀² + n/σ²   (BDA3 eq. 2.12)
//   normal_inverse_gamma(μ₀, κ₀, α₀, β₀, x)
//                                        μ | σ² ~ N(μ₀, σ²/κ₀), σ² ~ Inv-Gamma(α₀, β₀): κₙ = κ₀ + n, μₙ = (κ₀μ₀ + n x̄)/κₙ,
//                                        αₙ = α₀ + n/2, βₙ = β₀ + S/2 + κ₀ n (x̄ − μ₀)²/(2κₙ); the mean's marginal posterior a
//                                        Student t with 2αₙ degrees of freedom, location μₙ, scale² βₙ/(αₙκₙ); the predictive a t
//                                        with scale² βₙ(κₙ + 1)/(αₙκₙ); log p(x) = log Γ(αₙ) − log Γ(α₀) + α₀ log β₀ − αₙ log βₙ
//                                        + ½ log(κ₀/κₙ) − (n/2) log 2π                    (BDA3 §3.3; Murphy 2007, eqs. 85–95)
//   dirichlet_multinomial(α, counts)     Dirichlet(α) prior on a law of K classes: the posterior Dirichlet(α + counts), its mean,
//                                        and log p(counts) for the sequence observed
//
// Accuracy: closed forms through log Γ and log B (calx), so to about 10⁻¹⁴ relative for moderate parameters.

#pragma once

#include <tyche/bayes/common.hpp>
#include <tyche/probability/distributions.hpp>

#include <calx/special_functions/gamma.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::bayes {

namespace detail {
inline double lgamma(double x) { return std::lgamma(x); }
inline double lbeta(double a, double b) { return calx::special_functions::log_beta(a, b); }
}  // namespace detail

struct BetaBinomial {
  double a = 1, b = 1;           // the posterior Beta(a, b)
  double mean = 0, variance = 0; // of the probability
  double predictive = 0;         // P(the next trial succeeds)
  double log_marginal = 0;       // log p(k | n)
};

inline BetaBinomial beta_binomial(double a, double b, double k, double n) {
  detail::require(a > 0 && b > 0 && k >= 0 && n >= k, "beta_binomial: a, b > 0 and 0 ≤ k ≤ n");
  BetaBinomial r;
  r.a = a + k, r.b = b + n - k;
  r.mean = r.a / (r.a + r.b);
  r.variance = r.a * r.b / ((r.a + r.b) * (r.a + r.b) * (r.a + r.b + 1));
  r.predictive = r.mean;
  r.log_marginal = detail::lgamma(n + 1) - detail::lgamma(k + 1) - detail::lgamma(n - k + 1) + detail::lbeta(r.a, r.b) - detail::lbeta(a, b);
  return r;
}

struct GammaPoisson {
  double shape = 1, rate = 1;  // the posterior Gamma(shape, rate)
  double mean = 0, variance = 0;
  double log_marginal = 0;
};

inline GammaPoisson gamma_poisson(double alpha, double beta, std::span<const double> counts, std::span<const double> exposure = {}) {
  detail::require(alpha > 0 && beta > 0, "gamma_poisson: shape and rate > 0");
  detail::require(exposure.empty() || exposure.size() == counts.size(), "gamma_poisson: one exposure per count, or none");
  double sy = 0, st = 0, lm = 0;
  for (std::size_t i = 0; i < counts.size(); i++) {
    const double y = counts[i], t = exposure.empty() ? 1.0 : exposure[i];
    detail::require(y >= 0 && t > 0, "gamma_poisson: counts ≥ 0 and exposures > 0");
    sy += y, st += t;
    lm += y * std::log(t) - detail::lgamma(y + 1);
  }
  GammaPoisson r;
  r.shape = alpha + sy, r.rate = beta + st;
  r.mean = r.shape / r.rate, r.variance = r.shape / (r.rate * r.rate);
  r.log_marginal = lm + alpha * std::log(beta) - detail::lgamma(alpha) + detail::lgamma(r.shape) - r.shape * std::log(r.rate);
  return r;
}

struct NormalMean {
  double mean = 0, sd = 0;  // the posterior N(mean, sd²) of the mean
};

inline NormalMean normal_known_variance(double mu0, double tau0, std::span<const double> x, double sigma) {
  detail::require(tau0 > 0 && sigma > 0, "normal_known_variance: τ₀ > 0 and σ > 0");
  double s = 0;
  for (double v : x) s += v;
  const double n = static_cast<double>(x.size());
  const double precision = 1 / (tau0 * tau0) + n / (sigma * sigma);
  return {(mu0 / (tau0 * tau0) + s / (sigma * sigma)) / precision, std::sqrt(1 / precision)};
}

struct NormalInverseGamma {
  double mu = 0, kappa = 1, alpha = 1, beta = 1;  // the posterior
  probability::StudentT mean_law;                 // the standardised t of the mean: (μ − mu) / mean_scale ~ t
  double mean_scale = 0;
  double predictive_scale = 0;                    // likewise for a new observation
  double variance_mean = 0;                       // E[σ²] = β/(α − 1) when α > 1
  double log_marginal = 0;
};

inline NormalInverseGamma normal_inverse_gamma(double mu0, double kappa0, double alpha0, double beta0, std::span<const double> x) {
  detail::require(kappa0 > 0 && alpha0 > 0 && beta0 > 0, "normal_inverse_gamma: κ₀, α₀, β₀ > 0");
  const double n = static_cast<double>(x.size());
  double mean = 0;
  for (double v : x) mean += v;
  mean = n > 0 ? mean / n : 0.0;
  double S = 0;
  for (double v : x) S += (v - mean) * (v - mean);
  NormalInverseGamma r;
  r.kappa = kappa0 + n;
  r.mu = (kappa0 * mu0 + n * mean) / r.kappa;
  r.alpha = alpha0 + n / 2;
  r.beta = beta0 + S / 2 + kappa0 * n * (mean - mu0) * (mean - mu0) / (2 * r.kappa);
  r.mean_law = {2 * r.alpha};
  r.mean_scale = std::sqrt(r.beta / (r.alpha * r.kappa));
  r.predictive_scale = std::sqrt(r.beta * (r.kappa + 1) / (r.alpha * r.kappa));
  r.variance_mean = r.alpha > 1 ? r.beta / (r.alpha - 1) : INFINITY;
  r.log_marginal = detail::lgamma(r.alpha) - detail::lgamma(alpha0) + alpha0 * std::log(beta0) - r.alpha * std::log(r.beta) + 0.5 * std::log(kappa0 / r.kappa) -
                   n / 2 * std::log(2 * std::numbers::pi);
  return r;
}

struct DirichletMultinomial {
  std::vector<double> alpha, mean;
  double log_marginal = 0;  // of the sequence of observations (not of the unordered counts)
};

inline DirichletMultinomial dirichlet_multinomial(std::span<const double> alpha, std::span<const double> counts) {
  detail::require(alpha.size() == counts.size() && !alpha.empty(), "dirichlet_multinomial: one α and one count per class");
  DirichletMultinomial r;
  double a0 = 0, n = 0, A = 0;
  for (std::size_t k = 0; k < alpha.size(); k++) {
    detail::require(alpha[k] > 0 && counts[k] >= 0, "dirichlet_multinomial: α > 0 and counts ≥ 0");
    r.alpha.push_back(alpha[k] + counts[k]);
    a0 += alpha[k], n += counts[k];
    r.log_marginal += detail::lgamma(alpha[k] + counts[k]) - detail::lgamma(alpha[k]);
  }
  r.log_marginal += detail::lgamma(a0) - detail::lgamma(a0 + n);
  for (double v : r.alpha) A += v;
  for (double v : r.alpha) r.mean.push_back(v / A);
  return r;
}

}  // namespace tyche::bayes
