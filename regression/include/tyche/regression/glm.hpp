// Generalised linear models by iteratively reweighted least squares.
//
// THE MODEL (McCullagh and Nelder 1989, Generalized Linear Models, 2nd ed., ch. 2). Each yᵢ has
// a distribution of the exponential family with mean μᵢ and variance φ V(μᵢ) / wᵢ (wᵢ a prior
// weight), and g(μᵢ) = ηᵢ = xᵢβ + oᵢ for a link g and an offset oᵢ. The families and links here:
// - gaussian (V = 1) with the identity link: linear regression;
// - binomial (V = μ(1 − μ)), yᵢ the proportion of successes in mᵢ trials given as the prior
//   weights (default 1: yᵢ ∈ {0, 1}), with the logit link log(μ/(1 − μ)) (logistic regression)
//   or the probit link Φ⁻¹(μ);
// - poisson (V = μ) with the log link; an offset log(exposureᵢ) models rates. Poisson regression
//   with robust standard errors is the PPML estimator of gravity equations (Santos Silva and
//   Tenreyro 2006, "The log of gravity", Rev. Econ. Stat. 88, 641–658): consistent whenever
//   E(y|x) = exp(xβ), whatever the distribution of y ≥ 0, which need not be integer.
// Any family may take any of the links where its mean stays in range (a binomial with the log
// link, say), at the caller's risk.
//
// THE ALGORITHM (McCullagh and Nelder §2.5; Green 1984, JRSS B 46, 149–192). From μ⁽⁰⁾ (y for the
// gaussian, (m y + ½)/(m + 1) for the binomial, y + 0.1 for the poisson, as R's glm starts),
// repeat: the working response zᵢ = ηᵢ − oᵢ + (yᵢ − μᵢ) g′(μᵢ) and the working weights
// Wᵢ = wᵢ / (V(μᵢ) g′(μᵢ)²); β from the weighted least squares of z on X (by tyche's pivoted
// QR, never the normal equations); η = Xβ + o. It stops when the deviance D changes by less than
// tolerance · (|D| + 0.1) (R's criterion) and no coefficient moves by more than beta_tolerance ·
// max(1, |βⱼ|) (so that the linearly converging probit is not stopped early), halving the step
// towards the previous β when D is not
// finite or the mean leaves its range. For canonical links (identity, logit, log) IRLS is
// Newton's method on the log-likelihood and converges quadratically; for the probit, Fisher
// scoring, linearly. Separation in a binomial model (the MLE at infinity) shows as coefficients
// growing until max_iterations, with converged = false.
//
// THE COVARIANCE of β̂ (GlmOptions::covariance), with B = (XᵀWX)⁻¹ at convergence (the inverse
// expected information) and the score contributions uᵢ xᵢ, uᵢ = wᵢ(yᵢ − μᵢ)/(V(μᵢ) g′(μᵢ)):
// - model: φ̂ B, with φ̂ = Pearson χ²/(n − r) for the gaussian and φ = 1 for binomial and poisson;
// - hc0: B (Σ uᵢ² xᵢxᵢᵀ) B, the sandwich (Huber 1967; White 1982, Econometrica 50, 1–25);
//   hc1: hc0 · n/(n − r);
// - cluster: c · B (Σ_g (Σ_{i∈g} uᵢxᵢ)(…)ᵀ) B, c = G/(G − 1) · (n − 1)/(n − r) (Cameron and Miller
//   2015, as in linear.hpp; statsmodels' default correction).
// Wald statistics are referred to Student's t with n − r degrees of freedom for the gaussian (φ
// estimated, as R's summary.glm does) and to the normal otherwise; with clusters, t with G − 1.
//
// SUMMARIES. The deviance D = 2 Σ wᵢ [ℓᵢ(yᵢ) − ℓᵢ(μᵢ)] (McCullagh and Nelder Table 2.1): Σ w(y − μ)²
// (gaussian), 2 Σ m [y log(y/μ) + (1 − y) log((1 − y)/(1 − μ))] (binomial), 2 Σ w [y log(y/μ) −
// (y − μ)] (poisson), with 0 log 0 = 0; the null deviance is that of the model with only the
// intercept (and the offset), or of η = o when X has no constant column. The log-likelihood is
// the family's at μ̂ (the gaussian at σ² = Σ w(y − μ)²/n, plus ½ Σ log wᵢ; the binomial with its
// binomial coefficients, the poisson with log yᵢ!); AIC = −2ℓ + 2r and BIC = −2ℓ + r log n
// (statsmodels' convention: R's AIC for the gaussian adds 2 for φ); McFadden's pseudo-R² is
// 1 − ℓ/ℓ₀, ℓ₀ the null model's log-likelihood (McFadden 1974).

#pragma once

#include <tyche/probability/distributions.hpp>
#include <tyche/regression/linear.hpp>

#include <calx/special_functions/gamma.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::regression {

enum class Family { gaussian, binomial, poisson };
enum class Link { canonical, identity, logit, probit, log };
enum class GlmCovariance { model, hc0, hc1, cluster };

struct GlmOptions {
  Family family = Family::gaussian;
  Link link = Link::canonical;           // canonical: identity, logit, log for the three families
  std::vector<double> offset;            // empty, or one oᵢ per observation (e.g. log exposure)
  std::vector<double> weights;           // prior weights; for the binomial, the numbers of trials mᵢ
  GlmCovariance covariance = GlmCovariance::model;
  std::vector<std::int64_t> clusters;    // one label per observation, for GlmCovariance::cluster
  double confidence = 0.95;
  int max_iterations = 100;
  double tolerance = 1e-10;              // on |ΔD| / (|D| + 0.1)
  double beta_tolerance = 1e-10;         // and on |Δβⱼ| / max(1, |βⱼ|): both must hold
  bool drop_collinear = false;           // as OlsOptions::drop_collinear
};

struct GlmFit {
  std::size_t n = 0, k = 0, rank = 0;
  double df_residual = 0, df_inference = 0;
  Family family = Family::gaussian;
  Link link = Link::identity;            // the link used (canonical resolved)
  bool has_intercept = false;
  std::vector<bool> aliased;
  std::size_t clusters = 0;
  bool uses_t = false;                   // Wald statistics referred to t (gaussian, clusters) or the normal

  std::vector<double> beta, se, z, p, ci_lower, ci_upper;  // z: the Wald statistics β/se
  Matrix covariance;                     // k × k
  Matrix xtx_inverse;                    // (XᵀWX)⁻¹ at convergence

  std::vector<double> fitted;            // μ̂
  std::vector<double> linear_predictor;  // η̂ = Xβ̂ + o
  std::vector<double> residuals;         // y − μ̂ (response residuals)
  std::vector<double> pearson_residuals; // √w (y − μ̂)/√V(μ̂)
  std::vector<double> deviance_residuals;  // sign(y − μ̂) √dᵢ
  std::vector<double> working_weights;   // W at convergence

  double deviance = 0, null_deviance = 0, pearson_chi2 = 0, dispersion = 1;
  double log_likelihood = 0, null_log_likelihood = 0, aic = 0, bic = 0, pseudo_r_squared = 0;
  int iterations = 0;
  bool converged = false;
};

namespace detail {

inline Link resolve(Family f, Link l) {
  if (l != Link::canonical) return l;
  return f == Family::gaussian ? Link::identity : f == Family::binomial ? Link::logit : Link::log;
}

// μ = g⁻¹(η), with the probit's and logit's η clamped as R's binomial links clamp them.
inline double link_inverse(Link l, double eta) {
  switch (l) {
    case Link::identity: return eta;
    case Link::logit: {
      const double e = std::clamp(eta, -30.0, 30.0);
      return 1 / (1 + std::exp(-e));
    }
    case Link::probit: {
      const double e = std::clamp(eta, -8.125890664701906, 8.125890664701906);  // −Φ⁻¹(ε)
      return 0.5 * std::erfc(-e / std::numbers::sqrt2);
    }
    case Link::log: return std::exp(std::min(eta, 700.0));
    default: return eta;
  }
}
inline double link(Link l, double mu) {
  switch (l) {
    case Link::identity: return mu;
    case Link::logit: return std::log(mu / (1 - mu));
    case Link::probit: return probability::Normal::standard_quantile(mu);
    case Link::log: return std::log(mu);
    default: return mu;
  }
}
// g′(μ) = dη/dμ.
inline double link_derivative(Link l, double mu, double eta) {
  switch (l) {
    case Link::identity: return 1;
    case Link::logit: return 1 / (mu * (1 - mu));
    case Link::probit: {
      const double e = std::clamp(eta, -8.125890664701906, 8.125890664701906);
      return 1 / std::max(std::exp(-0.5 * e * e) / std::sqrt(2 * std::numbers::pi), std::numeric_limits<double>::min());
    }
    case Link::log: return 1 / mu;
    default: return 1;
  }
}
inline double variance_function(Family f, double mu) {
  switch (f) {
    case Family::gaussian: return 1;
    case Family::binomial: return mu * (1 - mu);
    case Family::poisson: return mu;
  }
  return 1;
}
inline bool valid_mean(Family f, double mu) {
  if (!std::isfinite(mu)) return false;
  if (f == Family::binomial) return mu > 0 && mu < 1;
  if (f == Family::poisson) return mu > 0;
  return true;
}
inline double xlogy(double x, double y) { return x == 0 ? 0.0 : x * std::log(y); }
// The unit deviance dᵢ (times the prior weight).
inline double unit_deviance(Family f, double y, double mu, double w) {
  switch (f) {
    case Family::gaussian: return w * (y - mu) * (y - mu);
    case Family::binomial: return 2 * w * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu)));
    case Family::poisson: return 2 * w * (xlogy(y, y / mu) - (y - mu));
  }
  return 0;
}
inline double deviance(Family f, std::span<const double> y, std::span<const double> mu, std::span<const double> w) {
  double d = 0;
  for (std::size_t i = 0; i < y.size(); i++) d += unit_deviance(f, y[i], mu[i], w.empty() ? 1.0 : w[i]);
  return d;
}
// The family's log-likelihood at μ (the gaussian's at its MLE σ²).
inline double log_likelihood(Family f, std::span<const double> y, std::span<const double> mu, std::span<const double> w) {
  const std::size_t n = y.size();
  double l = 0;
  switch (f) {
    case Family::gaussian: {
      double rss = 0, logw = 0;
      for (std::size_t i = 0; i < n; i++) {
        const double wi = w.empty() ? 1.0 : w[i];
        rss += wi * (y[i] - mu[i]) * (y[i] - mu[i]);
        logw += std::log(wi);
      }
      const double dn = static_cast<double>(n);
      return -0.5 * dn * (std::log(2 * std::numbers::pi * rss / dn) + 1) + 0.5 * logw;
    }
    case Family::binomial:
      for (std::size_t i = 0; i < n; i++) {
        const double m = w.empty() ? 1.0 : w[i], s = m * y[i];
        l += calx::special_functions::log_choose(m, std::round(s)) + xlogy(s, mu[i]) + xlogy(m - s, 1 - mu[i]);
      }
      return l;
    case Family::poisson:
      for (std::size_t i = 0; i < n; i++) {
        const double wi = w.empty() ? 1.0 : w[i];
        l += wi * (xlogy(y[i], mu[i]) - mu[i] - calx::special_functions::log_gamma(y[i] + 1));
      }
      return l;
  }
  return l;
}

struct IrlsResult {
  std::vector<double> beta, eta, mu, working_weight;
  WeightedLeastSquares last;  // the final weighted least squares (for covariances)
  int iterations = 0;
  bool converged = false;
  double deviance = 0;
};

inline IrlsResult irls(const Rectangular& X, std::span<const double> y, std::span<const double> offset, std::span<const double> w, Family family,
                       Link lk, int max_iterations, double tolerance, double beta_tolerance, bool drop_collinear) {
  const std::size_t n = X.rows, k = X.cols;
  IrlsResult r;
  r.mu.resize(n);
  r.eta.resize(n);
  for (std::size_t i = 0; i < n; i++) {
    const double m = w.empty() ? 1.0 : w[i];
    r.mu[i] = family == Family::binomial ? (m * y[i] + 0.5) / (m + 1) : family == Family::poisson ? y[i] + 0.1 : y[i];
    r.eta[i] = link(lk, r.mu[i]);
  }
  double dev_old = deviance(family, y, r.mu, w);
  std::vector<double> z(n), sw(n), beta_old;
  const double tol = static_cast<double>(std::max(n, k)) * std::numeric_limits<double>::epsilon();
  for (int it = 1; it <= max_iterations; it++) {
    r.working_weight.resize(n);
    for (std::size_t i = 0; i < n; i++) {
      const double gp = link_derivative(lk, r.mu[i], r.eta[i]);
      const double wi = w.empty() ? 1.0 : w[i];
      z[i] = r.eta[i] - (offset.empty() ? 0.0 : offset[i]) + (y[i] - r.mu[i]) * gp;
      r.working_weight[i] = wi / (variance_function(family, r.mu[i]) * gp * gp);
      sw[i] = std::sqrt(r.working_weight[i]);
    }
    r.last = weighted_least_squares(X, z, sw, tol, drop_collinear, "tyche::regression::glm");
    std::vector<double> beta = r.last.beta;
    // η, μ, and the deviance; halve the step while the mean is invalid or D is not finite.
    std::vector<double> eta(n), mu(n);
    double dev = 0;
    for (int half = 0; half < 40; half++) {
      bool ok = true;
      for (std::size_t i = 0; i < n && ok; i++) {
        double e = offset.empty() ? 0.0 : offset[i];
        for (std::size_t j = 0; j < k; j++)
          if (!r.last.aliased[j]) e += X(i, j) * beta[j];
        eta[i] = e;
        mu[i] = link_inverse(lk, e);
        ok = valid_mean(family, mu[i]);
      }
      if (ok) {
        dev = deviance(family, y, mu, w);
        ok = std::isfinite(dev);
      }
      if (ok || beta_old.empty()) break;
      for (std::size_t j = 0; j < k; j++)
        if (!r.last.aliased[j]) beta[j] = 0.5 * (beta[j] + beta_old[j]);
    }
    bool still = !beta_old.empty();
    for (std::size_t j = 0; j < k && still; j++)
      if (!r.last.aliased[j]) still = std::fabs(beta[j] - beta_old[j]) <= beta_tolerance * std::max(1.0, std::fabs(beta[j]));
    r.beta = beta;
    r.eta = eta;
    r.mu = mu;
    r.deviance = dev;
    r.iterations = it;
    beta_old = beta;
    if (still && std::fabs(dev - dev_old) / (std::fabs(dev) + 0.1) < tolerance) {
      r.converged = true;
      break;
    }
    dev_old = dev;
  }
  // The working weights at the final μ, for the covariance.
  for (std::size_t i = 0; i < n; i++) {
    const double gp = link_derivative(lk, r.mu[i], r.eta[i]);
    r.working_weight[i] = (w.empty() ? 1.0 : w[i]) / (variance_function(family, r.mu[i]) * gp * gp);
    sw[i] = std::sqrt(r.working_weight[i]);
    z[i] = r.eta[i] - (offset.empty() ? 0.0 : offset[i]);
  }
  r.last = weighted_least_squares(X, z, sw, tol, drop_collinear, "tyche::regression::glm");
  return r;
}

}  // namespace detail

inline GlmFit glm(const Rectangular& X, std::span<const double> y, const GlmOptions& options = {}) {
  const std::size_t n = X.rows, k = X.cols;
  if (y.size() != n) throw std::invalid_argument("tyche::regression::glm: y has one entry per row of X");
  if (!options.offset.empty() && options.offset.size() != n) throw std::invalid_argument("tyche::regression::glm: one offset per observation");
  if (!options.weights.empty() && options.weights.size() != n) throw std::invalid_argument("tyche::regression::glm: one weight per observation");
  if (options.covariance == GlmCovariance::cluster && options.clusters.size() != n)
    throw std::invalid_argument("tyche::regression::glm: one cluster label per observation for GlmCovariance::cluster");
  for (std::size_t i = 0; i < n; i++) {
    const double wi = options.weights.empty() ? 1.0 : options.weights[i];
    if (!(wi > 0 && std::isfinite(wi))) throw std::invalid_argument("tyche::regression::glm: weights positive and finite");
    if (options.family == Family::binomial && !(y[i] >= 0 && y[i] <= 1)) throw std::invalid_argument("tyche::regression::glm: binomial y is a proportion in [0, 1]");
    if (options.family == Family::poisson && !(y[i] >= 0)) throw std::invalid_argument("tyche::regression::glm: poisson y ≥ 0");
  }
  const Link lk = detail::resolve(options.family, options.link);
  const auto fitr = detail::irls(X, y, options.offset, options.weights, options.family, lk, options.max_iterations, options.tolerance, options.beta_tolerance, options.drop_collinear);
  const auto& w = fitr.last;
  const std::size_t r = w.rank;
  if (n <= r) throw std::invalid_argument("tyche::regression::glm: more observations than coefficients are needed");

  GlmFit g;
  g.n = n;
  g.k = k;
  g.rank = r;
  g.df_residual = static_cast<double>(n - r);
  g.family = options.family;
  g.link = lk;
  g.has_intercept = detail::has_constant_column(X);
  g.aliased = w.aliased;
  g.beta = fitr.beta;
  g.fitted = fitr.mu;
  g.linear_predictor = fitr.eta;
  g.working_weights = fitr.working_weight;
  g.iterations = fitr.iterations;
  g.converged = fitr.converged;
  g.deviance = fitr.deviance;

  // Residuals and the dispersion.
  g.residuals.resize(n);
  g.pearson_residuals.resize(n);
  g.deviance_residuals.resize(n);
  std::vector<double> u(n);  // √W (y − μ) g′(μ): the score contributions in √W X coordinates
  for (std::size_t i = 0; i < n; i++) {
    const double wi = options.weights.empty() ? 1.0 : options.weights[i];
    const double mu = g.fitted[i], V = detail::variance_function(options.family, mu);
    g.residuals[i] = y[i] - mu;
    g.pearson_residuals[i] = std::sqrt(wi) * (y[i] - mu) / std::sqrt(V);
    g.pearson_chi2 += g.pearson_residuals[i] * g.pearson_residuals[i];
    g.deviance_residuals[i] = std::copysign(std::sqrt(std::max(0.0, detail::unit_deviance(options.family, y[i], mu, wi))), y[i] - mu);
    u[i] = std::sqrt(g.working_weights[i]) * (y[i] - mu) * detail::link_derivative(lk, mu, g.linear_predictor[i]);
  }
  g.dispersion = options.family == Family::gaussian ? g.pearson_chi2 / g.df_residual : 1.0;

  // Covariance.
  const double dn = static_cast<double>(n), dr = static_cast<double>(r);
  const Matrix I = Matrix::identity(r);
  g.xtx_inverse = detail::unscale(w, I, 0.0);
  std::size_t groups = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  switch (options.covariance) {
    case GlmCovariance::model: g.covariance = detail::unscale(w, g.dispersion * I, nan); break;
    case GlmCovariance::hc0: g.covariance = detail::unscale(w, detail::meat(w, u, {}, {}, groups), nan); break;
    case GlmCovariance::hc1: g.covariance = detail::unscale(w, (dn / (dn - dr)) * detail::meat(w, u, {}, {}, groups), nan); break;
    case GlmCovariance::cluster: {
      const Matrix M = detail::meat(w, u, {}, options.clusters, groups);
      if (groups < 2) throw std::invalid_argument("tyche::regression::glm: at least two clusters");
      const double G = static_cast<double>(groups);
      g.covariance = detail::unscale(w, (G / (G - 1) * (dn - 1) / (dn - dr)) * M, nan);
      g.clusters = groups;
      break;
    }
  }
  g.uses_t = options.family == Family::gaussian || options.covariance == GlmCovariance::cluster;
  g.df_inference = options.covariance == GlmCovariance::cluster ? static_cast<double>(groups) - 1 : g.df_residual;

  // Wald inference on each coefficient.
  const probability::StudentT tdist{g.uses_t ? g.df_inference : probability::infinity};
  const double crit = tdist.upper_quantile((1 - options.confidence) / 2);
  g.se.assign(k, nan);
  g.z = g.p = g.ci_lower = g.ci_upper = g.se;
  for (std::size_t j = 0; j < k; j++) {
    if (w.aliased[j]) continue;
    g.se[j] = std::sqrt(g.covariance(j, j));
    g.z[j] = g.beta[j] / g.se[j];
    g.p[j] = std::min(1.0, 2 * tdist.survival(std::fabs(g.z[j])));
    g.ci_lower[j] = g.beta[j] - crit * g.se[j];
    g.ci_upper[j] = g.beta[j] + crit * g.se[j];
  }

  // Likelihoods: the fit and the null model (intercept and offset only, or η = o).
  g.log_likelihood = detail::log_likelihood(options.family, y, g.fitted, options.weights);
  std::vector<double> mu0(n);
  if (g.has_intercept) {
    Rectangular ones(n, 1);
    std::fill(ones.a.begin(), ones.a.end(), 1.0);
    mu0 = detail::irls(ones, y, options.offset, options.weights, options.family, lk, options.max_iterations, options.tolerance, options.beta_tolerance, false).mu;
  } else {
    for (std::size_t i = 0; i < n; i++) mu0[i] = detail::link_inverse(lk, options.offset.empty() ? 0.0 : options.offset[i]);
  }
  g.null_deviance = detail::deviance(options.family, y, mu0, options.weights);
  g.null_log_likelihood = detail::log_likelihood(options.family, y, mu0, options.weights);
  g.aic = -2 * g.log_likelihood + 2 * dr;
  g.bic = -2 * g.log_likelihood + dr * std::log(dn);
  g.pseudo_r_squared = 1 - g.log_likelihood / g.null_log_likelihood;
  return g;
}

}  // namespace tyche::regression
