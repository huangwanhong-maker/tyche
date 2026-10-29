// Autoregressive and ARIMA models: AR(p) by least squares and by Yule–Walker; ARIMA(p, d, q) by
// conditional sum of squares; ψ-weights and forecasts; stationarity and invertibility.
//
// CONVENTIONS. With B the backshift operator, w_t = (1 − B)^d x_t the d-th difference,
//   φ(B)(w_t − μ) = θ(B) ε_t,  φ(B) = 1 − φ₁B − … − φ_pB^p,  θ(B) = 1 + θ₁B + … + θ_qB^q,
// ε_t white noise of variance σ². The AR signs are Box and Jenkins's; the MA signs are R's and
// statsmodels' (Box, Jenkins and Reinsel 2008 write θ(B) = 1 − θ₁B − …, the opposite sign).
// μ is the mean of the differenced series (with d = 1, a drift); constant = false fixes μ = 0.
// The model is stationary when the roots of φ(z) lie outside the unit circle and invertible when
// those of θ(z) do; equivalently, when the companion matrices of the polynomials have spectral
// radius below 1 (calx::linalg::eigenvalues).
//
// - ar_ols(x, p): x_t = c + Σ φ_i x_{t−i} + ε_t by least squares on t = p + 1, …, n (tyche's
//   pivoted QR), with the fit's classical standard errors: the conditional maximum likelihood
//   estimator of a Gaussian AR(p).
// - ar_yule_walker(x, p): the φ solving the Yule–Walker equations Γ_p φ = γ_p of the sample
//   autocovariances (divisor n), by Durbin–Levinson; σ² = v_p, the mean x̄, c = x̄(1 − Σφ)
//   (Brockwell and Davis 1991, §8.1). Always stationary (Γ_p is positive definite).
// - arima(x, {p, d, q}): conditional sum of squares (Box, Jenkins and Reinsel 2008, §7.1):
//   residuals e_t = (w_t − μ) − Σ φ_i (w_{t−i} − μ) − Σ θ_j e_{t−j} for t = p + 1, …, N (N = n − d),
//   with e_t = 0 for t ≤ p, and S(μ, φ, θ) = Σ e_t² minimised by calx's BFGS (gradient by central
//   differences) from μ = w̄, Yule–Walker φ and θ = 0, polished by Nelder–Mead when BFGS stops
//   short. σ̂² = S/m with m = N − p residuals; the conditional log-likelihood
//   ℓ = −(m/2)(log 2πσ̂² + 1); AIC = −2ℓ + 2(k + 1) and BIC = −2ℓ + (k + 1) log m with k the number
//   of μ, φ, θ (σ² counted, as R's arima counts it); the covariance of the estimates
//   2σ̂² H_S⁻¹, H_S the Hessian of S by central differences (the profile likelihood's inverse
//   Hessian, as R's CSS reports it). For a pure AR the CSS estimate is ar_ols's (exactly: both are
//   least squares on the same residuals, with c = μ(1 − Σφ)).
// - psi_weights(phi, theta, n): ψ₀ = 1, ψ_j = θ_j + Σ_{i=1}^{min(j, p)} φ_i ψ_{j−i}, the weights of
//   the moving-average representation x_t = Σ ψ_j ε_{t−j} (Box, Jenkins and Reinsel, §5.2).
// - forecast(fit, h): the minimum mean-square-error forecasts of x_{n+1..n+h} given the CSS
//   residuals, with standard errors σ̂ (Σ_{j<k} ψ_j²)^{1/2} from the ψ-weights of φ(B)(1 − B)^d
//   (Box, Jenkins and Reinsel, §5.1–5.2), and normal intervals; parameter
//   uncertainty is not included.

#pragma once

#include <tyche/descriptive/summary.hpp>
#include <tyche/probability/distributions.hpp>
#include <tyche/regression/linear.hpp>
#include <tyche/time_series/autocorrelation.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigenvalues.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/optimisation/nelder_mead.hpp>
#include <calx/optimisation/quasi_newton.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

using calx::linalg::Matrix;

// The spectral radius of the companion matrix of z^p − c₁z^{p−1} − … − c_p: below 1 exactly when
// 1 − c₁z − … − c_pz^p has all its roots outside the unit circle.
inline double companion_radius(std::span<const double> c) {
  const std::size_t p = c.size();
  if (p == 0) return 0;
  Matrix C(p);
  for (std::size_t j = 0; j < p; j++) C(0, j) = c[j];
  for (std::size_t i = 1; i < p; i++) C(i, i - 1) = 1;
  return calx::linalg::spectral_radius(C);
}

// φ(z) = 1 − Σ φ_i z^i has its roots outside the unit circle.
inline bool stationary(std::span<const double> phi) { return companion_radius(phi) < 1; }

// θ(z) = 1 + Σ θ_j z^j has its roots outside the unit circle.
inline bool invertible(std::span<const double> theta) {
  std::vector<double> neg(theta.begin(), theta.end());
  for (double& v : neg) v = -v;
  return companion_radius(neg) < 1;
}

inline std::vector<double> psi_weights(std::span<const double> phi, std::span<const double> theta, std::size_t n) {
  std::vector<double> psi(n, 0.0);
  if (n == 0) return psi;
  psi[0] = 1;
  for (std::size_t j = 1; j < n; j++) {
    double v = j <= theta.size() ? theta[j - 1] : 0.0;
    for (std::size_t i = 1; i <= std::min(j, phi.size()); i++) v += phi[i - 1] * psi[j - i];
    psi[j] = v;
  }
  return psi;
}

// ---- AR(p) ----
struct ArFit {
  std::size_t p = 0;
  double constant = 0;            // c in x_t = c + Σ φ_i x_{t−i} + ε_t
  double mean = 0;                // μ = c / (1 − Σ φ_i)
  std::vector<double> phi;
  double sigma2 = 0;              // innovation variance (OLS: RSS/(m − p − 1); Yule–Walker: v_p)
  std::vector<double> se;         // OLS: standard errors of (c, φ₁, …, φ_p); Yule–Walker: empty
  bool is_stationary = false;
};

inline ArFit ar_ols(std::span<const double> x, std::size_t p, bool constant = true) {
  const std::size_t n = x.size();
  if (n <= 2 * p + 2) throw std::invalid_argument("tyche::time_series::ar_ols: too short a series for the order");
  const std::size_t m = n - p, c0 = constant ? 1 : 0;
  regression::Rectangular X(m, p + c0);
  std::vector<double> y(m);
  for (std::size_t t = 0; t < m; t++) {
    if (constant) X(t, 0) = 1;
    for (std::size_t i = 1; i <= p; i++) X(t, c0 + i - 1) = x[p + t - i];
    y[t] = x[p + t];
  }
  const auto f = regression::ols(X, y);
  ArFit a;
  a.p = p;
  a.constant = constant ? f.beta[0] : 0.0;
  a.phi.assign(f.beta.begin() + static_cast<std::ptrdiff_t>(c0), f.beta.end());
  double s = 1;
  for (double v : a.phi) s -= v;
  a.mean = a.constant / s;
  a.sigma2 = f.sigma * f.sigma;
  a.se = f.se;
  a.is_stationary = stationary(a.phi);
  return a;
}

inline ArFit ar_yule_walker(std::span<const double> x, std::size_t p) {
  const auto g = autocovariance(x, p);
  const auto d = durbin_levinson(g, p);
  ArFit a;
  a.p = p;
  a.phi = d.phi;
  a.sigma2 = d.variance.back();
  a.mean = descriptive::mean(x);
  double s = 1;
  for (double v : a.phi) s -= v;
  a.constant = a.mean * s;
  a.is_stationary = stationary(a.phi);
  return a;
}

// ---- ARIMA by conditional sum of squares ----
struct ArimaOrder {
  std::size_t p = 0, d = 0, q = 0;
};

struct ArimaOptions {
  bool constant = true;          // estimate μ, the mean of the differenced series
  double confidence = 0.95;      // for forecast intervals
};

struct ArimaFit {
  ArimaOrder order;
  bool has_constant = true;
  double mu = 0;                 // the mean of w (0 without a constant)
  std::vector<double> phi, theta;
  // Standard errors and covariance of the estimated parameters, in the order (μ,) φ₁…φ_p, θ₁…θ_q.
  std::vector<double> se;
  Matrix covariance;
  double sigma2 = 0, css = 0;
  std::size_t n_residuals = 0;   // m = n − d − p
  double log_likelihood = 0, aic = 0, bic = 0;
  std::vector<double> residuals; // e_t for t = p + 1, …, N of the differenced series (m values)
  bool converged = false, is_stationary = false, is_invertible = false;
  // Kept for forecasting: the series and its differences (level k = Δ^k x).
  std::vector<std::vector<double>> levels;
  double confidence = 0.95;
};

namespace detail {

inline std::vector<double> difference(std::span<const double> x) {
  std::vector<double> d;
  for (std::size_t t = 1; t < x.size(); t++) d.push_back(x[t] - x[t - 1]);
  return d;
}

// The CSS residuals of w under (μ, φ, θ).
inline double css_residuals(std::span<const double> w, double mu, std::span<const double> phi, std::span<const double> theta, std::vector<double>* out) {
  const std::size_t N = w.size(), p = phi.size(), q = theta.size();
  std::vector<double> e(N, 0.0);
  double s = 0;
  for (std::size_t t = p; t < N; t++) {
    double v = w[t] - mu;
    for (std::size_t i = 1; i <= p; i++) v -= phi[i - 1] * (w[t - i] - mu);
    for (std::size_t j = 1; j <= q && j <= t; j++) v -= theta[j - 1] * e[t - j];
    e[t] = v;
    s += v * v;
  }
  if (out) out->assign(e.begin() + static_cast<std::ptrdiff_t>(p), e.end());
  return s;
}

}  // namespace detail

inline ArimaFit arima(std::span<const double> x, ArimaOrder order, const ArimaOptions& options = {}) {
  const std::size_t p = order.p, d = order.d, q = order.q;
  ArimaFit fit;
  fit.order = order;
  fit.has_constant = options.constant;
  fit.confidence = options.confidence;
  fit.levels.emplace_back(x.begin(), x.end());
  for (std::size_t k = 0; k < d; k++) fit.levels.push_back(detail::difference(fit.levels.back()));
  const std::vector<double>& w = fit.levels.back();
  const std::size_t N = w.size();
  const std::size_t k = (options.constant ? 1 : 0) + p + q;
  if (N < p + k + 3) throw std::invalid_argument("tyche::time_series::arima: too short a series for the order");

  auto unpack = [&](std::span<const double> th, double& mu, std::vector<double>& ph, std::vector<double>& tq) {
    std::size_t i = 0;
    mu = options.constant ? th[i++] : 0.0;
    ph.assign(th.begin() + static_cast<std::ptrdiff_t>(i), th.begin() + static_cast<std::ptrdiff_t>(i + p));
    tq.assign(th.begin() + static_cast<std::ptrdiff_t>(i + p), th.begin() + static_cast<std::ptrdiff_t>(i + p + q));
  };
  // S scaled by the sample variance of w, so that the optimiser sees values of order m.
  const double scale = std::max(descriptive::variance(w, 0), std::numeric_limits<double>::min());
  auto objective = [&](std::span<const double> th) {
    double mu;
    std::vector<double> ph, tq;
    unpack(th, mu, ph, tq);
    const double s = detail::css_residuals(w, mu, ph, tq, nullptr) / scale;
    return std::isfinite(s) ? s : std::numeric_limits<double>::infinity();
  };
  std::vector<double> start;
  if (options.constant) start.push_back(descriptive::mean(w));
  if (p > 0) {
    const auto yw = ar_yule_walker(w, p);
    start.insert(start.end(), yw.phi.begin(), yw.phi.end());
  }
  for (std::size_t j = 0; j < q; j++) start.push_back(0.0);

  std::vector<double> theta_hat = start;
  if (k > 0) {
    calx::optimisation::BfgsOptions bo;
    bo.gradient_tolerance = 1e-8 * static_cast<double>(N);
    auto r = calx::optimisation::bfgs(objective, start, bo);
    theta_hat = r.x;
    fit.converged = r.converged;
    if (!r.converged) {  // polish without derivatives
      calx::optimisation::NelderMeadOptions no;
      no.x_tolerance = 1e-10;
      no.f_tolerance = 1e-12;
      no.adaptive = true;
      no.max_evaluations = 20000;
      const auto nm = calx::optimisation::nelder_mead(objective, theta_hat, no);
      if (nm.value <= r.value) theta_hat = nm.x;
      fit.converged = nm.converged;
    }
  } else {
    fit.converged = true;
  }
  unpack(theta_hat, fit.mu, fit.phi, fit.theta);
  fit.css = detail::css_residuals(w, fit.mu, fit.phi, fit.theta, &fit.residuals);
  fit.n_residuals = N - p;
  const double m = static_cast<double>(fit.n_residuals);
  fit.sigma2 = fit.css / m;
  fit.log_likelihood = -0.5 * m * (std::log(2 * std::numbers::pi * fit.sigma2) + 1);
  fit.aic = -2 * fit.log_likelihood + 2 * static_cast<double>(k + 1);
  fit.bic = -2 * fit.log_likelihood + static_cast<double>(k + 1) * std::log(m);
  fit.is_stationary = stationary(fit.phi);
  fit.is_invertible = invertible(fit.theta);
  // Covariance 2σ̂² H_S⁻¹, with H_S the Hessian of S (unscaled).
  fit.se.assign(k, std::numeric_limits<double>::quiet_NaN());
  fit.covariance = Matrix(k);
  if (k > 0) {
    Matrix H = calx::optimisation::hessian_by_differences(objective, theta_hat);
    for (double& v : H.a) v *= scale;
    try {
      fit.covariance = (2 * fit.sigma2) * calx::linalg::inverse(H);
      for (std::size_t i = 0; i < k; i++) fit.se[i] = fit.covariance(i, i) > 0 ? std::sqrt(fit.covariance(i, i)) : std::numeric_limits<double>::quiet_NaN();
    } catch (const std::runtime_error&) {
      // A singular Hessian (an unidentified model, e.g. a common AR and MA factor): no standard errors.
    }
  }
  return fit;
}

struct Forecast {
  std::vector<double> mean, se, lower, upper;  // for x_{n+1}, …, x_{n+h}
};

inline Forecast forecast(const ArimaFit& fit, std::size_t horizon) {
  const std::size_t p = fit.order.p, q = fit.order.q, d = fit.order.d;
  const std::vector<double>& w = fit.levels.back();
  const std::size_t N = w.size();
  // Residuals aligned with w (zero before p).
  std::vector<double> e(N, 0.0);
  for (std::size_t t = p; t < N; t++) e[t] = fit.residuals[t - p];
  // Forecasts of w.
  std::vector<double> ww(w.begin(), w.end());
  for (std::size_t h = 1; h <= horizon; h++) {
    const std::size_t t = N + h - 1;  // index of the forecast in ww
    double v = fit.mu;
    for (std::size_t i = 1; i <= p; i++) v += fit.phi[i - 1] * (ww[t - i] - fit.mu);
    for (std::size_t j = h; j <= q; j++)
      if (t >= j && t - j < N) v += fit.theta[j - 1] * e[t - j];
    ww.push_back(v);
  }
  // Integrate back: level k forecasts = last level-k value + cumulative sums of level k + 1's.
  std::vector<double> f(ww.begin() + static_cast<std::ptrdiff_t>(N), ww.end());
  for (std::size_t k = d; k-- > 0;) {
    double last = fit.levels[k].back();
    for (double& v : f) {
      last += v;
      v = last;
    }
  }
  // ψ-weights of φ(B)(1 − B)^d.
  std::vector<double> big(fit.phi.begin(), fit.phi.end());  // coefficients of 1 − Σ big_i B^i
  for (std::size_t k = 0; k < d; k++) {
    // (1 − Σ a_i B^i)(1 − B) = 1 − Σ (a_i − a_{i−1}) B^i with a_0 = −1 … as coefficients:
    std::vector<double> next(big.size() + 1, 0.0);
    for (std::size_t i = 0; i <= big.size(); i++) {
      const double ai = i < big.size() ? big[i] : 0.0;      // coefficient of B^{i+1}
      const double prev = i == 0 ? -1.0 : big[i - 1];       // coefficient of B^i (a_0 = −1)
      next[i] = ai - prev;
    }
    big = std::move(next);
  }
  const auto psi = psi_weights(big, fit.theta, horizon);
  const double z = probability::Normal{}.upper_quantile((1 - fit.confidence) / 2);
  Forecast out;
  double acc = 0;
  for (std::size_t h = 0; h < horizon; h++) {
    acc += psi[h] * psi[h];
    const double se = std::sqrt(fit.sigma2 * acc);
    out.mean.push_back(f[h]);
    out.se.push_back(se);
    out.lower.push_back(f[h] - z * se);
    out.upper.push_back(f[h] + z * se);
  }
  return out;
}

}  // namespace tyche::time_series
