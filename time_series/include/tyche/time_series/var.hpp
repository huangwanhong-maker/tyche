// Vector autoregressions: estimation by least squares, the choice of order, stability, impulse
// responses and forecasts.
//
// THE MODEL (Lütkepohl 2005, New Introduction to Multiple Time Series Analysis, ch. 2–4). A K-vector
// y_t = ν + A₁ y_{t−1} + … + A_p y_{t−p} + u_t, u_t white noise with covariance Σ_u.
//
// - var(Y, p): the least-squares estimator, equation by equation (tyche's pivoted QR), which is
//   the multivariate LS estimator and, for Gaussian u, the conditional ML estimator (Lütkepohl
//   §3.2–3.4); on t = p + 1, …, T, so T_e = T − p observations. Σ̂_u = ÛÛᵀ/(T_e − Kp − 1), with
//   Lütkepohl's degrees-of-freedom correction (§3.2.2), and the ML Σ̃_u = ÛÛᵀ/T_e. Each equation's
//   standard errors are its LinearFit's.
// - Information criteria (Lütkepohl §4.3), with Σ̃_u(m) on T_e observations:
//   FPE(m) = ((T_e + Km + 1)/(T_e − Km − 1))^K |Σ̃_u| (Akaike 1969; §4.3.1),
//   AIC(m) = ln|Σ̃_u| + 2mK²/T_e (Akaike 1973), HQ(m) = ln|Σ̃_u| + 2mK² ln ln T_e / T_e (Hannan and
//   Quinn 1979), SC(m) = ln|Σ̃_u| + mK² ln T_e / T_e (Schwarz 1978) (§4.3.2): the intercepts are not
//   counted (as in Lütkepohl), which leaves every choice of order unchanged.
//   select_order(Y, p_max) fits every m = 0, …, p_max on the same sample t = p_max + 1, …, T.
// - Stability: the companion matrix A of order Kp (A₁ … A_p in its first K rows, the identity
//   below) has spectral radius below 1 (Lütkepohl §2.1.1), by calx's eigenvalues.
// - Impulse responses: Φ₀ = I, Φ_i = Σ_{j=1}^{min(i,p)} Φ_{i−j} A_j, the coefficients of the
//   moving-average representation and the responses to a unit u (§2.1.2, §2.3.2); orthogonalised
//   Θ_i = Φ_i P with P the lower Cholesky factor of Σ̂_u (§2.3.2), the responses to
//   one-standard-deviation orthogonal shocks in the order of the variables. Θ_i(a, b) is the
//   response of variable a at horizon i to shock b.
// - Forecasts: ŷ_T(h) = ν + Σ_j A_j ŷ_T(h − j), with ŷ_T(h − j) = y_{T+h−j} for h ≤ j, and the MSE
//   Σ_ŷ(h) = Σ_{i<h} Φ_i Σ̂_u Φ_iᵀ (§2.2.2), which ignores the estimation uncertainty (Lütkepohl
//   §3.5 gives its correction, not applied here).

#pragma once

#include <tyche/regression/linear.hpp>

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigenvalues.hpp>
#include <calx/linalg/rectangular.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

using calx::linalg::Matrix;
using calx::linalg::Rectangular;

struct VarFit {
  std::size_t K = 0, p = 0, T = 0, T_effective = 0;  // variables, order, observations, T − p
  bool has_constant = true;
  std::vector<double> nu;                 // the intercepts ν (zero without a constant)
  std::vector<Matrix> A;                  // A₁, …, A_p, each K × K
  Matrix sigma_u;                         // Σ̂_u, divisor T_e − Kp − 1 (− 0 without a constant: T_e − Kp)
  Matrix sigma_ml;                        // Σ̃_u, divisor T_e
  Rectangular residuals;                  // T_e × K
  std::vector<regression::LinearFit> equations;  // one per variable: columns (1,) y_{t−1}, …, y_{t−p}
  double log_likelihood = 0;              // −(T_e/2)(K ln 2π + ln|Σ̃_u| + K)
  double aic = 0, bic = 0, hq = 0, fpe = 0;
  double spectral_radius = 0;             // of the companion matrix
  bool is_stable = false;
  Rectangular last;                       // the last p observations (p × K), oldest first, for forecasts
};

namespace detail {

inline double log_det(const Matrix& S) {
  const auto c = calx::linalg::cholesky(S);
  if (!c.positive_definite) return calx::linalg::log_determinant(S).log_abs;
  double s = 0;
  for (std::size_t i = 0; i < S.n; i++) s += 2 * std::log(c.l(i, i));
  return s;
}

// Fits order p on rows start, …, T − 1 of Y (start ≥ p).
inline VarFit fit_var(const Rectangular& Y, std::size_t p, bool constant, std::size_t start) {
  const std::size_t T = Y.rows, K = Y.cols, Te = T - start, c0 = constant ? 1 : 0, kc = c0 + K * p;
  if (Te <= kc) throw std::invalid_argument("tyche::time_series::var: too few observations for the order");
  Rectangular Z(Te, kc);
  for (std::size_t r = 0; r < Te; r++) {
    const std::size_t t = start + r;
    if (constant) Z(r, 0) = 1;
    for (std::size_t j = 1; j <= p; j++)
      for (std::size_t b = 0; b < K; b++) Z(r, c0 + (j - 1) * K + b) = Y(t - j, b);
  }
  VarFit v;
  v.K = K;
  v.p = p;
  v.T = T;
  v.T_effective = Te;
  v.has_constant = constant;
  v.nu.assign(K, 0.0);
  v.A.assign(p, Matrix(K));
  v.residuals = Rectangular(Te, K);
  for (std::size_t a = 0; a < K; a++) {
    std::vector<double> y(Te);
    for (std::size_t r = 0; r < Te; r++) y[r] = Y(start + r, a);
    if (kc == 0) {  // order 0 without a constant: y_t = u_t
      regression::LinearFit f;
      f.residuals = y;
      v.equations.push_back(f);
    } else {
      v.equations.push_back(regression::ols(Z, y));
    }
    const auto& f = v.equations.back();
    if (constant) v.nu[a] = f.beta[0];
    for (std::size_t j = 0; j < p; j++)
      for (std::size_t b = 0; b < K; b++) v.A[j](a, b) = f.beta[c0 + j * K + b];
    for (std::size_t r = 0; r < Te; r++) v.residuals(r, a) = f.residuals[r];
  }
  v.sigma_ml = calx::linalg::gram(v.residuals);
  v.sigma_u = v.sigma_ml;
  const double dTe = static_cast<double>(Te), dK = static_cast<double>(K), dp = static_cast<double>(p);
  for (double& s : v.sigma_ml.a) s /= dTe;
  for (double& s : v.sigma_u.a) s /= dTe - static_cast<double>(kc);
  const double ld = log_det(v.sigma_ml);
  v.log_likelihood = -0.5 * dTe * (dK * std::log(2 * 3.141592653589793) + ld + dK);
  const double free = dp * dK * dK;
  v.aic = ld + 2 * free / dTe;
  v.bic = ld + free * std::log(dTe) / dTe;
  v.hq = ld + 2 * free * std::log(std::log(dTe)) / dTe;
  v.fpe = std::pow((dTe + dK * dp + 1) / (dTe - dK * dp - 1), dK) * std::exp(ld);
  // Stability.
  if (p > 0) {
    Matrix C(K * p);
    for (std::size_t j = 0; j < p; j++)
      for (std::size_t a = 0; a < K; a++)
        for (std::size_t b = 0; b < K; b++) C(a, j * K + b) = v.A[j](a, b);
    for (std::size_t i = K; i < K * p; i++) C(i, i - K) = 1;
    v.spectral_radius = calx::linalg::spectral_radius(C);
  }
  v.is_stable = v.spectral_radius < 1;
  v.last = Rectangular(p, K);
  for (std::size_t j = 0; j < p; j++)
    for (std::size_t b = 0; b < K; b++) v.last(j, b) = Y(T - p + j, b);
  return v;
}

}  // namespace detail

// Y: T × K, one row per time, one column per variable.
inline VarFit var(const Rectangular& Y, std::size_t p, bool constant = true) { return detail::fit_var(Y, p, constant, p); }

struct VarOrder {
  std::size_t aic = 0, bic = 0, hq = 0, fpe = 0;           // the orders chosen
  std::vector<double> aic_values, bic_values, hq_values, fpe_values;  // for m = 0, …, p_max
};

inline VarOrder select_order(const Rectangular& Y, std::size_t p_max, bool constant = true) {
  VarOrder o;
  for (std::size_t m = 0; m <= p_max; m++) {
    const VarFit v = detail::fit_var(Y, m, constant, p_max);
    o.aic_values.push_back(v.aic);
    o.bic_values.push_back(v.bic);
    o.hq_values.push_back(v.hq);
    o.fpe_values.push_back(v.fpe);
  }
  auto argmin = [](const std::vector<double>& x) {
    std::size_t b = 0;
    for (std::size_t i = 1; i < x.size(); i++)
      if (x[i] < x[b]) b = i;
    return b;
  };
  o.aic = argmin(o.aic_values);
  o.bic = argmin(o.bic_values);
  o.hq = argmin(o.hq_values);
  o.fpe = argmin(o.fpe_values);
  return o;
}

// Φ₀, …, Φ_h (orthogonalised = false) or Θ₀, …, Θ_h = Φ_i P (true).
inline std::vector<Matrix> impulse_responses(const VarFit& v, std::size_t horizon, bool orthogonalised = true) {
  const std::size_t K = v.K;
  std::vector<Matrix> Phi{Matrix::identity(K)};
  for (std::size_t i = 1; i <= horizon; i++) {
    Matrix M(K);
    for (std::size_t j = 1; j <= std::min(i, v.p); j++) M = M + Phi[i - j] * v.A[j - 1];
    Phi.push_back(M);
  }
  if (!orthogonalised) return Phi;
  const auto c = calx::linalg::cholesky(v.sigma_u);
  if (!c.positive_definite) throw std::runtime_error("tyche::time_series::impulse_responses: Σ_u is not positive definite");
  for (auto& M : Phi) M = M * c.l;
  return Phi;
}

struct VarForecast {
  Rectangular mean;  // h × K
  Rectangular se;    // h × K: √ of the diagonal of Σ_ŷ(h)
};

inline VarForecast forecast(const VarFit& v, std::size_t horizon) {
  const std::size_t K = v.K, p = v.p;
  VarForecast f{Rectangular(horizon, K), Rectangular(horizon, K)};
  std::vector<std::vector<double>> path;  // the last p observations, then the forecasts
  for (std::size_t j = 0; j < p; j++) path.push_back(v.last.row(j));
  for (std::size_t h = 0; h < horizon; h++) {
    std::vector<double> y = v.nu;
    for (std::size_t j = 1; j <= p; j++) {
      const auto& prev = path[path.size() - j];
      for (std::size_t a = 0; a < K; a++)
        for (std::size_t b = 0; b < K; b++) y[a] += v.A[j - 1](a, b) * prev[b];
    }
    for (std::size_t a = 0; a < K; a++) f.mean(h, a) = y[a];
    path.push_back(y);
  }
  const auto Phi = impulse_responses(v, horizon == 0 ? 0 : horizon - 1, false);
  Matrix mse(K);
  for (std::size_t h = 0; h < horizon; h++) {
    mse = mse + Phi[h] * v.sigma_u * calx::linalg::transpose(Phi[h]);
    for (std::size_t a = 0; a < K; a++) f.se(h, a) = std::sqrt(mse(a, a));
  }
  return f;
}

}  // namespace tyche::time_series
