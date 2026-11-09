// Spatial regression: the spatial lag and spatial error models by maximum likelihood, the Lagrange multiplier tests that choose
// between them, and the lag model's impacts (tyche's decision 0015).
//
// THE MODELS (Ord 1975; Anselin 1988, Spatial Econometrics, Kluwer, ch. 6 and 8). With W a matrix of spatial weights (zero diagonal,
// rows as the caller declares them, row-standardised as a rule):
// - lag:   y = ρ W y + X β + ε,             ε ~ N(0, σ² I): an outcome spills into its neighbours' (a price that raises the
//   prices about it);
// - error: y = X β + u,  u = λ W u + ε,     ε ~ N(0, σ² I): what is left out is shared by neighbours (an unmeasured amenity).
//
// THE LIKELIHOOD. Concentrated in the spatial parameter, for the lag model (Anselin 1988, §12.1):
//   e(ρ) = e₀ − ρ e_L, with e₀, e_L the residuals of y and of W y on X;  σ̂²(ρ) = e(ρ)ᵀe(ρ)/n;
//   ln L(ρ) = −n/2 (ln 2π + 1) − n/2 ln σ̂²(ρ) + ln|I − ρW|,
// and for the error model, with y* = (I − λW) y and X* = (I − λW) X, β̂(λ) by least squares of y* on X*, σ̂²(λ) its mean square,
//   ln L(λ) = −n/2 (ln 2π + 1) − n/2 ln σ̂²(λ) + ln|I − λW|.
// The Jacobian ln|I − ρW| = Σᵢ ln|1 − ρ ωᵢ| over the eigenvalues ωᵢ of W (Ord 1975), computed once (calx's eigenvalues, complex
// where W is not symmetric: conjugate pairs give real products). The parameter is found by Brent's method over the interval where
// I − ρW stays non-singular, (1/ω_min, 1/ω_max) for the real extreme eigenvalues, within (−1, 1).
//
// THE COVARIANCE. The inverse of the information matrix (Anselin 1988, eq. 6.5 for the lag, 6.12 for the error), with
// A = I − ρW and B = W A⁻¹ (C = W (I − λW)⁻¹ for the error):
//   lag:  I_ββ = XᵀX/σ², I_βρ = Xᵀ(BXβ)/σ², I_ρρ = tr(B²) + tr(BᵀB) + (BXβ)ᵀ(BXβ)/σ², I_ρσ² = tr(B)/σ², I_σ²σ² = n/(2σ⁴), I_βσ² = 0;
//   error: I_ββ = X*ᵀX*/σ², I_λλ = tr(C²) + tr(CᵀC), I_λσ² = tr(C)/σ², I_σ²σ² = n/(2σ⁴), β apart from (λ, σ²).
// The whole matrix is inverted. B and C are dense (n × n): the models are for n of a few thousand at most.
//
// THE TESTS (Anselin 1988, §8.1; robust forms, Anselin, Bera, Florax and Yoon 1996, Reg. Sci. Urban Econ. 26, 77–104). From the
// least-squares residuals e, σ̂² = eᵀe/n, T = tr(WᵀW + W²), and nJ = (WXβ̂)ᵀM(WXβ̂)/σ̂² + T with M = I − X(XᵀX)⁻¹Xᵀ:
//   LM_error = (eᵀWe/σ̂²)² / T,  LM_lag = (eᵀWy/σ̂²)² / nJ,
//   robust LM_error = (eᵀWe/σ̂² − T eᵀWy/(σ̂² nJ))² / (T − T²/nJ),  robust LM_lag = (eᵀWy/σ̂² − eᵀWe/σ̂²)² / (nJ − T),
// each χ² with one degree of freedom under its null, and SARMA = (eᵀWy/σ̂² − eᵀWe/σ̂²)²/(nJ − T) + (eᵀWe/σ̂²)²/T with two.
//
// THE IMPACTS of the lag model (LeSage and Pace 2009, Introduction to Spatial Econometrics, §2.7): a change in xₖ everywhere moves y
// by S_k = (I − ρW)⁻¹ βₖ; the average direct impact is tr(S_k)/n, the average total impact the mean row sum of S_k (βₖ/(1 − ρ) for
// row-standardised W), and the indirect impact, what spills over, their difference.

#pragma once

#include <tyche/probability/distributions.hpp>
#include <tyche/regression/linear.hpp>
#include <tyche/spatial/autocorrelation.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigenvalues.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/optimisation/minimise.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::spatial {

struct SpatialRegression {
  std::vector<double> beta, se, z, p;       // per column of X
  double parameter = 0;                     // ρ (lag) or λ (error)
  double parameter_se = 0, parameter_z = 0, parameter_p = 1;
  double sigma2 = 0;                        // σ̂² = eᵀe/n
  double log_likelihood = 0, aic = 0;       // AIC = −2 ln L + 2 (k + 2): β, the parameter and σ²
  std::vector<double> residuals;            // ε̂: y − ρWy − Xβ̂ (lag); (I − λW)(y − Xβ̂) (error)
  std::vector<double> fitted;               // the reduced form (I − ρW)⁻¹ Xβ̂ (lag); Xβ̂ (error)
  calx::linalg::Matrix covariance;          // of (β, parameter, σ²), (k + 2) × (k + 2)
  std::size_t n = 0, k = 0;
  double lower = 0, upper = 0;              // the interval searched for the parameter
};

struct LagrangeTests {
  double lm_error = 0, lm_lag = 0, robust_error = 0, robust_lag = 0, sarma = 0;
  double p_error = 1, p_lag = 1, p_robust_error = 1, p_robust_lag = 1, p_sarma = 1;
};

struct Impacts {
  std::vector<double> direct, indirect, total;  // per column of X
};

namespace detail_regression {

inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::spatial (regression): ") + what);
}

inline calx::linalg::Matrix dense(const SpatialWeights& W) {
  calx::linalg::Matrix M(W.size());
  for (std::size_t i = 0; i < W.size(); i++)
    for (std::size_t q = W.row_start[i]; q < W.row_start[i + 1]; q++) M(i, W.column[q]) += W.weight[q];
  return M;
}

inline std::vector<double> lag(const SpatialWeights& W, std::span<const double> v) {
  std::vector<double> out(W.size(), 0.0);
  for (std::size_t i = 0; i < W.size(); i++)
    for (std::size_t q = W.row_start[i]; q < W.row_start[i + 1]; q++) out[i] += W.weight[q] * v[W.column[q]];
  return out;
}

inline calx::linalg::Rectangular lag_columns(const SpatialWeights& W, const calx::linalg::Rectangular& X) {
  calx::linalg::Rectangular out(X.rows, X.cols);
  for (std::size_t i = 0; i < W.size(); i++)
    for (std::size_t q = W.row_start[i]; q < W.row_start[i + 1]; q++)
      for (std::size_t j = 0; j < X.cols; j++) out(i, j) += W.weight[q] * X(W.column[q], j);
  return out;
}

// The eigenvalues of W, the interval of non-singularity within (−1, 1), and ln|I − ρW|.
struct Jacobian {
  std::vector<std::complex<double>> omega;
  double lower = -1, upper = 1;
  double operator()(double rho) const {
    double s = 0;
    for (const auto& w : omega) s += std::log(std::abs(1.0 - rho * w));
    return s;
  }
};

inline Jacobian jacobian(const calx::linalg::Matrix& Wd) {
  Jacobian J;
  J.omega = calx::linalg::eigenvalues(Wd);
  double lo = 0, hi = 0;
  for (const auto& w : J.omega) {
    if (std::fabs(w.imag()) > 1e-9 * (1 + std::abs(w))) continue;
    lo = std::min(lo, w.real()), hi = std::max(hi, w.real());
  }
  const double eps = 1e-9;
  J.lower = lo < 0 ? std::max(-1.0, 1.0 / lo) + eps : -1.0 + eps;
  J.upper = hi > 0 ? std::min(1.0, 1.0 / hi) - eps : 1.0 - eps;
  return J;
}

inline double trace(const calx::linalg::Matrix& M) {
  double t = 0;
  for (std::size_t i = 0; i < M.n; i++) t += M(i, i);
  return t;
}

inline calx::linalg::Matrix transpose(const calx::linalg::Matrix& M) {
  calx::linalg::Matrix T(M.n);
  for (std::size_t i = 0; i < M.n; i++)
    for (std::size_t j = 0; j < M.n; j++) T(j, i) = M(i, j);
  return T;
}

inline double trace_product(const calx::linalg::Matrix& A, const calx::linalg::Matrix& B) {  // tr(AB)
  double t = 0;
  for (std::size_t i = 0; i < A.n; i++)
    for (std::size_t j = 0; j < A.n; j++) t += A(i, j) * B(j, i);
  return t;
}

inline double trace_gram(const calx::linalg::Matrix& A) {  // tr(AᵀA)
  double t = 0;
  for (double v : A.a) t += v * v;
  return t;
}

inline double two_sided(double z) { return std::erfc(std::fabs(z) / std::numbers::sqrt2); }

inline void finish(SpatialRegression& f) {
  const std::size_t m = f.k + 2;
  calx::linalg::Matrix inv = calx::linalg::solve(f.covariance, calx::linalg::Matrix::identity(m));
  f.covariance = inv;
  f.se.resize(f.k), f.z.resize(f.k), f.p.resize(f.k);
  for (std::size_t j = 0; j < f.k; j++) {
    f.se[j] = std::sqrt(std::max(0.0, inv(j, j)));
    f.z[j] = f.beta[j] / f.se[j];
    f.p[j] = two_sided(f.z[j]);
  }
  f.parameter_se = std::sqrt(std::max(0.0, inv(f.k, f.k)));
  f.parameter_z = f.parameter / f.parameter_se;
  f.parameter_p = two_sided(f.parameter_z);
  f.aic = -2 * f.log_likelihood + 2.0 * static_cast<double>(m);
}

}  // namespace detail_regression

// The spatial lag model by maximum likelihood.
inline SpatialRegression spatial_lag(const calx::linalg::Rectangular& X, std::span<const double> y, const SpatialWeights& W, double tolerance = 1e-10) {
  namespace d = detail_regression;
  const std::size_t n = X.rows, k = X.cols;
  d::require(y.size() == n && W.size() == n, "y, X and W of one size");
  d::require(n > k + 2, "more observations than coefficients");
  const auto Wd = d::dense(W);
  const auto J = d::jacobian(Wd);
  const auto Wy = d::lag(W, y);
  const auto f0 = regression::ols(X, y), fL = regression::ols(X, Wy);
  const auto& e0 = f0.residuals;
  const auto& eL = fL.residuals;
  const double nd = static_cast<double>(n);
  const auto concentrated = [&](double rho) {
    double s = 0;
    for (std::size_t i = 0; i < n; i++) s += (e0[i] - rho * eL[i]) * (e0[i] - rho * eL[i]);
    return -0.5 * nd * (std::log(2 * std::numbers::pi) + 1) - 0.5 * nd * std::log(s / nd) + J(rho);
  };
  const auto m = calx::optimisation::brent_maximise(concentrated, J.lower, J.upper, tolerance);
  SpatialRegression f;
  f.n = n, f.k = k, f.lower = J.lower, f.upper = J.upper;
  f.parameter = m.x;
  f.log_likelihood = concentrated(m.x);
  f.beta.resize(k);
  for (std::size_t j = 0; j < k; j++) f.beta[j] = f0.beta[j] - m.x * fL.beta[j];
  f.residuals.resize(n);
  double s = 0;
  for (std::size_t i = 0; i < n; i++) f.residuals[i] = e0[i] - m.x * eL[i], s += f.residuals[i] * f.residuals[i];
  f.sigma2 = s / nd;
  // A⁻¹ and B = W A⁻¹; the reduced form A⁻¹Xβ.
  calx::linalg::Matrix A = calx::linalg::Matrix::identity(n);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) A(i, j) -= m.x * Wd(i, j);
  const auto Ai = calx::linalg::solve(A, calx::linalg::Matrix::identity(n));
  const auto B = Wd * Ai;
  std::vector<double> xb(n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < k; j++) xb[i] += X(i, j) * f.beta[j];
  f.fitted.assign(n, 0.0);
  std::vector<double> bxb(n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) f.fitted[i] += Ai(i, j) * xb[j], bxb[i] += B(i, j) * xb[j];
  // The information matrix of (β, ρ, σ²).
  const double s2 = f.sigma2;
  calx::linalg::Matrix I(k + 2);
  for (std::size_t a = 0; a < k; a++) {
    for (std::size_t b = 0; b < k; b++) {
      double v = 0;
      for (std::size_t i = 0; i < n; i++) v += X(i, a) * X(i, b);
      I(a, b) = v / s2;
    }
    double v = 0;
    for (std::size_t i = 0; i < n; i++) v += X(i, a) * bxb[i];
    I(a, k) = I(k, a) = v / s2;
  }
  double bb = 0;
  for (double v : bxb) bb += v * v;
  I(k, k) = d::trace_product(B, B) + d::trace_gram(B) + bb / s2;
  I(k, k + 1) = I(k + 1, k) = d::trace(B) / s2;
  I(k + 1, k + 1) = nd / (2 * s2 * s2);
  f.covariance = I;
  d::finish(f);
  return f;
}

// The spatial error model by maximum likelihood.
inline SpatialRegression spatial_error(const calx::linalg::Rectangular& X, std::span<const double> y, const SpatialWeights& W, double tolerance = 1e-10) {
  namespace d = detail_regression;
  const std::size_t n = X.rows, k = X.cols;
  d::require(y.size() == n && W.size() == n, "y, X and W of one size");
  d::require(n > k + 2, "more observations than coefficients");
  const auto Wd = d::dense(W);
  const auto J = d::jacobian(Wd);
  const auto Wy = d::lag(W, y);
  const auto WX = d::lag_columns(W, X);
  const double nd = static_cast<double>(n);
  const auto filtered = [&](double lambda, calx::linalg::Rectangular& Xs, std::vector<double>& ys) {
    Xs = calx::linalg::Rectangular(n, k);
    ys.assign(n, 0.0);
    for (std::size_t i = 0; i < n; i++) {
      ys[i] = y[i] - lambda * Wy[i];
      for (std::size_t j = 0; j < k; j++) Xs(i, j) = X(i, j) - lambda * WX(i, j);
    }
  };
  const auto concentrated = [&](double lambda) {
    calx::linalg::Rectangular Xs;
    std::vector<double> ys;
    filtered(lambda, Xs, ys);
    const auto fit = regression::ols(Xs, ys);
    return -0.5 * nd * (std::log(2 * std::numbers::pi) + 1) - 0.5 * nd * std::log(fit.rss / nd) + J(lambda);
  };
  const auto m = calx::optimisation::brent_maximise(concentrated, J.lower, J.upper, tolerance);
  SpatialRegression f;
  f.n = n, f.k = k, f.lower = J.lower, f.upper = J.upper;
  f.parameter = m.x;
  f.log_likelihood = concentrated(m.x);
  calx::linalg::Rectangular Xs;
  std::vector<double> ys;
  filtered(m.x, Xs, ys);
  const auto fit = regression::ols(Xs, ys);
  f.beta = fit.beta;
  f.residuals = fit.residuals;
  f.sigma2 = fit.rss / nd;
  f.fitted.assign(n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < k; j++) f.fitted[i] += X(i, j) * f.beta[j];
  calx::linalg::Matrix A = calx::linalg::Matrix::identity(n);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) A(i, j) -= m.x * Wd(i, j);
  const auto C = Wd * calx::linalg::solve(A, calx::linalg::Matrix::identity(n));
  const double s2 = f.sigma2;
  calx::linalg::Matrix I(k + 2);
  for (std::size_t a = 0; a < k; a++)
    for (std::size_t b = 0; b < k; b++) {
      double v = 0;
      for (std::size_t i = 0; i < n; i++) v += Xs(i, a) * Xs(i, b);
      I(a, b) = v / s2;
    }
  I(k, k) = d::trace_product(C, C) + d::trace_gram(C);
  I(k, k + 1) = I(k + 1, k) = d::trace(C) / s2;
  I(k + 1, k + 1) = nd / (2 * s2 * s2);
  f.covariance = I;
  d::finish(f);
  return f;
}

// The Lagrange multiplier tests on the least-squares residuals of y on X.
inline LagrangeTests lagrange_multipliers(const calx::linalg::Rectangular& X, std::span<const double> y, const SpatialWeights& W) {
  namespace d = detail_regression;
  const std::size_t n = X.rows, k = X.cols;
  d::require(y.size() == n && W.size() == n, "y, X and W of one size");
  const auto fit = regression::ols(X, y);
  const auto& e = fit.residuals;
  const double nd = static_cast<double>(n), s2 = fit.rss / nd;
  const auto We = d::lag(W, e), Wy = d::lag(W, y);
  double eWe = 0, eWy = 0;
  for (std::size_t i = 0; i < n; i++) eWe += e[i] * We[i], eWy += e[i] * Wy[i];
  const auto Wd = d::dense(W);
  const double T = d::trace_gram(Wd) + d::trace_product(Wd, Wd);
  // (WXβ̂)ᵀ M (WXβ̂): the residual sum of squares of WXβ̂ on X.
  std::vector<double> xb(n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < k; j++) xb[i] += X(i, j) * fit.beta[j];
  const auto wxb = d::lag(W, xb);
  const double nJ = regression::ols(X, wxb).rss / s2 + T;
  const double a = eWe / s2, b = eWy / s2;
  LagrangeTests t;
  t.lm_error = a * a / T;
  t.lm_lag = b * b / nJ;
  t.robust_error = (a - T * b / nJ) * (a - T * b / nJ) / (T - T * T / nJ);
  t.robust_lag = (b - a) * (b - a) / (nJ - T);
  t.sarma = t.robust_lag + t.lm_error;
  const probability::ChiSquared one{1}, two{2};
  t.p_error = one.survival(t.lm_error), t.p_lag = one.survival(t.lm_lag), t.p_robust_error = one.survival(t.robust_error), t.p_robust_lag = one.survival(t.robust_lag);
  t.p_sarma = two.survival(t.sarma);
  return t;
}

// The average direct, indirect and total impacts of each column of X in a fitted lag model.
inline Impacts lag_impacts(const SpatialRegression& f, const SpatialWeights& W) {
  namespace d = detail_regression;
  const std::size_t n = W.size();
  d::require(n == f.n, "the weights of the fit");
  const auto Wd = d::dense(W);
  calx::linalg::Matrix A = calx::linalg::Matrix::identity(n);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) A(i, j) -= f.parameter * Wd(i, j);
  const auto Ai = calx::linalg::solve(A, calx::linalg::Matrix::identity(n));
  double tr = 0, sum = 0;
  for (std::size_t i = 0; i < n; i++) {
    tr += Ai(i, i);
    for (std::size_t j = 0; j < n; j++) sum += Ai(i, j);
  }
  Impacts im;
  const double nd = static_cast<double>(n);
  for (double b : f.beta) {
    im.direct.push_back(b * tr / nd);
    im.total.push_back(b * sum / nd);
    im.indirect.push_back(im.total.back() - im.direct.back());
  }
  return im;
}

}  // namespace tyche::spatial
