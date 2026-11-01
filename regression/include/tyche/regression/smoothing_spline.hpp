// The cubic smoothing spline: the curve g minimising Σ wᵢ (yᵢ − g(tᵢ))² + λ ∫ g″(t)² dt.
//
// THE SOLUTION (Reinsch 1967, Numer. Math. 10, 177–183; Green and Silverman 1994, Nonparametric
// Regression and Generalized Linear Models, Chapman and Hall, §2.1–2.3). For distinct knots
// t₁ < … < t_n the minimiser over all twice-differentiable g is the natural cubic spline with
// knots at the tᵢ, determined by its values g = (g(t₁), …, g(t_n)) and its second derivatives γ at
// the inner knots (γ₁ = γ_n = 0): with hᵢ = t_{i+1} − tᵢ, the n × (n − 2) band matrix Q
// (q_{i−1,i} = 1/h_{i−1}, q_{ii} = −1/h_{i−1} − 1/hᵢ, q_{i+1,i} = 1/hᵢ) and the (n − 2) × (n − 2)
// tridiagonal R (r_ii = (h_{i−1} + hᵢ)/3, r_{i,i+1} = r_{i+1,i} = hᵢ/6), a natural spline has
// Qᵀg = Rγ and ∫ g″² = γᵀRγ = gᵀKg, K = QR⁻¹Qᵀ (their Theorem 2.1). The minimiser is then given by
// Reinsch's equations (their §2.3.3, weighted):
//   (R + λ QᵀW⁻¹Q) γ = Qᵀy,   g = y − λ W⁻¹Q γ,
// which stay well conditioned as λ → ∞, unlike g = (W + λK)⁻¹Wy. Between knots g is the cubic of
// their eq. 2.5; beyond the end knots it is linear.
//
// THE AMOUNT OF SMOOTHING. The fit is the linear smoother S_λ y, S_λ = I − λ W⁻¹Q(R + λQᵀW⁻¹Q)⁻¹Qᵀ,
// whose trace, n − λ tr((R + λM)⁻¹M) with M = QᵀW⁻¹Q, is its equivalent degrees of freedom (Green and
// Silverman §3.3.2; Hastie and Tibshirani 1990, Generalized Additive Models, §3.5): 2 as λ → ∞ (the
// weighted least-squares line, which the penalty does not see) and n as λ → 0 (interpolation).
// smoothing_spline(t, y, df) finds the λ whose trace is df by Brent's root of tr S_λ − df in log λ,
// on which it is monotone.
//
// *Remark (R's smooth.spline).* With fewer than 50 distinct x, R's smooth.spline places a knot at
// each and minimises over cubic B-splines; the minimiser would be this natural spline, but R's Gram
// matrix of the penalty (its Fortran sgram) weights the product of the two slopes of the piecewise
// linear second derivatives by 0.3330 instead of 1/3, so its fits differ from the exact spline by
// about 2·10⁻⁴ relative at the same degrees of freedom (the checks hold this code to an exact
// B-spline solution to 10⁻⁹, and to R to that difference; with 0.333 the B-spline solution is R's to
// 4·10⁻¹¹). R also matches df only to the tolerance of its search in `spar`.
//
// COMPLEXITY. Dense, O(n³): it serves the short grids it is used on (the λ grid of Storey's π₀,
// n = 19); a banded O(n) form is for long series, not yet needed.
// LIMITS. Knots distinct and sorted, weights positive, n ≥ 3.

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/solvers/roots.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::regression {

struct SmoothingSpline {
  std::vector<double> knots;     // t₁ < … < t_n
  std::vector<double> fitted;    // g(tᵢ)
  std::vector<double> second;    // g″(tᵢ), zero at the ends
  std::vector<double> leverage;  // the diagonal of S_λ
  double lambda = 0;             // the penalty's weight, on the knots' scale
  double df = 0;                 // tr S_λ

  // g(x): the natural cubic spline between the knots (Green and Silverman eq. 2.5), linear beyond.
  double operator()(double x) const {
    const std::size_t n = knots.size();
    if (x <= knots[0]) {
      const double h = knots[1] - knots[0];
      const double slope = (fitted[1] - fitted[0]) / h - h * second[1] / 6;
      return fitted[0] + slope * (x - knots[0]);
    }
    if (x >= knots[n - 1]) {
      const double h = knots[n - 1] - knots[n - 2];
      const double slope = (fitted[n - 1] - fitted[n - 2]) / h + h * second[n - 2] / 6;
      return fitted[n - 1] + slope * (x - knots[n - 1]);
    }
    const std::size_t i = static_cast<std::size_t>(std::upper_bound(knots.begin(), knots.end(), x) - knots.begin()) - 1;
    const double h = knots[i + 1] - knots[i], a = x - knots[i], b = knots[i + 1] - x;
    return (a * fitted[i + 1] + b * fitted[i]) / h - a * b / 6 * ((1 + a / h) * second[i + 1] + (1 + b / h) * second[i]);
  }
};

namespace detail {

// Q (n × (n − 2), row-major), R and M = QᵀW⁻¹Q for the knots and weights.
struct SplineSystem {
  std::size_t n = 0, m = 0;
  std::vector<double> q, w;
  calx::linalg::Matrix R, M;
};

inline SplineSystem spline_system(std::span<const double> t, std::span<const double> w) {
  SplineSystem s;
  s.n = t.size();
  s.m = s.n - 2;
  s.w.assign(w.begin(), w.end());
  const std::size_t n = s.n, m = s.m;
  s.q.assign(n * m, 0.0);
  s.R = calx::linalg::Matrix(m);
  for (std::size_t c = 0; c < m; c++) {
    const double h0 = t[c + 1] - t[c], h1 = t[c + 2] - t[c + 1];
    s.q[c * m + c] = 1 / h0;
    s.q[(c + 1) * m + c] = -1 / h0 - 1 / h1;
    s.q[(c + 2) * m + c] = 1 / h1;
    s.R(c, c) = (h0 + h1) / 3;
    if (c + 1 < m) s.R(c, c + 1) = s.R(c + 1, c) = h1 / 6;
  }
  s.M = calx::linalg::Matrix(m);
  for (std::size_t a = 0; a < m; a++)
    for (std::size_t b = 0; b < m; b++) {
      double v = 0;
      for (std::size_t i = 0; i < n; i++) v += s.q[i * m + a] * s.q[i * m + b] / w[i];
      s.M(a, b) = v;
    }
  return s;
}

inline calx::linalg::Matrix reinsch_matrix(const SplineSystem& s, double lambda) {
  calx::linalg::Matrix A = s.R;
  for (std::size_t i = 0; i < s.m * s.m; i++) A.a[i] += lambda * s.M.a[i];
  return A;
}

// tr S_λ = n − λ tr((R + λM)⁻¹ M).
inline double degrees_of_freedom(const SplineSystem& s, double lambda) {
  const auto X = calx::linalg::solve(reinsch_matrix(s, lambda), s.M);
  double t = 0;
  for (std::size_t i = 0; i < s.m; i++) t += X(i, i);
  return static_cast<double>(s.n) - lambda * t;
}

inline std::vector<double> unit_weights(std::span<const double> weights, std::size_t n) {
  std::vector<double> w(n, 1.0);
  if (!weights.empty()) {
    if (weights.size() != n) throw std::invalid_argument("tyche::regression::smoothing_spline: one weight per knot");
    for (std::size_t i = 0; i < n; i++) {
      if (!(weights[i] > 0)) throw std::invalid_argument("tyche::regression::smoothing_spline: positive weights");
      w[i] = weights[i];
    }
  }
  return w;
}

inline void require_knots(std::span<const double> t, std::size_t ny) {
  if (t.size() < 3 || ny != t.size()) throw std::invalid_argument("tyche::regression::smoothing_spline: at least three knots, one y each");
  for (std::size_t i = 1; i < t.size(); i++)
    if (!(t[i] > t[i - 1])) throw std::invalid_argument("tyche::regression::smoothing_spline: distinct knots in increasing order");
}

}  // namespace detail

// The spline with a given penalty λ ≥ 0 (weights empty: all 1).
inline SmoothingSpline smoothing_spline_lambda(std::span<const double> t, std::span<const double> y, double lambda, std::span<const double> weights = {}) {
  detail::require_knots(t, y.size());
  if (!(lambda >= 0)) throw std::invalid_argument("tyche::regression::smoothing_spline: λ ≥ 0");
  const std::size_t n = t.size();
  const auto w = detail::unit_weights(weights, n);
  const auto sys = detail::spline_system(t, w);
  const std::size_t m = sys.m;
  const auto A = detail::reinsch_matrix(sys, lambda);
  // γ = A⁻¹ Qᵀ y; g = y − λ W⁻¹ Q γ.
  calx::linalg::Matrix rhs(m);
  for (std::size_t c = 0; c < m; c++) {
    double v = 0;
    for (std::size_t i = 0; i < n; i++) v += sys.q[i * m + c] * y[i];
    rhs(c, 0) = v;
  }
  const auto gamma = calx::linalg::solve(A, rhs);
  SmoothingSpline s;
  s.knots.assign(t.begin(), t.end());
  s.lambda = lambda;
  s.second.assign(n, 0.0);
  for (std::size_t c = 0; c < m; c++) s.second[c + 1] = gamma(c, 0);
  s.fitted.resize(n);
  for (std::size_t i = 0; i < n; i++) {
    double v = 0;
    for (std::size_t c = 0; c < m; c++) v += sys.q[i * m + c] * gamma(c, 0);
    s.fitted[i] = y[i] - lambda * v / w[i];
  }
  // The leverages: S_ii = 1 − (λ/wᵢ) (Q A⁻¹ Qᵀ)_ii.
  const auto Ainv = calx::linalg::solve(A, calx::linalg::Matrix::identity(m));
  s.leverage.resize(n);
  s.df = 0;
  for (std::size_t i = 0; i < n; i++) {
    double v = 0;
    for (std::size_t a = 0; a < m; a++)
      for (std::size_t b = 0; b < m; b++) v += sys.q[i * m + a] * Ainv(a, b) * sys.q[i * m + b];
    s.leverage[i] = 1 - lambda * v / w[i];
    s.df += s.leverage[i];
  }
  return s;
}

// The spline whose equivalent degrees of freedom tr S_λ equal df, 2 < df < n.
inline SmoothingSpline smoothing_spline(std::span<const double> t, std::span<const double> y, double df, std::span<const double> weights = {}) {
  detail::require_knots(t, y.size());
  const std::size_t n = t.size();
  if (!(df > 2 && df < static_cast<double>(n))) throw std::invalid_argument("tyche::regression::smoothing_spline: 2 < df < n");
  const auto w = detail::unit_weights(weights, n);
  const auto sys = detail::spline_system(t, w);
  // λ on the scale tr R / tr M, so that log λ = 0 is a moderate smoothing whatever the units.
  double tr = 0, tm = 0;
  for (std::size_t i = 0; i < sys.m; i++) {
    tr += sys.R(i, i);
    tm += sys.M(i, i);
  }
  const double scale = tm > 0 ? tr / tm : 1.0;
  const auto excess = [&](double log_lambda) { return detail::degrees_of_freedom(sys, scale * std::exp(log_lambda)) - df; };
  double lo = -30, hi = 30;
  while (excess(lo) < 0 && lo > -300) lo -= 30;  // still too smooth: rougher
  while (excess(hi) > 0 && hi < 300) hi += 30;   // still too rough: smoother
  const auto root = calx::solvers::brent(excess, lo, hi, 1e-13);
  return smoothing_spline_lambda(t, y, scale * std::exp(root.x), weights);
}

}  // namespace tyche::regression
