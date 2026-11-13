// A sample of compositions described: its centre, its variation matrix and its total variance.
//
// A sample is an n × D table, a row for each composition (each in any units and need not be closed;
// every statistic here is invariant to the closure of each row).
//
// THE CENTRE (Aitchison 1986, The Statistical Analysis of Compositional Data; Pawlowsky-Glahn, Egozcue
// and Tolosana-Delgado 2015, Modelling and Analysis of Compositional Data) is the
// closure of the parts' geometric means, ĝ = C(g_1, …, g_D), g_j = (Π_i x_ij)^{1/n}. It is the
// Fréchet mean of the sample under d_a (clr ĝ = the mean of the rows' clr's) and is equivariant under
// perturbation and powering: cen(a ⊙ X ⊕ y) = a ⊙ cen(X) ⊕ y. The arithmetic mean of closed rows is
// not, and is not computed here.
//
// THE VARIATION MATRIX (Aitchison 1986) holds t_ij = var(ln(x_i / x_j)), the variances of the
// log-ratios of every pair of parts: symmetric, zero on the diagonal, and t_ij = 0 exactly when parts i
// and j are proportional throughout the sample.
//
// THE TOTAL VARIANCE (Pawlowsky-Glahn, Egozcue and Tolosana-Delgado 2015) is, with divisor
// n − ddof throughout,
//
//   totvar = (1/(2D)) Σ_i Σ_j t_ij = Σ_i var(clr_i) = (1/(n − ddof)) Σ_k d_a²(x_k, ĝ).
//
// Proof of the identities: t_ij = v_i + v_j − 2 c_ij for the clr variances v and covariances c, and
// Σ_ij c_ij = var(Σ_i clr_i) = 0, so Σ_ij t_ij = 2D Σ_i v_i; and Σ_i v_i is the mean squared Euclidean
// distance of the clr rows from their mean, clr ĝ. Equivalently totvar = (D/2) times the mean of the D²
// entries of the variation matrix.
//
// THE METRIC STANDARD DEVIATION √(totvar/(D − 1)) spreads the total variance equally over the D − 1
// dimensions of the simplex (Pawlowsky-Glahn and Egozcue 2001, Stoch. Environ. Res. Risk Assess. 15:
// 384–398; the `msd` of R's package compositions, van den Boogaart and Tolosana-Delgado 2013, Analyzing
// Compositional Data with R).
//
// DIVISOR. n − ddof, ddof = 1 by default (tyche's convention for variances, docs/glossary.md §1);
// Pawlowsky-Glahn et al. write the total variance with divisor n (ddof = 0).
//
// ACCURACY. Log-ratios are centred before they are squared, so the variances lose no digits to
// cancellation beyond those of the logarithms themselves.

#pragma once

#include <tyche/compositions/simplex.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace tyche::compositions {

using calx::linalg::Matrix;
using calx::linalg::Rectangular;

namespace detail {

// The logarithms of a sample's parts, checked positive (n × D).
inline Rectangular log_sample(const Rectangular& X, const char* where) {
  require(X.rows >= 1 && X.cols >= 2, where, "a sample of at least one composition of two or more parts");
  Rectangular L(X.rows, X.cols);
  for (std::size_t i = 0; i < X.rows; i++) {
    const auto row = X.row(i);
    require_positive(row, where);
    for (std::size_t j = 0; j < X.cols; j++) L(i, j) = std::log(X(i, j));
  }
  return L;
}

// The clr of each row.
inline Rectangular clr_sample(const Rectangular& X, const char* where) {
  auto L = log_sample(X, where);
  for (std::size_t i = 0; i < L.rows; i++) {
    double m = 0;
    for (std::size_t j = 0; j < L.cols; j++) m += L(i, j);
    m /= static_cast<double>(L.cols);
    for (std::size_t j = 0; j < L.cols; j++) L(i, j) -= m;
  }
  return L;
}

inline double divisor(std::size_t n, double ddof, const char* where) {
  require(ddof >= 0 && static_cast<double>(n) > ddof, where, "more compositions than ddof");
  return static_cast<double>(n) - ddof;
}

}  // namespace detail

// The centre ĝ = C(g_1, …, g_D), closed to `total`.
inline std::vector<double> centre(const Rectangular& X, double total = 1) {
  const auto L = detail::log_sample(X, "centre");
  std::vector<double> m(X.cols, 0.0);
  for (std::size_t i = 0; i < L.rows; i++)
    for (std::size_t j = 0; j < L.cols; j++) m[j] += L(i, j);
  for (double& v : m) v /= static_cast<double>(L.rows);
  return clr_inverse(m, total);
}

// Each row perturbed by the inverse of the centre, x ⊖ ĝ, closed to `total`: the sample moved so that
// its centre is the neutral element.
inline Rectangular centred(const Rectangular& X, double total = 1) {
  const auto g = centre(X, 1);
  Rectangular Y(X.rows, X.cols);
  for (std::size_t i = 0; i < X.rows; i++) {
    const auto row = X.row(i);
    const auto y = difference(row, g, total);
    for (std::size_t j = 0; j < X.cols; j++) Y(i, j) = y[j];
  }
  return Y;
}

// The variation matrix T, t_ij = var(ln(x_i/x_j)) with divisor n − ddof.
inline Matrix variation_matrix(const Rectangular& X, double ddof = 1) {
  const auto L = detail::log_sample(X, "variation_matrix");
  const double d = detail::divisor(L.rows, ddof, "variation_matrix");
  const std::size_t D = L.cols, n = L.rows;
  Matrix T(D);
  std::vector<double> r(n);
  for (std::size_t a = 0; a < D; a++)
    for (std::size_t b = a + 1; b < D; b++) {
      double m = 0;
      for (std::size_t i = 0; i < n; i++) m += r[i] = L(i, a) - L(i, b);
      m /= static_cast<double>(n);
      double s = 0;
      for (std::size_t i = 0; i < n; i++) s += (r[i] - m) * (r[i] - m);
      T(a, b) = T(b, a) = s / d;
    }
  return T;
}

// The covariance matrix of the clr coordinates (D × D, singular: its rows sum to zero).
inline Matrix clr_covariance(const Rectangular& X, double ddof = 1) {
  auto Z = detail::clr_sample(X, "clr_covariance");
  const double d = detail::divisor(Z.rows, ddof, "clr_covariance");
  const std::size_t D = Z.cols, n = Z.rows;
  std::vector<double> m(D, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < D; j++) m[j] += Z(i, j) / static_cast<double>(n);
  Matrix C(D);
  for (std::size_t a = 0; a < D; a++)
    for (std::size_t b = a; b < D; b++) {
      double s = 0;
      for (std::size_t i = 0; i < n; i++) s += (Z(i, a) - m[a]) * (Z(i, b) - m[b]);
      C(a, b) = C(b, a) = s / d;
    }
  return C;
}

// totvar = (1/(2D)) Σ_ij t_ij.
inline double total_variance(const Rectangular& X, double ddof = 1) {
  const auto T = variation_matrix(X, ddof);
  double s = 0;
  for (std::size_t a = 0; a < T.n; a++)
    for (std::size_t b = 0; b < T.n; b++) s += T(a, b);
  return s / (2 * static_cast<double>(T.n));
}

// √(totvar/(D − 1)).
inline double metric_standard_deviation(const Rectangular& X, double ddof = 1) {
  return std::sqrt(total_variance(X, ddof) / static_cast<double>(X.cols - 1));
}

struct Summary {
  std::size_t n = 0, parts = 0;
  double ddof = 1;
  std::vector<double> centre;         // ĝ, closed to the total asked
  Matrix variation;                   // T, D × D
  std::vector<double> clr_variances;  // v_i = var(clr_i)
  double total_variance = 0;          // Σ v_i = (1/(2D)) Σ t_ij
  double metric_standard_deviation = 0;
};

inline Summary describe(const Rectangular& X, double ddof = 1, double total = 1) {
  Summary s;
  s.n = X.rows;
  s.parts = X.cols;
  s.ddof = ddof;
  s.centre = centre(X, total);
  s.variation = variation_matrix(X, ddof);
  const auto C = clr_covariance(X, ddof);
  for (std::size_t j = 0; j < C.n; j++) s.clr_variances.push_back(C(j, j));
  double t = 0;
  for (std::size_t a = 0; a < s.variation.n; a++)
    for (std::size_t b = 0; b < s.variation.n; b++) t += s.variation(a, b);
  s.total_variance = t / (2 * static_cast<double>(s.parts));
  s.metric_standard_deviation = std::sqrt(s.total_variance / static_cast<double>(s.parts - 1));
  return s;
}

}  // namespace tyche::compositions
