// Principal coordinates: points in a space whose distances are as near as can be to given ones.
//
// Gower (1966, Biometrika 53: 325–338) places n objects with dissimilarities d_ij at points whose
// Euclidean distances reproduce them, when that can be done: with A_ij = −½ d_ij² and J = I − 11ᵀ/n,
// the centred matrix G = J A J is the Gram matrix of the points (centred at their mean) exactly when
// d is Euclidean, and its eigendecomposition G = V Λ Vᵀ gives the principal coordinates
// x_k = v_k √λ_k, axis k explaining λ_k of the total Σ λ. For Euclidean distances of points X this
// recovers X centred, up to an orthogonal transformation (the classical scaling of Torgerson 1952,
// Psychometrika 17: 401–419, and Gower 1966).
//
// A dissimilarity that is not Euclidean (Bray–Curtis, Sørensen) gives G negative eigenvalues; the
// points then live in a pseudo-Euclidean space, their positive axes real and their negative ones
// imaginary. The negative eigenvalues are reported (`negative`, their sum). Two corrections add a
// constant to the off-diagonal dissimilarities so that none is negative (Legendre and Legendre
// 2012, Numerical Ecology, §9.3.4; Legendre and Anderson 1999, Ecol. Monogr. 69: 1–24):
//
//   Lingoes (1971, Psychometrika 36: 195–203)   d̃_ij = √(d_ij² + 2c), c = −λ_min(G): G gains c on
//                                                every direction orthogonal to 1, so λ̃_min = 0.
//   Cailliez (1983, Psychometrika 48: 305–308)   d̃_ij = d_ij + c, c the largest real eigenvalue of
//                                                [[0, 2 G(d²)], [−I, −4 G(d)]], G(dᵖ) = J(−½dᵖ)J: the
//                                                least constant that makes d + c Euclidean.
//
// Either is applied only when G has a negative eigenvalue. Eigenvalues within 10⁻¹⁰ of the largest
// in magnitude are taken as zero.
//
// Accuracy: calx's symmetric eigenproblem (Householder and QL), backward stable, O(n³); Cailliez's
// constant from calx's general eigenvalues (balancing, Hessenberg, Francis QR) of a 2n × 2n matrix.

#pragma once

#include <tyche/multivariate/dissimilarity.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigen.hpp>
#include <calx/linalg/eigenvalues.hpp>
#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace tyche::multivariate {

enum class Correction { none, lingoes, cailliez };

struct PCoA {
  std::vector<double> eigenvalues;        // of G, descending, negative ones included
  calx::linalg::Rectangular coordinates;  // n × k, the principal coordinates of the k positive eigenvalues
  std::vector<double> explained;          // λ_k / Σ_{λ>0} λ for those k axes
  double negative = 0;                    // the sum of the negative eigenvalues (0 for Euclidean d)
  Correction correction = Correction::none;
  double constant = 0;                    // Lingoes's or Cailliez's constant, 0 when not applied
  calx::linalg::Matrix distances;         // the dissimilarities ordinated, corrected when a correction applied
};

// G = J(−½ d^power)J, Gower's centred matrix.
inline calx::linalg::Matrix gower_centred(const calx::linalg::Matrix& d, double power = 2) {
  const std::size_t n = d.n;
  calx::linalg::Matrix g(n);
  std::vector<double> row(n, 0.0);
  double all = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) {
      const double a = -0.5 * (power == 2 ? d(i, j) * d(i, j) : power == 1 ? d(i, j) : std::pow(d(i, j), power));
      g(i, j) = a;
      row[i] += a / static_cast<double>(n);
      all += a / static_cast<double>(n * n);
    }
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) g(i, j) += all - row[i] - row[j];
  return g;
}

namespace detail {

inline PCoA ordinate(const calx::linalg::Matrix& d) {
  const std::size_t n = d.n;
  PCoA r;
  r.distances = d;
  const auto e = calx::linalg::symmetric_eigen(gower_centred(d));
  double scale = 0;
  for (double v : e.values) scale = std::max(scale, std::fabs(v));
  const double tol = 1e-10 * scale;
  std::vector<std::size_t> positive;
  double total = 0;
  for (std::size_t k = n; k-- > 0;) {
    const double v = e.values[k];
    r.eigenvalues.push_back(std::fabs(v) <= tol ? 0.0 : v);
    if (v > tol) positive.push_back(k), total += v;
    if (v < -tol) r.negative += v;
  }
  r.coordinates = calx::linalg::Rectangular(n, positive.size());
  for (std::size_t c = 0; c < positive.size(); c++) {
    const std::size_t k = positive[c];
    const double s = std::sqrt(e.values[k]);
    // A fixed sign: the largest entry of each axis positive.
    double big = 0;
    for (std::size_t i = 0; i < n; i++)
      if (std::fabs(e.vectors(i, k)) > std::fabs(big)) big = e.vectors(i, k);
    const double sign = big < 0 ? -1 : 1;
    for (std::size_t i = 0; i < n; i++) r.coordinates(i, c) = sign * s * e.vectors(i, k);
    r.explained.push_back(e.values[k] / total);
  }
  return r;
}

}  // namespace detail

// Cailliez's additive constant: the largest real eigenvalue of [[0, 2G(d²)], [−I, −4G(d)]].
inline double cailliez_constant(const calx::linalg::Matrix& d) {
  const std::size_t n = d.n;
  const auto g2 = gower_centred(d, 2), g1 = gower_centred(d, 1);
  calx::linalg::Matrix m(2 * n);
  for (std::size_t i = 0; i < n; i++) {
    m(n + i, i) = -1;
    for (std::size_t j = 0; j < n; j++) {
      m(i, n + j) = 2 * g2(i, j);
      m(n + i, n + j) = -4 * g1(i, j);
    }
  }
  const auto ev = calx::linalg::eigenvalues(m);
  double best = -INFINITY, scale = 0;
  for (const auto& v : ev) scale = std::max(scale, std::abs(v));
  for (const auto& v : ev)
    if (std::fabs(v.imag()) <= 1e-8 * scale) best = std::max(best, v.real());
  if (!std::isfinite(best))
    for (const auto& v : ev) best = std::max(best, v.real());
  return best;
}

// The principal coordinates of a matrix of dissimilarities.
inline PCoA pcoa(const calx::linalg::Matrix& d, Correction correction = Correction::none) {
  require_dissimilarities(d, "pcoa");
  detail::check(d.n >= 2, "pcoa", "at least two objects");
  PCoA r = detail::ordinate(d);
  if (correction == Correction::none || r.negative == 0) return r;
  calx::linalg::Matrix c(d.n);
  double k = 0;
  if (correction == Correction::lingoes) {
    k = -r.eigenvalues.back();
    for (std::size_t i = 0; i < d.n; i++)
      for (std::size_t j = 0; j < d.n; j++) c(i, j) = i == j ? 0 : std::sqrt(d(i, j) * d(i, j) + 2 * k);
  } else {
    k = cailliez_constant(d);
    for (std::size_t i = 0; i < d.n; i++)
      for (std::size_t j = 0; j < d.n; j++) c(i, j) = i == j ? 0 : d(i, j) + k;
  }
  PCoA out = detail::ordinate(c);
  out.correction = correction;
  out.constant = k;
  return out;
}

}  // namespace tyche::multivariate
