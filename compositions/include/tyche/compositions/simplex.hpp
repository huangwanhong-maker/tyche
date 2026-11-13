// The simplex and its geometry: compositions as vectors of a Euclidean space (Aitchison's geometry).
//
// A COMPOSITION of D parts is a vector x of positive reals whose information lies in the ratios
// x_i/x_j alone (Aitchison 1986, The Statistical Analysis of Compositional Data): x and λx,
// λ > 0, are the same composition. Its representative is the closure
//
//   C(x) = c · (x_1, …, x_D) / Σ_i x_i
//
// to a stated total c: 1 for proportions, 100 for percentages, 1440 for the minutes of a day. The
// closed vectors form the simplex S^D.
//
// AITCHISON'S GEOMETRY makes S^D a real vector space of dimension D − 1 with an inner product
// (Aitchison 1986; Pawlowsky-Glahn and Egozcue 2001, Stoch. Environ. Res. Risk Assess. 15: 384–398;
// Pawlowsky-Glahn, Egozcue and Tolosana-Delgado 2015, Modelling and Analysis of Compositional Data):
//
//   perturbation    x ⊕ y = C(x_1 y_1, …, x_D y_D)          the group operation, neutral n = C(1, …, 1),
//                                                           inverse ⊖x = C(1/x_1, …, 1/x_D)
//   powering        a ⊙ x = C(x_1^a, …, x_D^a)               the multiplication by a real a
//   inner product   ⟨x, y⟩_a = (1/(2D)) Σ_i Σ_j ln(x_i/x_j) ln(y_i/y_j) = Σ_i clr_i(x) clr_i(y)
//   norm, distance  ‖x‖_a = √⟨x, x⟩_a,   d_a(x, y) = ‖x ⊖ y‖_a,   x ⊖ y = x ⊕ (⊖y)
//
// The CENTRED LOG-RATIO clr(x)_i = ln(x_i / g(x)), g(x) = (Π x_j)^{1/D} the geometric mean (Aitchison
// 1986), is an isometric isomorphism of S^D onto the hyperplane {z : Σ z_i = 0} of R^D:
// clr(x ⊕ y) = clr x + clr y, clr(a ⊙ x) = a clr x, ⟨x, y⟩_a = clr(x) · clr(y). Its inverse is
// clr⁻¹(z) = C(exp z), defined for any z ∈ R^D (adding a constant to z changes nothing). The second
// form of the inner product equals the first because Σ_i Σ_j (u_i − u_j)(v_i − v_j) = 2D Σ u_i v_i
// − 2 (Σ u_i)(Σ v_i) for u = ln x, v = ln y, and the clr removes the means.
//
// The ADDITIVE LOG-RATIO alr(x)_i = ln(x_i / x_k), i ≠ k (Aitchison 1986), is an isomorphism
// onto R^{D−1} but not an isometry; it is given for exchange with other software. Orthonormal
// coordinates (ilr) are in balances.hpp.
//
// CONVENTIONS. Every composition returned is closed to `total` (default 1). The operations are
// invariant to the closure of their arguments, so inputs may be given in any units (minutes, hours,
// shares) and need not be closed. Logarithms are natural. Parts must be positive and finite: a zero
// has no log-ratio, and is refused with the reason (replace rounded zeros first, zeros.hpp; leave an
// essential zero out of the subcomposition).
//
// ACCURACY. Double precision throughout; clr⁻¹ subtracts the largest entry before exponentiating, so
// it neither overflows nor underflows for |z| up to about 700 around the largest entry.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::compositions {

inline void require(bool ok, const char* where, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::compositions::") + where + ": " + what);
}

namespace detail {

inline void require_positive(std::span<const double> x, const char* where) {
  require(x.size() >= 2, where, "a composition has at least two parts");
  for (double v : x) {
    require(std::isfinite(v), where, "parts are finite");
    require(v > 0, where,
            "parts are positive: a zero part has no log-ratio; replace rounded zeros first (replace_zeros), and leave a part that is "
            "truly absent (an essential zero) out of the subcomposition");
  }
}

inline void require_total(double total, const char* where) { require(std::isfinite(total) && total > 0, where, "the total is positive and finite"); }

// Σ ln x_i / D.
inline double mean_log(std::span<const double> x) {
  double s = 0;
  for (double v : x) s += std::log(v);
  return s / static_cast<double>(x.size());
}

// C(exp z) without overflow.
inline std::vector<double> close_exp(std::span<const double> z, double total) {
  const double top = *std::max_element(z.begin(), z.end());
  std::vector<double> x(z.size());
  double s = 0;
  for (std::size_t i = 0; i < z.size(); i++) s += x[i] = std::exp(z[i] - top);
  for (double& v : x) v *= total / s;
  return x;
}

}  // namespace detail

// C(x) to `total`. Zeros are allowed here (closure is defined on the closed orthant); the sum must be
// positive.
inline std::vector<double> closure(std::span<const double> x, double total = 1) {
  detail::require_total(total, "closure");
  require(!x.empty(), "closure", "at least one part");
  double s = 0;
  for (double v : x) {
    require(std::isfinite(v) && v >= 0, "closure", "parts are non-negative and finite");
    s += v;
  }
  require(s > 0, "closure", "the parts do not all vanish");
  std::vector<double> c(x.size());
  for (std::size_t i = 0; i < x.size(); i++) c[i] = x[i] * (total / s);
  return c;
}

// The neutral element n = C(1, …, 1) of D parts.
inline std::vector<double> neutral(std::size_t parts, double total = 1) {
  require(parts >= 2, "neutral", "at least two parts");
  detail::require_total(total, "neutral");
  return std::vector<double>(parts, total / static_cast<double>(parts));
}

// x ⊕ y = C(x_i y_i).
inline std::vector<double> perturb(std::span<const double> x, std::span<const double> y, double total = 1) {
  detail::require_positive(x, "perturb");
  detail::require_positive(y, "perturb");
  require(x.size() == y.size(), "perturb", "two compositions of the same parts");
  std::vector<double> z(x.size());
  const double lx = detail::mean_log(x), ly = detail::mean_log(y);
  for (std::size_t i = 0; i < x.size(); i++) z[i] = std::log(x[i]) - lx + std::log(y[i]) - ly;
  return detail::close_exp(z, total);
}

// ⊖x = C(1/x_i), the inverse of x under perturbation.
inline std::vector<double> perturb_inverse(std::span<const double> x, double total = 1) {
  detail::require_positive(x, "perturb_inverse");
  std::vector<double> z(x.size());
  for (std::size_t i = 0; i < x.size(); i++) z[i] = -std::log(x[i]);
  return detail::close_exp(z, total);
}

// x ⊖ y = C(x_i / y_i).
inline std::vector<double> difference(std::span<const double> x, std::span<const double> y, double total = 1) {
  detail::require_positive(x, "difference");
  detail::require_positive(y, "difference");
  require(x.size() == y.size(), "difference", "two compositions of the same parts");
  std::vector<double> z(x.size());
  for (std::size_t i = 0; i < x.size(); i++) z[i] = std::log(x[i]) - std::log(y[i]);
  return detail::close_exp(z, total);
}

// a ⊙ x = C(x_i^a).
inline std::vector<double> power(std::span<const double> x, double a, double total = 1) {
  detail::require_positive(x, "power");
  require(std::isfinite(a), "power", "a finite real power");
  std::vector<double> z(x.size());
  const double lx = detail::mean_log(x);
  for (std::size_t i = 0; i < x.size(); i++) z[i] = a * (std::log(x[i]) - lx);
  return detail::close_exp(z, total);
}

// The geometric mean g(x) = exp(Σ ln x_i / D), in x's units.
inline double geometric_mean(std::span<const double> x) {
  detail::require_positive(x, "geometric_mean");
  return std::exp(detail::mean_log(x));
}

// clr(x)_i = ln(x_i / g(x)); Σ_i clr(x)_i = 0 up to rounding.
inline std::vector<double> clr(std::span<const double> x) {
  detail::require_positive(x, "clr");
  const double m = detail::mean_log(x);
  std::vector<double> z(x.size());
  for (std::size_t i = 0; i < x.size(); i++) z[i] = std::log(x[i]) - m;
  return z;
}

// clr⁻¹(z) = C(exp z), for any z ∈ R^D.
inline std::vector<double> clr_inverse(std::span<const double> z, double total = 1) {
  require(z.size() >= 2, "clr_inverse", "at least two coordinates");
  detail::require_total(total, "clr_inverse");
  for (double v : z) require(std::isfinite(v), "clr_inverse", "finite coordinates");
  return detail::close_exp(z, total);
}

// alr(x) = (ln(x_i / x_k))_{i ≠ k}, the parts in order with the denominator k left out; k = D − 1 by
// default (the last part, as Aitchison writes it).
inline std::vector<double> alr(std::span<const double> x, std::size_t denominator = std::numeric_limits<std::size_t>::max()) {
  detail::require_positive(x, "alr");
  const std::size_t k = denominator == std::numeric_limits<std::size_t>::max() ? x.size() - 1 : denominator;
  require(k < x.size(), "alr", "the denominator is one of the parts");
  std::vector<double> y;
  const double lk = std::log(x[k]);
  for (std::size_t i = 0; i < x.size(); i++)
    if (i != k) y.push_back(std::log(x[i]) - lk);
  return y;
}

// alr⁻¹(y): the composition of D = y.size() + 1 parts whose alr with denominator k is y.
inline std::vector<double> alr_inverse(std::span<const double> y, std::size_t denominator = std::numeric_limits<std::size_t>::max(), double total = 1) {
  require(!y.empty(), "alr_inverse", "at least one coordinate");
  const std::size_t D = y.size() + 1;
  const std::size_t k = denominator == std::numeric_limits<std::size_t>::max() ? D - 1 : denominator;
  require(k < D, "alr_inverse", "the denominator is one of the parts");
  detail::require_total(total, "alr_inverse");
  std::vector<double> z(D, 0.0);
  for (std::size_t i = 0, j = 0; i < D; i++)
    if (i != k) {
      require(std::isfinite(y[j]), "alr_inverse", "finite coordinates");
      z[i] = y[j++];
    }
  return detail::close_exp(z, total);
}

// ⟨x, y⟩_a = Σ_i clr_i(x) clr_i(y).
inline double inner(std::span<const double> x, std::span<const double> y) {
  require(x.size() == y.size(), "inner", "two compositions of the same parts");
  const auto u = clr(x), v = clr(y);
  double s = 0;
  for (std::size_t i = 0; i < u.size(); i++) s += u[i] * v[i];
  return s;
}

// ‖x‖_a.
inline double norm(std::span<const double> x) {
  const auto u = clr(x);
  double s = 0;
  for (double v : u) s += v * v;
  return std::sqrt(s);
}

// d_a(x, y) = ‖clr x − clr y‖.
inline double distance(std::span<const double> x, std::span<const double> y) {
  require(x.size() == y.size(), "distance", "two compositions of the same parts");
  const auto u = clr(x), v = clr(y);
  double s = 0;
  for (std::size_t i = 0; i < u.size(); i++) s += (u[i] - v[i]) * (u[i] - v[i]);
  return std::sqrt(s);
}

}  // namespace tyche::compositions
