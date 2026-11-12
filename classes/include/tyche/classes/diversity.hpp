// The diversity of a law: how many equally common classes it is worth.
//
// For a law p on S classes (p_i ≥ 0, Σ p_i = 1), the Hill number of order q ≥ 0 is
//
//   ^qD(p) = (Σ_{p_i>0} p_i^q)^{1/(1−q)},   q ≠ 1,
//   ^1D(p) = exp(−Σ p_i ln p_i)               (the limit q → 1: the exponential of Shannon's entropy),
//   ^∞D(p) = 1 / max p_i                      (the limit q → ∞: the reciprocal of the largest share),
//
// so that ^0D is the number of classes of positive probability, ^2D = 1/Σ p_i² the reciprocal of
// Simpson's concentration (Hill 1973, Ecology 54: 427–432, the numbers N_a; Jost 2006, Oikos 113:
// 363–375). A law of D equally probable classes has ^qD = D for every q, which is what makes ^qD an
// effective number of classes. ^qD is non-increasing in q, strictly unless p is uniform on its support
// (Hill 1973; a consequence of the monotonicity of power means). The Rényi entropy of order q is
// H_q = ln ^qD (Rényi 1961, Proc. 4th Berkeley Symp. Math. Statist. Probab. 1: 547–561, the entropy of
// order α).
//
// When classes resemble one another, a law spread over similar classes is less diverse than one spread
// over different ones. With a similarity matrix Z (Z_ij ∈ [0, 1], Z_ii = 1, the similarity of class j
// to class i), (Zp)_i = Σ_j Z_ij p_j is the ordinariness of class i, and Leinster and Cobbold's
// similarity-sensitive diversity (2012, Ecology 93: 477–489, the definition of ^qD^Z and its limits
// at q = 1 and q = ∞) is
//
//   ^qD^Z(p) = (Σ_{p_i>0} p_i (Zp)_i^{q−1})^{1/(1−q)},   ^1D^Z = exp(−Σ p_i ln (Zp)_i),
//   ^∞D^Z = 1 / max_{p_i>0} (Zp)_i.
//
// With Z = I it is the Hill number; with Z_ij = 1 within groups and 0 between, the Hill number of the
// groups' law; with Z all ones, 1. It is non-increasing in q (Leinster and Cobbold 2012). A
// similarity from distances d_ij is often taken as Z_ij = e^{−u d_ij} (their suggestion).
//
// Accuracy. The sums are formed so that no order loses precision: near q = 1 as
// log1p(Σ p_i expm1((q − 1) ln x_i)) / (1 − q), which tends to the q = 1 limit without cancellation,
// and elsewhere scaled by the largest (q > 1) or smallest (q < 1) term, so that no power under- or
// overflows. A law is accepted when its entries sum to one within 10⁻⁹ and is renormalised.

#pragma once

#include <tyche/classes/counts.hpp>

#include <calx/linalg/dense.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace tyche::classes {

namespace detail {

// p renormalised to sum to one; refused unless it is a law within 10⁻⁹.
inline std::vector<double> law(std::span<const double> p, const char* where) {
  double total = 0;
  for (double x : p) {
    require(std::isfinite(x) && x >= 0, where, "a law has non-negative entries");
    total += x;
  }
  require(std::fabs(total - 1) <= 1e-9, where, "a law sums to one");
  std::vector<double> r(p.begin(), p.end());
  for (double& x : r) x /= total;
  return r;
}

// ln Σ_{w_i>0} w_i x_i^{q−1} for weights w (a law) and positive x on the support of w, with
// Σ w = 1; near q = 1 by expm1 and log1p, elsewhere scaled by the extreme x.
inline double log_power_mean(std::span<const double> w, std::span<const double> x, double q) {
  const double r = q - 1;
  if (std::fabs(r) < 0.25) {
    double s = 0, c = 0;  // Neumaier's compensated sum of w_i expm1(r ln x_i) = Σ w x^r − 1
    for (std::size_t i = 0; i < w.size(); i++) {
      if (!(w[i] > 0)) continue;
      const double t = w[i] * std::expm1(r * std::log(x[i]));
      const double u = s + t;
      c += std::fabs(s) >= std::fabs(t) ? (s - u) + t : (t - u) + s;
      s = u;
    }
    return std::log1p(s + c);
  }
  double ref = r > 0 ? 0.0 : std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < w.size(); i++)
    if (w[i] > 0) ref = r > 0 ? std::max(ref, x[i]) : std::min(ref, x[i]);
  double s = 0;
  for (std::size_t i = 0; i < w.size(); i++)
    if (w[i] > 0) s += w[i] * std::pow(x[i] / ref, r);
  return r * std::log(ref) + std::log(s);
}

}  // namespace detail

// The Rényi entropy H_q = ln ^qD of a law, in nats; q ≥ 0 or +∞.
inline double renyi(std::span<const double> p, double q) {
  require(q >= 0, "renyi", "the order q is not negative");
  const auto w = detail::law(p, "renyi");
  if (q == 0) {
    double s = 0;
    for (double x : w) s += x > 0 ? 1 : 0;
    return std::log(s);
  }
  if (std::isinf(q)) return -std::log(*std::max_element(w.begin(), w.end()));
  if (q == 1) {
    double h = 0;
    for (double x : w)
      if (x > 0) h -= x * std::log(x);
    return h;
  }
  return detail::log_power_mean(w, w, q) / (1 - q);
}

// The Hill number ^qD of a law; q ≥ 0 or +∞.
inline double hill(std::span<const double> p, double q) {
  if (q == 0) {  // exact: a count
    detail::law(p, "hill");
    double s = 0;
    for (double x : p) s += x > 0 ? 1 : 0;
    return s;
  }
  return std::exp(renyi(p, q));
}

// ^qD for each order in `orders`: the diversity profile.
inline std::vector<double> hill_profile(std::span<const double> p, std::span<const double> orders) {
  std::vector<double> r;
  r.reserve(orders.size());
  for (double q : orders) r.push_back(hill(p, q));
  return r;
}

// The law of a sample of counts: X_i / n (zeros kept as zeros).
inline std::vector<double> law_of(std::span<const double> counts) {
  double n = 0;
  for (double x : counts) {
    require(std::isfinite(x) && x >= 0, "law_of", "counts are non-negative");
    n += x;
  }
  require(n > 0, "law_of", "some count is positive");
  std::vector<double> p(counts.begin(), counts.end());
  for (double& x : p) x /= n;
  return p;
}

// Z_ij = e^{−u d_ij} from a symmetric matrix of distances (d_ii = 0) and a scale u > 0.
inline calx::linalg::Matrix similarity_from_distances(const calx::linalg::Matrix& d, double u) {
  require(u > 0, "similarity_from_distances", "the scale u is positive");
  calx::linalg::Matrix z(d.n);
  for (std::size_t i = 0; i < d.n; i++)
    for (std::size_t j = 0; j < d.n; j++) {
      require(std::isfinite(d(i, j)) && d(i, j) >= 0, "similarity_from_distances", "distances are non-negative");
      z(i, j) = std::exp(-u * d(i, j));
    }
  return z;
}

// The ordinariness (Zp)_i of each class.
inline std::vector<double> ordinariness(std::span<const double> p, const calx::linalg::Matrix& z) {
  require(z.n == p.size(), "ordinariness", "a similarity for each pair of classes");
  std::vector<double> zp(p.size(), 0.0);
  for (std::size_t i = 0; i < z.n; i++)
    for (std::size_t j = 0; j < z.n; j++) zp[i] += z(i, j) * p[j];
  return zp;
}

// Leinster and Cobbold's ^qD^Z(p); q ≥ 0 or +∞.
inline double similarity_diversity(std::span<const double> p, const calx::linalg::Matrix& z, double q) {
  require(q >= 0, "similarity_diversity", "the order q is not negative");
  const auto w = detail::law(p, "similarity_diversity");
  require(z.n == w.size(), "similarity_diversity", "a similarity for each pair of classes");
  for (std::size_t i = 0; i < z.n; i++)
    for (std::size_t j = 0; j < z.n; j++) {
      const double v = z(i, j);
      require(std::isfinite(v) && v >= 0 && v <= 1, "similarity_diversity", "similarities lie in [0, 1]");
      if (i == j) require(v == 1, "similarity_diversity", "a class is wholly similar to itself (Z_ii = 1)");
    }
  const auto zp = ordinariness(w, z);
  if (std::isinf(q)) {
    double m = 0;
    for (std::size_t i = 0; i < w.size(); i++)
      if (w[i] > 0) m = std::max(m, zp[i]);
    return 1 / m;
  }
  if (q == 1) {
    double h = 0;
    for (std::size_t i = 0; i < w.size(); i++)
      if (w[i] > 0) h -= w[i] * std::log(zp[i]);
    return std::exp(h);
  }
  return std::exp(detail::log_power_mean(w, zp, q) / (1 - q));
}

inline std::vector<double> similarity_profile(std::span<const double> p, const calx::linalg::Matrix& z, std::span<const double> orders) {
  std::vector<double> r;
  r.reserve(orders.size());
  for (double q : orders) r.push_back(similarity_diversity(p, z, q));
  return r;
}

}  // namespace tyche::classes
