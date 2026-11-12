// The partition of diversity among units: gamma = alpha × beta.
//
// N units (plots, blocks, sites, years) each hold some of the classes: a table z, a row for each unit
// and a column for each class, of abundances z_ij ≥ 0 (or each row a law). Unit j has the law
// p_ij = z_ij / z_j+ and a weight w_j (Σ w_j = 1); the pooled law is p̄_i = Σ_j w_j p_ij, and the gamma
// diversity of order q is its Hill number ^qD(p̄). The alpha diversity is the effective number of
// classes in a typical unit, and beta = gamma / alpha the effective number of wholly distinct units:
// 1 when every unit holds the same law, N when N units of equal weight share no class.
//
// Jost (2007, Ecology 88: 2427–2439, §"Partitioning Hill numbers" and his Appendix) defines, for q ≠ 1,
//
//   ^qD_α = (Σ_j w_j^q Σ_i p_ij^q / Σ_j w_j^q)^{1/(1−q)},   and   ^1D_α = exp(Σ_j w_j H(p_j)),
//
// H the Shannon entropy. He proves that with this alpha, gamma ≥ alpha and beta is free to range over
// [1, N] whatever alpha (alpha and beta independent) for every q when the weights are equal, and with
// unequal weights only for q = 1. So:
//
//   `Weighting::equal`   w_j = 1/N, Jost's alpha, any q ≥ 0 or +∞;
//   given weights        Jost's alpha with those weights: any q if they are equal, otherwise q = 1
//                        only, where beta = exp(I), I the mutual information between unit and class in
//                        the joint law w_j p_ij (so beta ∈ [1, e^{H(w)}]); other q are refused;
//   `Weighting::size`    w_j = z_j+ / z_++, the units weighted by their totals, with the alpha that
//                        Chao, Chiu and Jost (2014, Annu. Rev. Ecol. Evol. Syst. 45: 297–324, §"Beta
//                        diversity", the alpha for units of unequal size) propose for every q:
//                          ^qD_α = (1/N) ^qD(c),   c_ij = z_ij / z_++,
//                        the Hill number of the table's cells over N. It equals Jost's alpha when the
//                        totals are equal; for q = 1 it differs from Jost's with unequal weights by the
//                        factor e^{H(w)}/N. Its beta is 1 exactly when the units hold the same absolute
//                        abundances: units of one law at sizes w have ^qD(c) = ^qD(w) ^qD(p), so
//                        beta = N / ^qD(w), which counts their difference in size as a difference.
//
// In the equal and size-weighted cases beta ∈ [1, N] for every q: with c_ij = w_j p_ij,
// Σ_j c_ij^q ≤ (Σ_j c_ij)^q ≤ N^{q−1} Σ_j c_ij^q for q ≥ 1 (reversed for q < 1), so
// beta = N (Σ_i p̄_i^q / Σ_ij c_ij^q)^{1/(1−q)} ∈ [1, N]; at q = ∞ beta = N max c_ij / max p̄_i.
//
// From beta, Chao, Chiu and Jost (2014, their table of the four classes of similarity measures, after
// Jost 2006 and Chiu, Jost and Chao 2014, Ecol. Monogr. 84: 21–44) normalise to overlaps in [0, 1]:
//
//   Sørensen-type   C_qN = [(1/β)^{q−1} − (1/N)^{q−1}] / [1 − (1/N)^{q−1}],
//   Jaccard-type    U_qN = [(1/β)^{1−q} − (1/N)^{1−q}] / [1 − (1/N)^{1−q}],
//   both at q = 1   1 − ln β / ln N (Horn's overlap for N = 2),
//
// 1 when the units are alike and 0 when they share nothing; C_qN = U_{(2−q)N}. For N = 2 and q = 0 they
// are Sørensen's and Jaccard's indices of the classes present, for q = 2 C_22 is Morisita and Horn's
// index. Their complements 1 − C_qN and 1 − U_qN are the dissimilarities (1 − C_0N = (β − 1)/(N − 1),
// Whittaker's normalised turnover).

#pragma once

#include <tyche/classes/diversity.hpp>

#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace tyche::classes {

enum class Weighting { equal, size };

struct Overlap {
  double sorensen = 0;                // C_qN
  double jaccard = 0;                 // U_qN
  double sorensen_dissimilarity = 0;  // 1 − C_qN
  double jaccard_dissimilarity = 0;   // 1 − U_qN
};

// The overlaps of N ≥ 2 units with the given beta, for a finite order q ≥ 0.
inline Overlap overlap(double beta, std::size_t units, double q) {
  require(units >= 2, "overlap", "at least two units");
  require(q >= 0 && std::isfinite(q), "overlap", "a finite order q ≥ 0");
  const double N = static_cast<double>(units);
  require(beta >= 1 - 1e-12 && beta <= N * (1 + 1e-12), "overlap", "beta lies in [1, N]");
  const double lb = std::log(std::clamp(beta, 1.0, N)), ln = std::log(N);
  Overlap o;
  const double r = q - 1;
  if (r == 0) {
    o.sorensen = o.jaccard = 1 - lb / ln;
  } else {
    // [e^{−r ln β} − e^{−r ln N}] / [1 − e^{−r ln N}], by expm1 so that q near 1 loses nothing.
    o.sorensen = (std::expm1(-r * lb) - std::expm1(-r * ln)) / (-std::expm1(-r * ln));
    o.jaccard = (std::expm1(r * lb) - std::expm1(r * ln)) / (-std::expm1(r * ln));
  }
  o.sorensen_dissimilarity = 1 - o.sorensen;
  o.jaccard_dissimilarity = 1 - o.jaccard;
  return o;
}

struct Partition {
  double q = 0;
  Weighting weighting = Weighting::equal;  // `equal` also for given weights that are equal
  bool given_weights = false;
  std::size_t units = 0;
  std::vector<double> weights;         // w_j
  std::vector<double> pooled;          // p̄_i
  std::vector<double> unit_diversity;  // ^qD(p_j) of each unit's own law
  double gamma = 0, alpha = 0, beta = 0;
  Overlap overlap;  // from beta, N and q; NaN at q = ∞ or N = 1
};

namespace detail {

struct Table {
  std::size_t N = 0, S = 0;
  std::vector<double> rows;  // z_j+
  double total = 0;
};

inline Table read_table(const calx::linalg::Rectangular& z, const char* where) {
  Table t;
  t.N = z.rows;
  t.S = z.cols;
  require(t.N >= 1 && t.S >= 1, where, "a table of units by classes");
  t.rows.assign(t.N, 0.0);
  for (std::size_t j = 0; j < t.N; j++) {
    for (std::size_t i = 0; i < t.S; i++) {
      const double v = z(j, i);
      require(std::isfinite(v) && v >= 0, where, "abundances are non-negative");
      t.rows[j] += v;
    }
    require(t.rows[j] > 0, where, "every unit holds something");
    t.total += t.rows[j];
  }
  return t;
}

inline void finish(Partition& r) {
  r.beta = r.gamma / r.alpha;
  if (r.units >= 2 && std::isfinite(r.q)) {
    r.overlap = overlap(std::clamp(r.beta, 1.0, static_cast<double>(r.units)), r.units, r.q);
  } else {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    r.overlap = {nan, nan, nan, nan};
  }
}

// Jost's alpha with weights w on the rows' laws.
inline Partition jost(const calx::linalg::Rectangular& z, double q, std::vector<double> w, bool equal, bool given, const char* where) {
  require(q >= 0, where, "the order q is not negative");
  const auto t = read_table(z, where);
  Partition r;
  r.q = q;
  r.weighting = Weighting::equal;
  r.given_weights = given;
  r.units = t.N;
  r.weights = w;
  r.pooled.assign(t.S, 0.0);
  std::vector<double> p(t.N * t.S);
  for (std::size_t j = 0; j < t.N; j++)
    for (std::size_t i = 0; i < t.S; i++) {
      p[j * t.S + i] = z(j, i) / t.rows[j];
      r.pooled[i] += w[j] * p[j * t.S + i];
    }
  for (std::size_t j = 0; j < t.N; j++) r.unit_diversity.push_back(hill(std::span<const double>(p.data() + j * t.S, t.S), q));
  r.gamma = hill(r.pooled, q);
  if (q == 1) {
    double h = 0;  // Σ_j w_j H(p_j)
    for (std::size_t j = 0; j < t.N; j++) h += w[j] * std::log(r.unit_diversity[j]);
    r.alpha = std::exp(h);
  } else {
    require(equal, where, "with unequal weights Jost's alpha is admissible only for q = 1 (Jost 2007); weight the units equally, or by size (Weighting::size)");
    if (std::isinf(q)) {
      r.alpha = 1 / *std::max_element(p.begin(), p.end());
    } else {
      // ((1/N) Σ_j Σ_i p_ij^q)^{1/(1−q)}: the power mean of the p_ij under the weights p_ij / N.
      std::vector<double> c(p.size());
      for (std::size_t k = 0; k < p.size(); k++) c[k] = p[k] / static_cast<double>(t.N);
      r.alpha = std::exp(log_power_mean(c, p, q) / (1 - q));
    }
  }
  finish(r);
  return r;
}

}  // namespace detail

// The partition of the diversity of order q among the units (rows) of a table of abundances.
inline Partition partition(const calx::linalg::Rectangular& z, double q, Weighting weighting = Weighting::equal) {
  require(q >= 0, "partition", "the order q is not negative");
  if (weighting == Weighting::equal) return detail::jost(z, q, std::vector<double>(z.rows, 1.0 / static_cast<double>(z.rows)), true, false, "partition");
  const auto t = detail::read_table(z, "partition");
  Partition r;
  r.q = q;
  r.weighting = Weighting::size;
  r.units = t.N;
  r.pooled.assign(t.S, 0.0);
  std::vector<double> c(t.N * t.S);
  for (std::size_t j = 0; j < t.N; j++) {
    r.weights.push_back(t.rows[j] / t.total);
    std::vector<double> pj(t.S);
    for (std::size_t i = 0; i < t.S; i++) {
      c[j * t.S + i] = z(j, i) / t.total;
      r.pooled[i] += c[j * t.S + i];
      pj[i] = z(j, i) / t.rows[j];
    }
    r.unit_diversity.push_back(hill(pj, q));
  }
  r.gamma = hill(r.pooled, q);
  r.alpha = hill(c, q) / static_cast<double>(t.N);
  detail::finish(r);
  return r;
}

// Jost's partition with given unit weights (positive, renormalised to sum to one).
inline Partition partition(const calx::linalg::Rectangular& z, double q, std::span<const double> weights) {
  require(weights.size() == z.rows, "partition", "a weight for each unit");
  double total = 0;
  for (double w : weights) {
    require(std::isfinite(w) && w > 0, "partition", "weights are positive");
    total += w;
  }
  std::vector<double> w(weights.begin(), weights.end());
  bool equal = true;
  for (double& x : w) {
    x /= total;
    if (std::fabs(x * static_cast<double>(w.size()) - 1) > 1e-12) equal = false;
  }
  return detail::jost(z, q, w, equal, true, "partition");
}

}  // namespace tyche::classes
