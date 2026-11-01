// Multiple comparisons: the family-wise error rate (Bonferroni, Holm), the false discovery rate
// (Benjamini and Hochberg) and Storey's q-values with their estimate of the proportion of true nulls.
//
// - Bonferroni: min(1, m p). Holm (1979, Scand. J. Statist. 6, 65–70): the step-down
//   p̃_(i) = max_{j ≤ i} min(1, (m − j + 1) p_(j)). Both control the family-wise error rate under any
//   dependence.
// - Benjamini and Hochberg (1995, JRSS B 57, 289–300): the step-up p̃_(i) = min_{j ≥ i} min(1, m p_(j)/j),
//   rejecting where p̃ ≤ q controls the false discovery rate at q π₀ ≤ q for independent (and
//   positively regression-dependent, Benjamini and Yekutieli 2001) p-values. R's p.adjust("BH").
// - Storey's q-values (Storey 2002, JRSS B 64, 479–498; Storey and Tibshirani 2003, PNAS 100,
//   9440–9445): q_(i) = π̂₀ min_{j ≥ i} min(1, m p_(j)/j), the BH adjustment scaled by an estimate of
//   the proportion π₀ of true nulls, q(p) estimating the positive FDR of the rule "reject below p".
//   π̂₀(λ) = #{p ≥ λ}/(m(1 − λ)) (the uniform density of null p-values above λ; Storey 2002, eq. 6,
//   with ≥ as R's qvalue counts), and λ is chosen by one of
//   · fixed: one λ (0.5 is Storey and Tibshirani's example);
//   · smoother: a cubic smoothing spline with 3 degrees of freedom through π̂₀(λ) on the grid
//     λ = 0.05, 0.10, …, 0.95, read at the largest λ (Storey and Tibshirani 2003, Remark B; the default
//     of R's qvalue::pi0est, which this follows: π̂₀ = min(ĝ(λ_max), 1));
//   · bootstrap: the λ minimising the bootstrap estimate of the mean squared error of π̂₀(λ)
//     (Storey, Taylor and Siegmund 2004, JRSS B 66, 187–205, §6), in the closed form of R's qvalue
//     (D. Robinson's): MSE(λ) = W(λ)(1 − W(λ)/m)/(m²(1 − λ)²) + (π̂₀(λ) − q₁₀)², W(λ) = #{p ≥ λ},
//     q₁₀ the 10% quantile (type 7) of π̂₀ over the grid.
//   q-values are R's qvalue(p)$qvalues to the precision of its spline (checks); the local FDR is
//   not computed.
//
// Complexity: a sort, O(m log m); the smoother is a dense spline on the 19 grid points.
// Limits: p-values in [0, 1]; NaN p-values are refused (filter them first).

#pragma once

#include <tyche/regression/smoothing_spline.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::testing {

namespace detail {

inline void require_p_values(std::span<const double> p, const char* who) {
  for (const double x : p)
    if (!(x >= 0 && x <= 1)) throw std::invalid_argument(std::string(who) + ": p-values in [0, 1]");
}

// The indices of p in increasing order of p (stable).
inline std::vector<std::size_t> order(std::span<const double> p) {
  std::vector<std::size_t> o(p.size());
  std::iota(o.begin(), o.end(), 0);
  std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return p[a] < p[b]; });
  return o;
}

// The type 7 quantile of a sample.
inline double quantile7(std::vector<double> x, double prob) {
  std::sort(x.begin(), x.end());
  const double h = (static_cast<double>(x.size()) - 1) * prob;
  const auto lo = static_cast<std::size_t>(std::floor(h));
  const std::size_t hi = std::min(lo + 1, x.size() - 1);
  return x[lo] + (h - static_cast<double>(lo)) * (x[hi] - x[lo]);
}

}  // namespace detail

inline std::vector<double> bonferroni(std::span<const double> p) {
  detail::require_p_values(p, "tyche::testing::bonferroni");
  std::vector<double> q(p.size());
  for (std::size_t i = 0; i < p.size(); i++) q[i] = std::min(1.0, static_cast<double>(p.size()) * p[i]);
  return q;
}

inline std::vector<double> holm(std::span<const double> p) {
  detail::require_p_values(p, "tyche::testing::holm");
  const std::size_t m = p.size();
  const auto o = detail::order(p);
  std::vector<double> q(m);
  double running = 0;
  for (std::size_t r = 0; r < m; r++) {
    running = std::max(running, std::min(1.0, static_cast<double>(m - r) * p[o[r]]));
    q[o[r]] = running;
  }
  return q;
}

inline std::vector<double> benjamini_hochberg(std::span<const double> p) {
  detail::require_p_values(p, "tyche::testing::benjamini_hochberg");
  const std::size_t m = p.size();
  const auto o = detail::order(p);
  std::vector<double> q(m);
  double running = 1;
  for (std::size_t r = m; r-- > 0;) {
    running = std::min(running, static_cast<double>(m) * p[o[r]] / static_cast<double>(r + 1));
    q[o[r]] = std::min(1.0, running);
  }
  return q;
}

enum class Pi0Method { fixed, smoother, bootstrap };

struct Pi0Options {
  Pi0Method method = Pi0Method::smoother;
  std::vector<double> lambda;  // the grid; empty: 0.05, 0.10, …, 0.95 (or 0.5 for fixed)
  double smooth_df = 3;
  bool smooth_log = false;     // smooth log π̂₀(λ) instead (qvalue's smooth.log.pi0)
};

struct Pi0 {
  double pi0 = 1;                  // the estimate, at most 1
  std::vector<double> lambda;      // the grid
  std::vector<double> by_lambda;   // π̂₀(λ) on the grid
  std::vector<double> smoothed;    // the spline at the grid (smoother only)
};

inline Pi0 pi0_estimate(std::span<const double> p, const Pi0Options& options = {}) {
  detail::require_p_values(p, "tyche::testing::pi0_estimate");
  const std::size_t m = p.size();
  if (m == 0) throw std::invalid_argument("tyche::testing::pi0_estimate: at least one p-value");
  Pi0 r;
  r.lambda = options.lambda;
  if (r.lambda.empty()) {
    if (options.method == Pi0Method::fixed) r.lambda = {0.5};
    else
      for (int k = 1; k <= 19; k++) r.lambda.push_back(0.05 * k);
  }
  std::sort(r.lambda.begin(), r.lambda.end());
  for (const double l : r.lambda)
    if (!(l >= 0 && l < 1)) throw std::invalid_argument("tyche::testing::pi0_estimate: λ in [0, 1)");
  const double dm = static_cast<double>(m);
  std::vector<double> sorted(p.begin(), p.end());
  std::sort(sorted.begin(), sorted.end());
  std::vector<double> W;
  for (const double l : r.lambda) {
    const double w = static_cast<double>(sorted.end() - std::lower_bound(sorted.begin(), sorted.end(), l));  // #{p ≥ λ}
    W.push_back(w);
    r.by_lambda.push_back(w / (dm * (1 - l)));
  }
  const std::size_t L = r.lambda.size();
  if (L == 1 || options.method == Pi0Method::fixed) {
    if (L != 1) throw std::invalid_argument("tyche::testing::pi0_estimate: the fixed method takes one λ");
    r.pi0 = std::min(r.by_lambda[0], 1.0);
  } else if (options.method == Pi0Method::smoother) {
    if (L < 4) throw std::invalid_argument("tyche::testing::pi0_estimate: the smoother needs at least four λ");
    std::vector<double> v = r.by_lambda;
    if (options.smooth_log)
      for (double& x : v) x = std::log(x);
    const auto spline = regression::smoothing_spline(r.lambda, v, options.smooth_df);
    r.smoothed = spline.fitted;
    if (options.smooth_log)
      for (double& x : r.smoothed) x = std::exp(x);
    r.pi0 = std::min(r.smoothed.back(), 1.0);
  } else {
    const double q10 = detail::quantile7(r.by_lambda, 0.1);
    double best = INFINITY;
    for (std::size_t k = 0; k < L; k++) {
      const double mse = W[k] / (dm * dm * (1 - r.lambda[k]) * (1 - r.lambda[k])) * (1 - W[k] / dm) + (r.by_lambda[k] - q10) * (r.by_lambda[k] - q10);
      if (mse < best) {
        best = mse;
        r.pi0 = std::min(r.by_lambda[k], 1.0);
      } else if (mse == best) {
        r.pi0 = std::min(r.pi0, std::min(r.by_lambda[k], 1.0));
      }
    }
  }
  if (!(r.pi0 > 0)) throw std::invalid_argument("tyche::testing::pi0_estimate: π̂₀ ≤ 0; the p-values are not valid, or λ's grid reaches past them");
  return r;
}

struct QValues {
  std::vector<double> q;  // one per p-value, in the input's order
  Pi0 pi0;
};

// Storey's q-values; with π₀ given (0 < π₀ ≤ 1) it is used as it is, and π₀ = 1 gives Benjamini and
// Hochberg's adjusted p-values.
inline QValues qvalues(std::span<const double> p, const Pi0Options& options = {}, double pi0 = -1) {
  QValues out;
  if (pi0 > 0) {
    if (pi0 > 1) throw std::invalid_argument("tyche::testing::qvalues: 0 < π₀ ≤ 1");
    detail::require_p_values(p, "tyche::testing::qvalues");
    out.pi0.pi0 = pi0;
  } else {
    out.pi0 = pi0_estimate(p, options);
  }
  out.q = benjamini_hochberg(p);
  for (double& x : out.q) x *= out.pi0.pi0;
  return out;
}

}  // namespace tyche::testing
