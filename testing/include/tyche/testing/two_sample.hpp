// Comparing two groups, or a group with a level (decision 0027): weighted means and their standard errors, Welch's test with
// Satterthwaite's degrees of freedom, a share against a level, and the difference of two shares with Newcombe's interval.
//
// - Weighted mean. x̄_w = Σ w x / Σ w. The weights are sampling weights: a row stands for w people, and the spread of the values
//   is the weighted variance s²_w = Σ w (x − x̄_w)² / Σ w · n_e/(n_e − 1), with Kish's effective size n_e = (Σ w)² / Σ w² (Kish
//   1965, Survey Sampling, §11.7). The standard error is s_w / √n_e. With equal weights this is the textbook s/√n exactly. It
//   approximates the design-based variance (peira's surveys hold the exact ultimate-cluster form when the design is known).
// - Welch's test (Welch 1947, Biometrika 34, 28–35): t = (x̄₁ − x̄₂)/√(se₁² + se₂²), its degrees of freedom by Satterthwaite (1946,
//   Biometrics Bull. 2, 110–114), ν = (se₁² + se₂²)² / (se₁⁴/(n₁ − 1) + se₂⁴/(n₂ − 1)) with the effective sizes; the interval
//   d ± t_{ν,1−α/2} √(se₁² + se₂²). scipy.stats.ttest_ind(equal_var=False) on equal weights.
// - A mean against a level: t = (x̄ − τ)/se on n_e − 1 degrees of freedom.
// - Shares. A share p̂ = k/n against a level π₀: the score test z = (p̂ − π₀)/√(π₀(1 − π₀)/n) with Wilson's interval (proportions.hpp).
//   Two shares: the difference p̂₁ − p̂₂ with Newcombe's hybrid score interval (Newcombe 1998, Statist. Med. 17, 873–890, method 10:
//   d − √((p̂₁ − l₁)² + (u₂ − p̂₂)²), d + √((u₁ − p̂₁)² + (p̂₂ − l₂)²), from each share's Wilson interval), and the pooled z test of
//   equal shares. Counts may be fractional (a weighted share times an effective size).
//
// Limits: rows with a missing value (NaN) or no weight are left out; fewer than two effective rows in a group gives no standard error
// (NaN), and the tests then answer NaN rather than a number.
#pragma once

#include <tyche/probability/distributions.hpp>
#include <tyche/probability/p_values.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace tyche::testing {

using tyche::probability::Normal;
using tyche::probability::StudentT;
using tyche::probability::two_sided_normal;
using tyche::probability::two_sided_t;

struct WeightedMean {
  double mean = std::numeric_limits<double>::quiet_NaN();
  double sd = std::numeric_limits<double>::quiet_NaN();    // the weighted standard deviation of the values
  double se = std::numeric_limits<double>::quiet_NaN();    // of the mean: sd / √n_e
  double total = 0;                                        // Σ w
  double effective = 0;                                    // Kish's n_e
  std::size_t rows = 0;                                    // rows counted
};

inline WeightedMean weighted_mean(std::span<const double> x, std::span<const double> w = {}) {
  if (!w.empty() && w.size() != x.size()) throw std::invalid_argument("weighted_mean: the weights are not one per value");
  WeightedMean m;
  double sw = 0, sww = 0, swx = 0;
  for (std::size_t i = 0; i < x.size(); i++) {
    const double wi = w.empty() ? 1.0 : w[i];
    if (!std::isfinite(x[i]) || !(wi > 0)) continue;
    sw += wi, sww += wi * wi, swx += wi * x[i], m.rows++;
  }
  if (sw <= 0) return m;
  m.total = sw, m.mean = swx / sw, m.effective = sw * sw / sww;
  double ss = 0;
  for (std::size_t i = 0; i < x.size(); i++) {
    const double wi = w.empty() ? 1.0 : w[i];
    if (!std::isfinite(x[i]) || !(wi > 0)) continue;
    ss += wi * (x[i] - m.mean) * (x[i] - m.mean);
  }
  if (m.effective > 1) {
    m.sd = std::sqrt(ss / sw * m.effective / (m.effective - 1));
    m.se = m.sd / std::sqrt(m.effective);
  }
  return m;
}

struct Comparison {
  double difference = std::numeric_limits<double>::quiet_NaN();  // first less second (or less the level)
  double se = std::numeric_limits<double>::quiet_NaN();
  double statistic = std::numeric_limits<double>::quiet_NaN();
  double df = std::numeric_limits<double>::quiet_NaN();
  double p_value = std::numeric_limits<double>::quiet_NaN();     // two-sided
  double low = std::numeric_limits<double>::quiet_NaN(), high = std::numeric_limits<double>::quiet_NaN();
};

namespace detail {
inline double t_quantile(double nu, double confidence) {
  if (!(nu > 0)) return std::numeric_limits<double>::quiet_NaN();
  return StudentT{nu}.quantile(0.5 + confidence / 2);
}
inline void require_confidence(double c) {
  if (!(c > 0 && c < 1)) throw std::invalid_argument("a confidence lies strictly between 0 and 1");
}
}  // namespace detail

// Welch's test of equal means, and the interval of their difference.
inline Comparison welch(const WeightedMean& a, const WeightedMean& b, double confidence = 0.95) {
  detail::require_confidence(confidence);
  Comparison c;
  c.difference = a.mean - b.mean;
  if (!(std::isfinite(a.se) && std::isfinite(b.se))) return c;
  const double va = a.se * a.se, vb = b.se * b.se;
  c.se = std::sqrt(va + vb);
  if (!(c.se > 0)) return c;
  c.statistic = c.difference / c.se;
  c.df = (va + vb) * (va + vb) / (va * va / (a.effective - 1) + vb * vb / (b.effective - 1));
  c.p_value = two_sided_t(c.statistic, c.df);
  const double q = detail::t_quantile(c.df, confidence);
  c.low = c.difference - q * c.se, c.high = c.difference + q * c.se;
  return c;
}

// A mean against a level.
inline Comparison against_level(const WeightedMean& a, double level, double confidence = 0.95) {
  detail::require_confidence(confidence);
  Comparison c;
  c.difference = a.mean - level;
  if (!(std::isfinite(a.se) && a.se > 0)) return c;
  c.se = a.se, c.df = a.effective - 1, c.statistic = c.difference / c.se;
  c.p_value = two_sided_t(c.statistic, c.df);
  const double q = detail::t_quantile(c.df, confidence);
  c.low = c.difference - q * c.se, c.high = c.difference + q * c.se;
  return c;
}

namespace detail {
// Wilson's score interval for fractional counts (proportions.hpp holds the integer form).
inline std::pair<double, double> wilson(double k, double n, double confidence) {
  const double z = Normal{}.quantile(0.5 + confidence / 2), p = k / n, z2 = z * z;
  const double centre = (p + z2 / (2 * n)) / (1 + z2 / n);
  const double half = z * std::sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n);
  return {std::max(0.0, centre - half), std::min(1.0, centre + half)};
}
}  // namespace detail

// A share k/n against a level π₀: the score test, with the share's Wilson interval read as an interval of p̂ − π₀.
inline Comparison share_against_level(double k, double n, double level, double confidence = 0.95) {
  detail::require_confidence(confidence);
  if (!(n > 0) || k < 0 || k > n) throw std::invalid_argument("share_against_level: needs 0 ≤ k ≤ n and n > 0");
  if (!(level > 0 && level < 1)) throw std::invalid_argument("share_against_level: the level lies strictly between 0 and 1");
  Comparison c;
  const double p = k / n;
  c.difference = p - level, c.se = std::sqrt(level * (1 - level) / n), c.statistic = c.difference / c.se;
  c.df = std::numeric_limits<double>::infinity();
  c.p_value = two_sided_normal(c.statistic);
  const auto [l, u] = detail::wilson(k, n, confidence);
  c.low = l - level, c.high = u - level;
  return c;
}

// The difference of two shares: Newcombe's hybrid score interval, and the pooled z test of equal shares.
inline Comparison share_difference(double k1, double n1, double k2, double n2, double confidence = 0.95) {
  detail::require_confidence(confidence);
  if (!(n1 > 0 && n2 > 0) || k1 < 0 || k2 < 0 || k1 > n1 || k2 > n2) throw std::invalid_argument("share_difference: needs 0 ≤ k ≤ n and n > 0 in both");
  Comparison c;
  const double p1 = k1 / n1, p2 = k2 / n2, d = p1 - p2;
  const auto [l1, u1] = detail::wilson(k1, n1, confidence);
  const auto [l2, u2] = detail::wilson(k2, n2, confidence);
  c.difference = d;
  c.low = d - std::sqrt((p1 - l1) * (p1 - l1) + (u2 - p2) * (u2 - p2));
  c.high = d + std::sqrt((u1 - p1) * (u1 - p1) + (p2 - l2) * (p2 - l2));
  const double pooled = (k1 + k2) / (n1 + n2);
  c.se = std::sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2));
  c.df = std::numeric_limits<double>::infinity();
  if (c.se > 0) c.statistic = d / c.se, c.p_value = two_sided_normal(c.statistic);
  return c;
}

}  // namespace tyche::testing
