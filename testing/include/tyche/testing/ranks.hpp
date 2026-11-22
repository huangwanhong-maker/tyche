// Ranks of two samples (decision 0036): Mann and Whitney's U, the area under the curve that it is, and its normal test with ties.
//
// - U counts the pairs (x, y), x from the first sample and y from the second, with x > y, ties counting one half:
//   U = Σ_i Σ_j ([x_i > y_j] + ½[x_i = y_j]). Its share of the n₁n₂ pairs is the area under the receiver operating characteristic
//   curve of the scores x (positives) against y (negatives), AUC = U/(n₁n₂) = P(X > Y) + ½P(X = Y) (Bamber 1975, J. Math. Psychol.
//   12, 387–415; Hanley and McNeil 1982, Radiology 143, 29–36). AUC = 1 when every positive outranks every negative, ½ when the two
//   samples are alike.
// - Under the null of exchangeable samples, E U = n₁n₂/2 and, with ties of sizes t_k among the n = n₁ + n₂ values,
//   Var U = n₁n₂/12 · ((n + 1) − Σ (t_k³ − t_k)/(n(n − 1))) (Lehmann 1975, Nonparametrics, §1.4); z = (U − n₁n₂/2)/√Var U, referred
//   to the standard normal two-sided, without a continuity correction (scipy.stats.mannwhitneyu(method="asymptotic",
//   use_continuity=False)). The normal approximation is asymptotic; for small samples a permutation test (permutation.hpp) is exact.
//
// Computed by sorting the pooled values once, in O(n log n). NaNs are left out.

#pragma once

#include <tyche/probability/p_values.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyche::testing {

struct RankSum {
  double u = 0, auc = std::numeric_limits<double>::quiet_NaN();
  double statistic = std::numeric_limits<double>::quiet_NaN(), p_value = std::numeric_limits<double>::quiet_NaN();
  std::size_t n1 = 0, n2 = 0;
};

inline RankSum mann_whitney(std::span<const double> x, std::span<const double> y) {
  std::vector<std::pair<double, int>> v;
  for (double a : x)
    if (!std::isnan(a)) v.push_back({a, 0});
  for (double b : y)
    if (!std::isnan(b)) v.push_back({b, 1});
  RankSum R;
  for (const auto& e : v) (e.second == 0 ? R.n1 : R.n2)++;
  if (R.n1 == 0 || R.n2 == 0) throw std::invalid_argument("tyche::testing::mann_whitney: each sample needs a value");
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  // Midranks; the first sample's rank sum gives U = R₁ − n₁(n₁ + 1)/2.
  const std::size_t n = v.size();
  double r1 = 0, ties = 0;
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i;
    while (j < n && v[j].first == v[i].first) j++;
    const double mid = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2, t = static_cast<double>(j - i);
    for (std::size_t k = i; k < j; k++)
      if (v[k].second == 0) r1 += mid;
    ties += t * t * t - t;
    i = j;
  }
  const double n1 = static_cast<double>(R.n1), n2 = static_cast<double>(R.n2), N = n1 + n2;
  R.u = r1 - n1 * (n1 + 1) / 2;
  R.auc = R.u / (n1 * n2);
  const double var = n1 * n2 / 12 * ((N + 1) - (N > 1 ? ties / (N * (N - 1)) : 0));
  if (var > 0) R.statistic = (R.u - n1 * n2 / 2) / std::sqrt(var), R.p_value = tyche::probability::two_sided_normal(R.statistic);
  return R;
}

}  // namespace tyche::testing
