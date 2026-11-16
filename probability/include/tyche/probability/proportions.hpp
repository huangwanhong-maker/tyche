// Intervals for a proportion: k successes in n trials (decision 0026).
//
// The estimate k/n has no honest interval by the textbook's p̂ ± z √(p̂(1 − p̂)/n), which has no width at k = 0 or k = n and
// covers badly near them (Brown, Cai and DasGupta 2001, Interval estimation for a binomial proportion, Statistical Science 16,
// 101–133). Two intervals are given.
//
// WILSON'S SCORE INTERVAL (Wilson 1927, Probable inference, the law of succession, and statistical inference, J. Amer. Statist.
// Assoc. 22, 209–212) is the set of p that a score test at the level would not reject: |k/n − p| ≤ z √(p(1 − p)/n), a quadratic
// in p whose roots are
//
//   [p̂ + z²/2n ∓ z √(p̂(1 − p̂)/n + z²/4n²)] / (1 + z²/n).
//
// Its coverage is close to nominal on average even for small n (Brown et al., §4), not at least nominal everywhere.
//
// THE CLOPPER–PEARSON INTERVAL (Clopper and Pearson 1934, The use of confidence or fiducial limits illustrated in the case of the
// binomial, Biometrika 26, 404–413) inverts the binomial's two tails exactly: the lower bound is the p at which P(X ≥ k) = α/2,
// the upper the p at which P(X ≤ k) = α/2, which are quantiles of beta laws, B(α/2; k, n − k + 1) and B(1 − α/2; k + 1, n − k).
// Its coverage is at least nominal for every p, and it is wider for it.

#pragma once

#include <tyche/probability/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace tyche::probability {

namespace detail {
inline void require_counts(std::int64_t k, std::int64_t n, double confidence) {
  if (n < 1 || k < 0 || k > n) throw std::invalid_argument("tyche::probability: a proportion is k of n, with n ≥ 1 and 0 ≤ k ≤ n");
  if (!(confidence > 0 && confidence < 1)) throw std::invalid_argument("tyche::probability: a confidence lies strictly between 0 and 1");
}
}  // namespace detail

// Wilson's score interval for k successes in n trials.
inline std::pair<double, double> wilson_interval(std::int64_t k, std::int64_t n, double confidence = 0.95) {
  detail::require_counts(k, n, confidence);
  const double z = Normal{}.quantile(0.5 + confidence / 2), nd = static_cast<double>(n), p = static_cast<double>(k) / nd;
  const double scale = 1 + z * z / nd, centre = (p + z * z / (2 * nd)) / scale, half = z * std::sqrt(p * (1 - p) / nd + z * z / (4 * nd * nd)) / scale;
  // At k = 0 and k = n the bound on that side is the estimate itself, which rounding must not move.
  return {k == 0 ? 0.0 : std::max(0.0, centre - half), k == n ? 1.0 : std::min(1.0, centre + half)};
}

// The Clopper–Pearson interval for k successes in n trials.
inline std::pair<double, double> clopper_pearson_interval(std::int64_t k, std::int64_t n, double confidence = 0.95) {
  detail::require_counts(k, n, confidence);
  const double alpha = 1 - confidence, kd = static_cast<double>(k), nd = static_cast<double>(n);
  const double lower = k == 0 ? 0.0 : Beta{kd, nd - kd + 1}.quantile(alpha / 2);
  const double upper = k == n ? 1.0 : Beta{kd + 1, nd - kd}.quantile(1 - alpha / 2);
  return {lower, upper};
}

}  // namespace tyche::probability
