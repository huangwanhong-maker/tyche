// p-values of test statistics under their null distributions.
//
// A p-value is the probability, under the null hypothesis, of a statistic at least as extreme as
// the one observed (Lehmann and Romano 2005, Testing Statistical Hypotheses, 3rd ed., §3.3).
// "Extreme" is the test's to say:
// - upper_tail(d, x) = P(X ≥ x) for tests that reject for large values (χ², F, Ljung–Box);
//   for a continuous d this is d.survival(x);
// - lower_tail(d, x) = P(X ≤ x) (the Dickey–Fuller test rejects for small values);
// - two_sided(d, x) = 2 min(P(X ≤ x), P(X ≥ x)), capped at 1, the equal-tailed p-value; for a
//   distribution symmetric about 0 it is P(|X| ≥ |x|) (t and z tests of a coefficient).
// Each tail is computed from the distribution's own cdf or survival function, so a p-value of
// 10⁻⁵⁰ keeps its digits.

#pragma once

#include <tyche/probability/distributions.hpp>

#include <algorithm>
#include <cmath>

namespace tyche::probability {

template <class D>
inline double upper_tail(const D& d, double x) {
  return d.survival(x);
}

template <class D>
inline double lower_tail(const D& d, double x) {
  return d.cdf(x);
}

template <class D>
inline double two_sided(const D& d, double x) {
  if (std::isnan(x)) return x;
  return std::min(1.0, 2 * std::min(d.cdf(x), d.survival(x)));
}

// P(|Z| ≥ |z|) for Z standard normal.
inline double two_sided_normal(double z) { return std::isnan(z) ? z : std::erfc(std::fabs(z) / std::numbers::sqrt2); }

// P(|T| ≥ |t|) for T Student's t with ν degrees of freedom.
inline double two_sided_t(double t, double nu) {
  if (std::isnan(t)) return t;
  return std::min(1.0, 2 * StudentT{nu}.survival(std::fabs(t)));
}

}  // namespace tyche::probability
