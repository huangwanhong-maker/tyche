// From belief functions to probabilities and expectations (decision 0023).
//
// A normal mass function m defines its credal set C(m) = {P : P(A) ≥ Bel(A) for every A ⊆ Ω}, the laws compatible
// with the evidence (Dempster 1967, Ann. Math. Statist. 38, 325–339). Then
//   probability_bounds(m, A)   min and max of P(A) over C(m): [Bel(A), Pl(A)] (Dempster 1967)
//   lower_expectation(m, f)    min over C(m) of E_P[f] = Σ_A m(A) min_{ω ∈ A} f(ω), the Choquet integral of f with
//   upper_expectation(m, f)    respect to Bel (and its max, with respect to Pl, Σ_A m(A) max_{ω ∈ A} f(ω)); that the
//                              minimum is attained at this value holds because Bel is a totally monotone capacity,
//                              whose core's vertices are the laws induced by the orderings of Ω (Shapley 1971,
//                              Int. J. Game Theory 1, 11–26; Chateauneuf and Jaffray 1989, Math. Social Sci. 17,
//                              263–283)
//   in_credal_set(m, p)        whether P(A) ≥ Bel(A) − tolerance for every A (frames of at most 20 elements)
//   pignistic(m)               BetP(ω) = Σ_{A ∋ ω} m(A) / (|A| (1 − m(∅))), the law on which the transferable belief
//                              model decides, each focal set's mass shared equally among its elements (Smets and
//                              Kennes 1994, Artificial Intelligence 66, 191–234); it lies in C(m) when m is normal
//   contour(m)                 Pl({ω}) for each ω, which determines Pl when m is consonant
// The credal operations need a normal mass function (an unnormalised one has an empty credal set) and refuse
// m(∅) > 0, with the reason; the pignistic law needs m(∅) < 1.

#pragma once

#include <tyche/beliefs/common.hpp>
#include <tyche/beliefs/mass.hpp>
#include <tyche/beliefs/transforms.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tyche::beliefs {

struct Interval {
  double lower = 0, upper = 0;
};

namespace detail {
inline void require_normal(const Mass& m, const char* what) {
  require(m.conflict() == 0, std::string(what) + ": m(∅) > 0 leaves the credal set empty; normalise the mass function (Dempster) or give the conflict to Ω (Yager) first");
}
}  // namespace detail

inline Interval probability_bounds(const Mass& m, Set A) {
  detail::require_normal(m, "probability_bounds");
  return {belief(m, A), plausibility(m, A)};
}

inline double lower_expectation(const Mass& m, std::span<const double> f) {
  detail::require_normal(m, "lower_expectation");
  detail::require(f.size() == m.frame, "lower_expectation: one value of the function per element of the frame");
  for (double x : f) detail::require(std::isfinite(x), "lower_expectation: the function's values are finite");
  double s = 0, c = 0;
  for (const auto& fc : m.focal) {
    double lo = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < m.frame; i++)
      if (fc.set >> i & 1) lo = std::min(lo, f[i]);
    detail::neumaier_add(s, c, fc.mass * lo);
  }
  return s + c;
}

inline double upper_expectation(const Mass& m, std::span<const double> f) {
  detail::require_normal(m, "upper_expectation");
  detail::require(f.size() == m.frame, "upper_expectation: one value of the function per element of the frame");
  for (double x : f) detail::require(std::isfinite(x), "upper_expectation: the function's values are finite");
  double s = 0, c = 0;
  for (const auto& fc : m.focal) {
    double hi = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < m.frame; i++)
      if (fc.set >> i & 1) hi = std::max(hi, f[i]);
    detail::neumaier_add(s, c, fc.mass * hi);
  }
  return s + c;
}

inline bool in_credal_set(const Mass& m, std::span<const double> p, double tolerance = 1e-9) {
  detail::require_normal(m, "in_credal_set");
  detail::check_dense(m.frame);
  detail::require(p.size() == m.frame, "in_credal_set: one probability per element of the frame");
  double total = 0;
  for (double x : p) {
    if (!(x >= -tolerance)) return false;
    total += x;
  }
  if (std::fabs(total - 1) > tolerance) return false;
  std::vector<double> pa(std::size_t{1} << m.frame, 0.0);
  for (std::size_t i = 0; i < m.frame; i++) pa[std::size_t{1} << i] = p[i];
  pa = morphe::combinatorics::subset_zeta(std::move(pa));
  const auto bel = belief_function(m);
  for (std::size_t A = 0; A < pa.size(); A++)
    if (pa[A] < bel[A] - tolerance) return false;
  return true;
}

inline std::vector<double> pignistic(const Mass& m) {
  std::vector<double> p(m.frame, 0.0);
  double keep = 0, c = 0;
  for (const auto& f : m.focal)
    if (f.set != 0) detail::neumaier_add(keep, c, f.mass);
  keep += c;
  detail::require(keep > 0, "pignistic: m(∅) = 1 leaves nothing to share");
  for (const auto& f : m.focal) {
    if (f.set == 0) continue;
    const double share = f.mass / (static_cast<double>(cardinality(f.set)) * keep);
    for (std::size_t i = 0; i < m.frame; i++)
      if (f.set >> i & 1) p[i] += share;
  }
  return p;
}

inline std::vector<double> contour(const Mass& m) {
  std::vector<double> pl(m.frame, 0.0);
  for (std::size_t i = 0; i < m.frame; i++) pl[i] = plausibility(m, singleton(i));
  return pl;
}

}  // namespace tyche::beliefs
