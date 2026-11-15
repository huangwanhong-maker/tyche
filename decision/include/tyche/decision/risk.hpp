// Lotteries: stochastic dominance and certainty equivalents.
//
// A lottery is a finite law on outcomes (money, years of life, anything ordered). Lottery A dominates B at first order when
// its distribution function lies nowhere above B's, F_A(x) ≤ F_B(x) for every x: then every decision maker who prefers more to
// less prefers A, E u(A) ≥ E u(B) for every increasing u. A dominates B at second order when ∫_{−∞}^x F_A ≤ ∫_{−∞}^x F_B for
// every x: then every such decision maker who is also averse to risk prefers A, for every increasing concave u (Hadar and
// Russell 1969, Rules for ordering uncertain prospects, American Economic Review 59; Rothschild and Stiglitz 1970, Increasing
// risk I, Journal of Economic Theory 2). The checks hold both theorems on random lotteries and random utilities.
//
// A certainty equivalent is the sure outcome a decision maker values as the lottery: u⁻¹(E u(X)). With constant absolute risk
// aversion a, u(x) = −e^{−ax}, it is −(1/a) log E e^{−aX}, and for a normal X it is μ − aσ²/2 exactly (Arrow 1965; Pratt
// 1964, Risk aversion in the small and in the large, Econometrica 32); with constant relative risk aversion ρ (outcomes
// positive), u(x) = x^{1−ρ}/(1−ρ), or log x at ρ = 1.

#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::decision {

struct Lottery {
  std::vector<double> outcomes, probabilities;
};

inline void validate(const Lottery& l) {
  if (l.outcomes.empty() || l.outcomes.size() != l.probabilities.size()) throw std::invalid_argument("tyche::decision: a lottery has outcomes, each with a probability");
  double t = 0;
  for (std::size_t i = 0; i < l.outcomes.size(); i++) {
    if (!std::isfinite(l.outcomes[i]) || !(l.probabilities[i] >= 0)) throw std::invalid_argument("tyche::decision: finite outcomes and probabilities not negative");
    t += l.probabilities[i];
  }
  if (std::abs(t - 1) > 1e-9) throw std::invalid_argument("tyche::decision: a lottery's probabilities sum to 1");
}

inline double expectation(const Lottery& l) {
  validate(l);
  double e = 0;
  for (std::size_t i = 0; i < l.outcomes.size(); i++) e += l.probabilities[i] * l.outcomes[i];
  return e;
}

namespace detail {

// The outcomes of both lotteries, sorted, without repeats.
inline std::vector<double> support(const Lottery& a, const Lottery& b) {
  std::vector<double> x = a.outcomes;
  x.insert(x.end(), b.outcomes.begin(), b.outcomes.end());
  std::sort(x.begin(), x.end());
  x.erase(std::unique(x.begin(), x.end()), x.end());
  return x;
}

inline double cdf(const Lottery& l, double x) {
  double f = 0;
  for (std::size_t i = 0; i < l.outcomes.size(); i++)
    if (l.outcomes[i] <= x) f += l.probabilities[i];
  return f;
}

}  // namespace detail

// Whether A dominates B at first order (F_A ≤ F_B everywhere, and somewhere strictly unless `weakly`).
inline bool first_order_dominates(const Lottery& a, const Lottery& b, double tolerance = 1e-12) {
  validate(a);
  validate(b);
  bool strict = false;
  for (const double x : detail::support(a, b)) {
    const double fa = detail::cdf(a, x), fb = detail::cdf(b, x);
    if (fa > fb + tolerance) return false;
    if (fa < fb - tolerance) strict = true;
  }
  return strict;
}

// Whether A dominates B at second order: the integral of F_A up to x lies nowhere above B's, and somewhere below. The integrals
// are piecewise linear between the outcomes, so they are compared at the outcomes.
inline bool second_order_dominates(const Lottery& a, const Lottery& b, double tolerance = 1e-12) {
  validate(a);
  validate(b);
  const auto xs = detail::support(a, b);
  double ia = 0, ib = 0;
  bool strict = false;
  for (std::size_t k = 1; k < xs.size(); k++) {
    const double h = xs[k] - xs[k - 1];
    ia += detail::cdf(a, xs[k - 1]) * h;
    ib += detail::cdf(b, xs[k - 1]) * h;
    if (ia > ib + tolerance) return false;
    if (ia < ib - tolerance) strict = true;
  }
  // Beyond the last outcome both distribution functions are 1, and the integrals keep the difference they have there.
  return strict;
}

// The certainty equivalent under constant absolute risk aversion a > 0 (a = 0: the expectation).
inline double certainty_equivalent_cara(const Lottery& l, double a) {
  validate(l);
  if (a < 0) throw std::invalid_argument("tyche::decision::certainty_equivalent_cara: a ≥ 0 (risk aversion)");
  if (a == 0) return expectation(l);
  // −(1/a) log Σ p e^{−a x}, computed stably about the least outcome.
  const double m = *std::min_element(l.outcomes.begin(), l.outcomes.end());
  double s = 0;
  for (std::size_t i = 0; i < l.outcomes.size(); i++) s += l.probabilities[i] * std::exp(-a * (l.outcomes[i] - m));
  return m - std::log(s) / a;
}

// The certainty equivalent under constant relative risk aversion ρ ≥ 0, for positive outcomes.
inline double certainty_equivalent_crra(const Lottery& l, double rho) {
  validate(l);
  if (rho < 0) throw std::invalid_argument("tyche::decision::certainty_equivalent_crra: ρ ≥ 0");
  for (double x : l.outcomes)
    if (!(x > 0)) throw std::invalid_argument("tyche::decision::certainty_equivalent_crra: outcomes are positive");
  if (std::abs(rho - 1) < 1e-12) {
    double s = 0;
    for (std::size_t i = 0; i < l.outcomes.size(); i++) s += l.probabilities[i] * std::log(l.outcomes[i]);
    return std::exp(s);
  }
  double s = 0;
  for (std::size_t i = 0; i < l.outcomes.size(); i++) s += l.probabilities[i] * std::pow(l.outcomes[i], 1 - rho);
  return std::pow(s, 1 / (1 - rho));
}

}  // namespace tyche::decision
