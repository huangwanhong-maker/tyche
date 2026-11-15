// Decisions in one step: a table of acts against states, chosen among under risk, under ignorance and under ambiguity.
//
// A decision table gives the utility u(a, s) of each act a in each state s of the world (Savage 1954, The Foundations of
// Statistics, ch. 2). What is known of the states fixes how an act is chosen:
//
// - Under risk, a probability p on the states is given, and an act is chosen for its expected utility Σ_s p(s) u(a, s)
//   (von Neumann and Morgenstern 1944; Savage 1954).
// - Under ignorance nothing is given (a game against nature; Milnor 1954, Games against nature, in Thrall, Coombs and Davis,
//   eds., Decision Processes). Wald's maximin takes the act whose worst outcome is best; maximax the act whose best is best;
//   Hurwicz's criterion weighs the worst and the best by α and 1 − α; Laplace's treats the states as equally likely; Savage's
//   minimax regret takes the act whose greatest regret, u(best act in s, s) − u(a, s), is least. Milnor (1954) showed that no
//   criterion has every property one would ask, and these five each give up a different one; the checks hold a table on which
//   all five choose differently.
// - Under ambiguity a set of probabilities is given, a credal set, here by finitely many points of which it is the convex hull
//   (Levi 1974, On indeterminate probabilities, Journal of Philosophy 71; Walley 1991, Statistical Reasoning with Imprecise
//   Probabilities). An act's expectation is then an interval [lower, upper], its ends attained at the points. Γ-maximin takes the
//   act of greatest lower expectation (Gilboa and Schmeidler 1989, Maxmin expected utility with non-unique prior, Journal of
//   Mathematical Economics 18); Γ-maximax the greatest upper; α-maxmin weighs the two (Ghirardato, Maccheroni and Marinacci 2004,
//   Differentiating ambiguity and ambiguity attitude, Journal of Economic Theory 118). Three rules choose a set rather than an act
//   (Troffaes 2007, Decision making under uncertainty using imprecise probabilities, International Journal of Approximate
//   Reasoning 45): interval dominance keeps the acts whose upper expectation is not below another's lower; maximality keeps those
//   that no other betters under every probability of the set; E-admissibility keeps those that are best under some probability of
//   the set, found by a linear programme on calx's simplex. Troffaes's inclusions (Γ-maximin and Γ-maximax acts are maximal;
//   E-admissible ⊆ maximal ⊆ interval-undominated) are held by the checks on random tables.
//
// Ties: every act whose value is within `tie` (absolute, 10⁻¹² by default) of the best is chosen, so a choice is a set.

#pragma once

#include <calx/optimisation/linear_programming.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::decision {

struct Table {
  std::vector<std::string> acts, states;
  std::vector<std::vector<double>> u;  // u[a][s]

  std::size_t n_acts() const { return u.size(); }
  std::size_t n_states() const { return u.empty() ? 0 : u[0].size(); }
};

inline void validate(const Table& t) {
  if (t.u.empty() || t.u[0].empty()) throw std::invalid_argument("tyche::decision: a table has at least one act and one state");
  for (const auto& row : t.u) {
    if (row.size() != t.u[0].size()) throw std::invalid_argument("tyche::decision: every act has a utility in every state");
    for (double x : row)
      if (!std::isfinite(x)) throw std::invalid_argument("tyche::decision: utilities are finite");
  }
  if (!t.acts.empty() && t.acts.size() != t.u.size()) throw std::invalid_argument("tyche::decision: a name for each act, or none");
  if (!t.states.empty() && t.states.size() != t.u[0].size()) throw std::invalid_argument("tyche::decision: a name for each state, or none");
}

inline void validate_law(std::span<const double> p, std::size_t states, const char* who) {
  if (p.size() != states) throw std::invalid_argument(std::string("tyche::decision::") + who + ": a probability for each state");
  double t = 0;
  for (double x : p) {
    if (!(x >= 0)) throw std::invalid_argument(std::string("tyche::decision::") + who + ": probabilities are not negative");
    t += x;
  }
  if (std::abs(t - 1) > 1e-9) throw std::invalid_argument(std::string("tyche::decision::") + who + ": probabilities sum to 1");
}

// The value each act is scored by, the acts chosen, and whether a higher value is better.
struct Choice {
  std::string criterion;
  std::vector<double> value;
  std::vector<std::size_t> chosen;
  bool higher_is_better = true;
};

inline constexpr double tie = 1e-12;

namespace detail {

inline Choice choose(std::string criterion, std::vector<double> value, bool higher = true) {
  Choice c{std::move(criterion), std::move(value), {}, higher};
  double best = higher ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
  for (double v : c.value) best = higher ? std::max(best, v) : std::min(best, v);
  for (std::size_t a = 0; a < c.value.size(); a++)
    if (std::abs(c.value[a] - best) <= tie * std::max(1.0, std::abs(best))) c.chosen.push_back(a);
  return c;
}

inline double dot(const std::vector<double>& u, std::span<const double> p) {
  double s = 0;
  for (std::size_t i = 0; i < u.size(); i++) s += u[i] * p[i];
  return s;
}

}  // namespace detail

// ---- Risk -------------------------------------------------------------------------------------------------------------------------

inline std::vector<double> expectations(const Table& t, std::span<const double> p) {
  validate(t);
  validate_law(p, t.n_states(), "expectations");
  std::vector<double> e;
  for (const auto& row : t.u) e.push_back(detail::dot(row, p));
  return e;
}

inline Choice expected_utility(const Table& t, std::span<const double> p) { return detail::choose("expected utility", expectations(t, p)); }

// ---- Dominance --------------------------------------------------------------------------------------------------------------------

// Whether act a dominates act b: at least as good in every state, and better in some (weakly), or better in every state
// (strictly).
inline bool dominates(const Table& t, std::size_t a, std::size_t b, bool strictly = false) {
  validate(t);
  bool better_somewhere = false;
  for (std::size_t s = 0; s < t.n_states(); s++) {
    if (t.u[a][s] < t.u[b][s]) return false;
    if (t.u[a][s] > t.u[b][s]) better_somewhere = true;
    else if (strictly) return false;
  }
  return better_somewhere;
}

// The admissible acts: those no act weakly dominates. Every criterion here but Laplace's and Hurwicz's with α ∈ {0, 1}
// chooses among them when the choice is unique.
inline std::vector<std::size_t> admissible(const Table& t) {
  std::vector<std::size_t> out;
  for (std::size_t a = 0; a < t.n_acts(); a++) {
    bool dominated = false;
    for (std::size_t b = 0; b < t.n_acts() && !dominated; b++) dominated = b != a && dominates(t, b, a);
    if (!dominated) out.push_back(a);
  }
  return out;
}

// ---- Ignorance --------------------------------------------------------------------------------------------------------------------

inline Choice maximin(const Table& t) {
  validate(t);
  std::vector<double> v;
  for (const auto& row : t.u) v.push_back(*std::min_element(row.begin(), row.end()));
  return detail::choose("maximin (Wald)", v);
}

inline Choice maximax(const Table& t) {
  validate(t);
  std::vector<double> v;
  for (const auto& row : t.u) v.push_back(*std::max_element(row.begin(), row.end()));
  return detail::choose("maximax", v);
}

// α times the worst plus (1 − α) times the best: α = 1 is maximin, α = 0 maximax (Hurwicz 1951).
inline Choice hurwicz(const Table& t, double alpha) {
  validate(t);
  if (!(alpha >= 0 && alpha <= 1)) throw std::invalid_argument("tyche::decision::hurwicz: α lies in [0, 1]");
  std::vector<double> v;
  for (const auto& row : t.u) v.push_back(alpha * *std::min_element(row.begin(), row.end()) + (1 - alpha) * *std::max_element(row.begin(), row.end()));
  return detail::choose("Hurwicz", v);
}

// Every state equally likely: the principle of insufficient reason (Laplace).
inline Choice laplace(const Table& t) {
  validate(t);
  const std::vector<double> p(t.n_states(), 1.0 / static_cast<double>(t.n_states()));
  auto c = detail::choose("Laplace", expectations(t, p));
  return c;
}

// The regret of each act in each state: what the best act in that state gains over it (Savage 1951, The theory of statistical
// decision, Journal of the American Statistical Association 46).
inline std::vector<std::vector<double>> regret(const Table& t) {
  validate(t);
  std::vector<std::vector<double>> r(t.n_acts(), std::vector<double>(t.n_states()));
  for (std::size_t s = 0; s < t.n_states(); s++) {
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < t.n_acts(); a++) best = std::max(best, t.u[a][s]);
    for (std::size_t a = 0; a < t.n_acts(); a++) r[a][s] = best - t.u[a][s];
  }
  return r;
}

// The act whose greatest regret is least; its value is that regret, lower being better.
inline Choice minimax_regret(const Table& t) {
  const auto r = regret(t);
  std::vector<double> v;
  for (const auto& row : r) v.push_back(*std::max_element(row.begin(), row.end()));
  return detail::choose("minimax regret (Savage)", v, false);
}

// ---- Ambiguity: credal sets -------------------------------------------------------------------------------------------------------

// A credal set as the convex hull of finitely many probabilities on the states.
struct Credal {
  std::vector<std::vector<double>> points;
};

inline void validate(const Credal& c, std::size_t states) {
  if (c.points.empty()) throw std::invalid_argument("tyche::decision: a credal set has at least one probability");
  for (const auto& p : c.points) validate_law(p, states, "credal");
}

// The vertices of {p : lower ≤ p ≤ upper, Σ p = 1}: at a vertex every coordinate but at most one is at one of its bounds.
// Refused (throws) when the bounds admit no probability.
inline Credal credal_from_bounds(std::span<const double> lower, std::span<const double> upper) {
  const std::size_t m = lower.size();
  if (upper.size() != m || m == 0) throw std::invalid_argument("tyche::decision::credal_from_bounds: a lower and an upper bound for each state");
  if (m > 20) throw std::invalid_argument("tyche::decision::credal_from_bounds: more than 20 states are refused");
  for (std::size_t i = 0; i < m; i++)
    if (!(lower[i] >= 0 && upper[i] <= 1 && lower[i] <= upper[i])) throw std::invalid_argument("tyche::decision::credal_from_bounds: 0 ≤ lower ≤ upper ≤ 1");
  Credal c;
  for (std::size_t free = 0; free < m; free++)
    for (std::size_t mask = 0; mask < (std::size_t{1} << (m - 1)); mask++) {
      std::vector<double> p(m);
      double rest = 0;
      std::size_t bit = 0;
      for (std::size_t j = 0; j < m; j++) {
        if (j == free) continue;
        p[j] = (mask >> bit++) & 1 ? upper[j] : lower[j];
        rest += p[j];
      }
      p[free] = 1 - rest;
      if (p[free] < lower[free] - 1e-12 || p[free] > upper[free] + 1e-12) continue;
      p[free] = std::clamp(p[free], lower[free], upper[free]);
      bool seen = false;
      for (const auto& q : c.points) {
        bool same = true;
        for (std::size_t j = 0; j < m && same; j++) same = std::abs(q[j] - p[j]) < 1e-12;
        seen = seen || same;
      }
      if (!seen) c.points.push_back(std::move(p));
    }
  if (c.points.empty()) throw std::invalid_argument("tyche::decision::credal_from_bounds: the bounds admit no probability");
  return c;
}

// The lower and upper expectations of a function of the states over the credal set: attained at its points.
inline double lower_expectation(const Credal& c, std::span<const double> f) {
  double v = std::numeric_limits<double>::infinity();
  for (const auto& p : c.points) v = std::min(v, detail::dot(p, f));
  return v;
}
inline double upper_expectation(const Credal& c, std::span<const double> f) {
  double v = -std::numeric_limits<double>::infinity();
  for (const auto& p : c.points) v = std::max(v, detail::dot(p, f));
  return v;
}

inline Choice gamma_maximin(const Table& t, const Credal& c) {
  validate(t);
  validate(c, t.n_states());
  std::vector<double> v;
  for (const auto& row : t.u) v.push_back(lower_expectation(c, row));
  return detail::choose("Γ-maximin (maxmin expected utility)", v);
}

inline Choice gamma_maximax(const Table& t, const Credal& c) {
  validate(t);
  validate(c, t.n_states());
  std::vector<double> v;
  for (const auto& row : t.u) v.push_back(upper_expectation(c, row));
  return detail::choose("Γ-maximax", v);
}

// α times the lower expectation plus (1 − α) times the upper.
inline Choice alpha_maxmin(const Table& t, const Credal& c, double alpha) {
  validate(t);
  validate(c, t.n_states());
  if (!(alpha >= 0 && alpha <= 1)) throw std::invalid_argument("tyche::decision::alpha_maxmin: α lies in [0, 1]");
  std::vector<double> v;
  for (const auto& row : t.u) v.push_back(alpha * lower_expectation(c, row) + (1 - alpha) * upper_expectation(c, row));
  return detail::choose("α-maxmin", v);
}

// The acts whose upper expectation is not below another act's lower expectation.
inline std::vector<std::size_t> interval_undominated(const Table& t, const Credal& c) {
  validate(t);
  validate(c, t.n_states());
  std::vector<double> lo, hi;
  for (const auto& row : t.u) lo.push_back(lower_expectation(c, row)), hi.push_back(upper_expectation(c, row));
  std::vector<std::size_t> out;
  for (std::size_t a = 0; a < t.n_acts(); a++) {
    bool dominated = false;
    for (std::size_t b = 0; b < t.n_acts() && !dominated; b++) dominated = b != a && lo[b] > hi[a] + tie;
    if (!dominated) out.push_back(a);
  }
  return out;
}

// The maximal acts (Walley 1991, §3.9): those for which no act b has a positive lower expectation of u(b) − u(a), that is,
// no act betters it under every probability of the set.
inline std::vector<std::size_t> maximal(const Table& t, const Credal& c) {
  validate(t);
  validate(c, t.n_states());
  std::vector<std::size_t> out;
  for (std::size_t a = 0; a < t.n_acts(); a++) {
    bool bettered = false;
    for (std::size_t b = 0; b < t.n_acts() && !bettered; b++) {
      if (b == a) continue;
      std::vector<double> d(t.n_states());
      for (std::size_t s = 0; s < t.n_states(); s++) d[s] = t.u[b][s] - t.u[a][s];
      bettered = lower_expectation(c, d) > tie;
    }
    if (!bettered) out.push_back(a);
  }
  return out;
}

// Whether act a is E-admissible (Levi 1974): best, among all acts, under some probability of the credal set. Decided by the
// feasibility of a linear programme in the weights λ of the set's points: λ ≥ 0, Σ λ = 1, and for every act b,
// Σ_k λ_k p_k · (u(b) − u(a)) ≤ 0.
inline bool e_admissible(const Table& t, const Credal& c, std::size_t a) {
  validate(t);
  validate(c, t.n_states());
  const std::size_t K = c.points.size(), A = t.n_acts();
  calx::optimisation::LinearProgram lp;
  lp.c.assign(K, 0.0);
  lp.a_eq = calx::linalg::Rectangular(1, K);
  for (std::size_t k = 0; k < K; k++) lp.a_eq(0, k) = 1;
  lp.b_eq = {1};
  lp.a_ub = calx::linalg::Rectangular(A - 1, K);
  std::size_t row = 0;
  for (std::size_t b = 0; b < A; b++) {
    if (b == a) continue;
    for (std::size_t k = 0; k < K; k++) {
      double d = 0;
      for (std::size_t s = 0; s < t.n_states(); s++) d += c.points[k][s] * (t.u[b][s] - t.u[a][s]);
      lp.a_ub(row, k) = d;
    }
    lp.b_ub.push_back(tie);
    row++;
  }
  if (A == 1) {
    lp.a_ub = calx::linalg::Rectangular(0, K);
    lp.b_ub.clear();
  }
  return calx::optimisation::simplex(lp).status == calx::optimisation::LpStatus::optimal;
}

inline std::vector<std::size_t> e_admissible(const Table& t, const Credal& c) {
  std::vector<std::size_t> out;
  for (std::size_t a = 0; a < t.n_acts(); a++)
    if (e_admissible(t, c, a)) out.push_back(a);
  return out;
}

}  // namespace tyche::decision
