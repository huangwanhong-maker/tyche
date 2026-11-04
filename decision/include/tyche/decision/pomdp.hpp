// Partially observable Markov decision processes on finite sets: the model, beliefs and their
// updates, and the Markov decision process beneath.
//
// A POMDP (Smallwood and Sondik 1973; Kaelbling, Littman and Cassandra 1998) is a tuple
// (S, A, O, T, Z, R, γ, b₀) of finite states, actions and observations; a transition law T(s' | s, a);
// a law Z(o | s', a) of what is observed of the state entered; a reward R(s, a) for taking a in s; a
// discount 0 ≤ γ ≤ 1 (γ = 1 only over a finite horizon); and the law b₀ of the first state. The
// decision maker sees its actions and observations, not the states. The belief b ∈ Δ(S), the law of
// the state given that history, is a sufficient statistic for it (Åström 1965; Smallwood and Sondik
// 1973), and Bayes's rule carries it forward:
//
//   τ(b, a, o)(s') = Z(o | s', a) Σ_s T(s' | s, a) b(s) / P(o | b, a),
//   P(o | b, a)    = Σ_{s'} Z(o | s', a) Σ_s T(s' | s, a) b(s),
//
// so that a POMDP is a Markov decision process on Δ(S), the belief MDP, with reward
// ρ(b, a) = Σ_s b(s) R(s, a).
//
// The MDP beneath (the same states, actions, transitions and rewards, the state seen) is solved by
// value iteration: V_{k+1}(s) = max_a [R(s, a) + γ Σ_{s'} T(s' | s, a) V_k(s')], which contracts in
// the sup norm with modulus γ; stopped when ‖V_{k+1} − V_k‖∞ < ε (1 − γ) / (2γ), its greedy policy is
// ε-optimal and V_{k+1} within ε/2 of V* (Puterman 1994, Markov Decision Processes, Theorem 6.3.1).
// Its values bound the POMDP's from above: a decision maker who saw the state could do no worse.
//
// Storage: dense. T(s' | s, a) at [(a S + s) S + s'], Z(o | s', a) at [(a S + s') O + o], R(s, a) at
// [a S + s]. States, actions and observations are numbered; their meaning is the caller's.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::decision {

namespace detail {
inline void require(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("tyche::decision: " + what);
}
}  // namespace detail

using Belief = std::vector<double>;

struct Pomdp {
  std::size_t states = 0, actions = 0, observations = 0;
  std::vector<double> T, Z, R;
  double discount = 0.95;
  Belief initial;

  Pomdp() = default;
  Pomdp(std::size_t S, std::size_t A, std::size_t O, double gamma = 0.95)
      : states(S), actions(A), observations(O), T(A * S * S, 0.0), Z(A * S * O, 0.0), R(A * S, 0.0), discount(gamma),
        initial(S, S ? 1.0 / static_cast<double>(S) : 0.0) {}

  double& transition(std::size_t a, std::size_t s, std::size_t s2) { return T[(a * states + s) * states + s2]; }
  double transition(std::size_t a, std::size_t s, std::size_t s2) const { return T[(a * states + s) * states + s2]; }
  double& observe(std::size_t a, std::size_t s2, std::size_t o) { return Z[(a * states + s2) * observations + o]; }
  double observe(std::size_t a, std::size_t s2, std::size_t o) const { return Z[(a * states + s2) * observations + o]; }
  double& reward(std::size_t s, std::size_t a) { return R[a * states + s]; }
  double reward(std::size_t s, std::size_t a) const { return R[a * states + s]; }

  // The range of rewards, R_max − R_min, which scales every bound and tolerance.
  double reward_span() const {
    if (R.empty()) return 0;
    const auto [lo, hi] = std::minmax_element(R.begin(), R.end());
    return *hi - *lo;
  }
};

// Every law a law (rows of T and Z, and b₀, non-negative and summing to 1 within `tolerance`), the
// discount in [0, 1], rewards finite.
inline void validate(const Pomdp& P, double tolerance = 1e-9) {
  const std::size_t S = P.states, A = P.actions, O = P.observations;
  detail::require(S > 0 && A > 0 && O > 0, "a POMDP needs states, actions and observations");
  detail::require(P.T.size() == A * S * S && P.Z.size() == A * S * O && P.R.size() == A * S, "T is A·S·S, Z is A·S·O and R is A·S");
  detail::require(P.discount >= 0 && P.discount <= 1, "the discount lies in [0, 1]");
  const auto law = [&](std::span<const double> p, const std::string& what) {
    double sum = 0;
    for (const double x : p) {
      detail::require(x >= -tolerance && std::isfinite(x), what + " has a negative or undefined entry");
      sum += x;
    }
    detail::require(std::fabs(sum - 1) <= tolerance * static_cast<double>(p.size()), what + " does not sum to 1");
  };
  for (std::size_t a = 0; a < A; a++)
    for (std::size_t s = 0; s < S; s++) {
      law(std::span<const double>(P.T).subspan((a * S + s) * S, S), "T(· | " + std::to_string(s) + ", " + std::to_string(a) + ")");
      law(std::span<const double>(P.Z).subspan((a * S + s) * O, O), "Z(· | " + std::to_string(s) + ", " + std::to_string(a) + ")");
    }
  detail::require(P.initial.size() == S, "the initial belief has one entry per state");
  law(P.initial, "the initial belief");
  for (const double r : P.R) detail::require(std::isfinite(r), "rewards are finite");
}

// The law of the next state: Σ_s T(s' | s, a) b(s).
inline Belief predict(const Pomdp& P, std::span<const double> b, std::size_t a) {
  const std::size_t S = P.states;
  Belief next(S, 0.0);
  for (std::size_t s = 0; s < S; s++) {
    if (b[s] == 0) continue;
    const double* row = &P.T[(a * S + s) * S];
    for (std::size_t s2 = 0; s2 < S; s2++) next[s2] += b[s] * row[s2];
  }
  return next;
}

// The law of the observation after a in b: P(o | b, a).
inline std::vector<double> observation_law(const Pomdp& P, std::span<const double> b, std::size_t a) {
  const std::size_t S = P.states, O = P.observations;
  const Belief next = predict(P, b, a);
  std::vector<double> p(O, 0.0);
  for (std::size_t s2 = 0; s2 < S; s2++) {
    if (next[s2] == 0) continue;
    const double* row = &P.Z[(a * S + s2) * O];
    for (std::size_t o = 0; o < O; o++) p[o] += next[s2] * row[o];
  }
  return p;
}

struct Update {
  Belief belief;       // τ(b, a, o)
  double probability;  // P(o | b, a)
};

// Bayes's rule: τ(b, a, o) and P(o | b, a). An observation of probability zero is refused.
inline Update update(const Pomdp& P, std::span<const double> b, std::size_t a, std::size_t o) {
  const std::size_t S = P.states, O = P.observations;
  Belief next = predict(P, b, a);
  double total = 0;
  for (std::size_t s2 = 0; s2 < S; s2++) total += (next[s2] *= P.Z[(a * S + s2) * O + o]);
  detail::require(total > 0, "an observation of probability zero under the belief");
  for (double& x : next) x /= total;
  return {std::move(next), total};
}

// ρ(b, a) = Σ_s b(s) R(s, a).
inline double expected_reward(const Pomdp& P, std::span<const double> b, std::size_t a) {
  double r = 0;
  for (std::size_t s = 0; s < P.states; s++) r += b[s] * P.R[a * P.states + s];
  return r;
}

// The MDP beneath: values, action values Q(s, a) at [a S + s], a greedy policy.
struct MdpSolution {
  std::vector<double> value, q;
  std::vector<std::size_t> policy;
  int iterations = 0;
  double residual = 0;  // ‖V_k − V_{k−1}‖∞ at the last sweep
  bool converged = false;
};

namespace detail {
inline void mdp_sweep(const Pomdp& P, const std::vector<double>& v, std::vector<double>& q, std::vector<double>& next, std::vector<std::size_t>& policy) {
  const std::size_t S = P.states, A = P.actions;
  for (std::size_t s = 0; s < S; s++) {
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < A; a++) {
      double x = P.R[a * S + s];
      const double* row = &P.T[(a * S + s) * S];
      double ev = 0;
      for (std::size_t s2 = 0; s2 < S; s2++) ev += row[s2] * v[s2];
      x += P.discount * ev;
      q[a * S + s] = x;
      if (x > best) best = x, policy[s] = a;
    }
    next[s] = best;
  }
}
}  // namespace detail

// Value iteration to ε-optimality (Puterman 1994, Theorem 6.3.1); γ < 1.
inline MdpSolution solve_mdp(const Pomdp& P, double epsilon = 1e-10, int max_iterations = 1000000) {
  detail::require(P.discount < 1, "solve_mdp needs γ < 1; use solve_mdp_horizon for γ = 1");
  const std::size_t S = P.states;
  MdpSolution M;
  M.value.assign(S, 0.0), M.q.assign(P.actions * S, 0.0), M.policy.assign(S, 0);
  std::vector<double> next(S);
  const double stop = P.discount > 0 ? epsilon * (1 - P.discount) / (2 * P.discount) : std::numeric_limits<double>::infinity();
  for (M.iterations = 1; M.iterations <= max_iterations; M.iterations++) {
    detail::mdp_sweep(P, M.value, M.q, next, M.policy);
    double d = 0;
    for (std::size_t s = 0; s < S; s++) d = std::max(d, std::fabs(next[s] - M.value[s]));
    M.value.swap(next);
    M.residual = d;
    if (d < stop) { M.converged = true; break; }
  }
  return M;
}

// The values of the first `horizon` steps, V_H, by H sweeps from V_0 = 0 (any γ ≤ 1).
inline MdpSolution solve_mdp_horizon(const Pomdp& P, int horizon) {
  const std::size_t S = P.states;
  MdpSolution M;
  M.value.assign(S, 0.0), M.q.assign(P.actions * S, 0.0), M.policy.assign(S, 0);
  std::vector<double> next(S);
  for (M.iterations = 0; M.iterations < horizon;) {
    detail::mdp_sweep(P, M.value, M.q, next, M.policy);
    M.value.swap(next);
    M.iterations++;
  }
  M.converged = true;
  return M;
}

}  // namespace tyche::decision
