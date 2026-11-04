// Value functions of a finite POMDP as sets of α-vectors: their bounds, pruning, the exact backup and
// the point backup.
//
// The value of the first H steps is piecewise linear and convex on Δ(S): V_H(b) = max_{α ∈ Γ_H} α·b
// for a finite set Γ_H of vectors on S, each the value of a conditional plan and labelled by the plan's
// first action (Smallwood and Sondik 1973). The Bellman backup carries Γ to Γ' = ⋃_a ⊕_o Γ^{a,o},
//
//   Γ^{a,o} = { g^{a,o}_α = R(·, a)/|O| + γ Σ_{s'} T(s' | ·, a) Z(o | s', a) α(s') : α ∈ Γ },
//
// the cross sum ⊕ adding one vector of each Γ^{a,o} in every way (Kaelbling, Littman and Cassandra 1998,
// §4). Most of the vectors so made are nowhere the maximum. Incremental pruning removes them after every
// cross sum, prune(prune(Γ^{a,1} ⊕ Γ^{a,2}) ⊕ Γ^{a,3}) …, which leaves the same function (Cassandra,
// Littman and Zhang 1997). A vector is kept when it is the strict maximum somewhere: when the linear
// programme
//
//   maximise δ   subject to   b·(α − α') ≥ δ for every other α',   Σ_s b(s) = 1,   b ≥ 0
//
// has δ > tolerance (its witness b), solved by calx's simplex; vectors dominated pointwise are removed
// first, and a vector strictly best at a corner of the simplex needs no programme (Lark's filter: White
// 1991, "A survey of solution techniques for the partially observed Markov decision process", Ann.
// Oper. Res. 32, 215–230).
//
// After H exact backups from Γ_0 = {0}, ‖V_H − V*‖∞ ≤ γ^H max_{s,a} |R(s, a)| / (1 − γ), the backup
// being a γ-contraction in the sup norm.
//
// The point backup at one belief b keeps only the vector of Γ' best at b (Pineau, Gordon and Thrun 2003,
// "Point-based value iteration: an anytime algorithm for POMDPs", IJCAI, 1025–1032, §3):
//
//   backup(Γ, b) = argmax_a b·[R(·, a) + Σ_o argmax_{α ∈ Γ} b·g^{a,o}_α],  g^{a,o}_α = γ Σ_{s'} T Z α,
//
// in O(|A| |O| (|S|² + |Γ| |S|)) operations.
//
// Bounds that need only the states (Hauskrecht 2000, "Value-function approximations for partially
// observable Markov decision processes", JAIR 13, 33–94):
// - QMDP (Littman, Cassandra and Kaelbling 1995, "Learning policies for partially observable
//   environments: scaling up", ICML): the MDP's action values Q(·, a) as vectors, an upper bound;
// - the fast informed bound, one vector per action, the fixed point of
//   α_a(s) = R(s, a) + γ Σ_o max_{a'} Σ_{s'} T(s' | s, a) Z(o | s', a) α_{a'}(s'),
//   below QMDP and above V* (iterated from QMDP's vectors, every iterate stays above V*);
// - blind policies, which take one action whatever is seen: α_a = R(·, a) + γ T_a α_a, a lower bound
//   (iterated from R_min / (1 − γ), every iterate stays below V*).
// So max_a QMDP_a·b ≥ max_a FIB_a·b ≥ V*(b) ≥ max_a blind_a·b for every belief b.

#pragma once

#include <tyche/decision/pomdp.hpp>

#include <calx/linalg/rectangular.hpp>
#include <calx/optimisation/linear_programming.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace tyche::decision {

struct AlphaVector {
  std::vector<double> values;  // one per state
  std::size_t action = 0;      // the first action of its plan
};

inline double dot(std::span<const double> a, std::span<const double> b) {
  double x = 0;
  for (std::size_t i = 0; i < a.size(); i++) x += a[i] * b[i];
  return x;
}

struct ValueFunction {
  std::vector<AlphaVector> vectors;

  // max_α α·b (−∞ when empty).
  double value(std::span<const double> b) const {
    double best = -std::numeric_limits<double>::infinity();
    for (const auto& v : vectors) best = std::max(best, dot(v.values, b));
    return best;
  }
  // The index of a maximising vector (the first among ties).
  std::size_t best(std::span<const double> b) const {
    detail::require(!vectors.empty(), "an empty value function has no best vector");
    std::size_t k = 0;
    double top = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < vectors.size(); i++) {
      const double x = dot(vectors[i].values, b);
      if (x > top) top = x, k = i;
    }
    return k;
  }
  std::size_t action(std::span<const double> b) const { return vectors[best(b)].action; }
  std::size_t size() const { return vectors.size(); }
};

// ---- bounds from the states ---------------------------------------------------------------------------------------

inline ValueFunction qmdp(const Pomdp& P, double epsilon = 1e-10) {
  const auto M = solve_mdp(P, epsilon);
  ValueFunction V;
  for (std::size_t a = 0; a < P.actions; a++) V.vectors.push_back({std::vector<double>(M.q.begin() + long(a * P.states), M.q.begin() + long((a + 1) * P.states)), a});
  return V;
}

inline ValueFunction fast_informed_bound(const Pomdp& P, double epsilon = 1e-10, int max_iterations = 1000000) {
  detail::require(P.discount < 1, "the fast informed bound needs γ < 1");
  const std::size_t S = P.states, A = P.actions, O = P.observations;
  ValueFunction V = qmdp(P, epsilon);
  const double stop = P.discount > 0 ? epsilon * (1 - P.discount) / (2 * P.discount) : std::numeric_limits<double>::infinity();
  std::vector<double> next(S);
  for (int it = 0; it < max_iterations; it++) {
    double change = 0;
    std::vector<std::vector<double>> fresh(A, std::vector<double>(S));
    for (std::size_t a = 0; a < A; a++)
      for (std::size_t s = 0; s < S; s++) {
        double sum = 0;
        const double* row = &P.T[(a * S + s) * S];
        for (std::size_t o = 0; o < O; o++) {
          double best = -std::numeric_limits<double>::infinity();
          for (std::size_t a2 = 0; a2 < A; a2++) {
            double x = 0;
            for (std::size_t s2 = 0; s2 < S; s2++) x += row[s2] * P.Z[(a * S + s2) * O + o] * V.vectors[a2].values[s2];
            best = std::max(best, x);
          }
          sum += best;
        }
        fresh[a][s] = P.R[a * S + s] + P.discount * sum;
        change = std::max(change, std::fabs(fresh[a][s] - V.vectors[a].values[s]));
      }
    for (std::size_t a = 0; a < A; a++) V.vectors[a].values = fresh[a];
    if (change < stop) break;
  }
  return V;
}

inline ValueFunction blind_lower_bound(const Pomdp& P, double epsilon = 1e-10, int max_iterations = 1000000) {
  detail::require(P.discount < 1, "the blind bound needs γ < 1");
  const std::size_t S = P.states, A = P.actions;
  const double rmin = *std::min_element(P.R.begin(), P.R.end());
  ValueFunction V;
  const double stop = P.discount > 0 ? epsilon * (1 - P.discount) / (2 * P.discount) : std::numeric_limits<double>::infinity();
  for (std::size_t a = 0; a < A; a++) {
    std::vector<double> v(S, rmin / (1 - P.discount)), next(S);
    for (int it = 0; it < max_iterations; it++) {
      double change = 0;
      for (std::size_t s = 0; s < S; s++) {
        const double* row = &P.T[(a * S + s) * S];
        double ev = 0;
        for (std::size_t s2 = 0; s2 < S; s2++) ev += row[s2] * v[s2];
        next[s] = P.R[a * S + s] + P.discount * ev;
        change = std::max(change, std::fabs(next[s] - v[s]));
      }
      v.swap(next);
      if (change < stop) break;
    }
    V.vectors.push_back({v, a});
  }
  return V;
}

// ---- pruning ------------------------------------------------------------------------------------------------------

namespace detail {

// δ of the witness programme of `vector` against `others` (+∞ when there are none), and its witness.
// The programme is posed in units of D = max |α'(s) − α(s)|, so that its coefficients are of order one
// however close the vectors, and with δ = D (t − 3), t ≥ 0, so that every right-hand side is positive:
// rows (α' − α)·b / D + t ≤ 3. (With δ free and zero right-hand sides every pivot is degenerate.)
inline double witness_margin(const std::vector<double>& vector, const std::vector<const std::vector<double>*>& others, std::vector<double>* witness = nullptr) {
  if (others.empty()) return std::numeric_limits<double>::infinity();
  const std::size_t S = vector.size(), n = S + 1;
  double D = 0;
  for (const auto* o : others)
    for (std::size_t s = 0; s < S; s++) D = std::max(D, std::fabs((*o)[s] - vector[s]));
  if (D == 0) {  // equal to another vector everywhere
    if (witness) witness->assign(S, 1.0 / static_cast<double>(S));
    return 0;
  }
  calx::optimisation::LinearProgram lp;
  lp.c.assign(n, 0.0), lp.c[S] = -1;  // minimise −t
  lp.a_ub = calx::linalg::Rectangular(others.size(), n);
  lp.b_ub.assign(others.size(), 3.0);
  for (std::size_t j = 0; j < others.size(); j++) {
    for (std::size_t s = 0; s < S; s++) lp.a_ub(j, s) = ((*others[j])[s] - vector[s]) / D;
    lp.a_ub(j, S) = 1;
  }
  lp.a_eq = calx::linalg::Rectangular(1, n);
  for (std::size_t s = 0; s < S; s++) lp.a_eq(0, s) = 1;
  lp.b_eq = {1.0};
  const auto r = calx::optimisation::simplex(lp);
  require(r.status == calx::optimisation::LpStatus::optimal,
          "the witness programme of pruning did not solve (status " + std::to_string(static_cast<int>(r.status)) + ", " + std::to_string(others.size()) + " rows)");
  if (witness) witness->assign(r.x.begin(), r.x.begin() + long(S));
  return D * (r.x[S] - 3);
}

// The index of the vector of `candidates` largest at b, ties (within tol) going to the lexicographically
// largest, so that a vector tied at b with a better one elsewhere is not taken for it.
inline std::size_t best_at(const std::vector<AlphaVector>& V, const std::vector<std::size_t>& candidates, std::span<const double> b, double tol) {
  std::size_t k = candidates[0];
  double top = dot(V[k].values, b);
  for (std::size_t c = 1; c < candidates.size(); c++) {
    const std::size_t i = candidates[c];
    const double x = dot(V[i].values, b);
    if (x > top + tol || (x >= top - tol && std::lexicographical_compare(V[k].values.begin(), V[k].values.end(), V[i].values.begin(), V[i].values.end())))
      top = std::max(top, x), k = i;
  }
  return k;
}

}  // namespace detail

// The vectors that are the strict maximum somewhere, by more than `tolerance` times the scale of the set
// (1 + max |α(s)|). Pointwise domination first; then Lark's filter (White 1991): a set W of vectors known
// to be needed, begun with the best at each corner of the simplex; each remaining candidate is tested
// against W alone, and where it has a witness b the candidate best at b joins W, else it is dropped. The
// programmes have |W| rows, not as many as there are candidates.
inline std::vector<AlphaVector> prune(std::vector<AlphaVector> V, double tolerance = 1e-9) {
  if (V.size() <= 1) return V;
  const std::size_t S = V[0].values.size();
  double scale = 1;
  for (const auto& v : V)
    for (const double x : v.values) scale = std::max(scale, 1 + std::fabs(x));
  const double tol = tolerance * scale;
  // Pointwise: drop a vector that another is at least everywhere (within tol); of equal ones keep the first.
  std::vector<bool> alive(V.size(), true);
  for (std::size_t i = 0; i < V.size(); i++) {
    if (!alive[i]) continue;
    for (std::size_t j = 0; j < V.size(); j++) {
      if (i == j || !alive[j]) continue;
      bool ge = true;
      for (std::size_t s = 0; s < S && ge; s++) ge = V[j].values[s] >= V[i].values[s] - tol;
      if (ge) { alive[i] = false; break; }
    }
  }
  std::vector<std::size_t> F;
  for (std::size_t i = 0; i < V.size(); i++)
    if (alive[i]) F.push_back(i);
  std::vector<std::size_t> W;
  const auto take = [&](std::size_t i) {
    W.push_back(i);
    F.erase(std::find(F.begin(), F.end(), i));
  };
  for (std::size_t s = 0; s < S && !F.empty(); s++) {
    std::vector<double> e(S, 0.0);
    e[s] = 1;
    const std::size_t k = detail::best_at(V, F, e, tol);
    bool known = false;
    for (const std::size_t w : W) known = known || dot(V[w].values, e) >= dot(V[k].values, e) - tol;
    if (!known) take(k);
  }
  std::vector<double> b;
  while (!F.empty()) {
    const std::size_t phi = F.back();
    std::vector<const std::vector<double>*> others;
    for (const std::size_t w : W) others.push_back(&V[w].values);
    const double delta = detail::witness_margin(V[phi].values, others, &b);
    if (delta > tol) take(detail::best_at(V, F, b, tol));
    else F.pop_back();
  }
  std::vector<AlphaVector> out;
  for (const std::size_t i : W) out.push_back(std::move(V[i]));
  return out;
}

// Pointwise domination only (no programmes): the cheap pruning of the point-based solvers, whose sets
// are as large as their beliefs.
inline std::vector<AlphaVector> prune_dominated(std::vector<AlphaVector> V, double tolerance = 1e-12) {
  if (V.size() <= 1) return V;
  const std::size_t S = V[0].values.size();
  std::vector<bool> alive(V.size(), true);
  for (std::size_t i = 0; i < V.size(); i++) {
    if (!alive[i]) continue;
    for (std::size_t j = 0; j < V.size(); j++) {
      if (i == j || !alive[j]) continue;
      bool ge = true;
      for (std::size_t s = 0; s < S && ge; s++) ge = V[j].values[s] >= V[i].values[s] - tolerance;
      if (ge) { alive[i] = false; break; }
    }
  }
  std::vector<AlphaVector> out;
  for (std::size_t i = 0; i < V.size(); i++)
    if (alive[i]) out.push_back(std::move(V[i]));
  return out;
}

// ---- the exact backup -----------------------------------------------------------------------------------------------

namespace detail {

// g^{a,o}_α(s) = share · R(s, a) + γ Σ_{s'} T(s' | s, a) Z(o | s', a) α(s').
inline std::vector<double> project(const Pomdp& P, std::size_t a, std::size_t o, const std::vector<double>& alpha, double share) {
  const std::size_t S = P.states, O = P.observations;
  std::vector<double> g(S);
  for (std::size_t s = 0; s < S; s++) {
    const double* row = &P.T[(a * S + s) * S];
    double x = 0;
    for (std::size_t s2 = 0; s2 < S; s2++) x += row[s2] * P.Z[(a * S + s2) * O + o] * alpha[s2];
    g[s] = share * P.R[a * S + s] + P.discount * x;
  }
  return g;
}

}  // namespace detail

// Γ' from Γ by incremental pruning (Cassandra, Littman and Zhang 1997).
inline ValueFunction exact_backup(const Pomdp& P, const ValueFunction& V, double tolerance = 1e-9) {
  detail::require(!V.vectors.empty(), "the exact backup needs at least one vector");
  const std::size_t A = P.actions, O = P.observations, S = P.states;
  const double share = 1.0 / static_cast<double>(O);
  std::vector<AlphaVector> all;
  for (std::size_t a = 0; a < A; a++) {
    std::vector<AlphaVector> W;
    for (std::size_t o = 0; o < O; o++) {
      std::vector<AlphaVector> G;
      for (const auto& v : V.vectors) G.push_back({detail::project(P, a, o, v.values, share), a});
      G = prune(std::move(G), tolerance);
      if (o == 0) { W = std::move(G); continue; }
      std::vector<AlphaVector> sum;
      sum.reserve(W.size() * G.size());
      for (const auto& w : W)
        for (const auto& g : G) {
          std::vector<double> x(S);
          for (std::size_t s = 0; s < S; s++) x[s] = w.values[s] + g.values[s];
          sum.push_back({std::move(x), a});
        }
      W = prune(std::move(sum), tolerance);
    }
    for (auto& w : W) all.push_back(std::move(w));
  }
  return {prune(std::move(all), tolerance)};
}

// Γ_1, …, Γ_H from Γ_0 = {0}: the values of the first H steps (any γ ≤ 1).
inline std::vector<ValueFunction> exact_value_iteration(const Pomdp& P, int horizon, double tolerance = 1e-9) {
  validate(P);
  std::vector<ValueFunction> out;
  ValueFunction V{{{std::vector<double>(P.states, 0.0), 0}}};
  for (int h = 0; h < horizon; h++) {
    V = exact_backup(P, V, tolerance);
    out.push_back(V);
  }
  return out;
}

// ---- the point backup -----------------------------------------------------------------------------------------------

inline AlphaVector point_backup(const Pomdp& P, const ValueFunction& V, std::span<const double> b) {
  detail::require(!V.vectors.empty(), "the point backup needs at least one vector");
  const std::size_t S = P.states, A = P.actions, O = P.observations;
  AlphaVector best;
  double top = -std::numeric_limits<double>::infinity();
  std::vector<double> weight(S);
  for (std::size_t a = 0; a < A; a++) {
    std::vector<double> alpha(S);
    for (std::size_t s = 0; s < S; s++) alpha[s] = P.R[a * S + s];
    const Belief next = predict(P, b, a);
    for (std::size_t o = 0; o < O; o++) {
      // b·g^{a,o}_α = γ Σ_{s'} [Σ_s b(s) T(s' | s, a)] Z(o | s', a) α(s'): the best α at the unnormalised posterior.
      bool any = false;
      for (std::size_t s2 = 0; s2 < S; s2++) weight[s2] = next[s2] * P.Z[(a * S + s2) * O + o], any = any || weight[s2] > 0;
      // An observation that cannot follow b and a still needs a branch, so that the vector stays the value of a
      // plan (and a lower bound elsewhere); any vector serves, and adds nothing at b.
      const std::size_t k = any ? V.best(weight) : V.best(next);
      const auto g = detail::project(P, a, o, V.vectors[k].values, 0.0);
      for (std::size_t s = 0; s < S; s++) alpha[s] += g[s];
    }
    const double x = dot(alpha, b);
    if (x > top) top = x, best = {std::move(alpha), a};
  }
  return best;
}

}  // namespace tyche::decision
