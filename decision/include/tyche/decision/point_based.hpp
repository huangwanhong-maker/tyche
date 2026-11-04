// Point-based solvers of finite POMDPs: PBVI, Perseus and heuristic search value iteration, and the
// exact lookahead on the tree of beliefs.
//
// Every vector these solvers make is a point backup of vectors that are the values of conditional
// plans, and so is itself the value of a plan: the lower bound V(b) = max_α α·b never exceeds V*(b)
// (Pineau, Gordon and Thrun 2003, §3). They differ in where they back up.
// - PBVI (Pineau, Gordon and Thrun 2003): at a growing set B of beliefs, all at once, H times; B grows
//   from b₀ by stochastic simulation with explorative actions, each belief adding the successor
//   farthest (in L1) from B. Its error on B's reachable beliefs is at most (R_max − R_min) δ_B / (1 − γ)²
//   for the density δ_B of B (their Theorem 1).
// - Perseus (Spaan and Vlassis 2005, "Perseus: randomized point-based value iteration for POMDPs", JAIR
//   24, 195–220): B sampled once by random walks from b₀; each stage backs up beliefs drawn at random
//   from those not yet improved, until every belief of B is, so that V_{n+1}(b) ≥ V_n(b) on B; it stops
//   when no backup can raise any belief of B (the Bellman residual on B).
// - HSVI (Smith and Simmons 2004, "Heuristic search value iteration for POMDPs", UAI, 520–527; 2005,
//   "Point-based POMDP algorithms: improved analysis and implementation", UAI): a lower bound (vectors,
//   from the blind policies) and an upper bound (points (b_i, v_i) over corner values from the fast
//   informed bound, read by the sawtooth interpolation of Hauskrecht 2000,
//   V̄(b) = min(c·b, min_i [c·b + (v_i − c·b_i) min_{s: b_i(s)>0} b(s)/b_i(s)])),
//   refined along trials from b₀: at depth t it takes the action best by the upper bound and the
//   observation of largest weighted excess width P(o | b, a) [V̄ − V](τ(b, a, o)) − ε γ^{−(t+1)}, and
//   stops where V̄ − V ≤ ε γ^{−t}; on the way back it backs up both bounds. When V̄(b₀) − V(b₀) ≤ ε it
//   stops, with V*(b₀) in [V(b₀), V̄(b₀)]; for γ < 1 it stops after finitely many trials (Smith and
//   Simmons 2004, Theorem 1).
//
// The exact lookahead: V_d(b) = max_a [ρ(b, a) + γ Σ_o P(o | b, a) V_{d−1}(τ(b, a, o))], V_0 a given
// leaf value; exponential in d, for checks and small horizons.

#pragma once

#include <tyche/decision/alpha.hpp>
#include <tyche/decision/pomdp.hpp>

#include <calx/random/categorical.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace tyche::decision {

inline double l1(std::span<const double> a, std::span<const double> b) {
  double x = 0;
  for (std::size_t i = 0; i < a.size(); i++) x += std::fabs(a[i] - b[i]);
  return x;
}

// ---- the exact lookahead --------------------------------------------------------------------------------------------

inline double lookahead(const Pomdp& P, std::span<const double> b, int depth, const std::function<double(std::span<const double>)>& leaf) {
  if (depth == 0) return leaf(b);
  double best = -std::numeric_limits<double>::infinity();
  for (std::size_t a = 0; a < P.actions; a++) {
    double q = expected_reward(P, b, a);
    const auto po = observation_law(P, b, a);
    for (std::size_t o = 0; o < P.observations; o++) {
      if (po[o] <= 0) continue;
      q += P.discount * po[o] * lookahead(P, update(P, b, a, o).belief, depth - 1, leaf);
    }
    best = std::max(best, q);
  }
  return best;
}

// ---- PBVI ---------------------------------------------------------------------------------------------------------

struct PointBasedResult {
  ValueFunction lower;
  std::vector<Belief> beliefs;
  int backups = 0;      // point backups made
  int iterations = 0;   // sweeps (PBVI) or stages (Perseus)
};

struct PbviOptions {
  int expansions = 10;           // times B grows (at most doubling each time)
  int sweeps = 30;               // backups of all of B between expansions
  std::uint64_t seed = 20261104;
};

inline PointBasedResult pbvi(const Pomdp& P, const PbviOptions& opt = {}) {
  validate(P);
  calx::random::Stream rng(opt.seed, 0x70627669u, 0, 0);  // "pbvi"
  PointBasedResult out;
  out.lower = blind_lower_bound(P);
  out.beliefs = {P.initial};
  for (int e = 0; e <= opt.expansions; e++) {
    for (int sweep = 0; sweep < opt.sweeps; sweep++, out.iterations++) {
      ValueFunction next;
      for (const auto& b : out.beliefs) next.vectors.push_back(point_backup(P, out.lower, b)), out.backups++;
      out.lower = {prune_dominated(std::move(next.vectors))};
    }
    if (e == opt.expansions) break;
    // Stochastic simulation with explorative actions: from each belief, one successor per action, the
    // farthest from B added.
    std::vector<Belief> grown = out.beliefs;
    for (const auto& b : out.beliefs) {
      Belief far;
      double dist = 0;
      for (std::size_t a = 0; a < P.actions; a++) {
        const auto po = observation_law(P, b, a);
        const std::size_t o = calx::random::categorical(rng, po);
        Belief nb = update(P, b, a, o).belief;
        double d = std::numeric_limits<double>::infinity();
        for (const auto& c : grown) d = std::min(d, l1(nb, c));
        if (d > dist) dist = d, far = std::move(nb);
      }
      if (dist > 1e-9) grown.push_back(std::move(far));
    }
    out.beliefs = std::move(grown);
  }
  return out;
}

// ---- Perseus --------------------------------------------------------------------------------------------------------

struct PerseusOptions {
  std::size_t beliefs = 500;     // the size of B
  int walk = 30;                 // the length of each random walk from b₀
  int stages = 200;              // at most
  double tolerance = 1e-9;       // stop when no backup can raise a belief of B by more than this (times the scale)
  std::uint64_t seed = 20261104;
};

inline std::vector<Belief> random_beliefs(const Pomdp& P, std::size_t n, int walk, calx::random::Stream& rng) {
  std::vector<Belief> B{P.initial};
  Belief b = P.initial;
  int step = 0;
  while (B.size() < n) {
    if (step++ == walk) b = P.initial, step = 1;
    const std::size_t a = static_cast<std::size_t>(rng.below(P.actions));
    const auto po = observation_law(P, b, a);
    b = update(P, b, a, calx::random::categorical(rng, po)).belief;
    B.push_back(b);
  }
  return B;
}

inline PointBasedResult perseus(const Pomdp& P, const PerseusOptions& opt = {}) {
  validate(P);
  calx::random::Stream rng(opt.seed, 0x70727365u, 0, 0);  // "prse"
  PointBasedResult out;
  out.beliefs = random_beliefs(P, opt.beliefs, opt.walk, rng);
  out.lower = blind_lower_bound(P);
  const double scale = 1 + P.reward_span() / (1 - std::min(P.discount, 0.999999));
  for (out.iterations = 0; out.iterations < opt.stages; out.iterations++) {
    std::vector<double> before(out.beliefs.size());
    for (std::size_t i = 0; i < out.beliefs.size(); i++) before[i] = out.lower.value(out.beliefs[i]);
    ValueFunction next;
    std::vector<std::size_t> todo(out.beliefs.size());
    for (std::size_t i = 0; i < todo.size(); i++) todo[i] = i;
    std::vector<double> now(out.beliefs.size(), -std::numeric_limits<double>::infinity());
    while (!todo.empty()) {
      const std::size_t pick = todo[rng.below(todo.size())];
      const auto& b = out.beliefs[pick];
      AlphaVector alpha = point_backup(P, out.lower, b);
      out.backups++;
      if (dot(alpha.values, b) < before[pick]) alpha = out.lower.vectors[out.lower.best(b)];
      for (std::size_t i = 0; i < out.beliefs.size(); i++) now[i] = std::max(now[i], dot(alpha.values, out.beliefs[i]));
      next.vectors.push_back(std::move(alpha));
      std::vector<std::size_t> rest;
      for (const std::size_t i : todo)
        if (now[i] < before[i]) rest.push_back(i);
      todo.swap(rest);
    }
    out.lower = std::move(next);
    // Converged when no belief of B can be raised by a backup: the Bellman residual on B. (A stage may end
    // with nothing raised, every belief merely kept, and the next stage's order may still find a gain.)
    double residual = 0;
    for (const auto& b : out.beliefs) residual = std::max(residual, dot(point_backup(P, out.lower, b).values, b) - out.lower.value(b));
    if (residual <= opt.tolerance * scale) break;
  }
  return out;
}

// ---- HSVI -----------------------------------------------------------------------------------------------------------

// The sawtooth upper bound: corner values and interior points.
struct UpperBound {
  std::vector<double> corners;                 // V̄(e_s)
  std::vector<std::pair<Belief, double>> points;

  double value(std::span<const double> b) const {
    const double flat = dot(corners, b);
    double best = flat;
    for (const auto& [bi, vi] : points) {
      double phi = std::numeric_limits<double>::infinity();
      for (std::size_t s = 0; s < b.size(); s++)
        if (bi[s] > 0) phi = std::min(phi, b[s] / bi[s]);
      best = std::min(best, flat + phi * (vi - dot(corners, bi)));
    }
    return best;
  }

  // Drops the points the others already make redundant: (b_i, v_i) with the sawtooth of the rest at b_i no
  // larger than v_i + tolerance. The bound is unchanged at every b_i, and nowhere raised by more than tolerance.
  void prune(double tolerance = 1e-12) {
    std::vector<bool> keep(points.size(), true);
    for (std::size_t i = 0; i < points.size(); i++) {
      const auto& [bi, vi] = points[i];
      const double flat = dot(corners, bi);
      double best = flat;
      for (std::size_t j = 0; j < points.size() && best > vi + tolerance; j++) {
        if (j == i || !keep[j]) continue;
        const auto& [bj, vj] = points[j];
        double phi = std::numeric_limits<double>::infinity();
        for (std::size_t s = 0; s < bi.size(); s++)
          if (bj[s] > 0) phi = std::min(phi, bi[s] / bj[s]);
        best = std::min(best, flat + phi * (vj - dot(corners, bj)));
      }
      if (best <= vi + tolerance) keep[i] = false;
    }
    std::vector<std::pair<Belief, double>> kept;
    for (std::size_t i = 0; i < points.size(); i++)
      if (keep[i]) kept.push_back(std::move(points[i]));
    points.swap(kept);
  }
};

struct HsviOptions {
  double epsilon = 1e-3;         // the width at b₀ to reach
  int max_trials = 100000;
  int max_depth = 0;             // 0: the depth at which ε γ^{−t} exceeds the widest possible gap
};

struct HsviResult {
  ValueFunction lower;
  UpperBound upper;
  double lower_value = 0, upper_value = 0;  // at b₀
  int trials = 0, backups = 0;
  bool converged = false;
};

inline HsviResult hsvi(const Pomdp& P, const HsviOptions& opt = {}) {
  validate(P);
  detail::require(P.discount < 1, "HSVI needs γ < 1");
  HsviResult H;
  H.lower = blind_lower_bound(P);
  const auto fib = fast_informed_bound(P);
  H.upper.corners.assign(P.states, -std::numeric_limits<double>::infinity());
  for (const auto& v : fib.vectors)
    for (std::size_t s = 0; s < P.states; s++) H.upper.corners[s] = std::max(H.upper.corners[s], v.values[s]);
  const double gamma = P.discount;
  int depth_cap = opt.max_depth;
  if (depth_cap <= 0) {
    const double widest = 1 + P.reward_span() / (1 - gamma);
    depth_cap = static_cast<int>(std::ceil(std::log(opt.epsilon / widest) / std::log(gamma))) + 1;
  }
  const auto q_upper = [&](std::span<const double> b, std::size_t a, std::vector<Belief>* next, std::vector<double>* po_out) {
    const auto po = observation_law(P, b, a);
    double q = expected_reward(P, b, a);
    for (std::size_t o = 0; o < P.observations; o++) {
      if (po[o] <= 0) { if (next) next->push_back({}); continue; }
      Belief nb = update(P, b, a, o).belief;
      q += gamma * po[o] * H.upper.value(nb);
      if (next) next->push_back(std::move(nb));
    }
    if (po_out) *po_out = po;
    return q;
  };
  const auto back_up = [&](const Belief& b) {
    H.lower.vectors.push_back(point_backup(P, H.lower, b));
    H.backups++;
    double best = -std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < P.actions; a++) best = std::max(best, q_upper(b, a, nullptr, nullptr));
    if (best < H.upper.value(b)) H.upper.points.push_back({b, best});
  };
  std::function<void(const Belief&, int)> explore = [&](const Belief& b, int t) {
    if (H.upper.value(b) - H.lower.value(b) <= opt.epsilon * std::pow(gamma, -t) || t >= depth_cap) return;
    std::size_t astar = 0;
    double qbest = -std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < P.actions; a++) {
      const double q = q_upper(b, a, nullptr, nullptr);
      if (q > qbest) qbest = q, astar = a;
    }
    std::vector<Belief> next;
    std::vector<double> po;
    q_upper(b, astar, &next, &po);
    std::size_t ostar = 0;
    double excess = -std::numeric_limits<double>::infinity();
    for (std::size_t o = 0; o < P.observations; o++) {
      if (po[o] <= 0) continue;
      const double x = po[o] * (H.upper.value(next[o]) - H.lower.value(next[o]) - opt.epsilon * std::pow(gamma, -(t + 1)));
      if (x > excess) excess = x, ostar = o;
    }
    explore(next[ostar], t + 1);
    back_up(b);
  };
  std::size_t upper_pruned = 0;  // the number of upper points after the last pruning
  for (H.trials = 0; H.trials < opt.max_trials; H.trials++) {
    H.lower_value = H.lower.value(P.initial), H.upper_value = H.upper.value(P.initial);
    if (H.upper_value - H.lower_value <= opt.epsilon) { H.converged = true; break; }
    explore(P.initial, 0);
    if (H.trials % 50 == 49) H.lower.vectors = prune_dominated(std::move(H.lower.vectors));
    if (H.upper.points.size() > 2 * upper_pruned + 64) H.upper.prune(), upper_pruned = H.upper.points.size();
  }
  H.lower.vectors = prune_dominated(std::move(H.lower.vectors));
  H.lower_value = H.lower.value(P.initial), H.upper_value = H.upper.value(P.initial);
  H.converged = H.converged || H.upper_value - H.lower_value <= opt.epsilon;
  return H;
}

}  // namespace tyche::decision
