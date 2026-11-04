// Constrained POMDPs: a reward maximised subject to an expected discounted cost within a budget, by Lagrangian
// duality over plans that carry both values.
//
// The problem (Altman 1999; Isom, Meyn and Braatz 2008): maximise V_R^π(b₀) = E^π[Σ γ^t R(s_t, a_t)] subject to
// V_C^π(b₀) = E^π[Σ γ^t C(s_t, a_t)] ≤ c. Optimal policies may randomise. For λ ≥ 0, the plan best for the
// scalarised reward R − λ C gives the dual function D(λ) = max_π [V_R^π − λ V_C^π] + λ c ≥ the constrained optimum
// (weak duality); its least value over λ equals the optimum over mixtures of plans (strong duality, the dual of the
// linear programme over plans), and a mixture of the plans optimal on either side of the least λ, weighted to spend
// the budget exactly, attains it.
//
// Plans carry two vectors, the reward's and the cost's (Isom et al. 2008; Kim, Lee, Kim and Poupart 2011,
// "Point-based value iteration for constrained POMDPs", IJCAI, 1968–1973): the point backup chooses by the
// scalarised value and carries both. Over a finite horizon, backups at every belief reachable from b₀ at each depth
// are exact there, so the plan found at b₀ is the scalarised optimum and the mixture the constrained optimum. Over
// an infinite horizon the backups run on a sampled set of beliefs (as Perseus's), and the policy is the finite-state
// controller on the final plans, evaluated exactly: the reward and cost reported are those of the policy executed,
// a feasible mixture; the dual computed from them is then not an upper bound, and HSVI's upper bound of the
// scalarised problem gives one.

#pragma once

#include <tyche/decision/alpha.hpp>
#include <tyche/decision/point_based.hpp>
#include <tyche/decision/pomdp.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace tyche::decision {

struct PlanVector {
  std::vector<double> reward, cost;
  std::size_t action = 0;
};

// The point backup of plans for R − λ C at b.
inline PlanVector point_backup(const Pomdp& P, std::span<const double> C, const std::vector<PlanVector>& V, std::span<const double> b, double lambda) {
  detail::require(!V.empty(), "the constrained backup needs at least one plan");
  const std::size_t S = P.states, A = P.actions, O = P.observations;
  PlanVector best;
  double top = -std::numeric_limits<double>::infinity();
  std::vector<double> weight(S);
  for (std::size_t a = 0; a < A; a++) {
    PlanVector p{std::vector<double>(S), std::vector<double>(S), a};
    for (std::size_t s = 0; s < S; s++) p.reward[s] = P.R[a * S + s], p.cost[s] = C[a * S + s];
    const Belief next = predict(P, b, a);
    for (std::size_t o = 0; o < O; o++) {
      bool any = false;
      for (std::size_t s2 = 0; s2 < S; s2++) weight[s2] = next[s2] * P.Z[(a * S + s2) * O + o], any = any || weight[s2] > 0;
      const auto& w = any ? weight : next;
      std::size_t k = 0;
      double kv = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < V.size(); i++) {
        const double x = dot(V[i].reward, w) - lambda * dot(V[i].cost, w);
        if (x > kv) kv = x, k = i;
      }
      const auto gr = detail::project(P, a, o, V[k].reward, 0.0);
      // The cost's projection: γ Σ_{s'} T Z cost(s'), the same operator on the cost vector.
      const auto gc = detail::project(P, a, o, V[k].cost, 0.0);
      for (std::size_t s = 0; s < S; s++) p.reward[s] += gr[s], p.cost[s] += gc[s];
    }
    const double x = dot(p.reward, b) - lambda * dot(p.cost, b);
    if (x > top) top = x, best = std::move(p);
  }
  return best;
}

// The beliefs reachable from b₀ at each depth below the horizon (distinct to 10⁻¹²).
inline std::vector<std::vector<Belief>> reachable_beliefs(const Pomdp& P, int horizon) {
  std::vector<std::vector<Belief>> levels{{P.initial}};
  for (int d = 1; d < horizon; d++) {
    std::vector<Belief> next;
    for (const auto& b : levels.back())
      for (std::size_t a = 0; a < P.actions; a++) {
        const auto po = observation_law(P, b, a);
        for (std::size_t o = 0; o < P.observations; o++) {
          if (po[o] <= 0) continue;
          auto nb = update(P, b, a, o).belief;
          bool seen = false;
          for (const auto& c : next) seen = seen || l1(c, nb) < 1e-12;
          if (!seen) next.push_back(std::move(nb));
        }
      }
    levels.push_back(std::move(next));
  }
  return levels;
}

// A finite-state controller: node k takes action[k] and, on observation o, moves to next[k][o]; reward[k] and
// cost[k] are its exact values from each state, the fixed point of
//   v(k, s) = R(s, a_k) + γ Σ_{s'} T(s' | s, a_k) Σ_o Z(o | s', a_k) v(next[k][o], s')
// (and the same for C), a γ-contraction (Hansen 1998; Poupart et al. 2015 evaluate constrained policies so).
struct Controller {
  std::vector<std::size_t> action;
  std::vector<std::vector<std::size_t>> next;
  std::vector<std::vector<double>> reward, cost;
  std::size_t start = 0;

  void evaluate(const Pomdp& P, std::span<const double> C, double epsilon = 1e-12, int max_iterations = 100000) {
    const std::size_t K = action.size(), S = P.states, O = P.observations;
    reward.assign(K, std::vector<double>(S, 0.0)), cost.assign(K, std::vector<double>(S, 0.0));
    auto nr = reward, nc = cost;
    const double scale = 1 + std::max(P.reward_span(), *std::max_element(C.begin(), C.end())) / (1 - P.discount);
    for (int it = 0; it < max_iterations; it++) {
      double change = 0;
      for (std::size_t k = 0; k < K; k++) {
        const std::size_t a = action[k];
        for (std::size_t s = 0; s < S; s++) {
          double er = 0, ec = 0;
          for (std::size_t s2 = 0; s2 < S; s2++) {
            const double t = P.T[(a * S + s) * S + s2];
            if (t == 0) continue;
            for (std::size_t o = 0; o < O; o++) {
              const double w = t * P.Z[(a * S + s2) * O + o];
              if (w == 0) continue;
              er += w * reward[next[k][o]][s2], ec += w * cost[next[k][o]][s2];
            }
          }
          nr[k][s] = P.R[a * S + s] + P.discount * er, nc[k][s] = C[a * S + s] + P.discount * ec;
          change = std::max(change, std::max(std::fabs(nr[k][s] - reward[k][s]), std::fabs(nc[k][s] - cost[k][s])));
        }
      }
      reward.swap(nr), cost.swap(nc);
      if (change < epsilon * scale) break;
    }
  }
};

struct ScalarisedPlan {
  PlanVector plan;                 // the plan chosen at b₀ (its values those of the controller, when there is one)
  double reward = 0, cost = 0;     // V_R and V_C at b₀ of the policy that will be executed
  double scalarised = 0;           // reward − λ cost
  Controller controller;           // over an infinite horizon: the policy executed, exactly evaluated
};

// Over a finite horizon: exact, by backups at every reachable belief from the last depth up.
inline ScalarisedPlan scalarised_finite(const Pomdp& P, std::span<const double> C, double lambda, int horizon, const std::vector<std::vector<Belief>>& levels) {
  std::vector<PlanVector> V{{std::vector<double>(P.states, 0.0), std::vector<double>(P.states, 0.0), 0}};
  for (int d = horizon - 1; d >= 0; d--) {
    std::vector<PlanVector> next;
    for (const auto& b : levels[d]) next.push_back(point_backup(P, C, V, b, lambda));
    V = std::move(next);
  }
  ScalarisedPlan out{V[0], dot(V[0].reward, P.initial), dot(V[0].cost, P.initial), 0, {}};
  out.scalarised = out.reward - lambda * out.cost;
  return out;
}

// Over an infinite horizon: Perseus's randomised backups on a sampled belief set, for R − λ C; then the controller
// whose nodes are the final plans, each moving on o to the plan best for R − λC at the posterior of the belief it
// was made at, evaluated exactly. Its values are those of the policy it executes (a lower bound of the scalarised
// optimum), whatever the backups' vectors claimed.
inline ScalarisedPlan scalarised_infinite(const Pomdp& P, std::span<const double> C, double lambda, const std::vector<Belief>& B, int stages = 200, std::uint64_t seed = 20261104) {
  calx::random::Stream rng(seed, 0x63706276u, 0, 0);  // "cpbv"
  // Start from the blind plans, both values carried, each made at b₀.
  std::vector<PlanVector> V;
  std::vector<Belief> made;
  for (std::size_t a = 0; a < P.actions; a++) {
    std::vector<double> r(P.states, 0.0), c(P.states, 0.0), nr(P.states), nc(P.states);
    for (int it = 0; it < 100000; it++) {
      double change = 0;
      for (std::size_t s = 0; s < P.states; s++) {
        double er = 0, ec = 0;
        for (std::size_t s2 = 0; s2 < P.states; s2++) er += P.transition(a, s, s2) * r[s2], ec += P.transition(a, s, s2) * c[s2];
        nr[s] = P.reward(s, a) + P.discount * er, nc[s] = C[a * P.states + s] + P.discount * ec;
        change = std::max(change, std::max(std::fabs(nr[s] - r[s]), std::fabs(nc[s] - c[s])));
      }
      r.swap(nr), c.swap(nc);
      if (change < 1e-12) break;
    }
    V.push_back({r, c, a}), made.push_back(P.initial);
  }
  const auto scal = [&](const PlanVector& p, std::span<const double> b) { return dot(p.reward, b) - lambda * dot(p.cost, b); };
  const auto best_of = [&](const std::vector<PlanVector>& W, std::span<const double> b) {
    std::size_t k = 0;
    double kv = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < W.size(); i++)
      if (const double x = scal(W[i], b); x > kv) kv = x, k = i;
    return k;
  };
  for (int stage = 0; stage < stages; stage++) {
    std::vector<double> before(B.size());
    for (std::size_t i = 0; i < B.size(); i++) before[i] = scal(V[best_of(V, B[i])], B[i]);
    std::vector<PlanVector> next;
    std::vector<Belief> next_made;
    std::vector<std::size_t> todo(B.size());
    for (std::size_t i = 0; i < todo.size(); i++) todo[i] = i;
    std::vector<double> now(B.size(), -std::numeric_limits<double>::infinity());
    while (!todo.empty()) {
      const std::size_t pick = todo[rng.below(todo.size())];
      PlanVector p = point_backup(P, C, V, B[pick], lambda);
      Belief at = B[pick];
      if (scal(p, B[pick]) < before[pick]) {
        const std::size_t k = best_of(V, B[pick]);
        p = V[k], at = made[k];
      }
      for (std::size_t i = 0; i < B.size(); i++) now[i] = std::max(now[i], scal(p, B[i]));
      next.push_back(std::move(p)), next_made.push_back(std::move(at));
      std::vector<std::size_t> rest;
      for (const std::size_t i : todo)
        if (now[i] < before[i]) rest.push_back(i);
      todo.swap(rest);
    }
    V = std::move(next), made = std::move(next_made);
    double residual = 0;
    for (const auto& b : B) residual = std::max(residual, scal(point_backup(P, C, V, b, lambda), b) - scal(V[best_of(V, b)], b));
    if (residual <= 1e-9 * (1 + P.reward_span() / (1 - P.discount))) break;
  }
  // The controller on the final plans.
  Controller K;
  for (std::size_t k = 0; k < V.size(); k++) {
    const std::size_t a = V[k].action;
    K.action.push_back(a);
    std::vector<std::size_t> succ(P.observations);
    const auto po = observation_law(P, made[k], a);
    for (std::size_t o = 0; o < P.observations; o++) succ[o] = best_of(V, po[o] > 0 ? update(P, made[k], a, o).belief : predict(P, made[k], a));
    K.next.push_back(std::move(succ));
  }
  K.evaluate(P, C);
  double kv = -std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < V.size(); k++)
    if (const double x = dot(K.reward[k], P.initial) - lambda * dot(K.cost[k], P.initial); x > kv) kv = x, K.start = k;
  ScalarisedPlan out{V[K.start], dot(K.reward[K.start], P.initial), dot(K.cost[K.start], P.initial), kv, std::move(K)};
  return out;
}

struct ConstrainedSolution {
  double lambda = 0;                 // the multiplier at which the budget binds (0 if it does not)
  ScalarisedPlan low, high;          // the plans on either side: low spends more than the budget, high not more
  double weight = 1;                 // the probability of taking `low` at the start (the rest `high`)
  double reward = 0, cost = 0;       // the mixture's values at b₀
  double dual = 0;                   // the least dual bound found, D(λ) = max (R − λC) + λ c
  int evaluations = 0;
  bool feasible = true;              // false if even the least cost found exceeds the budget
};

// The Lagrangian dual by bisection on λ; `solve(λ)` returns the scalarised plan at λ.
template <class Solve>
ConstrainedSolution lagrangian_dual(Solve&& solve, double budget, double lambda_max, int steps = 60) {
  ConstrainedSolution out;
  auto p0 = solve(0.0);
  out.evaluations = 1;
  out.dual = p0.scalarised + 0.0;
  if (p0.cost <= budget) {
    out.low = out.high = p0, out.weight = 0, out.reward = p0.reward, out.cost = p0.cost, out.lambda = 0;
    return out;
  }
  auto ph = solve(lambda_max);
  out.evaluations++;
  out.dual = std::min(out.dual, ph.scalarised + lambda_max * budget);
  if (ph.cost > budget) {
    out.feasible = false, out.low = out.high = ph, out.weight = 0, out.reward = ph.reward, out.cost = ph.cost, out.lambda = lambda_max;
    return out;
  }
  double lo = 0, hi = lambda_max;
  ScalarisedPlan pl = p0;
  for (int k = 0; k < steps; k++) {
    const double mid = (lo + hi) / 2;
    auto pm = solve(mid);
    out.evaluations++;
    out.dual = std::min(out.dual, pm.scalarised + mid * budget);
    if (pm.cost > budget) lo = mid, pl = std::move(pm);
    else hi = mid, ph = std::move(pm);
  }
  out.lambda = (lo + hi) / 2;
  out.low = pl, out.high = ph;
  out.weight = pl.cost > ph.cost ? (budget - ph.cost) / (pl.cost - ph.cost) : 0;
  out.reward = out.weight * pl.reward + (1 - out.weight) * ph.reward;
  out.cost = out.weight * pl.cost + (1 - out.weight) * ph.cost;
  return out;
}

}  // namespace tyche::decision
