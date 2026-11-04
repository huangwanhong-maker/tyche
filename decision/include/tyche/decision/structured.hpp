// Structured POMDPs: mixed observability, and many components joined by a budget.
//
// Mixed observability (Ong, Png, Hsu and Lee 2010; Araya-López, Thomas, Buffet and Charpillet 2010). The
// state is a pair (x, y), x seen and y hidden: a MOMDP is the POMDP whose observation is (x', o), and its
// beliefs lie on one slice {x} × Δ(Y). `to_pomdp` writes that POMDP exactly, so that every solver here
// applies; `MomdpBelief` keeps (x, b_Y), the belief in |Y| numbers rather than |X| |Y|. (Solvers whose
// vectors are indexed by x, α_x ∈ ℝ^Y, as Ong et al.'s are, are planned.)
//
// Weakly coupled components (Meuleau, Hauskrecht, Kim, Peshkin, Kaelbling, Dean and Boutilier 1998;
// Adelman and Mersereau 2008, "Relaxations of weakly coupled stochastic dynamic programs", Oper. Res. 56,
// 712–727). N components, each a POMDP of its own whose transitions and observations depend only on its
// own state and action, joined only by a budget: Σ_i c_i(a_i) ≤ B at every step. For every λ ≥ 0 the
// budget relaxed into the reward separates the components, and
//
//   V*(b) ≤ Σ_i V_i^λ(b_i) + λ B / (1 − γ),   V_i^λ the value of component i alone with reward R_i − λ c_i,
//
// since every policy feasible for the budget is feasible for the relaxation, and the relaxed reward adds
// λ (B − Σ c_i(a_i)) ≥ 0 at every step. With each V_i^λ replaced by HSVI's upper bound the inequality
// holds still, and its least value over λ is the bound reported. The policy that plans with it, at each
// step, ranks the components' actions by their gain in the relaxed action values (from HSVI's lower
// bound) per unit of cost, and takes them in that order while the budget allows, is feasible; its value,
// simulated, bounds V* from below in expectation. Together they give an interval for the coupled problem
// that the product (states ∏|S_i|, exponential in N) could not be solved for.

#pragma once

#include <tyche/decision/alpha.hpp>
#include <tyche/decision/point_based.hpp>
#include <tyche/decision/pomdp.hpp>

#include <calx/random/categorical.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace tyche::decision {

// ---- mixed observability --------------------------------------------------------------------------------------------

struct Momdp {
  std::size_t visible = 0, hidden = 0, actions = 0, observations = 0;
  // On the joint state s = x |Y| + y: T((x', y') | (x, y), a), Z(o | (x', y'), a), R((x, y), a), in Pomdp's layout.
  std::vector<double> T, Z, R;
  double discount = 0.95;
  std::size_t x0 = 0;
  std::vector<double> y0;  // the law of the hidden part at the start

  std::size_t states() const { return visible * hidden; }
};

// The POMDP whose observation is (x', o), numbered x' |O| + o.
inline Pomdp to_pomdp(const Momdp& M) {
  const std::size_t S = M.states(), A = M.actions, O = M.observations, Y = M.hidden;
  Pomdp P(S, A, M.visible * O, M.discount);
  P.T = M.T, P.R = M.R;
  for (std::size_t a = 0; a < A; a++)
    for (std::size_t s2 = 0; s2 < S; s2++)
      for (std::size_t o = 0; o < O; o++) P.observe(a, s2, (s2 / Y) * O + o) = M.Z[(a * S + s2) * O + o];
  P.initial.assign(S, 0.0);
  for (std::size_t y = 0; y < Y; y++) P.initial[M.x0 * Y + y] = M.y0[y];
  return P;
}

struct MomdpBelief {
  std::size_t x = 0;
  std::vector<double> y;
};

// The MOMDP's belief after a, seeing x' and o: the law of y' given them.
inline MomdpBelief update(const Momdp& M, const MomdpBelief& b, std::size_t a, std::size_t x2, std::size_t o) {
  const std::size_t S = M.states(), O = M.observations, Y = M.hidden;
  MomdpBelief out{x2, std::vector<double>(Y, 0.0)};
  double total = 0;
  for (std::size_t y = 0; y < Y; y++) {
    if (b.y[y] == 0) continue;
    const std::size_t s = b.x * Y + y;
    for (std::size_t y2 = 0; y2 < Y; y2++) {
      const std::size_t s2 = x2 * Y + y2;
      const double w = b.y[y] * M.T[(a * S + s) * S + s2] * M.Z[(a * S + s2) * O + o];
      out.y[y2] += w, total += w;
    }
  }
  detail::require(total > 0, "an observation of probability zero under the MOMDP's belief");
  for (double& v : out.y) v /= total;
  return out;
}

// ---- weakly coupled components --------------------------------------------------------------------------------------

struct Component {
  const Pomdp* model = nullptr;   // shared among components of one kind
  std::vector<double> cost;       // c(a), one per action, non-negative, per unit of weight
  Belief belief;                  // the component's belief now
  double weight = 1;              // scales the component's rewards and costs (an asset's area)
};

namespace detail {
inline Pomdp relaxed(const Pomdp& P, std::span<const double> cost, double lambda) {
  Pomdp Q = P;
  for (std::size_t a = 0; a < P.actions; a++)
    for (std::size_t s = 0; s < P.states; s++) Q.reward(s, a) -= lambda * cost[a];
  return Q;
}
}  // namespace detail

// Components that share a model and its costs form a group, solved once for each λ. While λ is sought, each group's
// upper bound is its fast informed bound (cheap, and above V* at every belief); at the λ found, HSVI from a few
// representative beliefs of the group, its trials capped, tightens it (its upper bound holds at every belief whether
// or not it has converged), and gives the lower bounds the ranked policy acts on. A component's upper bound is the
// least of these at its belief, its lower bound the greatest. The bound is Σ_i w_i V̄_g(b_i) + λ B / (1 − γ), the
// weights scaling rewards and costs alike.
struct CoupledGroup {
  const Pomdp* model = nullptr;
  std::vector<double> cost;
  Pomdp relaxed;                    // the unit model at the λ kept
  ValueFunction fib, blind;         // its fast informed bound (above V*) and blind policies (below)
  std::vector<HsviResult> solved;   // from each representative belief, at the λ kept

  double upper(std::span<const double> b) const {
    double v = fib.vectors.empty() ? std::numeric_limits<double>::infinity() : fib.value(b);
    for (const auto& H : solved) v = std::min(v, H.upper.value(b));
    return v;
  }
  double lower(std::span<const double> b) const {
    double v = blind.vectors.empty() ? -std::numeric_limits<double>::infinity() : blind.value(b);
    for (const auto& H : solved) v = std::max(v, H.lower.value(b));
    return v;
  }
};

struct CoupledSolution {
  double lambda = 0;                // the multiplier of the least bound found
  double upper = 0;                 // Σ_i w_i V̄_i^λ(b_i) + λ B / (1 − γ)
  std::vector<CoupledGroup> groups;
  std::vector<std::size_t> group_of;  // each component's group
  int evaluations = 0;              // values of λ tried
};

struct CoupledOptions {
  double epsilon = 1e-2;            // HSVI's width for each representative belief, at the λ found
  int max_trials = 300;             // HSVI's trials a representative belief (its bounds hold if it stops short)
  double lambda_max = 0;            // 0: the largest reward span over the cheapest positive cost
  int golden_steps = 24;
  std::size_t representatives = 4;  // beliefs a group is solved from (farthest first)
};

// The Lagrangian bound, minimised over λ ∈ [0, λ_max] by golden section (the exact bound is convex in λ; with
// HSVI's bounds it is so up to their widths, and every value tried is a valid bound).
inline CoupledSolution lagrangian_bound(const std::vector<Component>& C, double budget, const CoupledOptions& opt = {}) {
  detail::require(!C.empty(), "the coupled problem needs components");
  const double gamma = C[0].model->discount;
  // The groups and their representative beliefs.
  CoupledSolution out;
  std::vector<std::vector<Belief>> reps;
  for (const auto& c : C) {
    std::size_t g = 0;
    while (g < out.groups.size() && !(out.groups[g].model == c.model && out.groups[g].cost == c.cost)) g++;
    if (g == out.groups.size()) out.groups.push_back({c.model, c.cost, {}, {}}), reps.push_back({});
    out.group_of.push_back(g);
  }
  for (std::size_t g = 0; g < out.groups.size(); g++) {
    std::vector<const Belief*> mine;
    for (std::size_t i = 0; i < C.size(); i++)
      if (out.group_of[i] == g) mine.push_back(&C[i].belief);
    reps[g].push_back(*mine[0]);
    while (reps[g].size() < opt.representatives) {
      double far = 0;
      const Belief* pick = nullptr;
      for (const auto* b : mine) {
        double d = std::numeric_limits<double>::infinity();
        for (const auto& r : reps[g]) d = std::min(d, l1(*b, r));
        if (d > far) far = d, pick = b;
      }
      if (!pick || far < 0.02) break;
      reps[g].push_back(*pick);
    }
  }
  double lmax = opt.lambda_max;
  if (lmax <= 0) {
    double span = 0, cmin = std::numeric_limits<double>::infinity();
    for (const auto& c : C) {
      span = std::max(span, c.model->reward_span());
      for (const double x : c.cost)
        if (x > 0) cmin = std::min(cmin, x);
    }
    lmax = std::isfinite(cmin) ? 2 * span / cmin : 1;
  }
  const auto at = [&](double lambda, bool keep) {
    std::vector<CoupledGroup> solved = out.groups;
    for (std::size_t g = 0; g < solved.size(); g++) {
      solved[g].relaxed = detail::relaxed(*solved[g].model, solved[g].cost, lambda);
      solved[g].fib = fast_informed_bound(solved[g].relaxed);
      if (!keep) continue;
      solved[g].blind = blind_lower_bound(solved[g].relaxed);
      for (const auto& r : reps[g]) {
        auto Q = solved[g].relaxed;
        Q.initial = r;
        solved[g].solved.push_back(hsvi(Q, {.epsilon = opt.epsilon, .max_trials = opt.max_trials}));
      }
    }
    double total = lambda * budget / (1 - gamma);
    for (std::size_t i = 0; i < C.size(); i++) total += C[i].weight * solved[out.group_of[i]].upper(C[i].belief);
    if (keep) out.groups = std::move(solved), out.lambda = lambda, out.upper = total;
    return total;
  };
  const double phi = (std::sqrt(5.0) - 1) / 2;
  double a = 0, b = lmax, c = b - phi * (b - a), d = a + phi * (b - a), fc = at(c, false), fd = at(d, false);
  out.evaluations = 2;
  for (int k = 0; k < opt.golden_steps; k++, out.evaluations++) {
    if (fc < fd) b = d, d = c, fd = fc, c = b - phi * (b - a), fc = at(c, false);
    else a = c, c = d, fc = fd, d = a + phi * (b - a), fd = at(d, false);
  }
  const double f0 = at(0, false), lam = fc < fd ? c : d;
  out.evaluations += 2;
  at(f0 <= std::min(fc, fd) ? 0.0 : lam, true);
  return out;
}

// The ranked policy: one action per component within the budget. Each component's actions are valued by one step of
// lookahead on its group's lower bound, with the relaxed cost of this step given back (q_i(a) = w_i [Q^λ(b_i, a) +
// λ c(a)]); then the greedy of the multiple-choice knapsack: along each component's upper concave hull of
// (cost, value) from its cheapest action, the increments are taken in order of value per unit of cost while the
// budget allows, a component's later increments skipped once one of its increments does not fit.
inline std::vector<std::size_t> coupled_actions(const CoupledSolution& sol, const std::vector<Component>& C, double budget) {
  const std::size_t N = C.size();
  std::vector<std::size_t> choice(N);
  struct Step {
    double efficiency, dc;
    std::size_t component, from, to;
  };
  std::vector<Step> steps;
  double spent = 0;
  for (std::size_t i = 0; i < N; i++) {
    const auto& G = sol.groups[sol.group_of[i]];
    const auto& P = G.relaxed;
    std::vector<double> q(P.actions), c(P.actions);
    for (std::size_t a = 0; a < P.actions; a++) {
      double x = expected_reward(P, C[i].belief, a);
      const auto po = observation_law(P, C[i].belief, a);
      for (std::size_t o = 0; o < P.observations; o++)
        if (po[o] > 0) x += P.discount * po[o] * G.lower(update(P, C[i].belief, a, o).belief);
      q[a] = C[i].weight * (x + sol.lambda * C[i].cost[a]), c[a] = C[i].weight * C[i].cost[a];
    }
    // The upper concave hull from the cheapest action, ties in cost to the better value.
    std::vector<std::size_t> order(P.actions);
    for (std::size_t a = 0; a < P.actions; a++) order[a] = a;
    std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return c[x] < c[y] || (c[x] == c[y] && q[x] > q[y]); });
    std::size_t at = order[0];
    choice[i] = at, spent += c[at];
    while (true) {
      double best = 0;
      std::size_t next = P.actions;
      for (const std::size_t a : order) {
        if (c[a] < c[at] || q[a] <= q[at] || a == at) continue;
        const double e = c[a] > c[at] ? (q[a] - q[at]) / (c[a] - c[at]) : std::numeric_limits<double>::infinity();
        if (e > best) best = e, next = a;
      }
      if (next == P.actions) break;
      steps.push_back({best, c[next] - c[at], i, at, next});
      at = next;
    }
  }
  std::stable_sort(steps.begin(), steps.end(), [](const Step& x, const Step& y) { return x.efficiency > y.efficiency; });
  std::vector<bool> frozen(N, false);
  for (const auto& st : steps) {
    if (frozen[st.component] || choice[st.component] != st.from) continue;
    if (spent + st.dc > budget + 1e-12) { frozen[st.component] = true; continue; }
    spent += st.dc, choice[st.component] = st.to;
  }
  return choice;
}

// The product of the components with the budget as a constraint on joint actions: exact, for small N and checks.
// Joint states and actions in mixed radix, component 0 fastest; actions over the budget are replaced by the
// cheapest joint action (so that every action of the product is feasible). Rewards are summed; the joint
// observation is the tuple of the components'.
inline Pomdp joint_pomdp(const std::vector<Component>& C, double budget, std::vector<std::vector<std::size_t>>* feasible = nullptr) {
  const std::size_t N = C.size();
  std::size_t S = 1, A = 1, O = 1;
  for (const auto& c : C) S *= c.model->states, A *= c.model->actions, O *= c.model->observations;
  const auto split = [&](std::size_t k, auto size) {
    std::vector<std::size_t> parts(N);
    for (std::size_t i = 0; i < N; i++) parts[i] = k % size(i), k /= size(i);
    return parts;
  };
  const auto sS = [&](std::size_t i) { return C[i].model->states; };
  const auto sA = [&](std::size_t i) { return C[i].model->actions; };
  const auto sO = [&](std::size_t i) { return C[i].model->observations; };
  // The feasible joint actions.
  std::vector<std::vector<std::size_t>> acts;
  for (std::size_t ja = 0; ja < A; ja++) {
    const auto a = split(ja, sA);
    double cost = 0;
    for (std::size_t i = 0; i < N; i++) cost += C[i].cost[a[i]];
    if (cost <= budget + 1e-12) acts.push_back(a);
  }
  detail::require(!acts.empty(), "no joint action fits the budget");
  Pomdp P(S, acts.size(), O, C[0].model->discount);
  for (std::size_t k = 0; k < acts.size(); k++) {
    const auto& a = acts[k];
    for (std::size_t js = 0; js < S; js++) {
      const auto s = split(js, sS);
      double r = 0;
      for (std::size_t i = 0; i < N; i++) r += C[i].model->reward(s[i], a[i]);
      P.reward(js, k) = r;
      for (std::size_t js2 = 0; js2 < S; js2++) {
        const auto s2 = split(js2, sS);
        double t = 1;
        for (std::size_t i = 0; i < N && t > 0; i++) t *= C[i].model->transition(a[i], s[i], s2[i]);
        P.transition(k, js, js2) = t;
      }
    }
    for (std::size_t js2 = 0; js2 < S; js2++) {
      const auto s2 = split(js2, sS);
      for (std::size_t jo = 0; jo < O; jo++) {
        const auto o = split(jo, sO);
        double z = 1;
        for (std::size_t i = 0; i < N && z > 0; i++) z *= C[i].model->observe(a[i], s2[i], o[i]);
        P.observe(k, js2, jo) = z;
      }
    }
  }
  for (std::size_t js = 0; js < S; js++) {
    const auto s = split(js, sS);
    double p = 1;
    for (std::size_t i = 0; i < N; i++) p *= C[i].belief[s[i]];
    P.initial[js] = p;
  }
  if (feasible) *feasible = acts;
  return P;
}

}  // namespace tyche::decision
