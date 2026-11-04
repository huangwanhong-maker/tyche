// Checks of constrained POMDPs (decision/constrained.hpp) on the tiger problem with a cost for listening.
//
// - Over a horizon of 3 (undiscounted), against the linear programme over mixtures of all 2187 policy trees (calx's
//   simplex): the backups at every reachable belief give, for each λ tried, the best tree for R − λC; the dual's
//   mixture of two plans spends the budget of 1.5 listens exactly and attains the programme's optimum; the least
//   dual bound equals it (strong duality).
// - At discount 0.95, with a budget of 2 discounted listens: the mixture of two finite-state controllers spends the
//   budget, its reward no more than the unconstrained optimum and the dual bound (HSVI's upper bound of R − λC, plus
//   λ c); simulated node by node, its cost and reward equal the controllers' exact values within four standard
//   errors.

#include <tyche/decision/constrained.hpp>
#include <tyche/decision/multiagent.hpp>
#include <tyche/decision/point_based.hpp>

#include <calx/linalg/rectangular.hpp>
#include <calx/optimisation/linear_programming.hpp>
#include <calx/random/categorical.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace td = tyche::decision;

namespace {

td::Pomdp tiger(double gamma) {
  td::Pomdp P(2, 3, 2, gamma);
  for (std::size_t s = 0; s < 2; s++) {
    P.transition(0, s, s) = 1;
    P.observe(0, s, s) = 0.85, P.observe(0, s, 1 - s) = 0.15;
    P.reward(s, 0) = -1;
    for (std::size_t a = 1; a <= 2; a++) {
      P.transition(a, s, 0) = P.transition(a, s, 1) = 0.5;
      P.observe(a, s, 0) = P.observe(a, s, 1) = 0.5;
    }
  }
  P.reward(0, 1) = -100, P.reward(1, 1) = 10;
  P.reward(0, 2) = 10, P.reward(1, 2) = -100;
  return P;
}

// V_R and V_C of a tree from b₀.
std::pair<double, double> tree_values(const td::Pomdp& P, const std::vector<double>& C, const td::PolicyTree& q) {
  std::function<std::pair<double, double>(const std::vector<double>&, std::size_t, int)> v = [&](const std::vector<double>& b, std::size_t node, int depth) {
    const std::size_t a = q.action[node];
    double r = td::expected_reward(P, b, a), c = 0;
    for (std::size_t s = 0; s < P.states; s++) c += b[s] * C[a * P.states + s];
    if (depth + 1 < q.depth) {
      const auto po = td::observation_law(P, b, a);
      for (std::size_t o = 0; o < P.observations; o++) {
        if (po[o] <= 0) continue;
        const auto [rr, cc] = v(td::update(P, b, a, o).belief, q.child(node, o), depth + 1);
        r += P.discount * po[o] * rr, c += P.discount * po[o] * cc;
      }
    }
    return std::pair<double, double>{r, c};
  };
  return v(P.initial, 0, 0);
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- a finite horizon, against the programme over trees -----------------------------------------------------------
  {
    const auto P = tiger(1.0);
    std::vector<double> C(P.actions * P.states, 0.0);
    for (std::size_t st = 0; st < P.states; st++) C[0 * P.states + st] = 1;  // a listen costs 1
    const int h = 3;
    const double budget = 1.5;
    const auto trees = td::all_trees(P.actions, P.observations, h);
    std::vector<double> vr, vc;
    for (const auto& q : trees) {
      const auto [r, c] = tree_values(P, C, q);
      vr.push_back(r), vc.push_back(c);
    }
    calx::optimisation::LinearProgram lp;
    lp.c.resize(trees.size());
    for (std::size_t k = 0; k < trees.size(); k++) lp.c[k] = -vr[k];
    lp.a_ub = calx::linalg::Rectangular(1, trees.size());
    for (std::size_t k = 0; k < trees.size(); k++) lp.a_ub(0, k) = vc[k];
    lp.b_ub = {budget};
    lp.a_eq = calx::linalg::Rectangular(1, trees.size());
    for (std::size_t k = 0; k < trees.size(); k++) lp.a_eq(0, k) = 1;
    lp.b_eq = {1.0};
    const auto opt = calx::optimisation::simplex(lp);
    s.require("the programme over mixtures of the 2187 trees solves", opt.status == calx::optimisation::LpStatus::optimal);
    const double best = -opt.value;

    const auto levels = td::reachable_beliefs(P, h);
    double worst = 0;
    for (const double lambda : {0.0, 2.0, 7.5, 30.0}) {
      double brute = -1e300;
      for (std::size_t k = 0; k < trees.size(); k++) brute = std::max(brute, vr[k] - lambda * vc[k]);
      worst = std::max(worst, std::fabs(td::scalarised_finite(P, C, lambda, h, levels).scalarised - brute));
    }
    s.check("backups at every reachable belief give the best tree for R − λC (λ = 0, 2, 7.5, 30)", worst, 1e-9);
    const auto sol = td::lagrangian_dual([&](double lambda) { return td::scalarised_finite(P, C, lambda, h, levels); }, budget, 200);
    s.check("the dual's mixture spends the budget of 1.5 listens", std::fabs(sol.cost - budget), 1e-9);
    s.check("... and attains the programme's optimum " + std::to_string(best), std::fabs(sol.reward - best), 1e-6);
    s.check("the least dual bound equals it (strong duality)", std::fabs(sol.dual - best), 1e-6);
  }

  // ---- discount 0.95 ------------------------------------------------------------------------------------------------
  {
    const auto P = tiger(0.95);
    std::vector<double> C(P.actions * P.states, 0.0);
    for (std::size_t st = 0; st < P.states; st++) C[st] = 1;
    const double budget = 2.0;
    calx::random::Stream rng(20261104, 0x63707362u, 0, 0);  // "cpsb"
    const auto B = td::random_beliefs(P, 300, 20, rng);
    const auto sol = td::lagrangian_dual([&](double lambda) { return td::scalarised_infinite(P, C, lambda, B); }, budget, 200, 30);
    const auto free = td::hsvi(P, {.epsilon = 1e-2});
    s.check("the mixture's planned cost is the budget of 2 discounted listens", std::fabs(sol.cost - budget), 1e-6);
    // An upper bound on the constrained optimum: HSVI's upper bound of R − λC at the λ found, plus λ c.
    auto scal = P;
    for (std::size_t a = 0; a < P.actions; a++)
      for (std::size_t st = 0; st < P.states; st++) scal.reward(st, a) -= sol.lambda * C[a * P.states + st];
    const double dual = td::hsvi(scal, {.epsilon = 1e-2}).upper_value + sol.lambda * budget;
    s.require("its reward lies below the unconstrained optimum and the dual bound",
              sol.reward <= free.upper_value + 1e-9 && sol.reward <= dual + 1e-9,
              std::to_string(sol.reward) + " against " + std::to_string(free.upper_value) + " and " + std::to_string(dual));
    // Simulated: the plan is drawn at the start (low with its weight), then followed greedily on its set.
    const int episodes = 20000, steps = 300;
    double sr = 0, sr2 = 0, sc = 0, sc2 = 0;
    for (int e = 0; e < episodes; e++) {
      calx::random::Stream w(20261104, 0x63706570u, static_cast<std::uint32_t>(e), 0);  // "cpep"
      const bool low = w.uniform() < sol.weight;
      const auto& plan = low ? sol.low : sol.high;
      std::size_t st = calx::random::categorical(w, P.initial), node = plan.controller.start;
      double r = 0, c = 0, disc = 1;
      for (int t = 0; t < steps; t++) {
        const std::size_t a = plan.controller.action[node];
        r += disc * P.reward(st, a), c += disc * C[a * P.states + st], disc *= P.discount;
        const std::size_t s2 = calx::random::categorical(w, std::span<const double>(P.T).subspan((a * 2 + st) * 2, 2));
        const std::size_t o = calx::random::categorical(w, std::span<const double>(P.Z).subspan((a * 2 + s2) * 2, 2));
        node = plan.controller.next[node][o], st = s2;
      }
      sr += r, sr2 += r * r, sc += c, sc2 += c * c;
    }
    const double mr = sr / episodes, mc = sc / episodes;
    const double er = std::sqrt((sr2 - sr * sr / episodes) / (episodes - 1) / episodes), ec = std::sqrt((sc2 - sc * sc / episodes) / (episodes - 1) / episodes);
    const double trunc = std::pow(P.discount, steps) * 100 / (1 - P.discount);
    s.require("simulated, the mixture's discounted listens equal the budget within four standard errors",
              std::fabs(mc - budget) <= 4 * ec + trunc / 100, std::to_string(mc) + " ± " + std::to_string(ec));
    s.require("... and its reward the controllers' exact values", std::fabs(mr - sol.reward) <= 4 * er + trunc,
              std::to_string(mr) + " ± " + std::to_string(er) + " against " + std::to_string(sol.reward));
  }
  return s.done();
}
