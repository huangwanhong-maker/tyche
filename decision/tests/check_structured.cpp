// Checks of structured POMDPs (decision/structured.hpp).
//
// - Mixed observability: a random MOMDP written as a POMDP is a POMDP; along random histories its beliefs stay on
//   one visible slice and equal the MOMDP's own update of the hidden part.
// - Weakly coupled components: three deteriorating assets (good, worn, failed; do nothing, inspect, repair) under a
//   budget of one repair's cost a step. The product, 27 states, is solved by HSVI. The Lagrangian bound lies above
//   its upper bound's lower end (it bounds V*), and below the unconstrained sum (λ = 0); the ranked policy keeps to
//   the budget at every step, and its simulated value lies below the product's upper bound and within the bound's
//   interval.

#include <tyche/decision/point_based.hpp>
#include <tyche/decision/structured.hpp>

#include <calx/random/categorical.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace td = tyche::decision;

namespace {

std::vector<double> random_law(calx::random::Stream& rng, std::size_t n) {
  std::vector<double> p(n);
  double t = 0;
  for (auto& x : p) t += (x = rng.exponential());
  for (auto& x : p) x /= t;
  return p;
}

// An asset: good (0), worn (1), failed (2). Doing nothing (0) or inspecting (1) lets it wear; repairing (2) makes
// it good. Inspecting sees its state with probability 0.9 (else either other state, equally); the other actions
// see nothing (one uninformative symbol of the three, uniformly). Service: good 1, worn 0.6, failed −1; inspecting
// costs 0.05 and repairing 0.5 of reward, besides the budget's costs 0.2 and 1.
td::Pomdp asset(double wear) {
  td::Pomdp P(3, 3, 3, 0.9);
  const double service[3] = {1, 0.6, -1};
  for (std::size_t a = 0; a < 3; a++)
    for (std::size_t s = 0; s < 3; s++) {
      if (a == 2) P.transition(a, s, 0) = 1;
      else if (s == 0) P.transition(a, s, 0) = 1 - wear, P.transition(a, s, 1) = wear;
      else if (s == 1) P.transition(a, s, 1) = 1 - 2 * wear, P.transition(a, s, 2) = 2 * wear;
      else P.transition(a, s, 2) = 1;
      for (std::size_t o = 0; o < 3; o++) P.observe(a, s, o) = a == 1 ? (o == s ? 0.9 : 0.05) : 1.0 / 3;
      P.reward(s, a) = service[s] - (a == 1 ? 0.05 : a == 2 ? 0.5 : 0.0);
    }
  P.initial = {1, 0, 0};
  return P;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- mixed observability ------------------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261104, 0x6d6f6d64u, 0, 0);  // "momd"
    td::Momdp M;
    M.visible = 2, M.hidden = 3, M.actions = 2, M.observations = 2, M.discount = 0.9;
    const std::size_t S = M.states();
    M.T.assign(M.actions * S * S, 0.0), M.Z.assign(M.actions * S * M.observations, 0.0), M.R.assign(M.actions * S, 0.0);
    for (std::size_t a = 0; a < M.actions; a++)
      for (std::size_t s0 = 0; s0 < S; s0++) {
        const auto t = random_law(rng, S), z = random_law(rng, M.observations);
        for (std::size_t k = 0; k < S; k++) M.T[(a * S + s0) * S + k] = t[k];
        for (std::size_t o = 0; o < M.observations; o++) M.Z[(a * S + s0) * M.observations + o] = z[o];
        M.R[a * S + s0] = rng.uniform(-1, 1);
      }
    M.x0 = 1, M.y0 = random_law(rng, M.hidden);
    const auto P = td::to_pomdp(M);
    td::validate(P);
    bool sliced = true;
    double worst = 0;
    for (int run = 0; run < 50; run++) {
      td::Belief b = P.initial;
      td::MomdpBelief m{M.x0, M.y0};
      for (int t = 0; t < 8; t++) {
        const std::size_t a = static_cast<std::size_t>(rng.below(M.actions));
        const auto po = td::observation_law(P, b, a);
        const std::size_t jo = calx::random::categorical(rng, po);
        b = td::update(P, b, a, jo).belief;
        m = td::update(M, m, a, jo / M.observations, jo % M.observations);
        for (std::size_t s2 = 0; s2 < S; s2++) {
          const bool on = s2 / M.hidden == m.x;
          sliced = sliced && (on || b[s2] == 0);
          if (on) worst = std::max(worst, std::fabs(b[s2] - m.y[s2 % M.hidden]));
        }
      }
    }
    s.require("a MOMDP's beliefs, as a POMDP's, stay on the slice of what is seen", sliced);
    s.check("... and equal the MOMDP's update of the hidden part, along 50 random histories", worst, 1e-12);
  }

  // ---- weakly coupled components --------------------------------------------------------------------------------------
  {
    const td::Pomdp a1 = asset(0.10), a2 = asset(0.15), a3 = asset(0.20);
    const std::vector<double> cost{0, 0.2, 1};
    std::vector<td::Component> C = {{&a1, cost, {1, 0, 0}, 1}, {&a2, cost, {0.5, 0.5, 0}, 1}, {&a3, cost, {0.2, 0.5, 0.3}, 1}};
    const double budget = 1;
    const auto J = td::joint_pomdp(C, budget);
    td::validate(J);
    const auto t0 = std::chrono::steady_clock::now();
    const auto joint = td::hsvi(J, {.epsilon = 0.5});
    const double t_joint = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto L = td::lagrangian_bound(C, budget, {.epsilon = 0.05});
    const double t_bound = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - t_joint;
    std::printf("(the product by HSVI in %.1f s, %d trials; the Lagrangian bound in %.1f s, %d values of λ)\n", t_joint, joint.trials, t_bound, L.evaluations);
    double free = 0;  // λ = 0: each asset alone, the budget ignored
    for (const auto& c : C) {
      auto Q = *c.model;
      Q.initial = c.belief;
      free += td::hsvi(Q, {.epsilon = 0.05}).upper_value;
    }
    s.require("the joint product: " + std::to_string(J.states) + " states, " + std::to_string(J.actions) + " joint actions within the budget; HSVI's interval",
              joint.converged, "[" + std::to_string(joint.lower_value) + ", " + std::to_string(joint.upper_value) + "]");
    s.require("the Lagrangian bound lies above V* (above the product's lower bound)", L.upper >= joint.lower_value - 1e-9,
              std::to_string(L.upper) + " at λ = " + std::to_string(L.lambda) + " against [" + std::to_string(joint.lower_value) + ", " + std::to_string(joint.upper_value) + "]");
    s.require("... and no higher than the budget ignored (λ = 0)", L.upper <= free + 0.15, std::to_string(L.upper) + " against " + std::to_string(free));

    // The ranked policy, simulated on the true assets with exact beliefs.
    const int episodes = 1000, steps = 120;
    double sum = 0, sq = 0;
    bool within = true;
    for (int e = 0; e < episodes; e++) {
      calx::random::Stream rng(20261104, 0x636f7570u, static_cast<std::uint32_t>(e), 0);  // "coup"
      auto beliefs = C;
      std::vector<std::size_t> state(C.size());
      for (std::size_t i = 0; i < C.size(); i++) state[i] = calx::random::categorical(rng, C[i].belief);
      double g = 0, disc = 1;
      for (int t = 0; t < steps; t++) {
        const auto act = td::coupled_actions(L, beliefs, budget);
        double spent = 0;
        for (std::size_t i = 0; i < C.size(); i++) {
          const auto& P = *C[i].model;
          spent += cost[act[i]];
          g += disc * P.reward(state[i], act[i]);
          const std::size_t s2 = calx::random::categorical(rng, std::span<const double>(P.T).subspan((act[i] * 3 + state[i]) * 3, 3));
          const std::size_t o = calx::random::categorical(rng, std::span<const double>(P.Z).subspan((act[i] * 3 + s2) * 3, 3));
          beliefs[i].belief = td::update(P, beliefs[i].belief, act[i], o).belief;
          state[i] = s2;
        }
        within = within && spent <= budget + 1e-12;
        disc *= 0.9;
      }
      sum += g, sq += g * g;
    }
    const double mean = sum / episodes, se = std::sqrt((sq - sum * sum / episodes) / (episodes - 1) / episodes);
    s.require("the ranked policy keeps to the budget at every step", within);
    s.require("its simulated value lies below the product's upper bound (and the Lagrangian bound)",
              mean <= joint.upper_value + 4 * se && mean <= L.upper + 4 * se,
              std::to_string(mean) + " ± " + std::to_string(se));
    s.require("... and within 10 % of the product's lower bound: the interval [policy, bound] is informative",
              mean >= joint.lower_value - 0.1 * std::fabs(joint.lower_value) - 4 * se,
              std::to_string(mean) + " against " + std::to_string(joint.lower_value));
  }
  return s.done();
}
