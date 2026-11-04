// Checks of decentralised POMDPs (decision/multiagent.hpp) on Dec-Tiger.
//
// - Dec-Tiger (Nair, Tambe, Yokoo, Pynadath and Marsella 2003), with its published optimal values over the
//   horizons 2 and 3, −4.000000 and 5.190812 (Oliehoek, Spaan, Amato and Whiteson 2013, Table 2): by brute
//   force over every pair of trees at horizon 2, and at horizon 3 by every tree of one agent against the
//   other's exact best response.
// - A best response is the best of all the agent's trees (brute force over the 2187 trees of horizon 3), and
//   never worse than the tree it replaces.
// - JESP with restarts finds the optimum at horizons 2 and 3; at horizon 4 its value does not exceed the
//   published optimum 4.802755, and it is an equilibrium: no agent's best response improves it.

#include <tyche/decision/multiagent.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace td = tyche::decision;

namespace {

// Two agents, each hearing the tiger with probability 0.85 when both listen (0); opening the left (1) or right
// (2) door resets the problem. Rewards: both listen −2; both open the tiger's door −50, the other door +20;
// different doors −100; one listens while the other opens the tiger's door −101, the other door +9.
td::DecPomdp dec_tiger() {
  td::DecPomdp D;
  D.states = 2, D.actions = {3, 3}, D.observations = {2, 2}, D.discount = 1, D.initial = {0.5, 0.5};
  const std::size_t S = 2, JA = 9, JO = 4;
  D.T.assign(JA * S * S, 0.0), D.Z.assign(JA * S * JO, 0.0), D.R.assign(JA * S, 0.0);
  for (std::size_t a0 = 0; a0 < 3; a0++)
    for (std::size_t a1 = 0; a1 < 3; a1++) {
      const std::size_t ja = a0 + 3 * a1;
      const bool listen = a0 == 0 && a1 == 0;
      for (std::size_t s = 0; s < S; s++) {
        // The door with the tiger is 1 (left) in state 0 and 2 (right) in state 1.
        const std::size_t tiger = s == 0 ? 1 : 2;
        double r;
        if (listen) r = -2;
        else if (a0 != 0 && a1 != 0) r = a0 != a1 ? -100 : (a0 == tiger ? -50 : 20);
        else r = (a0 == 0 ? a1 : a0) == tiger ? -101 : 9;
        D.R[ja * S + s] = r;
        for (std::size_t s2 = 0; s2 < S; s2++) D.T[(ja * S + s) * S + s2] = listen ? (s2 == s ? 1.0 : 0.0) : 0.5;
        for (std::size_t o0 = 0; o0 < 2; o0++)
          for (std::size_t o1 = 0; o1 < 2; o1++) {
            const double p0 = o0 == s ? 0.85 : 0.15, p1 = o1 == s ? 0.85 : 0.15;
            D.Z[(ja * S + s) * JO + o0 + 2 * o1] = listen ? p0 * p1 : 0.25;
          }
      }
    }
  return D;
}

td::JointPolicy best_by_responses(const td::DecPomdp& D, int horizon) {
  td::JointPolicy best;
  for (const auto& t : td::all_trees(D.actions[0], D.observations[0], horizon)) {
    const auto br = td::best_response(D, {t, t}, 1);
    best.evaluated++;
    if (br.value > best.value) best = br;
  }
  return best;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto D = dec_tiger();
  td::validate(D);

  const auto e2 = td::exhaustive(D, 2);
  s.check("Dec-Tiger, horizon 2, by brute force over all 729 pairs of trees: −4.000000", std::fabs(e2.value - (-4.0)), 1e-6);
  const auto r2 = best_by_responses(D, 2);
  s.check("... and by every tree of one agent against the other's best response", std::fabs(r2.value - e2.value), 1e-12);
  const auto r3 = best_by_responses(D, 3);
  s.check("Dec-Tiger, horizon 3: 5.190812 (Oliehoek et al. 2013, Table 2)", std::fabs(r3.value - 5.190812), 1e-6);
  s.check("... the joint policy found has that value", std::fabs(td::value(D, r3.trees) - r3.value), 1e-12);

  {
    // A best response is the best of all the agent's trees, and never worse than the tree it replaces.
    calx::random::Stream rng(20261104, 0x6272u, 0, 0);
    td::PolicyTree q0{2, 3, std::vector<std::size_t>(td::PolicyTree::nodes(2, 3))}, q1 = q0;
    for (auto& a : q0.action) a = static_cast<std::size_t>(rng.below(3));
    for (auto& a : q1.action) a = static_cast<std::size_t>(rng.below(3));
    const auto br = td::best_response(D, {q0, q1}, 1);
    double brute = -1e300;
    for (const auto& t : td::all_trees(3, 2, 3)) brute = std::max(brute, td::value(D, {q0, t}));
    s.check("a best response at horizon 3 is the best of the agent's 2187 trees", std::fabs(br.value - brute), 1e-9);
    s.require("... never worse than the tree it replaces", br.value >= td::value(D, {q0, q1}) - 1e-12);
    s.check("... and its tree has the value it reports", std::fabs(td::value(D, br.trees) - br.value), 1e-9);
  }

  const auto j2 = td::jesp(D, 2), j3 = td::jesp(D, 3);
  s.check("JESP with 20 restarts finds the optimum at horizon 2", std::fabs(j2.value - e2.value), 1e-9);
  s.check("... and at horizon 3", std::fabs(j3.value - r3.value), 1e-6);
  const auto j4 = td::jesp(D, 4);
  bool equilibrium = true;
  for (std::size_t i = 0; i < 2; i++) equilibrium = equilibrium && td::best_response(D, j4.trees, i).value <= j4.value + 1e-9;
  s.require("JESP at horizon 4: no better than the published optimum 4.802755", j4.value <= 4.802755 + 1e-6, std::to_string(j4.value));
  s.require("... and an equilibrium: neither agent's best response improves it", equilibrium);
  return s.done();
}
