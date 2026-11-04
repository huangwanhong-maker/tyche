// Checks of online planning (decision/online.hpp) on the tiger problem, γ = 0.95, against HSVI's interval.
//
// - From b₀ POMCP chooses to listen, as the optimal policy does.
// - After listening and hearing the tiger on the left, its particles hold the tiger on the left in the
//   proportion of Bayes's rule, 0.85, within three binomial standard errors.
// - Played online for 50 episodes of 150 steps (the truncation then moves the value by at most 0.9), with the blind
//   bound's value as its leaf evaluation, it earns within four standard errors of the optimal value's interval, and
//   never above it. (With uniformly random rollouts the leaf estimates, near −600 with a spread of hundreds, swamp
//   the differences between actions, and 2000 simulations a step learn little: −17 ± 7 in 40 episodes.)

#include <tyche/decision/online.hpp>
#include <tyche/decision/point_based.hpp>

#include <calx/random/categorical.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
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

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto P = tiger(0.95);
  const td::TabularModel<td::Pomdp> M{P};
  const auto hs = td::hsvi(P, {.epsilon = 1e-2});

  {
    td::Pomcp<td::TabularModel<td::Pomdp>> planner(M, {.simulations = 20000, .reward_span = P.reward_span(), .particles = 4000});
    const std::size_t a = planner.plan();
    s.require("from b₀ POMCP listens, as the optimal policy does", a == 0, std::to_string(a));
    planner.advance(0, 0);
    double left = 0;
    for (const auto x : planner.belief()) left += x == 0;
    const double n = static_cast<double>(planner.belief().size()), f = left / n;
    s.check("after hearing the tiger on the left, the particles hold it there in Bayes's proportion 0.85 (in binomial standard errors)",
            std::fabs(f - 0.85) / std::sqrt(0.85 * 0.15 / n), 3);
  }

  // Online play: the planner advanced on each true observation.
  const int episodes = 50, steps = 150;
  const auto blind = td::blind_lower_bound(P);
  double sum = 0, sq = 0;
  for (int e = 0; e < episodes; e++) {
    calx::random::Stream world(20261104, 0x776f726cu, static_cast<std::uint32_t>(e), 0);  // "worl"
    td::Pomcp<td::TabularModel<td::Pomdp>> planner(M, {.simulations = 2000, .reward_span = P.reward_span(), .particles = 1000, .seed = 1000u + static_cast<std::uint64_t>(e)});
    planner.set_leaf([&](const std::size_t& x) { double v = -1e300; for (const auto& b : blind.vectors) v = std::max(v, b.values[x]); return v; });
    std::size_t state = calx::random::categorical(world, P.initial);
    double g = 0, disc = 1;
    for (int t = 0; t < steps; t++) {
      const std::size_t a = planner.plan();
      g += disc * P.reward(state, a), disc *= P.discount;
      const auto st = M.step(state, a, world);
      state = st.next;
      if (!planner.advance(a, st.observation)) break;
    }
    sum += g, sq += g * g;
  }
  const double mean = sum / episodes, se = std::sqrt((sq - sum * sum / episodes) / (episodes - 1) / episodes);
  const double truncation = std::pow(P.discount, steps) * 100 / (1 - P.discount);
  s.require("POMCP online earns the optimal value within four standard errors and the truncation",
            mean >= hs.lower_value - 4 * se - truncation && mean <= hs.upper_value + 4 * se,
            std::to_string(mean) + " ± " + std::to_string(se) + " against [" + std::to_string(hs.lower_value) + ", " + std::to_string(hs.upper_value) + "], truncation " + std::to_string(truncation));
  return s.done();
}
