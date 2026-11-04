// Simulation of a finite POMDP under a policy of beliefs, and the Monte Carlo estimate of its value.
//
// An episode draws s₀ from b₀ and, at each step, takes a = π(b), receives R(s, a), draws s' from
// T(· | s, a) and o from Z(· | s', a), and carries the belief by Bayes's rule. Its discounted return
// G = Σ_{t<n} γ^t R(s_t, a_t) is truncated after n steps, which moves the expectation by at most
// γ^n max |R| / (1 − γ). The estimate is the mean of independent episodes, with its standard error;
// episode e draws from its own stream (seed, "epis", e), so the estimate does not depend on the order
// in which episodes are run.

#pragma once

#include <tyche/decision/pomdp.hpp>

#include <calx/random/categorical.hpp>
#include <calx/random/stream.hpp>

#include <cmath>
#include <cstdint>
#include <span>

namespace tyche::decision {

struct Evaluation {
  double mean = 0, error = 0;  // the mean discounted return and its standard error
  double truncation = 0;       // γ^n max |R| / (1 − γ): the bias the truncation allows
  int episodes = 0, steps = 0;
};

// π: a callable taking the belief (std::span<const double>) and returning an action.
template <class Policy>
Evaluation evaluate(const Pomdp& P, Policy&& policy, int episodes, int steps, std::uint64_t seed = 20261104) {
  validate(P);
  const std::size_t S = P.states, O = P.observations;
  Evaluation E;
  E.episodes = episodes, E.steps = steps;
  double sum = 0, sq = 0;
  for (int e = 0; e < episodes; e++) {
    calx::random::Stream rng(seed, 0x65706973u, static_cast<std::uint32_t>(e), 0);  // "epis"
    std::size_t s = calx::random::categorical(rng, P.initial);
    Belief b = P.initial;
    double g = 0, disc = 1;
    for (int t = 0; t < steps; t++) {
      const std::size_t a = policy(std::span<const double>(b));
      g += disc * P.R[a * S + s];
      disc *= P.discount;
      const std::size_t s2 = calx::random::categorical(rng, std::span<const double>(P.T).subspan((a * S + s) * S, S));
      const std::size_t o = calx::random::categorical(rng, std::span<const double>(P.Z).subspan((a * S + s2) * O, O));
      b = update(P, b, a, o).belief;
      s = s2;
    }
    sum += g, sq += g * g;
  }
  E.mean = sum / episodes;
  E.error = episodes > 1 ? std::sqrt(std::max(0.0, (sq - sum * sum / episodes) / (episodes - 1)) / episodes) : 0;
  double rmax = 0;
  for (const double r : P.R) rmax = std::max(rmax, std::fabs(r));
  E.truncation = P.discount < 1 ? std::pow(P.discount, steps) * rmax / (1 - P.discount) : 0;
  return E;
}

}  // namespace tyche::decision
