// Online planning in POMDPs by Monte Carlo tree search: generative models, particle beliefs and POMCP.
//
// A generative model replaces the tables: it draws a first state, and from a state and an action draws
// the next state, an observation and a reward. States may be of any type (a district's cells, an
// agent-based model's configuration); observations are keys (64-bit integers); actions are numbered.
//
//   struct Model {
//     using State = ...;
//     std::size_t actions() const;
//     double discount() const;
//     State initial(calx::random::Stream&) const;
//     Step<State> step(const State&, std::size_t action, calx::random::Stream&) const;
//   };
//
// POMCP (Silver and Veness 2010, "Monte-Carlo planning in large POMDPs", NIPS 23, 2164–2172) grows a tree
// of histories (action and observation nodes) by simulations from states drawn from the belief, chooses
// actions in the tree by UCB1 (Q + c √(ln N / N_a); Kocsis and Szepesvári 2006; c by default the span of one
// step's rewards, where Silver and Veness took the range of returns seen), leaves it by a rollout
// policy, and backs up discounted returns; the states that reach a node are its belief's particles.
// After acting and observing, the subtree of the history taken becomes the new root, its particles the
// new belief. Silver and Veness prove that, for a POMDP whose observations are finite, the value at the
// root converges to the optimal value as the simulations grow (their Theorem 1); with finitely many
// simulations it is an estimate, not a bound.
//
// Particle deprivation: when too few particles reach the new root, it is refilled by simulating from the
// old root's particles and keeping the successors whose observation matches (rejection), up to a limit;
// a belief no particle reaches is reported, never invented.

#pragma once

#include <calx/random/categorical.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tyche::decision {

template <class State>
struct Step {
  State next;
  std::uint64_t observation = 0;
  double reward = 0;
};

struct PomcpOptions {
  int simulations = 10000;       // per decision
  double exploration = 0;        // c of UCB1; 0: the span of one step's rewards, R_max − R_min
  double reward_span = 1;        // R_max − R_min, the default c
  int depth = 0;                 // 0: where γ^depth falls below 10⁻³
  int rollout_depth = 0;         // steps of a rollout from a new node; 0: to the depth
  std::size_t particles = 1000;  // the belief at the root, kept or refilled to this many
  int refill_attempts = 100000;  // rejection steps when refilling
  std::uint64_t seed = 20261104;
};

template <class Model>
class Pomcp {
 public:
  using State = typename Model::State;

  Pomcp(const Model& model, PomcpOptions options = {}) : model_(model), opt_(options), rng_(options.seed, 0x706f6d63u, 0, 0) {  // "pomc"
    if (opt_.depth <= 0) opt_.depth = model_.discount() < 1 ? static_cast<int>(std::ceil(std::log(1e-3) / std::log(model_.discount()))) : 100;
    if (opt_.exploration <= 0) opt_.exploration = opt_.reward_span;
    root_ = std::make_unique<VNode>();
    for (std::size_t k = 0; k < opt_.particles; k++) root_->particles.push_back(model_.initial(rng_));
  }

  // Search from the current belief, and return the action of greatest value at the root.
  std::size_t plan() {
    if (root_->particles.empty()) throw std::runtime_error("tyche::decision::Pomcp: the belief has no particles left");
    for (int n = 0; n < opt_.simulations; n++) {
      const State s = root_->particles[rng_.below(root_->particles.size())];
      simulate(s, *root_, 0);
    }
    return best_action(*root_);
  }

  // The root's action values (their counts, when wanted), after plan().
  std::vector<double> values() const {
    std::vector<double> q(model_.actions(), -std::numeric_limits<double>::infinity());
    for (std::size_t a = 0; a < root_->actions.size(); a++)
      if (root_->actions[a].visits) q[a] = root_->actions[a].value;
    return q;
  }

  // Act a, observe o: the subtree of (a, o) becomes the root, its particles the belief, refilled by rejection.
  // Returns false if no particle could be found for the observation (the belief is then empty).
  bool advance(std::size_t a, std::uint64_t o) {
    std::unique_ptr<VNode> next;
    if (a < root_->actions.size()) {
      auto& children = root_->actions[a].children;
      if (auto it = children.find(o); it != children.end()) next = std::move(it->second);
    }
    if (!next) next = std::make_unique<VNode>();
    std::vector<State> old = std::move(root_->particles);
    for (int k = 0; k < opt_.refill_attempts && next->particles.size() < opt_.particles && !old.empty(); k++) {
      const State& s = old[rng_.below(old.size())];
      auto st = model_.step(s, a, rng_);
      if (st.observation == o) next->particles.push_back(std::move(st.next));
    }
    root_ = std::move(next);
    return !root_->particles.empty();
  }

  const std::vector<State>& belief() const { return root_->particles; }

  // A value for the states at new nodes in place of a rollout (a known policy's value, a bound): the leaf
  // evaluation of Monte Carlo tree search. Random rollouts are unbiased for the random policy but can be so noisy
  // that the tree learns nothing; a leaf value below V* keeps the search's estimates conservative.
  void set_leaf(std::function<double(const State&)> leaf) { leaf_ = std::move(leaf); }

 private:
  struct VNode;
  struct QNode {
    int visits = 0;
    double value = 0;
    std::unordered_map<std::uint64_t, std::unique_ptr<VNode>> children;
  };
  struct VNode {
    int visits = 0;
    std::vector<QNode> actions;  // empty until expanded
    std::vector<State> particles;
  };

  std::size_t best_action(const VNode& v) const {
    std::size_t best = 0;
    double top = -std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < v.actions.size(); a++)
      if (v.actions[a].visits > 0 && v.actions[a].value > top) top = v.actions[a].value, best = a;
    return best;
  }

  double rollout(State s, int depth) {
    if (leaf_) return leaf_(s);
    double g = 0, disc = 1;
    const int last = opt_.rollout_depth > 0 ? std::min(opt_.depth, depth + opt_.rollout_depth) : opt_.depth;
    for (int d = depth; d < last; d++) {
      auto st = model_.step(s, static_cast<std::size_t>(rng_.below(model_.actions())), rng_);
      g += disc * st.reward;
      disc *= model_.discount();
      s = std::move(st.next);
    }
    return g;
  }

  double simulate(const State& s, VNode& v, int depth) {
    if (depth >= opt_.depth) return 0;
    if (v.actions.empty()) {
      v.actions.resize(model_.actions());
      v.visits = 1;
      return rollout(s, depth);
    }
    // UCB1; an untried action first.
    std::size_t a = 0;
    double top = -std::numeric_limits<double>::infinity();
    const double logn = std::log(static_cast<double>(std::max(1, v.visits)));
    for (std::size_t k = 0; k < v.actions.size(); k++) {
      const auto& q = v.actions[k];
      const double u = q.visits == 0 ? std::numeric_limits<double>::infinity() : q.value + opt_.exploration * std::sqrt(logn / q.visits);
      if (u > top) top = u, a = k;
    }
    auto st = model_.step(s, a, rng_);
    auto& q = v.actions[a];
    auto& child = q.children[st.observation];
    if (!child) child = std::make_unique<VNode>();
    child->particles.push_back(st.next);
    const double g = st.reward + model_.discount() * simulate(st.next, *child, depth + 1);
    v.visits++;
    q.visits++;
    q.value += (g - q.value) / q.visits;
    return g;
  }

  const Model& model_;
  PomcpOptions opt_;
  calx::random::Stream rng_;
  std::unique_ptr<VNode> root_;
  std::function<double(const State&)> leaf_;
};

// A finite POMDP as a generative model (for checks of the online solvers against the exact ones).
template <class Tables>
struct TabularModel {
  using State = std::size_t;
  const Tables& P;
  std::size_t actions() const { return P.actions; }
  double discount() const { return P.discount; }
  State initial(calx::random::Stream& rng) const { return calx::random::categorical(rng, P.initial); }
  Step<State> step(const State& s, std::size_t a, calx::random::Stream& rng) const {
    const std::size_t S = P.states, O = P.observations;
    Step<State> out;
    out.reward = P.R[a * S + s];
    out.next = calx::random::categorical(rng, std::span<const double>(P.T).subspan((a * S + s) * S, S));
    out.observation = calx::random::categorical(rng, std::span<const double>(P.Z).subspan((a * S + out.next) * O, O));
    return out;
  }
};

}  // namespace tyche::decision
