// Decentralised POMDPs over a finite horizon: joint policies as trees, their exact values, exhaustive search,
// best responses and JESP.
//
// A Dec-POMDP (Bernstein, Givan, Immerman and Zilberstein 2002; Oliehoek and Amato 2016) is a POMDP whose
// action is a tuple (a_1, …, a_n), one per agent, whose observation is a tuple (o_1, …, o_n), and in which
// agent i chooses a_i from its own history of observations alone, all agents sharing one reward. Over a
// horizon h a policy of agent i is a tree: an action at the root and, below a node, one subtree for each of
// its observations (a tree of depth h has (|O_i|^h − 1)/(|O_i| − 1) nodes). The value of a joint policy
// (q_1, …, q_n) from the state s is
//
//   V(s, q) = R(s, a(q)) + γ Σ_{s'} T(s' | s, a(q)) Σ_o Z(o | s', a(q)) V(s', q/o),
//
// q/o the subtrees each agent enters on its observation; from b₀, Σ_s b₀(s) V(s, q). Finding the best
// joint policy is NEXP-complete for two agents or more (Bernstein et al. 2002, Theorem 1), so exhaustive
// search serves the smallest problems and checks, and JESP (Nair, Tambe, Yokoo, Pynadath and Marsella
// 2003) the rest: from any joint policy, each agent in turn takes its best response to the others' trees,
// computed exactly as a POMDP whose hidden state is (s, the others' current nodes), until no agent can
// improve: an equilibrium of the agents' common game, a local optimum, not in general the global one.
//
// Joint actions and joint observations are numbered in mixed radix, agent 0 fastest.

#pragma once

#include <tyche/decision/pomdp.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <vector>

namespace tyche::decision {

struct DecPomdp {
  std::size_t states = 0;
  std::vector<std::size_t> actions, observations;  // one per agent
  std::vector<double> T;  // T(s' | s, ja) at [(ja S + s) S + s']
  std::vector<double> Z;  // Z(jo | s', ja) at [(ja S + s') JO + jo]
  std::vector<double> R;  // R(s, ja) at [ja S + s]
  double discount = 1;
  Belief initial;

  std::size_t agents() const { return actions.size(); }
  std::size_t joint_actions() const {
    std::size_t n = 1;
    for (const auto a : actions) n *= a;
    return n;
  }
  std::size_t joint_observations() const {
    std::size_t n = 1;
    for (const auto o : observations) n *= o;
    return n;
  }
  std::size_t join(const std::vector<std::size_t>& parts, const std::vector<std::size_t>& radix) const {
    std::size_t k = 0, m = 1;
    for (std::size_t i = 0; i < parts.size(); i++) k += parts[i] * m, m *= radix[i];
    return k;
  }
  std::vector<std::size_t> split(std::size_t k, const std::vector<std::size_t>& radix) const {
    std::vector<std::size_t> parts(radix.size());
    for (std::size_t i = 0; i < radix.size(); i++) parts[i] = k % radix[i], k /= radix[i];
    return parts;
  }
};

// A policy tree of depth h for an agent with |O| observations: actions at nodes in breadth-first order,
// the children of node k at k |O| + 1 + o.
struct PolicyTree {
  std::size_t observations = 0;
  int depth = 0;
  std::vector<std::size_t> action;
  std::size_t child(std::size_t k, std::size_t o) const { return k * observations + 1 + o; }
  static std::size_t nodes(std::size_t O, int depth) {
    std::size_t n = 0, level = 1;
    for (int d = 0; d < depth; d++) n += level, level *= O;
    return n;
  }
};

inline void validate(const DecPomdp& D, double tolerance = 1e-9) {
  const std::size_t S = D.states, JA = D.joint_actions(), JO = D.joint_observations();
  detail::require(S > 0 && D.agents() > 0 && D.observations.size() == D.agents(), "a Dec-POMDP needs states, and actions and observations for each agent");
  detail::require(D.T.size() == JA * S * S && D.Z.size() == JA * S * JO && D.R.size() == JA * S, "T is JA·S·S, Z is JA·S·JO and R is JA·S");
  for (std::size_t ja = 0; ja < JA; ja++)
    for (std::size_t s = 0; s < S; s++) {
      double t = 0, z = 0;
      for (std::size_t k = 0; k < S; k++) t += D.T[(ja * S + s) * S + k];
      for (std::size_t o = 0; o < JO; o++) z += D.Z[(ja * S + s) * JO + o];
      detail::require(std::fabs(t - 1) <= tolerance * S && std::fabs(z - 1) <= tolerance * JO, "the rows of T and Z are laws");
    }
  detail::require(D.initial.size() == S, "the initial belief has one entry per state");
}

// The exact value of a joint policy from b₀.
inline double value(const DecPomdp& D, const std::vector<PolicyTree>& q) {
  const std::size_t S = D.states, JO = D.joint_observations(), n = D.agents();
  const int h = q[0].depth;
  std::vector<std::size_t> node(n, 0);
  std::function<double(std::size_t, std::vector<std::size_t>&, int)> v = [&](std::size_t s, std::vector<std::size_t>& at, int depth) -> double {
    std::vector<std::size_t> a(n);
    for (std::size_t i = 0; i < n; i++) a[i] = q[i].action[at[i]];
    const std::size_t ja = D.join(a, D.actions);
    double x = D.R[ja * S + s];
    if (depth + 1 == h) return x;
    for (std::size_t s2 = 0; s2 < S; s2++) {
      const double t = D.T[(ja * S + s) * S + s2];
      if (t == 0) continue;
      for (std::size_t jo = 0; jo < JO; jo++) {
        const double z = D.Z[(ja * S + s2) * JO + jo];
        if (z == 0) continue;
        const auto o = D.split(jo, D.observations);
        std::vector<std::size_t> next(n);
        for (std::size_t i = 0; i < n; i++) next[i] = q[i].child(at[i], o[i]);
        x += D.discount * t * z * v(s2, next, depth + 1);
      }
    }
    return x;
  };
  double total = 0;
  for (std::size_t s = 0; s < S; s++)
    if (D.initial[s] > 0) total += D.initial[s] * v(s, node, 0);
  return total;
}

// Every policy tree of depth h for an agent: |A|^{nodes} of them, in lexicographic order of their actions.
inline std::vector<PolicyTree> all_trees(std::size_t A, std::size_t O, int depth) {
  const std::size_t N = PolicyTree::nodes(O, depth);
  std::vector<PolicyTree> out;
  std::vector<std::size_t> act(N, 0);
  while (true) {
    out.push_back({O, depth, act});
    std::size_t k = 0;
    while (k < N && ++act[k] == A) act[k++] = 0;
    if (k == N) break;
  }
  return out;
}

struct JointPolicy {
  std::vector<PolicyTree> trees;
  double value = -std::numeric_limits<double>::infinity();
  long evaluated = 0;
};

// The best joint policy by exhaustive search (two agents; for the smallest horizons and for checks).
inline JointPolicy exhaustive(const DecPomdp& D, int horizon) {
  validate(D);
  detail::require(D.agents() == 2, "exhaustive search is written for two agents");
  const auto Q0 = all_trees(D.actions[0], D.observations[0], horizon), Q1 = all_trees(D.actions[1], D.observations[1], horizon);
  JointPolicy best;
  for (const auto& a : Q0)
    for (const auto& b : Q1) {
      const double x = value(D, {a, b});
      best.evaluated++;
      if (x > best.value) best.value = x, best.trees = {a, b};
    }
  return best;
}

// Grafts `sub` (depth d − 1) below the root of `t` (depth d) on observation o: in breadth-first order, node p
// of level ℓ of the subtree is node o |O|^ℓ + p of level ℓ + 1 of the tree.
inline void graft(PolicyTree& t, std::size_t o, const PolicyTree& sub) {
  const std::size_t O = t.observations;
  std::size_t width = 1;
  for (int level = 0; level < sub.depth; level++, width *= O)
    for (std::size_t p = 0; p < width; p++)
      t.action[PolicyTree::nodes(O, level + 1) + o * width + p] = sub.action[PolicyTree::nodes(O, level) + p];
}

// Agent i's best response to the others' trees: exact, by search over its own observation histories with a
// belief over (s, the others' current nodes).
inline JointPolicy best_response(const DecPomdp& D, const std::vector<PolicyTree>& q, std::size_t agent) {
  const std::size_t S = D.states, n = D.agents(), JO = D.joint_observations();
  const int h = q[0].depth;
  const std::size_t Ai = D.actions[agent], Oi = D.observations[agent];
  using Key = std::vector<std::size_t>;  // [s, node_0, …, node_{n−1}], the agent's own entry unused
  using Bel = std::map<Key, double>;
  struct Sub {
    double value;
    PolicyTree tree;
  };
  std::function<Sub(const Bel&, int)> solve = [&](const Bel& b, int remaining) -> Sub {
    Sub best{-std::numeric_limits<double>::infinity(), {}};
    for (std::size_t ai = 0; ai < Ai; ai++) {
      PolicyTree t{Oi, remaining, std::vector<std::size_t>(PolicyTree::nodes(Oi, remaining), 0)};
      t.action[0] = ai;
      double x = 0;
      std::vector<Bel> next(Oi);
      for (const auto& [key, p] : b) {
        std::vector<std::size_t> a(n);
        for (std::size_t j = 0; j < n; j++) a[j] = j == agent ? ai : q[j].action[key[1 + j]];
        const std::size_t ja = D.join(a, D.actions), s = key[0];
        x += p * D.R[ja * S + s];
        if (remaining == 1) continue;
        for (std::size_t s2 = 0; s2 < S; s2++) {
          const double tr = D.T[(ja * S + s) * S + s2];
          if (tr == 0) continue;
          for (std::size_t jo = 0; jo < JO; jo++) {
            const double z = D.Z[(ja * S + s2) * JO + jo];
            if (z == 0) continue;
            const auto o = D.split(jo, D.observations);
            Key k2(1 + n, 0);
            k2[0] = s2;
            for (std::size_t j = 0; j < n; j++) k2[1 + j] = j == agent ? 0 : q[j].child(key[1 + j], o[j]);
            next[o[agent]][k2] += p * tr * z;
          }
        }
      }
      if (remaining > 1)
        for (std::size_t oi = 0; oi < Oi; oi++) {
          double mass = 0;
          for (const auto& kv : next[oi]) mass += kv.second;
          if (mass <= 0) continue;  // unreachable: any subtree serves, left at action 0
          Bel normal;
          for (const auto& [key, p] : next[oi]) normal[key] = p / mass;
          const Sub sub = solve(normal, remaining - 1);
          x += D.discount * mass * sub.value;
          graft(t, oi, sub.tree);
        }
      if (x > best.value) best = {x, std::move(t)};
    }
    return best;
  };
  Bel b0;
  for (std::size_t s = 0; s < S; s++)
    if (D.initial[s] > 0) {
      Key k(1 + n, 0);
      k[0] = s;
      b0[k] += D.initial[s];
    }
  const Sub top = solve(b0, h);
  JointPolicy out;
  out.trees = q;
  out.trees[agent] = top.tree;
  out.value = top.value;
  return out;
}

struct JespOptions {
  int restarts = 20;
  int max_rounds = 100;
  std::uint64_t seed = 20261104;
};

// JESP with random restarts: the best equilibrium found.
inline JointPolicy jesp(const DecPomdp& D, int horizon, const JespOptions& opt = {}) {
  validate(D);
  JointPolicy best;
  for (int r = 0; r < opt.restarts; r++) {
    calx::random::Stream rng(opt.seed, 0x6a657370u, static_cast<std::uint32_t>(r), 0);  // "jesp"
    std::vector<PolicyTree> q;
    for (std::size_t i = 0; i < D.agents(); i++) {
      PolicyTree t{D.observations[i], horizon, std::vector<std::size_t>(PolicyTree::nodes(D.observations[i], horizon))};
      for (auto& a : t.action) a = static_cast<std::size_t>(rng.below(D.actions[i]));
      q.push_back(t);
    }
    double v = value(D, q);
    for (int round = 0; round < opt.max_rounds; round++) {
      bool improved = false;
      for (std::size_t i = 0; i < D.agents(); i++) {
        const auto br = best_response(D, q, i);
        if (br.value > v + 1e-12) q = br.trees, v = br.value, improved = true;
      }
      if (!improved) break;
    }
    best.evaluated++;
    if (v > best.value) best.value = v, best.trees = q;
  }
  return best;
}

}  // namespace tyche::decision
