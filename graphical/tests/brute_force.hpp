// For the checks of tyche::graphical: the joint law of a network given as nodes, by enumeration of every full
// assignment, written without the library's factors, eliminations or messages; and seeded random networks and
// polytrees.

#pragma once

#include <tyche/graphical/common.hpp>
#include <tyche/graphical/network.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace brute {

using tyche::graphical::Evidence;
using tyche::graphical::Node;

struct Joint {
  std::vector<std::size_t> card;
  std::vector<std::vector<std::size_t>> pa;
  std::vector<std::vector<double>> table;

  explicit Joint(const std::vector<Node>& nodes) {
    std::map<std::string, std::size_t> id;
    for (std::size_t v = 0; v < nodes.size(); v++) id[nodes[v].name] = v;
    for (const Node& node : nodes) {
      card.push_back(node.states.size());
      std::vector<std::size_t> p;
      for (const auto& name : node.parents) p.push_back(id.at(name));
      pa.push_back(p);
      table.push_back(node.table);
    }
  }
  std::size_t size() const { return card.size(); }
  // P(x) = Π_v table_v[row(x_pa) · k_v + x_v].
  double p(const std::vector<std::size_t>& x) const {
    double r = 1;
    for (std::size_t v = 0; v < size(); v++) {
      std::size_t row = 0;
      for (std::size_t u : pa[v]) row = row * card[u] + x[u];
      r *= table[v][row * card[v] + x[v]];
    }
    return r;
  }
  // The evidence's weight on x: the indicators of the observed states and the likelihoods.
  static double weight(const Evidence& e, const std::vector<std::size_t>& x) {
    double w = 1;
    for (const auto& [v, s] : e.observed)
      if (x[v] != s) return 0;
    for (const auto& [v, lam] : e.likelihood) w *= lam[x[v]];
    return w;
  }
  template <class F>
  void each(F f) const {
    std::vector<std::size_t> x(size(), 0);
    for (;;) {
      f(x);
      std::size_t j = size();
      while (j-- > 0) {
        if (++x[j] < card[j]) break;
        x[j] = 0;
      }
      if (j == static_cast<std::size_t>(-1)) return;
    }
  }
  double evidence(const Evidence& e) const {
    double s = 0;
    each([&](const std::vector<std::size_t>& x) { s += p(x) * weight(e, x); });
    return s;
  }
  // P(X_v | e) for every v.
  std::vector<std::vector<double>> marginals(const Evidence& e) const {
    std::vector<std::vector<double>> m(size());
    for (std::size_t v = 0; v < size(); v++) m[v].assign(card[v], 0.0);
    double z = 0;
    each([&](const std::vector<std::size_t>& x) {
      const double w = p(x) * weight(e, x);
      z += w;
      for (std::size_t v = 0; v < size(); v++) m[v][x[v]] += w;
    });
    for (auto& row : m)
      for (double& q : row) q /= z;
    return m;
  }
  // P(X_a, X_b | e), a varying slower.
  std::vector<double> pair(std::size_t a, std::size_t b, const Evidence& e) const {
    std::vector<double> m(card[a] * card[b], 0.0);
    double z = 0;
    each([&](const std::vector<std::size_t>& x) {
      const double w = p(x) * weight(e, x);
      z += w;
      m[x[a] * card[b] + x[b]] += w;
    });
    for (double& q : m) q /= z;
    return m;
  }
  // max_x P(x) w(x).
  double most_probable(const Evidence& e) const {
    double best = 0;
    each([&](const std::vector<std::size_t>& x) { best = std::max(best, p(x) * weight(e, x)); });
    return best;
  }
};

inline std::vector<double> dirichlet_row(calx::random::Stream& rng, std::size_t k) {
  std::vector<double> r(k);
  double s = 0;
  for (double& x : r) s += (x = rng.gamma(1.0));
  for (double& x : r) x /= s;
  return r;
}

inline std::vector<Node> with_tables(calx::random::Stream& rng, std::vector<Node> nodes, double lo = -1, double hi = -1) {
  std::map<std::string, std::size_t> id;
  for (std::size_t v = 0; v < nodes.size(); v++) id[nodes[v].name] = v;
  for (auto& node : nodes) {
    std::size_t rows = 1;
    for (const auto& p : node.parents) rows *= nodes[id.at(p)].states.size();
    node.table.clear();
    for (std::size_t r = 0; r < rows; r++) {
      std::vector<double> row;
      if (lo >= 0 && node.states.size() == 2) {
        const double q = rng.uniform(lo, hi);
        row = {q, 1 - q};
      } else {
        row = dirichlet_row(rng, node.states.size());
      }
      node.table.insert(node.table.end(), row.begin(), row.end());
    }
  }
  return nodes;
}

inline std::vector<std::size_t> permutation(calx::random::Stream& rng, std::size_t n) {
  std::vector<std::size_t> p(n);
  for (std::size_t i = 0; i < n; i++) p[i] = i;
  for (std::size_t i = n; i > 1; i--) std::swap(p[i - 1], p[rng.below(i)]);
  return p;
}

inline std::vector<Node> skeleton(calx::random::Stream& rng, std::size_t n, std::size_t max_card) {
  std::vector<Node> nodes(n);
  for (std::size_t v = 0; v < n; v++) {
    nodes[v].name = "X" + std::to_string(v);
    const std::size_t k = 2 + rng.below(max_card - 1);
    for (std::size_t s = 0; s < k; s++) nodes[v].states.push_back("s" + std::to_string(s));
  }
  return nodes;
}

// A random directed acyclic graph: the variables in a random order, each with up to max_parents earlier ones.
inline std::vector<Node> random_network(calx::random::Stream& rng, std::size_t n, std::size_t max_parents, std::size_t max_card, double lo = -1,
                                        double hi = -1) {
  auto nodes = skeleton(rng, n, max_card);
  const auto pos = permutation(rng, n);  // pos[i] is the variable at place i of the order
  for (std::size_t i = 1; i < n; i++) {
    const std::size_t v = pos[i];
    const std::size_t m = rng.below(std::min(i, max_parents) + 1);
    auto earlier = permutation(rng, i);
    for (std::size_t j = 0; j < m; j++) nodes[v].parents.push_back(nodes[pos[earlier[j]]].name);
  }
  return with_tables(rng, nodes, lo, hi);
}

// A random polytree: a random tree on the variables, each edge oriented at random.
inline std::vector<Node> random_polytree(calx::random::Stream& rng, std::size_t n, std::size_t max_card) {
  auto nodes = skeleton(rng, n, max_card);
  const auto pos = permutation(rng, n);
  for (std::size_t i = 1; i < n; i++) {
    const std::size_t v = pos[i], u = pos[rng.below(i)];
    if (rng.bernoulli(0.5)) nodes[v].parents.push_back(nodes[u].name);
    else nodes[u].parents.push_back(nodes[v].name);
  }
  return with_tables(rng, nodes);
}

}  // namespace brute
