// Relational data: a network of who is tied to whom, given as a list of ties.
//
// A NETWORK on n nodes (actors: persons, households, firms) is a set of ties (Wasserman and Faust 1994,
// Social Network Analysis, the graphs and digraphs of their Part II; Newman 2018, Networks, 2nd ed.):
//
//   undirected   a tie i — j is a symmetric relation (kin, co-residence, acquaintance as both report it);
//   directed     a tie i → j runs from a sender to a receiver (i names j as a friend, i helps j); i → j
//                and j → i are two ties, and the pair is then MUTUAL.
//
// Ties may carry positive weights (frequency, strength, amount); a tie absent has weight 0. Node
// attributes (a category, a number) are not part of the network: they are passed beside it, one value per
// node, to the functions that use them (mixing.hpp).
//
// CONVENTIONS (docs/glossary.md §7). Nodes are 0, …, n − 1. A tie from a node to itself is refused: the
// statistics here are of relations between two actors, and none of the cited definitions admits loops. A
// tie given twice is one tie, its weights summed; in an undirected network (i, j) and (j, i) are the same
// tie. Ties are kept sorted, an undirected tie as (min, max). The UNDERLYING graph of a directed network
// joins i and j when either names the other; clustering, cores and brokerage are computed on it, and say
// so.
//
// The representation keeps sorted adjacency lists, so that a tie is found in O(log d) and every statistic
// here costs O(n + m) or O(Σ d²).

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyche::networks {

using calx::linalg::Matrix;       // square: adjacency and dyadic matrices
using calx::linalg::Rectangular;  // rows × cols: mixing matrices, design matrices

inline void require(bool ok, const char* where, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::networks::") + where + ": " + what);
}

using Tie = std::array<std::size_t, 2>;

struct Network {
  std::size_t nodes = 0;
  bool directed = false;
  bool weighted = false;                   // weights were given (otherwise every weight is 1)
  std::vector<Tie> ties;                   // unique and sorted; undirected ties as (min, max)
  std::vector<double> weights;             // one per tie
  std::vector<std::vector<std::size_t>> out, in;            // sorted successors and predecessors; equal when undirected
  std::vector<std::vector<double>> out_weight, in_weight;   // their weights

  std::size_t size() const { return ties.size(); }  // m, the number of ties

  // Whether i → j (directed) or i — j (undirected).
  bool tied(std::size_t i, std::size_t j) const {
    const auto& a = out[i];
    return std::binary_search(a.begin(), a.end(), j);
  }
  // The weight of i → j (or i — j), 0 when absent.
  double weight(std::size_t i, std::size_t j) const {
    const auto& a = out[i];
    const auto it = std::lower_bound(a.begin(), a.end(), j);
    return it != a.end() && *it == j ? out_weight[i][static_cast<std::size_t>(it - a.begin())] : 0.0;
  }
};

// The network of n nodes with the given ties (each a pair of distinct nodes) and, optionally, positive
// weights, one per tie as given.
inline Network network(std::size_t nodes, std::span<const Tie> ties, bool directed, std::span<const double> weights = {}) {
  require(weights.empty() || weights.size() == ties.size(), "network", "one weight per tie, or none");
  std::vector<std::pair<Tie, double>> list;
  list.reserve(ties.size());
  for (std::size_t k = 0; k < ties.size(); k++) {
    auto [a, b] = ties[k];
    require(a < nodes && b < nodes, "network", "ties join nodes 0, …, n − 1");
    require(a != b, "network", "a tie joins two different nodes (loops are not relations between two actors)");
    const double w = weights.empty() ? 1.0 : weights[k];
    require(std::isfinite(w) && w > 0, "network", "weights are positive and finite (a tie of weight 0 is no tie)");
    if (!directed && a > b) std::swap(a, b);
    list.push_back({Tie{a, b}, w});
  }
  std::sort(list.begin(), list.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
  Network g;
  g.nodes = nodes;
  g.directed = directed;
  g.weighted = !weights.empty();
  for (const auto& [t, w] : list) {
    if (!g.ties.empty() && g.ties.back() == t) {
      g.weights.back() += w;
    } else {
      g.ties.push_back(t);
      g.weights.push_back(w);
    }
  }
  g.out.assign(nodes, {});
  g.in.assign(nodes, {});
  g.out_weight.assign(nodes, {});
  g.in_weight.assign(nodes, {});
  std::vector<std::vector<std::pair<std::size_t, double>>> o(nodes), i(nodes);
  for (std::size_t k = 0; k < g.ties.size(); k++) {
    const auto [a, b] = g.ties[k];
    o[a].push_back({b, g.weights[k]});
    i[b].push_back({a, g.weights[k]});
    if (!directed) {
      o[b].push_back({a, g.weights[k]});
      i[a].push_back({b, g.weights[k]});
    }
  }
  for (std::size_t v = 0; v < nodes; v++) {
    std::sort(o[v].begin(), o[v].end());
    std::sort(i[v].begin(), i[v].end());
    for (const auto& [u, w] : o[v]) g.out[v].push_back(u), g.out_weight[v].push_back(w);
    for (const auto& [u, w] : i[v]) g.in[v].push_back(u), g.in_weight[v].push_back(w);
  }
  return g;
}

// The network whose ties are the non-zero entries a_ij, i ≠ j, of a square matrix, weighted by them;
// undirected: the matrix must be symmetric and the entries i < j are read. The diagonal is ignored.
inline Network from_adjacency(const Matrix& A, bool directed) {
  std::vector<Tie> t;
  std::vector<double> w;
  for (std::size_t i = 0; i < A.n; i++)
    for (std::size_t j = 0; j < A.n; j++) {
      if (i == j) continue;
      const double a = A(i, j);
      require(std::isfinite(a) && a >= 0, "from_adjacency", "entries are non-negative and finite");
      if (!directed) {
        require(a == A(j, i), "from_adjacency", "an undirected network has a symmetric matrix");
        if (j < i) continue;
      }
      if (a > 0) t.push_back({i, j}), w.push_back(a);
    }
  return network(A.n, t, directed, w);
}

// The matrix of weights (0 where there is no tie, and on the diagonal); symmetric when undirected.
inline Matrix adjacency(const Network& g) {
  Matrix A(g.nodes);
  for (std::size_t v = 0; v < g.nodes; v++)
    for (std::size_t k = 0; k < g.out[v].size(); k++) A(v, g.out[v][k]) = g.out_weight[v][k];
  return A;
}

// The underlying undirected network: i — j wherever i → j or j → i, its weight the sum of the two.
inline Network underlying(const Network& g) {
  if (!g.directed) return g;
  Network u = network(g.nodes, g.ties, false, g.weights);
  u.weighted = g.weighted;
  return u;
}

// The sorted neighbours of each node in the underlying graph (the union of successors and predecessors).
inline std::vector<std::vector<std::size_t>> neighbourhoods(const Network& g) {
  std::vector<std::vector<std::size_t>> nb(g.nodes);
  for (std::size_t v = 0; v < g.nodes; v++) {
    if (!g.directed) {
      nb[v] = g.out[v];
      continue;
    }
    std::set_union(g.out[v].begin(), g.out[v].end(), g.in[v].begin(), g.in[v].end(), std::back_inserter(nb[v]));
  }
  return nb;
}

// The subnetwork induced by the given nodes (distinct), renumbered in their order: the ties among them.
inline Network induced(const Network& g, std::span<const std::size_t> nodes) {
  std::vector<std::size_t> index(g.nodes, g.nodes);
  for (std::size_t k = 0; k < nodes.size(); k++) {
    require(nodes[k] < g.nodes && index[nodes[k]] == g.nodes, "induced", "distinct nodes of the network");
    index[nodes[k]] = k;
  }
  std::vector<Tie> t;
  std::vector<double> w;
  for (std::size_t k = 0; k < g.ties.size(); k++) {
    const auto [a, b] = g.ties[k];
    if (index[a] < g.nodes && index[b] < g.nodes) t.push_back({index[a], index[b]}), w.push_back(g.weights[k]);
  }
  Network h = network(nodes.size(), t, g.directed, w);
  h.weighted = g.weighted;
  return h;
}

// The ego network of a node: the node (renumbered 0) and its neighbours in the underlying graph (1, 2, …
// in their order), with every tie among them (Wasserman and Faust 1994's first-order zone).
inline Network ego_network(const Network& g, std::size_t ego, std::vector<std::size_t>* members = nullptr) {
  require(ego < g.nodes, "ego_network", "the ego is a node of the network");
  const auto nb = neighbourhoods(g);
  std::vector<std::size_t> m{ego};
  m.insert(m.end(), nb[ego].begin(), nb[ego].end());
  if (members) *members = m;
  return induced(g, m);
}

struct Degrees {
  std::vector<std::size_t> out, in;      // ties sent and received (equal when undirected: the degree)
  std::vector<std::size_t> total;        // undirected: the degree; directed: out + in (a mutual pair counts twice)
  std::vector<std::size_t> neighbours;   // distinct neighbours in the underlying graph
  std::vector<double> out_strength, in_strength, strength;  // the same with weights
};

inline Degrees degrees(const Network& g) {
  Degrees d;
  const auto nb = neighbourhoods(g);
  for (std::size_t v = 0; v < g.nodes; v++) {
    d.out.push_back(g.out[v].size());
    d.in.push_back(g.in[v].size());
    d.total.push_back(g.directed ? g.out[v].size() + g.in[v].size() : g.out[v].size());
    d.neighbours.push_back(nb[v].size());
    double so = 0, si = 0;
    for (double w : g.out_weight[v]) so += w;
    for (double w : g.in_weight[v]) si += w;
    d.out_strength.push_back(so);
    d.in_strength.push_back(si);
    d.strength.push_back(g.directed ? so + si : so);
  }
  return d;
}

}  // namespace tyche::networks
