// Descriptive statistics of a network: density, reciprocity, the dyad and triad censuses, clustering and
// transitivity, components, cores and degree distributions.
//
// DENSITY is the share of possible ties present: m / (n(n − 1)) directed, 2m / (n(n − 1)) undirected
// (Wasserman and Faust 1994, Social Network Analysis; Newman 2018, Networks).
//
// RECIPROCITY of a directed network. With M the mutual pairs and A the asymmetric ones (m = 2M + A):
//
//   arc reciprocity      r = 2M / m, the share of ties whose reverse is also a tie (Newman 2018);
//   dyadic reciprocity   M / (M + A), the share of the connected pairs that are mutual (Wasserman and
//                        Faust 1994);
//   Garlaschelli and Loffredo's ρ = (r − ā)/(1 − ā), ā the density (2004, Phys. Rev. Lett. 93:
//                        268701): r measured against what the density alone would give, 0 when ties are
//                        reciprocated as often as chance, positive when more, negative when less.
//
// THE DYAD CENSUS (Holland and Leinhardt 1970, Amer. J. Sociol. 76: 492–513; Wasserman and Faust 1994)
// counts the C(n, 2) pairs as mutual, asymmetric or null: (M, A, N). THE TRIAD CENSUS counts the C(n, 3)
// triples by their isomorphism class, of which a directed graph on three nodes has sixteen, named by the
// counts of mutual, asymmetric and null pairs and a letter (Davis and Leinhardt 1972, in Sociological
// Theories in Progress 2: 218–251; Holland and Leinhardt 1976, Sociol. Methodol. 7: 1–45). In the order
// used here and by Batagelj and Mrvar, with A, B, C the three nodes:
//
//    0 003   empty                    6 111D  A↔B, C→B              11 120D  B→A, B→C, A↔C
//    1 012   A→B                      7 111U  A↔B, B→C              12 120U  A→B, C→B, A↔C
//    2 102   A↔B                      8 030T  A→B, C→B, A→C         13 120C  A→B, B→C, A↔C
//    3 021D  B→A, B→C (out-star)      9 030C  B→A, C→B, A→C (cycle) 14 210   A→B, B↔C, A↔C
//    4 021U  A→B, C→B (in-star)      10 201   A↔B, B↔C              15 300   complete
//    5 021C  A→B, B→C (path)
//
// D ("down") and U ("up") say whether the lone node sends to, or receives from, the others; for 111 the
// asymmetric tie runs into the mutual pair (D) or out of it (U). An undirected network is read as one
// whose every tie is mutual (only 003, 102, 201 and 300 occur). The census is computed by Batagelj and
// Mrvar's algorithm (2001, Social Networks 23: 237–243), in O(m Δ) for maximum degree Δ: each pair joined
// in the underlying graph, with the third nodes adjacent to either, visits every triple with two or more
// ties exactly once, triples with one tie are counted at once, and 003 is what remains of C(n, 3).
//
// CLUSTERING (on the underlying graph; Watts and Strogatz 1998, Nature 393: 440–442; Newman 2018): a
// node's local clustering is the share of pairs of its neighbours that are tied, 2t_i/(k_i(k_i − 1)) with
// t_i the triangles through it, and 0 when k_i < 2 (networkx's convention). The average clustering is
// their mean over all nodes (zeros included). TRANSITIVITY is 3 × triangles / connected triples
// = Σ t_i / Σ C(k_i, 2), the share of two-paths that close (Wasserman and Faust 1994; Newman 2018).
//
// COMPONENTS. Weak components (of the underlying graph) are morphe's spanning forest; strong components
// (every node reaches every other along directed ties) by Tarjan's algorithm (1972, SIAM J. Comput. 1:
// 146–160), iteratively. Components are numbered in order of their least node.
//
// CORES (Seidman 1983, Social Networks 5: 269–287): the k-core is the largest subgraph in which every node
// has at least k neighbours; a node's core number is the largest k whose core holds it. Computed on the
// underlying graph (distinct neighbours; networkx counts in + out degree for directed graphs instead) by
// Batagelj and Zaveršnik's algorithm (2011, Adv. Data Anal. Classif. 5: 129–145), in O(n + m).
//
// DEGREE DISTRIBUTIONS: the share p_k of nodes of each degree k, the share at least k, the mean and the
// variance (divisor n).

#pragma once

#include <tyche/networks/network.hpp>

#include <morphe/combinatorics/graph.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace tyche::networks {

inline double density(const Network& g) {
  require(g.nodes >= 2, "density", "at least two nodes");
  const double n = static_cast<double>(g.nodes), pairs = n * (n - 1) / (g.directed ? 1.0 : 2.0);
  return static_cast<double>(g.size()) / pairs;
}

struct DyadCensus {
  std::uint64_t mutual = 0, asymmetric = 0, null = 0;  // M, A, N; M + A + N = C(n, 2)
};

inline DyadCensus dyad_census(const Network& g) {
  DyadCensus d;
  const std::uint64_t n = g.nodes, pairs = n * (n - 1) / 2;
  if (!g.directed) {
    d.mutual = g.size();
  } else {
    for (const auto& [a, b] : g.ties)
      if (a < b && g.tied(b, a)) d.mutual++;
    d.asymmetric = g.size() - 2 * d.mutual;
  }
  d.null = pairs - d.mutual - d.asymmetric;
  return d;
}

struct Reciprocity {
  double arc = 0;                    // r = 2M/m
  double dyadic = 0;                 // M/(M + A)
  double garlaschelli_loffredo = 0;  // ρ = (r − ā)/(1 − ā)
  double density = 0;                // ā
};

inline Reciprocity reciprocity(const Network& g) {
  require(g.directed, "reciprocity", "reciprocity is a property of directed networks");
  require(g.size() > 0, "reciprocity", "at least one tie");
  const auto d = dyad_census(g);
  Reciprocity r;
  r.arc = 2.0 * static_cast<double>(d.mutual) / static_cast<double>(g.size());
  r.dyadic = static_cast<double>(d.mutual) / static_cast<double>(d.mutual + d.asymmetric);
  r.density = density(g);
  require(r.density < 1, "reciprocity", "Garlaschelli and Loffredo's ρ is undefined for a complete network");
  r.garlaschelli_loffredo = (r.arc - r.density) / (1 - r.density);
  return r;
}

// ---------------------------------------------------------------- triads

inline constexpr std::array<std::string_view, 16> triad_names{"003", "012", "102", "021D", "021U", "021C", "111D", "111U",
                                                              "030T", "030C", "201", "120D", "120U", "120C", "210", "300"};

// The class (0–15, in triad_names' order) of a triad given by its arcs: arc[i][j] for i → j, i ≠ j.
inline int triad_type(const std::array<std::array<bool, 3>, 3>& arc) {
  int M = 0, A = 0;
  for (int i = 0; i < 3; i++)
    for (int j = i + 1; j < 3; j++) {
      if (arc[i][j] && arc[j][i]) M++;
      else if (arc[i][j] || arc[j][i]) A++;
    }
  auto out = [&](int v) { return int(arc[v][(v + 1) % 3]) + int(arc[v][(v + 2) % 3]); };
  auto mutual = [&](int i, int j) { return arc[i][j] && arc[j][i]; };
  auto null = [&](int i, int j) { return !arc[i][j] && !arc[j][i]; };
  if (M == 0 && A == 0) return 0;
  if (M == 0 && A == 1) return 1;
  if (M == 1 && A == 0) return 2;
  if (M == 0 && A == 2) {
    // The node in both connected pairs is the centre.
    int c = 0;
    for (int v = 0; v < 3; v++)
      if (null((v + 1) % 3, (v + 2) % 3)) c = v;
    if (out(c) == 2) return 3;  // 021D
    if (out(c) == 0) return 4;  // 021U
    return 5;                   // 021C
  }
  if (M == 1 && A == 1) {
    // The mutual pair {p, q}; the lone node l tied to one of them, s.
    int l = 0;
    for (int v = 0; v < 3; v++)
      if (mutual((v + 1) % 3, (v + 2) % 3)) l = v;
    const int p = (l + 1) % 3, q = (l + 2) % 3;
    const int s = null(l, p) ? q : p;
    return arc[l][s] ? 6 : 7;  // 111D: l → s, into the mutual pair; 111U: s → l
  }
  if (M == 0 && A == 3) return out(0) == 1 && out(1) == 1 && out(2) == 1 ? 9 : 8;  // 030C a cycle, else 030T
  if (M == 2 && A == 0) return 10;
  if (M == 1 && A == 2) {
    int l = 0;
    for (int v = 0; v < 3; v++)
      if (mutual((v + 1) % 3, (v + 2) % 3)) l = v;
    if (out(l) == 2) return 11;  // 120D: the lone node sends to both
    if (out(l) == 0) return 12;  // 120U: it receives from both
    return 13;                   // 120C
  }
  if (M == 2 && A == 1) return 14;
  return 15;
}

// The class of the triad (a, b, c) of a network.
inline int triad_type(const Network& g, std::size_t a, std::size_t b, std::size_t c) {
  const std::size_t v[3] = {a, b, c};
  std::array<std::array<bool, 3>, 3> arc{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) arc[i][j] = i != j && g.tied(v[i], v[j]);
  return triad_type(arc);
}

using TriadCensus = std::array<std::uint64_t, 16>;

// The triad census by Batagelj and Mrvar's algorithm.
inline TriadCensus triad_census(const Network& g) {
  TriadCensus t{};
  const std::size_t n = g.nodes;
  const auto nb = neighbourhoods(g);
  std::vector<std::size_t> mark(n, n);  // mark[w] == v: w adjacent to v (in the underlying graph)
  for (std::size_t v = 0; v < n; v++) {
    for (std::size_t w : nb[v]) mark[w] = v;
    for (std::size_t u : nb[v]) {
      if (u <= v) continue;
      // S = N(u) ∪ N(v) \ {u, v}
      std::vector<std::size_t> S;
      std::set_union(nb[u].begin(), nb[u].end(), nb[v].begin(), nb[v].end(), std::back_inserter(S));
      std::size_t size = 0;
      for (std::size_t w : S)
        if (w != u && w != v) size++;
      const bool m = g.tied(u, v) && g.tied(v, u);
      t[m ? 2 : 1] += n - size - 2;
      for (std::size_t w : S) {
        if (w == u || w == v) continue;
        if (u < w || (v < w && w < u && mark[w] != v)) t[static_cast<std::size_t>(triad_type(g, v, u, w))]++;
      }
    }
  }
  const std::uint64_t nn = n, all = n >= 3 ? nn * (nn - 1) * (nn - 2) / 6 : 0;
  std::uint64_t counted = 0;
  for (std::size_t k = 1; k < 16; k++) counted += t[k];
  t[0] = all - counted;
  return t;
}

// ---------------------------------------------------------------- clustering

// t_i, the triangles through each node of the underlying graph.
inline std::vector<std::uint64_t> triangles(const Network& g) {
  const auto nb = neighbourhoods(g);
  std::vector<std::uint64_t> t(g.nodes, 0);
  std::vector<std::size_t> mark(g.nodes, g.nodes);
  for (std::size_t v = 0; v < g.nodes; v++) {
    for (std::size_t u : nb[v]) mark[u] = v;
    for (std::size_t u : nb[v])
      for (std::size_t w : nb[u])
        if (w > u && mark[w] == v) t[v]++;
  }
  return t;
}

// The local clustering of each node, 2t_i/(k_i(k_i − 1)), 0 when k_i < 2.
inline std::vector<double> local_clustering(const Network& g) {
  const auto t = triangles(g);
  const auto nb = neighbourhoods(g);
  std::vector<double> c(g.nodes, 0.0);
  for (std::size_t v = 0; v < g.nodes; v++) {
    const double k = static_cast<double>(nb[v].size());
    if (k >= 2) c[v] = 2.0 * static_cast<double>(t[v]) / (k * (k - 1));
  }
  return c;
}

// The mean local clustering over every node.
inline double average_clustering(const Network& g) {
  require(g.nodes >= 1, "average_clustering", "at least one node");
  const auto c = local_clustering(g);
  double s = 0;
  for (double v : c) s += v;
  return s / static_cast<double>(g.nodes);
}

// 3 × triangles / connected triples; 0 when there is no connected triple.
inline double transitivity(const Network& g) {
  const auto t = triangles(g);
  const auto nb = neighbourhoods(g);
  double closed = 0, triples = 0;
  for (std::size_t v = 0; v < g.nodes; v++) {
    const double k = static_cast<double>(nb[v].size());
    closed += static_cast<double>(t[v]);
    triples += k * (k - 1) / 2;
  }
  return triples > 0 ? closed / triples : 0.0;
}

// ---------------------------------------------------------------- components and cores

struct Components {
  std::vector<std::size_t> label;  // per node, 0, 1, … in order of the components' least nodes
  std::vector<std::size_t> sizes;  // per component
  std::size_t count() const { return sizes.size(); }
  std::size_t largest() const { return sizes.empty() ? 0 : *std::max_element(sizes.begin(), sizes.end()); }
};

namespace detail {

inline Components relabel(const std::vector<std::size_t>& raw, std::size_t n) {
  Components c;
  std::vector<std::size_t> map(n, n);
  c.label.resize(n);
  for (std::size_t v = 0; v < n; v++) {
    if (map[raw[v]] == n) {
      map[raw[v]] = c.sizes.size();
      c.sizes.push_back(0);
    }
    c.label[v] = map[raw[v]];
    c.sizes[c.label[v]]++;
  }
  return c;
}

}  // namespace detail

// Weak components: the components of the underlying graph (morphe's spanning forest).
inline Components weak_components(const Network& g) {
  morphe::combinatorics::Multigraph G;
  G.vertices = g.nodes;
  G.edges.assign(g.ties.begin(), g.ties.end());
  const auto F = morphe::combinatorics::spanning_forest(G);
  return detail::relabel(F.component, g.nodes);
}

// Strong components by Tarjan's algorithm; for an undirected network, the weak components.
inline Components strong_components(const Network& g) {
  if (!g.directed) return weak_components(g);
  const std::size_t n = g.nodes, unset = static_cast<std::size_t>(-1);
  std::vector<std::size_t> index(n, unset), low(n, 0), comp(n, unset), stack;
  std::vector<char> on(n, 0);
  std::size_t counter = 0, found = 0;
  std::vector<std::pair<std::size_t, std::size_t>> call;  // (node, next successor to visit)
  for (std::size_t s = 0; s < n; s++) {
    if (index[s] != unset) continue;
    index[s] = low[s] = counter++;
    stack.push_back(s);
    on[s] = 1;
    call.push_back({s, 0});
    while (!call.empty()) {
      const std::size_t v = call.back().first, pos = call.back().second;
      if (pos < g.out[v].size()) {
        call.back().second++;
        const std::size_t w = g.out[v][pos];
        if (index[w] == unset) {
          index[w] = low[w] = counter++;
          stack.push_back(w);
          on[w] = 1;
          call.push_back({w, 0});
        } else if (on[w]) {
          low[v] = std::min(low[v], index[w]);
        }
        continue;
      }
      if (low[v] == index[v]) {
        for (;;) {
          const std::size_t w = stack.back();
          stack.pop_back();
          on[w] = 0;
          comp[w] = found;
          if (w == v) break;
        }
        found++;
      }
      call.pop_back();
      if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[v]);
    }
  }
  return detail::relabel(comp, n);
}

// Core numbers on the underlying graph (Batagelj and Zaveršnik).
inline std::vector<std::size_t> core_numbers(const Network& g) {
  const auto nb = neighbourhoods(g);
  const std::size_t n = g.nodes;
  std::vector<std::size_t> deg(n), pos(n), vert(n);
  std::size_t md = 0;
  for (std::size_t v = 0; v < n; v++) deg[v] = nb[v].size(), md = std::max(md, deg[v]);
  std::vector<std::size_t> bin(md + 1, 0);
  for (std::size_t v = 0; v < n; v++) bin[deg[v]]++;
  std::size_t start = 0;
  for (std::size_t d = 0; d <= md; d++) {
    const std::size_t num = bin[d];
    bin[d] = start;
    start += num;
  }
  for (std::size_t v = 0; v < n; v++) {
    pos[v] = bin[deg[v]];
    vert[pos[v]] = v;
    bin[deg[v]]++;
  }
  for (std::size_t d = md; d >= 1; d--) bin[d] = bin[d - 1];
  bin[0] = 0;
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t v = vert[i];
    for (std::size_t u : nb[v]) {
      if (deg[u] > deg[v]) {
        const std::size_t du = deg[u], pu = pos[u], pw = bin[du], w = vert[pw];
        if (u != w) {
          pos[u] = pw, vert[pu] = w;
          pos[w] = pu, vert[pw] = u;
        }
        bin[du]++;
        deg[u]--;
      }
    }
  }
  return deg;
}

// The nodes of the k-core (core number ≥ k), in order.
inline std::vector<std::size_t> k_core(const Network& g, std::size_t k) {
  const auto c = core_numbers(g);
  std::vector<std::size_t> nodes;
  for (std::size_t v = 0; v < g.nodes; v++)
    if (c[v] >= k) nodes.push_back(v);
  return nodes;
}

// ---------------------------------------------------------------- degree distributions

struct DegreeDistribution {
  std::vector<double> share;     // p_k, k = 0, …, max
  std::vector<double> at_least;  // P(K ≥ k)
  double mean = 0, variance = 0; // divisor n
};

inline DegreeDistribution degree_distribution(std::span<const std::size_t> degree) {
  require(!degree.empty(), "degree_distribution", "at least one node");
  DegreeDistribution d;
  const std::size_t top = *std::max_element(degree.begin(), degree.end());
  d.share.assign(top + 1, 0.0);
  const double n = static_cast<double>(degree.size());
  for (std::size_t k : degree) d.share[k] += 1 / n, d.mean += static_cast<double>(k) / n;
  for (std::size_t k : degree) d.variance += (static_cast<double>(k) - d.mean) * (static_cast<double>(k) - d.mean) / n;
  d.at_least.assign(top + 1, 0.0);
  double tail = 0;
  for (std::size_t k = top + 1; k-- > 0;) {
    tail += d.share[k];
    d.at_least[k] = tail;
  }
  return d;
}

}  // namespace tyche::networks
