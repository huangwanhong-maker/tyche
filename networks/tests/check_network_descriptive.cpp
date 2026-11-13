// Checks of tyche/networks/network and descriptive: construction, density, reciprocity, the dyad and triad
// censuses, clustering, components, cores and degree distributions.
//
// Closed forms on small graphs (a triangle, a star, a complete graph, directed cycles); the triad classifier
// on all 64 labelled directed graphs of three nodes, whose classes have the sizes 3!/|Aut| and are invariant
// under relabelling; Batagelj and Mrvar's census against the classification of every triple, and both
// censuses summing to C(n, 2) and C(n, 3); clustering against triangles counted by brute force; strong
// components against mutual reachability by transitive closure; core numbers against repeated peeling.
// Cross-checks against networkx (tests/fixtures/make_networkx_fixtures.py).

#include <tyche/networks/descriptive.hpp>
#include <tyche/networks/network.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace tyche::networks;

namespace {

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

Network random_network(calx::random::Stream& rng, std::size_t n, double p, bool directed) {
  std::vector<Tie> t;
  for (std::size_t a = 0; a < n; a++)
    for (std::size_t b = directed ? 0 : a + 1; b < n; b++)
      if (a != b && rng.uniform() < p) t.push_back({a, b});
  return network(n, t, directed);
}

Network star(std::size_t leaves) {
  std::vector<Tie> t;
  for (std::size_t k = 1; k <= leaves; k++) t.push_back({0, k});
  return network(leaves + 1, t, false);
}

Network complete(std::size_t n, bool directed) {
  std::vector<Tie> t;
  for (std::size_t a = 0; a < n; a++)
    for (std::size_t b = directed ? 0 : a + 1; b < n; b++)
      if (a != b) t.push_back({a, b});
  return network(n, t, directed);
}

Network directed_cycle(std::size_t n) {
  std::vector<Tie> t;
  for (std::size_t a = 0; a < n; a++) t.push_back({a, (a + 1) % n});
  return network(n, t, true);
}

// The classes of a census as a map name → count, for messages.
std::uint64_t sum(const TriadCensus& t) {
  std::uint64_t s = 0;
  for (auto v : t) s += v;
  return s;
}

std::vector<double> parse_list(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) v.push_back(item == "nan" ? NAN : std::stod(item));
  return v;
}

Network parse_network(std::istringstream& ls) {
  int directed = 0, weighted = 0;
  std::size_t n = 0;
  std::string list;
  ls >> directed >> weighted >> n >> list;
  std::vector<Tie> t;
  std::vector<double> w;
  std::stringstream es(list);
  std::string e;
  while (std::getline(es, e, ';')) {
    const auto dash = e.find('-'), colon = e.find(':');
    t.push_back({std::stoul(e.substr(0, dash)), std::stoul(e.substr(dash + 1, colon - dash - 1))});
    w.push_back(std::stod(e.substr(colon + 1)));
  }
  return weighted ? network(n, t, directed != 0, w) : network(n, t, directed != 0);
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261113, 0x6e657477, 101, 0);

  // ---- construction ----
  {
    const std::vector<Tie> t{{0, 1}, {1, 0}, {2, 1}, {0, 1}};
    const std::vector<double> w{1, 2, 3, 4};
    const auto u = network(3, t, false, w);
    s.require("undirected: (0, 1), (1, 0) and (0, 1) again are one tie, weights summed", u.size() == 2 && u.weight(0, 1) == 7 && u.weight(1, 0) == 7 && u.weight(1, 2) == 3);
    const auto d = network(3, t, true, w);
    s.require("directed: 0 → 1 twice is one tie of weight 5; 1 → 0 another", d.size() == 3 && d.weight(0, 1) == 5 && d.weight(1, 0) == 2 && d.weight(1, 2) == 0 && d.tied(2, 1));
    s.require("a loop is refused", throws([] { network(2, std::vector<Tie>{{1, 1}}, false); }));
    s.require("a node out of range is refused", throws([] { network(2, std::vector<Tie>{{0, 2}}, true); }));
    s.require("a weight of 0 is refused", throws([] { network(2, std::vector<Tie>{{0, 1}}, true, std::vector<double>{0}); }));
    const auto back = from_adjacency(adjacency(d), true);
    s.require("from_adjacency(adjacency(g)) = g", back.ties == d.ties && back.weights == d.weights);
    const auto und = underlying(d);
    s.require("the underlying network joins 0 — 1 with weight 5 + 2", !und.directed && und.size() == 2 && und.weight(0, 1) == 7);
    const auto st = star(5);
    std::vector<std::size_t> members;
    const auto ego = ego_network(st, 0, &members);
    const auto leaf = ego_network(st, 3);
    s.require("the ego network of a star's centre is the star; of a leaf, the leaf and the centre", ego.size() == 5 && ego.nodes == 6 && members[0] == 0 && leaf.nodes == 2 && leaf.size() == 1);
    const auto dg = degrees(d);
    s.require("degrees: out, in, total and distinct neighbours of a directed network", dg.out[0] == 1 && dg.in[0] == 1 && dg.total[0] == 2 && dg.neighbours[0] == 1 && dg.out_strength[1] == 2 && dg.in_strength[1] == 8);
  }

  // ---- closed forms ----
  {
    const auto tri = complete(3, false);
    s.check("a triangle: density 1", density(tri) - 1, 0);
    s.check("a triangle: clustering 1 at every node, transitivity 1", std::fabs(average_clustering(tri) - 1) + std::fabs(transitivity(tri) - 1), 0);
    const auto tc = triad_census(tri);
    s.require("a triangle: one triad, of class 300; dyad census (3, 0, 0)", tc[15] == 1 && sum(tc) == 1 && dyad_census(tri).mutual == 3 && dyad_census(tri).null == 0);
    const auto st = star(5);
    s.check("a star of five leaves: density 5/15", density(st) - 1.0 / 3, 1e-16);
    s.require("a star: no triangles, clustering 0, transitivity 0", average_clustering(st) == 0 && transitivity(st) == 0);
    const auto cs = core_numbers(st);
    s.require("a star: every core number 1", std::all_of(cs.begin(), cs.end(), [](std::size_t c) { return c == 1; }));
    const auto dd = degree_distribution(degrees(st).total);
    s.check("a star: p₁ = 5/6, p₅ = 1/6, mean 10/6, variance 20/9", std::fabs(dd.share[1] - 5.0 / 6) + std::fabs(dd.share[5] - 1.0 / 6) + std::fabs(dd.mean - 10.0 / 6) + std::fabs(dd.variance - 20.0 / 9), 1e-15);
    s.require("a star: P(K ≥ 0) = 1, P(K ≥ 2) = 1/6", std::fabs(dd.at_least[0] - 1) < 1e-15 && std::fabs(dd.at_least[2] - 1.0 / 6) < 1e-15);
    const auto k6 = complete(6, false);
    const auto c6 = core_numbers(k6);
    const auto t6 = triad_census(k6);
    s.require("K₆: density 1, transitivity 1, every core number 5, all 20 triads 300",
              density(k6) == 1 && transitivity(k6) == 1 && std::all_of(c6.begin(), c6.end(), [](std::size_t c) { return c == 5; }) && t6[15] == 20 && sum(t6) == 20);
    const auto c5 = directed_cycle(5);
    const auto r5 = reciprocity(c5);
    s.check("a directed 5-cycle: arc and dyadic reciprocity 0; ρ = (0 − 1/4)/(1 − 1/4) = −1/3", std::fabs(r5.arc) + std::fabs(r5.dyadic) + std::fabs(r5.garlaschelli_loffredo + 1.0 / 3), 1e-15);
    const auto t5 = triad_census(c5);
    s.require("a directed 5-cycle: five triads 021C (paths) and five 012", t5[5] == 5 && t5[1] == 5 && sum(t5) == 10);
    const auto sc5 = strong_components(c5);
    s.require("a directed 5-cycle is one strong component", sc5.count() == 1 && sc5.largest() == 5);
    const auto c3 = directed_cycle(3);
    s.require("a directed 3-cycle is one triad 030C", triad_census(c3)[9] == 1);
    const auto path = network(3, std::vector<Tie>{{0, 1}, {1, 2}}, true);
    const auto sp = strong_components(path);
    const auto wp = weak_components(path);
    s.require("a directed path of three nodes: three strong components, one weak", sp.count() == 3 && wp.count() == 1);
    // Reciprocity by hand: 0 ↔ 1, 1 → 2, 2 → 3, 3 → 0 on four nodes: m = 5, M = 1, A = 3.
    const auto rh = network(4, std::vector<Tie>{{0, 1}, {1, 0}, {1, 2}, {2, 3}, {3, 0}}, true);
    const auto rr = reciprocity(rh);
    const double abar = 5.0 / 12;
    s.check("reciprocity by hand: r = 2/5, M/(M + A) = 1/4, ρ = (2/5 − 5/12)/(1 − 5/12)",
            std::fabs(rr.arc - 0.4) + std::fabs(rr.dyadic - 0.25) + std::fabs(rr.garlaschelli_loffredo - (0.4 - abar) / (1 - abar)), 1e-15);
    const auto dc = dyad_census(rh);
    s.require("dyad census by hand: (1, 3, 2)", dc.mutual == 1 && dc.asymmetric == 3 && dc.null == 2);
    s.require("reciprocity of an undirected network is refused", throws([&] { reciprocity(st); }));
  }

  // ---- the triad classifier on all 64 labelled triads ----
  {
    // Sizes of the sixteen classes among the 64 labelled digraphs on three nodes: 3!/|Aut|.
    const std::array<int, 16> sizes{1, 6, 3, 3, 3, 6, 6, 6, 6, 2, 3, 3, 3, 6, 6, 1};
    std::array<int, 16> got{};
    const int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
    bool invariant = true;
    for (int code = 0; code < 64; code++) {
      std::array<std::array<bool, 3>, 3> arc{};
      for (int k = 0; k < 6; k++)
        if (code >> k & 1) arc[pairs[k][0]][pairs[k][1]] = true;
      const int c = triad_type(arc);
      got[static_cast<std::size_t>(c)]++;
      int perm[3] = {0, 1, 2};
      do {
        std::array<std::array<bool, 3>, 3> rel{};
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) rel[perm[i]][perm[j]] = arc[i][j];
        invariant = invariant && triad_type(rel) == c;
      } while (std::next_permutation(perm, perm + 3));
    }
    s.require("the 64 labelled triads fall into the sixteen classes with sizes 3!/|Aut| (1, 6, 3, 3, 3, 6, 6, 6, 6, 2, 3, 3, 3, 6, 6, 1)", got == sizes);
    s.require("the class of a triad is invariant under the six relabellings", invariant);
    // Each class by the example in the header's table (A, B, C = 0, 1, 2).
    auto make = [](std::initializer_list<std::pair<int, int>> arcs) {
      std::array<std::array<bool, 3>, 3> a{};
      for (auto [i, j] : arcs) a[i][j] = true;
      return triad_type(a);
    };
    const bool table = make({}) == 0 && make({{0, 1}}) == 1 && make({{0, 1}, {1, 0}}) == 2 && make({{1, 0}, {1, 2}}) == 3 && make({{0, 1}, {2, 1}}) == 4 &&
                       make({{0, 1}, {1, 2}}) == 5 && make({{0, 1}, {1, 0}, {2, 1}}) == 6 && make({{0, 1}, {1, 0}, {1, 2}}) == 7 &&
                       make({{0, 1}, {2, 1}, {0, 2}}) == 8 && make({{1, 0}, {2, 1}, {0, 2}}) == 9 && make({{0, 1}, {1, 0}, {1, 2}, {2, 1}}) == 10 &&
                       make({{1, 0}, {1, 2}, {0, 2}, {2, 0}}) == 11 && make({{0, 1}, {2, 1}, {0, 2}, {2, 0}}) == 12 && make({{0, 1}, {1, 2}, {0, 2}, {2, 0}}) == 13 &&
                       make({{0, 1}, {1, 2}, {2, 1}, {0, 2}, {2, 0}}) == 14 && make({{0, 1}, {1, 0}, {1, 2}, {2, 1}, {0, 2}, {2, 0}}) == 15;
    s.require("each class named by the example of the header's table (111D: A↔B, C→B; 120D: B→A, B→C, A↔C; …)", table);
  }

  // ---- censuses against brute force ----
  {
    bool census_ok = true, sums_ok = true;
    for (int t = 0; t < 30; t++) {
      const std::size_t n = 3 + rng.below(16);
      const double p = 0.05 + 0.6 * rng.uniform();
      const auto g = random_network(rng, n, p, t % 3 != 0);
      TriadCensus brute{};
      for (std::size_t a = 0; a < n; a++)
        for (std::size_t b = a + 1; b < n; b++)
          for (std::size_t c = b + 1; c < n; c++) brute[static_cast<std::size_t>(triad_type(g, a, b, c))]++;
      const auto bm = triad_census(g);
      census_ok = census_ok && bm == brute;
      const auto dc = dyad_census(g);
      const std::uint64_t nn = n;
      sums_ok = sums_ok && dc.mutual + dc.asymmetric + dc.null == nn * (nn - 1) / 2 && sum(bm) == nn * (nn - 1) * (nn - 2) / 6 && 2 * dc.mutual + dc.asymmetric == (g.directed ? g.size() : 2 * g.size());
    }
    s.require("Batagelj and Mrvar's triad census equals the classification of every triple (30 random networks, directed and undirected)", census_ok);
    s.require("the dyad census sums to C(n, 2) and accounts for every tie; the triad census sums to C(n, 3)", sums_ok);
  }

  // ---- clustering, components and cores against brute force ----
  {
    bool tri_ok = true, strong_ok = true, weak_ok = true, core_ok = true;
    double trans_gap = 0;
    for (int t = 0; t < 25; t++) {
      const std::size_t n = 4 + rng.below(20);
      const auto g = random_network(rng, n, 0.04 + 0.3 * rng.uniform(), t % 2 == 0);
      const auto nb = neighbourhoods(g);
      auto adj = [&](std::size_t a, std::size_t b) { return std::binary_search(nb[a].begin(), nb[a].end(), b); };
      const auto tr = triangles(g);
      double closed = 0, triples = 0;
      for (std::size_t v = 0; v < n; v++) {
        std::uint64_t c = 0;
        for (std::size_t a = 0; a < n; a++)
          for (std::size_t b = a + 1; b < n; b++)
            if (a != v && b != v && adj(v, a) && adj(v, b) && adj(a, b)) c++;
        tri_ok = tri_ok && c == tr[v];
        closed += static_cast<double>(c);
        const double k = static_cast<double>(nb[v].size());
        triples += k * (k - 1) / 2;
      }
      trans_gap = std::max(trans_gap, std::fabs(transitivity(g) - (triples > 0 ? closed / triples : 0.0)));
      // Reachability by transitive closure.
      std::vector<std::vector<char>> R(n, std::vector<char>(n, 0)), W(n, std::vector<char>(n, 0));
      for (std::size_t v = 0; v < n; v++) {
        R[v][v] = W[v][v] = 1;
        for (std::size_t u : g.out[v]) R[v][u] = 1, W[v][u] = W[u][v] = 1;
      }
      for (std::size_t k = 0; k < n; k++)
        for (std::size_t i = 0; i < n; i++)
          for (std::size_t j = 0; j < n; j++) {
            if (R[i][k] && R[k][j]) R[i][j] = 1;
            if (W[i][k] && W[k][j]) W[i][j] = 1;
          }
      const auto sc = strong_components(g), wc = weak_components(g);
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) {
          strong_ok = strong_ok && ((sc.label[i] == sc.label[j]) == (R[i][j] && R[j][i]));
          weak_ok = weak_ok && ((wc.label[i] == wc.label[j]) == (W[i][j] != 0));
        }
      // Core numbers by peeling: the k-core is what remains after removing nodes of fewer than k neighbours.
      const auto cn = core_numbers(g);
      for (std::size_t k = 0; k <= n; k++) {
        std::vector<char> alive(n, 1);
        for (bool changed = true; changed;) {
          changed = false;
          for (std::size_t v = 0; v < n; v++) {
            if (!alive[v]) continue;
            std::size_t d = 0;
            for (std::size_t u : nb[v]) d += alive[u];
            if (d < k) alive[v] = 0, changed = true;
          }
        }
        for (std::size_t v = 0; v < n; v++) core_ok = core_ok && ((alive[v] != 0) == (cn[v] >= k));
      }
    }
    s.require("triangles through each node equal a count over every pair of neighbours (25 random networks)", tri_ok);
    s.check("transitivity = Σ t_i / Σ C(k_i, 2) counted directly", trans_gap, 1e-15);
    s.require("strong components: same label exactly when mutually reachable (transitive closure)", strong_ok);
    s.require("weak components: same label exactly when joined in the underlying graph", weak_ok);
    s.require("core numbers: the k-core is what repeated peeling of nodes with fewer than k neighbours leaves, for every k", core_ok);
  }

  // ---- cross-checks against networkx ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/networkx_networks.txt");
    if (!in) {
      s.skip("networkx fixture not found");
    } else {
      std::map<std::string, Network> nets;
      std::string line;
      double scalar_gap = 0, vector_gap = 0;
      bool census_ok = true, components_ok = true;
      int rows = 0;
      while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string kind, name;
        ls >> kind >> name;
        if (kind == "network") {
          nets[name] = parse_network(ls);
          continue;
        }
        if (kind == "census") {
          std::string list;
          ls >> list;
          const auto want = parse_list(list);
          const auto got = triad_census(nets.at(name));
          for (std::size_t k = 0; k < 16; k++) census_ok = census_ok && static_cast<double>(got[k]) == want[k];
          rows++;
          continue;
        }
        if (kind != "scalar" && kind != "vector") continue;
        std::string stat, list;
        ls >> stat >> list;
        const auto& g = nets.at(name);
        const auto want = parse_list(list);
        if (kind == "scalar") {
          double got;
          if (stat == "transitivity" || stat == "transitivity_underlying") got = transitivity(g);
          else if (stat == "average_clustering") got = average_clustering(g);
          else if (stat == "density") got = density(g);
          else if (stat == "reciprocity") got = reciprocity(g).arc;
          else if (stat == "strong_components") {
            components_ok = components_ok && static_cast<double>(strong_components(g).count()) == want[0];
            rows++;
            continue;
          } else if (stat == "weak_components") {
            components_ok = components_ok && static_cast<double>(weak_components(g).count()) == want[0];
            rows++;
            continue;
          } else continue;  // mixing statistics: check_mixing
          scalar_gap = std::max(scalar_gap, std::fabs(got - want[0]));
          rows++;
        } else {
          std::vector<double> got;
          if (stat == "clustering") got = local_clustering(g);
          else if (stat == "core_number" || stat == "core_number_underlying") {
            for (auto c : core_numbers(g)) got.push_back(static_cast<double>(c));
          } else if (stat == "triangles") {
            for (auto c : triangles(g)) got.push_back(static_cast<double>(c));
          } else if (stat == "strong_component_sizes") {
            for (auto c : strong_components(g).sizes) got.push_back(static_cast<double>(c));
            std::sort(got.begin(), got.end());
          } else continue;  // brokerage: check_mixing
          if (got.size() != want.size()) vector_gap = INFINITY;
          for (std::size_t k = 0; k < got.size() && k < want.size(); k++) vector_gap = std::max(vector_gap, std::fabs(got[k] - want[k]));
          rows++;
        }
      }
      s.check("cross-check (networkx 3.7): transitivity, average clustering, density and arc reciprocity of the karate club and two random networks", scalar_gap, 1e-15);
      s.check("cross-check (networkx 3.7): local clustering, triangles, core numbers (of the underlying graph) and strong components' sizes", vector_gap, 1e-15);
      s.require("cross-check (networkx 3.7): the triad census of a random directed network (30 nodes, 140 ties)", census_ok);
      s.require("cross-check (networkx 3.7): the numbers of strong and weak components", components_ok);
      s.require("cross-check: every fixture row read", rows == 16, std::to_string(rows) + " rows");
    }
  }

  return s.done();
}
