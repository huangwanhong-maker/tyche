// Checks of tyche/networks/mixing and brokerage: assortativity, the E–I index, inbreeding homophily, and
// Burt's constraint and effective size.
//
// The mixing matrix sums to 1 and is symmetric for undirected networks. Assortativity by degree equals
// Newman's (2002) form over the edges with remaining degrees and the Pearson correlation of the degrees at
// the two ends; it is +1 on disjoint cliques of different sizes and −1 on a star and a complete bipartite
// graph; by category, +1 on disjoint cliques of their own category and −1 across the sides of a complete
// bipartite graph. The E–I index, the group indices, the complete-network reference and inbreeding
// homophily are worked by hand on two triangles joined by a tie. Burt's constraint and effective size are
// worked by hand on a star and a triangle; the effective size equals Borgatti's n − 2t/n on unweighted
// undirected networks. Cross-checks against networkx.

#include <tyche/networks/brokerage.hpp>
#include <tyche/networks/descriptive.hpp>
#include <tyche/networks/mixing.hpp>
#include <tyche/networks/network.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
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

// Disjoint cliques of the given sizes; category[v] = the clique of v.
Network cliques(const std::vector<std::size_t>& sizes, std::vector<std::size_t>& category) {
  std::vector<Tie> t;
  std::size_t start = 0;
  category.clear();
  for (std::size_t c = 0; c < sizes.size(); c++) {
    for (std::size_t a = 0; a < sizes[c]; a++) {
      category.push_back(c);
      for (std::size_t b = a + 1; b < sizes[c]; b++) t.push_back({start + a, start + b});
    }
    start += sizes[c];
  }
  return network(start, t, false);
}

Network bipartite(std::size_t a, std::size_t b) {
  std::vector<Tie> t;
  for (std::size_t i = 0; i < a; i++)
    for (std::size_t j = 0; j < b; j++) t.push_back({i, a + j});
  return network(a + b, t, false);
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
  calx::random::Stream rng(20261113, 0x6e657477, 102, 0);

  // ---- assortativity ----
  {
    double newman = 0, pearson = 0, msum = 0, msym = 0;
    for (int t = 0; t < 30; t++) {
      const auto g = random_network(rng, 12 + rng.below(30), 0.08 + 0.3 * rng.uniform(), false);
      if (g.size() < 3) continue;
      const auto d = degrees(g);
      // Newman 2002: over the M edges, remaining degrees j, k at the ends.
      double M = 0, jk = 0, half = 0, sq = 0;
      std::vector<double> x, y;
      for (const auto& [a, b] : g.ties) {
        const double j = static_cast<double>(d.total[a]) - 1, k = static_cast<double>(d.total[b]) - 1;
        M += 1, jk += j * k, half += 0.5 * (j + k), sq += 0.5 * (j * j + k * k);
        x.push_back(j), y.push_back(k), x.push_back(k), y.push_back(j);
      }
      const double den = sq / M - (half / M) * (half / M);
      if (den <= 1e-12) continue;
      const double r2002 = (jk / M - (half / M) * (half / M)) / den;
      const double r = degree_assortativity(g);
      newman = std::max(newman, std::fabs(r - r2002));
      // Pearson over both orientations, directly.
      double mx = 0, my = 0;
      for (std::size_t k = 0; k < x.size(); k++) mx += x[k] / static_cast<double>(x.size()), my += y[k] / static_cast<double>(x.size());
      double sxy = 0, sxx = 0, syy = 0;
      for (std::size_t k = 0; k < x.size(); k++) sxy += (x[k] - mx) * (y[k] - my), sxx += (x[k] - mx) * (x[k] - mx), syy += (y[k] - my) * (y[k] - my);
      pearson = std::max(pearson, std::fabs(r - sxy / std::sqrt(sxx * syy)));
      std::vector<std::size_t> cat(g.nodes);
      for (auto& c : cat) c = rng.below(3);
      const auto e = mixing_matrix(g, cat, 3);
      double total = 0;
      for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 3; j++) total += e(i, j), msym = std::max(msym, std::fabs(e(i, j) - e(j, i)));
      msum = std::max(msum, std::fabs(total - 1));
    }
    s.check("degree assortativity equals Newman's (2002) form over edges with remaining degrees (30 random networks)", newman, 1e-12);
    s.check("degree assortativity is the Pearson correlation of the degrees at the two ends, both orientations", pearson, 1e-12);
    s.check("the mixing matrix sums to 1", msum, 1e-15);
    s.check("the mixing matrix of an undirected network is symmetric", msym, 0);
    std::vector<std::size_t> cat;
    const auto cl = cliques({2, 3, 4, 5}, cat);
    s.check("disjoint cliques of sizes 2, 3, 4, 5: degree assortativity +1", degree_assortativity(cl) - 1, 1e-14);
    s.check("disjoint cliques, each its own category: categorical assortativity +1", categorical_assortativity(cl, cat) - 1, 1e-15);
    std::vector<double> sizes;
    for (std::size_t c : cat) sizes.push_back(static_cast<double>(c) * 1.5 - 2);
    s.check("disjoint cliques: a number constant on each clique has numeric assortativity +1", numeric_assortativity(cl, sizes) - 1, 1e-14);
    std::vector<Tie> st;
    for (std::size_t k = 1; k <= 6; k++) st.push_back({0, k});
    s.check("a star of six leaves: degree assortativity −1", degree_assortativity(network(7, st, false)) + 1, 1e-14);
    const auto kb = bipartite(2, 5);
    s.check("K₂,₅: degree assortativity −1", degree_assortativity(kb) + 1, 1e-14);
    std::vector<std::size_t> side{0, 0, 1, 1, 1, 1, 1};
    s.check("K₂,₅ by side: categorical assortativity −1", categorical_assortativity(kb, side) + 1, 1e-15);
    // Directed: the tail's out-degree and the head's in-degree, by hand on a small network.
    const auto dg = network(4, std::vector<Tie>{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {3, 2}}, true);
    // outdeg 3, 1, 0, 1; indeg 0, 1, 3, 1. Pairs (out(tail), in(head)): (3,1), (3,3), (3,1), (1,3), (1,3).
    {
      const std::vector<double> x{3, 3, 3, 1, 1}, y{1, 3, 1, 3, 3};
      double mx = 2.2, my = 2.2, sxy = 0, sxx = 0, syy = 0;
      for (int k = 0; k < 5; k++) sxy += (x[k] - mx) * (y[k] - my), sxx += (x[k] - mx) * (x[k] - mx), syy += (y[k] - my) * (y[k] - my);
      s.check("directed degree assortativity (out, in) by hand", degree_assortativity(dg, DegreeEnds::out_in) - sxy / std::sqrt(sxx * syy), 1e-15);
    }
    s.require("assortativity of a single category is refused", throws([&] { categorical_assortativity(cl, std::vector<std::size_t>(cl.nodes, 0)); }));
  }

  // ---- E–I index and inbreeding homophily: two triangles joined by a tie ----
  {
    const auto g = network(6, std::vector<Tie>{{0, 1}, {1, 2}, {0, 2}, {3, 4}, {4, 5}, {3, 5}, {2, 3}}, false);
    const std::vector<std::size_t> group{0, 0, 0, 1, 1, 1};
    const auto ei = ei_index(g, group);
    s.check("E–I index: E = 1, I = 6, (1 − 6)/7", ei.index + 5.0 / 7, 1e-15);
    s.check("E–I of each group: E_g = 1, I_g = 3, (1 − 3)/4", std::fabs(ei.group[0] + 0.5) + std::fabs(ei.group[1] + 0.5), 1e-15);
    s.check("E–I of the complete network on two groups of three: (18 − 12)/30", ei.complete - 0.2, 1e-15);
    const auto h = inbreeding_homophily(g, group);
    s.check("inbreeding homophily: H = 6/7, w = 1/2, IH = (6/7 − 1/2)/(1/2) = 5/7", std::fabs(h.same[0] - 6.0 / 7) + std::fabs(h.inbreeding[0] - 5.0 / 7) + std::fabs(h.inbreeding[1] - 5.0 / 7), 1e-15);
    const auto all_in = ei_index(network(6, std::vector<Tie>{{0, 1}, {3, 4}}, false), group);
    const auto all_out = ei_index(network(6, std::vector<Tie>{{0, 3}, {1, 4}}, false), group);
    s.require("E–I is −1 when every tie is internal and +1 when every tie is external", all_in.index == -1 && all_out.index == 1);
    // Homophily at chance: a complete network has H_g = (n_g − 1)/(n − 1), below w_g.
    std::vector<Tie> kt;
    for (std::size_t a = 0; a < 6; a++)
      for (std::size_t b = a + 1; b < 6; b++) kt.push_back({a, b});
    const auto hk = inbreeding_homophily(network(6, kt, false), group);
    s.check("the complete network: H_g = (n_g − 1)/(n − 1) = 2/5, IH = (2/5 − 1/2)/(1/2) = −1/5", std::fabs(hk.same[0] - 0.4) + std::fabs(hk.inbreeding[0] + 0.2), 1e-15);
  }

  // ---- brokerage ----
  {
    std::vector<Tie> st;
    for (std::size_t k = 1; k <= 5; k++) st.push_back({0, k});
    const auto bs = brokerage(network(6, st, false));
    s.check("a star of five leaves: the centre's constraint 5 · (1/5)² = 1/5", bs.constraint[0] - 0.2, 1e-15);
    s.check("a star: the centre's effective size 5, efficiency 1", std::fabs(bs.effective_size[0] - 5) + std::fabs(bs.efficiency[0] - 1), 1e-15);
    s.check("a star: each leaf's constraint 1, effective size 1", std::fabs(bs.constraint[3] - 1) + std::fabs(bs.effective_size[3] - 1), 1e-15);
    const auto bt = brokerage(network(3, std::vector<Tie>{{0, 1}, {1, 2}, {0, 2}}, false));
    s.check("a closed triad: constraint 2 · (½ + ½·½)² = 9/8 at each node", std::fabs(bt.constraint[0] - 9.0 / 8) + std::fabs(bt.constraint[2] - 9.0 / 8), 1e-15);
    s.check("a closed triad: effective size 2 − 2·1/2 = 1", bt.effective_size[1] - 1, 1e-15);
    const auto iso = brokerage(network(3, std::vector<Tie>{{0, 1}}, false));
    s.require("an isolated node has no constraint (NaN)", std::isnan(iso.constraint[2]) && std::isnan(iso.effective_size[2]));
    double borgatti = 0;
    for (int t = 0; t < 20; t++) {
      const auto g = random_network(rng, 10 + rng.below(20), 0.1 + 0.3 * rng.uniform(), false);
      const auto b = brokerage(g);
      const auto nb = neighbourhoods(g);
      for (std::size_t v = 0; v < g.nodes; v++) {
        if (nb[v].empty()) continue;
        double ties = 0;  // ties among v's contacts
        for (std::size_t a : nb[v])
          for (std::size_t c : nb[v])
            if (a < c && g.tied(a, c)) ties += 1;
        const double n = static_cast<double>(nb[v].size());
        borgatti = std::max(borgatti, std::fabs(b.effective_size[v] - (n - 2 * ties / n)));
      }
    }
    s.check("effective size equals Borgatti's n − 2t/n on unweighted undirected networks (20 random networks)", borgatti, 1e-13);
  }

  // ---- cross-checks against networkx ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/networkx_networks.txt");
    if (!in) {
      s.skip("networkx fixture not found");
    } else {
      std::map<std::string, Network> nets;
      std::map<std::string, std::vector<std::size_t>> cats;
      std::map<std::string, std::vector<double>> vals;
      std::string line;
      double mix = 0, brok = 0;
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
        if (kind == "category" || kind == "value") {
          std::string list;
          ls >> list;
          const auto v = parse_list(list);
          if (kind == "value") vals[name] = v;
          else
            for (double c : v) cats[name].push_back(static_cast<std::size_t>(c));
          continue;
        }
        if (kind != "scalar" && kind != "vector") continue;
        std::string stat, list;
        ls >> stat >> list;
        const auto& g = nets.at(name);
        const auto want = parse_list(list);
        if (kind == "scalar") {
          double got;
          if (stat == "degree_assortativity" || stat == "degree_assortativity_out_in") got = degree_assortativity(g, DegreeEnds::out_in);
          else if (stat == "degree_assortativity_in_in") got = degree_assortativity(g, DegreeEnds::in_in);
          else if (stat == "degree_assortativity_out_out") got = degree_assortativity(g, DegreeEnds::out_out);
          else if (stat == "degree_assortativity_strength") got = degree_assortativity(g, DegreeEnds::out_in, true);
          else if (stat == "attribute_assortativity") got = categorical_assortativity(g, cats.at(name));
          else if (stat == "numeric_assortativity") got = numeric_assortativity(g, vals.at(name));
          else continue;
          mix = std::max(mix, std::fabs(got - want[0]));
          rows++;
        } else {
          std::vector<double> got;
          if (stat == "constraint") got = brokerage(g).constraint;
          else if (stat == "effective_size") got = brokerage(g).effective_size;
          else continue;
          for (std::size_t k = 0; k < got.size(); k++) brok = std::max(brok, std::isnan(want[k]) && std::isnan(got[k]) ? 0.0 : std::fabs(got[k] - want[k]));
          rows++;
        }
      }
      s.check("cross-check (networkx 3.7): assortativity by degree (out–in, in–in, out–out, strength), by category and by number", mix, 1e-13);
      s.check("cross-check (networkx 3.7): Burt's constraint and effective size, undirected, directed and weighted", brok, 1e-13);
      s.require("cross-check: every fixture row read", rows == 14, std::to_string(rows) + " rows");
    }
  }

  return s.done();
}
