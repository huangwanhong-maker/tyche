// Conditional uniform graph tests: a statistic of a network against networks drawn at random with some of
// its features held.
//
// A statistic of an observed network (its transitivity, its reciprocity, an assortativity) means little
// alone: a dense network has more triangles than a sparse one by its density only. The CONDITIONAL UNIFORM
// GRAPH (CUG) test compares it with its law over the networks drawn uniformly from those that share some
// feature of the observed one (Wasserman and Faust 1994, Social Network Analysis, on distributions of
// random graphs; Butts 2008, J. Stat. Softw. 24(6), whose `cug.test` names the conditionings):
//
//   ties      the same nodes and number of ties, uniformly (Erdős and Rényi's G(n, m)); drawn exactly, by
//             choosing m distinct pairs (or the pairs left out, when m exceeds half of them); the draws
//             are unweighted;
//   degrees   the same nodes and degrees (in- and out-degrees when directed), by double-edge swaps from
//             the observed network (Maslov and Sneppen 2002, Science 296: 910–913; Milo et al. 2003,
//             arXiv cond-mat/0312028): two ties (a, b), (c, d) are chosen at random (and, undirected, one
//             of them turned at random) and become (a, d), (c, b) unless that makes a loop or a tie already
//             present. The proposal is symmetric, so the chain's stationary law is uniform on the networks
//             it reaches; for undirected networks the swaps connect every simple graph of the given
//             degrees (Taylor 1981, in Combinatorial Mathematics VIII, Lecture Notes in Math. 884:
//             314–336). Caveat: for directed networks they need not (some digraphs differ by reversing a
//             directed triangle, which no swap does; Rao, Jana and Bandyopadhyay 1996, Sankhyā A 58:
//             225–242), and the number of swaps needed for a draw to forget its start is not known in
//             general: each draw restarts from the observed network and makes `swaps_per_tie` × m attempts,
//             a number reported with the result. Weights travel with the tails of the swapped ties.
//
// P-VALUES. For B draws, P(S ≥ s) as (1 + #{s* ≥ s})/(1 + B) and P(S ≤ s) as (1 + #{s* ≤ s})/(1 + B), the
// observed network counted among the draws, as a Monte Carlo test requires to be exact (Barnard 1963,
// J. R. Stat. Soc. B 25: 294; Besag and Clifford 1989, Biometrika 76: 633–642). (sna's cug.test reports
// #{s* ≥ s}/B instead.) Ties within 10⁻¹² (1 + |s|). Draw r is made on the calx stream (seed, identity
// (0x6e657477, 5, r)), so a test is reproduced by its seed and its draws do not depend on their order.

#pragma once

#include <tyche/networks/network.hpp>

#include <calx/random/stream.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

namespace tyche::networks {

// A network drawn uniformly from those with `nodes` nodes and `ties` ties (no loops, no repeated ties).
inline Network uniform_network(std::size_t nodes, std::size_t ties, bool directed, calx::random::Stream& rng) {
  require(nodes >= 2 || ties == 0, "uniform_network", "ties need at least two nodes");
  const std::uint64_t n = nodes, N = directed ? n * (n - 1) : n * (n - 1) / 2;
  require(ties <= N, "uniform_network", "no more ties than pairs of nodes");
  auto key = [&](std::uint64_t a, std::uint64_t b) { return directed || a < b ? a * n + b : b * n + a; };
  const bool complement = 2 * ties > N;
  const std::uint64_t draw = complement ? N - ties : ties;
  std::unordered_set<std::uint64_t> chosen;
  chosen.reserve(static_cast<std::size_t>(draw) * 2 + 1);
  while (chosen.size() < draw) {
    const std::uint64_t a = rng.below(n);
    std::uint64_t b = rng.below(n - 1);
    if (b >= a) b++;
    chosen.insert(key(a, b));
  }
  std::vector<Tie> t;
  t.reserve(ties);
  if (!complement) {
    for (std::uint64_t k : chosen) t.push_back({static_cast<std::size_t>(k / n), static_cast<std::size_t>(k % n)});
  } else {
    for (std::uint64_t a = 0; a < n; a++)
      for (std::uint64_t b = directed ? 0 : a + 1; b < n; b++)
        if (a != b && !chosen.count(key(a, b))) t.push_back({static_cast<std::size_t>(a), static_cast<std::size_t>(b)});
  }
  return network(nodes, t, directed);
}

// The network after `attempts` double-edge swap attempts from g, keeping every degree (in- and
// out-degrees when directed); `accepted`, when given, receives the number of swaps made.
inline Network degree_preserving(const Network& g, std::size_t attempts, calx::random::Stream& rng, std::size_t* accepted = nullptr) {
  std::vector<Tie> t = g.ties;
  std::vector<double> w = g.weights;
  const std::uint64_t n = g.nodes;
  auto key = [&](std::size_t a, std::size_t b) {
    return g.directed || a < b ? static_cast<std::uint64_t>(a) * n + b : static_cast<std::uint64_t>(b) * n + a;
  };
  std::unordered_set<std::uint64_t> present;
  present.reserve(t.size() * 2 + 1);
  for (const auto& [a, b] : t) present.insert(key(a, b));
  std::size_t done = 0;
  const std::size_t m = t.size();
  if (m >= 2) {
    for (std::size_t s = 0; s < attempts; s++) {
      const std::size_t k1 = static_cast<std::size_t>(rng.below(m));
      std::size_t k2 = static_cast<std::size_t>(rng.below(m - 1));
      if (k2 >= k1) k2++;
      auto [a, b] = t[k1];
      auto [c, d] = t[k2];
      if (!g.directed && rng.below(2)) std::swap(c, d);
      if (a == d || c == b) continue;                                       // a loop
      if (present.count(key(a, d)) || present.count(key(c, b))) continue;   // a tie already present
      present.erase(key(a, b));
      present.erase(key(c, d));
      present.insert(key(a, d));
      present.insert(key(c, b));
      t[k1] = {a, d};
      t[k2] = {c, b};
      done++;
    }
  }
  if (accepted) *accepted = done;
  Network h = g.weighted ? network(g.nodes, t, g.directed, w) : network(g.nodes, t, g.directed);
  return h;
}

enum class Conditioning { ties, degrees };

struct CugOptions {
  Conditioning conditioning = Conditioning::ties;
  std::size_t draws = 1000;
  std::uint64_t seed = 20261113;
  double swaps_per_tie = 10;  // degree-preserving draws: attempts = swaps_per_tie × m
};

struct CugTest {
  double observed = 0;
  std::vector<double> values;          // the statistic of each draw
  double mean = 0, sd = 0;             // of the draws (sd with divisor B − 1)
  double p_greater = 1, p_less = 1;    // (1 + #{s* ≥ s})/(1 + B), (1 + #{s* ≤ s})/(1 + B)
  std::size_t draws = 0, swaps = 0;    // swaps: the attempts per draw (degrees)
  Conditioning conditioning = Conditioning::ties;
};

// The CUG test of a statistic of g.
inline CugTest cug_test(const Network& g, const std::function<double(const Network&)>& statistic, const CugOptions& o = {}) {
  require(o.draws >= 1, "cug_test", "at least one draw");
  require(std::isfinite(o.swaps_per_tie) && o.swaps_per_tie >= 0, "cug_test", "a non-negative number of swaps per tie");
  CugTest c;
  c.conditioning = o.conditioning;
  c.draws = o.draws;
  c.observed = statistic(g);
  c.swaps = o.conditioning == Conditioning::degrees ? static_cast<std::size_t>(std::llround(o.swaps_per_tie * static_cast<double>(g.size()))) : 0;
  const double tol = 1e-12 * (1 + std::fabs(c.observed));
  std::size_t ge = 0, le = 0;
  for (std::size_t r = 0; r < o.draws; r++) {
    calx::random::Stream rng(o.seed, 0x6e657477, 5, static_cast<std::uint32_t>(r));
    const Network h = o.conditioning == Conditioning::ties ? uniform_network(g.nodes, g.size(), g.directed, rng) : degree_preserving(g, c.swaps, rng);
    const double v = statistic(h);
    c.values.push_back(v);
    if (v >= c.observed - tol) ge++;
    if (v <= c.observed + tol) le++;
  }
  double mean = 0;
  for (double v : c.values) mean += v / static_cast<double>(o.draws);
  double ss = 0;
  for (double v : c.values) ss += (v - mean) * (v - mean);
  c.mean = mean;
  c.sd = o.draws > 1 ? std::sqrt(ss / static_cast<double>(o.draws - 1)) : 0.0;
  c.p_greater = (1 + static_cast<double>(ge)) / (1 + static_cast<double>(o.draws));
  c.p_less = (1 + static_cast<double>(le)) / (1 + static_cast<double>(o.draws));
  return c;
}

}  // namespace tyche::networks
