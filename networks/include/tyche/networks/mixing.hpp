// Mixing and homophily: whether ties join nodes alike in a category, a number or their degree.
//
// THE PAIRS. Mixing is measured over the ends of ties: in a directed network each tie i → j once, as
// (tail, head) = (i, j); in an undirected one each tie twice, as (i, j) and (j, i), so that the measures
// are symmetric (Newman 2002, 2003). Optionally each pair counts with its tie's weight.
//
// THE MIXING MATRIX of a categorical attribute (Newman 2003, Phys. Rev. E 67: 026126) holds e_ab, the
// share of the pairs whose tail is of category a and head of category b (Σ e_ab = 1; symmetric when
// undirected), with margins a_a = Σ_b e_ab and b_b = Σ_a e_ab. Its ASSORTATIVITY is
//
//   r = (Σ_a e_aa − Σ_a a_a b_a) / (1 − Σ_a a_a b_a),
//
// 1 when every tie joins nodes of one category, 0 when ties fall as the margins alone would put them, and
// negative (down to −Σ a b / (1 − Σ a b)) when they prefer other categories.
//
// THE ASSORTATIVITY OF A NUMBER x (Newman 2003) is the Pearson correlation of x at the two ends of the
// pairs; BY DEGREE (Newman 2002, Phys. Rev. Lett. 89: 208701) the number is the degree. Newman wrote it
// for undirected networks with the remaining degree (degree − 1) at each end, which changes no
// correlation, as
//
//   r = [M⁻¹ Σ j_e k_e − (M⁻¹ Σ ½(j_e + k_e))²] / [M⁻¹ Σ ½(j_e² + k_e²) − (M⁻¹ Σ ½(j_e + k_e))²],
//
// over the M edges with degrees j_e, k_e at their ends: the Pearson correlation over both orientations
// (checked). For a directed network the degrees at the two ends are chosen: the tail's out-degree and the
// head's in-degree by default (Newman 2003; networkx's default), or any other pair.
//
// THE E–I INDEX (Krackhardt and Stern 1988, Soc. Psychol. Q. 51: 123–140) of a partition into groups is
// (E − I)/(E + I), E the ties between groups and I the ties within them (each tie once): −1 when all ties
// are internal, +1 when all are external. For a group, E_g and I_g count the ties incident to its members.
// Its value depends on the groups' sizes: the reference returned is the index of the complete network on
// the same groups, (P_between − P_within)/(P_between + P_within) over the pairs of nodes, which is also the
// ratio of the expected E and I when ties fall on pairs at random.
//
// INBREEDING HOMOPHILY (Coleman 1958, Hum. Organ. 17: 28–36, as Currarini, Jackson and Pin 2009,
// Econometrica 77: 1003–1045, write it): for a group g with population share w_g, and H_g the share of its
// members' ties that go to members of g,
//
//   IH_g = (H_g − w_g) / (1 − w_g),
//
// 0 when g's members tie to g no more than its share of the population, 1 when they tie only to g, negative
// when they tie to g less. In a directed network H_g is read from the ties its members send.

#pragma once

#include <tyche/networks/network.hpp>

#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace tyche::networks {

namespace detail {

// f(tail, head, weight) for each pair of ends: each directed tie once, each undirected tie both ways.
template <class F>
void each_pair(const Network& g, F f) {
  for (std::size_t k = 0; k < g.ties.size(); k++) {
    const auto [a, b] = g.ties[k];
    f(a, b, g.weights[k]);
    if (!g.directed) f(b, a, g.weights[k]);
  }
}

inline std::size_t category_count(std::span<const std::size_t> category, std::size_t categories, std::size_t nodes, const char* where) {
  require(category.size() == nodes, where, "one category per node");
  std::size_t K = categories;
  for (std::size_t c : category) {
    if (categories == 0) K = std::max(K, c + 1);
    else require(c < categories, where, "categories are 0, …, K − 1");
  }
  return K;
}

// Pearson's correlation of (x, y) over weighted pairs.
inline double weighted_pearson(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& w, const char* where) {
  double W = 0, mx = 0, my = 0;
  for (std::size_t k = 0; k < x.size(); k++) W += w[k], mx += w[k] * x[k], my += w[k] * y[k];
  require(W > 0, where, "at least one tie");
  mx /= W, my /= W;
  double sxy = 0, sxx = 0, syy = 0;
  for (std::size_t k = 0; k < x.size(); k++) {
    sxy += w[k] * (x[k] - mx) * (y[k] - my);
    sxx += w[k] * (x[k] - mx) * (x[k] - mx);
    syy += w[k] * (y[k] - my) * (y[k] - my);
  }
  require(sxx > 0 && syy > 0, where, "the values at the ends of the ties vary (otherwise the correlation is undefined)");
  return sxy / std::sqrt(sxx * syy);
}

}  // namespace detail

// e_ab, K × K, summing to 1 (K = categories, or the largest category + 1 when 0).
inline Rectangular mixing_matrix(const Network& g, std::span<const std::size_t> category, std::size_t categories = 0, bool weighted = false) {
  const std::size_t K = detail::category_count(category, categories, g.nodes, "mixing_matrix");
  Rectangular e(K, K);
  double total = 0;
  detail::each_pair(g, [&](std::size_t a, std::size_t b, double w) {
    const double v = weighted ? w : 1.0;
    e(category[a], category[b]) += v;
    total += v;
  });
  require(total > 0, "mixing_matrix", "at least one tie");
  for (double& v : e.a) v /= total;
  return e;
}

// Newman's assortativity of a categorical attribute.
inline double categorical_assortativity(const Network& g, std::span<const std::size_t> category, bool weighted = false) {
  const auto e = mixing_matrix(g, category, 0, weighted);
  double trace = 0, ab = 0;
  for (std::size_t a = 0; a < e.rows; a++) {
    double ra = 0, cb = 0;
    for (std::size_t b = 0; b < e.cols; b++) ra += e(a, b), cb += e(b, a);
    trace += e(a, a);
    ab += ra * cb;
  }
  require(ab < 1 - 1e-15, "categorical_assortativity", "the ties' ends are not all of one category (otherwise r is undefined)");
  return (trace - ab) / (1 - ab);
}

// Newman's assortativity of a numerical attribute: Pearson's correlation at the ends of the ties.
inline double numeric_assortativity(const Network& g, std::span<const double> value, bool weighted = false) {
  require(value.size() == g.nodes, "numeric_assortativity", "one value per node");
  for (double v : value) require(std::isfinite(v), "numeric_assortativity", "finite values");
  std::vector<double> x, y, w;
  detail::each_pair(g, [&](std::size_t a, std::size_t b, double t) {
    x.push_back(value[a]);
    y.push_back(value[b]);
    w.push_back(weighted ? t : 1.0);
  });
  return detail::weighted_pearson(x, y, w, "numeric_assortativity");
}

// Which degree is read at the tail and at the head of a directed tie.
enum class DegreeEnds { out_in, out_out, in_in, in_out };

// Newman's assortativity by degree (by strength, the sum of tie weights, when `strength`); for an
// undirected network the degree at both ends, and `ends` is ignored. Each pair counts once.
inline double degree_assortativity(const Network& g, DegreeEnds ends = DegreeEnds::out_in, bool strength = false) {
  const auto d = degrees(g);
  auto value = [&](std::size_t v, bool out) {
    if (strength) return out ? d.out_strength[v] : d.in_strength[v];
    return static_cast<double>(out ? d.out[v] : d.in[v]);
  };
  const bool tail_out = ends == DegreeEnds::out_in || ends == DegreeEnds::out_out;
  const bool head_out = ends == DegreeEnds::out_out || ends == DegreeEnds::in_out;
  std::vector<double> x, y, w;
  detail::each_pair(g, [&](std::size_t a, std::size_t b, double) {
    x.push_back(value(a, tail_out));
    y.push_back(value(b, head_out));
    w.push_back(1.0);
  });
  return detail::weighted_pearson(x, y, w, "degree_assortativity");
}

struct EiIndex {
  double index = 0;                // (E − I)/(E + I) over every tie
  double external = 0, internal = 0;  // E and I (weighted sums when weighted)
  std::vector<double> group;       // per category: (E_g − I_g)/(E_g + I_g) over the ties incident to it; NaN without ties
  double complete = 0;             // the index of the complete network on the same groups
};

inline EiIndex ei_index(const Network& g, std::span<const std::size_t> category, bool weighted = false) {
  const std::size_t K = detail::category_count(category, 0, g.nodes, "ei_index");
  require(g.size() > 0, "ei_index", "at least one tie");
  EiIndex r;
  std::vector<double> eg(K, 0.0), ig(K, 0.0);
  for (std::size_t k = 0; k < g.ties.size(); k++) {
    const auto [a, b] = g.ties[k];
    const double v = weighted ? g.weights[k] : 1.0;
    const std::size_t ca = category[a], cb = category[b];
    if (ca == cb) {
      r.internal += v;
      ig[ca] += v;
    } else {
      r.external += v;
      eg[ca] += v;
      eg[cb] += v;
    }
  }
  r.index = (r.external - r.internal) / (r.external + r.internal);
  for (std::size_t c = 0; c < K; c++)
    r.group.push_back(eg[c] + ig[c] > 0 ? (eg[c] - ig[c]) / (eg[c] + ig[c]) : std::numeric_limits<double>::quiet_NaN());
  std::vector<double> size(K, 0.0);
  for (std::size_t c : category) size[c] += 1;
  const double n = static_cast<double>(g.nodes);
  double within = 0, between = 0;
  for (double s : size) within += s * (s - 1), between += s * (n - s);
  r.complete = (between - within) / (between + within);
  return r;
}

struct Homophily {
  std::vector<std::size_t> members;  // n_g
  std::vector<double> share;         // w_g = n_g / n
  std::vector<double> same;          // H_g, the share of its members' ties that stay in g (NaN without ties)
  std::vector<double> inbreeding;    // IH_g = (H_g − w_g)/(1 − w_g) (NaN without ties or when w_g = 1)
};

inline Homophily inbreeding_homophily(const Network& g, std::span<const std::size_t> category, bool weighted = false) {
  const std::size_t K = detail::category_count(category, 0, g.nodes, "inbreeding_homophily");
  Homophily h;
  h.members.assign(K, 0);
  for (std::size_t c : category) h.members[c]++;
  std::vector<double> same(K, 0.0), all(K, 0.0);
  detail::each_pair(g, [&](std::size_t a, std::size_t b, double w) {
    const double v = weighted ? w : 1.0;
    all[category[a]] += v;
    if (category[a] == category[b]) same[category[a]] += v;
  });
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t c = 0; c < K; c++) {
    const double w = static_cast<double>(h.members[c]) / static_cast<double>(g.nodes);
    const double H = all[c] > 0 ? same[c] / all[c] : nan;
    h.share.push_back(w);
    h.same.push_back(H);
    h.inbreeding.push_back(all[c] > 0 && w < 1 ? (H - w) / (1 - w) : nan);
  }
  return h;
}

}  // namespace tyche::networks
