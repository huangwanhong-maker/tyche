// Brokerage: Burt's constraint, effective size and efficiency, the structural holes about a node.
//
// A node spans a STRUCTURAL HOLE when its contacts are not tied to one another, so that it brokers between
// them (Burt 1992, Structural Holes: The Social Structure of Competition). Burt measures it from the
// proportional strength of ties. With z_ij = w_ij + w_ji the strength of the relation of i and j in
// either direction (for an undirected tie, twice its weight; only ratios matter) and N(i) the contacts of i
// in the underlying graph:
//
//   p_ij = z_ij / Σ_{k∈N(i)} z_ik                the share of i's relations invested in j;
//   c_ij = (p_ij + Σ_{q∈N(i), q≠j} p_iq p_qj)²    the constraint of j on i: direct, and through the
//                                                 contacts q of i that are tied to j;
//   C_i  = Σ_{j∈N(i)} c_ij                        the constraint of i, high when i's contacts are few
//                                                 and tied to one another;
//   E_i  = Σ_{j∈N(i)} (1 − Σ_{q∈N(i), q≠j} p_iq m_jq),   m_jq = z_jq / max_k z_jk,
//                                                 the effective size: the contacts less their redundancy;
//   efficiency = E_i / |N(i)|.
//
// For an unweighted undirected network the effective size reduces to |N(i)| − 2t_i/|N(i)|, t_i the ties
// among i's contacts (Borgatti 1997, Connections 20: 35–38; checked). Worked by hand: the centre of a star
// with k leaves has C = k · (1/k)² = 1/k and E = k; each node of a triangle has c_ij = (½ + ¼)² and
// C = 9/8, E = 1. These are networkx's definitions (constraint, local_constraint, effective_size), against
// which they are cross-checked. An isolated node has no constraint, effective size or efficiency: NaN.

#pragma once

#include <tyche/networks/network.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tyche::networks {

struct Brokerage {
  std::vector<std::size_t> contacts;  // |N(i)|
  std::vector<double> constraint;     // C_i
  std::vector<double> effective_size; // E_i
  std::vector<double> efficiency;     // E_i / |N(i)|
};

namespace detail {

// z_ij = w_ij + w_ji on the underlying neighbourhoods, as sorted lists parallel to them.
inline std::vector<std::vector<double>> mutual_strength(const Network& g, const std::vector<std::vector<std::size_t>>& nb) {
  std::vector<std::vector<double>> z(g.nodes);
  for (std::size_t i = 0; i < g.nodes; i++)
    for (std::size_t j : nb[i]) z[i].push_back(g.weight(i, j) + g.weight(j, i));
  return z;
}

inline double lookup(const std::vector<std::size_t>& keys, const std::vector<double>& values, std::size_t j) {
  const auto it = std::lower_bound(keys.begin(), keys.end(), j);
  return it != keys.end() && *it == j ? values[static_cast<std::size_t>(it - keys.begin())] : 0.0;
}

}  // namespace detail

inline Brokerage brokerage(const Network& g) {
  const auto nb = neighbourhoods(g);
  const auto z = detail::mutual_strength(g, nb);
  const std::size_t n = g.nodes;
  std::vector<std::vector<double>> p(n);  // p_ij over j ∈ N(i)
  std::vector<double> zmax(n, 0.0);
  for (std::size_t i = 0; i < n; i++) {
    double s = 0;
    for (double v : z[i]) s += v, zmax[i] = std::max(zmax[i], v);
    for (double v : z[i]) p[i].push_back(v / s);
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Brokerage b;
  for (std::size_t i = 0; i < n; i++) {
    b.contacts.push_back(nb[i].size());
    if (nb[i].empty()) {
      b.constraint.push_back(nan);
      b.effective_size.push_back(nan);
      b.efficiency.push_back(nan);
      continue;
    }
    double C = 0, E = 0;
    for (std::size_t a = 0; a < nb[i].size(); a++) {
      const std::size_t j = nb[i][a];
      double indirect = 0, redundancy = 0;
      for (std::size_t c = 0; c < nb[i].size(); c++) {
        const std::size_t q = nb[i][c];
        if (q == j) continue;
        const double pqj = detail::lookup(nb[q], p[q], j);
        const double zjq = detail::lookup(nb[j], z[j], q);
        indirect += p[i][c] * pqj;
        redundancy += p[i][c] * (zmax[j] > 0 ? zjq / zmax[j] : 0.0);
      }
      C += (p[i][a] + indirect) * (p[i][a] + indirect);
      E += 1 - redundancy;
    }
    b.constraint.push_back(C);
    b.effective_size.push_back(E);
    b.efficiency.push_back(E / static_cast<double>(nb[i].size()));
  }
  return b;
}

}  // namespace tyche::networks
