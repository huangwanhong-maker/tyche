// Absorbing fixed effects: residualising on the indicators of one or more factors.
//
// A factor assigns each observation i a level f(i) among L levels (a unit, a period, an origin,
// a destination). The fixed effects of factors f₁, …, f_F span the columns D = [D₁ … D_F] of their
// indicators. By the theorem of Frisch, Waugh and Lovell (Frisch and Waugh 1933; Lovell 1963), the
// coefficients of a regression of y on [X D] are those of the regression of M_D y on M_D X, where
// M_D = I − D (DᵀWD)⁻ DᵀW is the W-orthogonal projection onto the complement of the span of D; the
// residuals are the same, and so is the sandwich of the coefficients on X (with M_D X in it).
//
// M_D is applied without forming D, by the method of alternating projections (von Neumann 1950;
// Halperin 1962): sweep over the factors, subtracting from the vector its W-weighted mean within
// each level of the factor; the iterates converge to M_D v (Guimarães and Portugal 2010; Gaure
// 2013, which introduced the method to econometrics; Correia 2016 for its acceleration). One
// factor is exact in one sweep. Two factors converge geometrically, at a rate set by the
// connectivity of the bipartite graph of their levels; a complete table with weights that factor
// as a_i b_j (as Poisson fits of two-way tables nearly have) is exact in one sweep. Iteration
// stops when a sweep changes no entry by more than tolerance · max|v|.
//
// The rank of D, the number of parameters the fixed effects absorb, is L for one factor (counting
// levels that occur) and L₁ + L₂ − c for two, c the number of connected components of the
// bipartite graph joining the levels that occur together (Abowd, Creecy and Kramarz 2002; Gaure
// 2013). For three factors and more it is not computed here.
//
// recover_effects() finds the effects themselves from a vector d that lies in the span of D (the
// fitted part, e.g. η − Xβ), by the same sweeps (backfitting): d = Σ_f α_f(f(i)). The effects of
// the second factor are normalised to have their first level occurring at zero in each connected
// component; one factor is unique.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::panel {

namespace detail {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::panel: ") + what);
}
}  // namespace detail

// The levels 0 … levels − 1 of a factor, one per observation.
struct Factor {
  std::vector<std::size_t> level;
  std::size_t levels = 0;
};

// A factor from arbitrary labels, numbered in order of first appearance.
inline Factor factor(std::span<const std::int64_t> labels) {
  Factor f;
  f.level.resize(labels.size());
  std::vector<std::int64_t> sorted(labels.begin(), labels.end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  std::vector<std::size_t> code(sorted.size(), static_cast<std::size_t>(-1));
  for (std::size_t i = 0; i < labels.size(); i++) {
    const auto k = static_cast<std::size_t>(std::lower_bound(sorted.begin(), sorted.end(), labels[i]) - sorted.begin());
    if (code[k] == static_cast<std::size_t>(-1)) code[k] = f.levels++;
    f.level[i] = code[k];
  }
  return f;
}

struct AbsorbOptions {
  double tolerance = 1e-13;  // relative to max|v|
  int max_sweeps = 100000;
};

// The W-weighted mean of v within each level of f (levels of zero weight get 0).
namespace detail {
inline void subtract_level_means(std::vector<double>& v, const Factor& f, std::span<const double> w, std::vector<double>& sum, std::vector<double>& mass,
                                 double& largest) {
  sum.assign(f.levels, 0.0);
  mass.assign(f.levels, 0.0);
  for (std::size_t i = 0; i < v.size(); i++) {
    const double wi = w.empty() ? 1.0 : w[i];
    sum[f.level[i]] += wi * v[i];
    mass[f.level[i]] += wi;
  }
  for (std::size_t l = 0; l < f.levels; l++) {
    sum[l] = mass[l] > 0 ? sum[l] / mass[l] : 0.0;
    largest = std::max(largest, std::fabs(sum[l]));
  }
  for (std::size_t i = 0; i < v.size(); i++) v[i] -= sum[f.level[i]];
}
}  // namespace detail

// M_D v for each column, in place; returns the largest number of sweeps used.
inline int demean(std::vector<std::vector<double>>& columns, const std::vector<Factor>& factors, std::span<const double> w = {}, const AbsorbOptions& options = {}) {
  int most = 0;
  std::vector<double> sum, mass;
  for (auto& v : columns) {
    for (const auto& f : factors) detail::require(f.level.size() == v.size(), "a level for every observation");
    if (factors.empty()) continue;
    double scale = 0;
    for (double x : v) scale = std::max(scale, std::fabs(x));
    if (scale == 0) continue;
    int sweep = 0;
    for (; sweep < options.max_sweeps; sweep++) {
      double largest = 0;
      for (const auto& f : factors) detail::subtract_level_means(v, f, w, sum, mass, largest);
      if (factors.size() == 1 || largest <= options.tolerance * scale) break;
    }
    most = std::max(most, sweep + 1);
  }
  return most;
}

// The number of connected components of the bipartite graph of two factors' levels (levels that
// occur only), by union–find.
inline std::size_t components(const Factor& a, const Factor& b) {
  std::vector<std::size_t> parent(a.levels + b.levels);
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  auto find = [&](std::size_t x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
  };
  std::vector<bool> used(a.levels + b.levels, false);
  for (std::size_t i = 0; i < a.level.size(); i++) {
    const std::size_t u = a.level[i], v = a.levels + b.level[i];
    used[u] = used[v] = true;
    parent[find(u)] = find(v);
  }
  std::size_t c = 0;
  for (std::size_t x = 0; x < parent.size(); x++)
    if (used[x] && find(x) == x) c++;
  return c;
}

// The number of levels of f that occur.
inline std::size_t occurring(const Factor& f) {
  std::vector<bool> seen(f.levels, false);
  for (std::size_t l : f.level) seen[l] = true;
  return static_cast<std::size_t>(std::count(seen.begin(), seen.end(), true));
}

// The rank of the indicators of one or two factors.
inline std::size_t absorbed_rank(const std::vector<Factor>& factors) {
  detail::require(factors.size() <= 2, "the absorbed rank of at most two factors");
  if (factors.empty()) return 0;
  if (factors.size() == 1) return occurring(factors[0]);
  return occurring(factors[0]) + occurring(factors[1]) - components(factors[0], factors[1]);
}

// Is every level of f contained in a single cluster? (Then its effects are nested in the clusters.)
inline bool nested(const Factor& f, std::span<const std::int64_t> clusters) {
  detail::require(clusters.size() == f.level.size(), "a cluster for every observation");
  std::vector<std::int64_t> owner(f.levels);
  std::vector<bool> seen(f.levels, false);
  for (std::size_t i = 0; i < clusters.size(); i++) {
    const std::size_t l = f.level[i];
    if (!seen[l]) seen[l] = true, owner[l] = clusters[i];
    else if (owner[l] != clusters[i]) return false;
  }
  return true;
}

// The effects of each factor, from d in the span of their indicators (backfitting); effects[f][l].
inline std::vector<std::vector<double>> recover_effects(std::span<const double> d, const std::vector<Factor>& factors, std::span<const double> w = {},
                                                        const AbsorbOptions& options = {}) {
  std::vector<std::vector<double>> effects;
  for (const auto& f : factors) {
    detail::require(f.level.size() == d.size(), "a level for every observation");
    effects.emplace_back(f.levels, 0.0);
  }
  if (factors.empty()) return effects;
  std::vector<double> r(d.begin(), d.end()), sum, mass;
  double scale = 0;
  for (double x : r) scale = std::max(scale, std::fabs(x));
  for (int sweep = 0; sweep < options.max_sweeps; sweep++) {
    double largest = 0;
    for (std::size_t k = 0; k < factors.size(); k++) {
      detail::subtract_level_means(r, factors[k], w, sum, mass, largest);
      for (std::size_t l = 0; l < factors[k].levels; l++) effects[k][l] += sum[l];
    }
    if (factors.size() == 1 || largest <= options.tolerance * std::max(scale, 1e-300)) break;
  }
  if (factors.size() == 2) {
    // Normalise: in each component, the first occurring level of the second factor at zero.
    const auto& a = factors[0];
    const auto& b = factors[1];
    std::vector<std::size_t> parent(a.levels + b.levels);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    auto find = [&](std::size_t x) {
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };
    for (std::size_t i = 0; i < a.level.size(); i++) parent[find(a.level[i])] = find(a.levels + b.level[i]);
    std::vector<double> shift(parent.size(), 0.0);
    std::vector<bool> fixed(parent.size(), false);
    for (std::size_t i = 0; i < b.level.size(); i++) {
      const std::size_t root = find(a.levels + b.level[i]);
      if (!fixed[root]) fixed[root] = true, shift[root] = effects[1][b.level[i]];
    }
    for (std::size_t l = 0; l < a.levels; l++) effects[0][l] += shift[find(l)];
    for (std::size_t l = 0; l < b.levels; l++) effects[1][l] -= shift[find(a.levels + l)];
  }
  return effects;
}

}  // namespace tyche::panel
