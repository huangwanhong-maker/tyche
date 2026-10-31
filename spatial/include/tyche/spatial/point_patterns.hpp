// Marked and unmarked point patterns: the co-location quotient of marks, and Clark and Evans's
// nearest-neighbour ratio.
//
// The co-location quotient. N points in the plane carry marks (kinds) 0 … marks − 1, N_A of kind A. The
// co-location quotient of A towards B (Leslie and Kronenfeld 2011, Geogr. Anal. 43, 306–326, eqs. 1–4) is
//   CLQ_{A→B} = (C_{A→B} / N_A) / (N'_B / (N − 1)),   N'_B = N_B − [A = B],
// C_{A→B} = Σ_{i of kind A} (the share of i's nearest neighbours that are of kind B, ties at the nearest
// distance shared equally): the proportion of B among A's nearest neighbours over the proportion of B
// among the others any point could have had. It is 1 in expectation when the marks are a random labelling
// of the points (exactly, with nearest neighbours: E C_{A→B} = N_A N'_B/(N − 1)), above 1 when B gathers
// about A (attraction), below when it keeps away; it is not symmetric. Σ_B C_{A→B} = N_A, so Σ_B CLQ_{A→B} N'_B/(N − 1) = 1.
// With radius > 0 the neighbours of a point are all the others within that distance, weighted equally, and
// a point with none is left out of C and of N_A (the count reported stays the kind's whole count).
//
// Significance by permuting the marks over the points (a Monte Carlo test of random labelling, Besag and
// Diggle 1977, Appl. Statist. 26, 327), `permutations` times from a seeded calx stream; with strata, the
// marks are permuted only within each stratum, so that every stratum keeps its composition: the null of
// marks that share a response to a common terrain (each stratum's mix), against which a CLQ above 1 says
// that B gathers about A beyond what the strata explain. One-sided p-values, upper (attraction) and lower
// (repulsion), each (1 + #{permuted ≥ observed}) / (1 + permutations) (Davison and Hinkley 1997, Bootstrap
// Methods and their Application, ch. 4), ties counted to the observed side; the null's mean and
// standard deviation (divisor permutations − 1) beside them.
//
// Clark and Evans's ratio. For n points in a region of area A, the mean distance r̄ from each point to its
// nearest neighbour, against its expectation under complete spatial randomness (a Poisson process),
// E r̄ = 0.5 √(A/n) on the unbounded plane (Clark and Evans 1954, Ecology 35, 445–453): R = r̄ / E r̄, below
// 1 for clustered points, above for regular ones (2 for a square lattice, 2.1491 for a hexagonal one, the
// largest possible). Points near the edge of a bounded region lose neighbours beyond it and so have longer
// distances; for a rectangle of perimeter P, Donnelly's (1978) corrections, fitted to simulations, are
//   E r̄ = 0.5 √(A/n) + (0.0514 + 0.041/√n) P/n,   Var r̄ = 0.0703 A/n² + 0.037 P √(A/n⁵)
// (Donnelly 1978, in Simulation Studies in Archaeology, ed. I. Hodder, Cambridge University Press, 91–95;
// as spatstat's clarkevans.test with correction "Donnelly"), and z = (r̄ − E r̄)/√Var r̄ is about standard
// normal: here R = r̄ / E r̄ with the corrected expectation, and p two-sided.
//
// Neighbours are found by a sweep over the points sorted in x, O(N log N + N k) for k candidates within
// the current nearest distance: fast for patterns of street scale (thousands of points); a permutation
// costs O(number of neighbour pairs).

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::spatial {

namespace detail {

// The neighbours of each point, as compressed rows: point i's are index[start[i] … start[i+1]) with weights
// summing to 1 (none for a point without neighbours).
struct Neighbours {
  std::vector<std::size_t> start, index;
  std::vector<double> weight;
};

// radius = 0: the nearest neighbours (ties at the nearest distance shared); radius > 0: all within it.
inline Neighbours neighbours(std::span<const double> x, std::span<const double> y, double radius) {
  const std::size_t N = x.size();
  std::vector<std::size_t> order(N);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  Neighbours nb;
  nb.start.assign(N + 1, 0);
  std::vector<std::vector<std::size_t>> lists(N);
  const double r2 = radius * radius;
  for (std::size_t p = 0; p < N; p++) {
    const std::size_t i = order[p];
    double best = radius > 0 ? r2 : std::numeric_limits<double>::infinity();
    std::vector<std::size_t>& found = lists[i];
    auto visit = [&](std::size_t j) {
      const double dx = x[j] - x[i], dy = y[j] - y[i], d2 = dx * dx + dy * dy;
      if (radius > 0) {
        if (d2 <= r2) found.push_back(j);
      } else if (d2 < best) {
        best = d2;
        found.assign(1, j);
      } else if (d2 == best) {
        found.push_back(j);
      }
    };
    for (std::size_t q = p + 1; q < N; q++) {
      const double dx = x[order[q]] - x[i];
      if (dx * dx > best) break;
      visit(order[q]);
    }
    for (std::size_t q = p; q-- > 0;) {
      const double dx = x[order[q]] - x[i];
      if (dx * dx > best) break;
      visit(order[q]);
    }
  }
  for (std::size_t i = 0; i < N; i++) {
    nb.start[i + 1] = nb.start[i] + lists[i].size();
    for (std::size_t j : lists[i]) {
      nb.index.push_back(j);
      nb.weight.push_back(1.0 / static_cast<double>(lists[i].size()));
    }
  }
  return nb;
}

// CLQ_{A→B} for every pair, from the neighbours and a labelling; total[B] = N_B of the whole pattern.
inline std::vector<double> colocation_quotients(const Neighbours& nb, std::span<const std::size_t> mark, std::size_t marks, const std::vector<double>& total) {
  const std::size_t N = mark.size();
  std::vector<double> C(marks * marks, 0.0), NA(marks, 0.0);
  for (std::size_t i = 0; i < N; i++) {
    if (nb.start[i + 1] == nb.start[i]) continue;
    const std::size_t A = mark[i];
    NA[A] += 1;
    for (std::size_t e = nb.start[i]; e < nb.start[i + 1]; e++) C[A * marks + mark[nb.index[e]]] += nb.weight[e];
  }
  std::vector<double> clq(marks * marks, std::numeric_limits<double>::quiet_NaN());
  const double others = static_cast<double>(N) - 1;
  for (std::size_t A = 0; A < marks; A++)
    for (std::size_t B = 0; B < marks; B++) {
      const double nb_ = total[B] - (A == B ? 1.0 : 0.0);
      if (NA[A] > 0 && nb_ > 0) clq[A * marks + B] = (C[A * marks + B] / NA[A]) / (nb_ / others);
    }
  return clq;
}

}  // namespace detail

// ---- the co-location quotient ----

struct Colocation {
  std::size_t marks = 0;
  std::vector<double> clq;            // row-major marks × marks, A→B at A*marks + B (NaN where N_A = 0)
  std::vector<double> null_mean, null_sd, p_upper, p_lower;   // same layout
  std::vector<std::size_t> count;     // N_A per mark
  std::size_t permutations = 0;
};

inline Colocation colocation(std::span<const double> x, std::span<const double> y, std::span<const std::size_t> mark, std::size_t marks,
                             std::span<const std::size_t> stratum = {}, double radius = 0, std::size_t permutations = 999, std::uint64_t seed = 1) {
  const std::size_t N = x.size();
  if (y.size() != N || mark.size() != N) throw std::invalid_argument("tyche::spatial::colocation: x, y and the marks of one length");
  if (!stratum.empty() && stratum.size() != N) throw std::invalid_argument("tyche::spatial::colocation: one stratum for each point, or none");
  if (N < 2) throw std::invalid_argument("tyche::spatial::colocation: two points or more");
  if (marks < 1) throw std::invalid_argument("tyche::spatial::colocation: one mark or more");
  if (!(radius >= 0)) throw std::invalid_argument("tyche::spatial::colocation: a radius of zero or more");
  for (std::size_t i = 0; i < N; i++) {
    if (mark[i] >= marks) throw std::invalid_argument("tyche::spatial::colocation: marks below `marks`");
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) throw std::invalid_argument("tyche::spatial::colocation: finite coordinates");
  }
  const detail::Neighbours nb = detail::neighbours(x, y, radius);
  Colocation out;
  out.marks = marks;
  out.permutations = permutations;
  out.count.assign(marks, 0);
  std::vector<double> total(marks, 0.0);
  for (std::size_t m : mark) {
    out.count[m]++;
    total[m] += 1;
  }
  out.clq = detail::colocation_quotients(nb, mark, marks, total);

  // ---- the permutation null ----
  const std::size_t M = marks * marks;
  std::vector<double> sum(M, 0.0), sum2(M, 0.0), ge(M, 0.0), le(M, 0.0), valid(M, 0.0);
  std::vector<std::vector<std::size_t>> groups;
  if (stratum.empty()) {
    groups.emplace_back(N);
    std::iota(groups[0].begin(), groups[0].end(), std::size_t{0});
  } else {
    // Strata by their labels, whatever their values: the points sorted by stratum, cut where it changes.
    std::vector<std::size_t> by(N);
    std::iota(by.begin(), by.end(), std::size_t{0});
    std::stable_sort(by.begin(), by.end(), [&](std::size_t a, std::size_t b) { return stratum[a] < stratum[b]; });
    for (std::size_t k = 0; k < N; k++) {
      if (k == 0 || stratum[by[k]] != stratum[by[k - 1]]) groups.emplace_back();
      groups.back().push_back(by[k]);
    }
  }
  calx::random::Stream rng(seed, 0x434C51u /* "CLQ" */, 0, 0);
  std::vector<std::size_t> perm(mark.begin(), mark.end()), labels;
  for (std::size_t r = 0; r < permutations; r++) {
    for (const auto& g : groups) {
      // Fisher–Yates on the marks of the group's points.
      labels.clear();
      for (std::size_t i : g) labels.push_back(mark[i]);
      for (std::size_t k = labels.size(); k > 1; k--) std::swap(labels[k - 1], labels[rng.below(k)]);
      for (std::size_t k = 0; k < g.size(); k++) perm[g[k]] = labels[k];
    }
    const auto q = detail::colocation_quotients(nb, perm, marks, total);
    for (std::size_t c = 0; c < M; c++) {
      if (!std::isfinite(q[c]) || !std::isfinite(out.clq[c])) continue;
      const double tol = 1e-12 * std::fabs(out.clq[c]);
      valid[c] += 1;
      sum[c] += q[c];
      sum2[c] += q[c] * q[c];
      if (q[c] >= out.clq[c] - tol) ge[c] += 1;
      if (q[c] <= out.clq[c] + tol) le[c] += 1;
    }
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  out.null_mean.assign(M, nan);
  out.null_sd.assign(M, nan);
  out.p_upper.assign(M, nan);
  out.p_lower.assign(M, nan);
  for (std::size_t c = 0; c < M; c++) {
    if (!std::isfinite(out.clq[c]) || valid[c] < 1) continue;
    out.null_mean[c] = sum[c] / valid[c];
    out.null_sd[c] = valid[c] > 1 ? std::sqrt(std::max(0.0, (sum2[c] - valid[c] * out.null_mean[c] * out.null_mean[c]) / (valid[c] - 1))) : 0.0;
    out.p_upper[c] = (1 + ge[c]) / (1 + valid[c]);
    out.p_lower[c] = (1 + le[c]) / (1 + valid[c]);
  }
  return out;
}

// ---- Clark and Evans's ratio ----

struct ClarkEvans { double ratio = 0, expected = 0, observed = 0, z = 0, p = 0; };

inline ClarkEvans clark_evans(std::span<const double> x, std::span<const double> y, double x0, double x1, double y0, double y1) {
  const std::size_t n = x.size();
  if (y.size() != n) throw std::invalid_argument("tyche::spatial::clark_evans: x and y of one length");
  if (n < 2) throw std::invalid_argument("tyche::spatial::clark_evans: two points or more");
  if (!(x1 > x0 && y1 > y0)) throw std::invalid_argument("tyche::spatial::clark_evans: a rectangle x0 < x1, y0 < y1");
  for (std::size_t i = 0; i < n; i++)
    if (!(x[i] >= x0 && x[i] <= x1 && y[i] >= y0 && y[i] <= y1)) throw std::invalid_argument("tyche::spatial::clark_evans: points inside the rectangle");
  const detail::Neighbours nb = detail::neighbours(x, y, 0);
  double sum = 0;
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t j = nb.index[nb.start[i]];  // any of the nearest: their distance is one
    sum += std::hypot(x[j] - x[i], y[j] - y[i]);
  }
  const double dn = static_cast<double>(n), A = (x1 - x0) * (y1 - y0), P = 2 * ((x1 - x0) + (y1 - y0));
  ClarkEvans out;
  out.observed = sum / dn;
  out.expected = 0.5 * std::sqrt(A / dn) + (0.0514 + 0.041 / std::sqrt(dn)) * P / dn;
  const double var = 0.0703 * A / (dn * dn) + 0.037 * P * std::sqrt(A / (dn * dn * dn * dn * dn));
  out.ratio = out.observed / out.expected;
  out.z = (out.observed - out.expected) / std::sqrt(var);
  out.p = std::erfc(std::fabs(out.z) / std::numbers::sqrt2);
  return out;
}

}  // namespace tyche::spatial
