// Spatial autocorrelation: whether values at neighbouring places resemble each other, globally (Moran's I) and
// place by place (local Moran's I).
//
// Weights. W is a sparse matrix of weights w_ij ≥ 0, w_ii = 0, by rows. `k_nearest_weights` gives each point its k
// nearest in the plane, each weighing 1/k, so that W is row-standardised (and not symmetric). Any other neighbours
// (cells sharing an edge, polygons sharing a boundary) come as rows through `weights_from_rows`.
//
// Moran's I (Moran 1950, Biometrika 37, 17–23; Cliff and Ord 1981, Spatial Processes, ch. 1–2). With z_i = x_i − x̄
// and S0 = Σ_ij w_ij,
//   I = (n / S0) Σ_ij w_ij z_i z_j / Σ_i z_i².
// Under the null of no association E I = −1/(n − 1). Its variance, with S1 = ½ Σ_ij (w_ij + w_ji)² and
// S2 = Σ_i (w_i· + w_·i)², is, for normal values,
//   Var_N I = (n² S1 − n S2 + 3 S0²) / ((n² − 1) S0²) − (E I)²,
// and under random assignment of the values to the places, with b = (Σ z⁴ / n) / (Σ z² / n)²,
//   Var_R I = (n ((n² − 3n + 3) S1 − n S2 + 3 S0²) − b ((n² − n) S1 − 2n S2 + 6 S0²)) / ((n − 1)(n − 2)(n − 3) S0²) − (E I)²
// (Cliff and Ord 1981, eqs. 1.38–1.41). z-scores and two-sided p-values are read from the standard normal.
//
// The permutation test assigns the values at random to the places `permutations` times from a seeded calx stream
// (Besag and Diggle 1977); its p-value is one-sided in the direction the observed value lies from the null's,
// (1 + #{as extreme}) / (1 + permutations) (Davison and Hinkley 1997, ch. 4).
//
// Local Moran's I (Anselin 1995, Geogr. Anal. 27, 93–115). I_i = (n − 1) z_i Σ_j w_ij z_j / Σ_k z_k², whose sum
// over the places is (n − 1) S0 / n × I when W is row-standardised, S0 = n, that is (n − 1) I. Its quadrant says what
// kind of association it is: 1 high among high, 2 low among high, 3 low among low, 4 high among low. Its p-value comes
// from conditional permutation: the place's own value is held, and the rest are assigned to its neighbours at random;
// it is not corrected for the many places tested at once (tyche's `testing` does that).
//
// The tests assume nothing of the values' distribution but are exchangeable: where a common trend runs across the
// map, I measures the trend and not contagion, so a regression's residuals are the thing to test. Agreement with PySAL's
// esda (Rey and Anselin 2007) is held by tests/fixtures/make_autocorrelation_fixtures.py.

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::spatial {

// Rows of a sparse matrix of weights: row i holds entries row_start[i] … row_start[i + 1] − 1.
struct SpatialWeights {
  std::vector<std::size_t> row_start{0};
  std::vector<std::size_t> column;
  std::vector<double> weight;
  std::size_t size() const { return row_start.size() - 1; }
};

struct WeightEntry {
  std::size_t to;
  double weight;
};

// Weights from each place's neighbours and their weights. `standardise` makes each row sum to one (a place with no
// neighbour keeps an empty row).
inline SpatialWeights weights_from_rows(const std::vector<std::vector<WeightEntry>>& rows, bool standardise = true) {
  SpatialWeights W;
  W.row_start.reserve(rows.size() + 1);
  for (std::size_t i = 0; i < rows.size(); i++) {
    double total = 0;
    for (const auto& e : rows[i]) {
      if (e.to >= rows.size()) throw std::invalid_argument("tyche::spatial::weights_from_rows: a neighbour beyond the places");
      if (e.to == i) throw std::invalid_argument("tyche::spatial::weights_from_rows: a place is not its own neighbour");
      if (!(e.weight >= 0)) throw std::invalid_argument("tyche::spatial::weights_from_rows: weights are not negative");
      total += e.weight;
    }
    for (const auto& e : rows[i]) {
      W.column.push_back(e.to);
      W.weight.push_back(standardise && total > 0 ? e.weight / total : e.weight);
    }
    W.row_start.push_back(W.column.size());
  }
  return W;
}

// Each point's k nearest others, each weighing 1/k, found by a sweep over the points sorted in x. The points lie in the
// plane, or in space if z is given: unit vectors then find neighbours on the sphere exactly, chord length being monotone
// in the angle between them.
inline SpatialWeights k_nearest_weights(std::span<const double> x, std::span<const double> y, std::span<const double> z, std::size_t k) {
  const std::size_t n = x.size();
  if (y.size() != n || (!z.empty() && z.size() != n)) throw std::invalid_argument("tyche::spatial::k_nearest_weights: the coordinates differ in length");
  if (k == 0 || k >= n) throw std::invalid_argument("tyche::spatial::k_nearest_weights: k must lie between 1 and the number of points less one");
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  std::vector<std::size_t> at(n);
  for (std::size_t p = 0; p < n; p++) at[order[p]] = p;
  std::vector<std::vector<WeightEntry>> rows(n);
  std::vector<std::pair<double, std::size_t>> best;
  for (std::size_t i = 0; i < n; i++) {
    best.clear();
    const auto consider = [&](std::size_t j) {
      const double dx = x[j] - x[i], dy = y[j] - y[i], dz = z.empty() ? 0.0 : z[j] - z[i], d = dx * dx + dy * dy + dz * dz;
      if (best.size() < k) {
        best.emplace_back(d, j);
        std::push_heap(best.begin(), best.end());
      } else if (std::pair(d, j) < best.front()) {
        std::pop_heap(best.begin(), best.end());
        best.back() = {d, j};
        std::push_heap(best.begin(), best.end());
      }
    };
    const auto outside = [&](std::size_t j) {
      const double dx = x[j] - x[i];
      return best.size() == k && dx * dx > best.front().first;
    };
    for (std::size_t p = at[i] + 1; p < n && !outside(order[p]); p++) consider(order[p]);
    for (std::size_t p = at[i]; p > 0 && !outside(order[p - 1]); p--) consider(order[p - 1]);
    auto& row = rows[i];
    row.reserve(k);
    for (const auto& [d, j] : best) row.push_back({j, 1.0 / static_cast<double>(k)});
    std::sort(row.begin(), row.end(), [](const WeightEntry& a, const WeightEntry& b) { return a.to < b.to; });
  }
  return weights_from_rows(rows, false);
}
inline SpatialWeights k_nearest_weights(std::span<const double> x, std::span<const double> y, std::size_t k) { return k_nearest_weights(x, y, {}, k); }

struct Moran {
  double I = 0, expected = 0;
  double variance_normal = 0, z_normal = 0, p_normal = 1;
  double variance_randomisation = 0, z_randomisation = 0, p_randomisation = 1;
  double p_permutation = std::numeric_limits<double>::quiet_NaN();
  std::size_t n = 0, permutations = 0;
};

struct LocalMoran {
  std::vector<double> I;        // one a place
  std::vector<int> quadrant;    // 1 high–high, 2 low–high, 3 low–low, 4 high–low; 0 where a place has no neighbour
  std::vector<double> p;        // by conditional permutation, uncorrected for many tests
  std::size_t permutations = 0;
};

namespace autocorrelation_detail {
inline std::vector<double> centred(std::span<const double> x) {
  double mean = 0;
  for (double v : x) {
    if (!std::isfinite(v)) throw std::invalid_argument("tyche::spatial: values must be finite");
    mean += v;
  }
  mean /= static_cast<double>(x.size());
  std::vector<double> z(x.size());
  for (std::size_t i = 0; i < x.size(); i++) z[i] = x[i] - mean;
  return z;
}
inline double two_sided(double z) { return std::erfc(std::abs(z) / std::sqrt(2.0)); }
inline double quadratic(const SpatialWeights& W, std::span<const double> z) {
  double s = 0;
  for (std::size_t i = 0; i < W.size(); i++)
    for (std::size_t e = W.row_start[i]; e < W.row_start[i + 1]; e++) s += z[i] * W.weight[e] * z[W.column[e]];
  return s;
}
}  // namespace autocorrelation_detail

inline Moran morans_i(std::span<const double> x, const SpatialWeights& W, std::size_t permutations = 0, std::uint64_t seed = 1) {
  namespace d = autocorrelation_detail;
  const std::size_t n = x.size();
  if (W.size() != n) throw std::invalid_argument("tyche::spatial::morans_i: one row of weights a value");
  if (n < 4) throw std::invalid_argument("tyche::spatial::morans_i: at least four places");
  const auto z = d::centred(x);
  double m2 = 0, m4 = 0;
  for (double v : z) {
    m2 += v * v;
    m4 += v * v * v * v;
  }
  if (!(m2 > 0)) throw std::invalid_argument("tyche::spatial::morans_i: the values do not vary");
  const double N = static_cast<double>(n);
  // S0, S1 and S2: S1 pairs each entry with its transpose, read by an index of the entries by (i, j).
  double S0 = 0;
  std::vector<double> out(n, 0.0), in(n, 0.0);
  std::vector<std::vector<std::pair<std::size_t, double>>> reverse(n);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t e = W.row_start[i]; e < W.row_start[i + 1]; e++) {
      S0 += W.weight[e];
      out[i] += W.weight[e];
      in[W.column[e]] += W.weight[e];
      reverse[W.column[e]].emplace_back(i, W.weight[e]);
    }
  if (!(S0 > 0)) throw std::invalid_argument("tyche::spatial::morans_i: no place has a neighbour");
  double S1 = 0;
  for (std::size_t i = 0; i < n; i++) {
    std::sort(reverse[i].begin(), reverse[i].end());
    for (std::size_t e = W.row_start[i]; e < W.row_start[i + 1]; e++) {
      const auto j = W.column[e];
      // w_ji, if j's row holds i: reverse[i] lists who points at i.
      const auto it = std::lower_bound(reverse[i].begin(), reverse[i].end(), std::pair(j, -std::numeric_limits<double>::infinity()));
      const double back = it != reverse[i].end() && it->first == j ? it->second : 0.0;
      const double sum = W.weight[e] + back;
      S1 += sum * sum;
      // An entry with no partner stands for both (i, j) and (j, i) in ½ Σ (w_ij + w_ji)²; one with a partner is met twice.
      if (back == 0.0) S1 += sum * sum;
    }
  }
  S1 *= 0.5;
  double S2 = 0;
  for (std::size_t i = 0; i < n; i++) S2 += (out[i] + in[i]) * (out[i] + in[i]);

  Moran M;
  M.n = n;
  M.I = N / S0 * d::quadratic(W, z) / m2;
  M.expected = -1.0 / (N - 1.0);
  const double E2 = M.expected * M.expected, S02 = S0 * S0;
  M.variance_normal = (N * N * S1 - N * S2 + 3 * S02) / ((N * N - 1) * S02) - E2;
  const double b = (m4 / N) / ((m2 / N) * (m2 / N));
  M.variance_randomisation = (N * ((N * N - 3 * N + 3) * S1 - N * S2 + 3 * S02) - b * ((N * N - N) * S1 - 2 * N * S2 + 6 * S02)) /
                                 ((N - 1) * (N - 2) * (N - 3) * S02) -
                             E2;
  M.z_normal = (M.I - M.expected) / std::sqrt(M.variance_normal);
  M.z_randomisation = (M.I - M.expected) / std::sqrt(M.variance_randomisation);
  M.p_normal = d::two_sided(M.z_normal);
  M.p_randomisation = d::two_sided(M.z_randomisation);
  if (permutations > 0) {
    calx::random::Stream rng(seed, 0x4D4F52u /* "MOR" */, 0, 0);
    auto values = z;
    std::size_t above = 0;
    for (std::size_t p = 0; p < permutations; p++) {
      for (std::size_t k = n; k > 1; k--) std::swap(values[k - 1], values[rng.below(k)]);
      if (N / S0 * d::quadratic(W, values) / m2 >= M.I) above++;
    }
    const std::size_t extreme = std::min(above, permutations - above);
    M.p_permutation = static_cast<double>(extreme + 1) / static_cast<double>(permutations + 1);
    M.permutations = permutations;
  }
  return M;
}

inline LocalMoran local_morans_i(std::span<const double> x, const SpatialWeights& W, std::size_t permutations = 499, std::uint64_t seed = 1) {
  namespace d = autocorrelation_detail;
  const std::size_t n = x.size();
  if (W.size() != n) throw std::invalid_argument("tyche::spatial::local_morans_i: one row of weights a value");
  if (n < 4) throw std::invalid_argument("tyche::spatial::local_morans_i: at least four places");
  const auto z = d::centred(x);
  double m2 = 0;
  for (double v : z) m2 += v * v;
  if (!(m2 > 0)) throw std::invalid_argument("tyche::spatial::local_morans_i: the values do not vary");
  const double scale = (static_cast<double>(n) - 1.0) / m2;

  LocalMoran L;
  L.I.resize(n);
  L.quadrant.assign(n, 0);
  L.p.assign(n, std::numeric_limits<double>::quiet_NaN());
  L.permutations = permutations;
  calx::random::Stream rng(seed, 0x4C4D4Fu /* "LMO" */, 0, 0);
  std::vector<double> others(n > 0 ? n - 1 : 0);
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t first = W.row_start[i], last = W.row_start[i + 1];
    double lag = 0;
    for (std::size_t e = first; e < last; e++) lag += W.weight[e] * z[W.column[e]];
    L.I[i] = scale * z[i] * lag;
    if (first == last) continue;
    L.quadrant[i] = z[i] >= 0 ? (lag >= 0 ? 1 : 4) : (lag >= 0 ? 2 : 3);
    if (permutations == 0) continue;
    // The place's own value held, and a partial shuffle of the rest gives its neighbours theirs.
    std::size_t c = 0;
    for (std::size_t j = 0; j < n; j++)
      if (j != i) others[c++] = z[j];
    const std::size_t degree = last - first;
    std::size_t above = 0;
    for (std::size_t p = 0; p < permutations; p++) {
      double sim = 0;
      for (std::size_t k = 0; k < degree; k++) {
        std::swap(others[k], others[k + rng.below(others.size() - k)]);
        sim += W.weight[first + k] * others[k];
      }
      if (scale * z[i] * sim >= L.I[i]) above++;
    }
    const std::size_t extreme = std::min(above, permutations - above);
    L.p[i] = static_cast<double>(extreme + 1) / static_cast<double>(permutations + 1);
  }
  return L;
}

}  // namespace tyche::spatial
