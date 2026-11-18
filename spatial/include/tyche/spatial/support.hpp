// Change of support: a quantity known on one set of zones carried to another (decision 0028).
//
// Zones of one partition (census areas) rarely coincide with those of another (city blocks, a grid). A quantity known on the
// first is carried to the second through their intersections, the parts, each with its measure (an area) and, when one is
// known, an ancillary amount inside it (residential floor space, dwellings, built area). What carries it is an assumption
// about how the quantity lies within a source zone, and each method states its own:
//
//   areal weighting (extensive)   a source's amount lies in proportion to measure: y_t = Σ_s y_s m_st / M_s, with M_s the source's
//                                 whole measure, so that Σ_t y_t = Σ_s y_s when the targets cover the sources (Goodchild and Lam
//                                 1980, Geo-Processing 1, 297–312)
//   areal weighting (intensive)   a density or rate is the same throughout a source: y_t = Σ_s y_s m_st / Σ_s m_st
//   dasymetric                    a source's amount lies in proportion to an ancillary amount a_st: y_t = Σ_s y_s a_st / Σ_t' a_st'
//                                 (Wright 1936, Geographical Review 26, 103–110; Eicher and Brewer 2001, CaGIS 28, 125–138); a
//                                 source with no ancillary amount inside the targets is carried by measure instead, and counted
//   class densities               the densities d_k of ancillary classes from the sources' totals, y_s ≈ Σ_k d_k A_sk with d ≥ 0, by
//                                 non-negative least squares (the regression approach of Langford 2006, Computers, Environment and
//                                 Urban Systems 30, 161–180; Mennis and Hultgren 2006, CaGIS 33, 179–194): the weights of a
//                                 dasymetric carriage, a_st = Σ_k d_k A_stk
//   pycnophylactic                a smooth surface on a grid whose sums over the zones are the zones' totals exactly (Tobler 1979,
//                                 JASA 74, 519–530): each sweep replaces a cell by the mean of its neighbours within the zones, then
//                                 restores every zone's total, first by adding the shortfall evenly, then, where that leaves a
//                                 negative, by clipping at zero and scaling. It recovers what varies smoothly across the zones'
//                                 borders (a gradient from a centre outward), not what lies within one zone and nowhere near its
//                                 border (a peak in a zone's middle), which it spreads; it converges slowly, in a number of
//                                 sweeps of the order of the grid's cells across
//
// Every method here preserves the sources' totals where the targets cover them: what is carried is not created or lost. The
// assumption is what errs, and no method can say by how much from the sources alone; checks hold each to a fine truth made for
// the purpose.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "calx/linalg/rectangular.hpp"
#include "tyche/regression/nnls.hpp"

namespace tyche::spatial {

using calx::linalg::Rectangular;

// The intersection of a source zone and a target zone: its measure, and the ancillary amount inside it.
struct Part {
  std::size_t source = 0, target = 0;
  double measure = 0;
  double ancillary = 0;
};

struct Carried {
  std::vector<double> values;   // per target
  std::vector<double> given;    // per source: what it gave to the targets
  std::size_t fell_back = 0;    // sources carried by measure because they held no ancillary amount inside the targets
  std::string method;
};

namespace detail {
inline void require_parts(std::size_t sources, std::size_t targets, const std::vector<Part>& parts) {
  for (const Part& p : parts) {
    if (p.source >= sources || p.target >= targets) throw std::invalid_argument("tyche::spatial: a part names a zone that is not given");
    if (!(p.measure >= 0) || !(p.ancillary >= 0)) throw std::invalid_argument("tyche::spatial: a part's measure and ancillary amount are not negative");
  }
}
}  // namespace detail

// An extensive quantity (a count, an amount) carried by measure. `source_measure` gives each source's whole measure; when it is
// empty, a source is taken to lie wholly within the targets.
inline Carried areal_extensive(std::size_t targets, const std::vector<Part>& parts, std::span<const double> y, std::span<const double> source_measure = {}) {
  const std::size_t S = y.size();
  detail::require_parts(S, targets, parts);
  std::vector<double> M(S, 0.0);
  if (source_measure.empty()) {
    for (const Part& p : parts) M[p.source] += p.measure;
  } else {
    if (source_measure.size() != S) throw std::invalid_argument("tyche::spatial::areal_extensive: one measure per source");
    M.assign(source_measure.begin(), source_measure.end());
  }
  Carried out{std::vector<double>(targets, 0.0), std::vector<double>(S, 0.0), 0, "areal weighting of an extensive quantity"};
  for (const Part& p : parts) {
    if (M[p.source] <= 0) continue;
    const double v = y[p.source] * p.measure / M[p.source];
    out.values[p.target] += v;
    out.given[p.source] += v;
  }
  return out;
}

// An intensive quantity (a density, a rate, a share) carried as the measure-weighted mean of the sources a target meets.
inline Carried areal_intensive(std::size_t targets, const std::vector<Part>& parts, std::span<const double> y) {
  detail::require_parts(y.size(), targets, parts);
  std::vector<double> num(targets, 0.0), den(targets, 0.0);
  for (const Part& p : parts) num[p.target] += y[p.source] * p.measure, den[p.target] += p.measure;
  Carried out{std::vector<double>(targets, 0.0), {}, 0, "areal weighting of an intensive quantity"};
  for (std::size_t t = 0; t < targets; t++) out.values[t] = den[t] > 0 ? num[t] / den[t] : std::numeric_limits<double>::quiet_NaN();
  return out;
}

// An extensive quantity carried in proportion to the ancillary amount of each part.
inline Carried dasymetric(std::size_t targets, const std::vector<Part>& parts, std::span<const double> y) {
  const std::size_t S = y.size();
  detail::require_parts(S, targets, parts);
  std::vector<double> A(S, 0.0), M(S, 0.0);
  for (const Part& p : parts) A[p.source] += p.ancillary, M[p.source] += p.measure;
  Carried out{std::vector<double>(targets, 0.0), std::vector<double>(S, 0.0), 0, "dasymetric, by the ancillary amount in each part"};
  std::vector<char> fell(S, 0);
  for (std::size_t s = 0; s < S; s++)
    if (A[s] <= 0 && y[s] != 0) fell[s] = 1, out.fell_back++;
  for (const Part& p : parts) {
    const double share = fell[p.source] ? (M[p.source] > 0 ? p.measure / M[p.source] : 0.0) : (A[p.source] > 0 ? p.ancillary / A[p.source] : 0.0);
    const double v = y[p.source] * share;
    out.values[p.target] += v;
    out.given[p.source] += v;
  }
  return out;
}

struct ClassDensities {
  std::vector<double> density;  // per class, ≥ 0
  std::vector<double> fitted;   // per source: Σ_k d_k A_sk
  double explained = 0;         // 1 − Σ(y − ŷ)² / Σ y², uncentred (the model has no intercept)
};

// The densities of ancillary classes from the sources' totals and the measure of each class in each source (sources × classes).
inline ClassDensities class_densities(const Rectangular& class_measure, std::span<const double> y) {
  if (class_measure.rows != y.size()) throw std::invalid_argument("tyche::spatial::class_densities: one row of class measures per source");
  const auto r = regression::nnls(class_measure, y);
  ClassDensities out;
  out.density = r.x;
  out.fitted.assign(y.size(), 0.0);
  double sse = 0, sst = 0;
  for (std::size_t s = 0; s < y.size(); s++) {
    for (std::size_t k = 0; k < class_measure.cols; k++) out.fitted[s] += class_measure(s, k) * r.x[k];
    sse += (y[s] - out.fitted[s]) * (y[s] - out.fitted[s]);
    sst += y[s] * y[s];
  }
  out.explained = sst > 0 ? 1 - sse / sst : 0;
  return out;
}

// A grid of cells, each in a zone (−1: in none).
struct ZoneGrid {
  std::size_t rows = 0, cols = 0;
  std::vector<int> zone;  // row-major
};

struct Surface {
  std::vector<double> values;  // per cell: the amount in it (0 outside every zone)
  std::size_t sweeps = 0;
  double change = 0;           // the largest change of a cell in the last sweep, relative to the mean amount of a cell
  bool converged = false;
};

inline Surface pycnophylactic(const ZoneGrid& g, std::span<const double> totals, std::size_t most = 20000, double tolerance = 1e-7, bool nonnegative = true) {
  const std::size_t N = g.rows * g.cols, Z = totals.size();
  if (g.zone.size() != N) throw std::invalid_argument("tyche::spatial::pycnophylactic: one zone label per cell");
  std::vector<std::vector<std::size_t>> cells(Z);
  for (std::size_t i = 0; i < N; i++) {
    const int z = g.zone[i];
    if (z >= static_cast<int>(Z)) throw std::invalid_argument("tyche::spatial::pycnophylactic: a cell names a zone with no total");
    if (z >= 0) cells[static_cast<std::size_t>(z)].push_back(i);
  }
  for (std::size_t z = 0; z < Z; z++)
    if (cells[z].empty() && totals[z] != 0) throw std::invalid_argument("tyche::spatial::pycnophylactic: a zone with a total has no cell");
  Surface out;
  out.values.assign(N, 0.0);
  double mean_cell = 0;
  std::size_t inside = 0;
  for (std::size_t z = 0; z < Z; z++) {
    for (std::size_t i : cells[z]) out.values[i] = totals[z] / static_cast<double>(cells[z].size());
    mean_cell += std::abs(totals[z]);
    inside += cells[z].size();
  }
  mean_cell = inside ? mean_cell / static_cast<double>(inside) : 1.0;
  if (mean_cell == 0) mean_cell = 1;
  std::vector<double> next(N);
  const auto restore = [&](std::vector<double>& v) {
    for (std::size_t z = 0; z < Z; z++) {
      if (cells[z].empty()) continue;
      double sum = 0;
      for (std::size_t i : cells[z]) sum += v[i];
      const double add = (totals[z] - sum) / static_cast<double>(cells[z].size());
      for (std::size_t i : cells[z]) v[i] += add;
      if (!nonnegative) continue;
      bool negative = false;
      for (std::size_t i : cells[z]) negative = negative || v[i] < 0;
      if (!negative) continue;
      double kept = 0;
      for (std::size_t i : cells[z]) v[i] = std::max(v[i], 0.0), kept += v[i];
      if (kept > 0) {
        for (std::size_t i : cells[z]) v[i] *= totals[z] / kept;
      } else {
        for (std::size_t i : cells[z]) v[i] = totals[z] / static_cast<double>(cells[z].size());
      }
    }
  };
  for (std::size_t sweep = 0; sweep < most; sweep++) {
    for (std::size_t r = 0; r < g.rows; r++)
      for (std::size_t c = 0; c < g.cols; c++) {
        const std::size_t i = r * g.cols + c;
        if (g.zone[i] < 0) {
          next[i] = 0;
          continue;
        }
        double s = 0;
        int n = 0;
        const auto add = [&](std::size_t j) {
          if (g.zone[j] >= 0) s += out.values[j], n++;
        };
        if (r > 0) add(i - g.cols);
        if (r + 1 < g.rows) add(i + g.cols);
        if (c > 0) add(i - 1);
        if (c + 1 < g.cols) add(i + 1);
        next[i] = n ? s / n : out.values[i];
      }
    restore(next);
    double change = 0;
    for (std::size_t i = 0; i < N; i++) change = std::max(change, std::abs(next[i] - out.values[i]));
    out.values.swap(next);
    out.sweeps = sweep + 1;
    out.change = change / mean_cell;
    if (out.change < tolerance) {
      out.converged = true;
      break;
    }
  }
  return out;
}

}  // namespace tyche::spatial
