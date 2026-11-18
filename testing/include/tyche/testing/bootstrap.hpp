// The nonparametric bootstrap of any statistic of weighted units, resampling units or the clusters they belong to (decision 0027).
//
// A sample of n units is resampled with replacement B times (Efron 1979, Ann. Statist. 7, 1–26); when units belong to clusters
// (households, areas, schools), whole clusters are resampled, which keeps the dependence within them (the cluster bootstrap;
// Davison and Hinkley 1997, Bootstrap Methods and their Application, §3.8). A resample is given to the statistic as a multiplicity
// per unit, the number of times its unit (or its cluster) was drawn, so that any statistic of weighted units serves: it multiplies
// its weights by the multiplicities. The answer is the B values, their mean and standard deviation (the bootstrap standard error),
// and the percentile interval (Efron and Tibshirani 1993, An Introduction to the Bootstrap, §13.3). A statistic that is not a number
// on a resample is counted and left out.
//
// Draws come from the stream the caller gives, so that a bootstrap is reproduced from its key.
#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace tyche::testing {

struct Bootstrap {
  std::vector<double> values;   // the statistic on each resample that gave a number
  double mean = std::numeric_limits<double>::quiet_NaN();
  double se = std::numeric_limits<double>::quiet_NaN();
  double low = std::numeric_limits<double>::quiet_NaN(), high = std::numeric_limits<double>::quiet_NaN();  // percentile interval
  std::size_t failed = 0;       // resamples on which it was not a number
};

namespace detail {
inline double quantile_sorted(const std::vector<double>& v, double p) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double h = (static_cast<double>(v.size()) - 1) * p;
  const auto lo = static_cast<std::size_t>(std::floor(h));
  const auto hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}
}  // namespace detail

// statistic(multiplicity) → a number; multiplicity has one entry per unit. clusters, when given, has one label per unit.
template <class Statistic>
Bootstrap bootstrap(std::size_t units, Statistic&& statistic, std::size_t resamples, calx::random::Stream& s, double confidence = 0.95,
                    const std::vector<long>& clusters = {}) {
  if (!(confidence > 0 && confidence < 1)) throw std::invalid_argument("bootstrap: a confidence lies strictly between 0 and 1");
  if (!clusters.empty() && clusters.size() != units) throw std::invalid_argument("bootstrap: the clusters are not one per unit");
  Bootstrap out;
  if (units == 0 || resamples == 0) return out;
  // Units of each cluster.
  std::vector<std::vector<std::size_t>> members;
  if (clusters.empty()) {
    members.resize(units);
    for (std::size_t i = 0; i < units; i++) members[i] = {i};
  } else {
    std::map<long, std::size_t> id;
    for (std::size_t i = 0; i < units; i++) {
      auto it = id.find(clusters[i]);
      if (it == id.end()) it = id.emplace(clusters[i], members.size()).first, members.emplace_back();
      members[it->second].push_back(i);
    }
  }
  const std::size_t C = members.size();
  std::vector<double> multiplicity(units);
  for (std::size_t b = 0; b < resamples; b++) {
    std::fill(multiplicity.begin(), multiplicity.end(), 0.0);
    for (std::size_t k = 0; k < C; k++)
      for (std::size_t i : members[s.below(C)]) multiplicity[i] += 1;
    const double v = statistic(multiplicity);
    if (std::isfinite(v)) out.values.push_back(v);
    else out.failed++;
  }
  if (out.values.empty()) return out;
  double sum = 0;
  for (double v : out.values) sum += v;
  out.mean = sum / static_cast<double>(out.values.size());
  if (out.values.size() > 1) {
    double ss = 0;
    for (double v : out.values) ss += (v - out.mean) * (v - out.mean);
    out.se = std::sqrt(ss / static_cast<double>(out.values.size() - 1));
  }
  std::vector<double> sorted = out.values;
  std::sort(sorted.begin(), sorted.end());
  out.low = detail::quantile_sorted(sorted, (1 - confidence) / 2);
  out.high = detail::quantile_sorted(sorted, (1 + confidence) / 2);
  return out;
}

}  // namespace tyche::testing
