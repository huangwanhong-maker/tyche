// Estimates of a quantity where it was not observed, from what was observed beside it, and the places where no estimate
// should be given (decision 0028).
//
// A quantity is observed at some units (training units) and wanted at others (targets); covariates are known at both. A linear
// model of the covariates (ridge on standardised covariates, λ = 0 being least squares) is fitted to the training units, and
// three things are asked of it before any estimate is given:
//
//   1. Does it predict at all? Blocked cross-validation leaves out one block of units at a time (a district, a region), so that
//      the units held out are not the near neighbours of those fitted on: random folds on spatially dependent data overstate a
//      model's skill (Roberts et al. 2017, Ecography 40, 913–929). Skill is 1 − SSE/SSE₀, against the mean of the units fitted on
//      for each block. A model whose skill is not above a declared floor (by default 0, the mean) gives no estimate anywhere.
//   2. How far off may it be? The absolute errors of the blocked cross-validation are the scores of a conformal interval: the
//      ⌈(n + 1) level⌉-th smallest of the n scores is the half-width (split conformal prediction, Vovk, Gammerman and Shafer 2005,
//      Algorithmic Learning in a Random World, §2; Lei et al. 2018, JASA 113, 1094–1111). Its coverage is exact for exchangeable
//      units and a model fitted apart from them; here the scores come from cross-validation, so it is approximate, and a check
//      measures it. An estimate whose interval is wider than the use allows (a declared share of the estimate, as statistical
//      offices withhold an estimate whose relative standard error exceeds a threshold) is withheld.
//   3. Is the target like the units it was learnt from? The dissimilarity index of Meyer and Pebesma (2021, Methods in Ecology and
//      Evolution 12, 1620–1633) is the distance in the space of standardised covariates, each weighted by its importance (here the
//      absolute standardised coefficient), from a target to its nearest training unit, over the mean distance between training
//      units. The area of applicability is where it does not exceed the largest training unit's index, computed against the other
//      blocks, that is not an outlier (the upper whisker, below Q3 + 1.5 IQR). Outside it the model extrapolates, and no estimate
//      is given.
//
// An estimate withheld is not a zero and not a missing value to be filled later by someone else's guess: it is a quantity that a
// person must declare, or a record that must be found. The reasons are given unit by unit.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "calx/linalg/dense.hpp"
#include "calx/linalg/rectangular.hpp"
#include "tyche/descriptive/summary.hpp"

namespace tyche::spatial {

using calx::linalg::Rectangular;

struct PredictOptions {
  double ridge = 0;                 // λ on standardised covariates; 0: least squares
  double level = 0.9;               // the coverage of the intervals
  double min_skill = 0;             // the blocked cross-validation's skill must exceed this for any estimate to be given
  double max_relative_halfwidth = std::numeric_limits<double>::infinity();  // withhold an estimate whose half-width exceeds this share of it
  bool log = false;                 // model log y (a positive quantity); estimates and intervals carried back
};

struct Prediction {
  std::vector<double> estimate, lower, upper, dissimilarity;  // per target
  std::vector<char> applicable, withheld;                     // per target
  std::vector<std::string> reason;                            // per target: empty when the estimate is given
  std::vector<double> cv_predicted;                           // per training unit, from the blocks that left it out
  std::vector<double> training_dissimilarity;                 // per training unit, against the other blocks
  std::vector<double> coefficients;                           // intercept first, then one per covariate, on the modelled scale
  std::vector<double> importance;                             // per covariate, summing to one
  double threshold = 0, halfwidth = 0, cv_rmse = 0, cv_skill = 0;
  std::size_t blocks = 0;
  std::string method;
};

namespace detail {

struct Linear {
  std::vector<double> mean, scale, beta;  // standardisation, and β on standardised covariates
  double intercept = 0;
  double predict(const Rectangular& X, std::size_t i) const {
    double v = intercept;
    for (std::size_t j = 0; j < beta.size(); j++)
      if (scale[j] > 0) v += beta[j] * (X(i, j) - mean[j]) / scale[j];
    return v;
  }
};

inline Linear fit_linear(const Rectangular& X, std::span<const double> y, const std::vector<std::size_t>& rows, double ridge) {
  const std::size_t p = X.cols, n = rows.size();
  Linear m;
  m.mean.assign(p, 0.0), m.scale.assign(p, 0.0), m.beta.assign(p, 0.0);
  for (std::size_t i : rows) m.intercept += y[i];
  m.intercept /= static_cast<double>(n);
  for (std::size_t j = 0; j < p; j++) {
    for (std::size_t i : rows) m.mean[j] += X(i, j);
    m.mean[j] /= static_cast<double>(n);
    double s = 0;
    for (std::size_t i : rows) s += (X(i, j) - m.mean[j]) * (X(i, j) - m.mean[j]);
    m.scale[j] = n > 1 ? std::sqrt(s / static_cast<double>(n - 1)) : 0;
  }
  std::vector<std::size_t> active;
  for (std::size_t j = 0; j < p; j++)
    if (m.scale[j] > 1e-300) active.push_back(j);
  const std::size_t k = active.size();
  if (k == 0) return m;
  calx::linalg::Matrix A(k), b(k);
  for (std::size_t a = 0; a < k; a++) {
    for (std::size_t c = 0; c < k; c++) {
      double s = 0;
      for (std::size_t i : rows)
        s += (X(i, active[a]) - m.mean[active[a]]) / m.scale[active[a]] * (X(i, active[c]) - m.mean[active[c]]) / m.scale[active[c]];
      A(a, c) = s;
    }
    A(a, a) += ridge;
    double s = 0;
    for (std::size_t i : rows) s += (X(i, active[a]) - m.mean[active[a]]) / m.scale[active[a]] * (y[i] - m.intercept);
    b(a, 0) = s;
  }
  const calx::linalg::Matrix sol = calx::linalg::solve(A, b);
  for (std::size_t a = 0; a < k; a++) {
    if (!std::isfinite(sol(a, 0))) throw std::invalid_argument("tyche::spatial::predict: the covariates are collinear; give a ridge");
    m.beta[active[a]] = sol(a, 0);
  }
  return m;
}

}  // namespace detail

inline Prediction predict(const Rectangular& X, std::span<const double> y, std::span<const std::int64_t> block, const Rectangular& Xt,
                          const PredictOptions& o = {}) {
  const std::size_t n = X.rows, p = X.cols, T = Xt.rows;
  if (y.size() != n || block.size() != n) throw std::invalid_argument("tyche::spatial::predict: one value and one block per training unit");
  if (Xt.cols != p) throw std::invalid_argument("tyche::spatial::predict: the targets carry the same covariates");
  if (n < p + 2) throw std::invalid_argument("tyche::spatial::predict: too few training units for the covariates");
  if (!(o.level > 0 && o.level < 1)) throw std::invalid_argument("tyche::spatial::predict: a level in (0, 1)");
  std::vector<double> z(n);
  for (std::size_t i = 0; i < n; i++) {
    if (o.log && !(y[i] > 0)) throw std::invalid_argument("tyche::spatial::predict: a logarithm needs positive values");
    z[i] = o.log ? std::log(y[i]) : y[i];
  }
  std::map<std::int64_t, std::vector<std::size_t>> blocks;
  for (std::size_t i = 0; i < n; i++) blocks[block[i]].push_back(i);
  if (blocks.size() < 2) throw std::invalid_argument("tyche::spatial::predict: cross-validation needs at least two blocks");

  Prediction out;
  out.blocks = blocks.size();
  out.method = o.ridge > 0 ? "ridge regression on standardised covariates" : "least squares on standardised covariates";
  if (o.log) out.method += ", of the logarithm";
  std::vector<std::size_t> all(n);
  for (std::size_t i = 0; i < n; i++) all[i] = i;
  const detail::Linear full = detail::fit_linear(X, z, all, o.ridge);

  // Blocked cross-validation.
  out.cv_predicted.assign(n, 0.0);
  double sse = 0, sse0 = 0;
  std::vector<double> scores;
  for (const auto& [b, held] : blocks) {
    std::vector<std::size_t> kept;
    for (std::size_t i = 0; i < n; i++)
      if (block[i] != b) kept.push_back(i);
    const detail::Linear m = detail::fit_linear(X, z, kept, o.ridge);
    for (std::size_t i : held) {
      out.cv_predicted[i] = m.predict(X, i);
      const double e = z[i] - out.cv_predicted[i], e0 = z[i] - m.intercept;
      sse += e * e, sse0 += e0 * e0;
      scores.push_back(std::abs(e));
    }
  }
  out.cv_rmse = std::sqrt(sse / static_cast<double>(n));
  out.cv_skill = sse0 > 0 ? 1 - sse / sse0 : 0;
  std::sort(scores.begin(), scores.end());
  const std::size_t rank = static_cast<std::size_t>(std::ceil((static_cast<double>(n) + 1) * o.level));
  out.halfwidth = rank <= n ? scores[rank - 1] : std::numeric_limits<double>::infinity();
  if (o.log)
    for (double& v : out.cv_predicted) v = std::exp(v);

  // Coefficients on the modelled scale, and the importance of each covariate.
  out.coefficients.assign(p + 1, 0.0);
  out.coefficients[0] = full.intercept;
  double total = 0;
  out.importance.assign(p, 0.0);
  for (std::size_t j = 0; j < p; j++) {
    if (full.scale[j] > 0) {
      out.coefficients[j + 1] = full.beta[j] / full.scale[j];
      out.coefficients[0] -= full.beta[j] * full.mean[j] / full.scale[j];
    }
    out.importance[j] = std::abs(full.beta[j]);
    total += out.importance[j];
  }
  for (double& w : out.importance) w = total > 0 ? w / total : 1.0 / static_cast<double>(p);

  // The dissimilarity index and the area of applicability.
  const auto coords = [&](const Rectangular& M, std::size_t i) {
    std::vector<double> v(p, 0.0);
    for (std::size_t j = 0; j < p; j++)
      if (full.scale[j] > 0) v[j] = out.importance[j] * (M(i, j) - full.mean[j]) / full.scale[j];
    return v;
  };
  std::vector<std::vector<double>> train(n);
  for (std::size_t i = 0; i < n; i++) train[i] = coords(X, i);
  const auto dist = [](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (std::size_t j = 0; j < a.size(); j++) s += (a[j] - b[j]) * (a[j] - b[j]);
    return std::sqrt(s);
  };
  double mean_d = 0;
  std::size_t pairs = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) mean_d += dist(train[i], train[j]), pairs++;
  mean_d = pairs && mean_d > 0 ? mean_d / static_cast<double>(pairs) : 1.0;
  out.training_dissimilarity.assign(n, 0.0);
  for (std::size_t i = 0; i < n; i++) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n; j++)
      if (block[j] != block[i]) best = std::min(best, dist(train[i], train[j]));
    out.training_dissimilarity[i] = best / mean_d;
  }
  {
    const double q1 = descriptive::quantile(out.training_dissimilarity, 0.25), q3 = descriptive::quantile(out.training_dissimilarity, 0.75);
    const double fence = q3 + 1.5 * (q3 - q1);
    out.threshold = 0;
    for (double d : out.training_dissimilarity)
      if (d <= fence) out.threshold = std::max(out.threshold, d);
  }

  // The targets.
  out.estimate.assign(T, 0.0), out.lower = out.upper = out.dissimilarity = out.estimate;
  out.applicable.assign(T, 0), out.withheld.assign(T, 0), out.reason.assign(T, "");
  const bool skilful = out.cv_skill > o.min_skill;
  for (std::size_t t = 0; t < T; t++) {
    const std::vector<double> c = coords(Xt, t);
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; i++) best = std::min(best, dist(c, train[i]));
    out.dissimilarity[t] = best / mean_d;
    out.applicable[t] = out.dissimilarity[t] <= out.threshold ? 1 : 0;
    const double eta = full.predict(Xt, t);
    out.estimate[t] = o.log ? std::exp(eta) : eta;
    out.lower[t] = o.log ? std::exp(eta - out.halfwidth) : eta - out.halfwidth;
    out.upper[t] = o.log ? std::exp(eta + out.halfwidth) : eta + out.halfwidth;
    const double relative = 0.5 * (out.upper[t] - out.lower[t]) / std::abs(out.estimate[t]);
    if (!skilful) {
      out.reason[t] = "the estimator does not beat the mean on blocks held out (skill " + std::to_string(out.cv_skill).substr(0, 6) + ")";
    } else if (!out.applicable[t]) {
      out.reason[t] = "outside the area of applicability: its covariates are farther from those learnt from (index " +
                      std::to_string(out.dissimilarity[t]).substr(0, 5) + ") than any training unit's from the other blocks (" +
                      std::to_string(out.threshold).substr(0, 5) + ")";
    } else if (!(relative <= o.max_relative_halfwidth)) {
      out.reason[t] = "its interval is wider than the use allows (half-width " + std::to_string(relative * 100).substr(0, 5) + "% of the estimate)";
    }
    out.withheld[t] = out.reason[t].empty() ? 0 : 1;
  }
  return out;
}

}  // namespace tyche::spatial
