// Layers of a loss, and how well a payout on an index follows a loss (decision 0029).
//
// A LAYER with attachment a and limit l takes of a loss L the part between a and a + l:
//
//     L_[a, a+l] = min(max(L − a, 0), l),
//
// so that a tower of contiguous layers 0 = a_0 < a_1 < … < a_k, the last without limit, splits every loss into parts that sum to it
// exactly. Its expectation is the stop-loss transform taken twice, E L_[a, a+l] = ∫_a^{a+l} P(L > x) dx (Bühlmann, Mathematical
// Methods in Risk Theory, 1970, §4.6; Klugman, Panjer and Willmot, Loss Models, 4th ed., 2012, §3.4 and §8.6): the area of the
// exceedance curve between the two heights. Reserves, contingent credit, insurance and catastrophe bonds are layers of one loss; which
// layer each takes, and what it costs, is a risk manager's to declare; how much of the loss each layer carries is this header's.
//
// From a record of losses over `years` years, read in one of two ways:
//   per year:   each loss is one year's total, the years without one counted as nothing; the variance is the years' own;
//   per event:  each loss is an event of a catalogue whose events arrive as a Poisson process (a catastrophe model's event-loss table);
//               a layer applies to each event on its own, as an occurrence cover does, and the annual layer loss is compound Poisson,
//               with mean Σ L_[a,a+l] / years and variance Σ L_[a,a+l]² / years (Klugman et al., §9.3).
// The rates are what attach or exhaust a year on average: probabilities per year, and per event the expected number of events a year.
//
// From a published exceedance curve only: the same area, under the curve's own conventions (exceedance.hpp's curve_area: the loss
// linear in p = 1/T between points, falling to `first_loss` at p = 1 below the first, held to p = 0 beyond the last if `hold_last`),
// integrated exactly segment by segment, so that the layer from 0 without limit is the curve's area itself.
//
// BASIS RISK. A parametric cover pays P(I) on an index I (a depth at a gauge, a wind speed, a modelled loss), not the loss; its basis
// risk is how far P(I) is from what an indemnity cover would have paid, C. Over paired samples with weights (events with their rates,
// or years): E|P − C|, the root mean square, the shortfall E max(C − P, 0) and the excess E max(P − C, 0), the rates at which a loss
// meets no payout and a payout meets no loss, and the correlation (Clarke et al. 2016, Policy Research Working Paper 7745, World Bank;
// Jensen, Barrett and Mude 2016, American Journal of Agricultural Economics 98, 1450). The payout most used is linear between a trigger
// and an exhaustion of the index, P = l · min(max((I − t)/(e − t), 0), 1); `fit_payout` chooses t < e among the quantiles of the index
// to least squared error, a search on a grid, said to be one.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::descriptive {

inline double layer_loss(double loss, double attachment, double limit) {
  return std::min(std::max(loss - attachment, 0.0), limit);
}

struct Layer {
  double attachment = 0;
  double limit = std::numeric_limits<double>::infinity();
};

enum class Record { per_year, per_event };

struct LayerStatistics {
  double expected = 0;        // the layer's expected loss a year
  double sd = 0;              // its standard deviation a year
  double attachment_rate = 0; // the years (per year) or events a year (per event) whose loss exceeds a
  double exhaustion_rate = 0; // ... whose loss reaches a + l
  double share = 0;           // expected over the record's expected loss a year
};

namespace layers_detail {
inline void check_layer(const Layer& L, const char* who) {
  if (!(L.attachment >= 0 && std::isfinite(L.attachment)) || !(L.limit > 0)) throw std::invalid_argument(std::string(who) + ": a layer attaches at a ≥ 0 with a limit > 0");
}
inline void check_record(const std::vector<double>& losses, double years, Record record, const char* who) {
  if (!(years > 0 && std::isfinite(years))) throw std::invalid_argument(std::string(who) + ": a record lasts more than no years");
  for (double v : losses)
    if (!(v >= 0 && std::isfinite(v))) throw std::invalid_argument(std::string(who) + ": losses are finite and ≥ 0");
  if (record == Record::per_year && static_cast<double>(losses.size()) > years) throw std::invalid_argument(std::string(who) + ": no more yearly losses than years");
}
}  // namespace layers_detail

inline LayerStatistics layer_statistics(const std::vector<double>& losses, double years, Layer layer, Record record = Record::per_event) {
  const char* who = "tyche::descriptive::layer_statistics";
  layers_detail::check_layer(layer, who);
  layers_detail::check_record(losses, years, record, who);
  LayerStatistics out;
  double sum = 0, sum2 = 0, all = 0;
  for (double v : losses) {
    const double x = layer_loss(v, layer.attachment, layer.limit);
    sum += x;
    sum2 += x * x;
    all += v;
    if (v > layer.attachment) out.attachment_rate += 1;
    if (v >= layer.attachment + layer.limit) out.exhaustion_rate += 1;
  }
  out.expected = sum / years;
  out.attachment_rate /= years;
  out.exhaustion_rate /= years;
  out.sd = record == Record::per_event ? std::sqrt(sum2 / years) : std::sqrt(std::max(0.0, sum2 / years - out.expected * out.expected));
  out.share = all > 0 ? sum / all : 0;
  return out;
}

// Each loss split over the tower attaching at 0 = attachments[0] < attachments[1] < …, the last layer without limit: parts[i][k] is
// the part of loss i in layer k; Σ_k parts[i][k] = loss i.
inline std::vector<std::vector<double>> split(const std::vector<double>& losses, const std::vector<double>& attachments) {
  const char* who = "tyche::descriptive::split";
  if (attachments.empty() || attachments.front() != 0) throw std::invalid_argument(std::string(who) + ": a tower begins at 0");
  for (std::size_t k = 1; k < attachments.size(); k++)
    if (!(attachments[k] > attachments[k - 1]) || !std::isfinite(attachments[k])) throw std::invalid_argument(std::string(who) + ": attachments ascend");
  std::vector<std::vector<double>> parts(losses.size(), std::vector<double>(attachments.size()));
  for (std::size_t i = 0; i < losses.size(); i++) {
    if (!(losses[i] >= 0 && std::isfinite(losses[i]))) throw std::invalid_argument(std::string(who) + ": losses are finite and ≥ 0");
    for (std::size_t k = 0; k < attachments.size(); k++) {
      const double limit = k + 1 < attachments.size() ? attachments[k + 1] - attachments[k] : std::numeric_limits<double>::infinity();
      parts[i][k] = layer_loss(losses[i], attachments[k], limit);
    }
  }
  return parts;
}

// The expected layer loss a year from a published curve of losses at return periods (ascending, > 1), under curve_area's conventions.
inline double layer_from_curve(const std::vector<double>& return_periods, const std::vector<double>& losses, Layer layer, double first_loss = 0,
                               bool hold_last = true) {
  const char* who = "tyche::descriptive::layer_from_curve";
  layers_detail::check_layer(layer, who);
  if (return_periods.size() != losses.size() || return_periods.empty()) throw std::invalid_argument(std::string(who) + ": a curve has a loss at each return period");
  for (std::size_t k = 0; k < return_periods.size(); k++) {
    if (!(return_periods[k] > 1)) throw std::invalid_argument(std::string(who) + ": return periods exceed one year");
    if (k > 0 && !(return_periods[k] > return_periods[k - 1])) throw std::invalid_argument(std::string(who) + ": return periods ascend");
  }
  const double a = layer.attachment, l = layer.limit;
  // ∫ over a segment of p of width w, the loss linear from y1 to y2, of min(max(y − a, 0), l): exact, on the pieces between breaks.
  const auto segment = [a, l](double w, double y1, double y2) {
    if (!(w > 0)) return 0.0;
    std::vector<double> u{0, 1};
    if (y1 != y2) {
      for (double level : {a, a + l})
        if (std::isfinite(level)) {
          const double t = (level - y1) / (y2 - y1);
          if (t > 0 && t < 1) u.push_back(t);
        }
    }
    std::sort(u.begin(), u.end());
    const auto g = [&](double t) { return layer_loss(y1 + (y2 - y1) * t, a, l); };
    double area = 0;
    for (std::size_t k = 1; k < u.size(); k++) area += (u[k] - u[k - 1]) * (g(u[k - 1]) + g(u[k])) / 2;
    return w * area;
  };
  double total = segment(1 - 1 / return_periods.front(), first_loss, losses.front());
  for (std::size_t k = 1; k < return_periods.size(); k++) total += segment(1 / return_periods[k - 1] - 1 / return_periods[k], losses[k - 1], losses[k]);
  if (hold_last) total += segment(1 / return_periods.back(), losses.back(), losses.back());
  return total;
}

// ---- basis risk --------------------------------------------------------------------------------------------------------------------

struct Payout {
  double trigger = 0, exhaustion = 1, limit = 1;
  double operator()(double index) const {
    return limit * std::min(std::max((index - trigger) / (exhaustion - trigger), 0.0), 1.0);
  }
};

struct BasisRisk {
  double covered = 0;         // E C: what an indemnity cover would pay, by weight
  double paid = 0;            // E P
  double absolute = 0;        // E |P − C|
  double rmse = 0;            // sqrt E (P − C)²
  double shortfall = 0;       // E max(C − P, 0)
  double excess = 0;          // E max(P − C, 0)
  double missed = 0;          // the weight of samples with C > 0 and P = 0: a loss met by no payout
  double unearned = 0;        // the weight of samples with P > 0 and C = 0: a payout with no loss
  double correlation = 0;     // between P and C (0 where either is constant)
};

// payout[i] and covered[i] of sample i with weight[i] (a rate a year, a probability; all 1 if empty). Expectations are Σ w x.
inline BasisRisk basis_risk(const std::vector<double>& payout, const std::vector<double>& covered, std::vector<double> weight = {}) {
  const char* who = "tyche::descriptive::basis_risk";
  const std::size_t n = payout.size();
  if (covered.size() != n || n == 0) throw std::invalid_argument(std::string(who) + ": a payout and a covered loss for each sample");
  if (weight.empty()) weight.assign(n, 1.0);
  if (weight.size() != n) throw std::invalid_argument(std::string(who) + ": one weight for each sample");
  BasisRisk b;
  double W = 0, sp = 0, sc = 0, spp = 0, scc = 0, spc = 0, sq = 0;
  for (std::size_t i = 0; i < n; i++) {
    const double w = weight[i], p = payout[i], c = covered[i];
    if (!(w >= 0 && std::isfinite(w)) || !std::isfinite(p) || !std::isfinite(c)) throw std::invalid_argument(std::string(who) + ": finite payouts, losses and weights ≥ 0");
    W += w;
    b.paid += w * p;
    b.covered += w * c;
    b.absolute += w * std::fabs(p - c);
    sq += w * (p - c) * (p - c);
    b.shortfall += w * std::max(c - p, 0.0);
    b.excess += w * std::max(p - c, 0.0);
    if (c > 0 && p == 0) b.missed += w;
    if (p > 0 && c == 0) b.unearned += w;
    sp += w * p;
    sc += w * c;
    spp += w * p * p;
    scc += w * c * c;
    spc += w * p * c;
  }
  b.rmse = std::sqrt(sq);
  if (W > 0) {
    const double vp = spp / W - (sp / W) * (sp / W), vc = scc / W - (sc / W) * (sc / W), cv = spc / W - (sp / W) * (sc / W);
    b.correlation = vp > 1e-300 && vc > 1e-300 ? cv / std::sqrt(vp * vc) : 0;
  }
  return b;
}

struct FittedPayout {
  Payout payout;
  BasisRisk risk;
  std::size_t tried = 0;
};

// The trigger and exhaustion, among `grid` quantiles of the index (by weight), that bring P(I) nearest C in weighted squared error.
inline FittedPayout fit_payout(const std::vector<double>& index, const std::vector<double>& covered, double limit, std::vector<double> weight = {},
                               std::size_t grid = 64) {
  const char* who = "tyche::descriptive::fit_payout";
  const std::size_t n = index.size();
  if (covered.size() != n || n < 2) throw std::invalid_argument(std::string(who) + ": an index and a covered loss for each of at least two samples");
  if (!(limit > 0 && std::isfinite(limit))) throw std::invalid_argument(std::string(who) + ": a limit > 0");
  if (grid < 2) throw std::invalid_argument(std::string(who) + ": a grid of at least two quantiles");
  if (weight.empty()) weight.assign(n, 1.0);
  if (weight.size() != n) throw std::invalid_argument(std::string(who) + ": one weight for each sample");
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; i++) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return index[x] < index[y]; });
  double W = 0;
  for (double w : weight) W += w;
  std::vector<double> levels;
  double run = 0;
  std::size_t at = 0;
  for (std::size_t g = 0; g <= grid; g++) {
    const double target = W * static_cast<double>(g) / static_cast<double>(grid);
    while (at + 1 < n && run + weight[order[at]] < target) run += weight[order[at++]];
    levels.push_back(index[order[at]]);
  }
  std::sort(levels.begin(), levels.end());
  levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
  if (levels.size() < 2) throw std::invalid_argument(std::string(who) + ": the index takes at least two values");
  FittedPayout best;
  double least = std::numeric_limits<double>::infinity();
  std::vector<double> paid(n);
  for (std::size_t i = 0; i < levels.size(); i++)
    for (std::size_t j = i + 1; j < levels.size(); j++) {
      const Payout P{levels[i], levels[j], limit};
      double e = 0;
      for (std::size_t k = 0; k < n; k++) {
        const double d = P(index[k]) - covered[k];
        e += weight[k] * d * d;
      }
      best.tried++;
      if (e < least) {
        least = e;
        best.payout = P;
      }
    }
  for (std::size_t k = 0; k < n; k++) paid[k] = best.payout(index[k]);
  best.risk = basis_risk(paid, covered, weight);
  return best;
}

}  // namespace tyche::descriptive
