// What a record of losses says about the year: the annual average loss, and how large a loss is to be expected once in so many
// years. Actuaries and flood modellers state risk this way (e.g. Woo, The Mathematics of Natural Catastrophes, 1999, ch. 7; Grossi and
// Kunreuther, Catastrophe Modeling, 2005, ch. 4).
//
// From a simulated catalogue of events over `years` years: the average annual loss is the sum of the losses over the years, exactly; and
// the occurrence curve, the largest loss of a year in order, is read from the years' maxima: the k-th largest has a return period of
// years / k (Weibull's plotting position with the record's length, as a catalogue's event-loss engines give it).
//
// From a published curve only: the average annual loss is the area under the curve of loss against annual exceedance probability,
// AAL = integral of L(p) dp, here by trapezoids between the given return periods, with what is assumed outside them stated:
//   below the first point, the loss falls linearly to `first_loss` at p = 1;
//   beyond the last, the last loss holds to p = 0 if `hold_last`, else nothing is counted.
// Caveat. An occurrence curve counts each year's largest event, so its area is below the true average when years can hold several; an
// aggregate curve counts their sum and overstates a little by the same integration. The two bracket it only roughly.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace tyche::descriptive {

inline double average_annual_loss(const std::vector<double>& losses, double years) {
  if (!(years > 0)) throw std::invalid_argument("exceedance: a record lasts more than no years");
  double sum = 0;
  for (const double v : losses) sum += v;
  return sum / years;
}

// The loss at each return period, linearly in log return period between the points (years / k, k-th largest) of the years' maxima.
// `maxima`: the largest loss of each year that had one. Return periods beyond the record, or below one, are refused.
inline std::vector<double> occurrence_curve(std::vector<double> maxima, double years, const std::vector<double>& return_periods) {
  if (!(years > 0)) throw std::invalid_argument("exceedance: a record lasts more than no years");
  std::sort(maxima.begin(), maxima.end(), std::greater<>());
  std::vector<double> out;
  out.reserve(return_periods.size());
  for (const double T : return_periods) {
    if (!(T >= 1) || T > years) throw std::invalid_argument("exceedance: a return period lies within the record");
    const double k = years / T;     // the rank, fractional
    const auto low = static_cast<std::size_t>(std::floor(k));
    const auto high = low + 1;
    const auto at = [&](std::size_t r) { return r >= 1 && r <= maxima.size() ? maxima[r - 1] : 0.0; };
    if (low == 0) { out.push_back(at(1)); continue; }
    if (std::abs(k - static_cast<double>(low)) < 1e-12) { out.push_back(at(low)); continue; }
    const double a = std::log(years / static_cast<double>(low)), b = std::log(years / static_cast<double>(high)), t = (std::log(T) - b) / (a - b);
    out.push_back(at(high) + t * (at(low) - at(high)));
  }
  return out;
}

struct CurveArea {
  double within = 0;     // between the first and the last return periods
  double below = 0;      // from the first point to p = 1
  double beyond = 0;     // from the last point to p = 0
  double total() const { return within + below + beyond; }
};

// The area under loss against annual exceedance probability p = 1 / T. Return periods are ascending and greater than 1.
inline CurveArea curve_area(const std::vector<double>& return_periods, const std::vector<double>& losses, double first_loss = 0, bool hold_last = true) {
  if (return_periods.size() != losses.size() || return_periods.empty()) throw std::invalid_argument("exceedance: a curve has a loss at each return period");
  CurveArea area;
  for (std::size_t k = 0; k < return_periods.size(); k++) {
    if (!(return_periods[k] > 1)) throw std::invalid_argument("exceedance: return periods exceed one year");
    if (k > 0 && !(return_periods[k] > return_periods[k - 1])) throw std::invalid_argument("exceedance: return periods ascend");
    if (k > 0) area.within += (1 / return_periods[k - 1] - 1 / return_periods[k]) * (losses[k - 1] + losses[k]) / 2;
  }
  area.below = (1 - 1 / return_periods.front()) * (first_loss + losses.front()) / 2;
  if (hold_last) area.beyond = losses.back() / return_periods.back();
  return area;
}

}  // namespace tyche::descriptive
