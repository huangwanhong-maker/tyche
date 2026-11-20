// Forecasts compared: the naive forecasts every model is held to, rolling one-step forecasts, the long-run variance and the
// test of equal accuracy (decision 0031).
//
// A forecaster here maps the history y_0 … y_{t−1} to a forecast of y_t. Three need no model and are the least a model must
// beat: the random walk (the last value), the historical mean, and the first-order autoregression fitted to the history by
// least squares, y_t = c + φ y_{t−1} (Makridakis et al. 2020, "The M4 Competition: 100,000 time series and 61 forecasting
// methods", Int. J. Forecast. 36, 54–74, on how often the simple ones are not beaten). Rolling forecasts from an origin
// give each forecaster only the history before the value it forecasts.
//
// Equal accuracy (Diebold and Mariano 1995, "Comparing predictive accuracy", J. Bus. Econ. Stat. 13, 253–263): with the
// loss differential d_t = e_{a,t}² − e_{b,t}² of two forecasts' errors, the statistic is d̄ / √(V̂/T), V̂ the long-run
// variance of d; for forecasts h steps ahead the differential is serially correlated to order h − 1, and V̂ takes that many
// autocovariances. Harvey, Leybourne and Newbold's correction (1997, "Testing the equality of prediction mean squared
// errors", Int. J. Forecast. 13, 281–291) multiplies it by √((T + 1 − 2h + h(h − 1)/T)/T) and refers it to Student's t
// with T − 1 degrees of freedom, which keeps the test's size in short samples. The test compares the forecasts, not the
// models: between nested models its null is not the models' (Clark and West 2007 give the test for that case).
//
// The long-run variance by Newey and West's estimator (1987, Econometrica 55, 703–708): γ̂_0 + 2 Σ_{j=1}^{L} (1 − j/(L+1)) γ̂_j,
// never negative, with L = ⌊4 (T/100)^{2/9}⌋ by default (Newey and West 1994's rule of thumb).
#pragma once

#include <tyche/probability/p_values.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

using Forecaster = std::function<double(std::span<const double>)>;

inline double random_walk(std::span<const double> h) {
  if (h.empty()) throw std::invalid_argument("tyche::random_walk: a history of at least one value");
  return h.back();
}

inline double historical_mean(std::span<const double> h) {
  if (h.empty()) throw std::invalid_argument("tyche::historical_mean: a history of at least one value");
  double s = 0;
  for (double v : h) s += v;
  return s / static_cast<double>(h.size());
}

// y_t = c + φ y_{t−1} by least squares on the history; the last value when it has fewer than three values or none varies.
inline double ar1(std::span<const double> h) {
  if (h.empty()) throw std::invalid_argument("tyche::ar1: a history of at least one value");
  const std::size_t n = h.size();
  if (n < 3) return h.back();
  double mx = 0, my = 0;
  for (std::size_t t = 1; t < n; t++) mx += h[t - 1], my += h[t];
  mx /= static_cast<double>(n - 1), my /= static_cast<double>(n - 1);
  double sxx = 0, sxy = 0;
  for (std::size_t t = 1; t < n; t++) sxx += (h[t - 1] - mx) * (h[t - 1] - mx), sxy += (h[t - 1] - mx) * (h[t] - my);
  if (!(sxx > 0)) return h.back();
  const double phi = sxy / sxx, c = my - phi * mx;
  return c + phi * h.back();
}

struct Rolling {
  std::size_t start = 0;
  std::vector<double> truth, forecast;  // y_t and its forecast, t = start … n − 1
  double mse() const {
    double s = 0;
    for (std::size_t i = 0; i < truth.size(); i++) s += (truth[i] - forecast[i]) * (truth[i] - forecast[i]);
    return truth.empty() ? 0 : s / static_cast<double>(truth.size());
  }
};

// One-step forecasts of y_t from y_0 … y_{t−1}, for t from `start` to the end.
inline Rolling rolling(std::span<const double> y, std::size_t start, const Forecaster& f) {
  if (start == 0 || start >= y.size()) throw std::invalid_argument("tyche::rolling: an origin after the first value and before the last");
  Rolling r;
  r.start = start;
  for (std::size_t t = start; t < y.size(); t++) {
    r.truth.push_back(y[t]);
    r.forecast.push_back(f(y.subspan(0, t)));
  }
  return r;
}

// Newey and West's long-run variance of a series (its mean removed); lags < 0: ⌊4 (T/100)^{2/9}⌋.
inline double long_run_variance(std::span<const double> d, int lags = -1) {
  const std::size_t T = d.size();
  if (T < 2) throw std::invalid_argument("tyche::long_run_variance: at least two values");
  double m = 0;
  for (double v : d) m += v;
  m /= static_cast<double>(T);
  const std::size_t L = lags >= 0 ? static_cast<std::size_t>(lags) : static_cast<std::size_t>(std::floor(4 * std::pow(static_cast<double>(T) / 100, 2.0 / 9)));
  auto gamma = [&](std::size_t j) {
    double s = 0;
    for (std::size_t t = j; t < T; t++) s += (d[t] - m) * (d[t - j] - m);
    return s / static_cast<double>(T);
  };
  double v = gamma(0);
  for (std::size_t j = 1; j <= std::min(L, T - 1); j++) v += 2 * (1 - static_cast<double>(j) / static_cast<double>(L + 1)) * gamma(j);
  return std::max(v, 0.0);
}

struct EqualAccuracy {
  std::size_t n = 0, lags = 0;
  double mse_a = 0, mse_b = 0;
  double mean_difference = 0;  // d̄ = mean of e_a² − e_b²: negative when a forecasts better
  double statistic = 0;        // corrected, referred to t with n − 1 degrees of freedom
  double p = 1;                // two-sided
};

// The test of equal accuracy of forecasts a and b of `truth`, h steps ahead; lags < 0: h − 1 autocovariances.
inline EqualAccuracy diebold_mariano(std::span<const double> truth, std::span<const double> a, std::span<const double> b, std::size_t h = 1, int lags = -1) {
  const std::size_t T = truth.size();
  if (a.size() != T || b.size() != T) throw std::invalid_argument("tyche::diebold_mariano: two forecasts of every value");
  if (T < 3 || h == 0) throw std::invalid_argument("tyche::diebold_mariano: at least three forecasts, at least one step ahead");
  std::vector<double> d(T);
  EqualAccuracy out;
  out.n = T;
  for (std::size_t t = 0; t < T; t++) {
    const double ea = truth[t] - a[t], eb = truth[t] - b[t];
    d[t] = ea * ea - eb * eb;
    out.mse_a += ea * ea, out.mse_b += eb * eb, out.mean_difference += d[t];
  }
  out.mse_a /= static_cast<double>(T), out.mse_b /= static_cast<double>(T), out.mean_difference /= static_cast<double>(T);
  out.lags = lags >= 0 ? static_cast<std::size_t>(lags) : h - 1;
  const double V = long_run_variance(d, static_cast<int>(out.lags));
  const double Td = static_cast<double>(T), hd = static_cast<double>(h);
  if (!(V > 0)) {
    out.statistic = out.mean_difference == 0 ? 0 : std::copysign(std::numeric_limits<double>::infinity(), out.mean_difference);
    out.p = out.mean_difference == 0 ? 1 : 0;
    return out;
  }
  const double dm = out.mean_difference / std::sqrt(V / Td);
  out.statistic = dm * std::sqrt((Td + 1 - 2 * hd + hd * (hd - 1) / Td) / Td);
  out.p = tyche::probability::two_sided_t(out.statistic, Td - 1);
  return out;
}

}  // namespace tyche::time_series
