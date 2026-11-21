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
//
// Nested models (Clark and West 2007, "Approximately normal tests for equal predictive accuracy in nested models", J.
// Econometrics 138, 291–311; decision 0034). When the larger model nests the smaller, under the null that the extra terms are
// zero its forecasts carry the noise of estimating them, so its squared errors are larger in the sample though the models are
// equally good, and the test of equal accuracy is undersized. Their adjustment adds back that noise:
// f_t = e_{1,t}² − (e_{2,t}² − (ŷ_{1,t} − ŷ_{2,t})²) = 2 e_{1,t} (ŷ_{2,t} − ŷ_{1,t}), model 1 the smaller; the statistic f̄ / √(V̂/T),
// V̂ Newey and West's long-run variance with h − 1 lags, is referred to the standard normal one-sided: large values favour the
// larger model.
//
// Conditional predictive ability (Giacomini and White 2006, "Tests of conditional predictive ability", Econometrica 74,
// 1545–1578; decision 0034). The question is not which forecast was better on average but whether the loss differential
// d_{t+τ} can be predicted from what was known when the forecasts were made: with test functions h_t (q of them, the first
// usually 1), Z_{t+τ} = h_t d_{t+τ}, the statistic n Z̄ᵀ Ω̂⁻¹ Z̄ is referred to χ² with q degrees of freedom, Ω̂ = (1/n) Σ Z Zᵀ one
// step ahead and its Newey–West form with τ − 1 lags beyond. With h_t = 1 alone it tests unconditional equality.
#pragma once

#include <tyche/probability/distributions.hpp>
#include <tyche/probability/p_values.hpp>

#include <calx/linalg/cholesky.hpp>

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

struct NestedAccuracy {
  std::size_t n = 0, lags = 0;
  double mse_small = 0, mse_large = 0;
  double mean_adjusted = 0;    // f̄: positive when the larger model forecasts better, once its estimation noise is added back
  double statistic = 0;        // f̄ / √(V̂/T), referred to the standard normal
  double p = 1;                // one-sided: the larger model better
};

// Clark and West's test: `small` the forecasts of the nested model, `large` those of the model that nests it.
inline NestedAccuracy clark_west(std::span<const double> truth, std::span<const double> small, std::span<const double> large, std::size_t h = 1, int lags = -1) {
  const std::size_t T = truth.size();
  if (small.size() != T || large.size() != T) throw std::invalid_argument("tyche::clark_west: two forecasts of every value");
  if (T < 3 || h == 0) throw std::invalid_argument("tyche::clark_west: at least three forecasts, at least one step ahead");
  std::vector<double> f(T);
  NestedAccuracy out;
  out.n = T;
  for (std::size_t t = 0; t < T; t++) {
    const double e1 = truth[t] - small[t], e2 = truth[t] - large[t], gap = small[t] - large[t];
    f[t] = e1 * e1 - (e2 * e2 - gap * gap);
    out.mse_small += e1 * e1, out.mse_large += e2 * e2, out.mean_adjusted += f[t];
  }
  const double Td = static_cast<double>(T);
  out.mse_small /= Td, out.mse_large /= Td, out.mean_adjusted /= Td;
  out.lags = lags >= 0 ? static_cast<std::size_t>(lags) : h - 1;
  const double V = long_run_variance(f, static_cast<int>(out.lags));
  if (!(V > 0)) {
    out.statistic = out.mean_adjusted == 0 ? 0 : std::copysign(std::numeric_limits<double>::infinity(), out.mean_adjusted);
    out.p = out.mean_adjusted > 0 ? 0 : 1;
    return out;
  }
  out.statistic = out.mean_adjusted / std::sqrt(V / Td);
  out.p = 0.5 * std::erfc(out.statistic / std::sqrt(2.0));
  return out;
}

struct ConditionalAbility {
  std::size_t n = 0, q = 0, lags = 0;
  double mean_difference = 0;      // d̄: negative when a forecasts better on average
  std::vector<double> mean_moment; // Z̄, one for each test function
  double statistic = 0;            // n Z̄ᵀ Ω̂⁻¹ Z̄, referred to χ² with q degrees of freedom
  double p = 1;
};

// Giacomini and White's test of a and b forecasting `truth` τ steps ahead, with test functions `h` (one row per forecast, q values
// known when it was made; empty: h_t = (1, d_{t−τ}), the forecasts' first τ values then dropped).
inline ConditionalAbility giacomini_white(std::span<const double> truth, std::span<const double> a, std::span<const double> b,
                                          std::vector<std::vector<double>> h = {}, std::size_t tau = 1) {
  const std::size_t T = truth.size();
  if (a.size() != T || b.size() != T) throw std::invalid_argument("tyche::giacomini_white: two forecasts of every value");
  if (tau == 0) throw std::invalid_argument("tyche::giacomini_white: at least one step ahead");
  std::vector<double> d(T);
  for (std::size_t t = 0; t < T; t++) {
    const double ea = truth[t] - a[t], eb = truth[t] - b[t];
    d[t] = ea * ea - eb * eb;
  }
  std::size_t first = 0;
  if (h.empty()) {
    first = tau;
    if (T <= tau + 2) throw std::invalid_argument("tyche::giacomini_white: too few forecasts for lagged differentials");
    h.assign(T, {});
    for (std::size_t t = tau; t < T; t++) h[t] = {1.0, d[t - tau]};
  } else if (h.size() != T) {
    throw std::invalid_argument("tyche::giacomini_white: test functions for every forecast");
  }
  ConditionalAbility out;
  out.q = h[first].size();
  out.n = T - first;
  out.lags = tau - 1;
  if (out.q == 0 || out.n <= out.q + 1) throw std::invalid_argument("tyche::giacomini_white: at least one test function and more forecasts than them");
  const std::size_t q = out.q, n = out.n;
  std::vector<std::vector<double>> Z(n, std::vector<double>(q));
  out.mean_moment.assign(q, 0.0);
  for (std::size_t k = 0; k < n; k++) {
    const std::size_t t = first + k;
    if (h[t].size() != q) throw std::invalid_argument("tyche::giacomini_white: as many test functions for every forecast");
    for (std::size_t j = 0; j < q; j++) Z[k][j] = h[t][j] * d[t], out.mean_moment[j] += Z[k][j] / static_cast<double>(n);
    out.mean_difference += d[t] / static_cast<double>(n);
  }
  // The moments' covariance: (1/n) Σ Z Zᵀ, uncentred under the null that E[Z] = 0, with Bartlett-weighted autocovariances beyond
  // one step.
  calx::linalg::Matrix W(q);
  auto add = [&](std::size_t lag, double weight) {
    for (std::size_t k = lag; k < n; k++)
      for (std::size_t i = 0; i < q; i++)
        for (std::size_t j = 0; j < q; j++) {
          const double v = weight * Z[k][i] * Z[k - lag][j] / static_cast<double>(n);
          W(i, j) += v;
          if (lag > 0) W(j, i) += v;
        }
  };
  add(0, 1.0);
  for (std::size_t L = 1; L <= out.lags && L < n; L++) add(L, 1 - static_cast<double>(L) / static_cast<double>(out.lags + 1));
  const auto C = calx::linalg::cholesky(W);
  if (!C.positive_definite)
    throw std::invalid_argument("tyche::giacomini_white: the moments' covariance is singular (a test function repeats another, or the losses are equal)");
  const auto x = calx::linalg::cholesky_solve(C, out.mean_moment);
  double quad = 0;
  for (std::size_t j = 0; j < q; j++) quad += out.mean_moment[j] * x[j];
  out.statistic = static_cast<double>(n) * quad;
  out.p = tyche::probability::ChiSquared{static_cast<double>(q)}.survival(out.statistic);
  return out;
}

}  // namespace tyche::time_series
