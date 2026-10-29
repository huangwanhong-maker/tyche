// The sample autocovariance, autocorrelation and partial autocorrelation functions of a series,
// the Durbin–Levinson recursion, and the Ljung–Box portmanteau test.
//
// - autocovariance: γ̂(h) = (1/n) Σ_{t=1}^{n−h} (x_t − x̄)(x_{t+h} − x̄), h = 0, …, max_lag, with
//   divisor n at every lag (Box, Jenkins and Reinsel 2008, Time Series Analysis, 4th ed., §2.1;
//   Brockwell and Davis 1991, Time Series: Theory and Methods, 2nd ed., §7.2):
//   biased, but the sequence is positive semidefinite, which the divisor n − h does not guarantee.
//   The mean is the compensated, corrected one of tyche::descriptive.
// - acf: r_h = γ̂(h)/γ̂(0), the r(1) of NIST's univariate reference datasets at h = 1.
// - durbin_levinson: the coefficients φ_{p,1..p} of the best linear predictor of order p, the
//   partial autocorrelations φ_kk and the prediction error variances v_k from an autocovariance
//   sequence (Durbin 1960; Brockwell and Davis 1991, Proposition 5.2.1): O(p²).
// - pacf: the φ_hh of the sample autocovariances, h = 0, …, max_lag (φ₀₀ = 1), i.e. the
//   Yule–Walker partial autocorrelations (statsmodels' pacf method "ywm").
// - ljung_box: Q = n(n + 2) Σ_{k=1}^{h} r_k²/(n − k), asymptotically χ² with h − m degrees of
//   freedom when the series is the residual of a fitted ARMA model with m parameters (Ljung and
//   Box 1978, Biometrika 65, 297–303; Box and Pierce 1970 for the unweighted Q).

#pragma once

#include <tyche/descriptive/summary.hpp>
#include <tyche/probability/distributions.hpp>

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

inline std::vector<double> autocovariance(std::span<const double> x, std::size_t max_lag, bool demean = true) {
  const std::size_t n = x.size();
  if (n == 0 || max_lag >= n) throw std::invalid_argument("tyche::time_series::autocovariance: max_lag < n");
  const double m = demean ? descriptive::mean(x) : 0.0;
  std::vector<double> g(max_lag + 1, 0.0);
  for (std::size_t h = 0; h <= max_lag; h++) {
    double s = 0, c = 0;  // Neumaier
    for (std::size_t t = 0; t + h < n; t++) {
      const double v = (x[t] - m) * (x[t + h] - m);
      const double u = s + v;
      c += std::fabs(s) >= std::fabs(v) ? (s - u) + v : (v - u) + s;
      s = u;
    }
    g[h] = (s + c) / static_cast<double>(n);
  }
  return g;
}

inline std::vector<double> acf(std::span<const double> x, std::size_t max_lag) {
  auto g = autocovariance(x, max_lag);
  const double g0 = g[0];
  if (!(g0 > 0)) throw std::invalid_argument("tyche::time_series::acf: a series that is not constant");
  for (double& v : g) v /= g0;
  return g;
}

struct DurbinLevinson {
  std::vector<double> phi;       // φ_{p,1}, …, φ_{p,p}: x̂_{t} = Σ φ_{p,j} x_{t−j} (mean removed)
  std::vector<double> partial;   // φ_11, …, φ_pp
  std::vector<double> variance;  // v_0 = γ(0), …, v_p
};

inline DurbinLevinson durbin_levinson(std::span<const double> gamma, std::size_t order) {
  if (gamma.size() < order + 1) throw std::invalid_argument("tyche::time_series::durbin_levinson: γ(0), …, γ(p)");
  if (!(gamma[0] > 0)) throw std::invalid_argument("tyche::time_series::durbin_levinson: γ(0) > 0");
  DurbinLevinson d;
  d.variance.push_back(gamma[0]);
  std::vector<double> phi;
  for (std::size_t k = 1; k <= order; k++) {
    double num = gamma[k];
    for (std::size_t j = 1; j < k; j++) num -= phi[j - 1] * gamma[k - j];
    const double pkk = num / d.variance.back();
    std::vector<double> next(k);
    for (std::size_t j = 1; j < k; j++) next[j - 1] = phi[j - 1] - pkk * phi[k - j - 1];
    next[k - 1] = pkk;
    phi = std::move(next);
    d.partial.push_back(pkk);
    d.variance.push_back(d.variance.back() * (1 - pkk * pkk));
  }
  d.phi = std::move(phi);
  return d;
}

inline std::vector<double> pacf(std::span<const double> x, std::size_t max_lag) {
  const auto g = autocovariance(x, max_lag);
  const auto d = durbin_levinson(g, max_lag);
  std::vector<double> out{1.0};
  out.insert(out.end(), d.partial.begin(), d.partial.end());
  return out;
}

struct LjungBox {
  double statistic = 0, p_value = 0;
  std::size_t lags = 0;
  double df = 0;  // lags − fitted_parameters
};

inline LjungBox ljung_box(std::span<const double> x, std::size_t lags, std::size_t fitted_parameters = 0) {
  const std::size_t n = x.size();
  if (lags == 0 || lags >= n) throw std::invalid_argument("tyche::time_series::ljung_box: 0 < lags < n");
  if (fitted_parameters >= lags) throw std::invalid_argument("tyche::time_series::ljung_box: fewer fitted parameters than lags");
  const auto r = acf(x, lags);
  const double dn = static_cast<double>(n);
  LjungBox lb;
  for (std::size_t k = 1; k <= lags; k++) lb.statistic += r[k] * r[k] / (dn - static_cast<double>(k));
  lb.statistic *= dn * (dn + 2);
  lb.lags = lags;
  lb.df = static_cast<double>(lags - fitted_parameters);
  lb.p_value = probability::ChiSquared{lb.df}.survival(lb.statistic);
  return lb;
}

}  // namespace tyche::time_series
