// The Lomb–Scargle periodogram of an unevenly sampled series, and the false-alarm probability of its
// highest peak.
//
// The law. At each frequency f the periodogram is the fraction of the series' variance that the best-
// fitting sinusoid of that frequency explains: with the times shifted by τ(ω) so that the sine and cosine
// terms decouple, tan 2ωτ = Σ sin 2ωt / Σ cos 2ωt (ω = 2πf),
//   P(f) = [ (Σ_i y_i cos ω(t_i − τ))² / Σ_i cos² ω(t_i − τ) + (Σ_i y_i sin ω(t_i − τ))² / Σ_i sin² ω(t_i − τ) ] / Σ_i y_i²,
// y centred on its mean (Lomb 1976, Astrophys. Space Sci. 39, 447; Scargle 1982, ApJ 263, 835, eqs. 10
// and 11). With a floating mean the sinusoid is fitted together with a constant, and the sums become
// covariances about the weighted means (Zechmeister and Kürster 2009, A&A 496, 577: the "generalised
// Lomb–Scargle periodogram"), so that P = 1 − χ²(f)/χ²₀ ∈ [0, 1] is the reduction of the residual sum
// of squares by the sinusoid: the "standard" normalisation (VanderPlas 2018, ApJS 236, 16). The
// computation is astropy's (astropy.timeseries.LombScargle(normalization='standard', fit_mean,
// center_data=True), method 'slow' or 'cython'): equal weights, the times taken from the first, which
// changes nothing but the rounding. O(n) per frequency; at f = 0 (and, for even sampling, at the
// Nyquist frequency) a term whose denominator vanishes is left out.
//
// The false-alarm probability. The highest of the periodogram's values over [0, fmax] exceeds a level Z
// by chance, for Gaussian white noise and a floating mean, with probability about
//   FAP = 1 − (1 − P₁(Z)) e^{−τ(Z)},  P₁(Z) = (1 − Z)^{(N−3)/2},
//   τ(Z) = γ(N_H) W (1 − Z)^{(N_K−1)/2} √(N_H Z / 2),  W = fmax √(4π Var t),
//   γ(N) = √(2/N) Γ(N/2)/Γ((N−1)/2),  N_H = N − 1,  N_K = N − 3
// (Baluev 2008, MNRAS 385, 1279, eq. 6 and Table 1; astropy's 'baluev' method, which it reproduces: an
// upper bound that is close for small FAP, from Davies's bound on the excursions of a random field).
// Var t is the variance of the times (divisor N). y does not enter the approximation and is taken only
// for its count.

#pragma once

#include <calx/special_functions/gamma.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

struct LombScargle { std::vector<double> frequency, power; };

inline LombScargle lomb_scargle(std::span<const double> t, std::span<const double> y, std::span<const double> frequency, bool floating_mean = true) {
  const std::size_t n = t.size();
  if (y.size() != n) throw std::invalid_argument("tyche::time_series::lomb_scargle: one value for each time");
  if (n < 3) throw std::invalid_argument("tyche::time_series::lomb_scargle: three observations or more");
  const double w = 1 / static_cast<double>(n);
  double ybar = 0;
  for (double v : y) ybar += w * v;
  std::vector<double> yc(n), tr(n);
  double YY = 0;
  for (std::size_t i = 0; i < n; i++) {
    yc[i] = y[i] - ybar;
    tr[i] = t[i] - t[0];
    YY += w * yc[i] * yc[i];
  }
  if (!(YY > 0)) throw std::invalid_argument("tyche::time_series::lomb_scargle: a series that is not constant");
  LombScargle out;
  out.frequency.assign(frequency.begin(), frequency.end());
  out.power.resize(frequency.size());
  for (std::size_t q = 0; q < frequency.size(); q++) {
    const double omega = 2 * std::numbers::pi * frequency[q];
    double S2 = 0, C2 = 0, S = 0, C = 0;
    for (std::size_t i = 0; i < n; i++) {
      const double s = std::sin(omega * tr[i]), c = std::cos(omega * tr[i]);
      S2 += 2 * w * s * c;
      C2 += 2 * w * (0.5 - s * s);
      S += w * s;
      C += w * c;
    }
    if (floating_mean) {
      S2 -= 2 * S * C;
      C2 -= C * C - S * S;
    }
    const double shift = 0.5 * std::atan2(S2, C2);
    double YC = 0, YS = 0, CC = 0, SS = 0, Ct = 0, St = 0;
    for (std::size_t i = 0; i < n; i++) {
      const double a = omega * tr[i] - shift;
      const double s = std::sin(a), c = std::cos(a);
      YC += w * yc[i] * c;
      YS += w * yc[i] * s;
      CC += w * c * c;
      SS += w * s * s;
      Ct += w * c;
      St += w * s;
    }
    if (floating_mean) {
      CC -= Ct * Ct;  // Y = 0: yc is centred
      SS -= St * St;
    }
    const double tiny = 1e-15;
    double p = 0;
    if (CC > tiny) p += YC * YC / CC;
    if (SS > tiny) p += YS * YS / SS;
    out.power[q] = p / YY;
  }
  return out;
}

inline double false_alarm_baluev(double power, std::span<const double> t, std::span<const double> y, double fmax) {
  const std::size_t N = t.size();
  if (y.size() != N) throw std::invalid_argument("tyche::time_series::false_alarm_baluev: one value for each time");
  if (N < 5) throw std::invalid_argument("tyche::time_series::false_alarm_baluev: five observations or more");
  if (!(power >= 0 && power <= 1)) throw std::invalid_argument("tyche::time_series::false_alarm_baluev: a standard power in [0, 1]");
  if (!(fmax > 0)) throw std::invalid_argument("tyche::time_series::false_alarm_baluev: a positive highest frequency");
  double mean = 0, var = 0;
  for (double v : t) mean += v;
  mean /= static_cast<double>(N);
  for (double v : t) var += (v - mean) * (v - mean);
  var /= static_cast<double>(N);
  const double NH = static_cast<double>(N) - 1, NK = static_cast<double>(N) - 3;
  namespace sf = calx::special_functions;
  const double gamma = std::sqrt(2 / NH) * std::exp(sf::log_gamma(NH / 2) - sf::log_gamma((NH - 1) / 2));
  const double W = fmax * std::sqrt(4 * std::numbers::pi * var);
  const double tau = gamma * W * std::pow(1 - power, 0.5 * (NK - 1)) * std::sqrt(0.5 * NH * power);
  const double single = std::pow(1 - power, 0.5 * NK);
  return -std::expm1(-tau) + single * std::exp(-tau);
}

}  // namespace tyche::time_series
