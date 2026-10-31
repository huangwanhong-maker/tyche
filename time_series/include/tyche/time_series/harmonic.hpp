// Harmonic analysis at given frequencies: the method of tidal analysis (Foreman 1977, Manual for Tidal
// Heights Analysis and Prediction, Pacific Marine Science Report 77-10; Pawlowicz, Beardsley and Lentz
// 2002, Comput. Geosci. 28, 929–937, T_TIDE), for any spacing of the times and any gaps.
//
// The model. y(t) = c₀ [+ c₁ (t − t₀)] + Σ_k [a_k cos 2πf_k(t − t₀) + b_k sin 2πf_k(t − t₀)] + e(t), the
// frequencies f_k known (cycles per unit of t), fitted by least squares on calx's QR with column pivoting
// (the trend's column scaled by the record length T = t_max − t_min). Each constituent is
// A_k cos(2πf_k(t − t₀) − φ_k) with A_k = √(a_k² + b_k²) and φ_k = atan2(b_k, a_k) ∈ [0, 2π): its maxima
// fall φ_k/(2πf_k) after t₀ (modulo a period). This is the phase convention of tidal analysis (the
// "phase lag" g of a constituent with respect to its argument at t₀, without astronomical arguments or
// nodal corrections, which belong to a domain that names the Moon).
//
// Resolution. Two constituents closer in frequency than R/T (Rayleigh's criterion, R = 1 by default)
// cannot be told apart by a record of length T: their columns are nearly collinear and their
// coefficients' errors large. They are still fitted (and reported unresolved); a constituent at zero
// frequency, or two at one frequency, is refused (Foreman 1977; Pawlowicz et al. 2002).
//
// Errors. Under white residual noise of variance σ² (estimated as RSS/(n − p)), the coefficients'
// covariance is σ² (XᵀX)⁻¹, for any spacing. Residuals of geophysical and social series are coloured:
// their variance near f_k, not the average over all frequencies, makes the error of a_k and b_k. With
// even spacing Δt (fs = 1/Δt) and `coloured`, the residual's one-sided density S_r (calx's periodogram,
// Hann-tapered) is averaged over the band |f − f_k| ≤ B (B = options.band, by default max(f_k/10, 10/T)),
// leaving out the bins within 2/T of every fitted frequency, where the fit has removed the residual (the
// Hann taper widens the notch of ±1/T to ±2/T), and the white-equivalent variance σ²_eff(f_k) = S_r fs/2
// replaces σ² for that constituent's block of the covariance (Pawlowicz et al. 2002, §3; for well-
// separated frequencies on a long record (XᵀX)⁻¹ ≈ (2/n) I, so var(â) = var(b̂) = 2σ²_eff/n as there).
// Without even spacing the errors are white, and the fit says so (`coloured` false). Amplitude and phase
// errors by linearisation (as Pawlowicz et al. 2002 do, without their Monte Carlo):
// A_se² = (a² var a + b² var b + 2ab cov)/A², φ_se² = (b² var a + a² var b − 2ab cov)/A⁴, so that with
// var a = var b = v and no covariance A_se = √v and φ_se = √v/A (radians). snr = (A/A_se)², T_TIDE's
// signal-to-noise ratio (T_TIDE marks a constituent with snr above 2 as significant).
//
// Accuracy. The coefficients are exact to rounding on noise-free data at any spacing (checked); the
// standard errors are exact under Gaussian white noise and, coloured, asymptotic in the length of the
// record and the band's number of bins (each band of m bins estimates S_r to about 1/√m).

#pragma once

#include <calx/linalg/qr.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/signal/spectrum.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::time_series {

struct Constituent { std::string name; double frequency = 0; };   // cycles per unit of t

struct HarmonicOptions {
  double t0 = 0;               // the reference instant of the phases
  bool trend = false;          // fit c₁ too
  bool coloured = true;        // errors from the residual spectrum near each f_k (else white)
  double rayleigh = 1.0;       // R: constituents closer than R/T are reported unresolved (T the record length)
  double band = 0;             // half-width of the band of the residual spectrum about each f_k (frequency
                               // units); 0: max(f_k/10, 10/T)
};

struct HarmonicTerm {
  std::string name;
  double frequency = 0, amplitude = 0, phase = 0, a = 0, b = 0;
  double amplitude_se = 0, phase_se = 0;   // standard errors (phase in radians)
  double snr = 0;                          // (A / A_se)², T_TIDE's signal-to-noise
  bool resolved = true;                    // false if within R/T of another constituent of the fit
};

struct HarmonicFit {
  double mean = 0, trend = 0;
  std::vector<HarmonicTerm> terms;
  std::vector<double> fitted, residual;
  double record_length = 0;                // t_max − t_min
  double variance_explained = 0;           // 1 − var(residual)/var(y)
  std::size_t observations = 0;
  bool coloured = false;                   // true when the errors came from the residual spectrum (even spacing)
};

// |f1 − f2| T ≥ R: Rayleigh's criterion.
inline bool resolvable(double f1, double f2, double record_length, double rayleigh = 1.0) {
  return std::fabs(f1 - f2) * record_length >= rayleigh;
}

inline HarmonicFit harmonic_fit(std::span<const double> t, std::span<const double> y, std::span<const Constituent> constituents, const HarmonicOptions& options = {}) {
  const std::size_t n = t.size(), m = constituents.size();
  if (y.size() != n) throw std::invalid_argument("tyche::time_series::harmonic_fit: one value for each time");
  const std::size_t p = 1 + (options.trend ? 1 : 0) + 2 * m;
  if (n <= p) throw std::invalid_argument("tyche::time_series::harmonic_fit: more observations than coefficients");
  for (const auto& c : constituents)
    if (!(c.frequency > 0) || !std::isfinite(c.frequency)) throw std::invalid_argument("tyche::time_series::harmonic_fit: frequencies above zero (the mean is fitted apart)");
  for (std::size_t i = 0; i < n; i++)
    if (!std::isfinite(t[i]) || !std::isfinite(y[i])) throw std::invalid_argument("tyche::time_series::harmonic_fit: finite times and values");
  const auto [lo, hi] = std::minmax_element(t.begin(), t.end());
  const double T = *hi - *lo;
  if (!(T > 0)) throw std::invalid_argument("tyche::time_series::harmonic_fit: a record of positive length");

  // ---- the design and the least-squares fit ----
  calx::linalg::Rectangular X(n, p);
  const std::size_t first = options.trend ? 2 : 1;
  for (std::size_t i = 0; i < n; i++) {
    const double s = t[i] - options.t0;
    X(i, 0) = 1;
    if (options.trend) X(i, 1) = s / T;
    for (std::size_t k = 0; k < m; k++) {
      const double arg = 2 * std::numbers::pi * constituents[k].frequency * s;
      X(i, first + 2 * k) = std::cos(arg);
      X(i, first + 2 * k + 1) = std::sin(arg);
    }
  }
  const calx::linalg::Qr qr = calx::linalg::qr(X);
  if (qr.rank < p) throw std::invalid_argument("tyche::time_series::harmonic_fit: the constituents' columns are collinear (a frequency repeated, or aliased onto another by the sampling)");
  const std::vector<double> beta = calx::linalg::least_squares(qr, y);
  const calx::linalg::Matrix C = calx::linalg::normal_inverse(qr);

  HarmonicFit fit;
  fit.observations = n;
  fit.record_length = T;
  fit.mean = beta[0];
  fit.trend = options.trend ? beta[1] / T : 0.0;
  fit.fitted.resize(n);
  fit.residual.resize(n);
  double rss = 0;
  for (std::size_t i = 0; i < n; i++) {
    double v = 0;
    for (std::size_t j = 0; j < p; j++) v += X(i, j) * beta[j];
    fit.fitted[i] = v;
    fit.residual[i] = y[i] - v;
    rss += fit.residual[i] * fit.residual[i];
  }
  const double sigma2 = rss / static_cast<double>(n - p);
  auto variance = [](std::span<const double> v) {
    double mean = 0, s = 0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    for (double x : v) s += (x - mean) * (x - mean);
    return s / static_cast<double>(v.size());
  };
  const double vy = variance(y);
  fit.variance_explained = vy > 0 ? 1 - variance(fit.residual) / vy : 0.0;

  // ---- even spacing, and the residual's spectrum ----
  bool even = n >= 3 && options.coloured;
  const double dt = (t[n - 1] - t[0]) / static_cast<double>(n - 1);
  if (even) {
    even = dt > 0;
    for (std::size_t i = 0; even && i + 1 < n; i++) even = std::fabs(t[i + 1] - t[i] - dt) <= 1e-6 * dt;
  }
  std::vector<double> S, f;
  double fs = 0;
  if (even) {
    fs = 1 / dt;
    const auto P = calx::signal::periodogram(fit.residual, fs, calx::signal::hann(n));
    S = P.density;
    f = P.frequency;
  }

  // ---- the terms ----
  for (std::size_t k = 0; k < m; k++) {
    HarmonicTerm term;
    term.name = constituents[k].name;
    term.frequency = constituents[k].frequency;
    term.a = beta[first + 2 * k];
    term.b = beta[first + 2 * k + 1];
    term.amplitude = std::hypot(term.a, term.b);
    term.phase = std::atan2(term.b, term.a);
    if (term.phase < 0) term.phase += 2 * std::numbers::pi;
    if (term.phase >= 2 * std::numbers::pi) term.phase = 0;
    for (std::size_t j = 0; j < m; j++)
      if (j != k && !resolvable(constituents[k].frequency, constituents[j].frequency, T, options.rayleigh)) term.resolved = false;

    double noise = sigma2;
    if (even) {
      const double band = options.band > 0 ? options.band : std::max(term.frequency / 10, 10 / T);
      double sum = 0;
      std::size_t count = 0;
      for (std::size_t j = 1; j < S.size(); j++) {
        if (n % 2 == 0 && j == n / 2) continue;  // the Nyquist bin, its own mirror
        if (std::fabs(f[j] - term.frequency) > band) continue;
        bool near = false;
        for (const auto& c : constituents) near = near || std::fabs(f[j] - c.frequency) <= 2 / T;
        if (near) continue;
        sum += S[j];
        count++;
      }
      if (count > 0) noise = sum / static_cast<double>(count) * fs / 2;
    }
    const std::size_t ia = first + 2 * k, ib = ia + 1;
    const double va = noise * C(ia, ia), vb = noise * C(ib, ib), cab = noise * C(ia, ib);
    if (term.amplitude > 0) {
      const double A2 = term.amplitude * term.amplitude;
      term.amplitude_se = std::sqrt(std::max(0.0, (term.a * term.a * va + term.b * term.b * vb + 2 * term.a * term.b * cab) / A2));
      term.phase_se = std::sqrt(std::max(0.0, (term.b * term.b * va + term.a * term.a * vb - 2 * term.a * term.b * cab) / (A2 * A2)));
    } else {
      term.amplitude_se = std::sqrt(0.5 * (va + vb));
      term.phase_se = std::numbers::pi;  // no phase: uniform
    }
    term.snr = term.amplitude_se > 0 ? (term.amplitude / term.amplitude_se) * (term.amplitude / term.amplitude_se) : std::numeric_limits<double>::infinity();
    fit.terms.push_back(term);
  }
  fit.coloured = even;
  return fit;
}

// The fitted model at the times t: mean + trend (t − t0) + Σ_k a_k cos 2πf_k(t − t0) + b_k sin 2πf_k(t − t0),
// t0 the reference instant of the fit.
inline std::vector<double> harmonic_predict(const HarmonicFit& fit, std::span<const double> t, double t0) {
  std::vector<double> out(t.size());
  for (std::size_t i = 0; i < t.size(); i++) {
    const double s = t[i] - t0;
    double v = fit.mean + fit.trend * s;
    for (const auto& term : fit.terms) {
      const double arg = 2 * std::numbers::pi * term.frequency * s;
      v += term.a * std::cos(arg) + term.b * std::sin(arg);
    }
    out[i] = v;
  }
  return out;
}

}  // namespace tyche::time_series
