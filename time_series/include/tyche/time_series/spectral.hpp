// Inference on spectra: confidence intervals of spectral estimates, the degrees of freedom of Welch's
// estimate, the levels that squared coherence and bicoherence exceed by chance, the red-noise background
// with its levels and surrogates, and aliasing.
//
// The estimates are calx's (calx/signal: periodogram, welch, multitaper, cross_spectrum, coherence,
// bispectrum), one-sided densities in one convention: a bin 0 < f < fs/2 holds the power of ±f.
//
// - spectral_interval: an estimate Ŝ with ν equivalent degrees of freedom is distributed as S χ²_ν/ν, so
//   [ν Ŝ / χ²_ν(1 − α/2), ν Ŝ / χ²_ν(α/2)] covers S with probability 1 − α (Percival and Walden 1993,
//   Spectral Analysis for Physical Applications, eq. 258), χ²_ν(p) the p-quantile (the upper one computed
//   as an upper quantile, tyche::probability). Exact for Gaussian series where Ŝ is a sum of ν/2
//   independent squared complex Gaussians; approximate (equivalent degrees of freedom) otherwise.
// - welch_dof: ν = 2N_B / (1 + 2 Σ_{m=1}^{N_B−1} (1 − m/N_B) ρ²(m)), ρ(m) = Σ_t w_t w_{t+mD} / Σ_t w_t²
//   the correlation of the Hann window with itself shifted by m steps D (Percival and Walden eq. 292b):
//   1.89 N_B for half-overlapping Hann segments (ρ(1) = 1/6), 2N_B without overlap. For a white or
//   slowly varying spectrum.
// - coherence_level: for two independent Gaussian series and K independent segments, the squared
//   coherence at 0 < f < fs/2 is Beta(1, K − 1), P(γ̂² > c) = (1 − c)^{K−1}, so the level exceeded with
//   probability α is 1 − α^{1/(K−1)} (Thompson 1979, J. Atmos. Sci. 36, 2020; Bendat and
//   Piersol 2010, Random Data, 4th ed., §9.2.3). With overlapping segments K is ν/2 (welch_dof).
// - bicoherence_level: for Gaussian noise 2S b² is asymptotically χ²₂ over S segments, P(b² > c) =
//   e^{−Sc}, so the level is −ln α / S (Elgar and Guza 1988, IEEE Trans. Acoust. Speech Signal Process.
//   36, 1667: b²₉₅ = 6/d, d = 2S). Asymptotic in S.
// - RedNoise: the AR(1) process x_t = ρ x_{t−1} + ε_t with the series' lag-one autocorrelation ρ and
//   variance σ² (divisor n, as calx's periodogram integrates to it), whose one-sided density is
//   (2σ²/fs)(1 − ρ²)/(1 + ρ² − 2ρ cos 2πf/fs) (Gilman, Fuglister and Mitchell 1963, J. Atmos. Sci. 20,
//   182; Torrence and Compo 1998, BAMS 79, 61, eq. 16): it integrates to σ² over [0, fs/2] (the Poisson
//   kernel), and at a bin 0 < f < fs/2 it is the expectation of the periodogram of a long AR(1) path.
//   The background of stochastic climate models (Hasselmann 1976, Tellus 28, 473): integrated white
//   forcing with damping.
// - red_noise_level: the level an estimate of ν degrees of freedom exceeds with probability 1 − confidence
//   under the background: background · χ²_ν(confidence)/ν (Torrence and Compo §4; Mann and Lees
//   1996, Climatic Change 33, 409). A level at each frequency, not over all of them together.
// - phase_surrogate: a copy with the periodogram kept exactly and the phases randomised (Theiler, Eubank,
//   Longtin, Galdrikian and Farmer 1992, Physica D 58, 77): the discrete Fourier coefficients 1 … ⌈n/2⌉−1
//   turned by independent uniform phases, their mirror images conjugate (so the copy is real), the mean
//   and (even n) the Nyquist coefficient kept. The null of a linear Gaussian process with the series'
//   autocovariance (circular).
// - red_noise_surrogate: an AR(1) path with the background's ρ and σ², started from its stationary law
//   (mean zero).
// - alias: a frequency f sampled at fs appears at |f − fs · round(f/fs)| ∈ [0, fs/2].

#pragma once

#include <calx/random/stream.hpp>
#include <calx/transforms/fft.hpp>
#include <tyche/probability/distributions.hpp>
#include <tyche/time_series/autocorrelation.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

// ---- intervals and degrees of freedom ----

struct Interval { double lower = 0, upper = 0; };

inline Interval spectral_interval(double estimate, double dof, double confidence = 0.95) {
  if (!(estimate >= 0) || !std::isfinite(estimate)) throw std::invalid_argument("tyche::time_series::spectral_interval: an estimate of zero or more");
  if (!(dof > 0)) throw std::invalid_argument("tyche::time_series::spectral_interval: degrees of freedom above zero");
  if (!(confidence > 0 && confidence < 1)) throw std::invalid_argument("tyche::time_series::spectral_interval: a confidence in (0, 1)");
  const double alpha = 1 - confidence;
  const probability::ChiSquared chi{dof};
  return {dof * estimate / chi.upper_quantile(alpha / 2), dof * estimate / chi.quantile(alpha / 2)};
}

inline double welch_dof(std::size_t segments, std::size_t L, double overlap) {
  if (segments < 1 || L < 2) throw std::invalid_argument("tyche::time_series::welch_dof: one segment or more, of two samples or more");
  if (!(overlap >= 0 && overlap < 1)) throw std::invalid_argument("tyche::time_series::welch_dof: the overlap lies in [0, 1)");
  // The step as calx's welch() takes it.
  const auto D = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(static_cast<double>(L) * (1 - overlap))));
  std::vector<double> w(L);
  double ww = 0;
  for (std::size_t t = 0; t < L; t++) {
    w[t] = 0.5 * (1 - std::cos(2 * std::numbers::pi * static_cast<double>(t) / static_cast<double>(L)));
    ww += w[t] * w[t];
  }
  const double NB = static_cast<double>(segments);
  double sum = 0;
  for (std::size_t m = 1; m < segments && m * D < L; m++) {
    double r = 0;
    for (std::size_t t = 0; t + m * D < L; t++) r += w[t] * w[t + m * D];
    r /= ww;
    sum += (1 - static_cast<double>(m) / NB) * r * r;
  }
  return 2 * NB / (1 + 2 * sum);
}

inline double coherence_level(double effective_segments, double alpha = 0.05) {
  if (!(effective_segments > 1)) throw std::invalid_argument("tyche::time_series::coherence_level: more than one effective segment");
  if (!(alpha > 0 && alpha < 1)) throw std::invalid_argument("tyche::time_series::coherence_level: α in (0, 1)");
  return 1 - std::pow(alpha, 1 / (effective_segments - 1));
}

inline double bicoherence_level(std::size_t segments, double alpha = 0.05) {
  if (segments < 1) throw std::invalid_argument("tyche::time_series::bicoherence_level: one segment or more");
  if (!(alpha > 0 && alpha < 1)) throw std::invalid_argument("tyche::time_series::bicoherence_level: α in (0, 1)");
  return -std::log(alpha) / static_cast<double>(segments);
}

// ---- the red-noise background ----

struct RedNoise {
  double rho = 0;          // lag-one autocorrelation
  double variance = 0;     // σ² of the series
  // One-sided density at f for sampling rate fs: (2σ²/fs)(1 − ρ²)/(1 + ρ² − 2ρ cos 2πf/fs) — integrates to σ².
  double density(double f, double fs) const {
    if (!(fs > 0)) throw std::invalid_argument("tyche::time_series::RedNoise::density: the sampling rate must be positive");
    return 2 * variance / fs * (1 - rho * rho) / (1 + rho * rho - 2 * rho * std::cos(2 * std::numbers::pi * f / fs));
  }
};

inline RedNoise red_noise(std::span<const double> x) {
  if (x.size() < 3) throw std::invalid_argument("tyche::time_series::red_noise: three values or more");
  const auto g = autocovariance(x, 1);
  if (!(g[0] > 0)) throw std::invalid_argument("tyche::time_series::red_noise: a series that is not constant");
  return {g[1] / g[0], g[0]};
}

inline double red_noise_level(const RedNoise& r, double f, double fs, double dof, double confidence = 0.95) {
  if (!(dof > 0)) throw std::invalid_argument("tyche::time_series::red_noise_level: degrees of freedom above zero");
  if (!(confidence > 0 && confidence < 1)) throw std::invalid_argument("tyche::time_series::red_noise_level: a confidence in (0, 1)");
  return r.density(f, fs) * probability::ChiSquared{dof}.quantile(confidence) / dof;
}

// ---- surrogates ----

inline std::vector<double> phase_surrogate(std::span<const double> x, calx::random::Stream& rng) {
  const std::size_t n = x.size();
  if (n < 2) throw std::invalid_argument("tyche::time_series::phase_surrogate: two values or more");
  std::vector<calx::transforms::Complex> X(x.begin(), x.end());
  calx::transforms::fft(X);
  for (std::size_t j = 1; j + j < n; j++) {  // j = 1 … ⌈n/2⌉ − 1
    X[j] *= std::polar(1.0, rng.uniform(0, 2 * std::numbers::pi));
    X[n - j] = std::conj(X[j]);
  }
  calx::transforms::fft(X, true);
  std::vector<double> out(n);
  for (std::size_t t = 0; t < n; t++) out[t] = X[t].real();
  return out;
}

inline std::vector<double> red_noise_surrogate(const RedNoise& r, std::size_t n, calx::random::Stream& rng) {
  if (!(std::fabs(r.rho) < 1)) throw std::invalid_argument("tyche::time_series::red_noise_surrogate: |ρ| < 1");
  if (!(r.variance >= 0)) throw std::invalid_argument("tyche::time_series::red_noise_surrogate: a variance of zero or more");
  std::vector<double> out(n);
  const double sd = std::sqrt(r.variance), innovation = sd * std::sqrt(1 - r.rho * r.rho);
  double v = sd * rng.normal();
  for (std::size_t t = 0; t < n; t++) {
    if (t > 0) v = r.rho * v + innovation * rng.normal();
    out[t] = v;
  }
  return out;
}

// ---- aliasing ----

inline double alias(double f, double fs) {
  if (!(fs > 0)) throw std::invalid_argument("tyche::time_series::alias: the sampling rate must be positive");
  return std::fabs(f - fs * std::round(f / fs));
}

}  // namespace tyche::time_series
