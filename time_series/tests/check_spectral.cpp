// Checks of tyche/time_series/spectral.hpp and lomb_scargle.hpp.
//
// - spectral_interval and red_noise_level against scipy.stats.chi2.ppf (make_spectral_fixtures.py), seven
//   (ν, confidence) pairs, ν fractional among them; the interval's coverage by simulation (multitaper
//   estimates of white noise, 2K degrees of freedom).
// - welch_dof in closed form (ρ(1) = 1/6 for half-overlapping Hann segments; 2N_B without overlap) and by
//   simulation (Var Ŝ / (E Ŝ)² = 2/ν on white noise).
// - coherence_level and bicoherence_level by simulation: independent Gaussian series exceed them at the
//   rate α (coherence: exactly Beta(1, K − 1); bicoherence: asymptotically).
// - The red-noise background: its density integrates to σ² (trapezoidal rule, spectrally exact here); the
//   mean periodogram of AR(1) paths is it (simulation); red_noise() recovers ρ and σ² of a long path, and
//   red_noise_surrogate() has them.
// - Phase surrogates keep the periodogram to rounding (even and odd lengths) and the mean, and are not the
//   series.
// - alias(): the M2 tide sampled daily appears at 14.765 days, sampled every 9.9156 days (TOPEX/Poseidon's
//   repeat) at 62.11 days; f and fs/2 cases in closed form.
// - Lomb–Scargle against astropy 8.0.2 (normalization 'standard', floating and fixed mean, the exact
//   'cython' sums) on 185 uneven times with a gap; its peak at the sinusoid's frequency; Baluev's false-
//   alarm probability against astropy's 'baluev'; the periodogram of even samples with a fixed mean equal
//   to the classical one in closed form at a Fourier frequency.

#include <tyche/time_series/lomb_scargle.hpp>
#include <tyche/time_series/spectral.hpp>

#include <calx/random/stream.hpp>
#include <calx/signal/bispectrum.hpp>
#include <calx/signal/spectrum.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace ts = tyche::time_series;
namespace cs = calx::signal;

namespace {

std::map<std::string, std::vector<std::vector<double>>> read_fixtures(const std::string& path) {
  std::map<std::string, std::vector<std::vector<double>>> out;
  std::ifstream in(path);
  std::string line, key;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream f(line);
    f >> key;
    std::vector<double> v;
    double x;
    while (f >> x) v.push_back(x);
    out[key].push_back(std::move(v));
  }
  return out;
}

std::vector<double> ar1(calx::random::Stream& z, std::size_t n, double rho, double sd = 1) {
  std::vector<double> x(n);
  double v = sd * z.normal() / std::sqrt(1 - rho * rho);
  for (auto& u : x) {
    v = rho * v + sd * z.normal();
    u = v;
  }
  return x;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto ref = read_fixtures(TYCHE_FIXTURES "/spectral-reference.txt");
  s.require("the reference values from scipy and astropy are there", ref.count("chi2") == 1 && ref.count("ls/t") == 1);
  const double two_pi = 2 * std::numbers::pi;

  // ---- intervals and levels against scipy's χ² quantiles ----
  {
    double worst = 0, worst_level = 0;
    const ts::RedNoise r{0.6, 2.5};
    for (const auto& c : ref.at("chi2")) {
      const double nu = c[0], conf = c[1], S = 1.7;
      const auto I = ts::spectral_interval(S, nu, conf);
      worst = std::max({worst, std::fabs(I.lower / (nu * S / c[3]) - 1), std::fabs(I.upper / (nu * S / c[2]) - 1)});
      worst_level = std::max(worst_level, std::fabs(ts::red_noise_level(r, 0.3, 2.0, nu, conf) / (r.density(0.3, 2.0) * c[4] / nu) - 1));
    }
    s.check("spectral_interval: [ν S/χ²_ν(1 − α/2), ν S/χ²_ν(α/2)] with scipy's quantiles (relative, 7 pairs)", worst, 1e-11);
    s.check("red_noise_level: background · χ²_ν(confidence)/ν with scipy's quantile (relative)", worst_level, 1e-11);
  }
  {
    // Coverage: plain multitaper estimates of white noise, ν = 2K = 10, at the interior bins of 40 series.
    const std::size_t n = 2048, K = 5;
    const double fs = 1, level = 2.0 / fs;
    double covered = 0, count = 0;
    for (std::uint32_t r = 0; r < 40; r++) {
      calx::random::Stream z(31, r, 0, 0);
      std::vector<double> x(n);
      for (auto& v : x) v = z.normal();
      const auto mt = cs::multitaper(x, fs, 3, K, false);
      for (std::size_t j = 6; j + 6 < mt.density.size(); j++) {
        const auto I = ts::spectral_interval(mt.density[j], mt.dof[j], 0.90);
        covered += (I.lower <= level && level <= I.upper) ? 1 : 0;
        count++;
      }
    }
    s.check("spectral_interval: a 90 % interval covers white noise's level at 90 % of the bins (fraction − 0.9)", covered / count - 0.9, 0.01);
  }

  // ---- Welch's degrees of freedom ----
  {
    const double NB = 40;
    s.check("welch_dof: 2N_B/(1 + 2(1 − 1/N_B)/36) for half-overlapping Hann segments (ρ(1) = 1/6)", ts::welch_dof(40, 256, 0.5) - 2 * NB / (1 + 2 * (1 - 1 / NB) / 36), 1e-12);
    s.check("welch_dof: 2N_B without overlap", ts::welch_dof(40, 256, 0.0) - 2 * NB, 1e-12);
    // Var Ŝ/(E Ŝ)² = 2/ν on white noise, over the interior bins of 400 series of 41 half-overlapping segments.
    const std::size_t L = 128, segments = 40, n = L / 2 * (segments + 1);
    std::vector<double> sum(L / 2 + 1, 0.0), sum2(L / 2 + 1, 0.0);
    const int R = 400;
    std::size_t got = 0;
    for (int r = 0; r < R; r++) {
      calx::random::Stream z(32, static_cast<std::uint32_t>(r), 0, 0);
      std::vector<double> x(n);
      for (auto& v : x) v = z.normal();
      const auto w = cs::welch(x, 1.0, L, 0.5);
      got = w.segments;
      for (std::size_t j = 0; j < w.density.size(); j++) {
        sum[j] += w.density[j];
        sum2[j] += w.density[j] * w.density[j];
      }
    }
    double ratio = 0, bins = 0;
    for (std::size_t j = 4; j + 4 < sum.size(); j++) {
      const double m = sum[j] / R, v = sum2[j] / R - m * m;
      ratio += v / (m * m);
      bins++;
    }
    s.require("welch_dof: 40 segments", got == segments);
    s.check("welch_dof: 2/ν equals Var Ŝ/(E Ŝ)² of Welch's estimate of white noise (relative, 400 series)", (ratio / bins) / (2 / ts::welch_dof(segments, L, 0.5)) - 1, 0.04);
  }

  // ---- the levels of coherence and bicoherence ----
  {
    s.check("coherence_level: 1 − α^{1/(K−1)} (K = 10, α = 0.05)", ts::coherence_level(10, 0.05) - (1 - std::pow(0.05, 1.0 / 9)), 1e-15);
    // Independent Gaussian series, K = 12 segments without overlap: the rate at which γ̂² exceeds the level.
    const std::size_t L = 64, K = 12;
    const double level = ts::coherence_level(static_cast<double>(K), 0.05);
    double over = 0, count = 0;
    for (std::uint32_t r = 0; r < 1000; r++) {
      calx::random::Stream z(33, r, 0, 0);
      std::vector<double> x(L * K), y(L * K);
      for (auto& v : x) v = z.normal();
      for (auto& v : y) v = z.normal();
      const auto coh = cs::coherence(cs::cross_spectrum(x, y, 1.0, L, 0.0));
      for (std::size_t j = 2; j + 2 < coh.size(); j++) {
        over += coh[j] > level ? 1 : 0;
        count++;
      }
    }
    s.check("coherence_level: independent Gaussian series exceed it at the rate α = 0.05 (rate − α)", over / count - 0.05, 0.006);
  }
  {
    s.check("bicoherence_level: −ln α / S (S = 64, α = 0.05; Elgar and Guza's 6/d with d = 2S)", ts::bicoherence_level(64, 0.05) - (-std::log(0.05) / 64), 1e-17);
    const std::size_t L = 64, S = 64;
    const double level = ts::bicoherence_level(S, 0.05);
    double over = 0, count = 0;
    for (std::uint32_t r = 0; r < 60; r++) {
      calx::random::Stream z(34, r, 0, 0);
      std::vector<double> x(L * S);
      for (auto& v : x) v = z.normal();
      const auto b = cs::bispectrum(x, 1.0, L, 0.0);
      for (std::size_t j = 1; j < L / 2; j++)
        for (std::size_t k = 1; k <= j && j + k < L / 2; k++) {
          over += b.b2(j, k) > level ? 1 : 0;
          count++;
        }
    }
    s.check("bicoherence_level: Gaussian noise exceeds it at about α = 0.05 (rate − α; asymptotic in S)", over / count - 0.05, 0.015);
  }

  // ---- the red-noise background ----
  {
    const ts::RedNoise r{0.8, 3.0};
    const double fs = 4;
    const int M = 4000;
    double integral = 0;
    for (int i = 0; i <= M; i++) integral += (i == 0 || i == M ? 0.5 : 1.0) * r.density(fs / 2 * i / M, fs) * (fs / 2) / M;
    s.check("RedNoise: the density integrates to σ² over [0, fs/2] (relative)", integral / r.variance - 1, 1e-12);

    // The mean periodogram of AR(1) paths, ρ = 0.7, against the density at the interior bins.
    const std::size_t n = 1024;
    const double rho = 0.7, sd = 1.3;
    std::vector<double> mean(n / 2 + 1, 0.0);
    const int R = 300;
    for (int k = 0; k < R; k++) {
      calx::random::Stream z(35, static_cast<std::uint32_t>(k), 0, 0);
      const auto x = ar1(z, n, rho, sd);
      const auto P = cs::periodogram(x, 1.0, cs::hann(n));
      for (std::size_t j = 0; j < mean.size(); j++) mean[j] += P.density[j] / R;
    }
    const ts::RedNoise truth{rho, sd * sd / (1 - rho * rho)};
    double ratio = 0, bins = 0;
    for (std::size_t j = 8; j + 1 < mean.size(); j++) {
      ratio += mean[j] / truth.density(static_cast<double>(j) / static_cast<double>(n), 1.0);
      bins++;
    }
    s.check("RedNoise: the mean periodogram of AR(1) paths (ρ = 0.7) is its density (relative, 300 paths)", ratio / bins - 1, 0.01);

    calx::random::Stream z(36, 0, 0, 0);
    const auto x = ar1(z, 200000, 0.6, 1.0);
    const auto est = ts::red_noise(x);
    s.check("red_noise: ρ of a long AR(1) path (ρ = 0.6, n = 2·10⁵)", est.rho - 0.6, 0.006);
    s.check("red_noise: its variance σ_ε²/(1 − ρ²) (relative)", est.variance / (1 / (1 - 0.36)) - 1, 0.02);
    calx::random::Stream z2(37, 0, 0, 0);
    const auto sur = ts::red_noise_surrogate(ts::RedNoise{-0.4, 2.0}, 200000, z2);
    const auto back = ts::red_noise(sur);
    s.check("red_noise_surrogate: an AR(1) path of the background's ρ (ρ = −0.4)", back.rho + 0.4, 0.006);
    s.check("red_noise_surrogate: and σ² (relative)", back.variance / 2.0 - 1, 0.02);
  }

  // ---- phase surrogates ----
  for (const std::size_t n : {std::size_t{1000}, std::size_t{999}}) {
    calx::random::Stream z(38, static_cast<std::uint32_t>(n), 0, 0);
    auto x = ar1(z, n, 0.9);
    for (auto& v : x) v += 5;
    calx::random::Stream rng(39, 0, 0, 0);
    const auto sur = ts::phase_surrogate(x, rng);
    const auto P = cs::periodogram(x, 1.0, {}, false), Q = cs::periodogram(sur, 1.0, {}, false);
    double worst = 0, top = 0, diff = 0, mx = 0, ms = 0;
    for (std::size_t j = 0; j < P.density.size(); j++) {
      worst = std::max(worst, std::fabs(P.density[j] - Q.density[j]));
      top = std::max(top, P.density[j]);
    }
    for (std::size_t t = 0; t < n; t++) {
      diff = std::max(diff, std::fabs(sur[t] - x[t]));
      mx += x[t] / static_cast<double>(n);
      ms += sur[t] / static_cast<double>(n);
    }
    const std::string what = " (n = " + std::to_string(n) + ")";
    s.check("phase_surrogate: the periodogram kept to rounding" + what, worst / top, 1e-12);
    s.check("phase_surrogate: and the mean" + what, ms - mx, 1e-12);
    s.require("phase_surrogate: but not the series" + what, diff > 1);
  }

  // ---- aliasing ----
  {
    const double m2 = 28.9841042 * 24 / 360;  // cycles per day (28.9841042° per hour)
    s.check("alias: M2 sampled once a day appears at a period of 14.765 days", 1 / ts::alias(m2, 1.0) - 14.765, 0.001, "d");
    s.check("alias: and sampled every 9.9156 days (TOPEX/Poseidon) at 62.11 days", 1 / ts::alias(m2, 1 / 9.9156) - 62.11, 0.01, "d");
    s.check("alias: 0.9 fs appears at 0.1 fs, −0.3 fs at 0.3 fs, 2.5 fs at fs/2, 7 fs at 0",
            std::fabs(ts::alias(0.9, 1) - 0.1) + std::fabs(ts::alias(-0.3, 1) - 0.3) + std::fabs(ts::alias(2.5, 1) - 0.5) + ts::alias(7, 1), 1e-15);
  }

  // ---- Lomb–Scargle against astropy ----
  {
    const auto& t = ref.at("ls/t")[0];
    const auto& y = ref.at("ls/y")[0];
    const auto& f = ref.at("ls/frequency")[0];
    for (const bool floating : {true, false}) {
      const auto ls = ts::lomb_scargle(t, y, f, floating);
      const auto& p = ref.at(floating ? "ls/power/floating" : "ls/power/fixed")[0];
      double worst = 0;
      std::size_t peak = 0;
      for (std::size_t q = 0; q < p.size(); q++) {
        worst = std::max(worst, std::fabs(ls.power[q] - p[q]));
        if (ls.power[q] > ls.power[peak]) peak = q;
      }
      const std::string what = floating ? "floating mean" : "fixed mean";
      s.check("lomb_scargle: = astropy's 'standard' power (" + what + "; 185 uneven times, 600 frequencies)", worst, 1e-12);
      s.check("lomb_scargle: its peak at the sinusoid's frequency 0.37 (" + what + ")", ls.frequency[peak] - 0.37, 0.006);
    }
    const auto& fap = ref.at("ls/fap")[0];
    double worst = 0;
    for (std::size_t i = 1; i + 1 < fap.size(); i += 2) worst = std::max(worst, std::fabs(ts::false_alarm_baluev(fap[i], t, y, fap[0]) / fap[i + 1] - 1));
    s.check("false_alarm_baluev: = astropy's 'baluev' (relative, 7 levels from 0.02 to 0.6: FAP from 1 to 10⁻³³)", worst, 1e-12);

    // Even samples, fixed mean, a Fourier frequency j/n: τ = 0 and the classical periodogram
    // P = (2/n)|Σ y e^{−2πijt/n}|² / Σ (y − ȳ)² ... in the standard normalisation |X_j|²·2/(n Σ yc²).
    const std::size_t n = 64;
    std::vector<double> te(n), ye(n);
    calx::random::Stream z(40, 0, 0, 0);
    for (std::size_t i = 0; i < n; i++) {
      te[i] = static_cast<double>(i);
      ye[i] = z.normal() + std::cos(two_pi * 5 * te[i] / n);
    }
    const std::vector<double> fj{5.0 / n, 11.0 / n};
    const auto ls = ts::lomb_scargle(te, ye, fj, false);
    double mean = 0, ss = 0, worst_even = 0;
    for (double v : ye) mean += v / n;
    for (double v : ye) ss += (v - mean) * (v - mean);
    for (std::size_t q = 0; q < fj.size(); q++) {
      std::complex<double> X = 0;
      for (std::size_t i = 0; i < n; i++) X += (ye[i] - mean) * std::polar(1.0, -two_pi * fj[q] * te[i]);
      worst_even = std::max(worst_even, std::fabs(ls.power[q] - 2 * std::norm(X) / (n * ss)));
    }
    s.check("lomb_scargle: even samples at a Fourier frequency: the classical periodogram 2|X_j|²/(n Σ(y − ȳ)²)", worst_even, 1e-13);
  }
  return s.done();
}
