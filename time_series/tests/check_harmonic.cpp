// Checks of tyche/time_series/harmonic.hpp: harmonic analysis at given frequencies.
//
// - Noise-free data recovered exactly (closed form): the mean, a trend, and the amplitudes and phases of
//   daily, half-daily and weekly constituents, from hourly samples and from uneven times with a gap of
//   eleven days; the phase convention in closed form: the constituent A cos(2πf(t − t₀) − φ) is greatest
//   φ/(2πf) after t₀, where harmonic_predict gives mean + A; the fitted values are harmonic_predict's.
// - Rayleigh's criterion: two constituents closer than 1/T are flagged unresolved, the others not;
//   resolvable() at both sides of R/T.
// - Standard errors by simulation (Gaussian white noise, 400 records): ±1.96 standard errors of the
//   amplitude and of the phase cover the truth 95 % of the time, with white errors and with coloured
//   ones (the band estimate of a white residual), and on uneven gapped times (white, as the fit says).
// - Red residuals (AR(1), ρ = 0.9, 300 records): the coloured errors at a low frequency exceed the white
//   ones by about the square root of the spectrum's excess there, and cover 95 %, while the white ones
//   fall well short.
// - Refusals: a constituent at zero frequency, two at one frequency.

#include <tyche/time_series/harmonic.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace ts = tyche::time_series;

namespace {

const double two_pi = 2 * std::numbers::pi;

struct Truth { double f, A, phi; };

double model(double t, double t0, double mean, double trend, const std::vector<Truth>& c) {
  double v = mean + trend * (t - t0);
  for (const auto& k : c) v += k.A * std::cos(two_pi * k.f * (t - t0) - k.phi);
  return v;
}

double wrap(double a) { return std::remainder(a, two_pi); }

}  // namespace

int main() {
  calx::verification::Suite s;
  const std::vector<Truth> truth{{1.0 / 24, 2.0, 1.1}, {1.0 / 12, 0.7, 5.9}, {1.0 / 168, 1.3, 3.3}};
  const std::vector<ts::Constituent> cons{{"day", 1.0 / 24}, {"half-day", 1.0 / 12}, {"week", 1.0 / 168}};

  // ---- noise-free recovery: even, and uneven with a gap ----
  for (const bool even : {true, false}) {
    std::vector<double> t, y;
    const double t0 = 1000.5, mean = 3, trend = 0.002;
    calx::random::Stream z(41, 0, 0, 0);
    for (std::size_t i = 0; i < 2000; i++) {
      const double ti = even ? static_cast<double>(i) : static_cast<double>(i) + z.uniform(-0.4, 0.4);
      if (!even && ti > 700 && ti < 964) continue;  // a gap of eleven days
      t.push_back(ti);
      y.push_back(model(ti, t0, mean, trend, truth));
    }
    ts::HarmonicOptions opt;
    opt.t0 = t0;
    opt.trend = true;
    const auto fit = ts::harmonic_fit(t, y, cons, opt);
    double worst = std::max(std::fabs(fit.mean - mean), std::fabs(fit.trend - trend) * 2000);
    for (std::size_t k = 0; k < truth.size(); k++)
      worst = std::max({worst, std::fabs(fit.terms[k].amplitude - truth[k].A), std::fabs(wrap(fit.terms[k].phase - truth[k].phi))});
    const std::string what = even ? " (hourly)" : " (uneven times, an eleven-day gap)";
    s.check("harmonic_fit: noise-free mean, trend, amplitudes and phases recovered" + what, worst, 1e-10);
    s.require("harmonic_fit: coloured errors only with even spacing" + what, fit.coloured == even);
    s.check("harmonic_fit: all the variance explained" + what, fit.variance_explained - 1, 1e-12);
    const auto pred = ts::harmonic_predict(fit, t, t0);
    double same = 0;
    for (std::size_t i = 0; i < t.size(); i++) same = std::max(same, std::fabs(pred[i] - fit.fitted[i]));
    s.check("harmonic_predict: the fitted values at the data's times" + what, same, 1e-11);
  }
  {
    // The phase convention: one constituent, no trend; at t₀ + φ/(2πf) the prediction is mean + A.
    std::vector<double> t, y;
    const double t0 = 17, f = 1.0 / 29.5, A = 4, phi = 4.4;
    for (int i = 0; i < 300; i++) {
      t.push_back(i * 0.5);
      y.push_back(-1 + A * std::cos(two_pi * f * (i * 0.5 - t0) - phi));
    }
    const std::vector<ts::Constituent> one{{"lunar", f}};
    ts::HarmonicOptions opt;
    opt.t0 = t0;
    const auto fit = ts::harmonic_fit(t, y, one, opt);
    const std::vector<double> at{t0 + fit.terms[0].phase / (two_pi * f), t0 + fit.terms[0].phase / (two_pi * f) + 0.3};
    const auto p = ts::harmonic_predict(fit, at, t0);
    s.check("harmonic_fit: A cos(2πf(t − t₀) − φ) is greatest φ/(2πf) after t₀ (prediction − (mean + A))", p[0] - (-1 + A), 1e-10);
    s.require("harmonic_fit: and only there", p[1] < p[0]);
    s.check("harmonic_fit: φ = atan2(b, a) ∈ [0, 2π) with a = A cos φ, b = A sin φ", std::fabs(fit.terms[0].a - A * std::cos(phi)) + std::fabs(fit.terms[0].b - A * std::sin(phi)), 1e-10);
  }

  // ---- Rayleigh's criterion ----
  {
    std::vector<double> t, y;
    calx::random::Stream z(42, 0, 0, 0);
    const double T = 999;
    for (int i = 0; i < 1000; i++) {
      t.push_back(i);
      y.push_back(z.normal() + std::cos(two_pi * i / 24.0));
    }
    const std::vector<ts::Constituent> close{{"a", 1.0 / 24}, {"b", 1.0 / 24 + 0.5 / T}, {"c", 1.0 / 12}};
    const auto fit = ts::harmonic_fit(t, y, close);
    s.require("harmonic_fit: constituents closer than 1/T flagged unresolved, the third not", !fit.terms[0].resolved && !fit.terms[1].resolved && fit.terms[2].resolved);
    s.require("resolvable: |f₁ − f₂| T ≥ R", ts::resolvable(0.1, 0.1 + 1.0 / T, T) && !ts::resolvable(0.1, 0.1 + 0.99 / T, T) && ts::resolvable(0.1, 0.1 + 1.0 / T, T, 0.5) && !ts::resolvable(0.1, 0.1 + 1.5 / T, T, 2));
    bool refused_zero = false, refused_twice = false;
    try { ts::harmonic_fit(t, y, std::vector<ts::Constituent>{{"mean", 0.0}}); } catch (const std::invalid_argument&) { refused_zero = true; }
    try { ts::harmonic_fit(t, y, std::vector<ts::Constituent>{{"a", 0.1}, {"a again", 0.1}}); } catch (const std::invalid_argument&) { refused_twice = true; }
    s.require("harmonic_fit: a constituent at zero frequency refused, and one frequency twice", refused_zero && refused_twice);
  }

  // ---- standard errors under white noise ----
  for (const int mode : {0, 1, 2}) {  // 0: white errors, even; 1: coloured, even; 2: uneven with a gap
    double cover_A = 0, cover_phi = 0, count = 0, ratio = 0;
    for (std::uint32_t r = 0; r < 400; r++) {
      calx::random::Stream z(43, r, static_cast<std::uint32_t>(mode), 0);
      std::vector<double> t, y;
      for (int i = 0; i < 2000; i++) {
        const double ti = mode == 2 ? i + z.uniform(-0.4, 0.4) : i;
        if (mode == 2 && ti > 700 && ti < 964) continue;
        t.push_back(ti);
        y.push_back(model(ti, 0, 0.5, 0, truth) + z.normal(0, 1.5));
      }
      ts::HarmonicOptions opt;
      opt.coloured = mode == 1;
      const auto fit = ts::harmonic_fit(t, y, cons, opt);
      for (std::size_t k = 0; k < truth.size(); k++) {
        const auto& term = fit.terms[k];
        cover_A += std::fabs(term.amplitude - truth[k].A) <= 1.96 * term.amplitude_se ? 1 : 0;
        cover_phi += std::fabs(wrap(term.phase - truth[k].phi)) <= 1.96 * term.phase_se ? 1 : 0;
        ratio += term.amplitude_se / (1.5 * std::sqrt(2.0 / static_cast<double>(t.size())));
        count++;
      }
    }
    const std::string what = mode == 0 ? " (white errors)" : mode == 1 ? " (coloured errors of a white residual)" : " (uneven times with a gap: white)";
    s.check("harmonic_fit: ±1.96 A_se covers the amplitude 95 % of the time" + what, cover_A / count - 0.95, 0.02);
    s.check("harmonic_fit: ±1.96 φ_se covers the phase 95 % of the time" + what, cover_phi / count - 0.95, 0.02);
    if (mode < 2) s.check("harmonic_fit: A_se ≈ σ √(2/n) for well-separated constituents" + what, ratio / count - 1, 0.03);
  }

  // ---- red residuals: coloured errors ----
  {
    const double rho = 0.9, f = 1.0 / 24;
    const double excess = (1 - rho * rho) / (1 + rho * rho - 2 * rho * std::cos(two_pi * f));  // S(f) / (2σ²/fs)
    double cover_white = 0, cover_coloured = 0, ratio = 0, count = 0;
    for (std::uint32_t r = 0; r < 300; r++) {
      calx::random::Stream z(44, r, 0, 0);
      std::vector<double> t, y;
      double e = z.normal() / std::sqrt(1 - rho * rho);
      for (int i = 0; i < 4000; i++) {
        e = rho * e + z.normal();
        t.push_back(i);
        y.push_back(2 * std::cos(two_pi * f * i - 1.1) + e);
      }
      const std::vector<ts::Constituent> day{{"day", f}};
      ts::HarmonicOptions white, coloured;
      white.coloured = false;
      const auto fw = ts::harmonic_fit(t, y, day, white);
      const auto fc = ts::harmonic_fit(t, y, day, coloured);
      cover_white += std::fabs(fw.terms[0].amplitude - 2) <= 1.96 * fw.terms[0].amplitude_se ? 1 : 0;
      cover_coloured += std::fabs(fc.terms[0].amplitude - 2) <= 1.96 * fc.terms[0].amplitude_se ? 1 : 0;
      ratio += fc.terms[0].amplitude_se / fw.terms[0].amplitude_se;
      count++;
    }
    s.check("harmonic_fit: under AR(1) residuals (ρ = 0.9) coloured A_se / white A_se ≈ √(S(f)/(2σ²/fs)) (relative)", (ratio / count) / std::sqrt(excess) - 1, 0.06);
    s.check("harmonic_fit: the coloured errors cover 95 %", cover_coloured / count - 0.95, 0.035);
    s.require("harmonic_fit: the white errors fall short (below 85 %)", cover_white / count < 0.85, "coverage " + std::to_string(cover_white / count));
  }
  return s.done();
}
