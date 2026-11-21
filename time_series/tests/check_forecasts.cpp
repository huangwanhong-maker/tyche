// Checks of tyche/time_series/forecasts: the long-run variance of white noise with no lags is its variance, and that of a
// first-order autoregression tends to σ²/(1 − φ)²; the test of equal accuracy gives 0 and p = 1 for identical forecasts,
// applies Harvey, Leybourne and Newbold's factor exactly, keeps its size near 5% when two forecasts are equally good,
// and finds the better one; rolling forecasts by the fitted autoregression and the random walk have, on an AR(1) series,
// the mean squared errors σ² and 2σ²/(1 + φ); the autoregression on a short or constant history is the last value. Clark and
// West's test exactly, near its size between nested models where the test of equal accuracy is undersized, and finding the
// larger model more often; Giacomini and White's test exactly with one and two test functions, near its size, and finding a
// persistent differential of mean zero more often than the unconditional test.

#include <tyche/time_series/forecasts.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace tyche::time_series;

namespace {
std::vector<double> ar1_series(std::size_t n, double phi, std::uint64_t seed) {
  calx::random::Stream r(seed, 7, 0, 0);
  std::vector<double> y(n);
  double x = 0;
  for (std::size_t t = 0; t < 200; t++) x = phi * x + r.normal();
  for (std::size_t t = 0; t < n; t++) x = phi * x + r.normal(), y[t] = x;
  return y;
}
}  // namespace

int main() {
  calx::verification::Suite s;
  {
    calx::random::Stream r(31, 1, 0, 0);
    std::vector<double> w(500);
    double m = 0;
    for (double& v : w) v = r.normal(), m += v;
    m /= 500;
    double var = 0;
    for (double v : w) var += (v - m) * (v - m);
    s.check("the long-run variance with no lags is the variance (divisor T)", long_run_variance(w, 0) - var / 500, 1e-12);
    // With L lags the estimator's expectation is γ₀[1 + 2 Σ_{j ≤ L} (1 − j/(L + 1)) φ^j], γ₀ = σ²/(1 − φ²): Bartlett's weights
    // bias it below σ²/(1 − φ)² = 4 by a few per cent.
    const auto y = ar1_series(400000, 0.5, 5);
    const int L = 60;
    double expect = 1;
    for (int j = 1; j <= L; j++) expect += 2 * (1 - j / (L + 1.0)) * std::pow(0.5, j);
    expect /= 1 - 0.25;
    s.check("the long-run variance of an AR(1) series against its expectation with Bartlett's weights", long_run_variance(y, L) / expect - 1, 0.03);
    s.check("... which is within 3% of σ²/(1 − φ)² = 4", expect / 4 - 1, 0.03);
  }
  {
    const std::vector<double> y{1, 2, 3, 2, 1, 0, 1, 2}, f{1.5, 1.5, 2.5, 2.5, 0.5, 0.5, 1.5, 1.5};
    const auto e = diebold_mariano(y, f, f);
    s.require("identical forecasts: statistic 0, p 1", e.statistic == 0 && e.p == 1);
    const std::vector<double> a{1.1, 2.3, 2.7, 2.2, 0.6, 0.4, 1.0, 2.5, 1.9, 0.8}, b{0.4, 2.9, 3.6, 1.1, 1.8, -0.7, 1.7, 1.2, 2.6, 0.1},
        t{1, 2, 3, 2, 1, 0, 1, 2, 2, 1};
    const auto r = diebold_mariano(t, a, b);
    std::vector<double> d(10);
    for (std::size_t i = 0; i < 10; i++) d[i] = (t[i] - a[i]) * (t[i] - a[i]) - (t[i] - b[i]) * (t[i] - b[i]);
    double dbar = 0;
    for (double v : d) dbar += v / 10;
    const double dm = dbar / std::sqrt(long_run_variance(d, 0) / 10);
    s.check("Harvey, Leybourne and Newbold's factor √((T − 1)/T) one step ahead, exactly", r.statistic - dm * std::sqrt(9.0 / 10), 1e-12);
    s.check("... referred to Student's t with T − 1 degrees of freedom", r.p - tyche::probability::two_sided_t(r.statistic, 9), 1e-15);
  }
  {
    // Two forecasts with independent errors of one variance: the test rejects about 5% of the time.
    calx::random::Stream r(41, 2, 0, 0);
    int rejected = 0, found = 0;
    const int runs = 4000;
    for (int k = 0; k < runs; k++) {
      std::vector<double> y(50), a(50), b(50), c(50);
      for (std::size_t t = 0; t < 50; t++) {
        y[t] = r.normal();
        a[t] = y[t] + r.normal(), b[t] = y[t] + r.normal(), c[t] = y[t] + std::sqrt(2.5) * r.normal();
      }
      rejected += diebold_mariano(y, a, b).p < 0.05;
      const auto better = diebold_mariano(y, a, c);
      found += better.p < 0.05 && better.statistic < 0;
    }
    s.check("equally good forecasts: rejected at 5% about 5% of the time (4,000 runs of 50)", static_cast<double>(rejected) / runs - 0.05, 0.012);
    s.require("a forecast with errors of variance 1 against 2.5: found the better in most runs of 50", found > runs * 0.7, std::to_string(found));
  }
  {
    const auto y = ar1_series(3000, 0.5, 9);
    const auto a = rolling(y, 300, ar1), w = rolling(y, 300, random_walk), m = rolling(y, 300, historical_mean);
    s.check("rolling forecasts by the fitted AR(1) on an AR(1) series: mean squared error σ² = 1", a.mse() - 1, 0.08);
    s.check("... by the random walk: 2σ²/(1 + φ) = 4/3", w.mse() - 4.0 / 3, 0.1);
    s.check("... by the mean: σ²/(1 − φ²) = 4/3", m.mse() - 4.0 / 3, 0.1);
    const auto e = diebold_mariano(a.truth, a.forecast, w.forecast);
    s.require("... and the autoregression beats the random walk by the test", e.p < 0.01 && e.statistic < 0);
  }
  {
    const std::vector<double> two{3, 5}, flat{2, 2, 2, 2};
    s.require("the autoregression on a history too short, or with nothing varying, is the last value", ar1(two) == 5 && ar1(flat) == 2);
  }
  // ---- nested models: Clark and West (decision 0034) ----
  {
    calx::random::Stream r(41, 3, 0, 0);
    std::vector<double> y(40), s1(40), s2(40);
    for (std::size_t t = 0; t < 40; t++) y[t] = r.normal(), s1[t] = 0.3 * r.normal(), s2[t] = s1[t] + 0.4 * r.normal();
    const auto cw = clark_west(y, s1, s2);
    double f = 0, ff = 0;
    for (std::size_t t = 0; t < 40; t++) f += 2 * (y[t] - s1[t]) * (s2[t] - s1[t]) / 40;
    for (std::size_t t = 0; t < 40; t++) {
      const double v = 2 * (y[t] - s1[t]) * (s2[t] - s1[t]) - f;
      ff += v * v / 40;
    }
    s.check("Clark and West's adjusted differential is 2 e₁ (ŷ₂ − ŷ₁), its mean", cw.mean_adjusted - f, 1e-12);
    s.check("... its statistic f̄ / √(V̂/T) with no lags one step ahead", cw.statistic - f / std::sqrt(ff / 40), 1e-12);
    s.check("... its p-value the standard normal's upper tail", cw.p - 0.5 * std::erfc(cw.statistic / std::sqrt(2.0)), 1e-15);
  }
  {
    // The smaller model the historical mean, the larger the fitted autoregression, which nests it. Under the null (no
    // autoregression) Clark and West's test keeps near its size where the test of equal accuracy, one-sided, is undersized; under
    // the alternative it finds the larger model more often.
    const int runs = 600;
    int cw_null = 0, dm_null = 0, cw_alt = 0, dm_alt = 0;
    const double t90 = 1.2816;   // the standard normal's 90% quantile; Student's with 99 degrees of freedom is within 0.01
    for (int k = 0; k < runs; k++) {
      calx::random::Stream r(100 + k, 5, 0, 0);
      std::vector<double> white(200);
      for (auto& v : white) v = 1 + r.normal();
      const auto ar = ar1_series(200, 0.25, 900 + k);
      for (int alt = 0; alt < 2; alt++) {
        const auto& y = alt ? ar : white;
        const auto m = rolling(y, 100, historical_mean), a = rolling(y, 100, ar1);
        const bool cw = clark_west(m.truth, m.forecast, a.forecast).p < 0.10;
        const bool dm = diebold_mariano(m.truth, m.forecast, a.forecast).statistic > t90;
        (alt ? cw_alt : cw_null) += cw, (alt ? dm_alt : dm_null) += dm;
      }
    }
    const double cwn = static_cast<double>(cw_null) / runs, dmn = static_cast<double>(dm_null) / runs;
    std::printf("nested, under the null at 10%%: Clark and West %.3f, equal accuracy one-sided %.3f; under φ = 0.25: %.3f and %.3f\n", cwn, dmn,
                static_cast<double>(cw_alt) / runs, static_cast<double>(dm_alt) / runs);
    s.require("nested models under the null: Clark and West's test rejects at 10% between 5% and 13% of 600 runs", cwn >= 0.05 && cwn <= 0.13, std::to_string(cwn));
    s.require("... where the test of equal accuracy, one-sided, rejects less often (it is undersized between nested models)", dmn < cwn, std::to_string(dmn));
    s.require("under an autoregression of φ = 0.25: Clark and West's test finds the larger model more often than the test of equal accuracy", cw_alt > dm_alt,
              std::to_string(cw_alt) + " against " + std::to_string(dm_alt));
  }
  // ---- conditional predictive ability: Giacomini and White (decision 0034) ----
  {
    calx::random::Stream r(43, 3, 0, 0);
    const std::size_t T = 60;
    std::vector<double> y(T), a(T), b(T), d(T);
    for (std::size_t t = 0; t < T; t++) y[t] = r.normal(), a[t] = y[t] + r.normal(), b[t] = y[t] + 1.2 * r.normal();
    for (std::size_t t = 0; t < T; t++) d[t] = (y[t] - a[t]) * (y[t] - a[t]) - (y[t] - b[t]) * (y[t] - b[t]);
    std::vector<std::vector<double>> one(T, std::vector<double>{1.0});
    const auto u = giacomini_white(y, a, b, one);
    double m = 0, m2 = 0;
    for (double v : d) m += v / T, m2 += v * v / T;
    s.check("Giacomini and White with h_t = 1: n d̄² / ((1/n) Σ d²)", u.statistic - T * m * m / m2, 1e-9);
    s.check("... referred to χ² with one degree of freedom", u.p - std::erfc(std::sqrt(u.statistic / 2)), 1e-12);
    // h_t = (1, d_{t−1}): the statistic n Z̄ᵀ Ω̂⁻¹ Z̄ with Ω̂ inverted by hand.
    const auto g = giacomini_white(y, a, b);
    const std::size_t n = T - 1;
    double z1 = 0, z2 = 0, w11 = 0, w12 = 0, w22 = 0;
    for (std::size_t t = 1; t < T; t++) {
      const double p = d[t], q = d[t - 1] * d[t];
      z1 += p / n, z2 += q / n, w11 += p * p / n, w12 += p * q / n, w22 += q * q / n;
    }
    const double det = w11 * w22 - w12 * w12, stat = n * (z1 * (w22 * z1 - w12 * z2) + z2 * (w11 * z2 - w12 * z1)) / det;
    s.check("... with h_t = (1, d_{t−1}): n Z̄ᵀ Ω̂⁻¹ Z̄, the 2 × 2 inverse by hand (relative)", (g.statistic - stat) / stat, 1e-10);
    s.require("... on n − 1 forecasts, two test functions, referred to χ² with two degrees of freedom", g.n == n && g.q == 2 && std::fabs(g.p - std::exp(-g.statistic / 2)) < 1e-12);
  }
  {
    // Losses equal and unpredictable: rejected at 5% about 5% of the time. A loss differential of mean zero that persists (the
    // better forecast changing rarely): the conditional test finds it where the unconditional one does not.
    const int runs = 1000;
    int null_rej = 0, gw_rej = 0, dm_rej = 0;
    for (int k = 0; k < runs; k++) {
      calx::random::Stream r(300 + k, 7, 0, 0);
      const std::size_t T = 200;
      std::vector<double> y(T), a(T), b(T), a2(T), b2(T);
      double regime = 1;
      for (std::size_t t = 0; t < T; t++) {
        if (r.uniform() < 0.05) regime = -regime;
        y[t] = r.normal();
        a[t] = y[t] + r.normal(), b[t] = y[t] + r.normal();
        a2[t] = y[t] + (regime > 0 ? 0.5 : 1.5) * r.normal(), b2[t] = y[t] + (regime > 0 ? 1.5 : 0.5) * r.normal();
      }
      null_rej += giacomini_white(y, a, b).p < 0.05;
      gw_rej += giacomini_white(y, a2, b2).p < 0.05;
      dm_rej += diebold_mariano(y, a2, b2).p < 0.05;
    }
    const double nr = static_cast<double>(null_rej) / runs;
    std::printf("Giacomini and White: size %.3f at 5%%; a persistent differential of mean zero found %.3f, by equal accuracy %.3f\n", nr,
                static_cast<double>(gw_rej) / runs, static_cast<double>(dm_rej) / runs);
    s.require("losses equal and unpredictable: Giacomini and White's test rejects at 5% between 3% and 8% of 1,000 runs", nr >= 0.03 && nr <= 0.08, std::to_string(nr));
    // The unconditional test, one step ahead, takes no autocovariances of the differential, which here is serially correlated:
    // it rejects often, for the wrong reason (a sample mean drawn by long runs of one forecast's advantage), and less often still
    // than the conditional test, which asks the right question.
    s.require("a persistent differential of mean zero: the conditional test finds it in most runs, and more often than the unconditional",
              gw_rej > runs / 2 && gw_rej > dm_rej, std::to_string(gw_rej) + " against " + std::to_string(dm_rej));
  }
  return s.done();
}
