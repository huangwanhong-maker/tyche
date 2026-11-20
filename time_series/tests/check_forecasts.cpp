// Checks of tyche/time_series/forecasts: the long-run variance of white noise with no lags is its variance, and that of a
// first-order autoregression tends to σ²/(1 − φ)²; the test of equal accuracy gives 0 and p = 1 for identical forecasts,
// applies Harvey, Leybourne and Newbold's factor exactly, keeps its size near 5% when two forecasts are equally good,
// and finds the better one; rolling forecasts by the fitted autoregression and the random walk have, on an AR(1) series,
// the mean squared errors σ² and 2σ²/(1 + φ); the autoregression on a short or constant history is the last value.

#include <tyche/time_series/forecasts.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
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
  return s.done();
}
