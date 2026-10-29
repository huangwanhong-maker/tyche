// Checks of tyche/time_series's AR and ARIMA models.
//
// - ψ-weights in closed form: ARMA(1, 1), ψ_j = (φ + θ)φ^{j−1} (Box, Jenkins and Reinsel 2008, §3.4);
//   stationarity and invertibility of AR(2) and MA(1) polynomials on either side of the boundary.
// - AR(2) by least squares and by Yule–Walker on a seeded simulation: φ within four standard errors
//   (Yule–Walker's asymptotic variance (1 − φ₂²)/n: Brockwell and Davis 1991, §8.1).
// - CSS for a pure AR is least squares: arima({2, 0, 0}) against ar_ols on the same series.
// - ARMA(1, 1) by CSS on a seeded simulation of 2000: φ and θ within four standard errors, and the
//   standard errors from the Hessian within 10% of the asymptotic ones,
//   Var φ̂ = (1 + φθ)²(1 − φ²)/(n(φ + θ)²), Var θ̂ = (1 + φθ)²(1 − θ²)/(n(φ + θ)²)
//   (Brockwell and Davis 1991, §8.8; Box, Jenkins and Reinsel 2008, §7.2).
// - ARIMA(0, 1, 1) with drift recovered; forecasts in closed form: an AR(1)'s
//   μ + φ^h (x_n − μ) with variance σ²(1 − φ^{2h})/(1 − φ²), a random walk with drift's x_n + hμ
//   with σ√h; the residuals of a correct model pass the Ljung–Box test.

#include <tyche/time_series/arima.hpp>
#include <tyche/time_series/autocorrelation.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace tyche::time_series;
using calx::verification::relative;

namespace {

// x_t − μ = Σ φ_i (x_{t−i} − μ) + ε_t + Σ θ_j ε_{t−j}, after a burn-in of 500.
std::vector<double> simulate_arma(std::uint32_t id, std::size_t n, double mu, std::vector<double> phi, std::vector<double> theta, double sigma) {
  calx::random::Stream rng(20261024, 61, id, 0);
  const std::size_t burn = 500, N = n + burn;
  std::vector<double> x(N, 0.0), e(N, 0.0);
  for (std::size_t t = 0; t < N; t++) {
    e[t] = sigma * rng.normal();
    double v = e[t];
    for (std::size_t i = 1; i <= phi.size() && i <= t; i++) v += phi[i - 1] * x[t - i];
    for (std::size_t j = 1; j <= theta.size() && j <= t; j++) v += theta[j - 1] * e[t - j];
    x[t] = v;
  }
  std::vector<double> out(x.begin() + static_cast<std::ptrdiff_t>(burn), x.end());
  for (double& v : out) v += mu;
  return out;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- ψ-weights and roots ----
  {
    const std::vector<double> phi{0.6}, theta{0.3};
    const auto psi = psi_weights(phi, theta, 8);
    double worst = std::fabs(psi[0] - 1);
    for (int j = 1; j < 8; j++) worst = std::max(worst, std::fabs(psi[j] - 0.9 * std::pow(0.6, j - 1)));
    s.check("ψ-weights: ARMA(1, 1), ψ_j = (φ + θ)φ^{j−1}", worst, 1e-15);
    s.require("roots: AR(2) (0.5, 0.3) is stationary, (1.2, −0.1) is not", stationary(std::vector<double>{0.5, 0.3}) && !stationary(std::vector<double>{1.2, -0.1}));
    s.require("roots: AR(1) at φ = 1 (a unit root) is not stationary", !stationary(std::vector<double>{1.0}));
    s.require("roots: MA(1) θ = 0.5 is invertible, θ = 2 is not", invertible(std::vector<double>{0.5}) && !invertible(std::vector<double>{2.0}));
  }

  // ---- AR(2) by least squares and Yule–Walker ----
  const auto ar2 = simulate_arma(1, 1000, 2.0, {0.5, 0.3}, {}, 1.0);
  {
    const ArFit o = ar_ols(ar2, 2);
    const double z = std::max(std::fabs(o.phi[0] - 0.5) / o.se[1], std::fabs(o.phi[1] - 0.3) / o.se[2]);
    s.check("ar_ols: AR(2) recovered within 4 standard errors (largest |φ̂ − φ|/se)", z, 4);
    s.check("ar_ols: its mean c/(1 − Σφ) near 2 (within 4 × σ/((1 − Σφ)√n))", std::fabs(o.mean - 2) / (1 / (0.2 * std::sqrt(1000.0))), 4);
    s.require("ar_ols: and it is stationary", o.is_stationary);
    const ArFit y = ar_yule_walker(ar2, 2);
    const double se = std::sqrt((1 - 0.09) / 1000);
    s.check("ar_yule_walker: within 4 asymptotic standard errors √((1 − φ₂²)/n)", std::max(std::fabs(y.phi[0] - 0.5), std::fabs(y.phi[1] - 0.3)) / se, 4);
    s.check("ar_yule_walker: close to least squares on a long series", std::max(std::fabs(y.phi[0] - o.phi[0]), std::fabs(y.phi[1] - o.phi[1])), 0.02);
  }

  // ---- CSS for a pure AR is least squares ----
  {
    const ArFit o = ar_ols(ar2, 2);
    const ArimaFit a = arima(ar2, {2, 0, 0});
    s.require("arima css: AR(2) converges", a.converged);
    s.check("arima css: AR(2)'s φ equal least squares'", std::max(std::fabs(a.phi[0] - o.phi[0]), std::fabs(a.phi[1] - o.phi[1])), 1e-6);
    s.check("arima css: μ(1 − Σφ) is least squares' constant", a.mu * (1 - a.phi[0] - a.phi[1]) - o.constant, 1e-5);
    s.check("arima css: the CSS is least squares' RSS", relative(a.css, o.sigma2 * (1000 - 2 - 3)), 1e-9);
    s.check("arima css: and the standard errors of φ agree with least squares' to 1%", std::max(relative(a.se[1], o.se[1]), relative(a.se[2], o.se[2])), 0.01);
  }

  // ---- ARMA(1, 1) ----
  {
    const double phi = 0.6, theta = 0.3, n = 2000;
    const auto x = simulate_arma(2, 2000, -1.0, {phi}, {theta}, 1.5);
    const ArimaFit a = arima(x, {1, 0, 1});
    s.require("arima css: ARMA(1, 1) converges, stationary and invertible", a.converged && a.is_stationary && a.is_invertible);
    const double zf = std::fabs(a.phi[0] - phi) / a.se[1], zt = std::fabs(a.theta[0] - theta) / a.se[2];
    s.check("arima css: φ and θ within 4 standard errors (largest)", std::max(zf, zt), 4);
    const double k = (1 + phi * theta) * (1 + phi * theta) / (n * (phi + theta) * (phi + theta));
    s.check("arima css: se(φ̂) within 10% of √((1 + φθ)²(1 − φ²)/(n(φ + θ)²))", relative(a.se[1], std::sqrt(k * (1 - phi * phi))), 0.1);
    s.check("arima css: se(θ̂) within 10% of √((1 + φθ)²(1 − θ²)/(n(φ + θ)²))", relative(a.se[2], std::sqrt(k * (1 - theta * theta))), 0.1);
    s.check("arima css: σ̂² near 2.25 (within 4 × σ²√(2/m))", std::fabs(a.sigma2 - 2.25) / (2.25 * std::sqrt(2 / n)), 4);
    const auto lb = ljung_box(a.residuals, 20, 2);
    s.require("arima css: its residuals pass the Ljung–Box test (p > 0.01)", lb.p_value > 0.01, std::to_string(lb.p_value));
    s.check("arima: AIC = −2ℓ + 2(k + 1) with k = 3", a.aic - (-2 * a.log_likelihood + 8), 1e-12);
  }

  // ---- ARIMA(0, 1, 1) with drift ----
  {
    const auto w = simulate_arma(3, 1500, 0.2, {}, {-0.4}, 1.0);  // the differences
    std::vector<double> x(w.size() + 1, 100.0);
    for (std::size_t t = 0; t < w.size(); t++) x[t + 1] = x[t] + w[t];
    const ArimaFit a = arima(x, {0, 1, 1});
    s.check("arima css: ARIMA(0, 1, 1) θ = −0.4 within 4 standard errors", std::fabs(a.theta[0] + 0.4) / a.se[1], 4);
    s.check("arima css: and its drift 0.2 within 4 standard errors", std::fabs(a.mu - 0.2) / a.se[0], 4);
  }

  // ---- forecasts in closed form ----
  {
    const auto x = simulate_arma(4, 400, 5.0, {0.7}, {}, 0.8);
    const ArimaFit a = arima(x, {1, 0, 0});
    const auto f = forecast(a, 6);
    double em = 0, es = 0;
    const double ph = a.phi[0], mu = a.mu, xn = x.back();
    for (int h = 1; h <= 6; h++) {
      em = std::max(em, std::fabs(f.mean[h - 1] - (mu + std::pow(ph, h) * (xn - mu))));
      es = std::max(es, relative(f.se[h - 1], std::sqrt(a.sigma2 * (1 - std::pow(ph, 2 * h)) / (1 - ph * ph))));
    }
    s.check("forecast: AR(1), μ + φ^h (x_n − μ)", em, 1e-12);
    s.check("forecast: and its standard error σ√((1 − φ^{2h})/(1 − φ²))", es, 1e-12);
    s.check("forecast: the 95% interval is ± 1.959963984540054 se", f.upper[2] - f.mean[2] - 1.959963984540054 * f.se[2], 1e-12);
  }
  {
    calx::random::Stream rng(20261024, 62, 0, 0);
    std::vector<double> x{10};
    for (int t = 1; t < 300; t++) x.push_back(x.back() + 0.5 + 2 * rng.normal());
    const ArimaFit a = arima(x, {0, 1, 0});
    const auto f = forecast(a, 4);
    double em = 0, es = 0;
    for (int h = 1; h <= 4; h++) {
      em = std::max(em, std::fabs(f.mean[h - 1] - (x.back() + h * a.mu)));
      es = std::max(es, relative(f.se[h - 1], std::sqrt(a.sigma2 * h)));
    }
    s.check("forecast: a random walk with drift, x_n + hμ", em, 1e-11);
    s.check("forecast: and σ√h", es, 1e-13);
    s.check("arima: the drift of (0, 1, 0) is the mean of the differences", a.mu - (x.back() - x.front()) / 299, 1e-6);
  }
  return s.done();
}
