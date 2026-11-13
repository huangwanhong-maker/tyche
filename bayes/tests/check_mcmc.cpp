// Checks of tyche::bayes's Markov chain Monte Carlo: Metropolis on a normal law and HMC on a correlated five-dimensional normal, their
// means within four Monte Carlo errors and their spreads and correlation near the truth, R̂ below 1.01; HMC's acceptance near its
// target; both samplers on a beta–binomial posterior, against the conjugate answer; the effective sample size of an AR(1) series
// against n(1 − φ)/(1 + φ); and R̂ flagging chains that do not mix.

#include <tyche/bayes/bayes.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <span>
#include <vector>

using namespace tyche::bayes;

int main() {
  calx::verification::Suite s;

  // ---- Metropolis on N(3, 2²) ----
  {
    Target t{1, [](std::span<const double> x) { return -0.5 * (x[0] - 3) * (x[0] - 3) / 4; }, {}};
    McmcOptions o;
    o.draws = 4000, o.warmup = 2000, o.seed = 3;
    const auto C = metropolis(t, o);
    s.check("Metropolis, N(3, 2²): the mean within four Monte Carlo errors", std::fabs(C.mean[0] - 3) / C.mcse[0], 4.0);
    s.check("Metropolis, N(3, 2²): the standard deviation within 5%", std::fabs(C.sd[0] / 2 - 1), 0.05);
    s.check("Metropolis, N(3, 2²): R̂ below 1.01", C.rhat[0] - 1, 0.01);
    s.check("Metropolis: the acceptance near 0.44 in one dimension, at least 0.2 (the target is 0.234 asymptotically)", std::fabs(C.acceptance[0] - 0.35), 0.15);
  }
  // ---- HMC on a correlated five-dimensional normal ----
  {
    const double r = 0.9, s0 = 1, s1 = 3;  // x₀, x₁ correlated 0.9 with scales 1 and 3; x₂…x₄ independent N(0, 1), N(−2, 0.5²), N(5, 10²)
    const double det = s0 * s0 * s1 * s1 * (1 - r * r);
    auto lp = [=](std::span<const double> x) {
      const double a = x[0], b = x[1] - 1;
      const double q = (s1 * s1 * a * a - 2 * r * s0 * s1 * a * b + s0 * s0 * b * b) / det;
      return -0.5 * q - 0.5 * x[2] * x[2] - 0.5 * (x[3] + 2) * (x[3] + 2) / 0.25 - 0.5 * (x[4] - 5) * (x[4] - 5) / 100;
    };
    auto grad = [=](std::span<const double> x, std::span<double> g) {
      const double a = x[0], b = x[1] - 1;
      g[0] = -(s1 * s1 * a - r * s0 * s1 * b) / det;
      g[1] = -(s0 * s0 * b - r * s0 * s1 * a) / det;
      g[2] = -x[2], g[3] = -(x[3] + 2) / 0.25, g[4] = -(x[4] - 5) / 100;
    };
    McmcOptions o;
    o.draws = 2000, o.warmup = 1000, o.seed = 5;
    const auto C = hmc(Target{5, lp, grad}, o);
    const double truth[5] = {0, 1, 0, -2, 5}, sd[5] = {1, 3, 1, 0.5, 10};
    double z = 0, spread = 0, rh = 0;
    for (int k = 0; k < 5; k++) {
      z = std::max(z, std::fabs(C.mean[k] - truth[k]) / C.mcse[k]);
      spread = std::max(spread, std::fabs(C.sd[k] / sd[k] - 1));
      rh = std::max(rh, C.rhat[k] - 1);
    }
    double cov = 0, n = 0;
    for (const auto& ch : C.draws)
      for (const auto& x : ch) cov += (x[0] - C.mean[0]) * (x[1] - C.mean[1]), n++;
    const double corr = cov / n / (C.sd[0] * C.sd[1]);
    s.check("HMC, five dimensions: every mean within four Monte Carlo errors", z, 4.0);
    s.check("HMC: every standard deviation within 6%, across scales from 0.5 to 10", spread, 0.06);
    s.check("HMC: the correlation 0.9 recovered within 0.02", std::fabs(corr - 0.9), 0.02);
    s.check("HMC: R̂ below 1.01 everywhere", rh, 0.01);
    s.check("HMC: the acceptance near its target 0.8", std::fabs(C.acceptance[0] - 0.8), 0.1);
    s.require("HMC: no divergent trajectory on a normal law", C.divergent == 0);
  }
  // ---- both samplers on a beta–binomial posterior, in the logit ----
  {
    const double a = 2, b = 3, k = 7, n = 10;
    // θ = logit p; p ~ Beta(a + k, b + n − k): log density (a + k) log p + (b + n − k) log(1 − p), the Jacobian p(1 − p) included.
    Target t{1, [=](std::span<const double> x) {
               const double p = 1 / (1 + std::exp(-x[0]));
               return (a + k) * std::log(p) + (b + n - k) * std::log(1 - p);
             }, {}};
    McmcOptions o;
    o.draws = 4000, o.warmup = 1000, o.seed = 7;
    const auto exact = beta_binomial(a, b, k, n);
    for (int which = 0; which < 2; which++) {
      const auto C = which == 0 ? metropolis(t, o) : hmc(t, o);
      double m = 0, cnt = 0;
      for (const auto& ch : C.draws)
        for (const auto& x : ch) m += 1 / (1 + std::exp(-x[0])), cnt++;
      m /= cnt;
      const double se = std::sqrt(exact.variance / C.ess[0]);
      s.check(which == 0 ? "Metropolis on a beta–binomial posterior: E[p] = 0.6 within four errors" : "HMC on a beta–binomial posterior: E[p] = 0.6 within four errors",
              std::fabs(m - exact.mean) / se, 4.0);
    }
  }
  // ---- the effective sample size of an AR(1) series; R̂ of chains that do not mix ----
  {
    calx::random::Stream rng(9, 0, 0, 0);
    const double phi = 0.7;
    std::vector<std::vector<std::vector<double>>> chains(4, std::vector<std::vector<double>>(5000, std::vector<double>(1)));
    for (auto& ch : chains) {
      double x = rng.normal() / std::sqrt(1 - phi * phi);
      for (auto& v : ch) x = phi * x + rng.normal(), v[0] = x;
    }
    const auto D = diagnose(chains);
    const double expected = 20000 * (1 - phi) / (1 + phi);
    s.check("ESS of AR(1), φ = 0.7: n (1 − φ)/(1 + φ) = 3529 within 15%", std::fabs(D.ess[0] / expected - 1), 0.15);
    for (std::size_t c = 0; c < chains.size(); c++)
      for (auto& v : chains[c]) v[0] += 3.0 * static_cast<double>(c);
    s.require("R̂ flags four chains about different means (R̂ > 1.5)", diagnose(chains).rhat[0] > 1.5);
  }
  return s.done();
}
