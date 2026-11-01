// Checks of the binomial mixture with known scales (binomial_mixture.hpp): the one-component fit
// against its closed form with equal scales and against the root of its score with unequal ones;
// the EM's monotone likelihood; recovery of three components from simulated counts and the choice
// of three by BIC; the bound on θ; the same fit for the same seed and data.

#include <tyche/multivariate/binomial_mixture.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace tyche::multivariate;

int main() {
  calx::verification::Suite s;
  calx::random::Stream r(42, 0, 0, 0);

  // ---- one component ---------------------------------------------------------------------------------
  {
    std::vector<double> k, n, c;
    for (int i = 0; i < 200; i++) {
      n.push_back(static_cast<double>(20 + r.below(80)));
      c.push_back(0.4);
      k.push_back(static_cast<double>(r.binomial(static_cast<std::int64_t>(n.back()), 0.4 * 0.7)));
    }
    BinomialMixtureOptions o;
    const auto f = fit_binomial_mixture(k, n, c, o);
    double sk = 0, sn = 0;
    for (std::size_t i = 0; i < k.size(); i++) { sk += k[i]; sn += n[i]; }
    s.check("one component, equal scales: θ = Σk / (c Σn)", f.theta[0] - sk / (0.4 * sn), 1e-12);
    s.check("one component: weight 1", f.weights[0] - 1, 1e-15);
    // Unequal scales: the score Σ [k/θ − (n − k) c/(1 − cθ)] vanishes at θ̂.
    for (auto& x : c) x = 0.2 + 0.6 * r.uniform();
    for (std::size_t i = 0; i < k.size(); i++) k[i] = static_cast<double>(r.binomial(static_cast<std::int64_t>(n[i]), c[i] * 0.55));
    const auto g = fit_binomial_mixture(k, n, c, o);
    double score = 0;
    for (std::size_t i = 0; i < k.size(); i++) score += k[i] / g.theta[0] - (n[i] - k[i]) * c[i] / (1 - c[i] * g.theta[0]);
    s.check("one component, unequal scales: the score at θ̂", score, 1e-6);
    s.check("one component: θ̂ near the truth 0.55", g.theta[0] - 0.55, 0.03);
  }

  // ---- three components ------------------------------------------------------------------------------
  {
    const std::vector<double> theta{1.0, 0.55, 0.25}, weight{0.5, 0.3, 0.2};
    std::vector<double> k, n, c;
    std::vector<int> z;
    for (int i = 0; i < 900; i++) {
      const double u = r.uniform();
      const int j = u < weight[0] ? 0 : u < weight[0] + weight[1] ? 1 : 2;
      z.push_back(j);
      n.push_back(static_cast<double>(r.poisson(80)) + 10);
      const int m = r.bernoulli(0.2) ? 2 : 1, N = r.bernoulli(0.3) ? 3 : 2;
      const double rho = 0.7;
      c.push_back(rho * m / (2 * (1 - rho) + rho * N));
      k.push_back(static_cast<double>(r.binomial(static_cast<std::int64_t>(n.back()), c.back() * theta[static_cast<std::size_t>(j)])));
    }
    BinomialMixtureOptions o;
    const auto sel = select_binomial_mixture(k, n, c, 6, o);
    const auto& f = sel.fits[sel.best];
    std::printf("      BIC by K:");
    for (const auto& x : sel.fits) std::printf(" %.1f", x.bic);
    std::printf("\n      fit: θ");
    for (double t : f.theta) std::printf(" %.3f", t);
    std::printf(", weights");
    for (double w : f.weights) std::printf(" %.3f", w);
    std::printf("\n");
    s.require("three components chosen by BIC", f.theta.size() == 3, std::to_string(f.theta.size()));
    double worst = 0, worst_w = 0;
    for (std::size_t j = 0; j < std::min<std::size_t>(3, f.theta.size()); j++) {
      worst = std::max(worst, std::fabs(f.theta[j] - theta[j]));
      worst_w = std::max(worst_w, std::fabs(f.weights[j] - weight[j]));
    }
    s.check("three components: θ recovered", worst, 0.03);
    s.check("three components: weights recovered", worst_w, 0.04);
    std::size_t right = 0;
    for (std::size_t i = 0; i < z.size(); i++) right += static_cast<int>(f.assignment[i]) == z[i];
    s.require("three components: 90% of the observations assigned to their own", static_cast<double>(right) >= 0.9 * static_cast<double>(z.size()), std::to_string(right));
    bool monotone = true;
    std::string iterations;
    for (const auto& x : sel.fits) {
      monotone = monotone && x.monotone;
      iterations += " " + std::to_string(x.iterations) + (x.converged ? "" : "*");
    }
    s.require("every EM run's likelihood is non-decreasing", monotone, "iterations by K:" + iterations);
    s.require("the chosen fit converged", f.converged, std::to_string(f.iterations) + " iterations");
    bool bounded = true;
    for (const auto& x : sel.fits)
      for (double t : x.theta) bounded = bounded && t > 0 && t <= 1;
    s.require("θ stays in (0, 1]", bounded);
    const auto again = select_binomial_mixture(k, n, c, 6, o);
    s.require("the same fit for the same data and seed", again.best == sel.best && again.fits[again.best].theta == f.theta);
  }
  return s.done();
}
