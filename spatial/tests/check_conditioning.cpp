// Checks of Gaussian conditioning (spatial/conditioning.hpp).
//
// - Against the batch formulas by one dense Cholesky factorisation: posterior means, variances and
//   covariances, the weights C⁻¹ d and the log marginal likelihood.
// - The posterior does not depend on the order of the observations.
// - One observation in closed form: Var = k − c²/(k_o + r), E = c d/(k_o + r).
// - An observation without noise at the query point fixes it: the posterior is the observation, of
//   variance zero.
// - Calibration by simulation (2000 seeded fields drawn from the prior): the errors of held-out values,
//   over their posterior standard deviations, have mean square 1 and cover 95 % at ±1.96; the whitened
//   innovations have mean square 1.

#include <tyche/spatial/conditioning.hpp>

#include <calx/linalg/cholesky.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

using tyche::spatial::GaussianConditioning;

namespace {

// A squared-exponential kernel on a line, σ² = 4, ℓ = 0.5.
double kernel(double x, double y) {
  const double d = (x - y) / 0.5;
  return 4.0 * std::exp(-0.5 * d * d);
}

std::vector<double> covariances(double q, const std::vector<double>& at) {
  std::vector<double> k;
  for (double x : at) k.push_back(kernel(q, x));
  return k;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261029, 0x6b726967u, 0, 0);  // "krig"

  // A field's observations at 20 points of [0, 6], noise variance 0.1.
  std::vector<double> at, d;
  for (int i = 0; i < 20; i++) at.push_back(rng.uniform(0, 6)), d.push_back(rng.normal(0, 2));
  const double noise = 0.1;
  GaussianConditioning G;
  for (std::size_t i = 0; i < at.size(); i++) G.observe(covariances(at[i], {at.begin(), at.begin() + long(i)}), kernel(at[i], at[i]), noise, d[i]);

  // ---- against the batch ------------------------------------------------------------------------------------------
  {
    const std::size_t n = at.size();
    calx::linalg::Matrix C(n);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) C(i, j) = kernel(at[i], at[j]) + (i == j ? noise : 0.0);
    const auto f = calx::linalg::cholesky(C);
    const std::vector<double> w = calx::linalg::cholesky_solve(f, d);
    double mean = 0, var = 0, cov = 0, weights = 0;
    for (int j = 0; j <= 60; j++) {
      const double q = 0.1 * j;
      const auto k = covariances(q, at);
      const auto Ck = calx::linalg::cholesky_solve(f, k);
      double m = 0, v = kernel(q, q);
      for (std::size_t i = 0; i < n; i++) m += k[i] * w[i], v -= k[i] * Ck[i];
      const auto p = G.predict(k, 0, kernel(q, q));
      mean = std::max(mean, std::fabs(p.mean - m) / 2), var = std::max(var, std::fabs(p.variance - std::max(v, 0.0)) / 4);
      mean = std::max(mean, std::fabs(G.mean(k, 0) - m) / 2);
      const double q2 = 6 - q;
      const auto k2 = covariances(q2, at);
      double c = kernel(q, q2);
      for (std::size_t i = 0; i < n; i++) c -= k2[i] * Ck[i];
      cov = std::max(cov, std::fabs(G.covariance(k, k2, kernel(q, q2)) - c) / 4);
    }
    for (std::size_t i = 0; i < n; i++) weights = std::max(weights, std::fabs(G.weights()[i] - w[i]) / std::max(1.0, std::fabs(w[i])));
    double logdet = 0, quad = 0;
    for (std::size_t i = 0; i < n; i++) logdet += 2 * std::log(f.l(i, i)), quad += d[i] * w[i];
    const double ll = -0.5 * quad - 0.5 * logdet - 0.5 * double(n) * std::log(2 * std::numbers::pi);
    s.check("batch: the posterior means, one observation at a time, are the batch's (61 queries; over the prior sd)", mean, 1e-12);
    s.check("batch: and their variances (over the prior variance)", var, 1e-12);
    s.check("batch: and the posterior covariances of pairs of queries", cov, 1e-12);
    {
      const auto ka = covariances(1.3, at), kb = covariances(4.2, at);
      const auto wa = G.whiten(ka), wb = G.whiten(kb);
      double c = kernel(1.3, 4.2), m = 0;
      for (std::size_t i = 0; i < n; i++) c -= wa[i] * wb[i], m += wa[i] * G.whitened()[i];
      s.check("batch: whitened covariances give the posterior covariance and mean, k_ab − (L⁻¹k_a)·(L⁻¹k_b) and (L⁻¹k)·z",
              std::max(std::fabs(c - G.covariance(ka, kb, kernel(1.3, 4.2))), std::fabs(m - G.mean(ka, 0))), 1e-13);
    }
    s.check("batch: the weights C⁻¹ d", weights, 1e-12);
    s.check("batch: the log marginal likelihood −½ dᵀC⁻¹d − ½ log det C − (n/2) log 2π", std::fabs(G.log_likelihood() - ll), 1e-11);
  }

  // ---- the order of the observations -------------------------------------------------------------------------------
  {
    std::vector<double> rat(at.rbegin(), at.rend()), rd(d.rbegin(), d.rend());
    GaussianConditioning R;
    for (std::size_t i = 0; i < rat.size(); i++) R.observe(covariances(rat[i], {rat.begin(), rat.begin() + long(i)}), kernel(rat[i], rat[i]), noise, rd[i]);
    double worst = 0;
    for (int j = 0; j <= 60; j++) {
      const double q = 0.1 * j;
      const auto a = G.predict(covariances(q, at), 0, 4), b = R.predict(covariances(q, rat), 0, 4);
      worst = std::max({worst, std::fabs(a.mean - b.mean) / 2, std::fabs(a.variance - b.variance) / 4});
    }
    s.check("order: the posterior of the observations reversed is the same", worst, 1e-12);
    s.check("order: and so is their likelihood", std::fabs(G.log_likelihood() - R.log_likelihood()), 1e-11);
  }

  // ---- closed forms --------------------------------------------------------------------------------------------------
  {
    GaussianConditioning one;
    const double ko = 4, r = 0.5, dd = 1.3, c = kernel(1.0, 1.4);
    one.observe({}, ko, r, dd);
    const auto p = one.predict(std::vector<double>{c}, 0.2, 4);
    s.check("one observation: E = E₀ + c d/(k_o + r), Var = k − c²/(k_o + r)",
            std::max(std::fabs(p.mean - (0.2 + c * dd / (ko + r))), std::fabs(p.variance - (4 - c * c / (ko + r)))), 1e-15);
    GaussianConditioning exact;
    exact.observe({}, 4, 0, 2.5);
    const auto e = exact.predict(std::vector<double>{4.0}, 0, 4);
    s.check("no noise: an observation at the query point fixes it (mean − the value, and the variance)", std::max(std::fabs(e.mean - 2.5), e.variance), 1e-15);
  }

  // ---- calibration: fields drawn from the prior ------------------------------------------------------------------------
  {
    const int grid = 61, trials = 2000, observed = 15;
    calx::linalg::Matrix K(grid);
    for (int i = 0; i < grid; i++)
      for (int j = 0; j < grid; j++) K(std::size_t(i), std::size_t(j)) = kernel(0.1 * i, 0.1 * j) + (i == j ? 1e-10 : 0.0);
    const auto chol = calx::linalg::cholesky(K);
    double square = 0, whitened = 0;
    int covered = 0, innovations = 0;
    for (int t = 0; t < trials; t++) {
      std::vector<double> z(grid), f(grid, 0.0);
      for (auto& x : z) x = rng.normal();
      for (int i = 0; i < grid; i++)
        for (int k = 0; k <= i; k++) f[std::size_t(i)] += chol.l(std::size_t(i), std::size_t(k)) * z[std::size_t(k)];
      // Observations at distinct random points of the grid, the last point held out.
      std::vector<int> order(grid);
      for (int i = 0; i < grid; i++) order[std::size_t(i)] = i;
      for (int i = grid - 1; i > 0; i--) std::swap(order[std::size_t(i)], order[rng.below(std::uint64_t(i) + 1)]);
      const double r = 0.05;
      GaussianConditioning P;
      std::vector<double> xs;
      for (int i = 0; i < observed; i++) {
        const double x = 0.1 * order[std::size_t(i)];
        P.observe(covariances(x, xs), kernel(x, x), r, f[std::size_t(order[std::size_t(i)])] + rng.normal(0, std::sqrt(r)));
        xs.push_back(x);
      }
      const int held = order[std::size_t(observed)];
      const auto p = P.predict(covariances(0.1 * held, xs), 0, kernel(0, 0));
      const double e = (f[std::size_t(held)] - p.mean) / p.sd();
      square += e * e;
      covered += std::fabs(e) < 1.959963984540054;
      for (double w : P.whitened()) whitened += w * w, innovations++;
    }
    // Standard errors: √(2/2000) ≈ 0.032 for the mean square, √(0.95 · 0.05/2000) ≈ 0.005 for the coverage,
    // √(2/30 000) ≈ 0.008 for the innovations; the limits are about four of them.
    s.check("calibration: held-out errors over their posterior sd have mean square 1 (2000 fields)", std::fabs(square / trials - 1), 0.13);
    s.check("calibration: and ±1.96 sd covers 95 % of them", std::fabs(double(covered) / trials - 0.95), 0.02);
    s.check("calibration: the whitened innovations have mean square 1", std::fabs(whitened / innovations - 1), 0.033);
  }

  return s.done();
}
