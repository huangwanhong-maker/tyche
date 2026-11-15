// Checks of linear mixed models (mixed.hpp): in the balanced one-way layout, REML gives the ANOVA estimators (σ̂² the
// mean square within, τ̂² the excess of the mean square between over it, per observation of a group) and ML its known
// variant, the intercept is the grand mean with standard error √(MSB/(mG)), and the BLUPs shrink each group's mean by
// τ̂²/(τ̂² + σ̂²/m); with random intercepts and slopes in unequal groups, the profiled criterion equals the restricted
// likelihood computed from the whole covariance matrix, no small step of θ lowers it, and simulated components are
// recovered.

#include <tyche/mixed/mixed.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

using namespace tyche::mixed;

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261115, 41, 0, 0);

  // ---- The balanced one-way layout ----------------------------------------------------------------------------------------------
  {
    const std::size_t G = 12, m = 6;
    Data d;
    std::vector<double> mean(G, 0);
    for (std::size_t g = 0; g < G; g++) {
      const double a = rng.normal(0, 1.5);
      for (std::size_t j = 0; j < m; j++) {
        d.y.push_back(5 + a + rng.normal());
        d.group.push_back(g);
        mean[g] += d.y.back() / m;
      }
    }
    double grand = 0;
    for (const double v : mean) grand += v / G;
    double ssb = 0, ssw = 0;
    for (std::size_t i = 0; i < d.y.size(); i++) ssw += (d.y[i] - mean[d.group[i]]) * (d.y[i] - mean[d.group[i]]);
    for (const double v : mean) ssb += m * (v - grand) * (v - grand);
    const double msb = ssb / (G - 1), msw = ssw / (G * (m - 1));
    const auto f = fit(d);
    const double tau2 = (msb - msw) / m;
    s.require("the mean square between exceeds the mean square within, so the ANOVA estimator is positive", tau2 > 0);
    s.check("REML: σ̂² is the mean square within", std::abs(f.sigma2 - msw) / msw, 1e-6);
    s.check("and τ̂² is (MSB − MSW)/m", std::abs(f.G(0, 0) - tau2) / tau2, 1e-6);
    s.check("the intercept is the grand mean, with standard error √(MSB/(mG))", std::abs(f.beta[0] - grand) + std::abs(f.se[0] - std::sqrt(msb / (m * G))), 1e-6);
    double worst = 0;
    for (std::size_t g = 0; g < G; g++) worst = std::max(worst, std::abs(f.blup[g][0] - tau2 / (tau2 + msw / m) * (mean[g] - grand)));
    s.check("each group's BLUP is its mean's departure shrunk by τ̂²/(τ̂² + σ̂²/m)", worst, 1e-6);
    s.check("the intraclass correlation is τ̂²/(τ̂² + σ̂²)", std::abs(f.icc - tau2 / (tau2 + msw)), 1e-6);
    Options ml;
    ml.reml = false;
    const auto fm = fit(d, ml);
    const double tau2_ml = ((1 - 1.0 / G) * msb - msw) / m;
    s.check("ML: σ̂² is again the mean square within, and τ̂² is ((1 − 1/G) MSB − MSW)/m, smaller",
            std::abs(fm.sigma2 - msw) / msw + std::abs(fm.G(0, 0) - tau2_ml) / tau2_ml, 1e-5);
  }

  // ---- Random intercepts and slopes, unequal groups ----------------------------------------------------------------------------------
  {
    const std::size_t G = 300;
    Data d;
    d.x.assign(1, {});
    d.z.assign(1, {});
    // G = [[1, 0.3], [0.3, 0.5]]: b0 = z1, b1 = 0.3 z1 + √(0.5 − 0.09) z2.
    for (std::size_t g = 0; g < G; g++) {
      const double z1 = rng.normal(), z2 = rng.normal();
      const double b0 = z1, b1 = 0.3 * z1 + std::sqrt(0.41) * z2;
      const std::size_t m = 3 + rng.below(10);
      for (std::size_t j = 0; j < m; j++) {
        const double x = rng.normal();
        d.x[0].push_back(x);
        d.z[0].push_back(x);
        d.group.push_back(g);
        d.y.push_back(2 + 1 * x + b0 + b1 * x + rng.normal(0, std::sqrt(0.8)));
      }
    }
    const auto f = fit(d);
    s.require("the search converges", f.converged, std::to_string(f.evaluations) + " evaluations");
    s.check("the fixed effects (2, 1) are recovered within 0.15", std::max(std::abs(f.beta[0] - 2), std::abs(f.beta[1] - 1)), 0.15);
    s.check("the covariance of the random effects [[1, 0.3], [0.3, 0.5]] within 0.2", std::max({std::abs(f.G(0, 0) - 1), std::abs(f.G(0, 1) - 0.3), std::abs(f.G(1, 1) - 0.5)}), 0.2);
    s.check("the residual variance 0.8 within 0.1", std::abs(f.sigma2 - 0.8), 0.1);
    double gain = 0;
    for (std::size_t k = 0; k < f.theta.size(); k++)
      for (const double step : {-1e-3, 1e-3}) {
        auto t = f.theta;
        t[k] += step;
        gain = std::max(gain, f.criterion - criterion(d, t));
      }
    s.check("no step of 0.001 in any entry of θ lowers the criterion", gain, 1e-9);
  }
  {
    // The first forty groups' worth of the same design, fitted, and its restricted likelihood computed again from the whole
    // covariance, built group by group into one n × n matrix.
    Data d;
    d.x.assign(1, {});
    d.z.assign(1, {});
    for (std::size_t g = 0; g < 40; g++) {
      const double b0 = rng.normal(), b1 = 0.3 * b0 + std::sqrt(0.41) * rng.normal();
      const std::size_t m = 3 + rng.below(10);
      for (std::size_t j = 0; j < m; j++) {
        const double x = rng.normal();
        d.x[0].push_back(x);
        d.z[0].push_back(x);
        d.group.push_back(g);
        d.y.push_back(2 + x + b0 + b1 * x + rng.normal(0, std::sqrt(0.8)));
      }
    }
    const auto f = fit(d);
    const std::size_t n = d.y.size(), p = 2;
    Matrix V(n);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) {
        if (d.group[i] != d.group[j]) continue;
        const double zi[2] = {1, d.z[0][i]}, zj[2] = {1, d.z[0][j]};
        double v = 0;
        for (int a = 0; a < 2; a++)
          for (int b = 0; b < 2; b++) v += zi[a] * f.G(a, b) * zj[b];
        V(i, j) = v + (i == j ? f.sigma2 : 0);
      }
    const auto cv = calx::linalg::cholesky(V);
    double logdet_v = 0;
    for (std::size_t i = 0; i < n; i++) logdet_v += 2 * std::log(cv.l(i, i));
    std::vector<double> one(n, 1.0);
    const auto vi1 = calx::linalg::cholesky_solve(cv, one), vix = calx::linalg::cholesky_solve(cv, d.x[0]), viy = calx::linalg::cholesky_solve(cv, d.y);
    Matrix xvx(p);
    double xvy[2] = {0, 0};
    for (std::size_t i = 0; i < n; i++) {
      xvx(0, 0) += vi1[i], xvx(0, 1) += vix[i], xvx(1, 1) += vix[i] * d.x[0][i];
      xvy[0] += viy[i], xvy[1] += viy[i] * d.x[0][i];
    }
    xvx(1, 0) = xvx(0, 1);
    const double det = xvx(0, 0) * xvx(1, 1) - xvx(0, 1) * xvx(1, 0);
    const double b0 = (xvx(1, 1) * xvy[0] - xvx(0, 1) * xvy[1]) / det, b1 = (xvx(0, 0) * xvy[1] - xvx(1, 0) * xvy[0]) / det;
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; i++) r[i] = d.y[i] - b0 - b1 * d.x[0][i];
    const auto vir = calx::linalg::cholesky_solve(cv, r);
    double quad = 0;
    for (std::size_t i = 0; i < n; i++) quad += r[i] * vir[i];
    const double dense = (n - p) * std::log(2 * std::numbers::pi) + logdet_v + std::log(det) + quad;
    s.check("the profiled criterion equals −2ℓ_R from the whole covariance matrix, with Woodbury's identity nowhere", std::abs(dense - f.criterion) / f.criterion, 1e-9);
  }
  bool refused = false;
  try { (void)fit(Data{{1, 2}, {}, {}, {0}}); } catch (const std::invalid_argument&) { refused = true; }
  s.require("a group label missing for an observation is refused", refused);
  return s.done();
}
