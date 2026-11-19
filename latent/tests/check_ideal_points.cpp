// Checks of ideal points from votes (ideal_points.hpp).
//
// - Recovery: on a simulated chamber of 120 members and 400 items, a fifth of the votes missing, the fitted ideal points correlate
//   with the true ones above 0.97, on the anchor's side, and the line classifies the votes as well as the true line does.
// - EM's monotonicity: the log posterior never falls from one iteration to the next.
// - The posterior mode by another route: on 15 members and 30 items, BFGS on the log posterior with its analytic gradient, started
//   near EM's answer, reaches the same parameters and the same value.
// - Reflection leaves the posterior unchanged; `standardised` leaves every α + β x unchanged.
// - The far tail: φ/Φ and log Φ continuous where the continued fraction takes over; a unanimous item and a member who never breaks
//   ranks give finite answers.
// - The bootstrap: in one chamber, four times the votes narrow the intervals; a member who casts a fifth of its votes has a wider interval
//   than when it casts them all; the 90% intervals cover the truth, carried onto the fit's location and scale, for most members.

#include <tyche/latent/ideal_points.hpp>

#include <calx/optimisation/quasi_newton.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <span>
#include <string>
#include <vector>

using namespace tyche::latent;

namespace {

struct Chamber {
  Votes y;
  std::vector<double> x, a, b;
};

Chamber simulate(std::size_t n, std::size_t m, double missing, calx::random::Stream& rng) {
  Chamber c;
  c.x.resize(n);
  for (auto& v : c.x) v = rng.normal();
  std::vector<double> a(m), b(m);
  for (std::size_t j = 0; j < m; j++) a[j] = rng.normal(0, 1), b[j] = rng.normal(0, 1.5);
  c.a = a, c.b = b;
  c.y.assign(n, std::vector<int>(m, -1));
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < m; j++) {
      if (rng.uniform() < missing) continue;
      c.y[i][j] = a[j] + b[j] * c.x[i] + rng.normal() > 0 ? 1 : 0;
    }
  return c;
}

double correlation(const std::vector<double>& u, const std::vector<double>& v) {
  const double n = static_cast<double>(u.size());
  double mu = 0, mv = 0;
  for (std::size_t i = 0; i < u.size(); i++) mu += u[i] / n, mv += v[i] / n;
  double suv = 0, suu = 0, svv = 0;
  for (std::size_t i = 0; i < u.size(); i++) suv += (u[i] - mu) * (v[i] - mv), suu += (u[i] - mu) * (u[i] - mu), svv += (v[i] - mv) * (v[i] - mv);
  return suv / std::sqrt(suu * svv);
}

std::size_t positive_member(const std::vector<double>& x) {
  for (std::size_t i = 0; i < x.size(); i++)
    if (x[i] > 0.5) return i;
  return 0;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- Recovery and monotonicity -------------------------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261119, 30, 0, 0);
    const auto c = simulate(120, 400, 0.2, rng);
    IdealPointOptions o;
    o.anchor = positive_member(c.x);
    const auto f = ideal_points(c.y, o);
    const double r = correlation(f.x, c.x);
    s.require("120 members, 400 items, a fifth of the votes missing: the fitted ideal points correlate with the true ones above 0.97, on the anchor's side",
              r > 0.97 && f.converged, "r = " + std::to_string(r) + ", " + std::to_string(f.iterations) + " iterations");
    // the share of votes the true line classifies: what any fit can hope for, the rest being the votes' own noise
    double right = 0, all = 0;
    for (std::size_t i = 0; i < 120; i++)
      for (std::size_t j = 0; j < 400; j++)
        if (c.y[i][j] == 0 || c.y[i][j] == 1) all += 1, right += ((c.a[j] + c.b[j] * c.x[i] > 0) == (c.y[i][j] == 1));
    s.require("the fitted line classifies the votes as well as the true line does, within two points", std::abs(f.classified - right / all) < 0.02 && f.apre > 0,
              "fitted " + std::to_string(f.classified) + ", true " + std::to_string(right / all) + ", APRE " + std::to_string(f.apre));
    double fall = 0;
    for (std::size_t k = 1; k < f.log_posterior.size(); k++) fall = std::max(fall, f.log_posterior[k - 1] - f.log_posterior[k]);
    s.check("EM never lowers the log posterior", fall, 1e-8);
  }

  // ---- The posterior mode by BFGS ---------------------------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261119, 31, 0, 0);
    const auto c = simulate(15, 30, 0.1, rng);
    IdealPointOptions o;
    o.anchor = positive_member(c.x), o.tolerance = 0, o.parameter_tolerance = 1e-13, o.most = 400000;
    const auto f = ideal_points(c.y, o);
    const std::size_t n = 15, m = 30;
    auto unpack = [&](std::span<const double> t, std::vector<double>& x, std::vector<double>& a, std::vector<double>& b) {
      x.assign(t.begin(), t.begin() + n), a.assign(t.begin() + n, t.begin() + n + m), b.assign(t.begin() + n + m, t.end());
    };
    auto value = [&](std::span<const double> t) {
      std::vector<double> x, a, b;
      unpack(t, x, a, b);
      return -log_posterior(c.y, x, a, b, o);
    };
    auto gradient = [&](std::span<const double> t) {
      std::vector<double> x, a, b;
      unpack(t, x, a, b);
      std::vector<double> g(n + 2 * m, 0.0);
      const double px = 1 / (o.prior_sd_x * o.prior_sd_x), pb = 1 / (o.prior_sd_item * o.prior_sd_item);
      for (std::size_t i = 0; i < n; i++) g[i] = px * x[i];
      for (std::size_t j = 0; j < m; j++) g[n + j] = pb * a[j], g[n + m + j] = pb * b[j];
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < m; j++) {
          if (c.y[i][j] != 0 && c.y[i][j] != 1) continue;
          const double q = c.y[i][j] == 1 ? 1 : -1, mm = a[j] + b[j] * x[i];
          const double d = q * detail::lambda_yes(q * mm);  // d/dm log Φ(q m)
          g[i] -= d * b[j], g[n + j] -= d, g[n + m + j] -= d * x[i];
        }
      return g;
    };
    std::vector<double> start;
    for (double v : f.x) start.push_back(v + 0.01 * rng.normal());
    for (double v : f.alpha) start.push_back(v + 0.01 * rng.normal());
    for (double v : f.beta) start.push_back(v + 0.01 * rng.normal());
    calx::optimisation::BfgsOptions bo;
    bo.gradient_tolerance = 1e-8;
    const auto res = calx::optimisation::bfgs(value, gradient, start, bo);
    double worst = 0;
    for (std::size_t i = 0; i < n; i++) worst = std::max(worst, std::abs(res.x[i] - f.x[i]));
    for (std::size_t j = 0; j < m; j++) worst = std::max({worst, std::abs(res.x[n + j] - f.alpha[j]), std::abs(res.x[n + m + j] - f.beta[j])});
    s.require("BFGS on the log posterior, from near EM's answer, comes to rest: its gradient vanishes", res.converged || res.gradient_norm < 1e-5, std::to_string(res.iterations) + " iterations, gradient " + std::to_string(res.gradient_norm));
    s.check("and reaches the same parameters as EM", worst, 1e-5);
    s.check("and the same log posterior", std::abs(-res.value - f.log_posterior.back()), 1e-9);

    std::vector<double> nx = f.x, nb = f.beta;
    for (double& v : nx) v = -v;
    for (double& v : nb) v = -v;
    s.check("reflecting x and β leaves the posterior unchanged", std::abs(log_posterior(c.y, nx, f.alpha, nb, o) - log_posterior(c.y, f.x, f.alpha, f.beta, o)), 1e-9);
    const auto g = standardised(f);
    double mu = 0, sd = 0, drift = 0;
    for (double v : g.x) mu += v / n;
    for (double v : g.x) sd += (v - mu) * (v - mu) / n;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < m; j++) drift = std::max(drift, std::abs((g.alpha[j] + g.beta[j] * g.x[i]) - (f.alpha[j] + f.beta[j] * f.x[i])));
    s.require("standardised: mean 0 and standard deviation 1, and every α + β x unchanged", std::abs(mu) < 1e-12 && std::abs(sd - 1) < 1e-12 && drift < 1e-12);
  }

  // ---- The far tail ---------------------------------------------------------------------------------------------------------------
  {
    const double below = detail::phi(-5.0) / (0.5 * std::erfc(5.0 / std::numbers::sqrt2)), above = 1 / detail::mills(5.0);
    s.check("φ/Φ at −5 by the error function and by Laplace's continued fraction agree", std::abs(below / above - 1), 1e-12);
    s.check("log Φ at −5 by both routes agrees", std::abs(std::log(0.5 * std::erfc(5.0 / std::numbers::sqrt2)) - (-12.5 - 0.5 * std::log(2 * std::numbers::pi) + std::log(detail::mills(5.0)))), 1e-12);
    s.require("log Φ(−40) is finite, about −804.6", std::isfinite(detail::log_Phi(-40)) && std::abs(detail::log_Phi(-40) + 804.608) < 0.01, std::to_string(detail::log_Phi(-40)));
    Votes y = {{1, 1, 0, 1}, {1, 1, 1, 0}, {1, 0, 1, 0}, {1, 0, 0, 1}, {1, 1, 0, 1}};  // the first item unanimous
    const auto f = ideal_points(y);
    bool finite = true;
    for (double v : f.x) finite = finite && std::isfinite(v);
    for (std::size_t j = 0; j < 4; j++) finite = finite && std::isfinite(f.alpha[j]) && std::isfinite(f.beta[j]);
    s.require("a unanimous item gives finite answers, its cut far on one side and its discrimination near none", finite && f.alpha[0] > 0.5 && std::abs(f.beta[0]) < std::abs(f.alpha[0]));
  }

  // ---- The bootstrap -------------------------------------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261119, 32, 0, 0);
    auto c = simulate(40, 480, 0.0, rng);
    Votes thin = c.y;
    for (std::size_t j = 0; j < 480; j++)
      if (j % 5) thin[1][j] = -1;  // member 1 casts a fifth of the votes
    Votes few = thin;
    for (auto& row : few) row.resize(120);  // the same chamber on its first 120 votes
    IdealPointOptions o;
    o.anchor = positive_member(c.x);
    calx::random::Stream draws(20261119, 33, 0, 0);
    auto full = ideal_points(c.y, o), all = ideal_points(thin, o), part = ideal_points(few, o);
    bootstrap(full, c.y, 30, draws, 0.9, o);
    bootstrap(all, thin, 30, draws, 0.9, o);
    bootstrap(part, few, 30, draws, 0.9, o);
    double wide_all = 0, wide_part = 0;
    for (std::size_t i = 2; i < 40; i++) wide_all += (all.upper[i] - all.lower[i]) / 38, wide_part += (part.upper[i] - part.lower[i]) / 38;
    s.require("in one chamber, four times the votes narrow the intervals", wide_all < 0.85 * wide_part, std::to_string(wide_all) + " against " + std::to_string(wide_part));
    const double w_thin = all.upper[1] - all.lower[1], w_full = full.upper[1] - full.lower[1];
    s.require("a member who casts a fifth of its votes has a wider interval than when it casts them all", w_thin > 1.4 * w_full,
              std::to_string(w_thin) + " against " + std::to_string(w_full));
    // the truth carried onto the fit's location and scale by least squares, then covered or not
    double mx = 0, mt = 0, sxt = 0, stt = 0;
    for (std::size_t i = 0; i < 40; i++) mx += all.x[i] / 40, mt += c.x[i] / 40;
    for (std::size_t i = 0; i < 40; i++) sxt += (all.x[i] - mx) * (c.x[i] - mt), stt += (c.x[i] - mt) * (c.x[i] - mt);
    std::size_t covered = 0;
    for (std::size_t i = 0; i < 40; i++) {
      const double t = mx + (sxt / stt) * (c.x[i] - mt);
      covered += all.lower[i] <= t && t <= all.upper[i];
    }
    s.require("the 90% intervals cover the truth, carried onto the fit's scale, for most members", covered >= 30, std::to_string(covered) + " of 40");
  }
  return s.done();
}
