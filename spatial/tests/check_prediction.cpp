// Checks of tyche/spatial/prediction.hpp: estimates where a quantity was not observed, and where none should be given.
//
// - An exact linear truth: the coefficients recovered, blocked cross-validation's skill one, the interval nil; a target among the
//   training units is in the area of applicability, one far outside it is not, and is withheld with that reason.
// - Noise of a known law: the conformal intervals cover about the level asked on fresh units, averaged over five samples.
// - A quantity that the covariates do not explain: no skill above a declared floor on blocks held out, so every estimate is
//   withheld, with the reason.
// - The logarithm of a positive quantity: an exponential truth recovered, its intervals carried back.
// - A declared limit on the interval: estimates wider than the use allows are withheld.
// - The index of a target that repeats a training unit is zero; refusals of a single block.

#include <tyche/spatial/prediction.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace tyche::spatial;

int main() {
  calx::verification::Suite s;
  calx::random::Stream rs(20261118, 28, 0, 0);

  // ---- an exact linear truth ----
  {
    const std::size_t n = 60;
    Rectangular X(n, 2);
    std::vector<double> y(n);
    std::vector<std::int64_t> b(n);
    for (std::size_t i = 0; i < n; i++) {
      X(i, 0) = rs.uniform(), X(i, 1) = rs.uniform();
      y[i] = 1 + 2 * X(i, 0) - X(i, 1);
      b[i] = static_cast<std::int64_t>(i % 5);
    }
    Rectangular Xt(2, 2);
    Xt(0, 0) = X(3, 0), Xt(0, 1) = X(3, 1);
    Xt(1, 0) = 100, Xt(1, 1) = 0.5;
    const Prediction p = predict(X, y, b, Xt);
    s.check("the coefficients of an exact truth: 1, 2, −1",
            std::abs(p.coefficients[0] - 1) + std::abs(p.coefficients[1] - 2) + std::abs(p.coefficients[2] + 1), 1e-10);
    s.check("blocked cross-validation's skill is one", p.cv_skill - 1, 1e-10);
    s.check("the interval is nil", p.halfwidth, 1e-10);
    s.check("a target that repeats a training unit has index zero", p.dissimilarity[0], 1e-12);
    s.require("... and is given", !p.withheld[0] && p.applicable[0]);
    s.require("a target far outside is not in the area of applicability, and is withheld with that reason",
              !p.applicable[1] && p.withheld[1] && p.reason[1].find("area of applicability") != std::string::npos, p.reason[1]);
  }

  // ---- conformal coverage ----
  {
    // Coverage conditional on one calibration set varies (by about ±0.013 for 500 units), so five samples are averaged.
    const std::size_t n = 500, m = 2000;
    const auto f = [](double a, double c, double d) { return 2 + a - 0.5 * c + 0.25 * d; };
    double cover = 0, ratio = 0;
    for (int rep = 0; rep < 5; rep++) {
      Rectangular X(n, 3), Xt(m, 3);
      std::vector<double> y(n), yt(m);
      std::vector<std::int64_t> b(n);
      for (std::size_t i = 0; i < n; i++) {
        X(i, 0) = rs.normal(), X(i, 1) = rs.normal(), X(i, 2) = rs.normal();
        y[i] = f(X(i, 0), X(i, 1), X(i, 2)) + 0.3 * rs.normal();
        b[i] = static_cast<std::int64_t>(i % 10);
      }
      for (std::size_t i = 0; i < m; i++) {
        Xt(i, 0) = rs.normal(), Xt(i, 1) = rs.normal(), Xt(i, 2) = rs.normal();
        yt[i] = f(Xt(i, 0), Xt(i, 1), Xt(i, 2)) + 0.3 * rs.normal();
      }
      PredictOptions o;
      o.level = 0.9;
      const Prediction p = predict(X, y, b, Xt, o);
      std::size_t covered = 0;
      for (std::size_t i = 0; i < m; i++) covered += (yt[i] >= p.lower[i] && yt[i] <= p.upper[i]);
      cover += static_cast<double>(covered) / static_cast<double>(m) / 5;
      ratio += p.halfwidth / p.cv_rmse / 5;
    }
    s.require("conformal intervals cover about 90% of fresh units, over five samples", cover > 0.88 && cover < 0.93, std::to_string(cover));
    s.require("... with a half-width near 1.645 times the error held out", ratio > 1.5 && ratio < 1.8, std::to_string(ratio));
  }

  // ---- no skill, nothing given ----
  {
    const std::size_t n = 200;
    Rectangular X(n, 2), Xt(3, 2);
    std::vector<double> y(n);
    std::vector<std::int64_t> b(n);
    for (std::size_t i = 0; i < n; i++) X(i, 0) = rs.normal(), X(i, 1) = rs.normal(), y[i] = rs.normal(), b[i] = static_cast<std::int64_t>(i % 4);
    for (std::size_t i = 0; i < 3; i++) Xt(i, 0) = 0, Xt(i, 1) = 0;
    PredictOptions o;
    o.min_skill = 0.1;  // a declared floor: the estimator must remove a tenth of the error of the mean on blocks held out
    const Prediction p = predict(X, y, b, Xt, o);
    bool all = true;
    for (char w : p.withheld) all = all && w;
    s.require("covariates that do not explain the quantity: no skill on blocks held out", p.cv_skill <= 0.1, std::to_string(p.cv_skill));
    s.require("... so every estimate is withheld, with the reason", all && p.reason[0].find("does not beat the mean") != std::string::npos, p.reason[0]);
  }

  // ---- a logarithm ----
  {
    const std::size_t n = 50;
    Rectangular X(n, 1), Xt(1, 1);
    std::vector<double> y(n);
    std::vector<std::int64_t> b(n);
    for (std::size_t i = 0; i < n; i++) X(i, 0) = rs.uniform(0, 2), y[i] = std::exp(1 + X(i, 0)), b[i] = static_cast<std::int64_t>(i % 5);
    Xt(0, 0) = 1;
    PredictOptions o;
    o.log = true;
    const Prediction p = predict(X, y, b, Xt, o);
    s.check("an exponential truth through its logarithm: e² at x = 1", p.estimate[0] - std::exp(2.0), 1e-9);
  }

  // ---- a declared limit on the interval ----
  {
    const std::size_t n = 100;
    Rectangular X(n, 1), Xt(2, 1);
    std::vector<double> y(n);
    std::vector<std::int64_t> b(n);
    for (std::size_t i = 0; i < n; i++) X(i, 0) = rs.uniform(0, 10), y[i] = X(i, 0) + rs.normal(), b[i] = static_cast<std::int64_t>(i % 5);
    Xt(0, 0) = 9, Xt(1, 0) = 1.5;
    PredictOptions o;
    o.max_relative_halfwidth = 0.5;
    const Prediction p = predict(X, y, b, Xt, o);
    s.require("an estimate of 9 ± 1.6 is given under a limit of half its value", !p.withheld[0], p.reason[0]);
    s.require("an estimate of 1.5 ± 1.6 is withheld, as wider than the use allows",
              p.withheld[1] && p.reason[1].find("wider than the use allows") != std::string::npos, p.reason[1]);
  }

  // ---- refusals ----
  {
    Rectangular X(10, 1), Xt(1, 1);
    std::vector<double> y(10, 1.0);
    std::vector<std::int64_t> b(10, 0);
    bool threw = false;
    try {
      predict(X, y, b, Xt);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    s.require("a single block is refused: cross-validation needs two", threw);
  }

  return s.done();
}
