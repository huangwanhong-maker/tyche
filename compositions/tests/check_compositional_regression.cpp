// Checks of tyche/compositions/regression: a composition as predictor, reallocation, a composition as response.
//
// On seeded compositions of a day (four parts summing to 1440 minutes, logistic-normal), the predictor model
// recovers known coefficients exactly without noise and within four standard errors with it; its fitted
// values, clr coefficients and R² do not depend on the basis; the clr coefficients sum to zero and equal the
// closed form of a two-part model; the pivot effect of a part equals the coefficient of the refitted model
// with that part first, with the same standard error. A reallocation's predicted change equals the
// difference of the fitted model at the two compositions and Σ b_i ln(x¹_i/x⁰_i), its standard error
// √(dᵀVd); the reverse move gives exactly the opposite change; one for remaining keeps the total and the
// ratios of the other parts. The response model recovers known coefficients, its effects and predictions do
// not depend on the basis, and its predictions are ilr⁻¹ of the fitted coordinates.

#include <tyche/compositions/regression.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

using namespace tyche::compositions;

namespace {

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

double gap(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return INFINITY;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

struct Day {
  Rectangular X;                // n × 4 minutes
  Rectangular Z;                // n × 3 pivot coordinates
  Rectangular W;                // n × 1 covariate
};

// Days of four parts (sleep, sedentary, light, vigorous) whose pivot coordinates are normal.
Day days(calx::random::Stream& rng, std::size_t n) {
  const std::vector<double> mu{1.2, 0.6, 0.9};
  const double L[3][3] = {{0.25, 0, 0}, {0.1, 0.3, 0}, {-0.05, 0.1, 0.35}};
  const auto b = pivot_basis(4);
  Day d{Rectangular(n, 4), Rectangular(n, 3), Rectangular(n, 1)};
  for (std::size_t i = 0; i < n; i++) {
    const double e[3] = {rng.normal(), rng.normal(), rng.normal()};
    std::vector<double> z(3);
    for (int a = 0; a < 3; a++) {
      z[a] = mu[a];
      for (int c = 0; c < 3; c++) z[a] += L[a][c] * e[c];
      d.Z(i, static_cast<std::size_t>(a)) = z[a];
    }
    const auto x = ilr_inverse(z, b, 1440);
    for (std::size_t j = 0; j < 4; j++) d.X(i, j) = x[j];
    d.W(i, 0) = rng.normal();
  }
  return d;
}

// A random sequential binary partition of four parts.
SignTable other_partition() { return {{1, -1, 1, -1}, {1, 0, -1, 0}, {0, 1, 0, -1}}; }

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261113, 0x636f6d70, 3, 0);
  const std::vector<double> gamma{-1.5, 0.8, 2.0};
  const double beta0 = 2, delta = 0.5;

  // ---- noise-free: exact recovery ----
  {
    const auto d = days(rng, 60);
    std::vector<double> y(60);
    for (std::size_t i = 0; i < 60; i++) y[i] = beta0 + gamma[0] * d.Z(i, 0) + gamma[1] * d.Z(i, 1) + gamma[2] * d.Z(i, 2) + delta * d.W(i, 0);
    const auto m = fit_predictor(d.X, y, d.W);
    s.check("without noise: the pivot coefficients γ recovered", gap({m.fit.beta[1], m.fit.beta[2], m.fit.beta[3]}, gamma), 1e-10);
    s.check("without noise: the intercept and the covariate's coefficient recovered", std::fabs(m.fit.beta[0] - beta0) + std::fabs(m.fit.beta[4] - delta), 1e-10);
    double sb = 0;
    for (double v : m.clr_coefficients) sb += v;
    s.check("the clr coefficients sum to zero", sb, 1e-13);
    s.check("the clr coefficients are Ψᵀγ", gap(m.clr_coefficients, calx::linalg::multiply_transpose(pivot_basis(4).contrast, gamma)), 1e-10);
    s.check("the model's total is the mean of the rows' sums (1440)", m.total - 1440, 1e-9);
  }

  // ---- two parts: the closed form ----
  {
    Rectangular X(20, 2);
    std::vector<double> y(20);
    for (std::size_t i = 0; i < 20; i++) {
      X(i, 0) = 1 + static_cast<double>(i), X(i, 1) = 30 - static_cast<double>(i);
      y[i] = 0.7 + 1.3 * std::log(X(i, 0) / X(i, 1));
    }
    const auto m = fit_predictor(X, y);
    s.check("two parts, y = a + b ln(x₁/x₂): the clr coefficients are (b, −b)", gap(m.clr_coefficients, {1.3, -1.3}), 1e-12);
    s.check("two parts: the ilr coefficient is b√2", m.fit.beta[1] - 1.3 * std::sqrt(2.0), 1e-12);
  }

  // ---- with noise: recovery, invariance, pivot effects ----
  const auto d = days(rng, 2000);
  std::vector<double> y(2000);
  for (std::size_t i = 0; i < 2000; i++) y[i] = beta0 + gamma[0] * d.Z(i, 0) + gamma[1] * d.Z(i, 1) + gamma[2] * d.Z(i, 2) + delta * d.W(i, 0) + 0.3 * rng.normal();
  const auto m = fit_predictor(d.X, y, d.W);
  {
    double worst = 0;
    for (std::size_t k = 0; k < 3; k++) worst = std::max(worst, std::fabs(m.fit.beta[1 + k] - gamma[k]) / m.fit.se[1 + k]);
    s.check("with noise (n = 2000, σ = 0.3): every γ̂_k within 4 standard errors of γ_k", worst, 4);
    PredictorOptions o;
    o.partition = other_partition();
    const auto m2 = fit_predictor(d.X, y, d.W, o);
    s.check("another basis: the same fitted values", gap(m.fit.fitted, m2.fit.fitted), 1e-10);
    s.check("another basis: the same clr coefficients", gap(m.clr_coefficients, m2.clr_coefficients), 1e-10);
    s.check("another basis: the same R²", m.fit.r_squared - m2.fit.r_squared, 1e-12);
    const auto c = d.X.row(7);
    Rectangular one(1, 4), w1(1, 1);
    for (std::size_t j = 0; j < 4; j++) one(0, j) = c[j] * 0.9 + 10;
    w1(0, 0) = 1.5;
    s.check("another basis: the same prediction and standard error at a new day",
            std::fabs(predict(m, one, w1).mean[0] - predict(m2, one, w1).mean[0]) + std::fabs(predict(m, one, w1).se[0] - predict(m2, one, w1).se[0]), 1e-10);
    double pe = 0;
    for (std::size_t l = 0; l < 4; l++) {
      std::vector<std::size_t> order{l};
      for (std::size_t j = 0; j < 4; j++)
        if (j != l) order.push_back(j);
      PredictorOptions ol;
      ol.partition = pivot_partition(4, order);
      const auto ml = fit_predictor(d.X, y, d.W, ol);
      const auto e = pivot_effect(m, l);
      pe = std::max({pe, std::fabs(e.estimate - ml.fit.beta[1]), std::fabs(e.se - ml.fit.se[1]), std::fabs(e.p - ml.fit.p[1])});
    }
    s.check("the pivot effect of each part equals the refitted model's first coefficient, with its standard error and p-value", pe, 1e-10);
  }

  // ---- reallocation ----
  {
    const auto& x0 = m.centre;
    const auto r = one_for_one(m, x0, 3, 1, 30);  // 30 minutes from sedentary to vigorous
    Rectangular A(1, 4), B(1, 4), w(1, 1);
    for (std::size_t j = 0; j < 4; j++) A(0, j) = r.before[j], B(0, j) = r.after[j];
    w(0, 0) = 0.2;
    const double diff = predict(m, B, w).mean[0] - predict(m, A, w).mean[0];
    s.check("one for one: the change equals the fitted model at x¹ less the fitted model at x⁰", r.change.estimate - diff, 1e-12);
    double lin = 0;
    for (std::size_t j = 0; j < 4; j++) lin += m.clr_coefficients[j] * std::log(r.after[j] / r.before[j]);
    s.check("one for one: the change equals Σ b_i ln(x¹_i/x⁰_i)", r.change.estimate - lin, 1e-12);
    const auto DA = design(m, A, w), DB = design(m, B, w);
    double v = 0;
    for (std::size_t a = 0; a < DA.cols; a++)
      for (std::size_t b = 0; b < DA.cols; b++) v += (DB(0, a) - DA(0, a)) * m.fit.covariance(a, b) * (DB(0, b) - DA(0, b));
    s.check("one for one: the standard error is √(dᵀVd) on the design's rows", r.change.se - std::sqrt(v), 1e-14);
    s.require("one for one: the move as asked, the other parts held",
              std::fabs(r.after[3] - r.before[3] - 30) < 1e-12 && std::fabs(r.before[1] - r.after[1] - 30) < 1e-12 && r.after[0] == r.before[0] && r.after[2] == r.before[2]);
    const auto back = one_for_one(m, r.after, 1, 3, 30);
    s.check("the reverse move gives exactly the opposite change", r.change.estimate + back.change.estimate, 1e-13);
    s.check("and the same standard error", r.change.se - back.change.se, 1e-15);
    const auto opposite = one_for_one(m, x0, 1, 3, 30);
    s.require("moving 30 minutes the other way from x⁰ is not exactly opposite (the model is nonlinear in Δ)",
              std::fabs(r.change.estimate + opposite.change.estimate) > 1e-6);
    double expected = 0;  // the true change, from γ
    {
      const auto z0 = ilr(r.before, pivot_basis(4)), z1 = ilr(r.after, pivot_basis(4));
      for (std::size_t k = 0; k < 3; k++) expected += gamma[k] * (z1[k] - z0[k]);
    }
    s.check("the true change lies within 4 standard errors of the estimate", (r.change.estimate - expected) / r.change.se, 4);
    s.require("the interval holds the estimate, at the fit's level", r.change.lower < r.change.estimate && r.change.estimate < r.change.upper);
    const auto rem = one_for_remaining(m, x0, 3, 30);
    double t0 = 0, t1 = 0, ratio = 0;
    for (std::size_t j = 0; j < 4; j++) t0 += rem.before[j], t1 += rem.after[j];
    for (std::size_t a : {0, 1, 2})
      for (std::size_t b : {0, 1, 2}) ratio = std::max(ratio, std::fabs(rem.after[a] / rem.after[b] - rem.before[a] / rem.before[b]));
    s.check("one for remaining: the total kept", t1 - t0, 1e-11);
    s.check("one for remaining: the ratios of the other parts kept", ratio, 1e-14);
    Rectangular C(1, 4);
    for (std::size_t j = 0; j < 4; j++) C(0, j) = rem.after[j];
    s.check("one for remaining: the change equals the fitted model's difference", rem.change.estimate - (predict(m, C, w).mean[0] - predict(m, A, w).mean[0]), 1e-12);
    const auto rem_back = reallocation(m, rem.after, rem.before);
    s.check("one for remaining: the reverse change is opposite", rem.change.estimate + rem_back.change.estimate, 1e-13);
    PredictorOptions robust;
    robust.ols.covariance = tyche::regression::Covariance::hc3;
    const auto mr = fit_predictor(d.X, y, d.W, robust);
    const auto rr = one_for_one(mr, x0, 3, 1, 30);
    s.require("a robust covariance (HC3) changes the standard error, not the estimate",
              std::fabs(rr.change.estimate - r.change.estimate) < 1e-12 && std::fabs(rr.change.se - r.change.se) > 1e-9);
    s.require("a move that empties a part is refused", throws([&] { one_for_one(m, x0, 3, 1, x0[1] + 1); }));
    s.require("a move between one part and itself is refused", throws([&] { one_for_one(m, x0, 2, 2, 10); }));
    s.require("one for remaining that empties the others is refused", throws([&] { one_for_remaining(m, x0, 0, 1440); }));
    s.require("covariates of the wrong shape are refused", throws([&] { fit_predictor(d.X, y, Rectangular(10, 1)); }));
  }

  // ---- a composition as response ----
  {
    // ilr(Y_i) = B₀ + B₁ w_i (+ noise), in pivot coordinates of 4 parts.
    const std::vector<double> b0{0.9, 0.4, 1.1}, b1{-0.3, 0.25, 0.6};
    const auto pb = pivot_basis(4);
    auto make = [&](std::size_t n, double sigma, Rectangular& Y, Rectangular& W) {
      Y = Rectangular(n, 4);
      W = Rectangular(n, 1);
      for (std::size_t i = 0; i < n; i++) {
        W(i, 0) = rng.normal();
        std::vector<double> z(3);
        for (std::size_t k = 0; k < 3; k++) z[k] = b0[k] + b1[k] * W(i, 0) + sigma * rng.normal();
        const auto x = ilr_inverse(z, pb, 1440);
        for (std::size_t j = 0; j < 4; j++) Y(i, j) = x[j];
      }
    };
    Rectangular Y, W;
    make(40, 0, Y, W);
    const auto r = fit_response(Y, W);
    s.check("response without noise: the intercept's coordinates recovered", gap(r.coefficients.row(0), b0), 1e-10);
    s.check("response without noise: the covariate's coordinates recovered", gap(r.coefficients.row(1), b1), 1e-10);
    s.check("the covariate's effect is the composition ilr⁻¹(B₁)", gap(r.effects.row(1), ilr_inverse(b1, pb)), 1e-12);
    Rectangular w(1, 1);
    w(0, 0) = 0.8;
    std::vector<double> zf(3);
    for (std::size_t k = 0; k < 3; k++) zf[k] = b0[k] + 0.8 * b1[k];
    s.check("prediction: ilr⁻¹ of the fitted coordinates, closed to the sample's total (1440)", gap(predict(r, w).row(0), ilr_inverse(zf, pb, 1440)), 1e-9);
    Rectangular Yn, Wn;
    make(1500, 0.2, Yn, Wn);
    const auto rn = fit_response(Yn, Wn);
    double worst = 0;
    for (std::size_t k = 0; k < 3; k++) worst = std::max({worst, std::fabs(rn.fits[k].beta[0] - b0[k]) / rn.fits[k].se[0], std::fabs(rn.fits[k].beta[1] - b1[k]) / rn.fits[k].se[1]});
    s.check("response with noise (n = 1500, σ = 0.2): every coefficient within 4 standard errors", worst, 4);
    double rc = 0;
    for (std::size_t k = 0; k < 3; k++) rc = std::max(rc, std::fabs(rn.residual_covariance(k, k) - 0.04) / 0.04);
    s.check("response with noise: the residual variances within 10% of σ² = 0.04", rc, 0.1);
    ResponseOptions o;
    o.partition = other_partition();
    const auto ro = fit_response(Yn, Wn, o);
    s.check("response in another basis: the same effects (compositions)", std::max(gap(ro.effects.row(0), rn.effects.row(0)), gap(ro.effects.row(1), rn.effects.row(1))), 1e-12);
    s.check("response in another basis: the same predictions", gap(predict(ro, w).row(0), predict(rn, w).row(0)), 1e-9);
    const auto only = fit_response(Yn, Rectangular());
    s.check("response on an intercept alone: the prediction is the sample's centre", distance(predict(only, Rectangular()).row(0), centre(Yn)), 1e-12);
  }

  return s.done();
}
