// Checks of the panel estimators (panel.hpp, absorb.hpp), on simulated panels.
//
// The within estimator against the least-squares dummy-variable regression (tyche's OLS on X and
// the indicators): one-way, the same coefficients, residuals and classical standard errors (with
// n − k − N degrees of freedom), and cluster-robust standard errors smaller by exactly
// √((n − k − N)/(n − k)), the effects being nested in the clusters (Stata's xtreg against areg);
// two-way on an unbalanced panel, the same coefficients, classical standard errors and fitted
// values, with N + T − 1 parameters absorbed, T − 1 of them not nested in unit clusters, and the
// effects reproducing the fitted values; a time-invariant regressor absorbed; the true
// coefficients recovered. Random effects: in a balanced panel the estimate is the matrix-weighted
// average of the within and between estimates (Maddala 1971), exactly; on a large panel Swamy and
// Arora's variance components are near the truth. The Hausman test accepts when the effects are
// independent of the regressors and rejects when they are not. (Differences in differences moved
// with their checks to peira's quasi; tyche's decision 0014.)

#include <tyche/panel/panel.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace ne = tyche::panel;
namespace tr = tyche::regression;
using calx::linalg::Rectangular;
using calx::verification::relative;

namespace {

calx::random::Stream z(20261025, 0x70616e6cu, 0, 0);  // "panl"

struct Data {
  Rectangular X;  // x1, x2, w (time-invariant)
  std::vector<double> y;
  std::vector<std::int64_t> unit, period;
};

// N units over T periods; each cell kept with probability `keep`; the effects correlated with x1 by `corr`;
// a common trend in time, which one-way models leave in their error, unless `trend` is 0.
Data simulate(std::size_t N, std::size_t T, double keep, double corr, double sigma_u = 1, double sigma_e = 0.5, bool hetero = true,
              double trend = 0.3) {
  std::vector<double> alpha(N), w(N), s(N);
  for (std::size_t i = 0; i < N; i++) {
    alpha[i] = sigma_u * z.normal();
    w[i] = z.normal();
    s[i] = hetero ? sigma_e * (0.5 + z.uniform()) : sigma_e;
  }
  std::vector<std::vector<double>> rows;
  Data d;
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t t = 0; t < T; t++) {
      if (z.uniform() > keep) continue;
      const double x1 = corr * alpha[i] + z.normal(), x2 = z.normal();
      rows.push_back({x1, x2, w[i]});
      d.y.push_back(alpha[i] + trend * static_cast<double>(t) + 1.5 * x1 - 0.7 * x2 + s[i] * z.normal());
      d.unit.push_back(static_cast<std::int64_t>(100 + i));
      d.period.push_back(static_cast<std::int64_t>(2000 + t));
    }
  d.X = Rectangular::from_rows(rows);
  return d;
}

Rectangular columns(const Rectangular& X, std::vector<std::size_t> which) {
  Rectangular out(X.rows, which.size());
  for (std::size_t i = 0; i < X.rows; i++)
    for (std::size_t j = 0; j < which.size(); j++) out(i, j) = X(i, which[j]);
  return out;
}

// [X, indicators of the units, indicators of the periods but the first].
Rectangular dummies(const Rectangular& X, const Data& d, bool periods) {
  const auto fu = ne::factor(d.unit), fp = ne::factor(d.period);
  const std::size_t extra = fu.levels + (periods ? fp.levels - 1 : 0);
  Rectangular D(X.rows, X.cols + extra);
  for (std::size_t i = 0; i < X.rows; i++) {
    for (std::size_t j = 0; j < X.cols; j++) D(i, j) = X(i, j);
    D(i, X.cols + fu.level[i]) = 1;
    if (periods && fp.level[i] > 0) D(i, X.cols + fu.levels + fp.level[i] - 1) = 1;
  }
  return D;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // One-way, balanced: within against LSDV.
  {
    const auto d = simulate(80, 6, 1.0, 0.7);
    const auto X = columns(d.X, {0, 1});
    ne::PanelOptions classical;
    classical.covariance = ne::PanelCovariance::classical;
    const auto fe = ne::within(X, d.y, d.unit, {}, classical);
    const auto lsdv = tr::ols(dummies(X, d, false), d.y);
    double db = 0, dse = 0, de = 0;
    for (std::size_t j = 0; j < 2; j++) {
      db = std::max(db, std::fabs(fe.fit.beta[j] - lsdv.beta[j]));
      dse = std::max(dse, relative(fe.fit.se[j], lsdv.se[j]));
    }
    for (std::size_t i = 0; i < d.y.size(); i++) de = std::max(de, std::fabs(fe.fit.residuals[i] - lsdv.residuals[i]));
    s.check("one-way within (80 units × 6 periods) against the dummy-variable regression: coefficients", db, 1e-10);
    s.check("... classical standard errors, with n − k − N degrees of freedom (relative)", dse, 1e-9);
    s.check("... residuals", de, 1e-10);
    s.require("... N parameters absorbed and t on n − k − N degrees of freedom", fe.absorbed == 80 && fe.fit.df_inference == 480.0 - 2 - 80);
    const auto fec = ne::within(X, d.y, d.unit);
    tr::OlsOptions oc;
    oc.covariance = tr::Covariance::cluster;
    oc.clusters = d.unit;
    const auto lsdvc = tr::ols(dummies(X, d, false), d.y, oc);
    const double n = 480, k = 2, N = 80;
    double ratio = 0;
    for (std::size_t j = 0; j < 2; j++) ratio = std::max(ratio, relative(fec.fit.se[j] / lsdvc.se[j], std::sqrt((n - k - N) / (n - k))));
    s.check("... clustered by unit: the effects are nested, so the errors are those of LSDV times √((n − k − N)/(n − k))", ratio, 1e-9);
    s.require("... with t on G − 1 = 79 degrees of freedom, and no absorbed parameter counted", fec.fit.df_inference == 79 && fec.absorbed_not_nested == 0);
    // The unit effects: α̂_i = ȳ_i − x̄_i β̂, against the LSDV coefficients of the dummies.
    double da = 0;
    for (std::size_t l = 0; l < fe.unit_effects.size(); l++) da = std::max(da, std::fabs(fe.unit_effects[l] - lsdv.beta[2 + l]));
    s.check("... the recovered unit effects are the dummies' coefficients", da, 1e-9);
  }
  // Two-way, unbalanced.
  {
    const auto d = simulate(70, 8, 0.8, 0.7);
    const auto X = columns(d.X, {0, 1});
    ne::PanelOptions o;
    o.effects = ne::PanelEffects::two_way;
    o.covariance = ne::PanelCovariance::classical;
    const auto fe = ne::within(X, d.y, d.unit, d.period, o);
    const auto lsdv = tr::ols(dummies(X, d, true), d.y);
    double db = 0, dse = 0, df = 0, deff = 0;
    for (std::size_t j = 0; j < 2; j++) {
      db = std::max(db, std::fabs(fe.fit.beta[j] - lsdv.beta[j]));
      dse = std::max(dse, relative(fe.fit.se[j], lsdv.se[j]));
    }
    const auto fu = ne::factor(d.unit), fp = ne::factor(d.period);
    for (std::size_t i = 0; i < d.y.size(); i++) {
      df = std::max(df, std::fabs(fe.fit.fitted[i] - lsdv.fitted[i]));
      const double rebuilt = X(i, 0) * fe.fit.beta[0] + X(i, 1) * fe.fit.beta[1] + fe.unit_effects[fu.level[i]] + fe.period_effects[fp.level[i]];
      deff = std::max(deff, std::fabs(rebuilt - lsdv.fitted[i]));
    }
    s.check("two-way within on an unbalanced panel (70 × 8, a fifth missing): coefficients against LSDV", db, 1e-9);
    s.check("... classical standard errors (relative)", dse, 1e-8);
    s.check("... fitted values", df, 1e-8);
    s.check("... and the recovered effects rebuild them", deff, 1e-8);
    s.require("... N + T − 1 parameters absorbed", fe.absorbed == 70 + 8 - 1, std::to_string(fe.absorbed) + " in " + std::to_string(fe.sweeps) + " sweeps");
    o.covariance = ne::PanelCovariance::cluster;
    const auto fc = ne::within(X, d.y, d.unit, d.period, o);
    s.require("... clustered by unit, the T − 1 period effects not nested in the clusters are counted", fc.absorbed_not_nested == 7);
    s.require("... and the true coefficients 1.5 and −0.7 lie within four standard errors",
              std::fabs(fc.fit.beta[0] - 1.5) < 4 * fc.fit.se[0] && std::fabs(fc.fit.beta[1] + 0.7) < 4 * fc.fit.se[1]);
    const auto with_w = ne::within(d.X, d.y, d.unit, d.period, o);
    s.require("a time-invariant regressor is absorbed (aliased, β = NaN) and changes nothing else",
              with_w.fit.aliased[2] && std::isnan(with_w.fit.beta[2]) && std::fabs(with_w.fit.beta[0] - fc.fit.beta[0]) < 1e-9);
  }
  // Random effects in a balanced panel: Maddala's weighted average, for one regressor.
  {
    const auto d = simulate(60, 5, 1.0, 0.0);
    const auto x = columns(d.X, {0});
    const auto re = ne::random_effects(x, d.y, d.unit);
    ne::PanelOptions classical;
    classical.covariance = ne::PanelCovariance::classical;
    const auto fe = ne::within(x, d.y, d.unit, {}, classical);
    const auto be = ne::between(x, d.y, d.unit);
    // S_W = Σ (x − x̄_i)², S_B = T Σ_i (x̄_i − x̄)².
    const auto f = ne::factor(d.unit);
    std::vector<double> xm(f.levels, 0.0);
    double xbar = 0;
    for (std::size_t i = 0; i < x.rows; i++) xm[f.level[i]] += x(i, 0) / 5, xbar += x(i, 0) / static_cast<double>(x.rows);
    double SW = 0, SB = 0;
    for (std::size_t i = 0; i < x.rows; i++) SW += (x(i, 0) - xm[f.level[i]]) * (x(i, 0) - xm[f.level[i]]);
    for (double v : xm) SB += 5 * (v - xbar) * (v - xbar);
    const double phi2 = (1 - re.theta[0]) * (1 - re.theta[0]);
    const double w = SW / (SW + phi2 * SB);
    s.check("random effects, balanced: β̂_RE = w β̂_W + (1 − w) β̂_B with w = S_W/(S_W + φ² S_B) (Maddala 1971)",
            re.fit.beta[1] - (w * fe.fit.beta[0] + (1 - w) * be.beta[1]), 1e-10);
  }
  // Variance components and the Hausman test on large panels, simulated as the one-way model
  // assumes: no common trend, which would be left in the error and inflate σ̂_ε.
  {
    const auto d = simulate(500, 5, 1.0, 0.0, 1.0, 0.5, false, 0.0);
    const auto X = columns(d.X, {0, 1});
    const auto re = ne::random_effects(X, d.y, d.unit);
    s.check("Swamy and Arora on 500 × 5: σ̂_u near 1 (relative)", relative(re.sigma_u, 1.0), 0.1);
    s.check("... σ̂_ε near 0.5 (relative)", relative(re.sigma_e, 0.5), 0.05);
    ne::PanelOptions classical;
    classical.covariance = ne::PanelCovariance::classical;
    const auto fe = ne::within(X, d.y, d.unit, {}, classical);
    const auto h0 = ne::hausman(fe, re);
    s.require("Hausman: effects independent of the regressors, not rejected (p > 0.01)", h0.p_value > 0.01,
              "H = " + calx::verification::exponential(h0.statistic) + ", p = " + calx::verification::exponential(h0.p_value));
    const auto d1 = simulate(500, 5, 1.0, 0.5, 1.0, 0.5, false, 0.0);
    const auto X1 = columns(d1.X, {0, 1});
    const auto h1 = ne::hausman(ne::within(X1, d1.y, d1.unit, {}, classical), ne::random_effects(X1, d1.y, d1.unit));
    s.require("... effects correlated with x₁, rejected (p < 10⁻⁶), V_FE − V_RE positive definite", h1.p_value < 1e-6 && h1.df == 2 && h1.positive_definite,
              "H = " + calx::verification::exponential(h1.statistic));
    s.require("... and positive definite under independence too", h0.positive_definite);
  }
  return s.done();
}
