// Checks of tyche/time_series's unit-root and cointegration tests.
//
// - MacKinnon (2010): the asymptotic critical values are β_∞ of his Tables 2–4 (τ_c, N = 1:
//   −3.43035, −2.86154, −2.56677; τ_ct, N = 1: −3.95877, −3.41049, −3.12705), and his worked
//   example, the 5% critical value of τ_ct with N = 5 at T = 100, −4.89111.
// - MacKinnon (1994) against (2010): the p-value at each asymptotic 1, 5 and 10% critical value of
//   τ_nc (N = 1), τ_c, τ_ct and τ_ctt (N = 1, …, 6) is 0.01, 0.05 and 0.10 to within 0.003.
// - The ADF statistic is the t statistic of ρ in the test regression, rebuilt here with ols().
// - Lütkepohl's data (log income, 1960Q1–1982Q4), against statsmodels 0.15.0's adfuller, kpss and
//   coint (computed 2026-10-24) as a cross-check: statistics, p-values, chosen lags, observations
//   and critical values; the Engle–Granger 1% value against MacKinnon's table rather than
//   statsmodels', whose β₂ for N = 2 is mistyped (−33.527 for −22.527).
// - KPSS's critical values are those of Kwiatkowski et al. (1992), Table 1; its statistic is
//   n⁻² Σ S_t² / s²(l), rebuilt here.
// - Seeded simulations: the ADF test's size on random walks and its power on a stationary AR(1);
//   KPSS's size on white noise; Engle–Granger on a cointegrated pair and on independent walks.

#include <tyche/regression/linear.hpp>
#include <tyche/time_series/unit_root.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tyche::time_series;
using calx::verification::relative;

namespace {

std::vector<double> e1_log(int column) {
  std::ifstream in(std::string(TYCHE_FIXTURES) + "/lutkepohl-e1.dat");
  std::vector<double> out;
  for (std::string l; std::getline(in, l);) {
    std::istringstream ss(l);
    double a, b, c;
    if (ss >> a >> b >> c) out.push_back(std::log(column == 0 ? a : column == 1 ? b : c));
  }
  return out;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- MacKinnon's tables ----
  {
    const auto c = mackinnon_critical(Deterministic::constant);
    const auto ct = mackinnon_critical(Deterministic::trend);
    s.check("mackinnon 2010: τ_c asymptotic 1, 5, 10% = −3.43035, −2.86154, −2.56677", std::fabs(c[0] + 3.43035) + std::fabs(c[1] + 2.86154) + std::fabs(c[2] + 2.56677), 1e-15);
    s.check("mackinnon 2010: τ_ct asymptotic = −3.95877, −3.41049, −3.12705", std::fabs(ct[0] + 3.95877) + std::fabs(ct[1] + 3.41049) + std::fabs(ct[2] + 3.12705), 1e-15);
    s.check("mackinnon 2010: his example, τ_ct, N = 5, 5%, T = 100: −4.89111", mackinnon_critical(Deterministic::trend, 5, 100)[1] + 4.89111, 5e-6);
    double worst = 0;
    const double levels[3] = {0.01, 0.05, 0.10};
    for (Deterministic d : {Deterministic::none, Deterministic::constant, Deterministic::trend, Deterministic::quadratic_trend}) {
      const int nmax = d == Deterministic::none ? 1 : 6;
      for (int N = 1; N <= nmax; N++) {
        const auto cv = mackinnon_critical(d, N);
        for (int l = 0; l < 3; l++) worst = std::max(worst, std::fabs(mackinnon_p(cv[l], d, N) - levels[l]));
      }
    }
    s.check("mackinnon 1994 against 2010: the p-values at the asymptotic critical values are 1, 5, 10% (worst deviation)", worst, 0.003);
    s.require("mackinnon 1994: p is 0 far left and 1 far right", mackinnon_p(-30, Deterministic::constant) == 0 && mackinnon_p(5, Deterministic::constant) == 1);
  }

  // ---- the ADF statistic is a t statistic ----
  const auto inc = e1_log(1), cons = e1_log(2), inv = e1_log(0);
  {
    AdfOptions o;
    o.deterministic = Deterministic::trend;
    o.selection = LagSelection::fixed;
    o.lags = 2;
    const AdfResult r = adf(inc, o);
    const std::size_t n = inc.size();
    tyche::regression::Rectangular X(n - 3, 5);
    std::vector<double> dy(n - 3);
    for (std::size_t t = 3; t < n; t++) {
      const std::size_t i = t - 3;
      dy[i] = inc[t] - inc[t - 1];
      X(i, 0) = 1;
      X(i, 1) = static_cast<double>(i + 1);
      X(i, 2) = inc[t - 1];
      X(i, 3) = inc[t - 1] - inc[t - 2];
      X(i, 4) = inc[t - 2] - inc[t - 3];
    }
    const auto f = tyche::regression::ols(X, dy);
    s.check("adf: τ is the t statistic of y_{t−1} in Δy on (1, t, y_{t−1}, Δy_{t−1}, Δy_{t−2})", relative(r.statistic, f.t[2]), 1e-12);
    s.require("adf: on n − k − 1 observations", r.n_obs == n - 3);
  }

  // ---- against statsmodels on Lütkepohl's data ----
  {
    AdfOptions o;  // constant, AIC, Schwert's maximum
    AdfResult r = adf(inc, o);
    s.check("adf (c, AIC): log income, τ = −2.29073824459 (statsmodels)", relative(r.statistic, -2.290738244591552), 1e-10);
    s.check("adf (c, AIC): its p-value 0.174972473641", relative(r.p_value, 0.17497247364131907), 1e-10);
    s.require("adf (c, AIC): 0 lags chosen, 91 observations", r.lags == 0 && r.n_obs == 91);
    s.check("adf (c, AIC): the critical values at T = 91", std::fabs(r.critical[0] + 3.50434289821397) + std::fabs(r.critical[1] + 2.8938659630479413) + std::fabs(r.critical[2] + 2.5840147047458037), 1e-12);
    s.check("adf (c, AIC): the AIC of the chosen lag on the common sample, −475.311008958", relative(r.information_criterion, -475.31100895774), 1e-11);
    o.deterministic = Deterministic::trend;
    r = adf(inc, o);
    s.check("adf (ct, AIC): τ = 0.120322886780", relative(r.statistic, 0.12032288678022628), 1e-9);
    s.check("adf (ct, AIC): p = 0.995266100362", relative(r.p_value, 0.9952661003623047), 1e-11);
    o.deterministic = Deterministic::none;
    r = adf(inc, o);
    s.check("adf (n, AIC): τ = 2.58814278288", relative(r.statistic, 2.588142782883076), 1e-10);
    s.require("adf (n, AIC): 3 lags chosen, 88 observations", r.lags == 3 && r.n_obs == 88);
    s.check("adf (n, AIC): critical values −2.59161518079, −1.94444098569, −1.61411506363",
            std::fabs(r.critical[0] + 2.5916151807851238) + std::fabs(r.critical[1] + 1.944440985689801) + std::fabs(r.critical[2] + 1.614115063626972), 1e-12);
    std::vector<double> d;
    for (std::size_t t = 1; t < inc.size(); t++) d.push_back(inc[t] - inc[t - 1]);
    AdfOptions ob;
    ob.selection = LagSelection::bic;
    r = adf(d, ob);
    s.check("adf (c, BIC): Δ log income, τ = −8.38881758294", relative(r.statistic, -8.38881758294472), 1e-10);
    s.check("adf (c, BIC): p = 2.3974659921e-13", relative(r.p_value, 2.3974659920958845e-13), 1e-8);
    AdfOptions of;
    of.deterministic = Deterministic::trend;
    of.selection = LagSelection::fixed;
    of.lags = 4;
    r = adf(inc, of);
    s.check("adf (ct, 4 lags fixed): τ = −0.750603462908", relative(r.statistic, -0.7506034629080272), 1e-10);
    s.check("adf (ct, 4 lags fixed): critical values at T = 87", std::fabs(r.critical[0] + 4.06678819414642) + std::fabs(r.critical[1] + 3.4622170894741555) + std::fabs(r.critical[2] + 3.1573220889654263), 1e-12);
  }
  {
    KpssResult k = kpss(inc);
    s.check("kpss (level): log income, η = 0.818266672955 with 12 lags (statsmodels)", relative(k.statistic, 0.8182666729547258), 1e-12);
    s.require("kpss (level): 12 lags, p reported as 0.01 beyond the table", k.lags == 12 && k.p_value == 0.01);
    s.require("kpss (level): Kwiatkowski et al.'s Table 1, 0.347, 0.463, 0.574, 0.739", k.critical[0] == 0.347 && k.critical[1] == 0.463 && k.critical[2] == 0.574 && k.critical[3] == 0.739);
    k = kpss(inc, true);
    s.check("kpss (trend): η = 0.116456986625", relative(k.statistic, 0.11645698662470931), 1e-12);
    s.require("kpss (trend): p reported as 0.10 below the table; critical values 0.119, 0.146, 0.176, 0.216",
              k.p_value == 0.10 && k.critical[0] == 0.119 && k.critical[3] == 0.216);
    std::vector<double> d;
    for (std::size_t t = 1; t < inc.size(); t++) d.push_back(inc[t] - inc[t - 1]);
    k = kpss(d, false, 4);
    s.check("kpss (level, 4 lags): Δ log income, η = 0.383012560021", relative(k.statistic, 0.3830125600208393), 1e-12);
    s.check("kpss: its interpolated p-value 0.0844773448186", k.p_value - 0.08447734481860376, 1e-14);
    // η rebuilt: residuals from the mean, partial sums, Bartlett long-run variance.
    double m = 0;
    for (double v : d) m += v / static_cast<double>(d.size());
    const double n = static_cast<double>(d.size());
    double s2 = 0, ps = 0, num = 0;
    for (double v : d) s2 += (v - m) * (v - m);
    for (int l = 1; l <= 4; l++) {
      double c = 0;
      for (std::size_t t = l; t < d.size(); t++) c += (d[t] - m) * (d[t - l] - m);
      s2 += 2 * (1 - l / 5.0) * c;
    }
    for (double v : d) {
      ps += v - m;
      num += ps * ps;
    }
    s.check("kpss: η = n⁻² Σ S_t² / s²(l), rebuilt", relative(k.statistic, num / (n * n * (s2 / n))), 1e-12);
  }
  {
    const tyche::regression::Rectangular X = tyche::regression::Rectangular::from_columns({inc});
    EngleGrangerResult r = engle_granger(cons, X);
    s.check("engle–granger: log consumption on log income, τ = −2.57478995979 (statsmodels coint)", relative(r.statistic, -2.5747899597942436), 1e-10);
    s.check("engle–granger: p = 0.247116251518 (MacKinnon 1994, N = 2)", relative(r.p_value, 0.24711625151821004), 1e-10);
    s.check("engle–granger: 5 and 10% critical values at T = n − 1 (statsmodels)", std::fabs(r.critical[1] + 3.404097890351407) + std::fabs(r.critical[2] + 3.091385056152638), 1e-12);
    const double T = static_cast<double>(cons.size() - 1);
    s.check("engle–granger: the 1% value from MacKinnon's Table 2 (N = 2: −3.89644 − 10.9519/T − 22.527/T²)", r.critical[0] - (-3.89644 - 10.9519 / T - 22.527 / (T * T)), 1e-12);
    const tyche::regression::Rectangular X2 = tyche::regression::Rectangular::from_columns({inc, inv});
    r = engle_granger(cons, X2, Deterministic::trend);
    s.check("engle–granger (ct, N = 3): τ = −2.61194770351", relative(r.statistic, -2.6119477035087155), 1e-10);
    s.check("engle–granger (ct, N = 3): p = 0.639860444587", relative(r.p_value, 0.6398604445866598), 1e-10);
    s.check("engle–granger (ct, N = 3): critical values", std::fabs(r.critical[0] + 4.875175137246524) + std::fabs(r.critical[1] + 4.2517790481587) + std::fabs(r.critical[2] + 3.935785553199366), 1e-12);
  }

  // ---- simulations ----
  {
    int rejected = 0;
    for (std::uint32_t rep = 0; rep < 400; rep++) {
      calx::random::Stream rng(20261024, 71, rep, 0);
      std::vector<double> x{0};
      for (int t = 1; t < 250; t++) x.push_back(x.back() + rng.normal());
      AdfOptions o;
      o.selection = LagSelection::fixed;
      o.lags = 0;
      rejected += adf(x, o).p_value < 0.05 ? 1 : 0;
    }
    s.check("adf: rejects random walks at 5% in 20 ± 4.4 of 400 (deviation in sd; size)", (rejected - 20) / 4.36, 3);
  }
  {
    calx::random::Stream rng(20261024, 72, 0, 0);
    std::vector<double> x{0};
    for (int t = 1; t < 400; t++) x.push_back(0.8 * x.back() + rng.normal());
    const AdfResult r = adf(x);
    s.require("adf: a stationary AR(1) with φ = 0.8, n = 400, is rejected at 1%", r.statistic < r.critical[0], std::to_string(r.statistic));
  }
  {
    int rejected = 0;
    for (std::uint32_t rep = 0; rep < 400; rep++) {
      calx::random::Stream rng(20261024, 73, rep, 0);
      std::vector<double> x(250);
      for (double& v : x) v = rng.normal();
      rejected += kpss(x).statistic > 0.463 ? 1 : 0;
    }
    s.check("kpss: rejects white noise at 5% in 20 ± 4.4 of 400 (deviation in sd; size)", (rejected - 20) / 4.36, 3);
  }
  {
    calx::random::Stream rng(20261024, 74, 0, 0);
    std::vector<double> x{0}, y, z{0};
    for (int t = 1; t < 300; t++) {
      x.push_back(x.back() + rng.normal());
      z.push_back(z.back() + rng.normal());
    }
    for (int t = 0; t < 300; t++) y.push_back(1 + 2 * x[t] + 0.5 * rng.normal());
    const auto X = tyche::regression::Rectangular::from_columns({x});
    const auto c = engle_granger(y, X);
    s.require("engle–granger: y = 1 + 2x + noise is cointegrated (rejected at 1%)", c.statistic < c.critical[0], std::to_string(c.statistic));
    s.check("engle–granger: its cointegrating slope near 2", c.cointegrating_regression.beta[1] - 2, 0.02);
    const auto Z = tyche::regression::Rectangular::from_columns({z});
    const auto u = engle_granger(y, Z);
    s.require("engle–granger: y and an independent walk are not (not rejected at 5%)", u.statistic > u.critical[1], std::to_string(u.statistic));
  }
  return s.done();
}
