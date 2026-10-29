// Checks of tyche/time_series's vector autoregressions and the Hodrick–Prescott filter.
//
// - Lütkepohl (2005), §3.2.3: the VAR(2) of the first differences of log investment, income and
//   consumption, 1960Q1–1978Q4 (73 observations used), whose estimates he prints to three decimals:
//   ν = (−0.017, 0.016, 0.013); A₁ = (−0.320 0.146 0.961; 0.044 −0.153 0.289; −0.002 0.225 −0.264);
//   A₂ = (−0.161 0.115 0.934; 0.050 0.019 −0.010; 0.034 0.355 −0.022); and Σ̂_u × 10⁴ =
//   (21.30 0.72 1.23; 0.72 1.37 0.61; 1.23 0.61 0.89). Against statsmodels 0.15.0's VAR (computed
//   2026-10-24) as a cross-check of the further digits, the log-likelihood, the standard errors,
//   the orthogonalised impulse responses and the forecasts with their standard errors.
// - Lütkepohl §4.3: with orders up to 4 on that sample, AIC and FPE choose 2, HQ and SC choose 0.
// - Closed forms: Θ₀ = P with P Pᵀ = Σ̂_u; Φ₁ = A₁ and Φ₂ = A₁² + A₂; a VAR(1)'s forecasts
//   ν + A ŷ and its MSE Σ_u + A Σ_u Aᵀ.
// - A simulated VAR(2) of two variables recovered within four standard errors; its order chosen by
//   SC; an explosive VAR flagged unstable.
// - The HP filter: a linear series is its own trend; the cycle is orthogonal to constants and to
//   time; λ = 0 leaves the series; the band solve against a dense one; log income against
//   statsmodels' hpfilter.

#include <tyche/time_series/filters.hpp>
#include <tyche/time_series/var.hpp>

#include <calx/linalg/dense.hpp>
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

std::vector<std::vector<double>> e1() {
  std::ifstream in(std::string(TYCHE_FIXTURES) + "/lutkepohl-e1.dat");
  std::vector<std::vector<double>> rows;
  for (std::string l; std::getline(in, l);) {
    std::istringstream ss(l);
    double a, b, c;
    if (ss >> a >> b >> c) rows.push_back({a, b, c});
  }
  return rows;
}

double printed(double v, double digits) { return std::round(v * std::pow(10, digits)) / std::pow(10, digits); }

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto rows = e1();
  // Δ log of 1960Q1–1978Q4 (76 quarters): 75 differences.
  Rectangular Y(75, 3);
  for (std::size_t t = 0; t < 75; t++)
    for (std::size_t j = 0; j < 3; j++) Y(t, j) = std::log(rows[t + 1][j]) - std::log(rows[t][j]);

  const VarFit v = var(Y, 2);
  {
    s.require("var (Lütkepohl §3.2.3): 73 observations used", v.T_effective == 73);
    const double nu[3] = {-0.017, 0.016, 0.013};
    const double A1[9] = {-0.320, 0.146, 0.961, 0.044, -0.153, 0.289, -0.002, 0.225, -0.264};
    const double A2[9] = {-0.161, 0.115, 0.934, 0.050, 0.019, -0.010, 0.034, 0.355, -0.022};
    const double S[9] = {21.30, 0.72, 1.23, 0.72, 1.37, 0.61, 1.23, 0.61, 0.89};
    bool ok_nu = true, ok_a = true, ok_s = true;
    for (int a = 0; a < 3; a++) ok_nu = ok_nu && printed(v.nu[a], 3) == nu[a];
    for (int i = 0; i < 9; i++) {
      ok_a = ok_a && printed(v.A[0].a[i], 3) == A1[i] && printed(v.A[1].a[i], 3) == A2[i];
      ok_s = ok_s && printed(v.sigma_u.a[i] * 1e4, 2) == S[i];
    }
    s.require("var (Lütkepohl §3.2.3): ν̂ as printed, (−0.017, 0.016, 0.013)", ok_nu);
    s.require("var (Lütkepohl §3.2.3): Â₁ and Â₂ as printed to three decimals", ok_a);
    s.require("var (Lütkepohl §3.2.3): Σ̂_u × 10⁴ as printed to two decimals", ok_s);
    const double sm_A1[9] = {-0.3196309715806485, 0.145988827066046, 0.9612190324601493, 0.0439310617186784, -0.1527319078222849,
                             0.2885016360024158, -0.0024226661299694, 0.2248126706873571, -0.263967508550022};
    const double sm_S[9] = {2.1296289187146896e-03, 7.1616666903583518e-05, 1.2324036430940279e-04, 7.1616666903583518e-05, 1.3733772760941436e-04,
                            6.1458667534994830e-05, 1.2324036430940279e-04, 6.1458667534994830e-05, 8.9203513932846956e-05};
    double ea = 0, es = 0;
    for (int i = 0; i < 9; i++) {
      ea = std::max(ea, std::fabs(v.A[0].a[i] - sm_A1[i]));
      es = std::max(es, relative(v.sigma_u.a[i], sm_S[i]));
    }
    s.check("var: Â₁ against statsmodels (cross-check)", ea, 1e-12);
    s.check("var: Σ̂_u against statsmodels, relative", es, 1e-11);
    s.check("var: the log-likelihood 606.306967527 (statsmodels)", relative(v.log_likelihood, 606.3069675270688), 1e-12);
    const auto& f = v.equations[0];
    s.check("var: standard errors of the investment equation's y_{t−1} coefficients (statsmodels)",
            std::max({relative(f.se[1], 0.125456432431894), relative(f.se[2], 0.5456658349459842), relative(f.se[3], 0.6643103193558623)}), 1e-11);
    s.require("var: stable, spectral radius 0.5705 (statsmodels' companion roots)", v.is_stable && std::fabs(v.spectral_radius - 0.5705) < 5e-5, std::to_string(v.spectral_radius));
  }
  {
    // Orders up to 4 on 1960Q1–1978Q4: the same 71 observations for every order.
    const VarOrder o = select_order(Y, 4);
    s.require("var order (Lütkepohl §4.3): AIC and FPE choose 2, HQ and SC choose 0", o.aic == 2 && o.fpe == 2 && o.hq == 0 && o.bic == 0,
              "aic " + std::to_string(o.aic) + ", fpe " + std::to_string(o.fpe) + ", hq " + std::to_string(o.hq) + ", sc " + std::to_string(o.bic));
    // statsmodels counts the K intercepts among the free parameters; Lütkepohl does not: they differ by 2K/T.
    s.check("var order: AIC(2) is statsmodels' −24.5096626031 less its 2K/T for the intercepts", o.aic_values[2] - (-24.509662603072684 - 2.0 * 3 / 71), 1e-10);
  }
  {
    const auto Th = impulse_responses(v, 2, true);
    const auto Ph = impulse_responses(v, 2, false);
    const auto P = calx::linalg::cholesky(v.sigma_u).l;
    double e0 = 0, e1v = 0, e2 = 0;
    const Matrix A1sq = v.A[0] * v.A[0] + v.A[1];
    for (int i = 0; i < 9; i++) {
      e0 = std::max(e0, std::fabs(Th[0].a[i] - P.a[i]));
      e1v = std::max(e1v, std::fabs(Ph[1].a[i] - v.A[0].a[i]));
      e2 = std::max(e2, std::fabs(Ph[2].a[i] - A1sq.a[i]));
    }
    s.check("irf: Θ₀ = P, the lower Cholesky factor of Σ̂_u", e0, 0);
    s.check("irf: Φ₁ = A₁", e1v, 0);
    s.check("irf: Φ₂ = A₁² + A₂", e2, 1e-16);
    const double sm_h1[9] = {-0.0119567545176408, 0.0064385599358988, 0.0073031242784761, 0.002560761145023, -0.0003506192478509,
                             0.0021919700204819, -0.0004678543634619, 0.001308957109975, -0.0020055652825418};
    double eh = 0;
    for (int i = 0; i < 9; i++) eh = std::max(eh, std::fabs(Th[1].a[i] - sm_h1[i]));
    s.check("irf: the orthogonalised responses at horizon 1 against statsmodels", eh, 1e-14);
    const auto fc = forecast(v, 3);
    const double sm_fc[9] = {-0.0108109430690531, 0.0199108377734452, 0.0216287280573333, 0.0107809079511874, 0.0203486771499882,
                             0.0146538755487314, 0.0211157020065115, 0.0169805876756368, 0.0198257446886965};
    const double sm_se[9] = {0.0461479026469751, 0.011719118038889, 0.0094447611898262, 0.0486557661082923, 0.0121992988191506,
                             0.0097548994078129, 0.0490331206743488, 0.0123144201713495, 0.0107874083644648};
    double ef = 0, ese = 0;
    for (int i = 0; i < 9; i++) {
      ef = std::max(ef, std::fabs(fc.mean.a[i] - sm_fc[i]));
      ese = std::max(ese, relative(fc.se.a[i], sm_se[i]));
    }
    s.check("forecast: three quarters ahead against statsmodels", ef, 1e-14);
    s.check("forecast: their standard errors (MSE Σ Φ_i Σ_u Φ_iᵀ) against statsmodels, relative", ese, 1e-12);
  }

  // ---- a simulated VAR(2) ----
  {
    calx::random::Stream rng(20261024, 81, 0, 0);
    const double a1[4] = {0.5, 0.1, 0.2, 0.3}, a2[4] = {-0.2, 0.0, 0.1, 0.25}, nu[2] = {1.0, -0.5};
    const std::size_t T = 3000, burn = 300;
    Rectangular Ys(T, 2);
    std::vector<double> prev1{0, 0}, prev2{0, 0};
    for (std::size_t t = 0; t < T + burn; t++) {
      const double e0 = rng.normal(), e1 = 0.5 * e0 + rng.normal();  // correlated shocks
      std::vector<double> y{nu[0] + a1[0] * prev1[0] + a1[1] * prev1[1] + a2[0] * prev2[0] + a2[1] * prev2[1] + e0,
                            nu[1] + a1[2] * prev1[0] + a1[3] * prev1[1] + a2[2] * prev2[0] + a2[3] * prev2[1] + e1};
      prev2 = prev1;
      prev1 = y;
      if (t >= burn) {
        Ys(t - burn, 0) = y[0];
        Ys(t - burn, 1) = y[1];
      }
    }
    const VarFit w = var(Ys, 2);
    double z = 0;
    for (std::size_t a = 0; a < 2; a++) {
      const auto& f = w.equations[a];
      z = std::max(z, std::fabs(f.beta[0] - nu[a]) / f.se[0]);
      for (std::size_t b = 0; b < 2; b++) {
        z = std::max(z, std::fabs(f.beta[1 + b] - a1[2 * a + b]) / f.se[1 + b]);
        z = std::max(z, std::fabs(f.beta[3 + b] - a2[2 * a + b]) / f.se[3 + b]);
      }
    }
    s.check("var: a simulated VAR(2) recovered within 4 standard errors (largest |β̂ − β|/se)", z, 4);
    s.check("var: and Σ_u = (1, 0.5; 0.5, 1.25)", std::max({std::fabs(w.sigma_u(0, 0) - 1), std::fabs(w.sigma_u(0, 1) - 0.5), std::fabs(w.sigma_u(1, 1) - 1.25)}), 0.1);
    s.require("var: SC chooses the true order 2 among 0–6", select_order(Ys, 6).bic == 2);
  }
  {
    // A VAR(1): forecasts ν + A ŷ, MSE Σ_u + A Σ_u Aᵀ at h = 2; an explosive one.
    calx::random::Stream rng(20261024, 82, 0, 0);
    Rectangular Ys(500, 2);
    std::vector<double> y{0, 0};
    for (std::size_t t = 0; t < 500; t++) {
      y = {0.3 + 0.6 * y[0] + 0.2 * y[1] + rng.normal(), 0.1 * y[0] + 0.4 * y[1] + rng.normal()};
      Ys(t, 0) = y[0];
      Ys(t, 1) = y[1];
    }
    const VarFit w = var(Ys, 1);
    const auto f = forecast(w, 2);
    const std::vector<double> last = Ys.row(499);
    double one[2], two[2];
    for (int a = 0; a < 2; a++) one[a] = w.nu[a] + w.A[0](a, 0) * last[0] + w.A[0](a, 1) * last[1];
    for (int a = 0; a < 2; a++) two[a] = w.nu[a] + w.A[0](a, 0) * one[0] + w.A[0](a, 1) * one[1];
    s.check("forecast: a VAR(1), ν + A y_T and ν + A ŷ_T(1)", std::max({std::fabs(f.mean(0, 0) - one[0]), std::fabs(f.mean(1, 1) - two[1])}), 1e-14);
    const Matrix M = w.sigma_u + w.A[0] * w.sigma_u * calx::linalg::transpose(w.A[0]);
    s.check("forecast: its MSE at h = 2, Σ_u + A Σ_u Aᵀ", relative(f.se(1, 0) * f.se(1, 0), M(0, 0)) + relative(f.se(1, 1) * f.se(1, 1), M(1, 1)), 1e-13);
    Rectangular Ze(80, 2);
    std::vector<double> u{1, 1};
    for (std::size_t t = 0; t < 80; t++) {
      u = {1.05 * u[0] + 0.1 * rng.normal(), 0.5 * u[1] + rng.normal()};
      Ze(t, 0) = u[0];
      Ze(t, 1) = u[1];
    }
    s.require("var: an explosive process (root 1.05) is flagged unstable", !var(Ze, 1).is_stable);
  }

  // ---- the Hodrick–Prescott filter ----
  {
    std::vector<double> lin(50);
    for (int t = 0; t < 50; t++) lin[t] = 3 - 0.25 * t;
    const auto h = hodrick_prescott(lin);
    double e = 0;
    for (int t = 0; t < 50; t++) e = std::max(e, std::fabs(h.trend[t] - lin[t]));
    s.check("hp: a linear series is its own trend", e, 1e-11);
    std::vector<double> inc;
    for (const auto& r : rows) inc.push_back(std::log(r[1]));
    const auto g = hodrick_prescott(inc, 1600);
    double c0 = 0, c1 = 0, scale = 0;
    for (std::size_t t = 0; t < inc.size(); t++) {
      c0 += g.cycle[t];
      c1 += static_cast<double>(t) * g.cycle[t];
      scale += std::fabs(g.cycle[t]) * static_cast<double>(t);
    }
    s.check("hp: the cycle sums to zero", c0, 1e-12);
    s.check("hp: and is orthogonal to time, Σ t c_t / Σ t |c_t|", c1 / scale, 1e-12);
    const double sm_tr[4] = {6.135870627446102, 6.345503209532945, 7.068884762282933, 7.913114196187519};
    const std::size_t at[4] = {0, 10, 45, 91};
    double et = 0;
    for (int i = 0; i < 4; i++) et = std::max(et, std::fabs(g.trend[at[i]] - sm_tr[i]));
    s.check("hp: log income's trend at λ = 1600 against statsmodels' hpfilter", et, 1e-12);
    const auto z0 = hodrick_prescott(inc, 0);
    double ez = 0;
    for (std::size_t t = 0; t < inc.size(); t++) ez = std::max(ez, std::fabs(z0.trend[t] - inc[t]));
    s.check("hp: λ = 0 leaves the series", ez, 0);
  }
  return s.done();
}
