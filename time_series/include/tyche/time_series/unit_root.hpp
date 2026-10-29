// Tests of unit roots and of cointegration: the augmented Dickey–Fuller test with MacKinnon's
// p-values and critical values, the KPSS test of stationarity, and the Engle–Granger test.
//
// ADF (Dickey and Fuller 1979, JASA 74, 427–431; Said and Dickey 1984, Biometrika 71, 599–607).
// The regression Δy_t = [α + βt + γ₂t²] + ρ y_{t−1} + Σ_{i=1}^{k} δ_i Δy_{t−i} + e_t by least squares
// on t = k + 2, …, n, and τ = ρ̂ / se(ρ̂); H₀: ρ = 0 (a unit root), rejected for small τ. The
// deterministic terms are none, a constant, a constant and trend, or a constant, trend and squared
// trend (MacKinnon's τ_nc, τ_c, τ_ct, τ_ctt). The lag length k is fixed, or chosen by AIC or BIC
// among 0, …, k_max, each fitted on the common sample t = k_max + 2, …, n so that the criteria are
// comparable, and the chosen one refitted on its full sample; k_max defaults to
// ⌈12 (n/100)^{1/4}⌉ (Schwert 1989, J. Bus. Econ. Stat. 7, 147–159), at most n/2 − (deterministic
// terms) − 1. This is statsmodels' adfuller, whose AIC is −2ℓ + 2·rank as tyche's.
//
// MACKINNON. The p-value is MacKinnon's (1994) approximate asymptotic distribution function,
// "Approximate asymptotic distribution functions for unit-root and cointegration tests", J. Bus.
// Econ. Stat. 12, 167–176: p = Φ(Σ γ_i τ^i), a quadratic in τ below τ* and a cubic above it, 0
// below τ_min and 1 above τ_max, for N = 1, …, 6 integrated series (N = 1 for the ADF test). Its
// coefficients are those transcribed in statsmodels (statsmodels/tsa/adfvalues.py, BSD-3, retrieved
// 2026-10-24; the journal's tables were not available to check them against, but the p-values at
// MacKinnon's 2010 critical values come out at 1, 5 and 10% to within 0.003, a check made here).
// The critical values are MacKinnon's (2010) response surfaces, "Critical values for
// cointegration tests", Queen's Economics Department Working Paper 1227, Tables 2–4:
// C(level, T) = β_∞ + β₁/T + β₂/T² + β₃/T³ for N = 1, …, 12 (τ_nc for N = 1 only), transcribed
// from the paper (statsmodels' copy differs from it in three coefficients: N = 2, τ_c, 1%, β₂ is
// −22.527 not −33.527; N = 7, τ_c, 10%, β₂ and β₃ are −6.922 and 67.721; N = 3 and 5, τ_c, 5%,
// β₁ in the last digit). T is the number of observations of the test regression (ADF), or n − 1
// for Engle–Granger (Stata's egranger, as statsmodels follows it).
//
// KPSS (Kwiatkowski, Phillips, Schmidt and Shin 1992, J. Econometrics 54, 159–178). The residuals
// e_t of y on a constant (level stationarity, η_μ) or a constant and trend (η_τ), their partial sums
// S_t, the long-run variance s²(l) = (1/n) Σ e_t² + (2/n) Σ_{s=1}^{l} (1 − s/(l + 1)) Σ_t e_t e_{t−s}
// (Bartlett weights, Newey and West 1987), and η = n⁻² Σ S_t² / s²(l) (KPSS §2), rejected
// for large η. l defaults to ⌈12 (n/100)^{1/4}⌉ (statsmodels' "legacy" choice). Critical values
// from KPSS Table 1: η_μ 0.347, 0.463, 0.574, 0.739 and η_τ 0.119, 0.146, 0.176, 0.216 at 10, 5,
// 2.5 and 1%; the p-value interpolated linearly between them and reported as 0.10 or 0.01 beyond.
//
// ENGLE–GRANGER (Engle and Granger 1987, Econometrica 55, 251–276). y on the other series X and
// the deterministic terms by least squares, then the ADF test without deterministic terms on the
// residuals; the statistic is referred to MacKinnon's distribution and critical values for
// N = 1 + (columns of X) and the deterministic terms of the first regression.

#pragma once

#include <tyche/probability/distributions.hpp>
#include <tyche/regression/linear.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

enum class Deterministic { none, constant, trend, quadratic_trend };  // MacKinnon's nc, c, ct, ctt
enum class LagSelection { fixed, aic, bic };

namespace detail {

inline std::size_t deterministic_terms(Deterministic d) {
  return d == Deterministic::none ? 0 : d == Deterministic::constant ? 1 : d == Deterministic::trend ? 2 : 3;
}

// MacKinnon (1994), as statsmodels transcribes it: [regression][N − 1].
struct Mk94 {
  double tau_star[6], tau_min[6], tau_max[6];
  double small[6][3];  // γ₀, γ₁, γ₂ (γ₂ already ×10⁻²)
  double large[6][4];  // γ₀, γ₁ (×10⁻¹), γ₂ (×10⁻¹), γ₃ (×10⁻²) already scaled
};

inline const Mk94& mk94(Deterministic d) {
  constexpr double inf = std::numeric_limits<double>::infinity();
  static const Mk94 nc{{-1.04, -1.53, -2.68, -3.09, -3.07, -3.77},
                       {-19.04, -19.62, -21.21, -23.25, -21.63, -25.74},
                       {inf, 1.51, 0.86, 0.88, 1.05, 1.24},
                       {{0.6344, 1.2378, 3.2496e-2}, {1.9129, 1.3857, 3.5322e-2}, {2.7648, 1.4502, 3.4186e-2}, {3.4336, 1.4835, 3.19e-2}, {4.0999, 1.5533, 3.59e-2}, {4.5388, 1.5344, 2.9807e-2}},
                       {{0.4797, 0.93557, -0.06999, 0.033066}, {1.5578, 0.8558, -0.2083, -0.033549}, {2.2268, 0.68093, -0.32362, -0.054448},
                        {2.7654, 0.64502, -0.30811, -0.044946}, {3.2684, 0.68051, -0.26778, -0.034972}, {3.7268, 0.7167, -0.23648, -0.028288}}};
  static const Mk94 c{{-1.61, -2.62, -3.13, -3.47, -3.78, -3.93},
                      {-18.83, -18.86, -23.48, -28.07, -25.96, -23.27},
                      {2.74, 0.92, 0.55, 0.61, 0.79, 1},
                      {{2.1659, 1.4412, 3.8269e-2}, {2.92, 1.5012, 3.9796e-2}, {3.4699, 1.4856, 3.164e-2}, {3.9673, 1.4777, 2.6315e-2}, {4.5509, 1.5338, 2.9545e-2}, {5.1399, 1.6036, 3.4445e-2}},
                      {{1.7339, 0.93202, -0.12745, -0.010368}, {2.1945, 0.64695, -0.29198, -0.042377}, {2.5893, 0.45168, -0.36529, -0.050074},
                       {3.0387, 0.45452, -0.33666, -0.041921}, {3.5049, 0.52098, -0.29158, -0.033468}, {3.9489, 0.58933, -0.25359, -0.02721}}};
  static const Mk94 ct{{-2.89, -3.19, -3.50, -3.65, -3.80, -4.36},
                       {-16.18, -21.15, -25.37, -26.63, -26.53, -26.18},
                       {0.7, 0.63, 0.71, 0.93, 1.19, 1.42},
                       {{3.2512, 1.6047, 4.9588e-2}, {3.6646, 1.5419, 3.6448e-2}, {4.0983, 1.5173, 2.9898e-2}, {4.5844, 1.5338, 2.8796e-2}, {5.0722, 1.5634, 2.9472e-2}, {5.53, 1.5914, 3.0392e-2}},
                       {{2.5261, 0.61654, -0.37956, -0.060285}, {2.85, 0.5272, -0.36622, -0.051695}, {3.221, 0.5255, -0.32685, -0.041501},
                        {3.652, 0.59758, -0.27483, -0.032081}, {4.0712, 0.66428, -0.23464, -0.02546}, {4.4735, 0.71757, -0.20681, -0.021196}}};
  static const Mk94 ctt{{-3.21, -3.51, -3.81, -3.83, -4.12, -4.63},
                        {-17.17, -21.1, -24.33, -24.03, -24.33, -28.22},
                        {0.54, 0.79, 1.08, 1.43, 3.49, 1.92},
                        {{4.0003, 1.658, 4.8288e-2}, {4.3534, 1.6016, 3.7947e-2}, {4.7343, 1.5768, 3.2396e-2}, {5.214, 1.6077, 3.3449e-2}, {5.6481, 1.6274, 3.3455e-2}, {5.9296, 1.5929, 2.8223e-2}},
                        {{3.0778, 0.49529, -0.41477, -0.059359}, {3.4713, 0.5967, -0.32507, -0.042286}, {3.8637, 0.67852, -0.26286, -0.031381},
                         {4.2736, 0.76199, -0.21534, -0.024026}, {4.6679, 0.82618, -0.1822, -0.019147}, {5.0009, 0.83735, -0.16994, -0.016928}}};
  switch (d) {
    case Deterministic::none: return nc;
    case Deterministic::constant: return c;
    case Deterministic::trend: return ct;
    default: return ctt;
  }
}

// MacKinnon (2010), Tables 2–4: [N − 1][level 1%, 5%, 10%] = {β∞, β₁, β₂, β₃}.
using Surface = std::array<std::array<std::array<double, 4>, 3>, 12>;
inline const Surface& mk2010(Deterministic d) {
  static const Surface c{{
      {{{-3.43035, -6.5393, -16.786, -79.433}, {-2.86154, -2.8903, -4.234, -40.040}, {-2.56677, -1.5384, -2.809, 0}}},
      {{{-3.89644, -10.9519, -22.527, 0}, {-3.33613, -6.1101, -6.823, 0}, {-3.04445, -4.2412, -2.720, 0}}},
      {{{-4.29374, -14.4354, -33.195, 47.433}, {-3.74066, -8.5631, -10.852, 27.982}, {-3.45218, -6.2143, -3.718, 0}}},
      {{{-4.64332, -18.1031, -37.972, 0}, {-4.09600, -11.2349, -11.175, 0}, {-3.81020, -8.3931, -4.137, 0}}},
      {{{-4.95756, -21.8883, -45.142, 0}, {-4.41519, -14.0406, -12.575, 0}, {-4.13157, -10.7417, -3.784, 0}}},
      {{{-5.24568, -25.6688, -57.737, 88.639}, {-4.70693, -16.9178, -17.492, 60.007}, {-4.42501, -13.1875, -5.104, 27.877}}},
      {{{-5.51233, -29.5760, -69.398, 164.295}, {-4.97684, -19.9021, -22.045, 110.761}, {-4.69648, -15.7315, -6.922, 67.721}}},
      {{{-5.76202, -33.5258, -82.189, 256.289}, {-5.22924, -23.0023, -24.646, 144.479}, {-4.95007, -18.3959, -7.344, 94.872}}},
      {{{-5.99742, -37.6572, -87.365, 248.316}, {-5.46697, -26.2057, -26.627, 176.382}, {-5.18897, -21.1377, -9.484, 172.704}}},
      {{{-6.22103, -41.7154, -102.680, 389.330}, {-5.69244, -29.4521, -30.994, 251.016}, {-5.41533, -24.0006, -7.514, 163.049}}},
      {{{-6.43377, -46.0084, -106.809, 352.752}, {-5.90714, -32.8336, -30.275, 249.994}, {-5.63086, -26.9693, -4.083, 151.427}}},
      {{{-6.63790, -50.2095, -124.156, 579.622}, {-6.11279, -36.2681, -32.505, 314.802}, {-5.83724, -29.9864, -2.686, 184.116}}},
  }};
  static const Surface ct{{
      {{{-3.95877, -9.0531, -28.428, -134.155}, {-3.41049, -4.3904, -9.036, -45.374}, {-3.12705, -2.5856, -3.925, -22.380}}},
      {{{-4.32762, -15.4387, -35.679, 0}, {-3.78057, -9.5106, -12.074, 0}, {-3.49631, -7.0815, -7.538, 21.892}}},
      {{{-4.66305, -18.7688, -49.793, 104.244}, {-4.11890, -11.8922, -19.031, 77.332}, {-3.83511, -9.0723, -8.504, 35.403}}},
      {{{-4.96940, -22.4694, -52.599, 51.314}, {-4.42871, -14.5876, -18.228, 39.647}, {-4.14633, -11.2500, -9.873, 54.109}}},
      {{{-5.25276, -26.2183, -59.631, 50.646}, {-4.71537, -17.3569, -22.660, 91.359}, {-4.43422, -13.6078, -10.238, 76.781}}},
      {{{-5.51727, -29.9760, -75.222, 202.253}, {-4.98228, -20.3050, -25.224, 132.030}, {-4.70233, -16.1253, -9.836, 94.272}}},
      {{{-5.76537, -33.9165, -84.312, 245.394}, {-5.23299, -23.3328, -28.955, 182.342}, {-4.95405, -18.7352, -10.168, 120.575}}},
      {{{-6.00003, -37.8892, -96.428, 335.920}, {-5.46971, -26.4771, -31.034, 220.165}, {-5.19183, -21.4328, -10.726, 157.955}}},
      {{{-6.22288, -41.9496, -109.881, 466.068}, {-5.69447, -29.7152, -33.784, 273.002}, {-5.41738, -24.2882, -8.584, 169.891}}},
      {{{-6.43551, -46.1151, -120.814, 566.823}, {-5.90887, -33.0251, -37.208, 346.189}, {-5.63255, -27.2042, -6.792, 177.666}}},
      {{{-6.63894, -50.4287, -128.997, 642.781}, {-6.11404, -36.4610, -36.246, 348.554}, {-5.83850, -30.1995, -5.163, 210.338}}},
      {{{-6.83488, -54.7119, -139.800, 736.376}, {-6.31127, -39.9676, -37.021, 406.051}, {-6.03650, -33.2381, -6.606, 317.776}}},
  }};
  static const Surface ctt{{
      {{{-4.37113, -11.5882, -35.819, -334.047}, {-3.83239, -5.9057, -12.490, -118.284}, {-3.55326, -3.6596, -5.293, -63.559}}},
      {{{-4.69276, -20.2284, -64.919, 88.884}, {-4.15387, -13.3114, -28.402, 72.741}, {-3.87346, -10.4637, -17.408, 66.313}}},
      {{{-4.99071, -23.5873, -76.924, 184.782}, {-4.45311, -15.7732, -32.316, 122.705}, {-4.17280, -12.4909, -17.912, 83.285}}},
      {{{-5.26780, -27.2836, -78.971, 137.871}, {-4.73244, -18.4833, -31.875, 111.817}, {-4.45268, -14.7199, -17.969, 101.920}}},
      {{{-5.52826, -30.9051, -92.490, 248.096}, {-4.99491, -21.2360, -37.685, 194.208}, {-4.71587, -17.0820, -18.631, 136.672}}},
      {{{-5.77379, -34.7010, -105.937, 393.991}, {-5.24217, -24.2177, -39.153, 232.528}, {-4.96397, -19.6064, -18.858, 174.919}}},
      {{{-6.00609, -38.7383, -108.605, 365.208}, {-5.47664, -27.3005, -39.498, 246.918}, {-5.19921, -22.2617, -17.910, 208.494}}},
      {{{-6.22758, -42.7154, -119.622, 421.395}, {-5.69983, -30.4365, -44.300, 345.480}, {-5.42320, -24.9686, -19.688, 274.462}}},
      {{{-6.43933, -46.7581, -136.691, 651.380}, {-5.91298, -33.7584, -42.686, 346.629}, {-5.63704, -27.8965, -13.880, 236.975}}},
      {{{-6.64235, -50.9783, -145.462, 752.228}, {-6.11753, -37.056, -48.719, 473.905}, {-5.84215, -30.8119, -14.938, 316.006}}},
      {{{-6.83743, -55.2861, -152.651, 792.577}, {-6.31396, -40.5507, -46.771, 487.185}, {-6.03921, -33.8950, -9.122, 285.164}}},
      {{{-7.02582, -59.6037, -166.368, 989.879}, {-6.50353, -44.0797, -47.242, 543.889}, {-6.22941, -36.9673, -10.868, 418.414}}},
  }};
  // τ_nc, N = 1 only (Table 2).
  static const Surface nc{{{{{-2.56574, -2.2358, -3.627, 0}, {-1.94100, -0.2686, -3.365, 31.223}, {-1.61682, 0.2656, -2.714, 25.364}}}}};
  switch (d) {
    case Deterministic::none: return nc;
    case Deterministic::constant: return c;
    case Deterministic::trend: return ct;
    default: return ctt;
  }
}

}  // namespace detail

// MacKinnon's (1994) asymptotic p-value of a Dickey–Fuller or Engle–Granger τ, N = 1, …, 6.
inline double mackinnon_p(double tau, Deterministic d, int N = 1) {
  if (N < 1 || N > 6) throw std::invalid_argument("tyche::time_series::mackinnon_p: N from 1 to 6");
  const auto& t = detail::mk94(d);
  const int i = N - 1;
  if (std::isnan(tau)) return tau;
  if (tau > t.tau_max[i]) return 1;
  if (tau < t.tau_min[i]) return 0;
  double z;
  if (tau <= t.tau_star[i]) z = t.small[i][0] + tau * (t.small[i][1] + tau * t.small[i][2]);
  else z = t.large[i][0] + tau * (t.large[i][1] + tau * (t.large[i][2] + tau * t.large[i][3]));
  return probability::Normal{}.cdf(z);
}

// MacKinnon's (2010) critical values at 1, 5 and 10% for T observations (T = ∞: asymptotic).
inline std::array<double, 3> mackinnon_critical(Deterministic d, int N = 1, double T = std::numeric_limits<double>::infinity()) {
  if (N < 1 || N > 12 || (d == Deterministic::none && N != 1)) throw std::invalid_argument("tyche::time_series::mackinnon_critical: N from 1 to 12 (1 without deterministic terms)");
  const auto& s = detail::mk2010(d)[static_cast<std::size_t>(N - 1)];
  std::array<double, 3> c{};
  const double u = std::isinf(T) ? 0.0 : 1 / T;
  for (int l = 0; l < 3; l++) c[l] = s[l][0] + u * (s[l][1] + u * (s[l][2] + u * s[l][3]));
  return c;
}

struct AdfOptions {
  Deterministic deterministic = Deterministic::constant;
  LagSelection selection = LagSelection::aic;
  int lags = -1;                 // fixed: the lag order; aic/bic: the largest (−1: Schwert's default)
};

struct AdfResult {
  double statistic = 0;          // τ = ρ̂ / se(ρ̂)
  double p_value = 0;            // MacKinnon (1994)
  std::array<double, 3> critical{};  // 1, 5, 10% (MacKinnon 2010, T = observations used)
  std::size_t lags = 0, n_obs = 0;
  double rho = 0;                // the coefficient on y_{t−1}
  double information_criterion = std::numeric_limits<double>::quiet_NaN();  // of the chosen lag (common sample)
  regression::LinearFit regression;  // the test regression: columns y_{t−1}, Δy_{t−1..t−k}, then deterministic terms
};

namespace detail {

// The ADF design on t = start, …, n − 1 (0-based; start ≥ k + 1): y_{t−1}, Δy_{t−1..t−k}, deterministic.
inline void adf_design(std::span<const double> y, std::size_t k, std::size_t start, Deterministic d, regression::Rectangular& X, std::vector<double>& dy) {
  const std::size_t n = y.size(), m = n - start, nd = deterministic_terms(d);
  X = regression::Rectangular(m, 1 + k + nd);
  dy.resize(m);
  for (std::size_t r = 0; r < m; r++) {
    const std::size_t t = start + r;
    dy[r] = y[t] - y[t - 1];
    X(r, 0) = y[t - 1];
    for (std::size_t i = 1; i <= k; i++) X(r, i) = y[t - i] - y[t - i - 1];
    // Deterministic terms; the trend counts observations of the regression from 1, as statsmodels' add_trend.
    const double tt = static_cast<double>(r + 1);
    if (nd >= 1) X(r, 1 + k) = 1;
    if (nd >= 2) X(r, 2 + k) = tt;
    if (nd >= 3) X(r, 3 + k) = tt * tt;
  }
}

}  // namespace detail

inline AdfResult adf(std::span<const double> y, const AdfOptions& options = {}) {
  const std::size_t n = y.size(), nd = detail::deterministic_terms(options.deterministic);
  if (n < 10) throw std::invalid_argument("tyche::time_series::adf: at least 10 observations");
  long long kmax;
  if (options.lags < 0) {
    kmax = static_cast<long long>(std::ceil(12 * std::pow(static_cast<double>(n) / 100, 0.25)));
    kmax = std::min<long long>(kmax, static_cast<long long>(n / 2) - static_cast<long long>(nd) - 1);
  } else {
    kmax = options.lags;
    if (kmax > static_cast<long long>(n / 2) - static_cast<long long>(nd) - 1) throw std::invalid_argument("tyche::time_series::adf: lags < n/2 − deterministic terms − 1");
  }
  if (kmax < 0) throw std::invalid_argument("tyche::time_series::adf: too short a series");
  const std::size_t K = static_cast<std::size_t>(kmax);
  AdfResult out;
  std::size_t k = K;
  regression::Rectangular X;
  std::vector<double> dy;
  if (options.selection != LagSelection::fixed) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j <= K; j++) {
      detail::adf_design(y, j, K + 1, options.deterministic, X, dy);
      const auto f = regression::ols(X, dy);
      const double ic = options.selection == LagSelection::aic ? f.aic : f.bic;
      if (ic < best) {
        best = ic;
        k = j;
      }
    }
    out.information_criterion = best;
  }
  detail::adf_design(y, k, k + 1, options.deterministic, X, dy);
  out.regression = regression::ols(X, dy);
  out.lags = k;
  out.n_obs = dy.size();
  out.rho = out.regression.beta[0];
  out.statistic = out.regression.t[0];
  out.p_value = mackinnon_p(out.statistic, options.deterministic, 1);
  out.critical = mackinnon_critical(options.deterministic, 1, static_cast<double>(out.n_obs));
  return out;
}

struct KpssResult {
  double statistic = 0;          // η
  double p_value = 0;            // interpolated in KPSS Table 1, within [0.01, 0.10]
  std::size_t lags = 0;
  std::array<double, 4> critical{};  // 10, 5, 2.5, 1%
};

inline KpssResult kpss(std::span<const double> y, bool trend = false, int lags = -1) {
  const std::size_t n = y.size();
  if (n < 4) throw std::invalid_argument("tyche::time_series::kpss: at least 4 observations");
  regression::Rectangular X(n, trend ? 2 : 1);
  for (std::size_t t = 0; t < n; t++) {
    X(t, 0) = 1;
    if (trend) X(t, 1) = static_cast<double>(t + 1);
  }
  const auto f = regression::ols(X, y);
  const auto& e = f.residuals;
  std::size_t l = lags >= 0 ? static_cast<std::size_t>(lags) : static_cast<std::size_t>(std::ceil(12 * std::pow(static_cast<double>(n) / 100, 0.25)));
  l = std::min(l, n - 1);
  const double dn = static_cast<double>(n);
  double s2 = 0;
  for (double v : e) s2 += v * v;
  for (std::size_t s = 1; s <= l; s++) {
    double c = 0;
    for (std::size_t t = s; t < n; t++) c += e[t] * e[t - s];
    s2 += 2 * (1 - static_cast<double>(s) / static_cast<double>(l + 1)) * c;
  }
  s2 /= dn;
  double partial = 0, eta = 0;
  for (double v : e) {
    partial += v;
    eta += partial * partial;
  }
  KpssResult r;
  r.statistic = eta / (dn * dn * s2);
  r.lags = l;
  r.critical = trend ? std::array<double, 4>{0.119, 0.146, 0.176, 0.216} : std::array<double, 4>{0.347, 0.463, 0.574, 0.739};
  const double pv[4] = {0.10, 0.05, 0.025, 0.01};
  if (r.statistic <= r.critical[0]) r.p_value = 0.10;
  else if (r.statistic >= r.critical[3]) r.p_value = 0.01;
  else
    for (int i = 0; i < 3; i++)
      if (r.statistic <= r.critical[i + 1]) {
        const double u = (r.statistic - r.critical[i]) / (r.critical[i + 1] - r.critical[i]);
        r.p_value = pv[i] + u * (pv[i + 1] - pv[i]);
        break;
      }
  return r;
}

struct EngleGrangerResult {
  double statistic = 0, p_value = 0;
  std::array<double, 3> critical{};  // 1, 5, 10% for N = 1 + columns of X, T = n − 1
  std::size_t lags = 0;
  regression::LinearFit cointegrating_regression;  // y on the deterministic terms, then X's columns
  AdfResult adf;                                   // on its residuals, without deterministic terms
};

// y and the columns of X are the series believed to be I(1); deterministic is that of the
// cointegrating regression (constant by default). adf_options.deterministic is ignored (none).
inline EngleGrangerResult engle_granger(std::span<const double> y, const regression::Rectangular& X, Deterministic deterministic = Deterministic::constant,
                                        AdfOptions adf_options = {}) {
  const std::size_t n = y.size();
  if (X.rows != n || X.cols == 0) throw std::invalid_argument("tyche::time_series::engle_granger: X has one row per observation");
  const std::size_t nd = detail::deterministic_terms(deterministic);
  regression::Rectangular Z(n, nd + X.cols);
  for (std::size_t t = 0; t < n; t++) {
    const double tt = static_cast<double>(t + 1);
    if (nd >= 1) Z(t, 0) = 1;
    if (nd >= 2) Z(t, 1) = tt;
    if (nd >= 3) Z(t, 2) = tt * tt;
    for (std::size_t j = 0; j < X.cols; j++) Z(t, nd + j) = X(t, j);
  }
  EngleGrangerResult r;
  r.cointegrating_regression = regression::ols(Z, y);
  adf_options.deterministic = Deterministic::none;
  r.adf = adf(r.cointegrating_regression.residuals, adf_options);
  r.statistic = r.adf.statistic;
  r.lags = r.adf.lags;
  const int N = static_cast<int>(1 + X.cols);
  r.p_value = N <= 6 ? mackinnon_p(r.statistic, deterministic, N) : std::numeric_limits<double>::quiet_NaN();
  r.critical = deterministic == Deterministic::none ? std::array<double, 3>{NAN, NAN, NAN} : mackinnon_critical(deterministic, N, static_cast<double>(n - 1));
  return r;
}

}  // namespace tyche::time_series
