// Linear models of panel data: the within (fixed-effects) estimator, one-way and two-way; the
// between estimator; random effects by the method of Swamy and Arora; the Hausman test.
//
// THE MODEL. y_it = x_it β + α_i + (γ_t) + ε_it for units i and periods t, the panel possibly
// unbalanced (Baltagi 2013, chs 2–3 and 9; Wooldridge 2010, ch. 10).
//
// THE WITHIN ESTIMATOR treats α_i (and γ_t) as parameters. By Frisch–Waugh–Lovell, β̂ is the OLS
// of M_D y on M_D X, M_D absorbing the effects (absorb.hpp): the deviations from unit means for
// one-way effects, and their two-way analogue by alternating projections, exact for unbalanced
// panels. It equals the least-squares dummy-variable regression on X and the indicators, with the
// same residuals (checks). A regressor constant within units (or, two-way, within periods) is
// absorbed: it is reported as aliased, with β = NaN, as Stata and R omit it.
//
// Degrees of freedom. The effects absorb a = rank(D) parameters (absorb.hpp), so the residual
// variance is σ̂² = RSS / (n − k − a) and the classical covariance σ̂² (X̃ᵀX̃)⁻¹, with t on n − k − a
// degrees of freedom: exactly the LSDV regression's. The cluster-robust covariance (CR1) is
// G/(G − 1) · (n − 1)/(n − k − a′) · (X̃ᵀX̃)⁻¹ (Σ_g X̃_gᵀ e_g e_gᵀ X̃_g) (X̃ᵀX̃)⁻¹, with t on G − 1, where
// a′ counts only the absorbed parameters not nested in the clusters: effects of a factor whose
// every level lies in one cluster (unit effects with clusters by unit) are not counted, since
// they do not shrink the space of the cluster sums (Cameron and Miller 2015; Stata's xtreg
// and reghdfe count them so, while the LSDV regression with clusters, Stata's areg, counts them,
// a difference of √((n − k − a)/(n − k)) in the standard errors). Clusters default to the units.
//
// THE BETWEEN ESTIMATOR is the OLS of the unit means ȳ_i on an intercept and x̄_i (Baltagi 2013,
// ch. 2), one observation per unit.
//
// RANDOM EFFECTS treats α_i as a random effect, uncorrelated with x, of variance σ²_u, and ε_it
// of variance σ²_ε. The feasible GLS estimator is the OLS of the quasi-demeaned
// y_it − θ_i ȳ_i on (1 − θ_i) and x_it − θ_i x̄_i, θ_i = 1 − √(σ²_ε / (T_i σ²_u + σ²_ε)), with the
// variances estimated as Swamy and Arora (1972): σ̂²_ε = RSS_within / (n − N − k_w) and
// σ̂²_u = max(0, σ̂²_b − σ̂²_ε / T̄), σ̂²_b = RSS_between / (N − k_b) and T̄ the harmonic mean of
// the T_i (Baltagi 2013, chs 2 and 9; Stata's xtreg, re). In a balanced panel β̂_RE is a matrix-
// weighted average of the within and between estimators (Maddala 1971), checked exactly for one
// regressor. The transformed errors have variance σ²_ε, so the classical covariance is σ̂²_ε (ZᵀZ)⁻¹
// with Swamy and Arora's σ̂²_ε, the scale of the within estimator's, not the transformed
// regression's own residual variance (which the effects inflate when they are correlated with x);
// a robust covariance asked for is the OLS's on the transformed data.
//
// THE HAUSMAN TEST (Hausman 1978) of H₀: E(α_i | x) = 0 compares the within estimator, consistent
// under H₀ and its alternative, with random effects, efficient under H₀:
// H = (β̂_FE − β̂_RE)ᵀ (V_FE − V_RE)⁻¹ (β̂_FE − β̂_RE) ~ χ²_k on the k coefficients both estimate. The
// covariances must both be classical (the efficiency of random effects is the premise). On one
// scale σ̂²_ε, V_FE − V_RE = σ̂²_ε ((X̃ᵀX̃)⁻¹ − [(ZᵀZ)⁻¹]_xx) is positive semi-definite, since the
// quasi-demeaned regressors keep the within variation and add a share of the between; when a
// sample's is not positive definite all the same, the statistic is computed with the inverse and
// the result says so.

#pragma once

#include <tyche/probability/distributions.hpp>
#include <tyche/regression/linear.hpp>

#include <calx/linalg/cholesky.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <vector>

#include <tyche/panel/absorb.hpp>

namespace tyche::panel {

namespace tr = tyche::regression;
using calx::linalg::Matrix;
using calx::linalg::Rectangular;

enum class PanelEffects { unit, two_way };
enum class PanelCovariance { classical, cluster };

struct PanelOptions {
  PanelEffects effects = PanelEffects::unit;
  PanelCovariance covariance = PanelCovariance::cluster;
  std::vector<std::int64_t> clusters;  // one label per observation; empty: the units
  double confidence = 0.95;
  AbsorbOptions absorb;
};

struct WithinFit {
  tr::LinearFit fit;                          // β on X's columns and its inference; fitted values and residuals of the full model
  std::vector<double> unit_effects;           // α̂ for each unit level (levels in order of first appearance)
  std::vector<double> period_effects;         // γ̂ for each period level (two-way; first period of each component at 0)
  std::vector<std::int64_t> unit_labels, period_labels;  // the label of each level
  std::size_t absorbed = 0;                   // a = rank(D)
  std::size_t absorbed_not_nested = 0;        // a′, counted in the cluster correction
  int sweeps = 0;                             // of the alternating projections
};

namespace detail {

inline std::vector<std::int64_t> labels_of(const Factor& f, std::span<const std::int64_t> labels) {
  std::vector<std::int64_t> out(f.levels);
  for (std::size_t i = 0; i < labels.size(); i++) out[f.level[i]] = labels[i];
  return out;
}

// Recomputes se, t, p, the intervals and the overall F from fit.covariance and fit.df_inference.
inline void reinfer(tr::LinearFit& fit) {
  const tyche::probability::StudentT tdist{fit.df_inference};
  const double tcrit = tdist.upper_quantile((1 - fit.confidence) / 2);
  std::vector<std::size_t> estimated;
  for (std::size_t j = 0; j < fit.k; j++) {
    if (fit.aliased[j]) continue;
    estimated.push_back(j);
    fit.se[j] = std::sqrt(fit.covariance(j, j));
    fit.t[j] = fit.beta[j] / fit.se[j];
    fit.p[j] = std::min(1.0, 2 * tdist.survival(std::fabs(fit.t[j])));
    fit.ci_lower[j] = fit.beta[j] - tcrit * fit.se[j];
    fit.ci_upper[j] = fit.beta[j] + tcrit * fit.se[j];
  }
  if (estimated.empty()) return;
  Rectangular R(estimated.size(), fit.k);
  for (std::size_t a = 0; a < estimated.size(); a++) R(a, estimated[a]) = 1;
  const auto w = tr::wald(fit, R, std::vector<double>(estimated.size(), 0.0));
  fit.f_statistic = w.f_statistic;
  fit.f_df1 = w.df1;
  fit.f_df2 = w.df2;
  fit.f_p_value = w.f_p_value;
}

inline void scale_covariance(tr::LinearFit& fit, double c) {
  for (double& v : fit.covariance.a) v *= c;
}

}  // namespace detail

// The within estimator of β, with unit (and period) effects absorbed.
inline WithinFit within(const Rectangular& X, std::span<const double> y, std::span<const std::int64_t> unit, std::span<const std::int64_t> period = {},
                        const PanelOptions& options = {}) {
  const std::size_t n = X.rows, k = X.cols;
  detail::require(y.size() == n && unit.size() == n, "y and a unit label for every row of X");
  const bool two = options.effects == PanelEffects::two_way;
  detail::require(!two || period.size() == n, "a period label for every observation (two-way effects)");
  detail::require(k > 0, "at least one regressor");
  WithinFit out;
  std::vector<Factor> factors{factor(unit)};
  if (two) factors.push_back(factor(period));
  // Demean y and every column of X.
  std::vector<std::vector<double>> cols(k + 1, std::vector<double>(n));
  for (std::size_t i = 0; i < n; i++) {
    cols[0][i] = y[i];
    for (std::size_t j = 0; j < k; j++) cols[j + 1][i] = X(i, j);
  }
  std::vector<double> norm(k + 1, 0.0);
  for (std::size_t j = 0; j <= k; j++)
    for (double v : cols[j]) norm[j] = std::max(norm[j], std::fabs(v));
  out.sweeps = demean(cols, factors, {}, options.absorb);
  Rectangular Xt(n, k);
  for (std::size_t j = 0; j < k; j++) {
    double left = 0;
    for (double v : cols[j + 1]) left = std::max(left, std::fabs(v));
    const bool absorbed = left <= 1e-9 * norm[j + 1];  // constant within the factors' levels
    for (std::size_t i = 0; i < n; i++) Xt(i, j) = absorbed ? 0.0 : cols[j + 1][i];
  }
  tr::OlsOptions o;
  o.confidence = options.confidence;
  o.drop_collinear = true;
  std::vector<std::int64_t> clusters;
  if (options.covariance == PanelCovariance::cluster) {
    clusters = options.clusters.empty() ? std::vector<std::int64_t>(unit.begin(), unit.end()) : options.clusters;
    detail::require(clusters.size() == n, "a cluster label for every observation");
    o.covariance = tr::Covariance::cluster;
    o.clusters = clusters;
  }
  out.fit = tr::ols(Xt, cols[0], o);
  auto& fit = out.fit;
  const double dn = static_cast<double>(n), dr = static_cast<double>(fit.rank);
  out.absorbed = absorbed_rank(factors);
  const double a = static_cast<double>(out.absorbed);
  detail::require(dn - dr - a >= 1, "more observations than coefficients and effects");
  if (options.covariance == PanelCovariance::classical) {
    out.absorbed_not_nested = out.absorbed;
    detail::scale_covariance(fit, (dn - dr) / (dn - dr - a));
    fit.df_inference = dn - dr - a;
  } else {
    std::size_t nested_levels = 0;
    for (const auto& f : factors)
      if (nested(f, clusters)) nested_levels += occurring(f);
    out.absorbed_not_nested = out.absorbed > nested_levels ? out.absorbed - nested_levels : 0;
    detail::scale_covariance(fit, (dn - dr) / (dn - dr - static_cast<double>(out.absorbed_not_nested)));
  }
  fit.df_residual = dn - dr - a;
  fit.sigma = std::sqrt(fit.rss / fit.df_residual);
  fit.adj_r_squared = std::numeric_limits<double>::quiet_NaN();  // r_squared is the within R²
  fit.aic = -2 * fit.log_likelihood + 2 * (dr + a);
  fit.bic = -2 * fit.log_likelihood + (dr + a) * std::log(dn);
  detail::reinfer(fit);
  // The full model's fitted values, and the effects from y − Xβ − e, which lies in the span of D.
  std::vector<double> d(n);
  for (std::size_t i = 0; i < n; i++) {
    double xb = 0;
    for (std::size_t j = 0; j < k; j++)
      if (!fit.aliased[j]) xb += X(i, j) * fit.beta[j];
    fit.fitted[i] = y[i] - fit.residuals[i];
    d[i] = fit.fitted[i] - xb;
  }
  const auto effects = recover_effects(d, factors, {}, options.absorb);
  out.unit_effects = effects[0];
  out.unit_labels = detail::labels_of(factors[0], unit);
  if (two) {
    out.period_effects = effects[1];
    out.period_labels = detail::labels_of(factors[1], period);
  }
  return out;
}

// The between estimator: OLS of the unit means of y on an intercept and the unit means of X.
// The coefficients are the intercept, then X's columns.
inline tr::LinearFit between(const Rectangular& X, std::span<const double> y, std::span<const std::int64_t> unit, const tr::OlsOptions& options = {}) {
  const std::size_t n = X.rows, k = X.cols;
  detail::require(y.size() == n && unit.size() == n, "y and a unit label for every row of X");
  const Factor f = factor(unit);
  Rectangular M(f.levels, k + 1);
  std::vector<double> ym(f.levels, 0.0), count(f.levels, 0.0);
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t l = f.level[i];
    count[l] += 1;
    ym[l] += y[i];
    for (std::size_t j = 0; j < k; j++) M(l, j + 1) += X(i, j);
  }
  for (std::size_t l = 0; l < f.levels; l++) {
    ym[l] /= count[l];
    M(l, 0) = 1;
    for (std::size_t j = 0; j < k; j++) M(l, j + 1) /= count[l];
  }
  return tr::ols(M, ym, options);
}

struct RandomEffectsFit {
  tr::LinearFit fit;            // the intercept, then X's columns
  double sigma_u = 0, sigma_e = 0;  // σ̂_u and σ̂_ε
  double rho = 0;               // σ̂²_u / (σ̂²_u + σ̂²_ε)
  std::vector<double> theta;    // θ_i for each unit level
};

// Random effects by feasible GLS with Swamy and Arora's variances.
inline RandomEffectsFit random_effects(const Rectangular& X, std::span<const double> y, std::span<const std::int64_t> unit, const tr::OlsOptions& options = {}) {
  const std::size_t n = X.rows, k = X.cols;
  PanelOptions po;
  po.covariance = PanelCovariance::classical;
  const auto fe = within(X, y, unit, {}, po);
  const auto be = between(X, y, unit);
  const Factor f = factor(unit);
  const double N = static_cast<double>(f.levels);
  std::vector<double> T(f.levels, 0.0);
  for (std::size_t l : f.level) T[l] += 1;
  double inverse_sum = 0;
  for (double t : T) inverse_sum += 1 / t;
  const double Tbar = N / inverse_sum;
  RandomEffectsFit r;
  const double s2e = fe.fit.rss / fe.fit.df_residual;
  detail::require(N > static_cast<double>(be.rank), "more units than between coefficients");
  const double s2b = be.rss / (N - static_cast<double>(be.rank));
  const double s2u = std::max(0.0, s2b - s2e / Tbar);
  r.sigma_e = std::sqrt(s2e);
  r.sigma_u = std::sqrt(s2u);
  r.rho = s2u / (s2u + s2e);
  r.theta.resize(f.levels);
  for (std::size_t l = 0; l < f.levels; l++) r.theta[l] = 1 - std::sqrt(s2e / (T[l] * s2u + s2e));
  // Unit means and the quasi-demeaned regression.
  std::vector<double> ym(f.levels, 0.0);
  Rectangular xm(f.levels, k);
  for (std::size_t i = 0; i < n; i++) {
    ym[f.level[i]] += y[i];
    for (std::size_t j = 0; j < k; j++) xm(f.level[i], j) += X(i, j);
  }
  for (std::size_t l = 0; l < f.levels; l++) {
    ym[l] /= T[l];
    for (std::size_t j = 0; j < k; j++) xm(l, j) /= T[l];
  }
  Rectangular Z(n, k + 1);
  std::vector<double> ys(n);
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t l = f.level[i];
    const double th = r.theta[l];
    Z(i, 0) = 1 - th;
    ys[i] = y[i] - th * ym[l];
    for (std::size_t j = 0; j < k; j++) Z(i, j + 1) = X(i, j) - th * xm(l, j);
  }
  r.fit = tr::ols(Z, ys, options);
  if (options.covariance == tr::Covariance::classical && r.fit.sigma > 0) {
    detail::scale_covariance(r.fit, s2e / (r.fit.sigma * r.fit.sigma));
    detail::reinfer(r.fit);
  }
  return r;
}

struct HausmanTest {
  double statistic = 0, df = 0, p_value = 0;
  bool positive_definite = true;  // whether V_FE − V_RE was positive definite
};

// Hausman's test of random against fixed effects, on the coefficients of X both estimate.
inline HausmanTest hausman(const WithinFit& fe, const RandomEffectsFit& re) {
  const std::size_t k = fe.fit.k;
  detail::require(re.fit.k == k + 1, "random effects with an intercept and the within fit's regressors");
  std::vector<std::size_t> common;
  for (std::size_t j = 0; j < k; j++)
    if (!fe.fit.aliased[j] && !re.fit.aliased[j + 1]) common.push_back(j);
  const std::size_t q = common.size();
  detail::require(q > 0, "at least one coefficient in common");
  Matrix D(q);
  std::vector<double> d(q);
  for (std::size_t a = 0; a < q; a++) {
    d[a] = fe.fit.beta[common[a]] - re.fit.beta[common[a] + 1];
    for (std::size_t b = 0; b < q; b++) D(a, b) = fe.fit.covariance(common[a], common[b]) - re.fit.covariance(common[a] + 1, common[b] + 1);
  }
  HausmanTest h;
  const auto chol = calx::linalg::cholesky(D);
  std::vector<double> x;
  if (chol.positive_definite) {
    x = calx::linalg::cholesky_solve(chol, d);
  } else {
    h.positive_definite = false;
    Matrix B(q);
    for (std::size_t a = 0; a < q; a++) B(a, 0) = d[a];
    const Matrix S = calx::linalg::solve(D, B);
    x.resize(q);
    for (std::size_t a = 0; a < q; a++) x[a] = S(a, 0);
  }
  for (std::size_t a = 0; a < q; a++) h.statistic += d[a] * x[a];
  h.df = static_cast<double>(q);
  h.p_value = h.statistic > 0 ? tyche::probability::ChiSquared{h.df}.survival(h.statistic) : 1.0;
  return h;
}

}  // namespace tyche::panel
