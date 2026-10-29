// Linear regression: ordinary and weighted least squares, with classical, heteroskedasticity-
// consistent and cluster-robust covariances, and Wald tests of linear restrictions.
//
// THE MODEL. y = X β + ε with X an n × k design matrix (one row per observation, one column per
// regressor; include a column of ones for an intercept, e.g. with design_matrix(columns, true)),
// E ε = 0. OLS estimates β by min ‖y − Xβ‖²; WLS, given weights wᵢ > 0 (proportional to
// 1/Var εᵢ), by min Σ wᵢ (yᵢ − xᵢβ)², i.e. OLS on √w X and √w y.
//
// HOW IT IS COMPUTED. Never by the normal equations, which square the condition number: the
// columns of √w X are scaled to unit norm and factored by Householder QR with column pivoting
// (calx::linalg::qr; Golub and Van Loan 2013, Algorithms 5.2.1 and 5.4.1), β̂ = R⁻¹ Qᵀ√w y, and
// every covariance is formed from R⁻¹ and the rows of Q. The coefficients of NIST's Longley data
// (condition number about 10⁷ after scaling) come out to about 12 significant digits, Norris's to
// 14 (checks). A column that is numerically a combination of the others (|R_jj| below
// rank_tolerance · |R_00| on the scaled columns) makes ols() throw std::invalid_argument naming
// it, or, with drop_collinear, gives it β = NaN as R's lm does and fits the rest.
//
// THE COVARIANCE OF β̂ (OlsOptions::covariance), with B = (XᵀWX)⁻¹, e the weighted residuals
// √wᵢ (yᵢ − xᵢβ̂), hᵢ the leverages (the diagonal of the hat matrix, ‖row i of Q‖²), r the rank:
// - classical: σ̂² B, σ̂² = Σ eᵢ² / (n − r);
// - hc0: B (Σ eᵢ² xᵢxᵢᵀ) B, White (1980), Econometrica 48, 817–838;
// - hc1: hc0 · n / (n − r); hc2: eᵢ² / (1 − hᵢ); hc3: eᵢ² / (1 − hᵢ)², MacKinnon and White (1985),
//   J. Econometrics 29, 305–325 (HC3 ≈ their jackknife; Long and Ervin 2000 recommend it for
//   small n);
// - cluster: c · B (Σ_g (Σ_{i∈g} eᵢxᵢ)(Σ_{i∈g} eᵢxᵢ)ᵀ) B with c = G/(G − 1) · (n − 1)/(n − r),
//   G clusters (one label per observation in OlsOptions::clusters): Cameron and Miller (2015),
//   "A practitioner's guide to cluster-robust inference", J. Human Resources 50, 317–372 (their
//   CR1 correction; Stata's and statsmodels' default).
// t statistics, p-values and confidence intervals use Student's t with df_inference degrees of
// freedom: n − r, or G − 1 with clusters (as Cameron and Miller recommend, and Stata does).
//
// THE SUMMARY STATISTICS. TSS is Σ wᵢ(yᵢ − ȳ_w)² when the model has an intercept (some column
// of X constant and non-zero) and Σ wᵢyᵢ² otherwise, as NIST's certified values and R define it;
// R² = 1 − RSS/TSS, adjusted R² = 1 − (1 − R²)(n − c)/(n − r) with c = 1 if there is an intercept.
// The F statistic tests every coefficient but the intercept: ((TSS − RSS)/q) / σ̂² with
// q = r − c under the classical covariance, and the Wald form (wald() below) under a robust one,
// with (q, df_inference) degrees of freedom. The Gaussian log-likelihood at the MLE σ² = RSS/n is
// ℓ = −(n/2)(log 2π + log(RSS/n) + 1) + ½ Σ log wᵢ; AIC = −2ℓ + 2r, BIC = −2ℓ + r log n
// (statsmodels' convention, counting the coefficients only; R's AIC for lm adds 2 for σ²).
//
// LIMITS. The inference is exact under normal, homoskedastic, independent errors (classical) and
// asymptotic otherwise (robust covariances). Weights must be positive and finite; observations
// with zero weight should be removed by the caller.

#pragma once

#include <tyche/probability/distributions.hpp>

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/dense.hpp>
#include <calx/linalg/qr.hpp>
#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::regression {

using calx::linalg::Matrix;       // square, row-major: covariances (k × k)
using calx::linalg::Rectangular;  // rows × cols, row-major: design matrices (n × k)

// The design matrix whose columns are the given regressors (all of length n), preceded by a
// column of ones when intercept is true.
inline Rectangular design_matrix(const std::vector<std::vector<double>>& columns, bool intercept = true) {
  const std::size_t n = columns.empty() ? 0 : columns[0].size();
  const std::size_t c0 = intercept ? 1 : 0;
  Rectangular X(n, columns.size() + c0);
  for (std::size_t i = 0; i < n; i++) {
    if (intercept) X(i, 0) = 1;
    for (std::size_t j = 0; j < columns.size(); j++) {
      if (columns[j].size() != n) throw std::invalid_argument("tyche::regression::design_matrix: columns of one length");
      X(i, j + c0) = columns[j][i];
    }
  }
  return X;
}

enum class Covariance { classical, hc0, hc1, hc2, hc3, cluster };

struct OlsOptions {
  std::vector<double> weights;          // empty: OLS; else one wᵢ > 0 per observation (WLS)
  Covariance covariance = Covariance::classical;
  std::vector<std::int64_t> clusters;   // one label per observation, for Covariance::cluster
  double confidence = 0.95;             // the level of the confidence intervals
  bool drop_collinear = false;          // false: throw on a collinear column; true: β = NaN for it
  double rank_tolerance = -1;           // relative, on unit-norm columns; < 0: max(n, k) ε
};

struct LinearFit {
  // Sizes.
  std::size_t n = 0, k = 0, rank = 0;   // observations, columns of X, coefficients estimated
  double df_residual = 0;               // n − rank
  double df_inference = 0;              // the t degrees of freedom: n − rank, or G − 1 with clusters
  bool has_intercept = false;           // a column of X is constant and non-zero
  std::vector<bool> aliased;            // columns dropped as collinear (all false unless drop_collinear)
  Covariance covariance_type = Covariance::classical;
  std::size_t clusters = 0;             // G, with Covariance::cluster
  double confidence = 0.95;

  // Coefficients (one per column of X; NaN for an aliased column) and their inference.
  std::vector<double> beta, se, t, p, ci_lower, ci_upper;
  Matrix covariance;                    // the chosen covariance of β̂, k × k (NaN rows for aliased)
  Matrix xtx_inverse;                   // B = (XᵀWX)⁻¹, unscaled, k × k (zero rows for aliased)

  // The fit.
  std::vector<double> fitted;           // X β̂
  std::vector<double> residuals;        // y − X β̂ (not multiplied by √w)
  std::vector<double> leverage;         // hᵢ, the diagonal of the hat matrix of √w X
  double rss = 0, tss = 0;              // weighted sums of squares (TSS centred iff has_intercept)
  double r_squared = 0, adj_r_squared = 0;
  double sigma = 0;                     // σ̂ = √(RSS / (n − rank)), the residual standard error
  double f_statistic = 0, f_df1 = 0, f_df2 = 0, f_p_value = 0;  // all coefficients but the intercept zero
  double log_likelihood = 0, aic = 0, bic = 0;
};

namespace detail {

// The weighted least-squares core shared by ols() and glm(): QR of √w X with unit-norm columns.
struct WeightedLeastSquares {
  std::size_t n = 0, k = 0, rank = 0;
  calx::linalg::Qr qr;                  // of √w X D⁻¹, D = diag(column norms)
  std::vector<double> scale;            // the column norms (0 for a zero column)
  std::vector<double> beta;             // in X's columns; NaN for aliased
  std::vector<bool> aliased;
  Matrix r_inverse;                     // R₁₁⁻¹ in scaled, pivoted coordinates (rank × rank)
  Rectangular q;                        // the first `rank` columns of Q (n × rank)
};

inline WeightedLeastSquares weighted_least_squares(const Rectangular& X, std::span<const double> y, std::span<const double> sqrt_w,
                                                   double rank_tolerance, bool drop_collinear, const char* who) {
  WeightedLeastSquares w;
  w.n = X.rows;
  w.k = X.cols;
  Rectangular A(w.n, w.k);
  std::vector<double> b(w.n);
  for (std::size_t i = 0; i < w.n; i++) {
    const double s = sqrt_w.empty() ? 1.0 : sqrt_w[i];
    for (std::size_t j = 0; j < w.k; j++) A(i, j) = s * X(i, j);
    b[i] = s * y[i];
  }
  w.scale.assign(w.k, 0.0);
  for (std::size_t j = 0; j < w.k; j++) {
    double ss = 0, big = 0;
    for (std::size_t i = 0; i < w.n; i++) big = std::max(big, std::fabs(A(i, j)));
    if (big > 0) {
      for (std::size_t i = 0; i < w.n; i++) ss += (A(i, j) / big) * (A(i, j) / big);
      w.scale[j] = big * std::sqrt(ss);
      for (std::size_t i = 0; i < w.n; i++) A(i, j) /= w.scale[j];
    }
  }
  w.qr = calx::linalg::qr(std::move(A), true, rank_tolerance);
  w.rank = w.qr.rank;
  w.aliased.assign(w.k, false);
  for (std::size_t j = w.rank; j < w.k; j++) w.aliased[w.qr.permutation[j]] = true;
  if (w.rank < w.k && !drop_collinear) {
    std::string cols;
    for (std::size_t j = 0; j < w.k; j++)
      if (w.aliased[j]) cols += (cols.empty() ? "" : ", ") + std::to_string(j);
    throw std::invalid_argument(std::string(who) + ": the columns of X are collinear; column(s) " + cols +
                                " are (numerically) combinations of the others (set drop_collinear to fit without them)");
  }
  if (w.rank == 0) throw std::invalid_argument(std::string(who) + ": X has no non-zero column");
  const auto bt = calx::linalg::least_squares(w.qr, b);
  w.beta.assign(w.k, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t j = 0; j < w.k; j++)
    if (!w.aliased[j]) w.beta[j] = bt[j] / w.scale[j];
  w.r_inverse = calx::linalg::upper_inverse(calx::linalg::r_factor(w.qr, w.rank));
  w.q = calx::linalg::thin_q(w.qr, w.rank);
  return w;
}

// R⁻¹ M R⁻ᵀ (scaled, pivoted, rank × rank) mapped to X's columns: entry (a, b) divided by the
// scales of the columns at pivots a and b; aliased rows and columns get `fill`.
inline Matrix unscale(const WeightedLeastSquares& w, const Matrix& middle, double fill) {
  const std::size_t r = w.rank;
  Matrix T(r);  // R⁻¹ M
  for (std::size_t i = 0; i < r; i++)
    for (std::size_t j = 0; j < r; j++) {
      double s = 0;
      for (std::size_t l = i; l < r; l++) s += w.r_inverse(i, l) * middle(l, j);
      T(i, j) = s;
    }
  Matrix out(w.k);
  std::fill(out.a.begin(), out.a.end(), fill);
  for (std::size_t i = 0; i < r; i++)
    for (std::size_t j = 0; j < r; j++) {
      double s = 0;
      for (std::size_t l = j; l < r; l++) s += T(i, l) * w.r_inverse(j, l);
      const std::size_t a = w.qr.permutation[i], b = w.qr.permutation[j];
      out(a, b) = s / (w.scale[a] * w.scale[b]);
    }
  return out;
}

// The sandwich's middle Σ_g (Σ_{i∈g} uᵢ qᵢ)(…)ᵀ in Q's coordinates; with no clusters each
// observation is its own group (weights ωᵢ multiply uᵢ²).
inline Matrix meat(const WeightedLeastSquares& w, std::span<const double> u, std::span<const double> omega, std::span<const std::int64_t> clusters,
                   std::size_t& groups) {
  const std::size_t r = w.rank;
  Matrix M(r);
  if (clusters.empty()) {
    for (std::size_t i = 0; i < w.n; i++) {
      const double c = u[i] * u[i] * (omega.empty() ? 1.0 : omega[i]);
      if (c == 0) continue;
      for (std::size_t a = 0; a < r; a++)
        for (std::size_t b = 0; b <= a; b++) M(a, b) += c * w.q(i, a) * w.q(i, b);
    }
    groups = w.n;
  } else {
    std::map<std::int64_t, std::vector<double>> sums;
    for (std::size_t i = 0; i < w.n; i++) {
      auto& v = sums[clusters[i]];
      if (v.empty()) v.assign(r, 0.0);
      for (std::size_t a = 0; a < r; a++) v[a] += u[i] * w.q(i, a);
    }
    for (const auto& [label, v] : sums)
      for (std::size_t a = 0; a < r; a++)
        for (std::size_t b = 0; b <= a; b++) M(a, b) += v[a] * v[b];
    groups = sums.size();
  }
  for (std::size_t a = 0; a < r; a++)
    for (std::size_t b = 0; b < a; b++) M(b, a) = M(a, b);
  return M;
}

inline bool has_constant_column(const Rectangular& X) {
  for (std::size_t j = 0; j < X.cols; j++) {
    if (X.rows == 0 || X(0, j) == 0) continue;
    bool constant = true;
    for (std::size_t i = 1; i < X.rows && constant; i++) constant = X(i, j) == X(0, j);
    if (constant) return true;
  }
  return false;
}

}  // namespace detail

struct WaldTest {
  double f_statistic = 0;     // W / q
  double chi2_statistic = 0;  // W = (Rβ − r)ᵀ (R V Rᵀ)⁻¹ (Rβ − r)
  double df1 = 0, df2 = 0;    // q, and the fit's df_inference
  double f_p_value = 0;       // P(F(q, df2) > W/q)
  double chi2_p_value = 0;    // P(χ²_q > W), the asymptotic form
};

// The Wald test of H₀: R β = r, R a q × k matrix (q restrictions on the k coefficients), with the
// fit's covariance (classical or robust): W = (Rβ̂ − r)ᵀ (R V Rᵀ)⁻¹ (Rβ̂ − r), F = W/q on
// (q, df_inference) degrees of freedom (Greene 2012, Econometric Analysis, 7th ed., ch. 5; under
// the classical covariance the F form is exact for normal errors). R must not load an aliased
// column.
inline WaldTest wald(const LinearFit& fit, const Rectangular& R, std::span<const double> r) {
  const std::size_t q = R.rows, k = fit.k;
  if (R.cols != k || r.size() != q || q == 0) throw std::invalid_argument("tyche::regression::wald: R is q × k and r has q entries, q ≥ 1");
  std::vector<double> d(q);
  Matrix RVR(q);
  for (std::size_t a = 0; a < q; a++) {
    double v = -r[a];
    for (std::size_t j = 0; j < k; j++) {
      if (R(a, j) == 0) continue;
      if (fit.aliased[j]) throw std::invalid_argument("tyche::regression::wald: a restriction loads an aliased column");
      v += R(a, j) * fit.beta[j];
    }
    d[a] = v;
  }
  for (std::size_t a = 0; a < q; a++)
    for (std::size_t b = 0; b <= a; b++) {
      double s = 0;
      for (std::size_t i = 0; i < k; i++) {
        if (R(a, i) == 0) continue;
        for (std::size_t j = 0; j < k; j++)
          if (R(b, j) != 0) s += R(a, i) * fit.covariance(i, j) * R(b, j);
      }
      RVR(a, b) = RVR(b, a) = s;
    }
  const auto chol = calx::linalg::cholesky(RVR);
  std::vector<double> x;
  if (chol.positive_definite) {
    x = calx::linalg::cholesky_solve(chol, d);
  } else {
    Matrix D(q);
    for (std::size_t a = 0; a < q; a++) D(a, 0) = d[a];
    const Matrix X = calx::linalg::solve(RVR, D);  // throws if R V Rᵀ is singular
    x.resize(q);
    for (std::size_t a = 0; a < q; a++) x[a] = X(a, 0);
  }
  WaldTest w;
  for (std::size_t a = 0; a < q; a++) w.chi2_statistic += d[a] * x[a];
  w.df1 = static_cast<double>(q);
  w.df2 = fit.df_inference;
  w.f_statistic = w.chi2_statistic / w.df1;
  w.f_p_value = probability::FisherF{w.df1, w.df2}.survival(w.f_statistic);
  w.chi2_p_value = probability::ChiSquared{w.df1}.survival(w.chi2_statistic);
  return w;
}

inline LinearFit ols(const Rectangular& X, std::span<const double> y, const OlsOptions& options = {}) {
  const std::size_t n = X.rows, k = X.cols;
  if (y.size() != n) throw std::invalid_argument("tyche::regression::ols: y has one entry per row of X");
  if (!options.weights.empty() && options.weights.size() != n) throw std::invalid_argument("tyche::regression::ols: one weight per observation");
  if (options.covariance == Covariance::cluster && options.clusters.size() != n)
    throw std::invalid_argument("tyche::regression::ols: one cluster label per observation for Covariance::cluster");
  if (!(options.confidence > 0 && options.confidence < 1)) throw std::invalid_argument("tyche::regression::ols: a confidence level in (0, 1)");
  std::vector<double> sw;
  if (!options.weights.empty()) {
    sw.resize(n);
    for (std::size_t i = 0; i < n; i++) {
      if (!(options.weights[i] > 0 && std::isfinite(options.weights[i]))) throw std::invalid_argument("tyche::regression::ols: weights positive and finite");
      sw[i] = std::sqrt(options.weights[i]);
    }
  }
  const double tol = options.rank_tolerance < 0 ? static_cast<double>(std::max(n, k)) * std::numeric_limits<double>::epsilon() : options.rank_tolerance;
  const auto w = detail::weighted_least_squares(X, y, sw, tol, options.drop_collinear, "tyche::regression::ols");
  const std::size_t r = w.rank;
  if (n <= r) throw std::invalid_argument("tyche::regression::ols: more observations than coefficients are needed");

  LinearFit fit;
  fit.n = n;
  fit.k = k;
  fit.rank = r;
  fit.df_residual = static_cast<double>(n - r);
  fit.has_intercept = detail::has_constant_column(X);
  fit.aliased = w.aliased;
  fit.covariance_type = options.covariance;
  fit.confidence = options.confidence;
  fit.beta = w.beta;

  // Fitted values and residuals.
  fit.fitted.assign(n, 0.0);
  fit.residuals.resize(n);
  std::vector<double> e(n);  // √w (y − Xβ)
  for (std::size_t i = 0; i < n; i++) {
    double v = 0;
    for (std::size_t j = 0; j < k; j++)
      if (!w.aliased[j]) v += X(i, j) * fit.beta[j];
    fit.fitted[i] = v;
    fit.residuals[i] = y[i] - v;
    e[i] = (sw.empty() ? 1.0 : sw[i]) * fit.residuals[i];
  }
  fit.leverage.assign(n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t a = 0; a < r; a++) fit.leverage[i] += w.q(i, a) * w.q(i, a);

  // Sums of squares.
  double sumw = 0, ybar = 0;
  for (std::size_t i = 0; i < n; i++) {
    const double wi = sw.empty() ? 1.0 : options.weights[i];
    sumw += wi;
    ybar += wi * y[i];
  }
  ybar /= sumw;
  for (std::size_t i = 0; i < n; i++) {
    const double wi = sw.empty() ? 1.0 : options.weights[i];
    fit.rss += e[i] * e[i];
    const double d = fit.has_intercept ? y[i] - ybar : y[i];
    fit.tss += wi * d * d;
  }
  const double c = fit.has_intercept ? 1.0 : 0.0, dn = static_cast<double>(n), dr = static_cast<double>(r);
  fit.r_squared = 1 - fit.rss / fit.tss;
  fit.adj_r_squared = 1 - (1 - fit.r_squared) * (dn - c) / (dn - dr);
  const double sigma2 = fit.rss / (dn - dr);
  fit.sigma = std::sqrt(sigma2);

  // Covariances.
  Matrix identity = Matrix::identity(r);
  fit.xtx_inverse = detail::unscale(w, identity, 0.0);
  std::size_t groups = 0;
  switch (options.covariance) {
    case Covariance::classical:
      fit.covariance = detail::unscale(w, sigma2 * identity, std::numeric_limits<double>::quiet_NaN());
      break;
    case Covariance::hc0:
    case Covariance::hc1:
    case Covariance::hc2:
    case Covariance::hc3: {
      std::vector<double> omega(n, 1.0);
      for (std::size_t i = 0; i < n; i++) {
        const double h = fit.leverage[i];
        if (options.covariance == Covariance::hc1) omega[i] = dn / (dn - dr);
        else if (options.covariance == Covariance::hc2) omega[i] = 1 / (1 - h);
        else if (options.covariance == Covariance::hc3) omega[i] = 1 / ((1 - h) * (1 - h));
      }
      fit.covariance = detail::unscale(w, detail::meat(w, e, omega, {}, groups), std::numeric_limits<double>::quiet_NaN());
      break;
    }
    case Covariance::cluster: {
      Matrix M = detail::meat(w, e, {}, options.clusters, groups);
      if (groups < 2) throw std::invalid_argument("tyche::regression::ols: at least two clusters");
      const double G = static_cast<double>(groups);
      fit.covariance = detail::unscale(w, (G / (G - 1) * (dn - 1) / (dn - dr)) * M, std::numeric_limits<double>::quiet_NaN());
      fit.clusters = groups;
      break;
    }
  }
  fit.df_inference = options.covariance == Covariance::cluster ? static_cast<double>(groups) - 1 : fit.df_residual;

  // Coefficient inference.
  const probability::StudentT tdist{fit.df_inference};
  const double tcrit = tdist.upper_quantile((1 - options.confidence) / 2);
  fit.se.assign(k, std::numeric_limits<double>::quiet_NaN());
  fit.t = fit.p = fit.ci_lower = fit.ci_upper = fit.se;
  for (std::size_t j = 0; j < k; j++) {
    if (w.aliased[j]) continue;
    fit.se[j] = std::sqrt(fit.covariance(j, j));
    fit.t[j] = fit.beta[j] / fit.se[j];
    fit.p[j] = std::min(1.0, 2 * tdist.survival(std::fabs(fit.t[j])));
    fit.ci_lower[j] = fit.beta[j] - tcrit * fit.se[j];
    fit.ci_upper[j] = fit.beta[j] + tcrit * fit.se[j];
  }

  // The overall F test: every coefficient but the intercept (and aliased ones) zero.
  const std::size_t q = r - static_cast<std::size_t>(fit.has_intercept && r > 0 ? 1 : 0);
  if (q == 0) {
    fit.f_statistic = fit.f_p_value = std::numeric_limits<double>::quiet_NaN();
  } else if (options.covariance == Covariance::classical) {
    fit.f_statistic = ((fit.tss - fit.rss) / static_cast<double>(q)) / sigma2;
    fit.f_df1 = static_cast<double>(q);
    fit.f_df2 = fit.df_residual;
    fit.f_p_value = probability::FisherF{fit.f_df1, fit.f_df2}.survival(fit.f_statistic);
  } else {
    // The columns tested: the estimated ones, less one constant column if there is an intercept.
    std::vector<std::size_t> tested;
    bool skipped_constant = !fit.has_intercept;
    for (std::size_t j = 0; j < k; j++) {
      if (w.aliased[j]) continue;
      if (!skipped_constant) {
        bool constant = X(0, j) != 0;
        for (std::size_t i = 1; i < n && constant; i++) constant = X(i, j) == X(0, j);
        if (constant) {
          skipped_constant = true;
          continue;
        }
      }
      tested.push_back(j);
    }
    Rectangular R(tested.size(), k);
    for (std::size_t a = 0; a < tested.size(); a++) R(a, tested[a]) = 1;
    const auto wt = wald(fit, R, std::vector<double>(tested.size(), 0.0));
    fit.f_statistic = wt.f_statistic;
    fit.f_df1 = wt.df1;
    fit.f_df2 = wt.df2;
    fit.f_p_value = wt.f_p_value;
  }

  // Likelihood and information criteria.
  double logw = 0;
  if (!sw.empty())
    for (double wi : options.weights) logw += std::log(wi);
  fit.log_likelihood = -0.5 * dn * (std::log(2 * std::numbers::pi) + std::log(fit.rss / dn) + 1) + 0.5 * logw;
  fit.aic = -2 * fit.log_likelihood + 2 * dr;
  fit.bic = -2 * fit.log_likelihood + dr * std::log(dn);
  return fit;
}

struct Prediction {
  std::vector<double> mean;     // x β̂ for each row of the new design
  std::vector<double> se;       // √(x V xᵀ), the standard error of the mean
  std::vector<double> lower, upper;  // the confidence interval of the mean at the fit's level
};

// Predictions at the rows of a new design matrix (the same columns as the fit's X).
inline Prediction predict(const LinearFit& fit, const Rectangular& Xnew) {
  if (Xnew.cols != fit.k) throw std::invalid_argument("tyche::regression::predict: the new design has the fit's columns");
  const double tcrit = probability::StudentT{fit.df_inference}.upper_quantile((1 - fit.confidence) / 2);
  Prediction p;
  for (std::size_t i = 0; i < Xnew.rows; i++) {
    double m = 0, v = 0;
    for (std::size_t a = 0; a < fit.k; a++) {
      if (fit.aliased[a]) continue;
      m += Xnew(i, a) * fit.beta[a];
      for (std::size_t b = 0; b < fit.k; b++)
        if (!fit.aliased[b]) v += Xnew(i, a) * fit.covariance(a, b) * Xnew(i, b);
    }
    const double se = std::sqrt(std::max(v, 0.0));
    p.mean.push_back(m);
    p.se.push_back(se);
    p.lower.push_back(m - tcrit * se);
    p.upper.push_back(m + tcrit * se);
  }
  return p;
}

}  // namespace tyche::regression
