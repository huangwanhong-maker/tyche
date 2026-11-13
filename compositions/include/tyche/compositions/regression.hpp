// Compositions in linear models: a composition as predictor, the reallocation of a fixed total among its
// parts, and a composition as response.
//
// A COMPOSITION AS PREDICTOR. The parts of a composition sum to a constant, so they cannot all enter a
// linear model (their columns with an intercept are collinear), and leaving one out makes every
// coefficient depend on which. In Aitchison's geometry the composition enters through D − 1 orthonormal
// coordinates (Hron, Filzmoser and Thompson 2012, J. Appl. Stat. 39: 1115–1128):
//
//   y = β₀ + Σ_{k=1}^{D−1} γ_k z_k(x) + δᵀ w + ε,       z = ilr(x), w the other covariates,
//
// fitted by tyche's least squares (regression::ols, with its covariances and weights). The fitted
// values do not depend on the basis: another orthonormal basis is z' = R z with R orthogonal, and the
// fit absorbs R. Neither do γᵀz and the CLR COEFFICIENTS b = Ψᵀγ, which write the model as
// ŷ = β₀ + Σ_i b_i ln x_i + δᵀ w with Σ_i b_i = 0. The coefficient of the first pivot coordinate with
// part l first (pivot_effect) is Hron et al.'s measure of part l against the geometric mean of the
// others; it is the l-th pivot rotation of γ, with its covariance rotated alike, so it needs no refit.
//
// REALLOCATION (compositional isotemporal substitution; Dumuid et al. 2018, Stat. Methods Med. Res. 27:
// 3726–3738). With the total fixed (the 1440 minutes of a day), more of one part is less of another.
// From a reference composition x⁰ (usually the sample's centre at the sample's total), the predicted
// change of the outcome when the composition becomes x¹ is
//
//   Δŷ = ŷ(x¹) − ŷ(x⁰) = γᵀ (ilr x¹ − ilr x⁰) = Σ_i b_i ln(x¹_i / x⁰_i),
//
// the same whatever the covariates' values in a model without interactions, with variance dᵀ V d,
// d = ilr x¹ − ilr x⁰ and V the covariance of γ̂ (classical or robust, as fitted), and an interval on
// Student's t with the fit's degrees of freedom. Two moves are offered:
//
//   one for one         Δ units moved from part j to part i, the others held:
//                       x¹_i = x⁰_i + Δ, x¹_j = x⁰_j − Δ;
//   one for remaining   Δ units added to part i and taken from all the others in proportion to their
//                       sizes, so that their ratios are kept (Δ < 0 gives to them): x¹_i = x⁰_i + Δ,
//                       x¹_k = x⁰_k (1 − Δ/(c − x⁰_i)), c = Σ x⁰ (the one-for-remaining reallocation of
//                       Dumuid et al. 2019, Stat. Methods Med. Res. 28: 846–857).
//
// The prediction is nonlinear in Δ: moving Δ from j to i and moving Δ from i to j from the same x⁰ do
// not give changes of opposite sign in general. What is exact is the reverse move: the change from x⁰ to
// x¹ is minus the change from x¹ back to x⁰ (checked).
//
// A COMPOSITION AS RESPONSE (Pawlowsky-Glahn, Egozcue and Tolosana-Delgado 2015, Modelling and Analysis
// of Compositional Data, on linear models in coordinates): each ilr coordinate of the response is
// regressed on the covariates by least squares, which on a common design is the multivariate
// least-squares fit; predictions are carried back by ilr⁻¹. In the simplex the model reads
// y = b₀ ⊕ (w₁ ⊙ b₁) ⊕ … ⊕ ε, each covariate's effect a composition b_a = ilr⁻¹ of its row of
// coefficients (`effects`); these and the predictions do not depend on the basis, the coordinates'
// separate t-tests do.
//
// LIMITS. The compositions must be positive (zeros.hpp first). The inference is that of least squares
// on the coordinates: exact under normal, homoskedastic, independent errors (classical covariance),
// asymptotic with robust covariances.

#pragma once

#include <tyche/compositions/balances.hpp>
#include <tyche/compositions/sample.hpp>
#include <tyche/compositions/simplex.hpp>

#include <tyche/probability/distributions.hpp>
#include <tyche/regression/linear.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace tyche::compositions {

// ---------------------------------------------------------------- a composition as predictor

struct PredictorOptions {
  SignTable partition;               // empty: pivot coordinates of the parts in order
  bool intercept = true;             // a column of ones before the coordinates
  regression::OlsOptions ols;        // weights, covariance (classical, HC0–HC3, cluster), confidence
};

struct CompositionPredictor {
  std::size_t n = 0, parts = 0, covariates = 0;
  bool intercept = true;
  Basis basis;
  regression::LinearFit fit;              // columns: [1], z_1, …, z_{D−1}, then the covariates
  std::vector<double> clr_coefficients;   // b = Ψᵀγ, Σ b_i = 0
  std::vector<double> centre;             // the sample's centre, closed to `total`
  double total = 0;                       // the mean total of the compositions as given

  std::size_t first_coordinate() const { return intercept ? 1 : 0; }
};

// The design matrix of a fitted model at new compositions (rows) and covariates (rows; empty when the
// model has none).
inline Rectangular design(const CompositionPredictor& m, const Rectangular& compositions, const Rectangular& covariates = {}) {
  require(compositions.cols == m.parts, "design", "the compositions have the model's parts");
  require(covariates.cols == m.covariates && (m.covariates == 0 || covariates.rows == compositions.rows), "design",
          "the covariates have the model's columns, one row per composition");
  const auto Z = ilr(compositions, m.basis);
  const std::size_t c0 = m.first_coordinate(), D = m.parts;
  Rectangular X(compositions.rows, c0 + D - 1 + m.covariates);
  for (std::size_t i = 0; i < X.rows; i++) {
    if (m.intercept) X(i, 0) = 1;
    for (std::size_t k = 0; k + 1 < D; k++) X(i, c0 + k) = Z(i, k);
    for (std::size_t a = 0; a < m.covariates; a++) X(i, c0 + D - 1 + a) = covariates(i, a);
  }
  return X;
}

// Fits y on the ilr coordinates of the compositions (n × D, positive, any units) and the covariates
// (n × p, or empty).
inline CompositionPredictor fit_predictor(const Rectangular& compositions, std::span<const double> y, const Rectangular& covariates = {},
                                          const PredictorOptions& o = {}) {
  const std::size_t n = compositions.rows, D = compositions.cols;
  require(n >= 1 && D >= 2, "fit_predictor", "compositions of two or more parts");
  require(y.size() == n, "fit_predictor", "one outcome per composition");
  require(covariates.cols == 0 || covariates.rows == n, "fit_predictor", "one row of covariates per composition, or none");
  CompositionPredictor m;
  m.n = n;
  m.parts = D;
  m.covariates = covariates.cols;
  m.intercept = o.intercept;
  m.basis = o.partition.empty() ? pivot_basis(D) : basis(o.partition);
  require(m.basis.parts == D, "fit_predictor", "the partition has the compositions' parts");
  double sum = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < D; j++) sum += compositions(i, j);
  m.total = sum / static_cast<double>(n);
  m.centre = centre(compositions, m.total);
  m.fit = regression::ols(design(m, compositions, covariates), y, o.ols);
  const std::size_t c0 = m.first_coordinate();
  std::vector<double> gamma(D - 1);
  for (std::size_t k = 0; k + 1 < D; k++) gamma[k] = m.fit.aliased[c0 + k] ? std::numeric_limits<double>::quiet_NaN() : m.fit.beta[c0 + k];
  m.clr_coefficients = calx::linalg::multiply_transpose(m.basis.contrast, gamma);
  return m;
}

// Predictions (with standard errors and intervals of the mean) at new compositions and covariates.
inline regression::Prediction predict(const CompositionPredictor& m, const Rectangular& compositions, const Rectangular& covariates = {}) {
  return regression::predict(m.fit, design(m, compositions, covariates));
}

struct Estimate {
  double estimate = 0, se = 0, t = 0, p = 1, lower = 0, upper = 0;  // p two-sided; the interval at the fit's level
};

namespace detail {

// a γ̂ and √(aᵀ V a) for a combination a of the coordinates' coefficients, with t, p and interval.
inline Estimate combination(const CompositionPredictor& m, std::span<const double> a) {
  const std::size_t c0 = m.first_coordinate(), K = m.parts - 1;
  Estimate e;
  double est = 0, var = 0;
  for (std::size_t k = 0; k < K; k++) {
    if (a[k] == 0) continue;
    if (m.fit.aliased[c0 + k]) return {std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan("")};
    est += a[k] * m.fit.beta[c0 + k];
    for (std::size_t l = 0; l < K; l++)
      if (a[l] != 0) var += a[k] * m.fit.covariance(c0 + k, c0 + l) * a[l];
  }
  e.estimate = est;
  e.se = std::sqrt(std::max(var, 0.0));
  const probability::StudentT tdist{m.fit.df_inference};
  e.t = e.se > 0 ? est / e.se : (est == 0 ? 0 : std::copysign(std::numeric_limits<double>::infinity(), est));
  e.p = e.se > 0 ? std::min(1.0, 2 * tdist.survival(std::fabs(e.t))) : (est == 0 ? 1 : 0);
  const double q = tdist.upper_quantile((1 - m.fit.confidence) / 2);
  e.lower = est - q * e.se;
  e.upper = est + q * e.se;
  return e;
}

}  // namespace detail

// The coefficient of the first pivot coordinate with `part` first: part l against the geometric mean
// of the others (Hron, Filzmoser and Thompson 2012), by rotating the fitted coefficients.
inline Estimate pivot_effect(const CompositionPredictor& m, std::size_t part) {
  require(part < m.parts, "pivot_effect", "the part is one of the model's");
  const auto R = change_of_basis(m.basis, pivot_basis(m.parts, part));
  const auto row = R.row(0);
  return detail::combination(m, row);
}

struct Reallocation {
  std::vector<double> before, after;  // x⁰ and x¹, in the reference's units
  Estimate change;                    // ŷ(x¹) − ŷ(x⁰), its standard error, t, p and interval
};

// The predicted change from one composition to another (any units, the same parts).
inline Reallocation reallocation(const CompositionPredictor& m, std::span<const double> before, std::span<const double> after) {
  require(before.size() == m.parts && after.size() == m.parts, "reallocation", "compositions of the model's parts");
  const auto z0 = ilr(before, m.basis), z1 = ilr(after, m.basis);
  std::vector<double> d(m.parts - 1);
  for (std::size_t k = 0; k < d.size(); k++) d[k] = z1[k] - z0[k];
  Reallocation r;
  r.before.assign(before.begin(), before.end());
  r.after.assign(after.begin(), after.end());
  r.change = detail::combination(m, d);
  return r;
}

// One for one: `delta` units moved from part `from` to part `to` of the reference, the others held.
inline Reallocation one_for_one(const CompositionPredictor& m, std::span<const double> reference, std::size_t to, std::size_t from, double delta) {
  require(reference.size() == m.parts, "one_for_one", "a reference of the model's parts");
  require(to < m.parts && from < m.parts && to != from, "one_for_one", "two different parts");
  require(std::isfinite(delta), "one_for_one", "a finite amount");
  std::vector<double> x1(reference.begin(), reference.end());
  x1[to] += delta;
  x1[from] -= delta;
  require(x1[to] > 0 && x1[from] > 0, "one_for_one", "the move leaves every part positive");
  return reallocation(m, reference, x1);
}

// One for remaining: `delta` units added to `part` and taken from the other parts in proportion to their
// sizes (delta < 0: taken from `part` and given to the others alike).
inline Reallocation one_for_remaining(const CompositionPredictor& m, std::span<const double> reference, std::size_t part, double delta) {
  require(reference.size() == m.parts, "one_for_remaining", "a reference of the model's parts");
  require(part < m.parts, "one_for_remaining", "the part is one of the model's");
  require(std::isfinite(delta), "one_for_remaining", "a finite amount");
  double c = 0;
  for (double v : reference) c += v;
  const double rest = c - reference[part];
  require(rest > 0, "one_for_remaining", "the other parts hold something");
  std::vector<double> x1(reference.begin(), reference.end());
  x1[part] += delta;
  for (std::size_t k = 0; k < m.parts; k++)
    if (k != part) x1[k] *= 1 - delta / rest;
  require(x1[part] > 0 && delta < rest, "one_for_remaining", "the move leaves every part positive");
  return reallocation(m, reference, x1);
}

// ---------------------------------------------------------------- a composition as response

struct ResponseOptions {
  SignTable partition;        // empty: pivot coordinates of the parts in order
  bool intercept = true;      // a column of ones before the covariates
  double confidence = 0.95;
};

struct CompositionResponse {
  std::size_t n = 0, parts = 0, columns = 0;   // columns of the design: [1] and the covariates
  bool intercept = true;
  Basis basis;
  std::vector<regression::LinearFit> fits;     // fits[k]: the k-th ilr coordinate on the design
  Rectangular coefficients;                    // columns × (D − 1): row a, the coordinates of column a's effect
  Rectangular effects;                         // columns × D: ilr⁻¹ of each row of coefficients, closed to 1
  Matrix residual_covariance;                  // (D − 1) × (D − 1), divisor n − columns
  double total = 0;                            // the mean total of the response compositions
};

namespace detail {

inline Rectangular response_design(bool intercept, const Rectangular& covariates, std::size_t rows) {
  Rectangular X(rows, (intercept ? 1 : 0) + covariates.cols);
  for (std::size_t i = 0; i < rows; i++) {
    std::size_t c = 0;
    if (intercept) X(i, c++) = 1;
    for (std::size_t a = 0; a < covariates.cols; a++) X(i, c++) = covariates(i, a);
  }
  return X;
}

}  // namespace detail

// Fits the compositions (n × D, positive) on the covariates (n × p) by least squares on their ilr
// coordinates.
inline CompositionResponse fit_response(const Rectangular& compositions, const Rectangular& covariates, const ResponseOptions& o = {}) {
  const std::size_t n = compositions.rows, D = compositions.cols;
  require(n >= 1 && D >= 2, "fit_response", "compositions of two or more parts");
  require(covariates.cols == 0 || covariates.rows == n, "fit_response", "one row of covariates per composition, or none");
  CompositionResponse r;
  r.n = n;
  r.parts = D;
  r.intercept = o.intercept;
  r.basis = o.partition.empty() ? pivot_basis(D) : basis(o.partition);
  require(r.basis.parts == D, "fit_response", "the partition has the compositions' parts");
  const auto X = detail::response_design(o.intercept, covariates, n);
  r.columns = X.cols;
  require(r.columns >= 1 && n > r.columns, "fit_response", "more compositions than columns of the design, and at least one column");
  double sum = 0;
  for (double v : compositions.a) sum += v;
  r.total = sum / static_cast<double>(n);
  const auto Z = ilr(compositions, r.basis);
  regression::OlsOptions ols;
  ols.confidence = o.confidence;
  r.coefficients = Rectangular(r.columns, D - 1);
  Rectangular residuals(n, D - 1);
  for (std::size_t k = 0; k + 1 < D; k++) {
    const auto zk = Z.column(k);
    r.fits.push_back(regression::ols(X, zk, ols));
    for (std::size_t a = 0; a < r.columns; a++) r.coefficients(a, k) = r.fits.back().beta[a];
    for (std::size_t i = 0; i < n; i++) residuals(i, k) = r.fits.back().residuals[i];
  }
  r.effects = ilr_inverse(r.coefficients, r.basis, 1);
  r.residual_covariance = Matrix(D - 1);
  const double dof = static_cast<double>(n - r.columns);
  for (std::size_t k = 0; k + 1 < D; k++)
    for (std::size_t l = k; l + 1 < D; l++) {
      double s = 0;
      for (std::size_t i = 0; i < n; i++) s += residuals(i, k) * residuals(i, l);
      r.residual_covariance(k, l) = r.residual_covariance(l, k) = s / dof;
    }
  return r;
}

// The fitted compositions at new covariates (rows; empty columns for a model with only an intercept),
// closed to `total` (NaN: the mean total of the fitted sample).
inline Rectangular predict(const CompositionResponse& r, const Rectangular& covariates, double total = std::numeric_limits<double>::quiet_NaN()) {
  require(covariates.cols + (r.intercept ? 1 : 0) == r.columns, "predict", "the covariates have the model's columns");
  const std::size_t rows = covariates.cols == 0 ? std::max<std::size_t>(covariates.rows, 1) : covariates.rows;
  const auto X = detail::response_design(r.intercept, covariates, rows);
  const auto Z = X * r.coefficients;
  return ilr_inverse(Z, r.basis, std::isnan(total) ? r.total : total);
}

}  // namespace tyche::compositions
