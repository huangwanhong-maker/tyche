// Penalised regression: ridge, the lasso and the elastic net, along a path of penalties, for the
// gaussian and binomial families, with K-fold cross-validation.
//
// THE PROBLEM (Friedman, Hastie and Tibshirani 2010, "Regularization paths for generalized linear
// models via coordinate descent", J. Stat. Softw. 33(1), eqs. 1–3 and 13–14). With observation weights
// vᵢ normalised to Σ v = 1 (all 1/N by default), minimise over (β₀, β)
//     gaussian:  ½ Σ vᵢ (yᵢ − β₀ − x̃ᵢᵀβ − oᵢ)² + λ P_α(β),
//     binomial:  −Σ vᵢ [yᵢ ηᵢ − log(1 + e^{ηᵢ})] + λ P_α(β),   ηᵢ = β₀ + x̃ᵢᵀβ + oᵢ, yᵢ ∈ [0, 1],
// with P_α(β) = Σ_j γ_j [½(1 − α) β_j² + α |β_j|]: the lasso at α = 1, ridge at α = 0, the elastic net
// between; γ_j ≥ 0 are penalty factors (glmnet's penalty.factor, rescaled to sum to the number of
// columns as glmnet rescales them; γ_j = 0 leaves a column unpenalised), oᵢ an offset. The columns
// x̃_j are X's centred at their weighted means (when there is an intercept) and, with standardize,
// divided by their weighted standard deviations (divisor Σ v = 1, i.e. N): the penalty then falls on
// the coefficients of standardised columns, and the coefficients returned are converted back to X's
// scale, as glmnet does. scikit-learn's ElasticNet minimises (1/2n)‖y − Xw‖² + a ρ ‖w‖₁ +
// ½ a (1 − ρ)‖w‖² without standardising: λ = a, α = ρ, standardize = false; its LogisticRegression
// with C and l1_ratio ρ is λ = 1/(C N), α = ρ.
//
// THE ALGORITHM. Cyclic coordinate descent (ibid., eqs. 5–10): with r the current residual and
// h_j = Σ uᵢ x̃ᵢⱼ² for working weights u,
//     β_j ← S(Σ uᵢ x̃ᵢⱼ rᵢ + h_j β_j, λ α γ_j) / (h_j + λ (1 − α) γ_j),   S(z, t) = sign(z)(|z| − t)₊,
// the intercept ← its weighted least-squares update, sweeps over all columns followed by sweeps over
// the active set until no coefficient moves (ibid., §2.6, active-set convergence), each λ warm-started
// from the last (§2.5). The path starts at λ_max = max_j |Σ vᵢ x̃ᵢⱼ rᵢ| / (α γ_j), r the residual of the
// unpenalised part (the intercept and γ_j = 0 columns), the least λ at which every penalised β_j is 0
// (with α < 10⁻³ in its place for ridge, as glmnet does), and descends geometrically to
// lambda_min_ratio · λ_max (default 10⁻⁴ if N ≥ p, 10⁻² otherwise, glmnet's). The binomial family adds
// an outer loop of iteratively reweighted least squares (eqs. 15–18): uᵢ = vᵢ pᵢ(1 − pᵢ),
// zᵢ = ηᵢ − oᵢ + (yᵢ − pᵢ)/(pᵢ(1 − pᵢ)), pᵢ kept in [10⁻⁵, 1 − 10⁻⁵] (§3), and the coordinate descent
// on the penalised weighted least squares inside; a step that raises the penalised objective is
// halved (glmnet does not; the safeguard costs nothing when the steps are good).
//
// CONVERGENCE AND ACCURACY. The sweeps stop when max_j h_j Δβ_j² (and Σu Δβ₀²) falls below tolerance
// times the null deviance's scale. Coordinate descent converges only linearly, slowly where active
// columns are correlated, so a small last step does not mean a small error; the sweeps are therefore
// finished exactly: with the active set and its signs fixed, the stationarity conditions are linear
// and are solved by calx's QR (a departure from glmnet, which stops at the sweeps). If a sign would
// change, the step goes only as far as the first coordinate reaching 0 (an active-set step, which
// lowers the objective) and the sweeps go on; the finish is final when no sign changes and every
// inactive coordinate satisfies its condition. Where the finish is not available (more than 400 active
// coordinates, or a singular system: the lasso with more active coordinates than N), the sweeps'
// answer stands, with coefficients good to about the square root of the threshold over the rate of
// convergence. The Karush–Kuhn–Tucker conditions (checked by
// penalized_kkt) hold to that order: for β_j ≠ 0, Σ vᵢ x̃ᵢⱼ ∂ℓᵢ = λγ_j [α sign β_j + (1 − α) β_j]; for
// β_j = 0, |Σ vᵢ x̃ᵢⱼ ∂ℓᵢ| ≤ λ α γ_j. A binomial fit with λ → 0 on separable data has no minimum (the
// coefficients diverge); the path stops before such λ only in the sense that its last fits stop
// converging (converged = false).
//
// RIDGE IN CLOSED FORM. ridge() solves the gaussian problem at α = 0 directly, (X̃ᵀVX̃ + λΓ) β = X̃ᵀV y,
// by calx's Householder QR of [√V X̃; √(λΓ)] (never the normal equations).
//
// CROSS-VALIDATION. cross_validate() fits the path on all data, then on each of K folds' complements
// at the same λ, and measures the held-out loss (squared error for the gaussian, deviance
// −2[y log p + (1 − y) log(1 − p)] for the binomial, or the misclassification rate). Per λ, cvm is the
// weighted mean of the folds' mean losses and cvsd = √(Σ_k W_k (cv_k − cvm)² / Σ W_k / (K − 1)), the
// standard error of that mean, W_k the fold's weight; λ_min minimises cvm, and λ_1se is the largest λ
// with cvm ≤ cvm(λ_min) + cvsd(λ_min), the one-standard-error rule (Hastie, Tibshirani and Friedman
// 2009, The Elements of Statistical Learning, 2nd ed., §7.10). Folds are balanced and deterministic: a
// permutation of the observations by a calx stream of the given seed, observation π(i) in fold i mod K,
// unless the caller gives them.

#pragma once

#include <tyche/regression/linear.hpp>

#include <calx/linalg/qr.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::regression {

enum class PenalizedFamily { gaussian, binomial };

struct PenalizedOptions {
  PenalizedFamily family = PenalizedFamily::gaussian;
  double alpha = 1;                     // the mixing: 1 the lasso, 0 ridge, between the elastic net
  std::vector<double> lambda;           // the path, decreasing; empty: lambda_count values from λ_max
  std::size_t lambda_count = 100;
  double lambda_min_ratio = -1;         // < 0: 10⁻⁴ if N ≥ p, else 10⁻² (glmnet's)
  bool standardize = true;              // penalise the coefficients of unit-variance columns
  bool intercept = true;                // an unpenalised intercept (X carries no column of ones then)
  std::vector<double> weights;          // observation weights > 0 (normalised to sum to 1); empty: equal
  std::vector<double> penalty_factor;   // γ_j ≥ 0 per column (rescaled to sum to p); empty: all 1
  std::vector<double> offset;           // oᵢ in the linear predictor; empty: 0
  double tolerance = 1e-12;             // of a sweep, relative to the null deviance
  std::size_t max_passes = 100000;      // sweeps over the coordinates, for each λ
};

struct PenalizedFit {
  PenalizedFamily family = PenalizedFamily::gaussian;
  double alpha = 1, lambda = 0;
  double intercept = 0;                 // β₀ on X's scale
  std::vector<double> beta;             // one per column of X, on X's scale
  std::size_t df = 0;                   // non-zero coefficients
  double deviance = 0, null_deviance = 0, deviance_ratio = 0;  // weighted (by the given weights), 1 − D/D₀
  double objective = 0;                 // the penalised objective (normalised weights, standardised β)
  std::size_t passes = 0;               // coordinate sweeps (all IRLS steps together)
  bool converged = false;
};

struct PenalizedPath {
  PenalizedFamily family = PenalizedFamily::gaussian;
  double alpha = 1;
  std::vector<double> lambda;           // decreasing
  std::vector<PenalizedFit> fits;       // one per λ
  double lambda_max = 0;                // the least λ with every penalised β zero
  bool converged = false;               // every fit converged
};

namespace penalized_detail {

inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::regression::penalized: ") + what);
}

inline double soft(double z, double t) { return z > t ? z - t : z < -t ? z + t : 0.0; }
inline double logistic(double eta) { return 1 / (1 + std::exp(-eta)); }
// log(1 + e^η), without overflow.
inline double log1pexp(double eta) { return eta > 0 ? eta + std::log1p(std::exp(-eta)) : std::log1p(std::exp(eta)); }

// The standardised problem: columns x̃_j (column-major), centres c_j and scales s_j, weights v (Σ = 1).
struct Problem {
  std::size_t n = 0, p = 0;
  std::vector<double> x;        // x̃, column-major: x[j n + i]
  std::vector<double> centre, scale;
  std::vector<bool> constant;   // a column with no variation (its β is 0)
  std::vector<double> v, y, offset, gamma;
  double weight_sum = 1;        // Σ of the given weights (1 per observation by default)
  bool intercept = true;
  PenalizedFamily family = PenalizedFamily::gaussian;
  double alpha = 1;
  double null_scale = 1;        // the null deviance per unit weight, the scale of the tolerance
};

inline Problem setup(const Rectangular& X, std::span<const double> y, const PenalizedOptions& o) {
  Problem P;
  P.n = X.rows;
  P.p = X.cols;
  P.family = o.family;
  P.alpha = o.alpha;
  P.intercept = o.intercept;
  const std::size_t n = P.n, p = P.p;
  require(n >= 1, "at least one observation");
  require(y.size() == n, "y has one entry per row of X");
  require(o.alpha >= 0 && o.alpha <= 1, "0 ≤ alpha ≤ 1");
  require(o.weights.empty() || o.weights.size() == n, "one weight per observation");
  require(o.offset.empty() || o.offset.size() == n, "one offset per observation");
  require(o.penalty_factor.empty() || o.penalty_factor.size() == p, "one penalty factor per column");
  P.v.assign(n, 1.0);
  if (!o.weights.empty()) {
    for (std::size_t i = 0; i < n; i++) {
      require(o.weights[i] > 0 && std::isfinite(o.weights[i]), "weights positive and finite");
      P.v[i] = o.weights[i];
    }
  }
  double sw = 0;
  for (const double w : P.v) sw += w;
  P.weight_sum = sw;
  for (double& w : P.v) w /= sw;
  P.y.assign(y.begin(), y.end());
  for (std::size_t i = 0; i < n; i++) {
    require(std::isfinite(y[i]), "finite responses");
    if (o.family == PenalizedFamily::binomial) require(y[i] >= 0 && y[i] <= 1, "binomial responses in [0, 1]");
  }
  P.offset = o.offset.empty() ? std::vector<double>(n, 0.0) : o.offset;
  P.gamma.assign(p, 1.0);
  if (!o.penalty_factor.empty()) {
    double s = 0;
    for (std::size_t j = 0; j < p; j++) {
      require(o.penalty_factor[j] >= 0 && std::isfinite(o.penalty_factor[j]), "penalty factors non-negative and finite");
      s += o.penalty_factor[j];
    }
    require(s > 0, "some penalty factor positive");
    for (std::size_t j = 0; j < p; j++) P.gamma[j] = o.penalty_factor[j] * static_cast<double>(p) / s;
  }
  P.x.assign(n * p, 0.0);
  P.centre.assign(p, 0.0);
  P.scale.assign(p, 1.0);
  P.constant.assign(p, false);
  for (std::size_t j = 0; j < p; j++) {
    double c = 0;
    if (o.intercept)
      for (std::size_t i = 0; i < n; i++) c += P.v[i] * X(i, j);
    double ss = 0;
    for (std::size_t i = 0; i < n; i++) ss += P.v[i] * (X(i, j) - c) * (X(i, j) - c);
    const double s = std::sqrt(ss);
    P.constant[j] = !(s > 0);
    P.centre[j] = c;
    P.scale[j] = o.standardize && s > 0 ? s : 1.0;
    for (std::size_t i = 0; i < n; i++) P.x[j * n + i] = P.constant[j] ? 0.0 : (X(i, j) - c) / P.scale[j];
  }
  // The null deviance per unit weight (the scale of the convergence tolerance).
  double ybar = 0;
  for (std::size_t i = 0; i < n; i++) ybar += P.v[i] * y[i];
  double d0 = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (o.family == PenalizedFamily::gaussian) d0 += P.v[i] * (y[i] - ybar) * (y[i] - ybar);
    else {
      const double q = std::clamp(ybar, 1e-5, 1 - 1e-5);
      d0 += -2 * P.v[i] * (y[i] * std::log(q) + (1 - y[i]) * std::log(1 - q));
    }
  }
  P.null_scale = d0 > 0 ? d0 : 1.0;
  return P;
}

// The state of a fit in standardised coordinates.
struct State {
  double b0 = 0;
  std::vector<double> b;
  std::size_t passes = 0;
  bool converged = false;
};

// Coordinate descent on ½ Σ uᵢ (zᵢ − b₀ − x̃ᵢᵀb)² + λ P_α(b), from the state given; r is z − b₀ − x̃b on
// entry and on exit. frozen: the penalised coordinates are held at 0 (λ = ∞).
inline void descend(const Problem& P, std::span<const double> u, std::vector<double>& r, State& s, double lambda, bool frozen,
                    std::size_t max_passes, double tol) {
  const std::size_t n = P.n, p = P.p;
  std::vector<double> h(p, 0.0);
  double usum = 0;
  for (std::size_t i = 0; i < n; i++) usum += u[i];
  for (std::size_t j = 0; j < p; j++) {
    if (P.constant[j]) continue;
    const double* xj = &P.x[j * n];
    double hj = 0;
    for (std::size_t i = 0; i < n; i++) hj += u[i] * xj[i] * xj[i];
    h[j] = hj;
  }
  auto update = [&](std::size_t j) -> double {
    if (P.constant[j]) return 0;
    const bool penalised = P.gamma[j] > 0;
    if (penalised && frozen) return 0;
    const double* xj = &P.x[j * n];
    double g = 0;
    for (std::size_t i = 0; i < n; i++) g += u[i] * xj[i] * r[i];
    g += h[j] * s.b[j];
    const double l1 = penalised ? lambda * P.alpha * P.gamma[j] : 0.0;
    const double l2 = penalised ? lambda * (1 - P.alpha) * P.gamma[j] : 0.0;
    const double denom = h[j] + l2;
    const double nb = denom > 0 ? soft(g, l1) / denom : 0.0;
    const double d = nb - s.b[j];
    if (d == 0) return 0;
    for (std::size_t i = 0; i < n; i++) r[i] -= d * xj[i];
    s.b[j] = nb;
    return h[j] * d * d;
  };
  auto update_intercept = [&]() -> double {
    if (!P.intercept || !(usum > 0)) return 0;
    double g = 0;
    for (std::size_t i = 0; i < n; i++) g += u[i] * r[i];
    const double d = g / usum;
    if (d == 0) return 0;
    for (std::size_t i = 0; i < n; i++) r[i] -= d;
    s.b0 += d;
    return usum * d * d;
  };
  // The exact finish on the active set A (the non-zero penalised coordinates, the unpenalised ones and
  // the intercept): with the signs σ of A fixed, the stationarity conditions are linear,
  //     (X̃_Aᵀ U X̃_A + Λ₂) c = X̃_Aᵀ U z − Λ₁ σ,   Λ₁ = diag(λαγ), Λ₂ = diag(λ(1 − α)γ),
  // solved by calx's Householder QR of [√U X̃_A; √Λ₂] (Rᵀ R c = rhs, by two triangular solves). It is accepted only if no sign changes; it is the minimiser if, besides, every coordinate
  // outside A satisfies |x̃ⱼᵀ U r| ≤ λαγⱼ. Coordinate descent alone converges linearly, slowly on
  // correlated columns; the finish makes the solution exact to rounding once the active set is found.
  enum class Finish { exact, failed, unavailable };
  auto refine = [&]() -> Finish {
    std::vector<std::size_t> A;
    for (std::size_t j = 0; j < p; j++) {
      if (P.constant[j]) continue;
      const bool penalised = P.gamma[j] > 0;
      if (!penalised || (!frozen && s.b[j] != 0)) A.push_back(j);
    }
    const std::size_t off = P.intercept ? 1 : 0, q = A.size() + off;
    if (q == 0) return Finish::exact;
    if (q > 400 || static_cast<double>(n) * static_cast<double>(q) * static_cast<double>(q) > 4e8) return Finish::unavailable;
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; i++) z[i] = r[i] + s.b0;
    for (const std::size_t j : A)
      for (std::size_t i = 0; i < n; i++) z[i] += P.x[j * n + i] * s.b[j];
    Rectangular M(n + q, q);
    std::vector<double> rhs(q, 0.0);
    for (std::size_t i = 0; i < n; i++) {
      const double su = std::sqrt(u[i]);
      if (off) {
        M(i, 0) = su;
        rhs[0] += u[i] * z[i];
      }
      for (std::size_t k = 0; k < A.size(); k++) {
        const double x = P.x[A[k] * n + i];
        M(i, k + off) = su * x;
        rhs[k + off] += u[i] * x * z[i];
      }
    }
    for (std::size_t k = 0; k < A.size(); k++) {
      const std::size_t j = A[k];
      if (!(P.gamma[j] > 0)) continue;
      M(n + k + off, k + off) = std::sqrt(lambda * (1 - P.alpha) * P.gamma[j]);
      rhs[k + off] -= lambda * P.alpha * P.gamma[j] * (s.b[j] > 0 ? 1.0 : -1.0);
    }
    const auto f = calx::linalg::qr(M, false);
    if (f.rank < q) return Finish::unavailable;
    const auto R = calx::linalg::r_factor(f);
    const auto c = calx::linalg::solve_upper(R, calx::linalg::solve_upper_transpose(R, rhs));
    // If a sign would change, step from b towards c only as far as the first coordinate reaching 0 (the
    // objective, a convex quadratic on the orthant of b's signs, decreases along the step), set it to 0,
    // and let the sweeps go on: an active-set step.
    double t = 1;
    std::size_t hit = A.size();
    for (std::size_t k = 0; k < A.size(); k++) {
      const std::size_t j = A[k];
      if (P.gamma[j] > 0 && P.alpha > 0 && !(c[k + off] * s.b[j] > 0)) {
        const double tk = s.b[j] / (s.b[j] - c[k + off]);
        if (tk < t) {
          t = tk;
          hit = k;
        }
      }
    }
    if (off) s.b0 += t * (c[0] - s.b0);
    for (std::size_t k = 0; k < A.size(); k++) s.b[A[k]] = k == hit ? 0.0 : s.b[A[k]] + t * (c[k + off] - s.b[A[k]]);
    for (std::size_t i = 0; i < n; i++) {
      double fit = s.b0;
      for (const std::size_t j : A) fit += P.x[j * n + i] * s.b[j];
      r[i] = z[i] - fit;
    }
    if (hit < A.size()) return Finish::failed;
    if (frozen) return Finish::exact;
    for (std::size_t j = 0; j < p; j++) {
      if (P.constant[j] || !(P.gamma[j] > 0) || s.b[j] != 0) continue;
      double g = 0;
      for (std::size_t i = 0; i < n; i++) g += u[i] * P.x[j * n + i] * r[i];
      if (std::fabs(g) > lambda * P.alpha * P.gamma[j] * (1 + 1e-10) + 1e-15 * std::sqrt(P.null_scale)) return Finish::failed;
    }
    return Finish::exact;
  };
  const double thr = tol * P.null_scale;
  s.converged = false;
  std::size_t passes = 0, finishes = 0;
  std::vector<std::size_t> active;
  while (passes < max_passes) {
    // A full sweep.
    double change = update_intercept();
    for (std::size_t j = 0; j < p; j++) change = std::max(change, update(j));
    passes++;
    if (change < thr) {
      // Converged by the sweeps' measure: finish exactly on the active set if it is right; if not
      // (a sign changed, or an inactive coordinate wants in), sweep on from the partial step and try
      // again (100 times at most). Where the finish is not available (an active set larger than 400,
      // or a singular system), the sweeps' answer stands.
      if (finishes >= 100 || refine() != Finish::failed) {
        s.converged = true;
        break;
      }
      finishes++;
      continue;
    }
    // Sweeps over the active set until they settle.
    active.clear();
    for (std::size_t j = 0; j < p; j++)
      if (s.b[j] != 0) active.push_back(j);
    while (passes < max_passes) {
      double c = update_intercept();
      for (const std::size_t j : active) c = std::max(c, update(j));
      passes++;
      if (c < thr) break;
    }
  }
  s.passes += passes;
}

inline double penalty(const Problem& P, const State& s, double lambda) {
  double pen = 0;
  for (std::size_t j = 0; j < P.p; j++)
    pen += P.gamma[j] * (0.5 * (1 - P.alpha) * s.b[j] * s.b[j] + P.alpha * std::fabs(s.b[j]));
  return lambda * pen;
}

inline std::vector<double> linear_predictor(const Problem& P, const State& s) {
  std::vector<double> eta(P.n, s.b0);
  for (std::size_t i = 0; i < P.n; i++) eta[i] += P.offset[i];
  for (std::size_t j = 0; j < P.p; j++) {
    if (s.b[j] == 0) continue;
    const double* xj = &P.x[j * P.n];
    for (std::size_t i = 0; i < P.n; i++) eta[i] += s.b[j] * xj[i];
  }
  return eta;
}

// The negative log-likelihood part of the objective (normalised weights).
inline double loss(const Problem& P, std::span<const double> eta) {
  double l = 0;
  for (std::size_t i = 0; i < P.n; i++) {
    if (P.family == PenalizedFamily::gaussian) l += 0.5 * P.v[i] * (P.y[i] - eta[i]) * (P.y[i] - eta[i]);
    else l += P.v[i] * (log1pexp(eta[i]) - P.y[i] * eta[i]);
  }
  return l;
}

// Fits at one λ from the state given (warm start).
inline void solve(const Problem& P, State& s, double lambda, bool frozen, std::size_t max_passes, double tol) {
  const std::size_t n = P.n;
  if (P.family == PenalizedFamily::gaussian) {
    auto eta = linear_predictor(P, s);
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; i++) r[i] = P.y[i] - eta[i];
    descend(P, P.v, r, s, lambda, frozen, max_passes, tol);
    return;
  }
  // Binomial: IRLS around coordinate descent.
  std::vector<double> u(n), r(n);
  auto eta = linear_predictor(P, s);
  double objective = loss(P, eta) + (frozen ? 0.0 : penalty(P, s, lambda));
  bool all_converged = true;
  std::size_t used = 0;
  for (int outer = 0; outer < 200; outer++) {
    for (std::size_t i = 0; i < n; i++) {
      const double pr = std::clamp(logistic(eta[i]), 1e-5, 1 - 1e-5);
      const double w = pr * (1 - pr);
      u[i] = P.v[i] * w;
      const double z = eta[i] - P.offset[i] + (P.y[i] - pr) / w;
      r[i] = z - (eta[i] - P.offset[i]);
    }
    const State before = s;
    descend(P, u, r, s, lambda, frozen, max_passes > used ? max_passes - used : 1, tol);
    used += s.passes - before.passes;
    all_converged = all_converged && s.converged;
    auto eta_new = linear_predictor(P, s);
    double obj = loss(P, eta_new) + (frozen ? 0.0 : penalty(P, s, lambda));
    // Halve the step while the penalised objective rises.
    for (int half = 0; half < 30 && obj > objective + 1e-15 * (1 + std::fabs(objective)); half++) {
      s.b0 = 0.5 * (s.b0 + before.b0);
      for (std::size_t j = 0; j < P.p; j++) s.b[j] = 0.5 * (s.b[j] + before.b[j]);
      eta_new = linear_predictor(P, s);
      obj = loss(P, eta_new) + (frozen ? 0.0 : penalty(P, s, lambda));
    }
    double move = s.b0 - before.b0;
    move *= move;
    for (std::size_t j = 0; j < P.p; j++) move = std::max(move, (s.b[j] - before.b[j]) * (s.b[j] - before.b[j]));
    const bool settled = std::fabs(obj - objective) <= tol * (std::fabs(objective) + 1e-3) && move <= tol * 1e2;
    eta = std::move(eta_new);
    objective = obj;
    if (settled) break;
    if (used >= max_passes) {
      all_converged = false;
      break;
    }
  }
  s.converged = all_converged;
}

inline PenalizedFit report(const Problem& P, const State& s, double lambda, std::size_t passes) {
  PenalizedFit f;
  f.family = P.family;
  f.alpha = P.alpha;
  f.lambda = lambda;
  f.beta.assign(P.p, 0.0);
  double b0 = s.b0;
  for (std::size_t j = 0; j < P.p; j++) {
    f.beta[j] = s.b[j] / P.scale[j];
    b0 -= f.beta[j] * P.centre[j];
    if (s.b[j] != 0) f.df++;
  }
  f.intercept = b0;
  const auto eta = linear_predictor(P, s);
  f.objective = loss(P, eta) + penalty(P, s, lambda);
  // Deviances with the given (unnormalised) weights.
  double ybar = 0;
  for (std::size_t i = 0; i < P.n; i++) ybar += P.v[i] * P.y[i];
  double dev = 0, dev0 = 0;
  for (std::size_t i = 0; i < P.n; i++) {
    const double w = P.v[i] * P.weight_sum;
    if (P.family == PenalizedFamily::gaussian) {
      dev += w * (P.y[i] - eta[i]) * (P.y[i] - eta[i]);
      dev0 += w * (P.y[i] - ybar) * (P.y[i] - ybar);
    } else {
      const double pr = std::clamp(logistic(eta[i]), 1e-15, 1 - 1e-15), q = std::clamp(ybar, 1e-15, 1 - 1e-15);
      auto xlogy = [](double a, double b) { return a == 0 ? 0.0 : a * std::log(b); };
      dev += -2 * w * (xlogy(P.y[i], pr) + xlogy(1 - P.y[i], 1 - pr));
      dev0 += -2 * w * (xlogy(P.y[i], q) + xlogy(1 - P.y[i], 1 - q));
    }
  }
  f.deviance = dev;
  f.null_deviance = dev0;
  f.deviance_ratio = dev0 > 0 ? 1 - dev / dev0 : 0.0;
  f.passes = passes;
  f.converged = s.converged;
  return f;
}

// The gradient of the loss at the state: gⱼ = Σ vᵢ x̃ᵢⱼ (yᵢ − μᵢ), μ the mean (η, or the probability).
inline std::vector<double> score(const Problem& P, const State& s) {
  const auto eta = linear_predictor(P, s);
  std::vector<double> g(P.p, 0.0);
  for (std::size_t j = 0; j < P.p; j++) {
    const double* xj = &P.x[j * P.n];
    double v = 0;
    for (std::size_t i = 0; i < P.n; i++) {
      const double mu = P.family == PenalizedFamily::gaussian ? eta[i] : logistic(eta[i]);
      v += P.v[i] * xj[i] * (P.y[i] - mu);
    }
    g[j] = v;
  }
  return g;
}

// The state (standardised coordinates) of a fit reported on X's scale.
inline State state_of(const Problem& P, const PenalizedFit& f) {
  State s;
  s.b.assign(P.p, 0.0);
  s.b0 = f.intercept;
  for (std::size_t j = 0; j < P.p; j++) {
    s.b[j] = f.beta[j] * P.scale[j];
    s.b0 += f.beta[j] * P.centre[j];
  }
  return s;
}

}  // namespace penalized_detail

// The path of elastic-net fits over a decreasing sequence of λ.
inline PenalizedPath elastic_net_path(const Rectangular& X, std::span<const double> y, const PenalizedOptions& o = {}) {
  using namespace penalized_detail;
  const Problem P = setup(X, y, o);
  State s;
  s.b.assign(P.p, 0.0);
  // The unpenalised part first: the intercept (and γ = 0 columns) at λ = ∞.
  if (P.family == PenalizedFamily::binomial && P.intercept) {
    double ybar = 0;
    for (std::size_t i = 0; i < P.n; i++) ybar += P.v[i] * P.y[i];
    const double q = std::clamp(ybar, 1e-5, 1 - 1e-5);
    s.b0 = std::log(q / (1 - q));
  }
  solve(P, s, 0, true, o.max_passes, o.tolerance);
  const auto g = score(P, s);
  double lmax = 0;
  const double a = std::max(o.alpha, 1e-3);
  for (std::size_t j = 0; j < P.p; j++)
    if (P.gamma[j] > 0 && !P.constant[j]) lmax = std::max(lmax, std::fabs(g[j]) / (a * P.gamma[j]));
  PenalizedPath path;
  path.family = o.family;
  path.alpha = o.alpha;
  path.lambda_max = lmax;
  if (!o.lambda.empty()) {
    path.lambda = o.lambda;
    for (std::size_t k = 0; k < path.lambda.size(); k++) {
      require(path.lambda[k] >= 0 && std::isfinite(path.lambda[k]), "λ finite and ≥ 0");
      require(k == 0 || path.lambda[k] <= path.lambda[k - 1], "λ decreasing");
    }
  } else {
    const double ratio = o.lambda_min_ratio > 0 ? o.lambda_min_ratio : (P.n >= P.p ? 1e-4 : 1e-2);
    const std::size_t K = std::max<std::size_t>(o.lambda_count, 1);
    const double top = lmax > 0 ? lmax : 1.0;
    for (std::size_t k = 0; k < K; k++)
      path.lambda.push_back(K == 1 ? top : top * std::pow(ratio, static_cast<double>(k) / static_cast<double>(K - 1)));
  }
  path.converged = true;
  for (const double lambda : path.lambda) {
    const std::size_t before = s.passes;
    solve(P, s, lambda, false, o.max_passes, o.tolerance);
    path.fits.push_back(report(P, s, lambda, s.passes - before));
    path.converged = path.converged && s.converged;
  }
  return path;
}

// The elastic-net fit at one λ (by the path from λ_max down to it, for the warm starts: about twenty
// geometric steps).
inline PenalizedFit elastic_net(const Rectangular& X, std::span<const double> y, double lambda, const PenalizedOptions& o = {}) {
  using namespace penalized_detail;
  require(lambda >= 0 && std::isfinite(lambda), "λ finite and ≥ 0");
  PenalizedOptions first = o;
  first.lambda = {std::numeric_limits<double>::max()};
  first.max_passes = o.max_passes;
  // λ_max from a one-point path at a huge λ (every penalised coefficient zero).
  const double lmax = elastic_net_path(X, y, first).lambda_max;
  PenalizedOptions q = o;
  q.lambda.clear();
  if (lmax > lambda && lambda > 0) {
    const std::size_t steps = 20;
    for (std::size_t k = 0; k < steps; k++) q.lambda.push_back(lmax * std::pow(lambda / lmax, static_cast<double>(k) / static_cast<double>(steps)));
  }
  q.lambda.push_back(lambda);
  auto path = elastic_net_path(X, y, q);
  return path.fits.back();
}

// The lasso at one λ (α = 1).
inline PenalizedFit lasso(const Rectangular& X, std::span<const double> y, double lambda, PenalizedOptions o = {}) {
  o.alpha = 1;
  return elastic_net(X, y, lambda, o);
}

// Ridge regression at one λ in closed form (gaussian family, α = 0).
inline PenalizedFit ridge(const Rectangular& X, std::span<const double> y, double lambda, PenalizedOptions o = {}) {
  using namespace penalized_detail;
  require(o.family == PenalizedFamily::gaussian, "ridge in closed form is gaussian");
  require(lambda >= 0 && std::isfinite(lambda), "λ finite and ≥ 0");
  o.alpha = 0;
  const Problem P = setup(X, y, o);
  const std::size_t n = P.n, p = P.p;
  // z = y − o, centred when there is an intercept.
  std::vector<double> z(n);
  double zbar = 0;
  for (std::size_t i = 0; i < n; i++) {
    z[i] = P.y[i] - P.offset[i];
    zbar += P.v[i] * z[i];
  }
  if (!P.intercept) zbar = 0;
  Rectangular A(n + p, p);
  std::vector<double> rhs(n + p, 0.0);
  for (std::size_t i = 0; i < n; i++) {
    const double sv = std::sqrt(P.v[i]);
    for (std::size_t j = 0; j < p; j++) A(i, j) = sv * P.x[j * n + i];
    rhs[i] = sv * (z[i] - zbar);
  }
  for (std::size_t j = 0; j < p; j++) A(n + j, j) = std::sqrt(lambda * P.gamma[j]);
  State s;
  s.b = calx::linalg::least_squares(A, rhs);
  for (std::size_t j = 0; j < p; j++)
    if (P.constant[j]) s.b[j] = 0;
  s.b0 = zbar;
  s.converged = true;
  if (P.intercept) {
    // With γ_j = 0 columns the centred design keeps Σ v x̃ = 0, so the intercept is z̄ exactly.
    double d = 0;
    for (std::size_t i = 0; i < n; i++) {
      double fit = 0;
      for (std::size_t j = 0; j < p; j++) fit += P.x[j * n + i] * s.b[j];
      d += P.v[i] * (z[i] - fit);
    }
    s.b0 = d;
  }
  return report(P, s, lambda, 0);
}

// The linear predictor β₀ + Xβ + o of a fit at new rows (response = true: the mean, μ = η for the
// gaussian, the probability for the binomial).
inline std::vector<double> predict(const PenalizedFit& f, const Rectangular& X, std::span<const double> offset = {}, bool response = true) {
  penalized_detail::require(X.cols == f.beta.size(), "X has the fit's columns");
  penalized_detail::require(offset.empty() || offset.size() == X.rows, "one offset per row");
  std::vector<double> eta(X.rows, f.intercept);
  for (std::size_t i = 0; i < X.rows; i++) {
    for (std::size_t j = 0; j < X.cols; j++) eta[i] += X(i, j) * f.beta[j];
    if (!offset.empty()) eta[i] += offset[i];
    if (response && f.family == PenalizedFamily::binomial) eta[i] = penalized_detail::logistic(eta[i]);
  }
  return eta;
}

// The largest violation of the Karush–Kuhn–Tucker conditions of a fit (on the standardised scale,
// relative to λ): for β_j ≠ 0, |gⱼ − λγⱼ[α sign βⱼ + (1 − α)βⱼ]|; for βⱼ = 0, (|gⱼ| − λαγⱼ)₊; for
// unpenalised and intercept coordinates |gⱼ|; divided by max(λ, 10⁻¹²).
inline double penalized_kkt(const Rectangular& X, std::span<const double> y, const PenalizedFit& f, const PenalizedOptions& o = {}) {
  using namespace penalized_detail;
  PenalizedOptions q = o;
  q.family = f.family;
  q.alpha = f.alpha;
  const Problem P = setup(X, y, q);
  const State s = state_of(P, f);
  const auto g = score(P, s);
  double worst = 0;
  const double lambda = f.lambda;
  for (std::size_t j = 0; j < P.p; j++) {
    if (P.constant[j]) continue;
    double v;
    if (P.gamma[j] == 0) v = std::fabs(g[j]);
    else if (s.b[j] != 0) v = std::fabs(g[j] - lambda * P.gamma[j] * (P.alpha * (s.b[j] > 0 ? 1.0 : -1.0) + (1 - P.alpha) * s.b[j]));
    else v = std::max(0.0, std::fabs(g[j]) - lambda * P.alpha * P.gamma[j]);
    worst = std::max(worst, v);
  }
  if (P.intercept) {
    const auto eta = linear_predictor(P, s);
    double g0 = 0;
    for (std::size_t i = 0; i < P.n; i++) g0 += P.v[i] * (P.y[i] - (P.family == PenalizedFamily::gaussian ? eta[i] : logistic(eta[i])));
    worst = std::max(worst, std::fabs(g0));
  }
  return worst / std::max(lambda, 1e-12);
}

// ---------------------------------------------------------------------------------------------
// Cross-validation.

enum class CvMeasure { automatic, mse, deviance, misclassification };

struct CvOptions {
  std::size_t folds = 10;
  std::uint64_t seed = 1;               // of the permutation that makes the folds
  std::vector<std::size_t> fold;        // given folds (0..K−1, one per observation); empty: made here
  CvMeasure measure = CvMeasure::automatic;  // automatic: mse (gaussian), deviance (binomial)
};

struct CrossValidation {
  PenalizedPath path;                   // the fits on all the data
  std::vector<double> mean_error;       // cvm, per λ
  std::vector<double> standard_error;   // cvsd, per λ
  std::size_t index_min = 0, index_1se = 0;
  double lambda_min = 0, lambda_1se = 0;
  std::vector<std::size_t> fold;        // the fold of each observation
  CvMeasure measure = CvMeasure::mse;
};

// Balanced deterministic folds: observation π(i) in fold i mod K, π a permutation by calx's stream.
inline std::vector<std::size_t> make_folds(std::size_t n, std::size_t K, std::uint64_t seed) {
  std::vector<std::size_t> perm(n), fold(n);
  for (std::size_t i = 0; i < n; i++) perm[i] = i;
  calx::random::Stream rng(seed, 0x464f4c44u, 0, 0);  // "FOLD"
  for (std::size_t i = n; i > 1; i--) std::swap(perm[i - 1], perm[rng.below(i)]);
  for (std::size_t i = 0; i < n; i++) fold[perm[i]] = i % K;
  return fold;
}

inline CrossValidation cross_validate(const Rectangular& X, std::span<const double> y, const PenalizedOptions& o = {}, const CvOptions& cv = {}) {
  using namespace penalized_detail;
  const std::size_t n = X.rows;
  require(y.size() == n, "y has one entry per row of X");
  CrossValidation out;
  out.measure = cv.measure == CvMeasure::automatic ? (o.family == PenalizedFamily::gaussian ? CvMeasure::mse : CvMeasure::deviance) : cv.measure;
  require(!(out.measure == CvMeasure::misclassification && o.family == PenalizedFamily::gaussian), "misclassification needs the binomial family");
  out.path = elastic_net_path(X, y, o);
  const std::size_t L = out.path.lambda.size();
  std::size_t K = cv.folds;
  if (!cv.fold.empty()) {
    require(cv.fold.size() == n, "one fold per observation");
    K = *std::max_element(cv.fold.begin(), cv.fold.end()) + 1;
    out.fold = cv.fold;
  } else {
    require(K >= 2 && K <= n, "2 ≤ folds ≤ n");
    out.fold = make_folds(n, K, cv.seed);
  }
  std::vector<std::vector<double>> err(K, std::vector<double>(L, 0.0));
  std::vector<double> W(K, 0.0);
  for (std::size_t k = 0; k < K; k++) {
    std::vector<std::size_t> train, test;
    for (std::size_t i = 0; i < n; i++) (out.fold[i] == k ? test : train).push_back(i);
    require(!test.empty() && !train.empty(), "every fold holds some observations and leaves some");
    Rectangular Xt(train.size(), X.cols);
    std::vector<double> yt(train.size());
    PenalizedOptions q = o;
    q.lambda = out.path.lambda;
    q.weights.clear();
    q.offset.clear();
    for (std::size_t r = 0; r < train.size(); r++) {
      for (std::size_t j = 0; j < X.cols; j++) Xt(r, j) = X(train[r], j);
      yt[r] = y[train[r]];
      if (!o.weights.empty()) q.weights.push_back(o.weights[train[r]]);
      if (!o.offset.empty()) q.offset.push_back(o.offset[train[r]]);
    }
    const auto fp = elastic_net_path(Xt, yt, q);
    double wk = 0;
    for (const std::size_t i : test) wk += o.weights.empty() ? 1.0 : o.weights[i];
    W[k] = wk;
    for (std::size_t l = 0; l < L; l++) {
      const auto& f = fp.fits[l];
      double e = 0;
      for (const std::size_t i : test) {
        double eta = f.intercept + (o.offset.empty() ? 0.0 : o.offset[i]);
        for (std::size_t j = 0; j < X.cols; j++) eta += X(i, j) * f.beta[j];
        const double w = o.weights.empty() ? 1.0 : o.weights[i];
        double lossi = 0;
        if (o.family == PenalizedFamily::gaussian) lossi = (y[i] - eta) * (y[i] - eta);
        else {
          const double pr = std::clamp(logistic(eta), 1e-5, 1 - 1e-5);
          if (out.measure == CvMeasure::deviance) lossi = -2 * (y[i] * std::log(pr) + (1 - y[i]) * std::log(1 - pr));
          else if (out.measure == CvMeasure::mse) lossi = (y[i] - pr) * (y[i] - pr);
          else lossi = std::fabs(y[i] - (pr > 0.5 ? 1.0 : 0.0));
        }
        e += w * lossi;
      }
      err[k][l] = e / wk;
    }
  }
  double Wsum = 0;
  for (const double w : W) Wsum += w;
  out.mean_error.assign(L, 0.0);
  out.standard_error.assign(L, 0.0);
  for (std::size_t l = 0; l < L; l++) {
    double m = 0;
    for (std::size_t k = 0; k < K; k++) m += W[k] * err[k][l];
    m /= Wsum;
    double v = 0;
    for (std::size_t k = 0; k < K; k++) v += W[k] * (err[k][l] - m) * (err[k][l] - m);
    out.mean_error[l] = m;
    out.standard_error[l] = std::sqrt(v / Wsum / static_cast<double>(K - 1));
  }
  std::size_t best = 0;
  for (std::size_t l = 1; l < L; l++)
    if (out.mean_error[l] < out.mean_error[best]) best = l;
  out.index_min = best;
  const double limit = out.mean_error[best] + out.standard_error[best];
  std::size_t one = best;
  for (std::size_t l = 0; l <= best; l++)
    if (out.mean_error[l] <= limit) {
      one = l;
      break;
    }
  out.index_1se = one;
  out.lambda_min = out.path.lambda[best];
  out.lambda_1se = out.path.lambda[one];
  return out;
}

}  // namespace tyche::regression
