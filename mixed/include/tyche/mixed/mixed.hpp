// Linear mixed models: observations grouped, each group with effects of its own drawn from a common law.
//
//   y_g = X_g β + Z_g b_g + e_g,   b_g ~ N(0, G),  e_g ~ N(0, σ² I),  independent across groups,
//
// with X the fixed design (p columns), Z the random design (q columns: 1 for a random intercept, 1 and a covariate for a
// random intercept and slope) and G an unstructured q × q covariance (Laird and Ware 1982, Random-effects models for
// longitudinal data, Biometrics 38, 963–974). The marginal covariance of a group is V_g = Z_g G Z_gᵀ + σ² I.
//
// Estimation (Bates, Mächler, Bolker and Walker 2015, Fitting linear mixed-effects models using lme4, Journal of
// Statistical Software 67(1), §3). G is written σ² Λ Λᵀ with Λ lower triangular, its q(q + 1)/2 entries θ the only
// parameters searched; for a given θ, β and σ² have closed forms and are profiled out. With H_g = V_g/σ² = I + Z_g Λ Λᵀ Z_gᵀ,
// Woodbury's identity gives H_g⁻¹ = I − Z_g Λ M_g⁻¹ Λᵀ Z_gᵀ and log|H_g| = log|M_g| with the q × q matrix
// M_g = I + Λᵀ Z_gᵀ Z_g Λ, so that no n_g × n_g matrix is formed. Then β̂ = (XᵀH⁻¹X)⁻¹XᵀH⁻¹y, RSS = (y − Xβ̂)ᵀH⁻¹(y − Xβ̂) and
//   restricted (REML):  −2ℓ_R(θ) = (n − p)(1 + log(2π RSS/(n − p))) + log|H| + log|XᵀH⁻¹X|,
//   full (ML):          −2ℓ(θ)   = n(1 + log(2π RSS/n)) + log|H|,
// the restricted likelihood being that of the error contrasts, free of β and of ML's downward bias in the variances
// (Patterson and Thompson 1971; Harville 1977, JASA 72, 320–340, without the constant −log|XᵀX|). θ is found by calx's
// Nelder–Mead from Λ = I (and from a second start, the better kept); only ΛΛᵀ is identified, not the signs of Λ.
//
// Predictions: the best linear unbiased predictors b̂_g = G Z_gᵀ V_g⁻¹ (y_g − X_g β̂) = Λ M_g⁻¹ Λᵀ Z_gᵀ (y_g − X_g β̂)
// (Henderson 1975; Robinson 1991, That BLUP is a good thing, Statistical Science 6, 15–51); the fixed effects' covariance
// σ̂²(XᵀH⁻¹X)⁻¹, which ignores the uncertainty of θ; with a random intercept alone, the intraclass correlation
// τ²/(τ² + σ²).
//
// A group's rows need not be contiguous. Costs: O(n q² + G q³ + n p²) an evaluation.

#pragma once

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/dense.hpp>
#include <calx/optimisation/nelder_mead.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::mixed {

using calx::linalg::Matrix;

struct Data {
  std::vector<double> y;
  std::vector<std::vector<double>> x;  // fixed-effect columns, each of length n (an intercept is added when asked)
  std::vector<std::vector<double>> z;  // random-effect columns (a random intercept is added when asked)
  std::vector<std::size_t> group;      // each observation's group, 0 … G − 1
};

struct Options {
  bool fixed_intercept = true, random_intercept = true;
  bool reml = true;
  calx::optimisation::NelderMeadOptions search{1e-10, 1e-12, 0, {}, false};
};

struct Fit {
  std::vector<double> beta, se;          // fixed effects (the intercept first when added), and their standard errors
  Matrix G;                              // the covariance of the random effects
  double sigma2 = 0;                     // the residual variance
  std::vector<std::vector<double>> blup; // per group, its predicted random effects
  double criterion = 0;                  // −2ℓ_R (REML) or −2ℓ (ML) at the estimate
  double icc = std::numeric_limits<double>::quiet_NaN();  // with a random intercept alone
  std::vector<double> theta;             // the relative Cholesky factor Λ, its lower triangle by rows
  std::size_t groups = 0, evaluations = 0;
  bool converged = false;
};

namespace detail {

struct Design {
  std::size_t n = 0, p = 0, q = 0, groups = 0;
  std::vector<std::vector<double>> X, Z;  // [observation][column]
  std::vector<double> y;
  std::vector<std::vector<std::size_t>> members;  // the observations of each group
};

inline Design design(const Data& d, const Options& o) {
  Design s;
  s.n = d.y.size();
  if (s.n == 0 || d.group.size() != s.n) throw std::invalid_argument("tyche::mixed: one group label for each observation");
  for (const auto& c : d.x)
    if (c.size() != s.n) throw std::invalid_argument("tyche::mixed: a fixed-effect column of the wrong length");
  for (const auto& c : d.z)
    if (c.size() != s.n) throw std::invalid_argument("tyche::mixed: a random-effect column of the wrong length");
  s.p = d.x.size() + (o.fixed_intercept ? 1 : 0);
  s.q = d.z.size() + (o.random_intercept ? 1 : 0);
  if (s.p == 0 || s.q == 0) throw std::invalid_argument("tyche::mixed: at least one fixed and one random effect");
  s.y = d.y;
  s.X.assign(s.n, {});
  s.Z.assign(s.n, {});
  for (std::size_t i = 0; i < s.n; i++) {
    if (o.fixed_intercept) s.X[i].push_back(1);
    for (const auto& c : d.x) s.X[i].push_back(c[i]);
    if (o.random_intercept) s.Z[i].push_back(1);
    for (const auto& c : d.z) s.Z[i].push_back(c[i]);
    if (d.group[i] >= s.members.size()) s.members.resize(d.group[i] + 1);
    s.members[d.group[i]].push_back(i);
  }
  s.groups = s.members.size();
  for (std::size_t g = 0; g < s.groups; g++)
    if (s.members[g].empty()) throw std::invalid_argument("tyche::mixed: the group " + std::to_string(g) + " has no observations");
  if (s.n <= s.p) throw std::invalid_argument("tyche::mixed: more observations than fixed effects");
  return s;
}

inline Matrix lambda(const std::vector<double>& theta, std::size_t q) {
  Matrix L(q);
  std::size_t k = 0;
  for (std::size_t i = 0; i < q; i++)
    for (std::size_t j = 0; j <= i; j++) L(i, j) = theta[k++];
  return L;
}

inline double log_det(const calx::linalg::Cholesky& c) {
  double s = 0;
  for (std::size_t i = 0; i < c.l.n; i++) s += 2 * std::log(c.l(i, i));
  return s;
}

struct Evaluation {
  double criterion = std::numeric_limits<double>::infinity();
  std::vector<double> beta;
  double sigma2 = 0;
  Matrix xhx;  // XᵀH⁻¹X
  std::vector<std::vector<double>> blup;
};

// The profiled criterion at θ, and with `predict` the BLUPs.
inline Evaluation evaluate(const Design& s, const std::vector<double>& theta, bool reml, bool predict) {
  const std::size_t p = s.p, q = s.q;
  const Matrix L = lambda(theta, q);
  Matrix xhx(p);
  std::vector<double> xhy(p, 0.0);
  double yhy = 0, logdet_h = 0;
  std::vector<calx::linalg::Cholesky> factors(s.groups);
  for (std::size_t g = 0; g < s.groups; g++) {
    const auto& m = s.members[g];
    // A = Λᵀ Z_gᵀ (q × n_g) as rows; M = I + A Aᵀ.
    std::vector<std::vector<double>> A(q, std::vector<double>(m.size(), 0.0));
    for (std::size_t r = 0; r < q; r++)
      for (std::size_t t = 0; t < m.size(); t++)
        for (std::size_t c = r; c < q; c++) A[r][t] += L(c, r) * s.Z[m[t]][c];
    Matrix M = Matrix::identity(q);
    for (std::size_t r = 0; r < q; r++)
      for (std::size_t c = 0; c < q; c++)
        for (std::size_t t = 0; t < m.size(); t++) M(r, c) += A[r][t] * A[c][t];
    factors[g] = calx::linalg::cholesky(M);
    if (!factors[g].positive_definite) return {};
    logdet_h += log_det(factors[g]);
    // H⁻¹v = v − Aᵀ M⁻¹ A v, for v the columns of X and y.
    const auto apply_inverse = [&](const std::vector<double>& v) {
      std::vector<double> av(q, 0.0);
      for (std::size_t r = 0; r < q; r++)
        for (std::size_t t = 0; t < m.size(); t++) av[r] += A[r][t] * v[t];
      const auto w = calx::linalg::cholesky_solve(factors[g], av);
      std::vector<double> out = v;
      for (std::size_t t = 0; t < m.size(); t++)
        for (std::size_t r = 0; r < q; r++) out[t] -= A[r][t] * w[r];
      return out;
    };
    std::vector<std::vector<double>> xcols(p, std::vector<double>(m.size()));
    std::vector<double> yg(m.size());
    for (std::size_t t = 0; t < m.size(); t++) {
      yg[t] = s.y[m[t]];
      for (std::size_t c = 0; c < p; c++) xcols[c][t] = s.X[m[t]][c];
    }
    const auto hy = apply_inverse(yg);
    for (std::size_t t = 0; t < m.size(); t++) yhy += yg[t] * hy[t];
    for (std::size_t c = 0; c < p; c++) {
      const auto hx = apply_inverse(xcols[c]);
      for (std::size_t t = 0; t < m.size(); t++) {
        xhy[c] += hx[t] * yg[t];
        for (std::size_t e = 0; e < p; e++) xhx(c, e) += hx[t] * xcols[e][t];
      }
    }
  }
  const auto cx = calx::linalg::cholesky(xhx);
  if (!cx.positive_definite) throw std::invalid_argument("tyche::mixed: the fixed design is collinear");
  Evaluation ev;
  ev.beta = calx::linalg::cholesky_solve(cx, xhy);
  double rss = yhy;
  for (std::size_t c = 0; c < p; c++) rss -= ev.beta[c] * xhy[c];
  const double n = static_cast<double>(s.n), pp = static_cast<double>(p);
  const double dof = reml ? n - pp : n;
  ev.sigma2 = rss / dof;
  ev.criterion = dof * (1 + std::log(2 * std::numbers::pi * ev.sigma2)) + logdet_h + (reml ? log_det(cx) : 0.0);
  ev.xhx = xhx;
  if (predict) {
    for (std::size_t g = 0; g < s.groups; g++) {
      const auto& m = s.members[g];
      // b̂_g = Λ M⁻¹ Λᵀ Z_gᵀ r_g.
      std::vector<double> lzr(q, 0.0);
      for (std::size_t t = 0; t < m.size(); t++) {
        double r = s.y[m[t]];
        for (std::size_t c = 0; c < p; c++) r -= s.X[m[t]][c] * ev.beta[c];
        for (std::size_t a = 0; a < q; a++)
          for (std::size_t c = a; c < q; c++) lzr[a] += L(c, a) * s.Z[m[t]][c] * r;
      }
      const auto w = calx::linalg::cholesky_solve(factors[g], lzr);
      std::vector<double> b(q, 0.0);
      for (std::size_t a = 0; a < q; a++)
        for (std::size_t c = 0; c <= a; c++) b[a] += L(a, c) * w[c];
      ev.blup.push_back(std::move(b));
    }
  }
  return ev;
}

}  // namespace detail

// The profiled criterion −2ℓ_R (or −2ℓ) at a relative Cholesky factor θ.
inline double criterion(const Data& d, const std::vector<double>& theta, const Options& o = {}) {
  const auto s = detail::design(d, o);
  if (theta.size() != s.q * (s.q + 1) / 2) throw std::invalid_argument("tyche::mixed::criterion: θ has q(q + 1)/2 entries");
  return detail::evaluate(s, theta, o.reml, false).criterion;
}

inline Fit fit(const Data& d, const Options& o = {}) {
  const auto s = detail::design(d, o);
  const std::size_t q = s.q, k = q * (q + 1) / 2;
  const auto objective = [&](std::span<const double> theta) { return detail::evaluate(s, std::vector<double>(theta.begin(), theta.end()), o.reml, false).criterion; };
  std::vector<double> start(k, 0.0), second(k, 0.0);
  for (std::size_t i = 0, idx = 0; i < q; i++)
    for (std::size_t j = 0; j <= i; j++, idx++)
      if (i == j) start[idx] = 1, second[idx] = 0.2;
  auto best = calx::optimisation::nelder_mead(objective, start, o.search);
  const auto other = calx::optimisation::nelder_mead(objective, second, o.search);
  Fit f;
  f.evaluations = static_cast<std::size_t>(best.evaluations + other.evaluations);
  if (other.value < best.value) best = other;
  // A restart from the best point guards against a simplex that collapsed early.
  const auto again = calx::optimisation::nelder_mead(objective, best.x, o.search);
  f.evaluations += static_cast<std::size_t>(again.evaluations);
  if (again.value <= best.value) best = again;
  f.converged = best.converged;
  f.theta = best.x;
  const auto ev = detail::evaluate(s, f.theta, o.reml, true);
  f.beta = ev.beta;
  f.sigma2 = ev.sigma2;
  f.criterion = ev.criterion;
  f.groups = s.groups;
  const Matrix L = detail::lambda(f.theta, q);
  f.G = Matrix(q);
  for (std::size_t i = 0; i < q; i++)
    for (std::size_t j = 0; j < q; j++) {
      double v = 0;
      for (std::size_t c = 0; c < q; c++) v += L(i, c) * L(j, c);
      f.G(i, j) = f.sigma2 * v;
    }
  f.blup = ev.blup;  // on the scale of the data: Λ is relative, and σ² cancels in Λ M⁻¹ Λᵀ
  const auto inv = calx::linalg::solve(ev.xhx, Matrix::identity(s.p));
  for (std::size_t c = 0; c < s.p; c++) f.se.push_back(std::sqrt(f.sigma2 * inv(c, c)));
  if (q == 1 && o.random_intercept) f.icc = f.G(0, 0) / (f.G(0, 0) + f.sigma2);
  return f;
}

}  // namespace tyche::mixed
