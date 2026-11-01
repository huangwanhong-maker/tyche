// Variance components: y = Xβ + Σ_k u_k + e with u_k ~ N(0, σ_k² K_k) and e ~ N(0, σ_e² I), all
// independent, the K_k known positive semi-definite n × n matrices (kinship, similarity, adjacency:
// whatever relates the observations). Estimated by restricted maximum likelihood and by
// Haseman and Elston's regression of products on the kernel.
//
// RESTRICTED LIKELIHOOD (Patterson and Thompson 1971, Biometrika 58, 545–554; Harville 1977, JASA 72,
// 320–340). The likelihood of the error contrasts Aᵀy (AᵀX = 0, AᵀA = I), free of β, is, with
// V = Σ_k σ_k² K_k + σ_e² I and P = V⁻¹ − V⁻¹X(XᵀV⁻¹X)⁻¹XᵀV⁻¹,
//   ℓ_R = −½ [(n − r) log 2π + log|V| + log|XᵀV⁻¹X| − log|XᵀX| + yᵀPy],  r = rank X,
// (Harville 1977, eq. 2.1 with log|AᵀVA| = log|V| + log|XᵀV⁻¹X| − log|XᵀX|); its maximiser does not
// share ML's downward bias of order r/n in the variances.
//
// - reml_spectral (one kernel). With K = U S Uᵀ (calx's symmetric eigensolver), ỹ = Uᵀy, X̃ = UᵀX and
//   V = σ²(h² S + (1 − h²) I), V is diagonal in the rotated coordinates and σ² is profiled out:
//   σ̂²(h²) = RSS/(n − r), RSS = Σ_i ẽ_i²/w_i, w_i = h² s_i + 1 − h², ẽ = ỹ − X̃β̂ the GLS residual, and
//   ℓ_R(h²) = −½ [(n − r)(log 2πσ̂² + 1) + Σ log w_i + log|X̃ᵀW⁻¹X̃| − log|XᵀX|]. One decomposition
//   serves every h² (Kang et al. 2008, EMMA, Genetics 178, 1709–1723; Lippert et al. 2011, FaST-LMM,
//   Nature Methods 8, 833–835), so the maximum over h² ∈ [0, 1) is found exactly: a grid of 200
//   points, then Brent's method in the best one's neighbourhood. O(n³) for the decomposition, then
//   O(n r²) an evaluation.
// - reml (any number of kernels), by average-information REML (Gilmour, Thompson and Cullis 1995,
//   Biometrics 51, 1440–1450; Johnson and Thompson 1995; as GCTA does, Yang, Lee, Goddard and
//   Visscher 2011, AJHG 88, 76–82, eqs. 4–6): the score ∂ℓ_R/∂σ_i² = −½[tr(PK_i) − yᵀPK_iPy] and the
//   average of the observed and expected information, AI_ij = ½ yᵀPK_iPK_jPy, give the step
//   σ ← σ + AI⁻¹ ∂ℓ_R (K_e = I for the residual). It starts at σ_i² = Var(y)/(c + 1) for c kernels
//   with one EM-REML step, σ_i² ← [σ_i⁴ yᵀPK_iPy + tr(σ_i² I − σ_i⁴ PK_i)]/n (GCTA's first iteration),
//   halves a step that lowers ℓ_R (GCTA does not; the halving only guards against overshooting),
//   sets a component that turns negative to 10⁻⁶ Var(y) (GCTA's constraint) and stops when ℓ_R gains
//   less than tolerance (10⁻⁴, GCTA's). Standard errors are AI⁻¹'s. O(n³) an iteration (Cholesky
//   of V); use reml_spectral for one kernel and large n.
// - The proportion h² = σ_1²/(Σ σ²) and its standard error by the delta method on AI⁻¹ (Yang et
//   al. 2011, eq. 7's form): var ĥ² ≈ ∇hᵀ AI⁻¹ ∇h.
//
// HASEMAN–ELSTON REGRESSION (Haseman and Elston 1972, Behav. Genet. 2, 3–19, in the cross-product
// form of Elston et al. 2000, Genet. Epidemiol. 19, 1–17). With y regressed on X and its residuals
// standardised (mean 0, variance 1), E[y_i y_j] = h² K_ij for i ≠ j when the K_ij are the kernel's
// correlations, so the least-squares slope of y_i y_j on K_ij over the n(n − 1)/2 pairs (with an
// intercept) estimates h², a method of moments (Golan, Lander and Rosset 2014, PNAS 111, E5272). As
// Σ c_ij = 0 for c_ij = K_ij − K̄, the slope is the quadratic form ĥ² = zᵀCz/(2S), C the matrix of the
// c_ij with a zero diagonal and S = Σ_{i<j} c_ij²; for Gaussian z ~ N(0, Σ), Var(zᵀCz) = 2 tr(CΣCΣ)
// (Searle 1971, Linear Models, §2.5, Theorem 4), and Cov(zᵀCz, zᵀz) = 2 tr(CΣ²), Var(zᵀz) = 2 tr(Σ²);
// the model-based standard error is the delta method's on zᵀCz/(2S T), T = zᵀz/(n − 1) the
// standardisation's variance, with Σ̂ = ĥ²K + (1 − ĥ²)I and ĥ² clamped to [0, 1] (O(n³)). Under h² = 0 it reduces
// to √(2/(n² Var K_ij)) nearly (Visscher et al. 2014, PLoS Genetics 10, e1004269), also reported. The
// delete-one-observation jackknife (exact in O(n²) from row sums, the standardisation held fixed) is
// reported too, but it is conservative here: the slope is close to a degenerate U-statistic of degree
// two (each pair's term has conditional mean near zero given one member), whose variance the jackknife
// overestimates by about a factor of two (Efron and Stein 1981, Ann. Statist. 9, 586–596, bound it
// from above; the checks find its SE 1.6 times the spread at h² = 0.4). O(n²), and O(n³) for the
// model-based error.
//
// LIMITS. REML's standard errors are asymptotic; near the boundary (h² ≈ 0 or 1) the estimate's law
// is a mixture with a point mass there. Haseman–Elston is unbiased but less efficient than REML.

#pragma once

#include <tyche/regression/linear.hpp>

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigen.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/optimisation/minimise.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::regression {

struct VarianceComponents {
  std::size_t n = 0, k = 0;              // observations, fixed effects (rank of X)
  std::vector<double> sigma2;            // σ_1², …, σ_c², then σ_e²
  std::vector<double> se;                // their standard errors (AI⁻¹)
  calx::linalg::Matrix covariance;       // AI⁻¹, (c + 1) × (c + 1)
  double h2 = 0, h2_se = 0;              // σ_1² / Σσ², and its standard error
  std::vector<double> beta, beta_se;     // the GLS fixed effects
  double log_likelihood = 0;             // ℓ_R at the estimate
  int iterations = 0;
  bool converged = false;
  std::vector<double> profile_h2, profile_log_likelihood;  // reml_spectral: ℓ_R over its grid of h²
};

namespace detail {

// log|M| for a symmetric positive definite M by Cholesky (−∞ when it is not).
inline double log_det_spd(const calx::linalg::Matrix& M) {
  const auto c = calx::linalg::cholesky(M);
  if (!c.positive_definite) return -INFINITY;
  double s = 0;
  for (std::size_t i = 0; i < M.n; i++) s += 2 * std::log(c.l(i, i));
  return s;
}

inline calx::linalg::Matrix gram_weighted(const calx::linalg::Rectangular& X, std::span<const double> inv_w) {
  const std::size_t n = X.rows, k = X.cols;
  calx::linalg::Matrix G(k);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t a = 0; a < k; a++) {
      const double xa = X(i, a) * (inv_w.empty() ? 1.0 : inv_w[i]);
      for (std::size_t b = 0; b <= a; b++) G(a, b) += xa * X(i, b);
    }
  for (std::size_t a = 0; a < k; a++)
    for (std::size_t b = 0; b < a; b++) G(b, a) = G(a, b);
  return G;
}

inline void require_design(const calx::linalg::Rectangular& X, std::size_t n, const char* who) {
  if (X.rows != n) throw std::invalid_argument(std::string(who) + ": X has one row per observation");
  if (X.cols == 0) throw std::invalid_argument(std::string(who) + ": X has at least one column (an intercept)");
}

}  // namespace detail

// REML for one kernel by its eigendecomposition.
inline VarianceComponents reml_spectral(const calx::linalg::Rectangular& X, std::span<const double> y, const calx::linalg::Matrix& K) {
  const std::size_t n = y.size(), k = X.cols;
  detail::require_design(X, n, "tyche::regression::reml_spectral");
  if (K.n != n) throw std::invalid_argument("tyche::regression::reml_spectral: K is n × n");
  if (n <= k + 1) throw std::invalid_argument("tyche::regression::reml_spectral: more observations than fixed effects");
  const auto eig = calx::linalg::symmetric_eigen(K);
  std::vector<double> s(n), yt(n, 0.0);
  calx::linalg::Rectangular Xt(n, k);
  for (std::size_t c = 0; c < n; c++) {
    s[c] = std::max(0.0, eig.values[c]);
    double v = 0;
    for (std::size_t i = 0; i < n; i++) v += eig.vectors(i, c) * y[i];
    yt[c] = v;
    for (std::size_t a = 0; a < k; a++) {
      double u = 0;
      for (std::size_t i = 0; i < n; i++) u += eig.vectors(i, c) * X(i, a);
      Xt(c, a) = u;
    }
  }
  const double logdet_xtx = detail::log_det_spd(detail::gram_weighted(X, {}));
  if (!std::isfinite(logdet_xtx)) throw std::invalid_argument("tyche::regression::reml_spectral: X has full column rank");
  const double dn = static_cast<double>(n), dk = static_cast<double>(k);
  std::vector<double> inv_w(n);
  struct Eval {
    double ll, sigma2;
    std::vector<double> beta;
    calx::linalg::Matrix gram;
  };
  const auto evaluate = [&](double h2) {
    double logw = 0;
    for (std::size_t i = 0; i < n; i++) {
      const double w = h2 * s[i] + 1 - h2;
      inv_w[i] = 1 / w;
      logw += std::log(w);
    }
    Eval e;
    e.gram = detail::gram_weighted(Xt, inv_w);
    std::vector<double> rhs(k, 0.0);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t a = 0; a < k; a++) rhs[a] += Xt(i, a) * inv_w[i] * yt[i];
    const auto ch = calx::linalg::cholesky(e.gram);
    if (!ch.positive_definite) {
      e.ll = -INFINITY;
      return e;
    }
    e.beta = calx::linalg::cholesky_solve(ch, rhs);
    double rss = 0;
    for (std::size_t i = 0; i < n; i++) {
      double r = yt[i];
      for (std::size_t a = 0; a < k; a++) r -= Xt(i, a) * e.beta[a];
      rss += r * r * inv_w[i];
    }
    e.sigma2 = rss / (dn - dk);
    double lg = 0;
    for (std::size_t i = 0; i < k; i++) lg += 2 * std::log(ch.l(i, i));
    e.ll = -0.5 * ((dn - dk) * (std::log(2 * std::numbers::pi * e.sigma2) + 1) + logw + lg - logdet_xtx);
    if (!std::isfinite(e.ll)) e.ll = -INFINITY;
    return e;
  };
  VarianceComponents out;
  out.n = n;
  out.k = k;
  constexpr int G = 200;
  const double top = 1 - 1e-6;
  int best = 0;
  double best_ll = -INFINITY;
  for (int g = 0; g <= G; g++) {
    const double h = top * g / G;
    const double ll = evaluate(h).ll;
    out.profile_h2.push_back(h);
    out.profile_log_likelihood.push_back(ll);
    if (ll > best_ll) {
      best_ll = ll;
      best = g;
    }
  }
  double h2 = top * best / G;
  {
    const double lo = top * std::max(0, best - 1) / G, hi = top * std::min(G, best + 1) / G;
    const auto m = calx::optimisation::brent_minimise([&](double h) { return -evaluate(h).ll; }, lo, hi, 1e-10);
    if (-m.f > best_ll) h2 = m.x;
  }
  const Eval e = evaluate(h2);
  out.h2 = h2;
  out.log_likelihood = e.ll;
  out.iterations = 1;
  out.converged = std::isfinite(e.ll);
  const double sg = h2 * e.sigma2, se2 = (1 - h2) * e.sigma2;
  out.sigma2 = {sg, se2};
  out.beta = e.beta;
  // Fixed effects' covariance σ̂² (X̃ᵀW⁻¹X̃)⁻¹; the AI matrix in (σ_g², σ_e²) at the estimate.
  const auto gi = calx::linalg::inverse(e.gram);
  out.beta_se.resize(k);
  for (std::size_t a = 0; a < k; a++) out.beta_se[a] = std::sqrt(e.sigma2 * gi(a, a));
  std::vector<double> vinv(n), py(n);
  for (std::size_t i = 0; i < n; i++) {
    vinv[i] = 1 / (sg * s[i] + se2);
    double r = yt[i];
    for (std::size_t a = 0; a < k; a++) r -= Xt(i, a) * e.beta[a];
    py[i] = r * vinv[i];
  }
  // P u = V⁻¹u − V⁻¹X̃ (X̃ᵀV⁻¹X̃)⁻¹ X̃ᵀV⁻¹u, with X̃ᵀV⁻¹X̃ = gram/σ̂².
  const auto P = [&](const std::vector<double>& u) {
    std::vector<double> t(k, 0.0), out_v(n);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t a = 0; a < k; a++) t[a] += Xt(i, a) * vinv[i] * u[i];
    std::vector<double> c(k, 0.0);
    for (std::size_t a = 0; a < k; a++)
      for (std::size_t b = 0; b < k; b++) c[a] += e.sigma2 * gi(a, b) * t[b];
    for (std::size_t i = 0; i < n; i++) {
      double v = 0;
      for (std::size_t a = 0; a < k; a++) v += Xt(i, a) * c[a];
      out_v[i] = vinv[i] * (u[i] - v);
    }
    return out_v;
  };
  std::vector<double> kpy(n), ipy = py;
  for (std::size_t i = 0; i < n; i++) kpy[i] = s[i] * py[i];
  const auto pk = P(kpy), pi = P(ipy);
  calx::linalg::Matrix ai(2);
  for (std::size_t i = 0; i < n; i++) {
    ai(0, 0) += 0.5 * kpy[i] * pk[i];
    ai(0, 1) += 0.5 * kpy[i] * pi[i];
    ai(1, 1) += 0.5 * ipy[i] * pi[i];
  }
  ai(1, 0) = ai(0, 1);
  const double det = ai(0, 0) * ai(1, 1) - ai(0, 1) * ai(0, 1);
  out.covariance = calx::linalg::Matrix(2);
  if (det > 0) {
    out.covariance(0, 0) = ai(1, 1) / det;
    out.covariance(1, 1) = ai(0, 0) / det;
    out.covariance(0, 1) = out.covariance(1, 0) = -ai(0, 1) / det;
  } else {
    std::fill(out.covariance.a.begin(), out.covariance.a.end(), std::numeric_limits<double>::quiet_NaN());
  }
  out.se = {std::sqrt(out.covariance(0, 0)), std::sqrt(out.covariance(1, 1))};
  const double tot = sg + se2, d0 = se2 / (tot * tot), d1 = -sg / (tot * tot);
  out.h2_se = std::sqrt(std::max(0.0, d0 * d0 * out.covariance(0, 0) + 2 * d0 * d1 * out.covariance(0, 1) + d1 * d1 * out.covariance(1, 1)));
  return out;
}

struct RemlOptions {
  int max_iterations = 100;
  double tolerance = 1e-4;   // on the gain in ℓ_R
  bool constrain = true;     // a negative component set to 10⁻⁶ Var y
};

// AI-REML for several kernels (each n × n, positive semi-definite) and the residual.
inline VarianceComponents reml(const calx::linalg::Rectangular& X, std::span<const double> y, const std::vector<calx::linalg::Matrix>& kernels,
                               const RemlOptions& options = {}) {
  const std::size_t n = y.size(), k = X.cols, c = kernels.size(), r = c + 1;
  detail::require_design(X, n, "tyche::regression::reml");
  if (c == 0) throw std::invalid_argument("tyche::regression::reml: at least one kernel");
  for (const auto& K : kernels)
    if (K.n != n) throw std::invalid_argument("tyche::regression::reml: each kernel is n × n");
  const double dn = static_cast<double>(n), dk = static_cast<double>(k);
  double mean = 0;
  for (const double v : y) mean += v;
  mean /= dn;
  double vy = 0;
  for (const double v : y) vy += (v - mean) * (v - mean);
  vy /= dn - 1;
  const double logdet_xtx = detail::log_det_spd(detail::gram_weighted(X, {}));
  if (!std::isfinite(logdet_xtx)) throw std::invalid_argument("tyche::regression::reml: X has full column rank");

  struct State {
    double ll = -INFINITY;
    calx::linalg::Matrix P;
    std::vector<double> py, beta;
    calx::linalg::Matrix xvx_inv;
  };
  const auto component = [&](std::size_t i) -> const calx::linalg::Matrix* { return i < c ? &kernels[i] : nullptr; };
  const auto evaluate = [&](const std::vector<double>& sig) {
    State st;
    calx::linalg::Matrix V(n);
    for (std::size_t q = 0; q < c; q++)
      for (std::size_t i = 0; i < n * n; i++) V.a[i] += sig[q] * kernels[q].a[i];
    for (std::size_t i = 0; i < n; i++) V(i, i) += sig[c];
    const auto ch = calx::linalg::cholesky(V);
    if (!ch.positive_definite) return st;
    double logdet_v = 0;
    for (std::size_t i = 0; i < n; i++) logdet_v += 2 * std::log(ch.l(i, i));
    // V⁻¹ column by column, and V⁻¹X.
    calx::linalg::Matrix Vi(n);
    std::vector<double> e(n);
    for (std::size_t j = 0; j < n; j++) {
      std::fill(e.begin(), e.end(), 0.0);
      e[j] = 1;
      const auto col = calx::linalg::cholesky_solve(ch, e);
      for (std::size_t i = 0; i < n; i++) Vi(i, j) = col[i];
    }
    calx::linalg::Rectangular ViX(n, k);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t a = 0; a < k; a++) {
        double s = 0;
        for (std::size_t j = 0; j < n; j++) s += Vi(i, j) * X(j, a);
        ViX(i, a) = s;
      }
    calx::linalg::Matrix xvx(k);
    for (std::size_t a = 0; a < k; a++)
      for (std::size_t b = 0; b < k; b++) {
        double s = 0;
        for (std::size_t i = 0; i < n; i++) s += X(i, a) * ViX(i, b);
        xvx(a, b) = s;
      }
    const double logdet_xvx = detail::log_det_spd(xvx);
    if (!std::isfinite(logdet_xvx)) return st;
    st.xvx_inv = calx::linalg::inverse(xvx);
    st.P = Vi;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) {
        double s = 0;
        for (std::size_t a = 0; a < k; a++)
          for (std::size_t b = 0; b < k; b++) s += ViX(i, a) * st.xvx_inv(a, b) * ViX(j, b);
        st.P(i, j) -= s;
      }
    st.py.assign(n, 0.0);
    double ypy = 0;
    for (std::size_t i = 0; i < n; i++) {
      double s = 0;
      for (std::size_t j = 0; j < n; j++) s += st.P(i, j) * y[j];
      st.py[i] = s;
      ypy += y[i] * s;
    }
    st.beta.assign(k, 0.0);
    std::vector<double> t(k, 0.0);
    for (std::size_t a = 0; a < k; a++)
      for (std::size_t i = 0; i < n; i++) t[a] += ViX(i, a) * y[i];
    for (std::size_t a = 0; a < k; a++)
      for (std::size_t b = 0; b < k; b++) st.beta[a] += st.xvx_inv(a, b) * t[b];
    st.ll = -0.5 * ((dn - dk) * std::log(2 * std::numbers::pi) + logdet_v + logdet_xvx - logdet_xtx + ypy);
    return st;
  };
  // tr(P K) and K u.
  const auto trace_pk = [&](const State& st, const calx::linalg::Matrix* K) {
    double t = 0;
    if (!K) {
      for (std::size_t i = 0; i < n; i++) t += st.P(i, i);
    } else {
      for (std::size_t i = 0; i < n * n; i++) t += st.P.a[i] * K->a[i];  // both symmetric
    }
    return t;
  };
  const auto times = [&](const calx::linalg::Matrix* K, const std::vector<double>& u) {
    if (!K) return u;
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; i++) {
      double s = 0;
      for (std::size_t j = 0; j < n; j++) s += (*K)(i, j) * u[j];
      v[i] = s;
    }
    return v;
  };
  const auto ptimes = [&](const State& st, const std::vector<double>& u) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; i++) {
      double s = 0;
      for (std::size_t j = 0; j < n; j++) s += st.P(i, j) * u[j];
      v[i] = s;
    }
    return v;
  };

  std::vector<double> sig(r, vy / static_cast<double>(r));
  State st = evaluate(sig);
  if (!std::isfinite(st.ll)) throw std::invalid_argument("tyche::regression::reml: V is not positive definite at the start");
  // One EM-REML step (GCTA's first iteration).
  {
    std::vector<double> next(r);
    for (std::size_t i = 0; i < r; i++) {
      const auto kpy = times(component(i), st.py);
      double q = 0;
      for (std::size_t j = 0; j < n; j++) q += st.py[j] * kpy[j];
      next[i] = (sig[i] * sig[i] * q + dn * sig[i] - sig[i] * sig[i] * trace_pk(st, component(i))) / dn;
      if (options.constrain && next[i] <= 0) next[i] = 1e-6 * vy;
    }
    const State nst = evaluate(next);
    if (std::isfinite(nst.ll)) {
      sig = next;
      st = nst;
    }
  }
  VarianceComponents out;
  out.n = n;
  out.k = k;
  calx::linalg::Matrix ai(r);
  for (int it = 1; it <= options.max_iterations; it++) {
    out.iterations = it;
    std::vector<std::vector<double>> kpy(r);
    std::vector<double> score(r);
    for (std::size_t i = 0; i < r; i++) {
      kpy[i] = times(component(i), st.py);
      double q = 0;
      for (std::size_t j = 0; j < n; j++) q += st.py[j] * kpy[i][j];
      score[i] = -0.5 * (trace_pk(st, component(i)) - q);
    }
    std::vector<std::vector<double>> pkpy(r);
    for (std::size_t i = 0; i < r; i++) pkpy[i] = ptimes(st, kpy[i]);
    for (std::size_t i = 0; i < r; i++)
      for (std::size_t j = 0; j <= i; j++) {
        double s = 0;
        for (std::size_t t = 0; t < n; t++) s += kpy[i][t] * pkpy[j][t];
        ai(i, j) = ai(j, i) = 0.5 * s;
      }
    const auto ai_inv = calx::linalg::inverse(ai);
    std::vector<double> delta(r, 0.0);
    for (std::size_t i = 0; i < r; i++)
      for (std::size_t j = 0; j < r; j++) delta[i] += ai_inv(i, j) * score[j];
    double step = 1;
    State nst;
    std::vector<double> next(r);
    for (int h = 0; h < 20; h++) {
      for (std::size_t i = 0; i < r; i++) {
        next[i] = sig[i] + step * delta[i];
        if (options.constrain && next[i] <= 0) next[i] = 1e-6 * vy;
      }
      nst = evaluate(next);
      if (std::isfinite(nst.ll) && nst.ll >= st.ll - 1e-10) break;
      step /= 2;
    }
    if (!std::isfinite(nst.ll)) break;
    const double gain = nst.ll - st.ll;
    sig = next;
    st = std::move(nst);
    if (std::fabs(gain) < options.tolerance) {
      out.converged = true;
      break;
    }
  }
  // The AI matrix at the estimate, for the standard errors.
  {
    std::vector<std::vector<double>> kpy(r), pkpy(r);
    for (std::size_t i = 0; i < r; i++) {
      kpy[i] = times(component(i), st.py);
      pkpy[i] = ptimes(st, kpy[i]);
    }
    for (std::size_t i = 0; i < r; i++)
      for (std::size_t j = 0; j <= i; j++) {
        double s = 0;
        for (std::size_t t = 0; t < n; t++) s += kpy[i][t] * pkpy[j][t];
        ai(i, j) = ai(j, i) = 0.5 * s;
      }
  }
  out.covariance = calx::linalg::inverse(ai);
  out.sigma2 = sig;
  out.se.resize(r);
  for (std::size_t i = 0; i < r; i++) out.se[i] = std::sqrt(std::max(0.0, out.covariance(i, i)));
  out.log_likelihood = st.ll;
  out.beta = st.beta;
  out.beta_se.resize(k);
  for (std::size_t a = 0; a < k; a++) out.beta_se[a] = std::sqrt(st.xvx_inv(a, a));
  double tot = 0;
  for (const double v : sig) tot += v;
  out.h2 = sig[0] / tot;
  std::vector<double> grad(r, -sig[0] / (tot * tot));
  grad[0] = (tot - sig[0]) / (tot * tot);
  double v = 0;
  for (std::size_t i = 0; i < r; i++)
    for (std::size_t j = 0; j < r; j++) v += grad[i] * out.covariance(i, j) * grad[j];
  out.h2_se = std::sqrt(std::max(0.0, v));
  return out;
}

struct HasemanElston {
  double h2 = 0;              // the slope
  double intercept = 0;
  double se = 0;              // model-based: the delta method on zᵀCz/(2S T) for Gaussian z
  double se_jackknife = 0;    // delete-one-observation jackknife (conservative)
  double se_null = 0;         // √(2/(n² Var K_ij)), the approximation under h² = 0
  std::size_t pairs = 0;
};

// Haseman–Elston regression of y_i y_j on K_ij (i < j), y first residualised on X and standardised.
inline HasemanElston haseman_elston(const calx::linalg::Matrix& K, std::span<const double> y, const calx::linalg::Rectangular& X) {
  const std::size_t n = y.size();
  if (K.n != n || n < 3) throw std::invalid_argument("tyche::regression::haseman_elston: K is n × n, n ≥ 3");
  detail::require_design(X, n, "tyche::regression::haseman_elston");
  const auto fit = ols(X, y);
  std::vector<double> z = fit.residuals;
  double m = 0, v = 0;
  for (const double x : z) m += x;
  m /= static_cast<double>(n);
  for (const double x : z) v += (x - m) * (x - m);
  const double sd = std::sqrt(v / static_cast<double>(n - 1));
  for (double& x : z) x = (x - m) / sd;
  // Sums over pairs i < j, and each observation's share of them.
  double N = 0, sk = 0, skk = 0, sp = 0, skp = 0;
  std::vector<double> rk(n, 0.0), rkk(n, 0.0), rp(n, 0.0), rkp(n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) {
      const double a = K(i, j), b = z[i] * z[j];
      N++;
      sk += a;
      skk += a * a;
      sp += b;
      skp += a * b;
      rk[i] += a; rk[j] += a;
      rkk[i] += a * a; rkk[j] += a * a;
      rp[i] += b; rp[j] += b;
      rkp[i] += a * b; rkp[j] += a * b;
    }
  const auto slope = [](double N_, double sk_, double skk_, double sp_, double skp_, double& icept) {
    const double sxx = skk_ - sk_ * sk_ / N_, sxy = skp_ - sk_ * sp_ / N_;
    const double b = sxy / sxx;
    icept = (sp_ - b * sk_) / N_;
    return b;
  };
  HasemanElston he;
  he.pairs = static_cast<std::size_t>(N);
  he.h2 = slope(N, sk, skk, sp, skp, he.intercept);
  const double dn = static_cast<double>(n);
  std::vector<double> loo(n);
  double mean_loo = 0;
  for (std::size_t i = 0; i < n; i++) {
    double ic = 0;
    loo[i] = slope(N - (dn - 1), sk - rk[i], skk - rkk[i], sp - rp[i], skp - rkp[i], ic);
    mean_loo += loo[i];
  }
  mean_loo /= dn;
  double jv = 0;
  for (const double b : loo) jv += (b - mean_loo) * (b - mean_loo);
  he.se_jackknife = std::sqrt((dn - 1) / dn * jv);
  const double var_k = skk / N - (sk / N) * (sk / N);
  he.se_null = std::sqrt(2 / (dn * dn * var_k));
  // The model-based error: C (centred kernel, zero diagonal) and Σ̂ = ĥ²K + (1 − ĥ²)I, with the
  // standardisation's own variance taken in by the delta method on Q/(2S T), T = zᵀz/(n − 1).
  const double kbar = sk / N, h = std::clamp(he.h2, 0.0, 1.0);
  const auto sigma = [&](std::size_t i, std::size_t j) { return h * K(i, j) + (i == j ? 1 - h : 0.0); };
  std::vector<double> M(n * n, 0.0);  // C Σ̂, row-major
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t l = 0; l < n; l++) {
      if (l == i) continue;
      const double c = K(i, l) - kbar;
      const double* srow = &K.a[l * n];
      for (std::size_t j = 0; j < n; j++) M[i * n + j] += c * h * srow[j];
      M[i * n + l] += c * (1 - h);
    }
  double tr_csc = 0, tr_cs = 0, tr_css = 0, tr_s = 0, tr_ss = 0;
  for (std::size_t i = 0; i < n; i++) {
    tr_cs += M[i * n + i];
    tr_s += sigma(i, i);
    for (std::size_t j = 0; j < n; j++) {
      tr_csc += M[i * n + j] * M[j * n + i];
      tr_css += M[i * n + j] * sigma(j, i);
      tr_ss += sigma(i, j) * sigma(j, i);
    }
  }
  const double sxx = skk - sk * sk / N, d1 = dn - 1, tau = tr_s / d1, eq = tr_cs;
  const double var_q = 2 * tr_csc, var_t = 2 * tr_ss / (d1 * d1), cov_qt = 2 * tr_css / d1;
  const double var_theta = (var_q - 2 * (eq / tau) * cov_qt + (eq / tau) * (eq / tau) * var_t) / (4 * sxx * sxx * tau * tau);
  he.se = std::sqrt(std::max(0.0, var_theta));
  return he;
}

}  // namespace tyche::regression
