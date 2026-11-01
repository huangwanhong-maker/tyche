// The sum of single effects (SuSiE): Bayesian variable selection in linear regression whose
// posterior is summarised by credible sets, each holding one effect.
//
// THE MODEL (Wang, Sarkar, Carbonetto and Stephens 2020, "A simple new approach to variable
// selection in regression, with application to genetic fine mapping", JRSS B 82, 1273–1300, §2).
// y = Xb + e, e ~ N(0, σ² I), b = Σ_{l=1}^L b_l, each b_l = γ_l β_l a single effect: γ_l ~ Mult(1, π)
// picks one of the p columns and β_l ~ N(0, σ₀ₗ²) is its size. One effect alone is the single-effect
// regression (SER), whose posterior is exact and cheap (§2.1): with d_j = x_jᵀx_j, b̂_j = x_jᵀy/d_j and
// s_j² = σ²/d_j, the Bayes factor of column j against b = 0 is
//   log BF_j = log N(b̂_j; 0, σ₀² + s_j²) − log N(b̂_j; 0, s_j²)           (their eq. 2.8; Wakefield 2009)
// the posterior inclusion probabilities α_j ∝ π_j BF_j (eq. 2.7), and given γ = j, β is
// N(μ_j, σ₁ⱼ²) with σ₁ⱼ² = (1/s_j² + 1/σ₀²)⁻¹ and μ_j = σ₁ⱼ² b̂_j/s_j² (eqs. 2.9–2.10).
//
// THE FIT. Iterative Bayesian stepwise selection (IBSS, their Algorithm 2 and §3.1): coordinate
// ascent on the evidence lower bound F(q, σ², σ₀²) over q = Π_l q_l, each step the SER fitted to the
// expected residual r̄_l = y − X Σ_{l′≠l} b̄_{l′}, which is the exact maximiser of F in q_l (their
// Proposition 1); then σ² ← E‖y − Xb‖²/n. Every step raises F, so IBSS converges to a stationary
// point of it (Proposition 2 and Appendix B). F = −(n/2) log 2πσ² − ERSS/(2σ²) − Σ_l KL_l (eq. B.6),
//   ERSS = ‖y − X b̄‖² − Σ_l ‖X b̄_l‖² + Σ_l Σ_j d_j α_lj μ2_lj,  μ2 = σ₁² + μ²,
//   KL_l = −log Σ_j π_j BF_j − (1/(2σ²)) (−2 r̄_lᵀ X b̄_l + Σ_j d_j α_lj μ2_lj),
// the second because KL(q*‖p) = E log p(r̄ | b) − log p(r̄) for the exact posterior q* of the SER,
// where ‖r̄‖² cancels. Each σ₀ₗ² is set before its SER to the maximiser of the SER's marginal
// likelihood Σ_j π_j BF_j(σ₀²) over log σ₀² ∈ [−30, 15] by Brent's method (susieR's "optim"), kept at
// its value if that is no worse, and set to 0 when 0 is within null_threshold of it (susieR's
// check_null_threshold), or by the EM step σ₀² = Σ_j α_j μ2_j after it (method em).
//
// Everything is a function of the sufficient statistics XᵀX, Xᵀy, yᵀy and n (Xᵀr̄ = Xᵀy − XᵀX b̄),
// so one core serves individual data (susie: centred, and scaled to unit standard deviation when
// standardize) and summary data (susie_suff_stat; susie_rss from z-scores and their correlation
// matrix R: Zou, Carbonetto, Wang and Stephens 2022, "Fine-mapping from summary data with the 'Sum of
// Single Effects' model", PLoS Genetics 18, e1010299, §2 and eq. 13, with the z-scores adjusted for
// the proportion of variance they explain, z̃ = z √((n − 1)/(z² + n − 2)), so that for standardised x
// and y, Xᵀy = (n − 1) r = √(n − 1) z̃ and XᵀX = (n − 1) R, yᵀy = n − 1). The order of the updates,
// the convergence test (the gain in F below tolerance, F taken before σ² is updated), the defaults
// (L = 10, σ₀² = 0.2 Var y, tolerance 10⁻³, at most 100 iterations) and the summaries follow the
// authors' R package susieR (version 0.14), against which the checks hold this code.
//
// *Remark (susieR 0.14's guard).* susieR adds √ε ≈ 1.5·10⁻⁸ to every prior weight inside the
// logarithm (lpo = lbf + log(π + √ε)), so that a zero weight does not give log 0; the effective prior is
// (π_j + √ε)/(1 + p√ε) and every log Σ_j π_j BF_j is raised by log(1 + p√ε) (1.2·10⁻⁶ for p = 80).
// prior_weight_guard reproduces it by default; 0 gives the paper's formulas exactly. (susieR also gives
// a constant column, whose ŝ² is infinite, the log-weight 0 rather than log π_j; here its BF is 1 and
// its weight π_j, as the model has it.)
//
// SUMMARIES. The posterior inclusion probability PIP_j = 1 − Π_l (1 − α_lj) over the effects whose
// σ₀ₗ² exceeds prior_tolerance (10⁻⁹). A level-ρ credible set of effect l is the smallest set of
// columns whose α_l sum to at least ρ (their Definition 1, §2.2); a set is reported when its effect is
// kept (σ₀ₗ² > prior_tolerance), it differs from the sets of earlier effects, and its purity, the least
// absolute correlation between two of its columns, is at least min_abs_corr (0.5; §3.2.2), sets
// ordered by purity. susieR estimates purity on a random 100 of a set's columns when it has more;
// here every pair is used.
//
// COMPLEXITY. Individual data: O(n p²) for XᵀX once (in parallel over rows of its triangle when a
// pool is given), then O(L p²) an iteration. Limits: the posterior is a variational approximation:
// exact for L = 1 (the SER), and for L > 1 it can miss the joint structure of effects in strong LD
// (§3.4); the credible sets' coverage is calibrated in simulations (theirs and the checks').

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/optimisation/minimise.hpp>
#include <calx/scheduler/parallel_for.hpp>
#include <calx/scheduler/thread_pool.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::regression {

enum class PriorVariance { fixed, optimise, em };

struct SusieOptions {
  std::size_t effects = 10;                // L
  double scaled_prior_variance = 0.2;      // σ₀² at the start, as a fraction of Var y
  std::vector<double> prior_weights;       // π; empty: uniform (normalised to sum 1)
  double residual_variance = -1;           // σ² at the start; < 0: Var y
  bool estimate_residual_variance = true;  // susie_rss callers usually fix it (σ² = 1 on the z scale)
  PriorVariance estimate_prior_variance = PriorVariance::optimise;
  double null_threshold = 0;               // σ₀ₗ² set to 0 when log BF(0) + this ≥ log BF(σ̂₀ₗ²)
  double prior_weight_guard = 1.4901161193847656e-08;  // π_j enter as π_j + this (susieR 0.14's √ε); 0: the paper's π_j
  double prior_tolerance = 1e-9;           // effects with σ₀ₗ² at or below it are left out of PIPs and sets
  double tolerance = 1e-3;                 // on the gain in the ELBO
  int max_iterations = 100;
  bool standardize = true;                 // individual data: scale columns to unit sd
  bool intercept = true;                   // individual data: centre X and y
  double coverage = 0.95;
  double min_abs_corr = 0.5;               // purity threshold of a credible set
  double residual_variance_lowerbound = -1;  // < 0: Var y / 10⁴ (individual data), 0 (summary data)
};

struct CredibleSet {
  std::size_t effect = 0;                // l
  std::vector<std::size_t> variables;    // in decreasing α_l
  double coverage = 0;                   // Σ α_l over the set
  double min_abs_corr = 1, mean_abs_corr = 1, median_abs_corr = 1;
  double log_bayes_factor = 0;           // the effect's log Σ_j π_j BF_j
};

struct SusieFit {
  std::size_t n = 0, p = 0, effects = 0;
  std::vector<double> alpha, mu, mu2;    // L × p, row l the effect l; on the scaled columns
  std::vector<double> lbf_variable;      // L × p: log BF_j of each SER at its last update
  std::vector<double> lbf;               // L: log Σ_j π_j BF_j
  std::vector<double> prior_variance;    // σ₀ₗ², L (on the scaled columns)
  std::vector<double> kl;                // L
  double residual_variance = 0;          // σ²
  std::vector<double> elbo;              // after each iteration
  int iterations = 0;
  bool converged = false;
  std::vector<double> pip;               // p
  std::vector<double> coef;              // p: Σ_l α_l μ_l on X's own scale
  double intercept = 0;
  std::vector<double> fitted;            // individual data: X̂b + intercept
  std::vector<double> center, scale;     // of X's columns (0 and 1 when not applied)
  std::vector<CredibleSet> sets;
};

namespace detail {

// log N(x; 0, v₀ + s²) − log N(x; 0, s²).
inline double single_effect_log_bf(double bhat, double shat2, double prior_variance) {
  if (!std::isfinite(shat2)) return 0;
  if (prior_variance <= 0) return 0;
  const double t = prior_variance + shat2;
  return 0.5 * std::log(shat2 / t) + 0.5 * bhat * bhat * (1 / shat2 - 1 / t);
}

// log Σ_j π_j BF_j(v₀).
inline double ser_log_marginal(std::span<const double> bhat, std::span<const double> shat2, std::span<const double> pi, double v0) {
  double top = -INFINITY;
  const std::size_t p = bhat.size();
  std::vector<double> l(p);
  for (std::size_t j = 0; j < p; j++) top = std::max(top, l[j] = single_effect_log_bf(bhat[j], shat2[j], v0));
  double s = 0;
  for (std::size_t j = 0; j < p; j++) s += pi[j] * std::exp(l[j] - top);
  return top + std::log(s);
}

struct SuffStat {
  calx::linalg::Matrix xtx;  // of centred (and scaled) columns
  std::vector<double> xty, d;
  double yty = 0;
  std::size_t n = 0;
  double var_y = 0;
};

inline void ibss(const SuffStat& S, const SusieOptions& o, double lower_bound, SusieFit& f) {
  const std::size_t p = S.xty.size(), L = std::max<std::size_t>(1, o.effects);
  if (L > p) throw std::invalid_argument("tyche::regression::susie: at most p effects");
  f.p = p;
  f.n = S.n;
  f.effects = L;
  std::vector<double> pi(p, 1.0 / static_cast<double>(p));
  if (!o.prior_weights.empty()) {
    if (o.prior_weights.size() != p) throw std::invalid_argument("tyche::regression::susie: one prior weight per column");
    double s = 0;
    for (const double w : o.prior_weights) {
      if (!(w >= 0)) throw std::invalid_argument("tyche::regression::susie: prior weights ≥ 0");
      s += w;
    }
    for (std::size_t j = 0; j < p; j++) pi[j] = o.prior_weights[j] / s;
  }
  if (!(o.prior_weight_guard >= 0)) throw std::invalid_argument("tyche::regression::susie: a guard ≥ 0");
  for (double& w : pi) w += o.prior_weight_guard;
  f.alpha.assign(L * p, 1.0 / static_cast<double>(p));
  f.mu.assign(L * p, 0.0);
  f.mu2.assign(L * p, 0.0);
  f.lbf_variable.assign(L * p, 0.0);
  f.lbf.assign(L, 0.0);
  f.kl.assign(L, 0.0);
  f.prior_variance.assign(L, o.scaled_prior_variance * S.var_y);
  double sigma2 = o.residual_variance >= 0 ? o.residual_variance : S.var_y;
  if (!(sigma2 > 0)) throw std::invalid_argument("tyche::regression::susie: a positive residual variance (Var y > 0)");
  const double dn = static_cast<double>(S.n);
  std::vector<double> xtxr(p, 0.0), bl(p), xtr(p), bhat(p), shat2(p);
  const auto xtx_times = [&](const std::vector<double>& b, std::vector<double>& out, double sign) {
    for (std::size_t j = 0; j < p; j++) {
      const double* row = &S.xtx.a[j * p];
      double s = 0;
      for (std::size_t k = 0; k < p; k++) s += row[k] * b[k];
      out[j] += sign * s;
    }
  };
  // ERSS from the current posterior (susieR's get_ER2_ss).
  const auto erss = [&] {
    std::vector<double> bbar(p, 0.0), tmp(p);
    double xb2 = 0, post = 0;
    for (std::size_t l = 0; l < L; l++) {
      for (std::size_t j = 0; j < p; j++) {
        tmp[j] = f.alpha[l * p + j] * f.mu[l * p + j];
        bbar[j] += tmp[j];
        post += S.d[j] * f.alpha[l * p + j] * f.mu2[l * p + j];
      }
      std::vector<double> xt(p, 0.0);
      xtx_times(tmp, xt, 1);
      for (std::size_t j = 0; j < p; j++) xb2 += tmp[j] * xt[j];
    }
    std::vector<double> xb(p, 0.0);
    xtx_times(bbar, xb, 1);
    double r = S.yty;
    for (std::size_t j = 0; j < p; j++) r += -2 * bbar[j] * S.xty[j] + bbar[j] * xb[j];
    return r - xb2 + post;
  };
  double previous = -INFINITY;
  f.elbo.clear();
  f.converged = false;
  for (int it = 1; it <= o.max_iterations; it++) {
    f.iterations = it;
    for (std::size_t l = 0; l < L; l++) {
      double* al = &f.alpha[l * p];
      double* ml = &f.mu[l * p];
      double* m2 = &f.mu2[l * p];
      for (std::size_t j = 0; j < p; j++) bl[j] = al[j] * ml[j];
      xtx_times(bl, xtxr, -1);
      for (std::size_t j = 0; j < p; j++) {
        xtr[j] = S.xty[j] - xtxr[j];
        if (S.d[j] > 0) {
          bhat[j] = xtr[j] / S.d[j];
          shat2[j] = sigma2 / S.d[j];
        } else {
          bhat[j] = 0;
          shat2[j] = INFINITY;
        }
      }
      double V = f.prior_variance[l];
      if (o.estimate_prior_variance == PriorVariance::optimise) {
        const auto neg = [&](double lv) { return -ser_log_marginal(bhat, shat2, pi, std::exp(lv)); };
        const auto m = calx::optimisation::brent_minimise(neg, -30, 15, 1.4901161193847656e-08);
        const double current = V > 0 ? neg(std::log(V)) : -ser_log_marginal(bhat, shat2, pi, 0);
        if (!(m.f > current)) V = std::exp(m.x);
        if (ser_log_marginal(bhat, shat2, pi, 0) + o.null_threshold >= ser_log_marginal(bhat, shat2, pi, V)) V = 0;
      }
      // The SER's posterior under V.
      double top = -INFINITY;
      for (std::size_t j = 0; j < p; j++) top = std::max(top, f.lbf_variable[l * p + j] = single_effect_log_bf(bhat[j], shat2[j], V));
      double wsum = 0;
      for (std::size_t j = 0; j < p; j++) wsum += pi[j] * std::exp(f.lbf_variable[l * p + j] - top);
      double post_e2 = 0, cross = 0;
      for (std::size_t j = 0; j < p; j++) {
        al[j] = pi[j] * std::exp(f.lbf_variable[l * p + j] - top) / wsum;
        const double v1 = V > 0 ? 1 / (1 / V + S.d[j] / sigma2) : 0;
        ml[j] = v1 * xtr[j] / sigma2;
        m2[j] = v1 + ml[j] * ml[j];
        post_e2 += S.d[j] * al[j] * m2[j];
        cross += al[j] * ml[j] * xtr[j];
      }
      f.lbf[l] = top + std::log(wsum);
      if (o.estimate_prior_variance == PriorVariance::em) {
        double v = 0;
        for (std::size_t j = 0; j < p; j++) v += al[j] * m2[j];
        V = v;
        if (ser_log_marginal(bhat, shat2, pi, 0) + o.null_threshold >= ser_log_marginal(bhat, shat2, pi, V)) V = 0;
      }
      f.prior_variance[l] = V;
      f.kl[l] = -f.lbf[l] - 0.5 / sigma2 * (-2 * cross + post_e2);
      for (std::size_t j = 0; j < p; j++) bl[j] = al[j] * ml[j];
      xtx_times(bl, xtxr, 1);
    }
    const double e = erss();
    double kl = 0;
    for (const double k : f.kl) kl += k;
    const double elbo = -dn / 2 * std::log(2 * std::numbers::pi * sigma2) - e / (2 * sigma2) - kl;
    f.elbo.push_back(elbo);
    if (elbo - previous < o.tolerance) {
      f.converged = true;
      break;
    }
    previous = elbo;
    if (o.estimate_residual_variance) sigma2 = std::max(lower_bound, e / dn);
  }
  f.residual_variance = sigma2;
  // PIPs over the kept effects.
  f.pip.assign(p, 0.0);
  std::vector<double> keep_prod(p, 1.0);
  for (std::size_t l = 0; l < L; l++)
    if (f.prior_variance[l] > o.prior_tolerance)
      for (std::size_t j = 0; j < p; j++) keep_prod[j] *= 1 - f.alpha[l * p + j];
  for (std::size_t j = 0; j < p; j++) f.pip[j] = 1 - keep_prod[j];
}

// The credible sets of the kept effects, filtered by purity from the correlations of the columns.
template <class Corr>
inline std::vector<CredibleSet> credible_sets(const SusieFit& f, const SusieOptions& o, Corr&& corr) {
  const std::size_t p = f.p, L = f.effects;
  std::vector<CredibleSet> sets;
  std::vector<std::vector<std::size_t>> seen;
  for (std::size_t l = 0; l < L; l++) {
    std::vector<std::size_t> idx(p);
    std::iota(idx.begin(), idx.end(), 0);
    const double* a = &f.alpha[l * p];
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t x, std::size_t y) { return a[x] > a[y]; });
    // susieR's n_in_CS: #{k : cumulative sum of the sorted α below the coverage} + 1.
    double c = 0;
    std::size_t count = 0;
    for (std::size_t k = 0; k < p; k++) {
      c += a[idx[k]];
      if (c < o.coverage) count++;
    }
    count = std::min(p, count + 1);
    CredibleSet cs;
    cs.effect = l;
    cs.variables.assign(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(count));
    for (const std::size_t j : cs.variables) cs.coverage += a[j];
    cs.log_bayes_factor = f.lbf[l];
    // Duplicates are judged against every earlier effect, kept or not, as susieR's duplicated() is.
    std::vector<std::size_t> key = cs.variables;
    std::sort(key.begin(), key.end());
    const bool duplicate = std::find(seen.begin(), seen.end(), key) != seen.end();
    seen.push_back(key);
    if (duplicate || !(f.prior_variance[l] > o.prior_tolerance)) continue;
    if (count > 1) {
      std::vector<double> r;
      for (std::size_t x = 0; x < count; x++)
        for (std::size_t y = x + 1; y < count; y++) r.push_back(std::fabs(corr(cs.variables[x], cs.variables[y])));
      std::sort(r.begin(), r.end());
      cs.min_abs_corr = r.front();
      double s = 0;
      for (const double v : r) s += v;
      cs.mean_abs_corr = s / static_cast<double>(r.size());
      const std::size_t h = r.size() / 2;
      cs.median_abs_corr = r.size() % 2 ? r[h] : (r[h - 1] + r[h]) / 2;
    }
    if (cs.min_abs_corr >= o.min_abs_corr) sets.push_back(std::move(cs));
  }
  std::stable_sort(sets.begin(), sets.end(), [](const CredibleSet& x, const CredibleSet& y) { return x.min_abs_corr > y.min_abs_corr; });
  return sets;
}

}  // namespace detail

// SuSiE on individual data: X (n × p) and y.
inline SusieFit susie(const calx::linalg::Rectangular& X, std::span<const double> y, const SusieOptions& o = {}, calx::scheduler::ThreadPool* pool = nullptr) {
  const std::size_t n = X.rows, p = X.cols;
  if (y.size() != n || n < 2 || p == 0) throw std::invalid_argument("tyche::regression::susie: X is n × p, n ≥ 2, and y has n entries");
  SusieFit f;
  f.center.assign(p, 0.0);
  f.scale.assign(p, 1.0);
  const double dn = static_cast<double>(n);
  double ymean = 0;
  for (const double v : y) ymean += v;
  ymean /= dn;
  if (!o.intercept) ymean = 0;
  for (std::size_t j = 0; j < p; j++) {
    double m = 0;
    for (std::size_t i = 0; i < n; i++) m += X(i, j);
    m /= dn;
    double ss = 0;
    for (std::size_t i = 0; i < n; i++) ss += (X(i, j) - m) * (X(i, j) - m);
    f.center[j] = o.intercept ? m : 0;
    // A constant column is left unscaled (susieR sets its scale to 1).
    const double sd = std::sqrt(ss / (dn - 1));
    f.scale[j] = o.standardize && sd > 0 ? sd : 1;
  }
  // The scaled, centred columns, column-major for the Gram matrix.
  std::vector<double> Z(n * p);
  for (std::size_t j = 0; j < p; j++)
    for (std::size_t i = 0; i < n; i++) Z[j * n + i] = (X(i, j) - f.center[j]) / f.scale[j];
  detail::SuffStat S;
  S.n = n;
  S.xtx = calx::linalg::Matrix(p);
  S.xty.assign(p, 0.0);
  S.d.assign(p, 0.0);
  const auto rows = [&](std::size_t begin, std::size_t end, std::size_t) {
    for (std::size_t j = begin; j < end; j++) {
      const double* a = &Z[j * n];
      for (std::size_t k = 0; k <= j; k++) {
        const double* b = &Z[k * n];
        double s = 0;
        for (std::size_t i = 0; i < n; i++) s += a[i] * b[i];
        S.xtx(j, k) = s;
      }
      double s = 0;
      for (std::size_t i = 0; i < n; i++) s += a[i] * (y[i] - ymean);
      S.xty[j] = s;
    }
  };
  if (pool) calx::scheduler::parallel_for(*pool, p, 16, rows);
  else rows(0, p, 0);
  for (std::size_t j = 0; j < p; j++) {
    for (std::size_t k = 0; k < j; k++) S.xtx(k, j) = S.xtx(j, k);
    S.d[j] = S.xtx(j, j);
  }
  double vy = 0, yy = 0;
  for (const double v : y) {
    yy += (v - ymean) * (v - ymean);
    vy += v;
  }
  S.yty = yy;
  {
    double m = vy / dn, s = 0;
    for (const double v : y) s += (v - m) * (v - m);
    S.var_y = s / (dn - 1);
  }
  const double lower = o.residual_variance_lowerbound >= 0 ? o.residual_variance_lowerbound : S.var_y / 1e4;
  detail::ibss(S, o, lower, f);
  // Coefficients, intercept, fitted values.
  f.coef.assign(p, 0.0);
  for (std::size_t l = 0; l < f.effects; l++)
    for (std::size_t j = 0; j < p; j++) f.coef[j] += f.alpha[l * p + j] * f.mu[l * p + j] / f.scale[j];
  f.intercept = ymean;
  for (std::size_t j = 0; j < p; j++) f.intercept -= f.center[j] * f.coef[j];
  f.fitted.assign(n, f.intercept);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < p; j++) f.fitted[i] += X(i, j) * f.coef[j];
  f.sets = detail::credible_sets(f, o, [&](std::size_t a, std::size_t b) {
    const double da = S.d[a], db = S.d[b];
    return da > 0 && db > 0 ? S.xtx(a, b) / std::sqrt(da * db) : 0.0;
  });
  return f;
}

// SuSiE on sufficient statistics: XᵀX and Xᵀy of centred X and y, yᵀy and n (susieR's susie_suff_stat).
inline SusieFit susie_suff_stat(const calx::linalg::Matrix& XtX, std::span<const double> Xty, double yty, std::size_t n, const SusieOptions& o = {}) {
  const std::size_t p = Xty.size();
  if (XtX.n != p || n < 2 || p == 0) throw std::invalid_argument("tyche::regression::susie_suff_stat: XᵀX is p × p for p = |Xᵀy|, n ≥ 2");
  SusieFit f;
  f.center.assign(p, 0.0);
  f.scale.assign(p, 1.0);
  detail::SuffStat S;
  S.n = n;
  S.yty = yty;
  S.var_y = yty / static_cast<double>(n - 1);
  S.xtx = XtX;
  S.xty.assign(Xty.begin(), Xty.end());
  if (o.standardize)
    for (std::size_t j = 0; j < p; j++) {
      const double sd = std::sqrt(XtX(j, j) / static_cast<double>(n - 1));
      f.scale[j] = sd > 0 ? sd : 1;
    }
  for (std::size_t j = 0; j < p; j++) {
    for (std::size_t k = 0; k < p; k++) S.xtx(j, k) = XtX(j, k) / (f.scale[j] * f.scale[k]);
    S.xty[j] = Xty[j] / f.scale[j];
  }
  S.d.resize(p);
  for (std::size_t j = 0; j < p; j++) S.d[j] = S.xtx(j, j);
  detail::ibss(S, o, o.residual_variance_lowerbound >= 0 ? o.residual_variance_lowerbound : 0.0, f);
  f.coef.assign(p, 0.0);
  for (std::size_t l = 0; l < f.effects; l++)
    for (std::size_t j = 0; j < p; j++) f.coef[j] += f.alpha[l * p + j] * f.mu[l * p + j] / f.scale[j];
  f.sets = detail::credible_sets(f, o, [&](std::size_t a, std::size_t b) {
    const double da = S.d[a], db = S.d[b];
    return da > 0 && db > 0 ? S.xtx(a, b) / std::sqrt(da * db) : 0.0;
  });
  return f;
}

// SuSiE-RSS from z-scores, their correlation matrix R and the sample size n (susieR's susie_rss with
// n given; it fixes σ² unless asked: pass estimate_residual_variance as susieR's default, false).
inline SusieFit susie_rss(std::span<const double> z, const calx::linalg::Matrix& R, std::size_t n, const SusieOptions& o) {
  const std::size_t p = z.size();
  if (R.n != p || n < 3) throw std::invalid_argument("tyche::regression::susie_rss: R is p × p for p z-scores, n ≥ 3");
  const double dn = static_cast<double>(n);
  calx::linalg::Matrix XtX = (dn - 1) * R;
  std::vector<double> Xty(p);
  for (std::size_t j = 0; j < p; j++) {
    const double adj = (dn - 1) / (z[j] * z[j] + dn - 2);
    Xty[j] = std::sqrt(dn - 1) * std::sqrt(adj) * z[j];
  }
  return susie_suff_stat(XtX, Xty, dn - 1, n, o);
}

}  // namespace tyche::regression
