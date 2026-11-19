// Ideal points from votes: the one-dimensional probit model of roll calls, fitted by EM.
//
// Members i = 1..n vote yes or no on items j = 1..m, some votes missing (absent, abstaining, not yet a member). Each member has an
// ideal point x_i on a line and each item a cut and a discrimination; a member votes yes when
//
//   y*_ij = α_j + β_j x_i + ε_ij > 0,   ε_ij ~ N(0, 1) independent,
//
// the quadratic-utility model of spatial voting written as an item response model (Clinton, Jackman and Rivers 2004, CJR). With the
// priors x_i ~ N(0, σ_x²) and α_j, β_j ~ N(0, σ_b²), the posterior mode is found by the EM algorithm of Imai, Lo and Olmsted (2016,
// "Fast estimation of ideal points with massive data", American Political Science Review 110(4), 631–656), on the latent y*:
//
//   E-step   y*_ij ← m_ij + φ(m_ij)/Φ(m_ij) for a yes, m_ij − φ(m_ij)/(1 − Φ(m_ij)) for a no, with m_ij = α_j + β_j x_i;
//            missing votes enter no sum;
//   M-step   x_i ← Σ_j β_j (y*_ij − α_j) / (Σ_j β_j² + 1/σ_x²), over i's votes;
//            (α_j, β_j) ← (Xᵀ X + I/σ_b²)⁻¹ Xᵀ y*_j, X the rows (1, x_i) of j's voters.
//
// Each step raises the log posterior, which is reported by iteration (EM's monotonicity; Dempster, Laird and Rubin 1977, Theorem 1).
// EM approaches the mode slowly along the direction the prior barely pins (the scale of the line), so its steps are accelerated by
// SQUAREM with a safeguard that keeps the log posterior from falling; it stops when no parameter moves by more than a tolerance in an
// iteration (Imai, Lo and Olmsted stop on the correlation of successive parameter vectors), or when the log posterior has stopped
// changing. An iteration is SQUAREM's cycle of three steps of EM.
// The model is identified only up to reflection, x ↦ −x with β ↦ −β, which leaves the posterior unchanged; an anchor member is
// taken to lie on the positive side. Its location and scale are fixed by the prior on x; `standardised` re-expresses a fit with
// the members' mean 0 and standard deviation 1, the items' parameters carried with it, which changes no probability.
//
// The ratios φ/Φ are computed by Laplace's continued fraction for Mills's ratio in the far tail, so that unanimous items and
// members who never break with their side do not overflow. Starting values are the leading principal component of the centred
// votes. Intervals come from a parametric bootstrap: votes drawn from the fitted model where they were cast, the model refitted
// from the fit (Imai, Lo and Olmsted 2016, §4; Lewis and Poole 2004), each replicate's line put on the fit's location and scale, so
// that an interval is of a member's place relative to the others.
//
// How well a line fits is told by the share of votes it classifies correctly and by the aggregate proportional reduction in error,
// APRE = Σ_j (minority_j − errors_j) / Σ_j minority_j (Poole and Rosenthal 1997, ch. 3).
//
// Caveat. Votes record public positions, not preferences (the survey behind the diplomacy domain, §A.11.4); a line fitted to them is
// a summary of how members divided, under an assumption of one dimension, and is not a measure of what they want.
//
// References. Clinton, J., Jackman, S. and Rivers, D. (2004), The statistical analysis of roll call data, APSR 98(2), 355–370.
// Imai, K., Lo, J. and Olmsted, J. (2016), above. Poole, K. T. and Rosenthal, H. (1997), Congress: A Political-Economic History of
// Roll Call Voting, Oxford University Press. Lewis, J. B. and Poole, K. T. (2004), Measuring bias and uncertainty in ideal point
// estimates via the parametric bootstrap, Political Analysis 12(2), 105–127. Abramowitz, M. and Stegun, I. A. (1964), Handbook of
// Mathematical Functions, 26.2.14 (Laplace's continued fraction).

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace tyche::latent {

// A matrix of votes: 1 yes, 0 no, any other value missing.
using Votes = std::vector<std::vector<int>>;

struct IdealPointOptions {
  double prior_sd_x = 1;      // x_i ~ N(0, σ_x²)
  double prior_sd_item = 5;   // α_j, β_j ~ N(0, σ_b²)
  std::size_t most = 5000;             // iterations (each SQUAREM's cycle of three steps of EM)
  double parameter_tolerance = 1e-6;   // it stops when no parameter moves by more than this in an iteration,
  double tolerance = 1e-12;            // or when the log posterior changes by less than this, relatively
  std::size_t anchor = 0;     // a member taken to lie on the positive side
  std::vector<double> start;  // starting ideal points (empty: the leading principal component)
};

struct IdealPoints {
  std::vector<double> x, alpha, beta;
  std::vector<double> log_posterior;  // by iteration
  std::size_t iterations = 0;
  bool converged = false;
  double classified = 0, apre = 0;
  std::vector<double> lower, upper;   // a bootstrap interval for each x, when asked
};

namespace detail {

inline double phi(double m) { return std::exp(-0.5 * m * m) / std::sqrt(2 * std::numbers::pi); }

// Mills's ratio R(t) = (1 − Φ(t))/φ(t) for t > 0 by Laplace's continued fraction 1/(t + 1/(t + 2/(t + 3/(t + …)))).
inline double mills(double t) {
  double r = t;
  for (int k = 60; k >= 1; k--) r = t + k / r;
  return 1 / r;
}

// φ(m)/Φ(m), stable for every m.
inline double lambda_yes(double m) {
  if (m > -5) return phi(m) / (0.5 * std::erfc(-m / std::numbers::sqrt2));
  return 1 / mills(-m);
}

// log Φ(m), stable for every m.
inline double log_Phi(double m) {
  if (m > -5) return std::log(0.5 * std::erfc(-m / std::numbers::sqrt2));
  const double t = -m;
  return -0.5 * t * t - 0.5 * std::log(2 * std::numbers::pi) + std::log(mills(t));
}

inline bool cast(int v) { return v == 0 || v == 1; }

inline void check(const Votes& y) {
  if (y.empty() || y[0].empty()) throw std::invalid_argument("tyche::latent::ideal_points: at least one member and one item");
  for (const auto& row : y)
    if (row.size() != y[0].size()) throw std::invalid_argument("tyche::latent::ideal_points: one vote, or none, by every member on every item");
}

// The leading principal component of the votes, centred by item, missing votes as the item's mean.
inline std::vector<double> leading_component(const Votes& y) {
  const std::size_t n = y.size(), m = y[0].size();
  std::vector<double> mean(m, 0.0);
  std::vector<std::size_t> count(m, 0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < m; j++)
      if (cast(y[i][j])) mean[j] += y[i][j], count[j]++;
  for (std::size_t j = 0; j < m; j++) mean[j] = count[j] ? mean[j] / count[j] : 0.5;
  auto c = [&](std::size_t i, std::size_t j) { return cast(y[i][j]) ? y[i][j] - mean[j] : 0.0; };
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; i++) v[i] = 1 + 0.01 * static_cast<double>(i % 7);  // a start not orthogonal to the component
  for (int it = 0; it < 200; it++) {
    std::vector<double> u(m, 0.0), w(n, 0.0);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < m; j++) u[j] += c(i, j) * v[i];
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < m; j++) w[i] += c(i, j) * u[j];
    double norm = 0;
    for (double x : w) norm += x * x;
    norm = std::sqrt(norm);
    if (norm == 0) break;
    for (std::size_t i = 0; i < n; i++) v[i] = w[i] / norm;
  }
  // scaled to unit variance
  double mu = 0, sd = 0;
  for (double x : v) mu += x / n;
  for (double x : v) sd += (x - mu) * (x - mu) / n;
  sd = std::sqrt(sd);
  for (double& x : v) x = sd > 0 ? (x - mu) / sd : 0;
  return v;
}

}  // namespace detail

// The log posterior (up to a constant) of the parameters given the votes.
inline double log_posterior(const Votes& y, const std::vector<double>& x, const std::vector<double>& alpha, const std::vector<double>& beta,
                            const IdealPointOptions& o = {}) {
  double lp = 0;
  for (std::size_t i = 0; i < y.size(); i++)
    for (std::size_t j = 0; j < y[i].size(); j++)
      if (detail::cast(y[i][j])) {
        const double mm = alpha[j] + beta[j] * x[i];
        lp += detail::log_Phi(y[i][j] == 1 ? mm : -mm);
      }
  for (double v : x) lp -= v * v / (2 * o.prior_sd_x * o.prior_sd_x);
  for (std::size_t j = 0; j < alpha.size(); j++) lp -= (alpha[j] * alpha[j] + beta[j] * beta[j]) / (2 * o.prior_sd_item * o.prior_sd_item);
  return lp;
}

inline IdealPoints ideal_points(const Votes& y, const IdealPointOptions& o = {}) {
  detail::check(y);
  const std::size_t n = y.size(), m = y[0].size();
  if (o.anchor >= n) throw std::invalid_argument("tyche::latent::ideal_points: the anchor is a member");
  IdealPoints f;
  f.x = o.start.empty() ? detail::leading_component(y) : o.start;
  if (f.x.size() != n) throw std::invalid_argument("tyche::latent::ideal_points: one starting point per member");
  f.alpha.assign(m, 0.0), f.beta.assign(m, 0.0);
  std::vector<std::vector<double>> z(n, std::vector<double>(m, 0.0));
  const double px = 1 / (o.prior_sd_x * o.prior_sd_x), pb = 1 / (o.prior_sd_item * o.prior_sd_item);
  auto e_step = [&] {
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < m; j++) {
        if (!detail::cast(y[i][j])) continue;
        const double mm = f.alpha[j] + f.beta[j] * f.x[i];
        z[i][j] = y[i][j] == 1 ? mm + detail::lambda_yes(mm) : mm - detail::lambda_yes(-mm);
      }
  };
  auto m_items = [&] {
    for (std::size_t j = 0; j < m; j++) {
      double s00 = pb, s01 = 0, s11 = pb, r0 = 0, r1 = 0;
      for (std::size_t i = 0; i < n; i++) {
        if (!detail::cast(y[i][j])) continue;
        s00 += 1, s01 += f.x[i], s11 += f.x[i] * f.x[i], r0 += z[i][j], r1 += f.x[i] * z[i][j];
      }
      const double det = s00 * s11 - s01 * s01;
      f.alpha[j] = (s11 * r0 - s01 * r1) / det;
      f.beta[j] = (s00 * r1 - s01 * r0) / det;
    }
  };
  auto m_members = [&] {
    for (std::size_t i = 0; i < n; i++) {
      double num = 0, den = px;
      for (std::size_t j = 0; j < m; j++) {
        if (!detail::cast(y[i][j])) continue;
        num += f.beta[j] * (z[i][j] - f.alpha[j]), den += f.beta[j] * f.beta[j];
      }
      f.x[i] = num / den;
    }
  };
  // One step of EM as a map of the parameters θ = (x, α, β), accelerated by SQUAREM (Varadhan and Roland 2008, "Simple and globally
  // convergent methods for accelerating the convergence of any EM algorithm", Scandinavian Journal of Statistics 35, 335–353): from θ,
  // two steps θ₁ = F(θ), θ₂ = F(θ₁); r = θ₁ − θ, v = θ₂ − θ₁ − r; the step length α = −‖r‖/‖v‖ (their S3), at most −1; the point
  // θ − 2αr + α²v, and one step of EM from it. That step is kept only when its log posterior is at least θ₂'s, else θ₂ is: so the log
  // posterior never falls, as under plain EM.
  const std::size_t dim = n + 2 * m;
  auto get = [&] {
    std::vector<double> t(dim);
    for (std::size_t i = 0; i < n; i++) t[i] = f.x[i];
    for (std::size_t j = 0; j < m; j++) t[n + j] = f.alpha[j], t[n + m + j] = f.beta[j];
    return t;
  };
  auto set = [&](const std::vector<double>& t) {
    for (std::size_t i = 0; i < n; i++) f.x[i] = t[i];
    for (std::size_t j = 0; j < m; j++) f.alpha[j] = t[n + j], f.beta[j] = t[n + m + j];
  };
  auto em = [&](const std::vector<double>& t) {
    set(t);
    e_step(), m_members();
    e_step(), m_items();
    return get();
  };
  auto lp = [&](const std::vector<double>& t) {
    set(t);
    return log_posterior(y, f.x, f.alpha, f.beta, o);
  };
  // first the items from the starting points, then EM in full
  e_step(), m_items();
  std::vector<double> theta = get();
  double previous = lp(theta);
  f.log_posterior.push_back(previous);
  for (f.iterations = 1; f.iterations <= o.most; f.iterations++) {
    const std::vector<double> t1 = em(theta), t2 = em(t1);
    double rr = 0, vv = 0;
    std::vector<double> r(dim), v(dim);
    for (std::size_t k = 0; k < dim; k++) {
      r[k] = t1[k] - theta[k], v[k] = t2[k] - t1[k] - r[k];
      rr += r[k] * r[k], vv += v[k] * v[k];
    }
    std::vector<double> next = t2;
    double now = lp(t2);
    if (vv > 0) {
      const double step = std::min(-1.0, -std::sqrt(rr / vv));
      std::vector<double> jump(dim);
      for (std::size_t k = 0; k < dim; k++) jump[k] = theta[k] - 2 * step * r[k] + step * step * v[k];
      const std::vector<double> settled = em(jump);
      const double ls = lp(settled);
      if (std::isfinite(ls) && ls >= now) next = settled, now = ls;
    }
    double moved = 0;
    for (std::size_t k = 0; k < dim; k++) moved = std::max(moved, std::abs(next[k] - theta[k]));
    theta = next;
    f.log_posterior.push_back(now);
    if (moved <= o.parameter_tolerance || std::abs(now - previous) <= o.tolerance * (1 + std::abs(now))) {
      f.converged = true;
      break;
    }
    previous = now;
  }
  set(theta);
  if (f.x[o.anchor] < 0) {
    for (double& v : f.x) v = -v;
    for (double& b : f.beta) b = -b;
  }
  // how well the line classifies the votes
  double right = 0, all = 0, minority = 0, errors = 0;
  for (std::size_t j = 0; j < m; j++) {
    double yes = 0, no = 0, wrong = 0;
    for (std::size_t i = 0; i < n; i++) {
      if (!detail::cast(y[i][j])) continue;
      const bool predicted = f.alpha[j] + f.beta[j] * f.x[i] > 0;
      (y[i][j] == 1 ? yes : no) += 1;
      if (predicted != (y[i][j] == 1)) wrong += 1;
    }
    right += yes + no - wrong, all += yes + no, minority += std::min(yes, no), errors += wrong;
  }
  f.classified = all > 0 ? right / all : 0;
  f.apre = minority > 0 ? (minority - errors) / minority : 0;
  return f;
}

// The same fit with the members' ideal points of mean 0 and standard deviation 1; α + β x is unchanged for every vote.
inline IdealPoints standardised(IdealPoints f) {
  const double n = static_cast<double>(f.x.size());
  double mu = 0, sd = 0;
  for (double v : f.x) mu += v / n;
  for (double v : f.x) sd += (v - mu) * (v - mu) / n;
  sd = std::sqrt(sd);
  if (sd == 0) return f;
  for (double& v : f.x) v = (v - mu) / sd;
  for (std::size_t j = 0; j < f.alpha.size(); j++) f.alpha[j] += f.beta[j] * mu, f.beta[j] *= sd;
  for (std::size_t i = 0; i < f.lower.size(); i++) f.lower[i] = (f.lower[i] - mu) / sd, f.upper[i] = (f.upper[i] - mu) / sd;
  return f;
}

// Intervals for the ideal points by a parametric bootstrap: votes drawn from the fit where they were cast, refitted from the fit.
inline void bootstrap(IdealPoints& f, const Votes& y, std::size_t replicates, calx::random::Stream& rng, double level = 0.95, const IdealPointOptions& o = {}) {
  const std::size_t n = y.size(), m = y[0].size();
  std::vector<std::vector<double>> draws(n);
  IdealPointOptions warm = o;
  warm.start = f.x;
  // a replicate needs its line to the precision of the interval, not of the fit: refitted from the fit, to a looser rule
  warm.parameter_tolerance = std::max(o.parameter_tolerance, 1e-4);
  warm.most = std::min<std::size_t>(o.most, 2000);
  warm.anchor = o.anchor;
  for (std::size_t r = 0; r < replicates; r++) {
    Votes sim = y;
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < m; j++)
        if (detail::cast(y[i][j])) {
          const double p = 0.5 * std::erfc(-(f.alpha[j] + f.beta[j] * f.x[i]) / std::numbers::sqrt2);
          sim[i][j] = rng.uniform() < p ? 1 : 0;
        }
    const auto g = ideal_points(sim, warm);
    // the model fixes the line's location and scale only through its prior, so each replicate is put on the fit's location and scale:
    // what the interval measures is a member's place relative to the others, as the points are reported
    double mf = 0, sf = 0, mg = 0, sg = 0;
    for (std::size_t i = 0; i < n; i++) mf += f.x[i] / static_cast<double>(n), mg += g.x[i] / static_cast<double>(n);
    for (std::size_t i = 0; i < n; i++) sf += (f.x[i] - mf) * (f.x[i] - mf), sg += (g.x[i] - mg) * (g.x[i] - mg);
    const double scale = sg > 0 ? std::sqrt(sf / sg) : 1;
    for (std::size_t i = 0; i < n; i++) draws[i].push_back(mf + (g.x[i] - mg) * scale);
  }
  f.lower.assign(n, 0.0), f.upper.assign(n, 0.0);
  const double a = (1 - level) / 2;
  for (std::size_t i = 0; i < n; i++) {
    auto& d = draws[i];
    std::sort(d.begin(), d.end());
    auto at = [&](double q) {
      const double pos = q * static_cast<double>(d.size() - 1);
      const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
      const std::size_t hi = std::min(lo + 1, d.size() - 1);
      return d[lo] + (pos - static_cast<double>(lo)) * (d[hi] - d[lo]);
    };
    f.lower[i] = at(a), f.upper[i] = at(1 - a);
  }
}

}  // namespace tyche::latent
