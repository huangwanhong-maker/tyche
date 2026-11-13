// Markov chain Monte Carlo: draws from a law known up to its normalising constant, with the diagnostics that say whether they
// can be trusted (decision 0021).
//
// A target is a log density on ℝᵈ, with its gradient if the caller has one (else central differences are taken, d + d
// evaluations each). Two samplers, each running several chains from dispersed starting points, warming up, then drawing:
//
//   metropolis(target, o)   random-walk Metropolis whose proposal covariance is learnt in the warm-up (Haario, Saksman and
//                           Tamminen 2001, Bernoulli 7: 223–242), scaled by 2.38²/d (Roberts, Gelman and Gilks 1997) and by a
//                           factor tuned on the log scale toward an acceptance of 0.234 (Robbins–Monro, Andrieu and Thoms 2008,
//                           Stat. Comput. 18: 343–373); the proposal is fixed after the warm-up, so the chain is then exactly
//                           reversible with respect to the target
//   hmc(target, o)          Hamiltonian Monte Carlo with the leapfrog (Neal 2011, Handbook of MCMC, ch. 5): a diagonal mass matrix
//                           from the warm-up's variances, the step size by dual averaging toward an acceptance of 0.8 (Hoffman and
//                           Gelman 2014, JMLR 15: 1593–1623, §3.2, eqs. 5–6), a path of length about π/2 in the scaled
//                           coordinates with a random number of steps, so that no period locks in; a trajectory whose energy
//                           error exceeds 1000 is counted as divergent
//
// Diagnostics (Gelman et al. 2013, BDA3, §11.4–11.5; Vehtari et al. 2021, Bayesian Anal. 16: 667–718):
//   split R̂                 each chain split in halves; R̂ = √(var⁺/W) with var⁺ = (n − 1)/n W + B/n (BDA3 eq. 11.3–11.4)
//   effective sample size   from the chains' autocorrelations combined as ρ̂_t = 1 − (W − mean autocovariance_t)/var⁺, summed
//                           over Geyer's initial positive and monotone sequence of pairs (BDA3 eq. 11.7–11.8; Geyer 1992)
//   Monte Carlo error       sd / √ESS of each mean
//
// Accuracy: the draws' laws are the target's in the limit; what the diagnostics measure is how far a finite run is from it.
// R̂ below 1.01 and an effective sample size above a few hundred per quantity are the usual bars (Vehtari et al. 2021).

#pragma once

#include <tyche/bayes/common.hpp>

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/dense.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::bayes {

struct Target {
  std::size_t dim = 0;
  std::function<double(std::span<const double>)> log_density;
  std::function<void(std::span<const double>, std::span<double>)> gradient;  // optional
};

struct McmcOptions {
  std::size_t chains = 4, warmup = 1000, draws = 1000;
  std::uint64_t seed = 1;
  std::vector<std::vector<double>> init;  // one point per chain, or empty: N(0, init_spread²) in every coordinate
  double init_spread = 1;
  double target_accept = -1;              // −1: 0.234 for Metropolis, 0.8 for HMC
  double path = std::numbers::pi / 2;     // HMC: the integration time in the scaled coordinates
};

struct Chains {
  std::vector<std::vector<std::vector<double>>> draws;  // [chain][draw][coordinate]
  std::vector<double> acceptance, step;                 // per chain, after the warm-up
  std::vector<double> mean, sd, rhat, ess, mcse;        // per coordinate, over all chains
  std::size_t divergent = 0;
};

namespace detail {

inline std::vector<double> numeric_gradient(const Target& t, std::span<const double> x) {
  std::vector<double> g(t.dim), y(x.begin(), x.end());
  for (std::size_t i = 0; i < t.dim; i++) {
    const double h = 1e-5 * std::max(1.0, std::fabs(x[i]));
    y[i] = x[i] + h;
    const double up = t.log_density(y);
    y[i] = x[i] - h;
    const double down = t.log_density(y);
    y[i] = x[i];
    g[i] = (up - down) / (2 * h);
  }
  return g;
}
inline std::vector<double> gradient_of(const Target& t, std::span<const double> x) {
  if (!t.gradient) return numeric_gradient(t, x);
  std::vector<double> g(t.dim);
  t.gradient(x, g);
  return g;
}
inline std::vector<double> start(const McmcOptions& o, std::size_t c, std::size_t d, calx::random::Stream& rng) {
  if (!o.init.empty()) {
    require(o.init.size() == o.chains && o.init[c].size() == d, "mcmc: one starting point per chain, of the target's dimension");
    return o.init[c];
  }
  std::vector<double> x(d);
  for (double& v : x) v = o.init_spread * rng.normal();
  return x;
}

}  // namespace detail

// ---- diagnostics ----

struct Diagnostics {
  std::vector<double> mean, sd, rhat, ess, mcse;
};

inline Diagnostics diagnose(const std::vector<std::vector<std::vector<double>>>& draws) {
  detail::require(!draws.empty() && draws[0].size() >= 4, "diagnose: at least one chain of four draws");
  const std::size_t m0 = draws.size(), n0 = draws[0].size(), d = draws[0][0].size();
  for (const auto& c : draws) detail::require(c.size() == n0, "diagnose: chains of one length");
  Diagnostics D;
  const std::size_t half = n0 / 2, m = 2 * m0;  // split chains
  for (std::size_t k = 0; k < d; k++) {
    std::vector<std::vector<double>> s(m);
    for (std::size_t c = 0; c < m0; c++)
      for (std::size_t i = 0; i < 2 * half; i++) s[2 * c + (i >= half)].push_back(draws[c][i][k]);
    const double n = static_cast<double>(half);
    std::vector<double> means(m), vars(m);
    double grand = 0;
    for (std::size_t j = 0; j < m; j++) {
      double mu = 0;
      for (double v : s[j]) mu += v;
      mu /= n;
      double var = 0;
      for (double v : s[j]) var += (v - mu) * (v - mu);
      means[j] = mu, vars[j] = var / (n - 1);
      grand += mu / static_cast<double>(m);
    }
    double B = 0, W = 0;
    for (std::size_t j = 0; j < m; j++) B += (means[j] - grand) * (means[j] - grand), W += vars[j];
    B *= n / static_cast<double>(m - 1), W /= static_cast<double>(m);
    const double varp = (n - 1) / n * W + B / n;
    D.mean.push_back(grand);
    D.sd.push_back(std::sqrt(varp));
    D.rhat.push_back(W > 0 ? std::sqrt(varp / W) : NAN);
    // Autocovariances of each split chain, combined (BDA3 eq. 11.7), summed over Geyer's initial monotone sequence.
    const std::size_t T = static_cast<std::size_t>(n) - 1;
    std::vector<double> rho(T + 1, 0.0);
    for (std::size_t t = 0; t <= T; t++) {
      double acov = 0;
      for (std::size_t j = 0; j < m; j++) {
        double a = 0;
        for (std::size_t i = 0; i + t < s[j].size(); i++) a += (s[j][i] - means[j]) * (s[j][i + t] - means[j]);
        acov += a / n;
      }
      acov /= static_cast<double>(m);
      rho[t] = varp > 0 ? 1 - (W - acov) / varp : 0.0;
      if (t >= 3 && t % 2 == 1 && rho[t - 1] + rho[t] < 0) break;  // the first negative pair ends the sequence
    }
    double sum = 0, prev = INFINITY;
    for (std::size_t t = 0; t + 1 <= T; t += 2) {
      double pair = rho[t] + rho[t + 1];
      if (pair < 0) break;
      pair = std::min(pair, prev);  // initial monotone sequence
      prev = pair;
      sum += pair;
    }
    const double tau = std::max(-1 + 2 * sum, 1.0 / std::log10(static_cast<double>(m) * n));
    D.ess.push_back(static_cast<double>(m) * n / tau);
    D.mcse.push_back(D.sd.back() / std::sqrt(D.ess.back()));
  }
  return D;
}

namespace detail {
inline Chains finish(Chains C) {
  const auto D = diagnose(C.draws);
  C.mean = D.mean, C.sd = D.sd, C.rhat = D.rhat, C.ess = D.ess, C.mcse = D.mcse;
  return C;
}
}  // namespace detail

// ---- random-walk Metropolis, adaptive in the warm-up ----

inline Chains metropolis(const Target& t, const McmcOptions& o = {}) {
  const std::size_t d = t.dim;
  detail::require(d > 0 && t.log_density && o.chains >= 1 && o.warmup >= 10 && o.draws >= 4, "metropolis: a target, a dimension, and enough draws");
  const double target_accept = o.target_accept > 0 ? o.target_accept : 0.234;
  Chains C;
  for (std::size_t c = 0; c < o.chains; c++) {
    calx::random::Stream rng(o.seed, static_cast<std::uint32_t>(c), 1, 0);
    std::vector<double> x = detail::start(o, c, d, rng);
    double lp = t.log_density(x);
    detail::require(std::isfinite(lp), "metropolis: the log density is finite at the starting point");
    calx::linalg::Matrix cov = calx::linalg::Matrix::identity(d);
    std::vector<double> mean(x);
    double scale = 2.38 * 2.38 / static_cast<double>(d), log_factor = 0;
    calx::linalg::Matrix L = calx::linalg::Matrix::identity(d);
    std::size_t accepted = 0;
    std::vector<std::vector<double>> kept;
    const std::size_t total = o.warmup + o.draws;
    for (std::size_t it = 0; it < total; it++) {
      std::vector<double> z(d), y(d);
      for (double& v : z) v = rng.normal();
      const double f = std::sqrt(scale * std::exp(log_factor));
      for (std::size_t i = 0; i < d; i++) {
        double s = 0;
        for (std::size_t j = 0; j <= i; j++) s += L(i, j) * z[j];
        y[i] = x[i] + f * s;
      }
      const double ly = t.log_density(y);
      const double a = std::isfinite(ly) ? std::min(1.0, std::exp(ly - lp)) : 0.0;
      if (rng.uniform() < a) x = y, lp = ly, accepted += it >= o.warmup;
      if (it < o.warmup) {
        // Robbins–Monro on the log scale, and the running mean and covariance (Haario et al.), refactored now and then.
        log_factor += (a - target_accept) / std::pow(static_cast<double>(it + 1), 0.6);
        const double w = 1.0 / static_cast<double>(it + 2);
        for (std::size_t i = 0; i < d; i++) {
          const double di = x[i] - mean[i];
          for (std::size_t j = 0; j < d; j++) cov(i, j) = (1 - w) * cov(i, j) + w * (1 - w) * di * (x[j] - mean[j]);
        }
        for (std::size_t i = 0; i < d; i++) mean[i] += w * (x[i] - mean[i]);
        if (it > 50 && it % 50 == 0) {
          calx::linalg::Matrix S = cov;
          for (std::size_t i = 0; i < d; i++) S(i, i) += 1e-10 + 1e-8 * std::fabs(S(i, i));
          const auto f = calx::linalg::cholesky(S);
          if (f.positive_definite) L = f.l;  // else keep the last factor
        }
      } else {
        kept.push_back(x);
      }
    }
    C.draws.push_back(std::move(kept));
    C.acceptance.push_back(static_cast<double>(accepted) / static_cast<double>(o.draws));
    C.step.push_back(std::sqrt(scale * std::exp(log_factor)));
  }
  return detail::finish(std::move(C));
}

// ---- Hamiltonian Monte Carlo ----

inline Chains hmc(const Target& t, const McmcOptions& o = {}) {
  const std::size_t d = t.dim;
  detail::require(d > 0 && t.log_density && o.chains >= 1 && o.warmup >= 20 && o.draws >= 4, "hmc: a target, a dimension, and enough draws");
  const double delta = o.target_accept > 0 ? o.target_accept : 0.8;
  Chains C;
  for (std::size_t c = 0; c < o.chains; c++) {
    calx::random::Stream rng(o.seed, static_cast<std::uint32_t>(c), 2, 0);
    std::vector<double> x = detail::start(o, c, d, rng);
    double lp = t.log_density(x);
    detail::require(std::isfinite(lp), "hmc: the log density is finite at the starting point");
    std::vector<double> g = detail::gradient_of(t, x);
    std::vector<double> inv_mass(d, 1.0);  // the diagonal of M⁻¹: the warm-up's variances
    // Dual averaging (Hoffman and Gelman, eq. 6): γ = 0.05, t₀ = 10, κ = 0.75, μ = log(10 ε₀).
    double eps = 0.1, log_eps_bar = 0, H_bar = 0, mu = std::log(10 * eps);
    std::size_t m_adapt = 0;
    std::vector<double> wmean(d, 0.0), wvar(d, 0.0);
    std::size_t wn = 0;
    const std::size_t window_start = o.warmup / 4, window_end = (3 * o.warmup) / 4;
    std::size_t accepted_draws = 0;
    double accept_sum = 0;
    std::vector<std::vector<double>> kept;
    for (std::size_t it = 0; it < o.warmup + o.draws; it++) {
      std::vector<double> p(d), q = x, gq = g;
      for (std::size_t i = 0; i < d; i++) p[i] = rng.normal() / std::sqrt(inv_mass[i]);
      auto kinetic = [&](const std::vector<double>& pp) {
        double k = 0;
        for (std::size_t i = 0; i < d; i++) k += 0.5 * pp[i] * pp[i] * inv_mass[i];
        return k;
      };
      const double H0 = -lp + kinetic(p);
      const std::size_t steps = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(o.path / eps * (0.5 + rng.uniform()))));
      double lq = lp;
      bool divergent = false;
      for (std::size_t s = 0; s < steps && !divergent; s++) {
        for (std::size_t i = 0; i < d; i++) p[i] += 0.5 * eps * gq[i];
        for (std::size_t i = 0; i < d; i++) q[i] += eps * inv_mass[i] * p[i];
        lq = t.log_density(q);
        if (!std::isfinite(lq)) {
          divergent = true;
          break;
        }
        gq = detail::gradient_of(t, q);
        for (std::size_t i = 0; i < d; i++) p[i] += 0.5 * eps * gq[i];
        if (-lq + kinetic(p) - H0 > 1000) divergent = true;
      }
      const double H1 = divergent ? INFINITY : -lq + kinetic(p);
      const double a = std::isfinite(H1) ? std::min(1.0, std::exp(H0 - H1)) : 0.0;
      if (rng.uniform() < a) x = q, lp = lq, g = gq, accepted_draws += it >= o.warmup;
      if (it >= o.warmup) {
        accept_sum += a;
        C.divergent += divergent;
        kept.push_back(x);
        continue;
      }
      // Warm-up: the step by dual averaging; the mass matrix from the middle window, after which the averaging restarts.
      m_adapt++;
      const double mm = static_cast<double>(m_adapt);
      H_bar = (1 - 1 / (mm + 10)) * H_bar + (delta - a) / (mm + 10);
      const double log_eps = mu - std::sqrt(mm) / 0.05 * H_bar;
      eps = std::exp(log_eps);
      const double w = std::pow(mm, -0.75);
      log_eps_bar = w * log_eps + (1 - w) * log_eps_bar;
      if (it >= window_start && it < window_end) {
        wn++;
        for (std::size_t i = 0; i < d; i++) {
          const double dlt = x[i] - wmean[i];
          wmean[i] += dlt / static_cast<double>(wn);
          wvar[i] += dlt * (x[i] - wmean[i]);
        }
      }
      if (it + 1 == window_end && wn > 10) {
        for (std::size_t i = 0; i < d; i++) {
          const double v = wvar[i] / static_cast<double>(wn - 1);
          // Regularised toward the unit scale as Stan does: (n / (n + 5)) v + 10⁻³ (5 / (n + 5)).
          inv_mass[i] = (static_cast<double>(wn) / (static_cast<double>(wn) + 5)) * v + 1e-3 * (5.0 / (static_cast<double>(wn) + 5));
        }
        m_adapt = 0, H_bar = 0, log_eps_bar = 0, mu = std::log(10 * eps);
      }
      if (it + 1 == o.warmup) eps = std::exp(log_eps_bar);
    }
    C.draws.push_back(std::move(kept));
    C.acceptance.push_back(accept_sum / static_cast<double>(o.draws));
    C.step.push_back(eps);
    (void)accepted_draws;
  }
  return detail::finish(std::move(C));
}

}  // namespace tyche::bayes
