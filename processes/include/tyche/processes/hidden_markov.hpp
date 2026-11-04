// Hidden Markov models with Gaussian emissions: filtering, smoothing, the most probable path, and their fit to many
// sequences at once by expectation–maximisation.
//
// A chain on K hidden states with initial law π and transition matrix A emits, at each step, an observation
// y_t ~ N(μ_k, σ_k²) in its state k (Hamilton 1989, "A new approach to the economic analysis of nonstationary time
// series and the business cycle", Econometrica 57, 357–384, reads the business cycle so). The belief about the state
// given y_1 … y_t is carried by Bayes's rule, the forward filter, scaled at each step so that it is the filtered law
// itself and the log-likelihood the sum of the logs of the scales (Rabiner 1989, "A tutorial on hidden Markov models
// and selected applications in speech recognition", Proc. IEEE 77, 257–286, §V.A); the backward pass gives the law
// given the whole sequence (smoothed); Viterbi's recursion in logs the most probable path. A missing observation (NaN)
// emits with likelihood 1: the filter then only predicts.
//
// The fit (Baum, Petrie, Soules and Weiss 1970; Dempster, Laird and Rubin 1977): given the smoothed laws γ_t(k) and
// the pairs ξ_t(i, j) of each sequence, π is the mean of the first γ, A_ij = Σ ξ(i, j) / Σ_{t<T} γ(i), μ_k and σ_k the
// γ-weighted mean and deviation of the observations; each iteration does not lower the likelihood. The states are
// numbered at the end in increasing order of their means (the labels being otherwise arbitrary), and σ_k is held above
// 10⁻³ of the pooled deviation, where the likelihood is unbounded (a state that explains one observation exactly).
//
// A shared deviation, σ_k = σ for every state (Hamilton's own model switches only the mean), has the M-step
// σ² = Σ_k Σ_t γ_t(k) (y_t − μ_k)² / Σ_k Σ_t γ_t(k); the states then differ by their means alone.
//
// Caveat. EM finds a local maximum: the fit starts from the pooled quantiles and keeps the best of several starts.

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::processes {

struct GaussianHmm {
  std::size_t states = 0;
  std::vector<double> initial;     // π
  std::vector<double> transition;  // A(i → j) at [i K + j]
  std::vector<double> mean, sd;

  double emission(std::size_t k, double y) const {
    if (std::isnan(y)) return 1;
    const double z = (y - mean[k]) / sd[k];
    return std::exp(-0.5 * z * z) / (sd[k] * std::sqrt(2 * std::numbers::pi));
  }
  double a(std::size_t i, std::size_t j) const { return transition[i * states + j]; }
};

struct Decoded {
  std::vector<std::vector<double>> filtered, smoothed;  // P(s_t | y_1..t), P(s_t | y_1..T)
  std::vector<std::size_t> path;                        // the most probable path (Viterbi)
  double log_likelihood = 0;
};

namespace detail {
inline void check(const GaussianHmm& M) {
  const std::size_t K = M.states;
  if (K == 0 || M.initial.size() != K || M.transition.size() != K * K || M.mean.size() != K || M.sd.size() != K)
    throw std::invalid_argument("tyche::processes::GaussianHmm: π of K, A of K², means and deviations of K");
  for (const double s : M.sd)
    if (!(s > 0)) throw std::invalid_argument("tyche::processes::GaussianHmm: deviations are positive");
}
}  // namespace detail

inline Decoded decode(const GaussianHmm& M, std::span<const double> y) {
  detail::check(M);
  const std::size_t K = M.states, T = y.size();
  Decoded D;
  if (T == 0) return D;
  std::vector<std::vector<double>> alpha(T, std::vector<double>(K)), beta(T, std::vector<double>(K, 1.0));
  std::vector<double> scale(T);
  for (std::size_t t = 0; t < T; t++) {
    double c = 0;
    for (std::size_t j = 0; j < K; j++) {
      double prior = 0;
      if (t == 0) prior = M.initial[j];
      else
        for (std::size_t i = 0; i < K; i++) prior += alpha[t - 1][i] * M.a(i, j);
      c += (alpha[t][j] = prior * M.emission(j, y[t]));
    }
    if (!(c > 0)) throw std::invalid_argument("tyche::processes::decode: an observation of likelihood zero under every state");
    for (double& v : alpha[t]) v /= c;
    scale[t] = c;
    D.log_likelihood += std::log(c);
  }
  for (std::size_t t = T - 1; t-- > 0;) {
    for (std::size_t i = 0; i < K; i++) {
      double b = 0;
      for (std::size_t j = 0; j < K; j++) b += M.a(i, j) * M.emission(j, y[t + 1]) * beta[t + 1][j];
      beta[t][i] = b / scale[t + 1];
    }
  }
  D.filtered = alpha;
  D.smoothed.assign(T, std::vector<double>(K));
  for (std::size_t t = 0; t < T; t++) {
    double s = 0;
    for (std::size_t k = 0; k < K; k++) s += (D.smoothed[t][k] = alpha[t][k] * beta[t][k]);
    for (double& v : D.smoothed[t]) v /= s;
  }
  // Viterbi, in logs.
  std::vector<std::vector<double>> v(T, std::vector<double>(K));
  std::vector<std::vector<std::size_t>> back(T, std::vector<std::size_t>(K, 0));
  const auto lg = [](double x) { return x > 0 ? std::log(x) : -std::numeric_limits<double>::infinity(); };
  for (std::size_t k = 0; k < K; k++) v[0][k] = lg(M.initial[k]) + lg(M.emission(k, y[0]));
  for (std::size_t t = 1; t < T; t++)
    for (std::size_t j = 0; j < K; j++) {
      double best = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < K; i++) {
        const double x = v[t - 1][i] + lg(M.a(i, j));
        if (x > best) best = x, back[t][j] = i;
      }
      v[t][j] = best + lg(M.emission(j, y[t]));
    }
  D.path.assign(T, 0);
  D.path[T - 1] = static_cast<std::size_t>(std::max_element(v[T - 1].begin(), v[T - 1].end()) - v[T - 1].begin());
  for (std::size_t t = T - 1; t-- > 0;) D.path[t] = back[t + 1][D.path[t + 1]];
  return D;
}

struct HmmFit {
  GaussianHmm model;
  double log_likelihood = 0;
  int iterations = 0;
  bool converged = false;
  std::vector<double> trace;  // the log-likelihood after each iteration of the best start
};

namespace detail {

// One EM run from `M`; returns the fit.
inline HmmFit em(GaussianHmm M, const std::vector<std::vector<double>>& sequences, double floor_sd, int max_iterations, double tolerance, bool shared_sd) {
  const std::size_t K = M.states;
  HmmFit F;
  double previous = -std::numeric_limits<double>::infinity();
  for (F.iterations = 0; F.iterations < max_iterations; F.iterations++) {
    std::vector<double> pi(K, 0.0), num(K * K, 0.0), den(K, 0.0), w(K, 0.0), wy(K, 0.0), wyy(K, 0.0);
    double ll = 0;
    std::size_t used = 0;
    for (const auto& y : sequences) {
      if (y.empty()) continue;
      const auto D = decode(M, y);
      ll += D.log_likelihood, used++;
      for (std::size_t k = 0; k < K; k++) pi[k] += D.smoothed[0][k];
      for (std::size_t t = 0; t < y.size(); t++)
        for (std::size_t k = 0; k < K; k++)
          if (!std::isnan(y[t])) w[k] += D.smoothed[t][k], wy[k] += D.smoothed[t][k] * y[t], wyy[k] += D.smoothed[t][k] * y[t] * y[t];
      // ξ_t(i, j) = α_t(i) A_ij γ_{t+1}(j) / Σ_l α_t(l) A_lj, from the filtered and smoothed laws alone.
      for (std::size_t t = 0; t + 1 < y.size(); t++) {
        std::vector<double> pred(K, 0.0);
        for (std::size_t j = 0; j < K; j++)
          for (std::size_t i = 0; i < K; i++) pred[j] += D.filtered[t][i] * M.a(i, j);
        for (std::size_t i = 0; i < K; i++)
          for (std::size_t j = 0; j < K; j++)
            if (pred[j] > 0) {
              const double xi = D.filtered[t][i] * M.a(i, j) * D.smoothed[t + 1][j] / pred[j];
              num[i * K + j] += xi, den[i] += xi;
            }
      }
    }
    F.trace.push_back(ll);
    // M-step.
    for (std::size_t k = 0; k < K; k++) M.initial[k] = pi[k] / static_cast<double>(used);
    for (std::size_t i = 0; i < K; i++)
      if (den[i] > 0)
        for (std::size_t j = 0; j < K; j++) M.transition[i * K + j] = num[i * K + j] / den[i];
    double pooled_ss = 0, pooled_w = 0;
    for (std::size_t k = 0; k < K; k++)
      if (w[k] > 0) {
        M.mean[k] = wy[k] / w[k];
        const double ss = std::max(0.0, wyy[k] - w[k] * M.mean[k] * M.mean[k]);  // Σ γ (y − μ_k)²
        M.sd[k] = std::max(floor_sd, std::sqrt(ss / w[k]));
        pooled_ss += ss, pooled_w += w[k];
      }
    if (shared_sd && pooled_w > 0)
      for (std::size_t k = 0; k < K; k++) M.sd[k] = std::max(floor_sd, std::sqrt(pooled_ss / pooled_w));
    F.model = M, F.log_likelihood = ll;
    if (std::fabs(ll - previous) <= tolerance * (1 + std::fabs(ll))) { F.converged = true; break; }
    previous = ll;
  }
  return F;
}

}  // namespace detail

// The fit to sequences (missing values NaN), from `starts` starts: the first at the pooled quantiles, the others
// perturbed; the best kept, its states ordered by their means.
inline HmmFit fit_hmm(const std::vector<std::vector<double>>& sequences, std::size_t K, int starts = 5, int max_iterations = 500, double tolerance = 1e-9,
                      std::uint64_t seed = 20261104, bool shared_sd = false) {
  std::vector<double> pooled;
  for (const auto& y : sequences)
    for (const double v : y)
      if (!std::isnan(v)) pooled.push_back(v);
  if (pooled.size() < 2 * K) throw std::invalid_argument("tyche::processes::fit_hmm: too few observations for the states");
  std::sort(pooled.begin(), pooled.end());
  double m = 0, s2 = 0;
  for (const double v : pooled) m += v;
  m /= static_cast<double>(pooled.size());
  for (const double v : pooled) s2 += (v - m) * (v - m);
  const double sd = std::sqrt(s2 / static_cast<double>(pooled.size() - 1));
  calx::random::Stream rng(seed, 0x686d6d66u, 0, 0);  // "hmmf"
  HmmFit best;
  best.log_likelihood = -std::numeric_limits<double>::infinity();
  for (int r = 0; r < starts; r++) {
    GaussianHmm M;
    M.states = K;
    M.initial.assign(K, 1.0 / static_cast<double>(K));
    M.transition.assign(K * K, K > 1 ? 0.2 / static_cast<double>(K - 1) : 1.0);
    for (std::size_t k = 0; k < K; k++) M.transition[k * K + k] = K > 1 ? 0.8 : 1.0;
    for (std::size_t k = 0; k < K; k++) {
      const double q = (k + 0.5) / static_cast<double>(K);
      M.mean.push_back(pooled[static_cast<std::size_t>(q * static_cast<double>(pooled.size() - 1))] + (r == 0 ? 0.0 : rng.normal(0, 0.5 * sd)));
      M.sd.push_back(sd / std::sqrt(static_cast<double>(K)) * (r == 0 ? 1.0 : std::exp(rng.normal(0, 0.3))));
    }
    auto F = detail::em(M, sequences, 1e-3 * sd, max_iterations, tolerance, shared_sd);
    if (F.log_likelihood > best.log_likelihood) best = std::move(F);
  }
  // Order the states by their means.
  std::vector<std::size_t> order(K);
  for (std::size_t k = 0; k < K; k++) order[k] = k;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return best.model.mean[a] < best.model.mean[b]; });
  GaussianHmm O = best.model;
  for (std::size_t a = 0; a < K; a++) {
    O.initial[a] = best.model.initial[order[a]], O.mean[a] = best.model.mean[order[a]], O.sd[a] = best.model.sd[order[a]];
    for (std::size_t b = 0; b < K; b++) O.transition[a * K + b] = best.model.a(order[a], order[b]);
  }
  best.model = O;
  return best;
}

}  // namespace tyche::processes
