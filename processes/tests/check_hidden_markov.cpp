// Checks of Gaussian hidden Markov models (processes/hidden_markov.hpp).
//
// - Against brute force over all 3⁶ paths of a three-state model and six observations (one missing): the
//   log-likelihood, every smoothed marginal, and the most probable path.
// - The filtered law at the last step equals the smoothed one; every law sums to 1.
// - The fit to 300 sequences of 60 steps simulated from a two-state model whose observations overlap (5 % missing):
//   its log-likelihood never falls along EM, and it recovers the means and deviations within 0.15 and the staying
//   probabilities within 0.05 (about two and a half of their standard errors), the states ordered by their means.
// - With a shared deviation (Hamilton's model), three states' means and their one σ recovered.

#include <tyche/processes/hidden_markov.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace tp = tyche::processes;

int main() {
  calx::verification::Suite s;
  tp::GaussianHmm M;
  M.states = 3;
  M.initial = {0.5, 0.3, 0.2};
  M.transition = {0.7, 0.2, 0.1, 0.15, 0.7, 0.15, 0.05, 0.25, 0.7};
  M.mean = {-1.5, 1.0, 4.0};
  M.sd = {1.2, 0.8, 1.5};
  const std::vector<double> y = {0.3, -2.1, std::nan(""), 3.6, 4.4, 0.9};
  const auto D = tp::decode(M, y);
  // Brute force.
  const std::size_t T = y.size(), K = 3;
  double total = 0, best = -1;
  std::vector<std::size_t> best_path;
  std::vector<std::vector<double>> marg(T, std::vector<double>(K, 0.0));
  std::vector<std::size_t> path(T, 0);
  for (std::size_t code = 0; code < 729; code++) {
    std::size_t c = code;
    for (std::size_t t = 0; t < T; t++) path[t] = c % 3, c /= 3;
    double p = M.initial[path[0]] * M.emission(path[0], y[0]);
    for (std::size_t t = 1; t < T; t++) p *= M.a(path[t - 1], path[t]) * M.emission(path[t], y[t]);
    total += p;
    for (std::size_t t = 0; t < T; t++) marg[t][path[t]] += p;
    if (p > best) best = p, best_path = path;
  }
  double worst = 0;
  for (std::size_t t = 0; t < T; t++)
    for (std::size_t k = 0; k < K; k++) worst = std::max(worst, std::fabs(D.smoothed[t][k] - marg[t][k] / total));
  s.check("the log-likelihood against the sum over all 729 paths", std::fabs(D.log_likelihood - std::log(total)), 1e-12);
  s.check("every smoothed marginal against brute force", worst, 1e-12);
  s.require("the most probable path against brute force", D.path == best_path);
  double last = 0, sums = 0;
  for (std::size_t k = 0; k < K; k++) last = std::max(last, std::fabs(D.filtered[T - 1][k] - D.smoothed[T - 1][k]));
  for (const auto& row : D.filtered) {
    double t = 0;
    for (const double v : row) t += v;
    sums = std::max(sums, std::fabs(t - 1));
  }
  s.check("the last filtered law is the last smoothed one, and every law sums to 1", std::max(last, sums), 1e-14);

  // ---- the fit ---------------------------------------------------------------------------------------------------
  tp::GaussianHmm truth;
  truth.states = 2;
  truth.initial = {0.2, 0.8};
  truth.transition = {0.6, 0.4, 0.1, 0.9};
  truth.mean = {-1.0, 3.0};
  truth.sd = {2.0, 1.5};
  calx::random::Stream rng(20261104, 0x686d6d73u, 0, 0);  // "hmms"
  std::vector<std::vector<double>> seqs;
  for (int n = 0; n < 300; n++) {
    std::vector<double> q;
    std::size_t k = rng.uniform() < truth.initial[0] ? 0 : 1;
    for (int t = 0; t < 60; t++) {
      if (t > 0) k = rng.uniform() < truth.a(k, 0) ? 0 : 1;
      q.push_back(rng.uniform() < 0.05 ? std::nan("") : rng.normal(truth.mean[k], truth.sd[k]));
    }
    seqs.push_back(q);
  }
  const auto F = tp::fit_hmm(seqs, 2);
  bool rising = true;
  for (std::size_t i = 1; i < F.trace.size(); i++) rising = rising && F.trace[i] >= F.trace[i - 1] - 1e-7 * std::fabs(F.trace[i - 1]);
  s.require("EM's log-likelihood never falls (" + std::to_string(F.iterations) + " iterations)", rising && F.converged);
  const auto& m = F.model;
  const bool ok = std::fabs(m.mean[0] + 1) < 0.15 && std::fabs(m.mean[1] - 3) < 0.15 && std::fabs(m.sd[0] - 2) < 0.15 && std::fabs(m.sd[1] - 1.5) < 0.15 &&
                  std::fabs(m.a(0, 0) - 0.6) < 0.05 && std::fabs(m.a(1, 1) - 0.9) < 0.05;
  s.require("the fit recovers the two-state model, its states ordered by their means", ok,
            "means " + std::to_string(m.mean[0]) + ", " + std::to_string(m.mean[1]) + "; deviations " + std::to_string(m.sd[0]) + ", " + std::to_string(m.sd[1]) +
                "; staying " + std::to_string(m.a(0, 0)) + ", " + std::to_string(m.a(1, 1)));
  {
    // A shared deviation (Hamilton's model): three states of means −2, 0.5 and 3, σ = 1.2 for all, recovered.
    tp::GaussianHmm h;
    h.states = 3, h.initial = {0.2, 0.6, 0.2}, h.transition = {0.7, 0.25, 0.05, 0.1, 0.8, 0.1, 0.05, 0.25, 0.7}, h.mean = {-2, 0.5, 3}, h.sd = {1.2, 1.2, 1.2};
    std::vector<std::vector<double>> q;
    for (int n = 0; n < 200; n++) {
      std::vector<double> y;
      std::size_t k = 1;
      for (int t = 0; t < 60; t++) {
        const double u = rng.uniform();
        double c = 0;
        for (std::size_t j = 0; j < 3; j++)
          if ((c += h.a(k, j)) > u) { k = j; break; }
        y.push_back(rng.normal(h.mean[k], 1.2));
      }
      q.push_back(y);
    }
    const auto G = tp::fit_hmm(q, 3, 5, 500, 1e-9, 20261104, true);
    const auto& g = G.model;
    s.require("a shared deviation: one σ for all states, and the means and σ recovered within 0.15",
              g.sd[0] == g.sd[1] && g.sd[1] == g.sd[2] && std::fabs(g.sd[0] - 1.2) < 0.15 && std::fabs(g.mean[0] + 2) < 0.15 && std::fabs(g.mean[1] - 0.5) < 0.15 && std::fabs(g.mean[2] - 3) < 0.15,
              "means " + std::to_string(g.mean[0]) + ", " + std::to_string(g.mean[1]) + ", " + std::to_string(g.mean[2]) + "; σ " + std::to_string(g.sd[0]));
  }
  return s.done();
}
