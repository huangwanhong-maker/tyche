// Checks of panel data on continuous-time Markov chains (processes/panel_markov.hpp).
//
// - Two states, 0 → 1 at rate λ: the log-likelihood in closed form, n₀ (−λΔ) + n₁ log(1 − e^{−λΔ}).
// - Current-status data simulated from a sequential chain of five stages by its holding times (no matrix
//   exponential): 6000 units of ages uniform on [0, 40]; the four rates recovered within three of Laplace's standard
//   errors, and the errors within 35 % of the spread of the estimates over 15 replicates.
// - Uniformisation's rows against calx's expm.
// - Refusals: a state beyond the chain, a negative interval.

#include <tyche/processes/panel_markov.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp = tyche::processes;

namespace {

std::vector<tp::PanelObservation> simulate(const std::vector<double>& rates, int units, std::uint64_t seed) {
  calx::random::Stream rng(seed, 0x70616e6cu, 0, 0);  // "panl"
  std::vector<tp::PanelObservation> data;
  for (int u = 0; u < units; u++) {
    const double age = rng.uniform(0, 40);
    double t = 0;
    std::size_t k = 0;
    while (k < rates.size()) {
      t += rng.exponential() / rates[k];
      if (t > age) break;
      k++;
    }
    data.push_back({0, k, age, 1});
  }
  return data;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  {
    const double lambda = 0.3, delta = 2.0;
    const std::vector<tp::PanelObservation> d = {{0, 0, delta, 3}, {0, 1, delta, 5}};
    const double ll = tp::panel_log_likelihood(tp::sequential_generator, {std::log(lambda)}, d);
    const double closed = 3 * (-lambda * delta) + 5 * std::log(1 - std::exp(-lambda * delta));
    s.check("two states: the log-likelihood in closed form", std::fabs(ll - closed), 1e-12);
  }
  const std::vector<double> truth = {0.15, 0.08, 0.12, 0.25};
  std::vector<double> theta0(4, std::log(0.1));
  const auto fit = tp::fit_panel(tp::sequential_generator, simulate(truth, 6000, 1), theta0);
  bool within = fit.converged;
  std::string got;
  for (std::size_t k = 0; k < 4; k++) {
    within = within && std::fabs(fit.theta[k] - std::log(truth[k])) <= 3 * fit.error[k];
    got += std::to_string(std::exp(fit.theta[k])) + " ";
  }
  s.require("current-status data of a sequential chain: the four rates recovered within three standard errors", within, got);
  // The errors against the spread of 40 replicates.
  std::vector<double> sum(4, 0.0), sq(4, 0.0), err(4, 0.0);
  const int reps = 15;
  for (int r = 0; r < reps; r++) {
    const auto f = tp::fit_panel(tp::sequential_generator, simulate(truth, 4000, 100 + static_cast<std::uint64_t>(r)), theta0);
    for (std::size_t k = 0; k < 4; k++) sum[k] += f.theta[k], sq[k] += f.theta[k] * f.theta[k], err[k] += f.error[k] / reps;
  }
  double worst = 0;
  for (std::size_t k = 0; k < 4; k++) {
    const double sd = std::sqrt((sq[k] - sum[k] * sum[k] / reps) / (reps - 1));
    worst = std::max(worst, std::fabs(err[k] / sd - 1));
  }
  s.check("Laplace's standard errors against the spread of 15 replicates of 4000 units (relative)", worst, 0.4);
  {
    const auto q = tp::sequential_generator({std::log(0.4), std::log(1.3), std::log(0.2)});
    double w = 0;
    for (const double delta : {0.0, 0.01, 1.7, 25.0, 900.0}) {
      const auto E = calx::linalg::expm(delta * q);
      for (std::size_t i = 0; i < 4; i++) {
        const auto r = tp::transition_row(q, i, delta);
        for (std::size_t j = 0; j < 4; j++) w = std::max(w, std::fabs(r[j] - E(i, j)));
      }
    }
    s.check("uniformisation's rows equal calx's expm (intervals 0 to 900, rates 0.2 to 1.3)", w, 1e-12);
  }
  {
    // Misclassification: a state recorded one off with probability 0.1 each side; the likelihood by hand for one
    // observation, and the rates recovered from records simulated with that error, including new units recorded worse.
    const std::vector<double> rates = {0.3, 0.2};
    tp::Misclassification E = {{0.9, 0.1, 0.0}, {0.1, 0.8, 0.1}, {0.0, 0.1, 0.9}};
    std::vector<double> th = {std::log(rates[0]), std::log(rates[1])};
    const auto row = tp::transition_row(tp::sequential_generator(th), 0, 2.0);
    const double by_hand = std::log(row[0] * E[0][1] + row[1] * E[1][1] + row[2] * E[2][1]);
    s.check("misclassification: one observation's likelihood by hand", std::fabs(tp::panel_log_likelihood(tp::sequential_generator, th, {{0, 1, 2.0, 1}}, E) - by_hand), 1e-12);
    calx::random::Stream rng(20261104, 0x6d697363u, 0, 0);  // "misc"
    std::vector<tp::PanelObservation> d;
    for (int u = 0; u < 5000; u++) {
      const double age = u < 200 ? 0.0 : rng.uniform(0, 20);
      double t = 0;
      std::size_t k = 0;
      while (k < 2) {
        t += rng.exponential() / rates[k];
        if (t > age) break;
        k++;
      }
      const double v = rng.uniform();
      std::size_t j = k;
      if (k == 0 && v < 0.1) j = 1;
      else if (k == 1) j = v < 0.1 ? 0 : (v < 0.2 ? 2 : 1);
      else if (k == 2 && v < 0.1) j = 1;
      d.push_back({0, j, age, 1});
    }
    bool impossible = false;
    try { tp::fit_panel(tp::sequential_generator, d, {0.0, 0.0}); } catch (const std::invalid_argument&) { impossible = true; }
    s.require("without misclassification, a new unit recorded worse than new is refused as impossible", impossible);
    const auto F = tp::fit_panel(tp::sequential_generator, d, {std::log(0.1), std::log(0.1)}, 500, E);
    bool ok = F.converged;
    for (std::size_t k = 0; k < 2; k++) ok = ok && std::fabs(F.theta[k] - std::log(rates[k])) <= 3 * F.error[k];
    s.require("with it, the rates are recovered within three standard errors", ok, std::to_string(std::exp(F.theta[0])) + " " + std::to_string(std::exp(F.theta[1])));
  }
  int refused = 0;
  try { tp::panel_log_likelihood(tp::sequential_generator, {0.0}, {{0, 3, 1.0, 1}}); } catch (const std::invalid_argument&) { refused++; }
  try { tp::panel_log_likelihood(tp::sequential_generator, {0.0}, {{0, 1, -1.0, 1}}); } catch (const std::invalid_argument&) { refused++; }
  s.require("refusals: a state beyond the chain, a negative interval", refused == 2);
  return s.done();
}
