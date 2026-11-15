// The value of information: what knowing more before choosing is worth.
//
// A decision maker with a prior p on the states chooses the act of greatest expected utility, worth max_a Σ_s p(s) u(a, s).
// Told the state before choosing, they would take the best act in each state, worth Σ_s p(s) max_a u(a, s); the difference is
// the expected value of perfect information, EVPI (Raiffa and Schlaifer 1961, Applied Statistical Decision Theory, ch. 4). A
// test whose signal y has likelihood P(y | s) is worth, in prospect, Σ_y P(y) max_a E[u(a, ·) | y] − max_a E[u(a, ·)]: the
// expected value of sample information, EVSI, found by preposterior analysis (ibid.). Information that can be ignored has
// non-negative value (Good 1967, On the principle of total evidence, British Journal for the Philosophy of Science 17), and no
// test is worth more than perfect information: 0 ≤ EVSI ≤ EVPI, with EVSI = EVPI for a test that reveals the state and EVSI = 0
// for one whose signal is independent of it. The checks hold all four on random problems.
//
// For a choice between adopting and not adopting, with a net benefit B ~ N(μ, σ²), EVPI is σ times the unit normal loss at
// |μ|/σ, σ φ(|μ|/σ) − |μ| Φ(−|μ|/σ); a study of n units of sampling variance s² each moves the posterior mean by a normal
// amount of variance σ̃² = σ⁴/(σ² + s²/n), and its EVSI is the same loss at σ̃ (Raiffa and Schlaifer 1961, ch. 5).

#pragma once

#include <tyche/decision/choice.hpp>
#include <tyche/probability/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::decision {

// The expected value of perfect information.
inline double evpi(const Table& t, std::span<const double> p) {
  validate(t);
  validate_law(p, t.n_states(), "evpi");
  double informed = 0;
  for (std::size_t s = 0; s < t.n_states(); s++) {
    double best = t.u[0][s];
    for (std::size_t a = 1; a < t.n_acts(); a++) best = std::max(best, t.u[a][s]);
    informed += p[s] * best;
  }
  const auto e = expectations(t, p);
  return informed - *std::max_element(e.begin(), e.end());
}

// A test: its signals and, for each state, the law of the signal.
struct Test {
  std::vector<std::string> signals;
  std::vector<std::vector<double>> likelihood;  // likelihood[s][y] = P(y | s)
};

struct Preposterior {
  double evsi = 0;
  double prior_value = 0;                      // max_a E[u(a)] without the test
  std::vector<double> p_signal;                // P(y)
  std::vector<std::vector<double>> posterior;  // posterior[y][s]; empty for a signal of probability 0
  std::vector<std::size_t> act;                // the act best after each signal (the first of the best)
};

// The preposterior analysis of a test: what each signal would make one believe and do, and what the test is worth.
inline Preposterior evsi(const Table& t, std::span<const double> p, const Test& test) {
  validate(t);
  validate_law(p, t.n_states(), "evsi");
  if (test.likelihood.size() != t.n_states()) throw std::invalid_argument("tyche::decision::evsi: a law of the signal for each state");
  const std::size_t Y = test.likelihood[0].size();
  for (const auto& row : test.likelihood) validate_law(row, Y, "evsi (a likelihood)");
  Preposterior out;
  const auto e = expectations(t, p);
  out.prior_value = *std::max_element(e.begin(), e.end());
  out.p_signal.assign(Y, 0);
  out.posterior.assign(Y, {});
  out.act.assign(Y, 0);
  double with = 0;
  for (std::size_t y = 0; y < Y; y++) {
    double py = 0;
    for (std::size_t s = 0; s < t.n_states(); s++) py += p[s] * test.likelihood[s][y];
    out.p_signal[y] = py;
    if (py <= 0) continue;
    std::vector<double> q(t.n_states());
    for (std::size_t s = 0; s < t.n_states(); s++) q[s] = p[s] * test.likelihood[s][y] / py;
    double best = 0;
    for (std::size_t a = 0; a < t.n_acts(); a++) {
      double v = 0;
      for (std::size_t s = 0; s < t.n_states(); s++) v += q[s] * t.u[a][s];
      if (a == 0 || v > best + tie) best = v, out.act[y] = a;
    }
    out.posterior[y] = std::move(q);
    with += py * best;
  }
  out.evsi = with - out.prior_value;
  return out;
}

// ---- Normal net benefit -------------------------------------------------------------------------------------------------------

// σ L(|μ|/σ), with L(z) = φ(z) − z (1 − Φ(z)) the unit normal loss.
inline double normal_loss(double mean, double sd) {
  if (!(sd > 0) || !std::isfinite(mean)) throw std::invalid_argument("tyche::decision::normal_loss: a finite mean and a positive standard deviation");
  const probability::Normal z{0, 1};
  const double a = std::abs(mean) / sd;
  return sd * z.pdf(a) - std::abs(mean) * z.cdf(-a);
}

// EVPI for adopting or not, the net benefit N(mean, sd²).
inline double evpi_normal(double mean, double sd) { return normal_loss(mean, sd); }

// EVSI of a study of n units, each of sampling variance s2, for the same choice.
inline double evsi_normal(double mean, double sd, double s2, double n) {
  if (!(s2 > 0) || !(n > 0)) throw std::invalid_argument("tyche::decision::evsi_normal: a positive sampling variance and number of units");
  const double var = sd * sd;
  return normal_loss(mean, std::sqrt(var * var / (var + s2 / n)));
}

}  // namespace tyche::decision
