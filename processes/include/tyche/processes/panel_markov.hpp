// Continuous-time Markov chains observed at discrete times: the likelihood of panel data and its maximum.
//
// A chain on states 0 … n−1 with generator Q(θ) (rows summing to zero, off-diagonal rates ≥ 0) is observed in state
// i and, an interval Δ later, in state j: the observation's likelihood is P(Δ)_{ij} = [exp(Q Δ)]_{ij}, and a panel's
// log-likelihood is Σ_k w_k log P(Δ_k)_{i_k j_k} (Kalbfleisch and Lawless 1985, "The analysis of panel data under a
// Markov assumption", JASA 80, 863–871). Current-status data, each unit seen once at a known age from a known
// starting state, are the special case i_k = the start, Δ_k = the age. Each observation needs one row of exp(QΔ),
// computed by uniformisation (Jensen 1953; Grassmann 1977): with Λ ≥ max_i |Q_ii| and M = I + Q/Λ a stochastic
// matrix, e_i exp(QΔ) = Σ_{k≥0} e^{−ΛΔ} (ΛΔ)^k / k! · e_i M^k, a sum of non-negative terms, cut where the Poisson
// tail falls far below 10⁻¹⁵, at ΛΔ + 10 √(ΛΔ) + 20 terms (calx's expm for ΛΔ above 500, where e^{−ΛΔ} underflows). The maximum comes from
// calx's BFGS on numerical gradients (θ is the caller's: log-rates keep them positive); the errors from the inverse
// of the Hessian of −ℓ by differences at the maximum (Laplace).
//
// Misclassification. When what is recorded is the state read with error, E(j | k) the chance of recording j in the
// state k, an observation's likelihood is Σ_k [exp(QΔ)]_{ik} E(j | k): the hidden Markov model of panel data (Jackson,
// Sharples, Thompson, Duffy and Couto 2003, "Multistate Markov models for disease progression with classification
// error", J. Roy. Statist. Soc. D 52, 193–209). The starting state is taken as known.
//
// Caveat. The likelihood is that of the chain's law only if the times of observation do not depend on the states
// (non-informative observation); and a chain fitted to current-status data is identified only through the spread of
// ages.

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/expm.hpp>
#include <calx/optimisation/quasi_newton.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <utility>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::processes {

struct PanelObservation {
  std::size_t from = 0, to = 0;
  double interval = 0;  // Δ ≥ 0, in the generator's unit of time
  double weight = 1;
};

using Generator = std::function<calx::linalg::Matrix(const std::vector<double>&)>;
using Misclassification = std::vector<std::vector<double>>;  // E[k][j]: record j in state k; empty: no error

// Σ w log P(Δ)_{from,to}; −∞ if an observation is impossible under θ.
// The row e_i exp(QΔ) by uniformisation.
inline std::vector<double> transition_row(const calx::linalg::Matrix& q, std::size_t i, double delta) {
  const std::size_t n = q.n;
  double lambda = 0;
  for (std::size_t k = 0; k < n; k++) lambda = std::max(lambda, -q(k, k));
  std::vector<double> row(n, 0.0);
  if (lambda == 0 || delta == 0) {
    row[i] = 1;
    return row;
  }
  const double mean = lambda * delta;
  if (mean > 500) {
    const auto E = calx::linalg::expm(delta * q);
    for (std::size_t j = 0; j < n; j++) row[j] = E(i, j);
    return row;
  }
  std::vector<double> v(n, 0.0), w(n);
  v[i] = 1;
  // Terms to mean + 10 √mean + 20: the Poisson tail beyond lies far below 10⁻¹⁵. (A test on 1 − Σ weights would
  // never be met, the sum's rounding keeping it above 10⁻¹⁵.)
  const int last = static_cast<int>(std::ceil(mean + 10 * std::sqrt(mean) + 20));
  double weight = std::exp(-mean);
  for (int k = 0; k <= last; k++) {
    for (std::size_t j = 0; j < n; j++) row[j] += weight * v[j];
    // v ← v M, M = I + Q/Λ.
    for (std::size_t j = 0; j < n; j++) {
      double x = v[j];
      for (std::size_t l = 0; l < n; l++) x += v[l] * q(l, j) / lambda;
      w[j] = x;
    }
    v.swap(w);
    weight *= mean / (k + 1);
  }
  return row;
}

inline double panel_log_likelihood(const Generator& Q, const std::vector<double>& theta, const std::vector<PanelObservation>& data,
                                   const Misclassification& E = {}) {
  const calx::linalg::Matrix q = Q(theta);
  std::map<std::pair<std::size_t, double>, std::vector<double>> cache;
  double ll = 0;
  for (const auto& d : data) {
    if (d.from >= q.n || d.to >= q.n || !(d.interval >= 0)) throw std::invalid_argument("tyche::processes::panel_log_likelihood: states within the chain and intervals ≥ 0");
    const auto key = std::make_pair(d.from, d.interval);
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, transition_row(q, d.from, d.interval)).first;
    double p = 0;
    if (E.empty()) p = it->second[d.to];
    else
      for (std::size_t k = 0; k < q.n; k++) p += it->second[k] * E[k][d.to];
    if (!(p > 0)) return -std::numeric_limits<double>::infinity();
    ll += d.weight * std::log(p);
  }
  return ll;
}

struct PanelFit {
  std::vector<double> theta, error;  // the maximum, and Laplace's standard errors (NaN if the Hessian is not positive)
  double log_likelihood = 0;
  int iterations = 0;
  bool converged = false;
};

inline PanelFit fit_panel(const Generator& Q, const std::vector<PanelObservation>& data, std::vector<double> theta0, int max_iterations = 500,
                          const Misclassification& E = {}) {
  // Parameters beyond ±30 (log-rates of 10⁻¹³ to 10¹³ a unit of time) are refused: there the generator's exponential
  // overflows into numbers that are finite but meaningless, which a line search could otherwise accept.
  const auto f = [&](std::span<const double> th) {
    for (const double x : th)
      if (!(std::fabs(x) <= 30)) return 1e300;
    const double ll = panel_log_likelihood(Q, std::vector<double>(th.begin(), th.end()), data, E);
    return std::isfinite(ll) ? -ll : 1e300;
  };
  // An observation impossible at the start would leave the search on a flat 10³⁰⁰: refused, with the reason.
  if (!std::isfinite(panel_log_likelihood(Q, theta0, data, E)))
    throw std::invalid_argument("tyche::processes::fit_panel: an observation is impossible at the starting parameters (a state the chain cannot reach in "
                                "its interval; misclassification, or another start, may allow it)");
  // The gradient's tolerance scales with the data: −ℓ grows with the weight of the observations, and central
  // differences cannot resolve an absolute 10⁻⁶ of a sum of thousands of terms.
  double weight = 0;
  for (const auto& d : data) weight += d.weight;
  calx::optimisation::BfgsOptions opt;
  opt.max_iterations = max_iterations;
  opt.gradient_tolerance = 1e-7 * std::max(1.0, weight);
  const auto r = calx::optimisation::bfgs(f, std::move(theta0), opt);
  PanelFit out;
  out.theta = r.x, out.iterations = r.iterations, out.converged = r.converged, out.log_likelihood = -r.value;
  const auto H = calx::optimisation::hessian_by_differences(f, out.theta);
  out.error.assign(out.theta.size(), std::numeric_limits<double>::quiet_NaN());
  // The inverse of H by Gauss–Jordan (small: one row per parameter).
  const std::size_t p = out.theta.size();
  std::vector<std::vector<double>> A(p, std::vector<double>(2 * p, 0.0));
  for (std::size_t i = 0; i < p; i++) {
    for (std::size_t j = 0; j < p; j++) A[i][j] = H(i, j);
    A[i][p + i] = 1;
  }
  bool ok = true;
  for (std::size_t c = 0; c < p && ok; c++) {
    std::size_t piv = c;
    for (std::size_t r2 = c + 1; r2 < p; r2++)
      if (std::fabs(A[r2][c]) > std::fabs(A[piv][c])) piv = r2;
    if (std::fabs(A[piv][c]) < 1e-300) { ok = false; break; }
    std::swap(A[piv], A[c]);
    const double d = A[c][c];
    for (auto& x : A[c]) x /= d;
    for (std::size_t r2 = 0; r2 < p; r2++)
      if (r2 != c) {
        const double m = A[r2][c];
        for (std::size_t k = 0; k < 2 * p; k++) A[r2][k] -= m * A[c][k];
      }
  }
  if (ok)
    for (std::size_t i = 0; i < p; i++)
      if (A[i][p + i] > 0) out.error[i] = std::sqrt(A[i][p + i]);
  return out;
}

// A sequential chain: states 0 → 1 → … → n−1 (absorbing) with rates exp(θ_k) out of state k: the generator of
// deterioration through ordered classes, and of any chain of stages.
inline calx::linalg::Matrix sequential_generator(const std::vector<double>& theta) {
  const std::size_t n = theta.size() + 1;
  calx::linalg::Matrix q(n);
  for (std::size_t k = 0; k + 1 < n; k++) {
    const double r = std::exp(theta[k]);
    q(k, k + 1) = r, q(k, k) = -r;
  }
  return q;
}

}  // namespace tyche::processes
