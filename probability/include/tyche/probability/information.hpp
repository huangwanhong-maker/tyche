// Information: how far one law is from another, and how much of that a record shows.
//
// Laws here are finite: p_i ≥ 0 over n outcomes (histories, say), summing to one. A kernel K is a
// row-stochastic n × m matrix, K(i, y) the probability of the record y given the outcome i; a
// deterministic record (a function of the outcome, a coarsening) is a kernel of zeros and ones.
//
//   entropy             H(p) = −Σ p ln p, in nats
//   relative entropy    D(p ‖ q) = Σ p ln(p/q) (physis's, used here): +∞ unless p ≪ q
//   observed            pK, the law of the record
//   discrepancy         the chain rule of relative entropy through a record,
//
//                         D(p ‖ q) = D(pK ‖ qK) + E_{y ~ pK} D(p(· | y) ‖ q(· | y)),
//
//                       its first term what the record shows of the difference between p and q
//                       (the visible part) and its second what it hides. Both are ≥ 0; a record
//                       that is a garbling of another shows no more than it (data processing).
//   kernel_discrepancy  E_p D(K(i, ·) ‖ K′(i, ·)): the term a joint law gains when the two
//                       candidates also differ in how the outcome is recorded,
//                       D(p ⊗ K ‖ q ⊗ K′) = D(p ‖ q) + E_p D(K ‖ K′).
//   garbled             K G, a record passed through a second kernel
//
// For two Markov jump processes with rates k and k′ on one finite state space, the laws of their
// paths over [0, T] from a common law of the starting state differ by the integral of
//
//   jump_discrepancy_rate    Σ_i p_i(t) Σ_{j ≠ i} [k_ij ln(k_ij/k′_ij) − k_ij + k′_ij],
//
// p(t) the first process's law at t (Girsanov's formula for jump processes; Kipnis and Landim 1999,
// Appendix 1, Proposition 2.6). `path_discrepancy_rate` is that rate at the first process's
// stationary law: the nats per unit of time by which an event-complete record tells the two apart.
//
// References. Kullback and Leibler (1951), Ann. Math. Statist. 22: 79–86. Cover and Thomas (2006),
// Elements of Information Theory, 2nd ed., Theorems 2.5.3 (chain rule) and 2.8.1 (data processing).

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>
#include <physis/statistical_mechanics/master_equation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::probability {

using Kernel = calx::linalg::Rectangular;

inline void require_law(std::span<const double> p, const char* where, double tolerance = 1e-9) {
  double total = 0;
  for (double x : p) {
    if (!(x >= 0)) throw std::invalid_argument(std::string("tyche::probability::") + where + ": a law has non-negative entries");
    total += x;
  }
  if (std::fabs(total - 1) > tolerance) throw std::invalid_argument(std::string("tyche::probability::") + where + ": a law sums to one");
}

inline void require_kernel(const Kernel& K, std::size_t outcomes, const char* where, double tolerance = 1e-9) {
  if (K.rows != outcomes || K.cols == 0) throw std::invalid_argument(std::string("tyche::probability::") + where + ": a kernel has a row for each outcome");
  for (std::size_t i = 0; i < K.rows; i++) require_law(std::span<const double>(K.a.data() + i * K.cols, K.cols), where, tolerance);
}

inline double entropy(std::span<const double> p) {
  double h = 0;
  for (double x : p)
    if (x > 0) h -= x * std::log(x);
  return h;
}

using physis::statistical_mechanics::relative_entropy;

// D(Bernoulli(a) ‖ Bernoulli(b)).
inline double bernoulli_relative_entropy(double a, double b) {
  const double p[2] = {a, 1 - a}, q[2] = {b, 1 - b};
  return relative_entropy(p, q);
}

// The law of the record: (pK)_y = Σ_i p_i K(i, y).
inline std::vector<double> observed(std::span<const double> p, const Kernel& K) {
  std::vector<double> r(K.cols, 0.0);
  for (std::size_t i = 0; i < K.rows; i++)
    for (std::size_t y = 0; y < K.cols; y++) r[y] += p[i] * K(i, y);
  return r;
}

// The record that is a function of the outcome: outcome i is recorded as cell[i].
inline Kernel coarsening(std::span<const std::size_t> cell) {
  std::size_t m = 0;
  for (std::size_t c : cell) m = std::max(m, c + 1);
  Kernel K(cell.size(), m);
  for (std::size_t i = 0; i < cell.size(); i++) K(i, cell[i]) = 1;
  return K;
}

inline Kernel garbled(const Kernel& K, const Kernel& G) {
  if (K.cols != G.rows) throw std::invalid_argument("tyche::probability::garbled: the second kernel takes the first's records");
  return K * G;
}

struct Discrepancy {
  double total = 0, visible = 0, hidden = 0;  // total = visible + hidden, in nats
  std::vector<double> record_p, record_q;     // pK and qK
  std::vector<double> hidden_at;              // D(p(· | y) ‖ q(· | y)) for each record y of positive probability under p, else 0
};

// The chain rule through a record. Where p is not absolutely continuous with respect to q the
// total is +∞, and so is the part (visible or hidden) in which the failure lies.
inline Discrepancy discrepancy(std::span<const double> p, std::span<const double> q, const Kernel& K) {
  if (p.size() != q.size()) throw std::invalid_argument("tyche::probability::discrepancy: two laws on one set of outcomes");
  require_law(p, "discrepancy");
  require_law(q, "discrepancy");
  require_kernel(K, p.size(), "discrepancy");
  Discrepancy d;
  d.total = relative_entropy(p, q);
  d.record_p = observed(p, K);
  d.record_q = observed(q, K);
  d.visible = relative_entropy(d.record_p, d.record_q);
  d.hidden_at.assign(K.cols, 0.0);
  for (std::size_t y = 0; y < K.cols; y++) {
    if (d.record_p[y] <= 0) continue;
    double h = 0;
    for (std::size_t i = 0; i < p.size(); i++) {
      const double a = p[i] * K(i, y) / d.record_p[y];  // p(i | y)
      if (a <= 0) continue;
      const double b = d.record_q[y] > 0 ? q[i] * K(i, y) / d.record_q[y] : 0.0;
      if (b <= 0) { h = std::numeric_limits<double>::infinity(); break; }
      h += a * std::log(a / b);
    }
    d.hidden_at[y] = h;
    d.hidden += d.record_p[y] * h;
  }
  return d;
}

// E_p D(K(i, ·) ‖ K′(i, ·)).
inline double kernel_discrepancy(std::span<const double> p, const Kernel& K, const Kernel& K2) {
  require_kernel(K, p.size(), "kernel_discrepancy");
  require_kernel(K2, p.size(), "kernel_discrepancy");
  if (K.cols != K2.cols) throw std::invalid_argument("tyche::probability::kernel_discrepancy: two kernels to one set of records");
  double d = 0;
  for (std::size_t i = 0; i < p.size(); i++) {
    if (p[i] <= 0) continue;
    d += p[i] * relative_entropy(std::span<const double>(K.a.data() + i * K.cols, K.cols), std::span<const double>(K2.a.data() + i * K.cols, K.cols));
  }
  return d;
}

// The rate at which the laws of the paths of two jump processes part, at the law p of the first.
inline double jump_discrepancy_rate(const calx::linalg::Matrix& k, const calx::linalg::Matrix& k2, std::span<const double> p) {
  if (k.n != k2.n || p.size() != k.n) throw std::invalid_argument("tyche::probability::jump_discrepancy_rate: two sets of rates on one state space, and a law on it");
  double rate = 0;
  for (std::size_t i = 0; i < k.n; i++) {
    if (p[i] <= 0) continue;
    for (std::size_t j = 0; j < k.n; j++) {
      if (i == j) continue;
      const double a = k(i, j), b = k2(i, j);
      if (a < 0 || b < 0) throw std::invalid_argument("tyche::probability::jump_discrepancy_rate: rates are not negative");
      if (a > 0 && b == 0) return std::numeric_limits<double>::infinity();
      rate += p[i] * ((a > 0 ? a * std::log(a / b) : 0.0) - a + b);
    }
  }
  return rate;
}

inline double path_discrepancy_rate(const calx::linalg::Matrix& k, const calx::linalg::Matrix& k2) {
  return jump_discrepancy_rate(k, k2, physis::statistical_mechanics::stationary(k));
}

}  // namespace tyche::probability
