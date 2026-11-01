// Non-negative least squares: min ‖Ax − b‖₂ subject to x ≥ 0.
//
// THE CONDITIONS. x is optimal exactly when, with the dual w = Aᵀ(b − Ax) (the negative gradient of
// ½‖Ax − b‖²), x ≥ 0, w ≤ 0 and x_j w_j = 0 for every j (Karush–Kuhn–Tucker; Lawson and Hanson 1974,
// Solving Least Squares Problems, Prentice-Hall, ch. 23). The solution is unique when A
// has full column rank; its residual Ax − b is unique always.
//
// THE ALGORITHM. Lawson and Hanson's (1974, ch. 23, §3, Algorithm NNLS) active-set method: the passive
// set P (the coordinates free to be positive) grows by the coordinate of largest dual w_j > 0; the
// unconstrained least squares on the columns in P gives z; if some z_j ≤ 0 the step from x towards z
// stops where the first coordinate reaches 0 (α = min x_j/(x_j − z_j) over z_j ≤ 0), which then leaves
// P, and the least squares is solved again; when every z_j > 0, x = z and the duals are recomputed. It
// stops when every w_j for j outside P is ≤ tolerance. Each least-squares problem is solved by calx's
// Householder QR (never the normal equations). The method terminates in finitely many steps in
// exact arithmetic (ibid., §3); here an iteration limit (default 3n outer steps, as Lawson and Hanson
// suggest) guards against cycling by rounding. The tolerance on w is by default
// 10 max(m, n) ε ‖A‖₁ ‖b‖₂.

#pragma once

#include <tyche/regression/linear.hpp>

#include <calx/linalg/qr.hpp>
#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::regression {

struct NnlsOptions {
  std::size_t max_iterations = 0;  // outer steps (coordinates entering P); 0: 3n
  double tolerance = -1;           // on the duals w_j; < 0: 10 max(m, n) ε ‖A‖₁ ‖b‖₂
};

struct NnlsResult {
  std::vector<double> x;           // x ≥ 0
  std::vector<double> dual;        // w = Aᵀ(b − Ax): 0 where x_j > 0, ≤ 0 elsewhere at the optimum
  double residual_norm = 0;        // ‖Ax − b‖₂
  std::size_t iterations = 0;
  bool converged = false;
};

inline NnlsResult nnls(const Rectangular& A, std::span<const double> b, const NnlsOptions& o = {}) {
  const std::size_t m = A.rows, n = A.cols;
  if (b.size() != m) throw std::invalid_argument("tyche::regression::nnls: b has one entry per row of A");
  for (const double v : A.a)
    if (!std::isfinite(v)) throw std::invalid_argument("tyche::regression::nnls: A finite");
  for (const double v : b)
    if (!std::isfinite(v)) throw std::invalid_argument("tyche::regression::nnls: b finite");
  NnlsResult r;
  r.x.assign(n, 0.0);
  double norm1 = 0, bnorm = 0;
  for (std::size_t j = 0; j < n; j++) {
    double s = 0;
    for (std::size_t i = 0; i < m; i++) s += std::fabs(A(i, j));
    norm1 = std::max(norm1, s);
  }
  for (const double v : b) bnorm += v * v;
  bnorm = std::sqrt(bnorm);
  const double tol = o.tolerance >= 0 ? o.tolerance : 10 * static_cast<double>(std::max(m, n)) * std::numeric_limits<double>::epsilon() * norm1 * bnorm;
  const std::size_t limit = o.max_iterations > 0 ? o.max_iterations : 3 * std::max<std::size_t>(n, 1);
  std::vector<bool> passive(n, false);
  auto residual = [&]() {
    std::vector<double> res(b.begin(), b.end());
    for (std::size_t i = 0; i < m; i++)
      for (std::size_t j = 0; j < n; j++)
        if (r.x[j] != 0) res[i] -= A(i, j) * r.x[j];
    return res;
  };
  auto duals = [&]() {
    const auto res = residual();
    return calx::linalg::multiply_transpose(A, res);
  };
  // The unconstrained least squares on the passive columns.
  auto subproblem = [&]() {
    std::vector<std::size_t> cols;
    for (std::size_t j = 0; j < n; j++)
      if (passive[j]) cols.push_back(j);
    Rectangular Ap(m, cols.size());
    for (std::size_t i = 0; i < m; i++)
      for (std::size_t k = 0; k < cols.size(); k++) Ap(i, k) = A(i, cols[k]);
    const auto zp = calx::linalg::least_squares(Ap, b);
    std::vector<double> z(n, 0.0);
    for (std::size_t k = 0; k < cols.size(); k++) z[cols[k]] = zp[k];
    return z;
  };
  std::vector<double> w = duals();
  while (r.iterations < limit) {
    // The most promising coordinate outside P.
    std::size_t t = n;
    double best = tol;
    for (std::size_t j = 0; j < n; j++)
      if (!passive[j] && w[j] > best) {
        best = w[j];
        t = j;
      }
    if (t == n) {
      r.converged = true;
      break;
    }
    r.iterations++;
    passive[t] = true;
    for (int inner = 0; inner < static_cast<int>(3 * n + 3); inner++) {
      const auto z = subproblem();
      bool feasible = true;
      for (std::size_t j = 0; j < n; j++)
        if (passive[j] && z[j] <= 0) feasible = false;
      if (feasible) {
        r.x = z;
        break;
      }
      // Step from x towards z until a coordinate reaches 0.
      double alpha = 1;
      for (std::size_t j = 0; j < n; j++)
        if (passive[j] && z[j] <= 0) alpha = std::min(alpha, r.x[j] / (r.x[j] - z[j]));
      for (std::size_t j = 0; j < n; j++) {
        if (!passive[j]) continue;
        r.x[j] += alpha * (z[j] - r.x[j]);
        if (r.x[j] <= 0 || (z[j] <= 0 && r.x[j] <= tol)) {
          r.x[j] = 0;
          passive[j] = false;
        }
      }
      bool any = false;
      for (std::size_t j = 0; j < n; j++) any = any || passive[j];
      if (!any) break;
    }
    w = duals();
    // A coordinate that just entered but could not stay: stop it from re-entering at once.
    if (!passive[t] && w[t] > tol) w[t] = 0;
  }
  r.dual = duals();
  const auto res = residual();
  double s = 0;
  for (const double v : res) s += v * v;
  r.residual_norm = std::sqrt(s);
  return r;
}

// NNLS for every column of B: X with column k the solution for B's column k (n × q).
inline Rectangular nnls_columns(const Rectangular& A, const Rectangular& B, const NnlsOptions& o = {}) {
  if (B.rows != A.rows) throw std::invalid_argument("tyche::regression::nnls_columns: B has A's rows");
  Rectangular X(A.cols, B.cols);
  for (std::size_t k = 0; k < B.cols; k++) {
    const auto col = B.column(k);
    const auto r = nnls(A, col, o);
    for (std::size_t j = 0; j < A.cols; j++) X(j, k) = r.x[j];
  }
  return X;
}

}  // namespace tyche::regression
