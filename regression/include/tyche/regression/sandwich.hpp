// The pieces of robust covariances that estimators share: columns joined, the "meat" of a sandwich (heteroskedasticity-robust,
// or summed within clusters), the sandwich itself, and a solve of a symmetric positive (semi-)definite system.
//
// For an estimator whose scores are u_i v_i (residuals times the rows of a design V), the robust covariance is B M B, with B the
// inverse of the Hessian (the bread) and M = Σ_g s_g s_gᵀ, s_g = Σ_{i∈g} u_i v_i (the meat), each observation its own group when
// there are no cluster labels: White's (1980) estimator, and Liang and Zeger's (1986) for clusters. The small-sample factors
// (n/(n − k), G/(G − 1) · (n − 1)/(n − k)) are the caller's (Cameron and Miller 2015). These pieces served nomos's instrumental
// variables, Poisson with absorbed effects and multinomial logit; they are tyche's since peira's stage 2 (tyche's decision 0014).

#pragma once

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/rectangular.hpp>

#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace tyche::regression {

using calx::linalg::Matrix;
using calx::linalg::Rectangular;

// [a b], the columns of a then those of b (either may have none).
inline Rectangular join_columns(const Rectangular& a, const Rectangular& b) {
  const std::size_t n = a.cols ? a.rows : b.rows;
  Rectangular m(n, a.cols + b.cols);
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < a.cols; j++) m(i, j) = a(i, j);
    for (std::size_t j = 0; j < b.cols; j++) m(i, a.cols + j) = b(i, j);
  }
  return m;
}

// Σ_g s_g s_gᵀ with s_g = Σ_{i∈g} u_i v_i (each observation its own group when there are no labels).
inline Matrix sandwich_meat(const Rectangular& V, std::span<const double> u, std::span<const std::int64_t> clusters, std::size_t& groups) {
  const std::size_t n = V.rows, k = V.cols;
  Matrix M(k);
  std::map<std::int64_t, std::vector<double>> sums;
  for (std::size_t i = 0; i < n; i++) {
    if (clusters.empty()) {
      for (std::size_t a = 0; a < k; a++)
        for (std::size_t b = 0; b < k; b++) M(a, b) += u[i] * u[i] * V(i, a) * V(i, b);
    } else {
      auto& s = sums[clusters[i]];
      if (s.empty()) s.assign(k, 0.0);
      for (std::size_t a = 0; a < k; a++) s[a] += u[i] * V(i, a);
    }
  }
  if (clusters.empty()) {
    groups = n;
  } else {
    for (const auto& [label, s] : sums)
      for (std::size_t a = 0; a < k; a++)
        for (std::size_t b = 0; b < k; b++) M(a, b) += s[a] * s[b];
    groups = sums.size();
  }
  return M;
}

inline Matrix sandwich(const Matrix& B, const Matrix& M) { return B * M * B; }

inline std::vector<double> solve_spd(const Matrix& A, std::span<const double> b) {
  const auto c = calx::linalg::cholesky(A);
  if (c.positive_definite) return calx::linalg::cholesky_solve(c, b);
  Matrix R(A.n);
  for (std::size_t i = 0; i < A.n; i++) R(i, 0) = b[i];
  const Matrix S = calx::linalg::solve(A, R);
  std::vector<double> x(A.n);
  for (std::size_t i = 0; i < A.n; i++) x[i] = S(i, 0);
  return x;
}

}  // namespace tyche::regression
