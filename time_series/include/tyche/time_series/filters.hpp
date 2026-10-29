// The Hodrick–Prescott filter: a series split into a smooth trend and a cycle.
//
// The trend τ minimises Σ_t (y_t − τ_t)² + λ Σ_{t=2}^{n−1} (τ_{t+1} − 2τ_t + τ_{t−1})² (Hodrick and
// Prescott 1997, "Postwar U.S. business cycles: an empirical investigation", J. Money, Credit and
// Banking 29, 1–16, §2), so that (I + λ KᵀK) τ = y with K the (n − 2) × n second-difference
// matrix. The matrix is symmetric positive definite and pentadiagonal; it is solved by calx's band
// Cholesky factorisation in O(n) (calx::linalg::band_solve), exact to rounding (the condition
// number grows like λ, about 10⁴ at λ = 1600, so the trend keeps some 12 digits). The cycle is
// y − τ. λ = 1600 for quarterly data (Hodrick and Prescott's choice); Ravn and Uhlig (2002, Rev.
// Econ. Stat. 84, 371–376) scale it by the fourth power of the frequency: 6.25 for annual and
// 129 600 for monthly data.
//
// Properties checked: K annihilates constants and linear trends, so a linear series is its own
// trend, and the cycle is orthogonal to both (Σ c_t = Σ t c_t = 0, since c = λ KᵀK τ).

#pragma once

#include <calx/linalg/cholesky.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::time_series {

struct HpFilter {
  std::vector<double> trend, cycle;
};

inline HpFilter hodrick_prescott(std::span<const double> y, double lambda = 1600) {
  const std::size_t n = y.size();
  if (n < 3) throw std::invalid_argument("tyche::time_series::hodrick_prescott: at least 3 observations");
  if (!(lambda >= 0)) throw std::invalid_argument("tyche::time_series::hodrick_prescott: λ ≥ 0");
  calx::linalg::SymmetricBand A(n, 2);
  for (std::size_t i = 0; i < n; i++) A(i, i) = 1;
  const double d[3] = {1, -2, 1};
  for (std::size_t t = 0; t + 2 < n; t++)  // row t of K: (1, −2, 1) at columns t, t + 1, t + 2
    for (int a = 0; a < 3; a++)
      for (int b = 0; b <= a; b++) A(t + a, t + b) += lambda * d[a] * d[b];
  HpFilter f;
  f.trend = calx::linalg::band_solve(A, y);
  f.cycle.resize(n);
  for (std::size_t t = 0; t < n; t++) f.cycle[t] = y[t] - f.trend[t];
  return f;
}

}  // namespace tyche::time_series
