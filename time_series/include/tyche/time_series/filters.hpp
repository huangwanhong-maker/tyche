// The Hodrick–Prescott filter: a series split into a smooth trend and a cycle.
//
// The trend τ minimises Σ_t (y_t − τ_t)² + λ Σ_{t=2}^{n−1} (τ_{t+1} − 2τ_t + τ_{t−1})² (Hodrick and
// Prescott 1997, "Postwar U.S. business cycles: an empirical investigation", J. Money, Credit and
// Banking 29, 1–16, §2), so that (I + λ KᵀK) τ = y with K the (n − 2) × n second-difference
// matrix. λ = 1600 for quarterly data (Hodrick and Prescott's choice); Ravn and Uhlig (2002, Rev.
// Econ. Stat. 84, 371–376) scale it by the fourth power of the frequency: 6.25 for annual and
// 129 600 for monthly data.
//
// The cycle is solved for first. With w = Kτ, applying K to the normal equations gives
//   (I + λ KKᵀ) w = K y,      c = y − τ = λ Kᵀ w,      τ = y − c,
// an (n − 2) × (n − 2) system, symmetric positive definite and pentadiagonal (KKᵀ has 6 on its
// diagonal, −4 and 1 beside it), solved by calx's band Cholesky factorisation in O(n)
// (calx::linalg::band_solve). Its condition number grows like 1 + 16λ, as the trend's system's
// does, but its right-hand side is the second differences of y, so the cycle's error is relative
// to the cycle, not to the series; the identities hold to rounding by construction: y = τ + c
// exactly, and Σ c_t = Σ t c_t = 0 up to the rounding of the sums, since every column of Kᵀ sums
// to zero and is orthogonal to t (K annihilates constants and linear trends). Solving for the trend
// first, as (I + λKᵀK)τ = y, keeps only some 12 digits of τ, and c = y − τ inherits that error at
// the scale of y.
//
// Properties checked: a linear series is its own trend; the cycle sums to zero and is orthogonal
// to time; statsmodels' hpfilter on the same series; λ = 0 leaves the series.

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
  const std::size_t m = n - 2;
  // B = I + λ KKᵀ, row i of K being (1, −2, 1) at columns i, i + 1, i + 2.
  calx::linalg::SymmetricBand B(m, 2);
  for (std::size_t i = 0; i < m; i++) {
    B(i, i) = 1 + 6 * lambda;
    if (i >= 1) B(i, i - 1) = -4 * lambda;
    if (i >= 2) B(i, i - 2) = lambda;
  }
  std::vector<double> ky(m);
  for (std::size_t i = 0; i < m; i++) ky[i] = y[i] - 2 * y[i + 1] + y[i + 2];
  const auto w = calx::linalg::band_solve(B, ky);
  HpFilter f;
  f.cycle.assign(n, 0.0);
  f.trend.resize(n);
  for (std::size_t s = 0; s < n; s++) {
    // (Kᵀw)_s = w_s − 2 w_{s−1} + w_{s−2}, the terms that exist.
    double k = 0;
    if (s < m) k += w[s];
    if (s >= 1 && s - 1 < m) k -= 2 * w[s - 1];
    if (s >= 2 && s - 2 < m) k += w[s - 2];
    f.cycle[s] = lambda * k;
    f.trend[s] = y[s] - f.cycle[s];
  }
  return f;
}

}  // namespace tyche::time_series
