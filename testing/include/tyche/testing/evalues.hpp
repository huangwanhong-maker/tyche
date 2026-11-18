// E-values (decision 0027): non-negative statistics whose expectation under the null is at most 1. By Markov's inequality
// P(E ≥ 1/α) ≤ α, so an e-value is a test at every level at once; and unlike p-values, e-values combine under any dependence.
//
// - Calibrators (Vovk and Wang 2021, Ann. Statist. 49, 1736–1754, §2): a decreasing f with ∫₀¹ f = 1 turns a p-value into an
//   e-value. f_κ(p) = κ p^{κ−1} for κ ∈ (0, 1); and its mixture over κ, ∫₀¹ κ p^{κ−1} dκ = (1 − p + p log p)/(p (log p)²), which
//   dominates every f_κ up to a constant (Shafer, Shen, Vereshchagin and Vovk 2011, Ann. Statist. 39, 1121–1141).
// - Combination (Vovk and Wang 2021, §3–4): the mean of e-values is an e-value under any dependence; their product is one when
//   they are sequentially independent (each valid given the ones before).
// - e-BH (Wang and Ramdas 2022, JRSS B 84, 822–852): reject the k* largest e-values, k* = max{k : e_(k) ≥ n/(α k)}. FDR ≤ α under
//   arbitrary dependence, with no correction.
// - An e-value is read back as a p-value by min(1, 1/e), which is valid by Markov's inequality.
//
// Limits: e-values from calibrated p-values are conservative (a p-value of 0.01 gives 4.45 by the mixture, 5 by κ = 1/2); e-values
// built for the alternative, by likelihood ratios or betting, are stronger and are built where the test is.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::testing {

// κ p^{κ−1}.
inline double calibrate(double p, double kappa) {
  if (!(p >= 0 && p <= 1)) throw std::invalid_argument("calibrate: a p-value lies in [0, 1]");
  if (!(kappa > 0 && kappa < 1)) throw std::invalid_argument("calibrate: κ lies in (0, 1)");
  if (p == 0) return std::numeric_limits<double>::infinity();
  return kappa * std::pow(p, kappa - 1);
}

// The mixture calibrator ∫₀¹ κ p^{κ−1} dκ = (1 − p + p log p)/(p (log p)²); 1/2 at p = 1, its limit.
inline double calibrate_mixture(double p) {
  if (!(p >= 0 && p <= 1)) throw std::invalid_argument("calibrate_mixture: a p-value lies in [0, 1]");
  if (p == 0) return std::numeric_limits<double>::infinity();
  const double l = std::log(p);
  if (std::fabs(l) < 1e-4) {
    // Near p = 1, by the series of the integral in u = −log p: ∫₀¹ κ e^{−(κ−1)u}... = 1/2 + u/6 + u²/24 + O(u³).
    const double u = -l;
    return 0.5 + u / 6 + u * u / 24;
  }
  return (1 - p + p * l) / (p * l * l);
}

inline double e_to_p(double e) {
  if (!(e >= 0)) throw std::invalid_argument("e_to_p: an e-value is non-negative");
  return e <= 1 ? 1.0 : 1 / e;
}

inline double e_mean(std::span<const double> e) {
  if (e.empty()) throw std::invalid_argument("e_mean: no e-values");
  double s = 0;
  for (double v : e) {
    if (!(v >= 0)) throw std::invalid_argument("e_mean: an e-value is non-negative");
    s += v;
  }
  return s / static_cast<double>(e.size());
}

inline double e_product(std::span<const double> e) {
  double s = 1;
  for (double v : e) {
    if (!(v >= 0)) throw std::invalid_argument("e_product: an e-value is non-negative");
    s *= v;
  }
  return s;
}

// e-BH: which hypotheses are rejected (one flag each, in the given order).
inline std::vector<char> e_bh(std::span<const double> e, double alpha) {
  if (!(alpha > 0 && alpha < 1)) throw std::invalid_argument("e_bh: α lies strictly between 0 and 1");
  const std::size_t n = e.size();
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  for (double v : e)
    if (!(v >= 0)) throw std::invalid_argument("e_bh: an e-value is non-negative");
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return e[a] > e[b]; });
  std::size_t kstar = 0;
  for (std::size_t k = 1; k <= n; k++)
    if (e[idx[k - 1]] >= static_cast<double>(n) / (alpha * static_cast<double>(k))) kstar = k;
  std::vector<char> out(n, 0);
  for (std::size_t k = 0; k < kstar; k++) out[idx[k]] = 1;
  return out;
}

}  // namespace tyche::testing
