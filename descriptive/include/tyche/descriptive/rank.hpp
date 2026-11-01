// Ranks and rank correlation: the mid-ranks of a sample, Spearman's ρ and Kendall's τ_b.
//
// Definitions.
// - ranks(x): the rank of each value in increasing order, from 1; tied values share the mean of the
//   ranks they occupy (mid-ranks; Kendall and Gibbons 1990, Rank Correlation Methods, 5th ed.,
//   §3.1), so that the ranks always sum to n(n + 1)/2.
// - spearman(x, y): Pearson's correlation of the mid-ranks (Spearman 1904; Kendall and Gibbons,
//   §3.8, eq. 3.8 with ties by mid-ranks), which is 1 − 6 Σ d²/(n(n² − 1)) when there are no ties.
//   Ties need no other correction in this form.
// - kendall_tau_b(x, y): (n_c − n_d) / √((n₀ − n₁)(n₀ − n₂)), n_c and n_d the concordant and
//   discordant pairs, n₀ = n(n − 1)/2, n₁ and n₂ the pairs tied in x and in y (Kendall 1945,
//   Biometrika 33, 239–251; Kendall and Gibbons, eq. 3.11), by Knight's O(n log n) merge count
//   (Knight 1966, J. Amer. Statist. Assoc. 61, 436–439).
//
// Accuracy: exact up to the rounding of Pearson's correlation (summary.hpp's correlation, compensated
// and two-pass). NaN when a sample is constant. Complexity O(n log n).

#pragma once

#include <tyche/descriptive/summary.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace tyche::descriptive {

inline std::vector<double> ranks(std::span<const double> x) {
  const std::size_t n = x.size();
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i;
    while (j + 1 < n && x[order[j + 1]] == x[order[i]]) j++;
    const double mid = 0.5 * static_cast<double>(i + j) + 1;  // the mean of ranks i + 1 … j + 1
    for (std::size_t k = i; k <= j; k++) r[order[k]] = mid;
    i = j + 1;
  }
  return r;
}

inline double spearman(std::span<const double> x, std::span<const double> y) {
  detail::require(x.size() == y.size() && x.size() >= 2, "spearman: two samples of one size, at least 2");
  const auto rx = ranks(x), ry = ranks(y);
  return correlation(rx, ry);
}

namespace detail {

// The number of inversions of v (pairs i < j with v[i] > v[j]), by merge sort; v is sorted on return.
inline std::uint64_t inversions(std::vector<double>& v, std::vector<double>& buffer, std::size_t lo, std::size_t hi) {
  if (hi - lo < 2) return 0;
  const std::size_t mid = lo + (hi - lo) / 2;
  std::uint64_t count = inversions(v, buffer, lo, mid) + inversions(v, buffer, mid, hi);
  std::size_t i = lo, j = mid, k = lo;
  while (i < mid && j < hi) {
    if (v[j] < v[i]) {
      count += mid - i;
      buffer[k++] = v[j++];
    } else {
      buffer[k++] = v[i++];
    }
  }
  while (i < mid) buffer[k++] = v[i++];
  while (j < hi) buffer[k++] = v[j++];
  std::copy(buffer.begin() + static_cast<std::ptrdiff_t>(lo), buffer.begin() + static_cast<std::ptrdiff_t>(hi), v.begin() + static_cast<std::ptrdiff_t>(lo));
  return count;
}

// Σ t(t − 1)/2 over runs of equal values in a sorted sequence.
template <class Equal>
inline std::uint64_t tied_pairs(std::size_t n, Equal&& equal) {
  std::uint64_t t = 0, run = 1;
  for (std::size_t i = 1; i <= n; i++) {
    if (i < n && equal(i - 1, i)) { run++; continue; }
    t += run * (run - 1) / 2;
    run = 1;
  }
  return t;
}

}  // namespace detail

inline double kendall_tau_b(std::span<const double> x, std::span<const double> y) {
  detail::require(x.size() == y.size() && x.size() >= 2, "kendall_tau_b: two samples of one size, at least 2");
  const std::size_t n = x.size();
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x[a] != x[b] ? x[a] < x[b] : y[a] < y[b]; });
  const std::uint64_t n0 = static_cast<std::uint64_t>(n) * (n - 1) / 2;
  const std::uint64_t n1 = detail::tied_pairs(n, [&](std::size_t a, std::size_t b) { return x[order[a]] == x[order[b]]; });
  const std::uint64_t n3 = detail::tied_pairs(n, [&](std::size_t a, std::size_t b) { return x[order[a]] == x[order[b]] && y[order[a]] == y[order[b]]; });
  std::vector<double> v(n), buffer(n);
  for (std::size_t i = 0; i < n; i++) v[i] = y[order[i]];
  const std::uint64_t swaps = detail::inversions(v, buffer, 0, n);  // v is now y sorted
  const std::uint64_t n2 = detail::tied_pairs(n, [&](std::size_t a, std::size_t b) { return v[a] == v[b]; });
  // Knight: n_c − n_d = n₀ − n₁ − n₂ + n₃ − 2 · swaps.
  const double s = static_cast<double>(n0) - static_cast<double>(n1) - static_cast<double>(n2) + static_cast<double>(n3) - 2.0 * static_cast<double>(swaps);
  const double denominator = std::sqrt(static_cast<double>(n0 - n1) * static_cast<double>(n0 - n2));
  return denominator > 0 ? s / denominator : std::nan("");
}

}  // namespace tyche::descriptive
