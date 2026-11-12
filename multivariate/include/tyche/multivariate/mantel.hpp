// Do two matrices of dissimilarities among the same objects agree? The Mantel test.
//
// Mantel (1967, Cancer Res. 27: 209–220) compares two matrices A and B of dissimilarities among the
// same n objects by Z = Σ_{i<j} a_ij b_ij; standardised, this is Pearson's correlation r between their
// n(n − 1)/2 off-diagonal entries (Legendre and Legendre 2012, Numerical Ecology, §10.5.1). Under the
// null hypothesis that the objects' labels in one matrix are exchangeable with respect to the other,
// relabelling the rows and columns of B together by a permutation π leaves the law of r unchanged, so
// the one-sided p-value for a positive association is (b + 1)/(B + 1), b of B random relabellings with
// r* ≥ r (Phipson and Smyth 2010), or, by `mantel_exact`, the share of all n! relabellings with
// r* ≥ r. Ties are r* ≥ r − 10⁻¹². The entries of a pair are not independent, so r's usual t-test does
// not apply; only the permutation law does. Relabellings are drawn by Fisher–Yates shuffles on a calx
// stream (seed, identity (0x7065, 2, 0)).

#pragma once

#include <tyche/multivariate/dissimilarity.hpp>
#include <tyche/multivariate/permanova.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace tyche::multivariate {

struct Mantel {
  double r = 0;        // Pearson's correlation of the off-diagonal entries
  double p_value = 1;  // one-sided, r* ≥ r
  std::size_t permutations = 0;
  bool exact = false;
};

namespace detail {

struct MantelSetup {
  std::size_t n = 0;
  std::vector<double> a, b;  // standardised: a over the pairs i < j in order; b as a full n × n matrix
  double r = 0;
};

inline MantelSetup mantel_setup(const calx::linalg::Matrix& A, const calx::linalg::Matrix& B, const char* where) {
  check(A.n == B.n && A.n >= 3, where, "two matrices among the same objects, at least three");
  const std::size_t n = A.n;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < i; j++)
      check(std::isfinite(A(i, j)) && std::isfinite(B(i, j)) && A(i, j) == A(j, i) && B(i, j) == B(j, i), where, "symmetric and finite");
  MantelSetup s;
  s.n = n;
  const double m = static_cast<double>(n * (n - 1) / 2);
  double ma = 0, mb = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) ma += A(i, j) / m, mb += B(i, j) / m;
  double va = 0, vb = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) va += (A(i, j) - ma) * (A(i, j) - ma), vb += (B(i, j) - mb) * (B(i, j) - mb);
  check(va > 0 && vb > 0, where, "neither matrix is constant off the diagonal");
  const double sa = std::sqrt(va), sb = std::sqrt(vb);
  s.b.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++)
      if (i != j) s.b[i * n + j] = (B(i, j) - mb) / sb;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) s.a.push_back((A(i, j) - ma) / sa);
  double r = 0;
  std::size_t k = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) r += s.a[k++] * s.b[i * n + j];
  s.r = r;
  return s;
}

inline double relabelled(const MantelSetup& s, const std::vector<std::size_t>& p) {
  double r = 0;
  std::size_t k = 0;
  for (std::size_t i = 0; i < s.n; i++)
    for (std::size_t j = i + 1; j < s.n; j++) r += s.a[k++] * s.b[p[i] * s.n + p[j]];
  return r;
}

}  // namespace detail

inline Mantel mantel(const calx::linalg::Matrix& A, const calx::linalg::Matrix& B, std::size_t permutations = 999, std::uint64_t seed = 20261112) {
  const auto s = detail::mantel_setup(A, B, "mantel");
  Mantel m;
  m.r = s.r;
  m.permutations = permutations;
  std::vector<std::size_t> p(s.n);
  std::iota(p.begin(), p.end(), std::size_t{0});
  calx::random::Stream rng(seed, 0x7065, 2, 0);
  std::size_t b = 0;
  for (std::size_t k = 0; k < permutations; k++) {
    detail::shuffle(p, rng);
    if (detail::relabelled(s, p) >= s.r - 1e-12) b++;
  }
  m.p_value = (static_cast<double>(b) + 1) / (static_cast<double>(permutations) + 1);
  return m;
}

// The exact p-value over all n! relabellings; n ≤ 10.
inline Mantel mantel_exact(const calx::linalg::Matrix& A, const calx::linalg::Matrix& B) {
  const auto s = detail::mantel_setup(A, B, "mantel_exact");
  detail::check(s.n <= 10, "mantel_exact", "at most ten objects (n! relabellings)");
  Mantel m;
  m.r = s.r;
  m.exact = true;
  std::vector<std::size_t> p(s.n);
  std::iota(p.begin(), p.end(), std::size_t{0});
  std::size_t total = 0, b = 0;
  do {
    total++;
    if (detail::relabelled(s, p) >= s.r - 1e-12) b++;
  } while (std::next_permutation(p.begin(), p.end()));
  m.permutations = total;
  m.p_value = static_cast<double>(b) / static_cast<double>(total);
  return m;
}

}  // namespace tyche::multivariate
