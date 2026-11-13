// Inference with relational dependence: the quadratic assignment procedure (QAP).
//
// DYADIC DATA are n × n matrices over the same n actors: a network's ties, and covariates of pairs (the
// same group, the distance between homes, a difference of ages). Their n(n − 1) cells are not independent
// (the cells of a row share an actor), so the usual standard errors of a correlation or a regression over
// cells do not hold. QAP tests instead against the null law obtained by relabelling the actors: permuting
// the rows and columns of one matrix together by a permutation π of the actors keeps every dependence
// among the cells of a row or column, and breaks only the association between the matrices (Hubert and
// Schultz 1976, Br. J. Math. Statist. Psychol. 29: 190–241; Krackhardt 1987, Soc. Networks 9: 171–186).
//
// QAP CORRELATION. r is Pearson's correlation of the off-diagonal cells of A and B; its null law is that of
// r(A, B^π), B^π_ij = B_π(i)π(j). Since the off-diagonal cells of B^π are those of B rearranged, their mean
// and variance do not change, and r(A, B^π) is a sum of products of standardised cells.
//
// MULTIPLE REGRESSION QAP (MRQAP) regresses Y on X₁, …, X_k over the cells by least squares
// (regression::ols) and tests each coefficient by its t statistic against a permutation law:
//
//   Y-permutation           Y is relabelled and the whole model refitted (Krackhardt 1988, Soc. Networks
//                           10: 359–381). Exact for the null that Y is unrelated to every X, but biased when
//                           the X's are correlated and one of them matters (Dekker, Krackhardt and Snijders
//                           2007, Psychometrika 72: 563–581).
//   double semi-partialling for coefficient k, X_k is regressed on the other columns (with the intercept),
//                           its residual matrix E_k relabelled, and Y regressed on the other columns and
//                           E_k^π; the statistic is the t of E_k^π (Dekker, Krackhardt and Snijders 2007,
//                           who found it the most robust of the procedures they compared). At π = identity
//                           this t is the full model's t of X_k, by the theorem of Frisch, Waugh and Lovell
//                           (checked).
//
// P-VALUES. For B random relabellings, (b + 1)/(B + 1), b the relabellings whose statistic is at least as
// extreme (greater, less, or greater in absolute value), the observed relabelling counted (Phipson and
// Smyth 2010, Stat. Appl. Genet. Mol. Biol. 9: 39); with `exact`, the share of all n! relabellings
// (n ≤ 10 for a correlation, n ≤ 8 for MRQAP). Ties: within 10⁻¹² (correlation) or 10⁻¹⁰ (1 + |t|)
// (t). Relabellings are Fisher–Yates shuffles on a calx stream (seed, identity (0x6e657477, purpose, 0),
// purpose 1 for the correlation, 2 for MRQAP), so a result is reproduced by its seed.
//
// SYMMETRIC DATA. With `symmetric`, the matrices must be symmetric and only the cells i < j are used (each
// pair once): the correlation is the same as over all cells, but the t statistics of MRQAP are not.
// The diagonal is never used.

#pragma once

#include <tyche/networks/network.hpp>

#include <tyche/regression/linear.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace tyche::networks {

struct QapOptions {
  std::size_t permutations = 999;
  std::uint64_t seed = 20261113;
  bool exact = false;      // every relabelling (n ≤ 10)
  bool symmetric = false;  // symmetric matrices: the cells i < j only
};

struct QapCorrelation {
  double r = 0;
  double p_greater = 1, p_less = 1, p_two_sided = 1;  // r* ≥ r, r* ≤ r, |r*| ≥ |r|
  std::size_t permutations = 0;
  bool exact = false;
};

namespace detail {

inline std::vector<std::pair<std::size_t, std::size_t>> cells(std::size_t n, bool symmetric) {
  std::vector<std::pair<std::size_t, std::size_t>> c;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = symmetric ? i + 1 : 0; j < n; j++)
      if (i != j) c.push_back({i, j});
  return c;
}

inline void check_dyadic(const Matrix& A, std::size_t n, bool symmetric, const char* where) {
  require(A.n == n, where, "matrices among the same actors");
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++) {
      if (i == j) continue;
      require(std::isfinite(A(i, j)), where, "finite cells");
      if (symmetric) require(A(i, j) == A(j, i), where, "symmetric matrices, as declared");
    }
}

template <class Stream>
void shuffle(std::vector<std::size_t>& p, Stream& rng) {
  for (std::size_t i = p.size(); i > 1; i--) std::swap(p[i - 1], p[rng.below(i)]);
}

inline std::uint64_t factorial(std::size_t n) {
  std::uint64_t f = 1;
  for (std::size_t k = 2; k <= n; k++) f *= k;
  return f;
}

// Calls f(π) for each relabelling: B successive shuffles, or every permutation (the identity among them).
template <class F>
std::size_t relabellings(std::size_t n, bool exact, std::size_t permutations, std::uint64_t seed, std::uint32_t purpose, F f) {
  std::vector<std::size_t> p(n);
  std::iota(p.begin(), p.end(), std::size_t{0});
  if (exact) {
    std::size_t count = 0;
    do {
      f(p);
      count++;
    } while (std::next_permutation(p.begin(), p.end()));
    return count;
  }
  calx::random::Stream rng(seed, 0x6e657477, purpose, 0);
  for (std::size_t b = 0; b < permutations; b++) {
    shuffle(p, rng);
    f(p);
  }
  return permutations;
}

struct Tally {
  std::size_t greater = 0, less = 0, two = 0;
  void add(double stat, double observed, double tol) {
    if (stat >= observed - tol) greater++;
    if (stat <= observed + tol) less++;
    if (std::fabs(stat) >= std::fabs(observed) - tol) two++;
  }
  // (b + 1)/(B + 1) for random relabellings, b/n! for all of them.
  double p(std::size_t b, std::size_t count, bool exact) const {
    return exact ? static_cast<double>(b) / static_cast<double>(count) : (static_cast<double>(b) + 1) / (static_cast<double>(count) + 1);
  }
};

}  // namespace detail

// The QAP correlation of two dyadic matrices and its permutation p-values.
inline QapCorrelation qap_correlation(const Matrix& A, const Matrix& B, const QapOptions& o = {}) {
  const std::size_t n = A.n;
  require(n >= 3, "qap_correlation", "at least three actors");
  detail::check_dyadic(A, n, o.symmetric, "qap_correlation");
  detail::check_dyadic(B, n, o.symmetric, "qap_correlation");
  require(!o.exact || n <= 10, "qap_correlation", "an exact test enumerates n! relabellings: at most ten actors");
  const auto c = detail::cells(n, o.symmetric);
  const double m = static_cast<double>(c.size());
  double ma = 0, mb = 0;
  for (const auto& [i, j] : c) ma += A(i, j) / m, mb += B(i, j) / m;
  double va = 0, vb = 0;
  for (const auto& [i, j] : c) va += (A(i, j) - ma) * (A(i, j) - ma), vb += (B(i, j) - mb) * (B(i, j) - mb);
  require(va > 0 && vb > 0, "qap_correlation", "neither matrix is constant off the diagonal");
  std::vector<double> a;
  for (const auto& [i, j] : c) a.push_back((A(i, j) - ma) / std::sqrt(va));
  Matrix b(n);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++)
      if (i != j) b(i, j) = (B(i, j) - mb) / std::sqrt(vb);
  auto r_of = [&](const std::vector<std::size_t>& p) {
    double r = 0;
    for (std::size_t k = 0; k < c.size(); k++) r += a[k] * b(p[c[k].first], p[c[k].second]);
    return r;
  };
  std::vector<std::size_t> id(n);
  std::iota(id.begin(), id.end(), std::size_t{0});
  QapCorrelation q;
  q.r = r_of(id);
  q.exact = o.exact;
  detail::Tally t;
  q.permutations = detail::relabellings(n, o.exact, o.permutations, o.seed, 1, [&](const std::vector<std::size_t>& p) { t.add(r_of(p), q.r, 1e-12); });
  q.p_greater = t.p(t.greater, q.permutations, o.exact);
  q.p_less = t.p(t.less, q.permutations, o.exact);
  q.p_two_sided = t.p(t.two, q.permutations, o.exact);
  return q;
}

enum class MrqapNull { y_permutation, double_semi_partialling };

struct MrqapOptions {
  MrqapNull method = MrqapNull::double_semi_partialling;
  bool intercept = true;
  std::size_t permutations = 999;
  std::uint64_t seed = 20261113;
  bool exact = false;      // every relabelling (n ≤ 8)
  bool symmetric = false;  // symmetric matrices: the cells i < j only
};

struct Mrqap {
  regression::LinearFit fit;      // Y on [1,] X₁, …, X_k over the cells (classical covariance)
  std::vector<double> p_two_sided, p_greater, p_less;  // per coefficient of fit, from t; NaN for the intercept
  std::size_t cells = 0, permutations = 0;
  bool exact = false;
  MrqapNull method = MrqapNull::double_semi_partialling;
};

// MRQAP: Y regressed on the X's over the off-diagonal cells, each coefficient tested by QAP.
inline Mrqap mrqap(const Matrix& Y, const std::vector<Matrix>& X, const MrqapOptions& o = {}) {
  const std::size_t n = Y.n;
  require(n >= 3, "mrqap", "at least three actors");
  require(!X.empty(), "mrqap", "at least one explanatory matrix");
  require(!o.exact || n <= 8, "mrqap", "an exact test enumerates n! relabellings, each a regression: at most eight actors");
  detail::check_dyadic(Y, n, o.symmetric, "mrqap");
  for (const auto& x : X) detail::check_dyadic(x, n, o.symmetric, "mrqap");
  const auto c = detail::cells(n, o.symmetric);
  const std::size_t c0 = o.intercept ? 1 : 0, k = c0 + X.size();
  Rectangular D(c.size(), k);
  std::vector<double> y(c.size());
  for (std::size_t r = 0; r < c.size(); r++) {
    const auto [i, j] = c[r];
    if (o.intercept) D(r, 0) = 1;
    for (std::size_t a = 0; a < X.size(); a++) D(r, c0 + a) = X[a](i, j);
    y[r] = Y(i, j);
  }
  Mrqap m;
  m.method = o.method;
  m.exact = o.exact;
  m.cells = c.size();
  m.fit = regression::ols(D, y);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  m.p_two_sided.assign(k, nan);
  m.p_greater.assign(k, nan);
  m.p_less.assign(k, nan);
  std::vector<detail::Tally> tally(k);
  auto tol = [](double t) { return 1e-10 * (1 + std::fabs(t)); };
  if (o.method == MrqapNull::y_permutation) {
    std::vector<double> ys(c.size());
    m.permutations = detail::relabellings(n, o.exact, o.permutations, o.seed, 2, [&](const std::vector<std::size_t>& p) {
      for (std::size_t r = 0; r < c.size(); r++) ys[r] = Y(p[c[r].first], p[c[r].second]);
      const auto f = regression::ols(D, ys);
      for (std::size_t a = c0; a < k; a++) tally[a].add(f.t[a], m.fit.t[a], tol(m.fit.t[a]));
    });
  } else {
    // For each coefficient: E_a, the residual of X_a on the other columns, as a matrix over the actors.
    std::vector<Rectangular> others;
    std::vector<Matrix> E;
    for (std::size_t a = c0; a < k; a++) {
      Rectangular O(c.size(), k - 1);
      std::vector<double> xa(c.size());
      for (std::size_t r = 0; r < c.size(); r++) {
        for (std::size_t b = 0, col = 0; b < k; b++)
          if (b != a) O(r, col++) = D(r, b);
        xa[r] = D(r, a);
      }
      const auto res = regression::ols(O, xa).residuals;
      Matrix e(n);
      for (std::size_t r = 0; r < c.size(); r++) {
        e(c[r].first, c[r].second) = res[r];
        if (o.symmetric) e(c[r].second, c[r].first) = res[r];
      }
      others.push_back(O);
      E.push_back(e);
    }
    std::uint32_t purpose = 2;
    m.permutations = detail::relabellings(n, o.exact, o.permutations, o.seed, purpose, [&](const std::vector<std::size_t>& p) {
      for (std::size_t a = c0; a < k; a++) {
        const auto& O = others[a - c0];
        Rectangular Dp(c.size(), k);
        for (std::size_t r = 0; r < c.size(); r++) {
          for (std::size_t b = 0; b + 1 < k; b++) Dp(r, b) = O(r, b);
          Dp(r, k - 1) = E[a - c0](p[c[r].first], p[c[r].second]);
        }
        const auto f = regression::ols(Dp, y);
        tally[a].add(f.t[k - 1], m.fit.t[a], tol(m.fit.t[a]));
      }
    });
  }
  for (std::size_t a = c0; a < k; a++) {
    m.p_greater[a] = tally[a].p(tally[a].greater, m.permutations, o.exact);
    m.p_less[a] = tally[a].p(tally[a].less, m.permutations, o.exact);
    m.p_two_sided[a] = tally[a].p(tally[a].two, m.permutations, o.exact);
  }
  return m;
}

// The t statistic of coefficient `column` (≥ 1 with an intercept) under double semi-partialling at one
// relabelling π of the actors: exposed so that its value at the identity can be held to the full model's.
inline double dsp_statistic(const Matrix& Y, const std::vector<Matrix>& X, std::size_t column, const std::vector<std::size_t>& relabelling,
                            bool intercept = true, bool symmetric = false) {
  const std::size_t n = Y.n, c0 = intercept ? 1 : 0, k = c0 + X.size();
  require(column >= c0 && column < k, "dsp_statistic", "a coefficient of an explanatory matrix");
  require(relabelling.size() == n, "dsp_statistic", "a relabelling of the actors");
  const auto c = detail::cells(n, symmetric);
  Rectangular O(c.size(), k - 1);
  std::vector<double> xa(c.size()), y(c.size());
  for (std::size_t r = 0; r < c.size(); r++) {
    const auto [i, j] = c[r];
    std::size_t col = 0;
    for (std::size_t b = 0; b < k; b++) {
      const double v = b < c0 ? 1.0 : X[b - c0](i, j);
      if (b == column) xa[r] = v;
      else O(r, col++) = v;
    }
    y[r] = Y(i, j);
  }
  const auto res = regression::ols(O, xa).residuals;
  Matrix e(n);
  for (std::size_t r = 0; r < c.size(); r++) {
    e(c[r].first, c[r].second) = res[r];
    if (symmetric) e(c[r].second, c[r].first) = res[r];
  }
  Rectangular Dp(c.size(), k);
  for (std::size_t r = 0; r < c.size(); r++) {
    for (std::size_t b = 0; b + 1 < k; b++) Dp(r, b) = O(r, b);
    Dp(r, k - 1) = e(relabelling[c[r].first], relabelling[c[r].second]);
  }
  return regression::ols(Dp, y).t[k - 1];
}

}  // namespace tyche::networks
