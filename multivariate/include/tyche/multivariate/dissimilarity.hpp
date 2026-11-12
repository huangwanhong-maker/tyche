// Dissimilarities between samples of counts: how different two samples' compositions are.
//
// Each sample is a row of non-negative abundances over the same classes (a table of samples by
// classes: plots by species, blocks by kinds of business, documents by words). Legendre and Legendre
// (2012, Numerical Ecology, 3rd English ed., Elsevier, ch. 7, §7.3–7.5 and Table 7.2) give the
// coefficients and their properties; with x and y two rows, x₊ = Σ x_i:
//
//   Bray–Curtis   Σ|x_i − y_i| / Σ(x_i + y_i)          (their D14, the percentage difference); 0 for
//                 two empty samples, 1 when one alone is empty. Semimetric: it can break the triangle
//                 inequality.
//   Jaccard       1 − a/(a + b + c) on presence           (one minus their S7), a the classes present in
//                 both, b and c in one only; 0 for two empty samples. Metric.
//   Sørensen      1 − 2a/(2a + b + c) on presence         (one minus their S8): Bray–Curtis on presence
//                 and absence. Semimetric.
//   Hellinger     √Σ(√(x_i/x₊) − √(y_i/y₊))²              (their D17): the Euclidean distance between the
//                 square roots of the samples' laws, at most √2. Metric; Euclidean.
//   chord         √Σ(x_i/‖x‖ − y_i/‖y‖)² = √(2(1 − cos θ)) (their D3), ‖·‖ the Euclidean norm, at most √2.
//                 Metric; Euclidean.
//   Euclidean     √Σ(x_i − y_i)²                          (their D1). Metric; Euclidean. Its rows may be
//                 any real values (coordinates, standardised variables), not only abundances.
//
// Hellinger and chord distances are refused for an empty sample (its law is undefined). Bray–Curtis,
// Hellinger and chord ignore classes absent from both samples (the double zeros), as a coefficient of
// composition should (Legendre and Legendre 2012, §7.2.2); Euclidean does not.

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace tyche::multivariate {

enum class Dissimilarity { bray_curtis, jaccard, sorensen, hellinger, chord, euclidean };

namespace detail {

inline void check(bool ok, const char* where, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::multivariate::") + where + ": " + what);
}

}  // namespace detail

// The dissimilarity of two samples of non-negative abundances over the same classes.
inline double dissimilarity(std::span<const double> x, std::span<const double> y, Dissimilarity kind) {
  detail::check(x.size() == y.size(), "dissimilarity", "two samples over the same classes");
  double sx = 0, sy = 0, nx = 0, ny = 0;
  for (std::size_t i = 0; i < x.size(); i++) {
    detail::check(std::isfinite(x[i]) && std::isfinite(y[i]), "dissimilarity", "values are finite");
    detail::check(kind == Dissimilarity::euclidean || (x[i] >= 0 && y[i] >= 0), "dissimilarity", "abundances are non-negative");
    sx += x[i], sy += y[i], nx += x[i] * x[i], ny += y[i] * y[i];
  }
  switch (kind) {
    case Dissimilarity::bray_curtis: {
      if (sx + sy == 0) return 0;
      double d = 0;
      for (std::size_t i = 0; i < x.size(); i++) d += std::fabs(x[i] - y[i]);
      return d / (sx + sy);
    }
    case Dissimilarity::jaccard:
    case Dissimilarity::sorensen: {
      double a = 0, bc = 0;
      for (std::size_t i = 0; i < x.size(); i++) {
        const bool px = x[i] > 0, py = y[i] > 0;
        a += px && py;
        bc += px != py;
      }
      if (a + bc == 0) return 0;
      return kind == Dissimilarity::jaccard ? bc / (a + bc) : bc / (2 * a + bc);
    }
    case Dissimilarity::hellinger: {
      detail::check(sx > 0 && sy > 0, "dissimilarity", "the Hellinger distance needs samples that are not empty");
      double d = 0;
      for (std::size_t i = 0; i < x.size(); i++) {
        const double u = std::sqrt(x[i] / sx) - std::sqrt(y[i] / sy);
        d += u * u;
      }
      return std::sqrt(d);
    }
    case Dissimilarity::chord: {
      detail::check(nx > 0 && ny > 0, "dissimilarity", "the chord distance needs samples that are not empty");
      const double rx = std::sqrt(nx), ry = std::sqrt(ny);
      double d = 0;
      for (std::size_t i = 0; i < x.size(); i++) {
        const double u = x[i] / rx - y[i] / ry;
        d += u * u;
      }
      return std::sqrt(d);
    }
    case Dissimilarity::euclidean: {
      double d = 0;
      for (std::size_t i = 0; i < x.size(); i++) d += (x[i] - y[i]) * (x[i] - y[i]);
      return std::sqrt(d);
    }
  }
  return 0;
}

// The matrix of dissimilarities between the rows of a table (samples by classes): symmetric, with a
// zero diagonal.
inline calx::linalg::Matrix dissimilarities(const calx::linalg::Rectangular& table, Dissimilarity kind) {
  const std::size_t n = table.rows;
  calx::linalg::Matrix d(n);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) {
      const double v = dissimilarity(std::span<const double>(table.a.data() + i * table.cols, table.cols), std::span<const double>(table.a.data() + j * table.cols, table.cols), kind);
      d(i, j) = d(j, i) = v;
    }
  return d;
}

// Checks that d is a matrix of dissimilarities: square, symmetric, finite, non-negative, zero on the
// diagonal.
inline void require_dissimilarities(const calx::linalg::Matrix& d, const char* where) {
  for (std::size_t i = 0; i < d.n; i++) {
    detail::check(d(i, i) == 0, where, "dissimilarities vanish on the diagonal");
    for (std::size_t j = 0; j < i; j++)
      detail::check(std::isfinite(d(i, j)) && d(i, j) >= 0 && d(i, j) == d(j, i), where, "dissimilarities are finite, non-negative and symmetric");
  }
}

}  // namespace tyche::multivariate
