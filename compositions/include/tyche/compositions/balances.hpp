// Isometric log-ratio coordinates: a composition in an orthonormal basis of the simplex.
//
// ILR. An orthonormal basis e_1, …, e_{D−1} of S^D under Aitchison's inner product gives coordinates
// ilr(x)_k = ⟨x, e_k⟩_a, an isometric isomorphism of S^D onto R^{D−1} (Egozcue, Pawlowsky-Glahn,
// Mateu-Figueras and Barceló-Vidal 2003, Math. Geol. 35: 279–300): ⟨x, y⟩_a = ilr(x) · ilr(y), and
// d_a(x, y) is the Euclidean distance of the coordinates. Write the clr's of the basis vectors as the
// rows of the (D−1) × D CONTRAST MATRIX Ψ, so that ΨΨᵀ = I_{D−1} and Ψ1 = 0. Then
//
//   ilr(x) = Ψ clr(x) = Ψ ln x,      ilr⁻¹(z) = C(exp(Ψᵀ z)),      clr(x) = Ψᵀ ilr(x),
//
// the second equality since Ψ's rows sum to zero, the last since ΨᵀΨ = I − (1/D)11ᵀ is the projection
// onto the hyperplane where clr lies.
//
// SEQUENTIAL BINARY PARTITIONS (Egozcue and Pawlowsky-Glahn 2005, Math. Geol. 37: 795–828) build Ψ so
// that each coordinate is interpretable. The first step splits the D parts into two groups; each later
// step splits one group made by the steps before it, until every group is a single part: D − 1 steps.
// Step k marks the r parts of one group +1 and the s parts of the other −1 (the rest 0), and its
// coordinate is the BALANCE
//
//   b_k = √(rs/(r + s)) · ln( g(x_+) / g(x_−) ),       g the geometric mean of the marked parts,
//
// whose contrast row holds √(s/(r(r + s))) on the + parts and −√(r/(s(r + s))) on the − parts. The
// rows have unit norm by construction, and two rows are orthogonal because the later one's support lies
// within one group of the earlier one, on which the earlier row is constant, while the later row sums
// to zero. ORIENTATION: a balance is positive when the parts marked +1 are, in geometric mean, larger
// than the parts marked −1.
//
// PIVOT COORDINATES (Hron, Filzmoser and Thompson 2012, J. Appl. Stat. 39: 1115–1128) are the
// partition whose step j sets the j-th part of an ordering against all the parts after it:
//
//   z_j = √((D − j)/(D − j + 1)) · ln( x_(j) / (Π_{k>j} x_(k))^{1/(D−j)} ),   j = 1, …, D − 1.
//
// The first coordinate holds all the relative information of the first part against the rest, so a
// part's role is read by putting it first (`pivot_basis(D, part)`); they are this module's default.
//
// Any two orthonormal bases differ by an orthogonal (D−1) × (D−1) matrix, Ψ_b Ψ_aᵀ (change_of_basis):
// linear models fitted on ilr coordinates give the same fitted values whatever the basis.

#pragma once

#include <tyche/compositions/simplex.hpp>

#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace tyche::compositions {

using calx::linalg::Rectangular;

// A sequential binary partition: D − 1 rows of D signs, each +1, −1 or 0.
using SignTable = std::vector<std::vector<int>>;

struct Basis {
  std::size_t parts = 0;   // D
  SignTable partition;     // the partition it was built from (D − 1 rows)
  Rectangular contrast;    // Ψ, (D − 1) × D: the clr's of the basis vectors, orthonormal rows summing to zero
};

// The orthonormal basis of a sequential binary partition, validated: each row has at least one +1 and
// one −1; the first row marks every part; each later row's marked parts are exactly a group (the +1 or
// the −1 parts) of an earlier row that no row between has split; every group of two or more parts is
// split. Rows must be given in an order in which each splits a group made before it.
inline Basis basis(const SignTable& partition) {
  require(!partition.empty(), "basis", "a partition of at least two parts (one row)");
  const std::size_t D = partition[0].size();
  require(D >= 2, "basis", "at least two parts");
  require(partition.size() == D - 1, "basis", "a sequential binary partition of D parts has D − 1 rows");
  std::vector<std::vector<std::size_t>> open;  // groups still to be split, each sorted
  {
    std::vector<std::size_t> all(D);
    for (std::size_t i = 0; i < D; i++) all[i] = i;
    open.push_back(all);
  }
  Basis b;
  b.parts = D;
  b.partition = partition;
  b.contrast = Rectangular(D - 1, D);
  for (std::size_t k = 0; k < partition.size(); k++) {
    const auto& row = partition[k];
    require(row.size() == D, "basis", "every row has a sign for each part");
    std::vector<std::size_t> plus, minus, support;
    for (std::size_t i = 0; i < D; i++) {
      require(row[i] == 1 || row[i] == -1 || row[i] == 0, "basis", "signs are +1, −1 or 0");
      if (row[i] == 1) plus.push_back(i);
      if (row[i] == -1) minus.push_back(i);
      if (row[i] != 0) support.push_back(i);
    }
    require(!plus.empty() && !minus.empty(), "basis", "every row marks at least one part +1 and one −1");
    const auto at = std::find(open.begin(), open.end(), support);
    require(at != open.end(), "basis",
            "each row splits exactly one group made by the rows before it (the first row marks every part); this row's marked parts are no such group");
    open.erase(at);
    if (plus.size() >= 2) open.push_back(plus);
    if (minus.size() >= 2) open.push_back(minus);
    const double r = static_cast<double>(plus.size()), s = static_cast<double>(minus.size());
    const double a = std::sqrt(s / (r * (r + s))), c = -std::sqrt(r / (s * (r + s)));
    for (std::size_t i : plus) b.contrast(k, i) = a;
    for (std::size_t i : minus) b.contrast(k, i) = c;
  }
  require(open.empty(), "basis", "every group of two or more parts is split");
  return b;
}

// The pivot partition of D parts in the given order (a permutation of 0, …, D − 1; empty: 0, 1, …):
// row j marks order[j] +1 and order[j + 1], …, order[D − 1] −1.
inline SignTable pivot_partition(std::size_t parts, std::span<const std::size_t> order = {}) {
  require(parts >= 2, "pivot_partition", "at least two parts");
  std::vector<std::size_t> o(parts);
  if (order.empty()) {
    for (std::size_t i = 0; i < parts; i++) o[i] = i;
  } else {
    require(order.size() == parts, "pivot_partition", "the order is a permutation of the parts");
    std::vector<char> seen(parts, 0);
    for (std::size_t i = 0; i < parts; i++) {
      require(order[i] < parts && !seen[order[i]], "pivot_partition", "the order is a permutation of the parts");
      seen[order[i]] = 1;
      o[i] = order[i];
    }
  }
  SignTable t(parts - 1, std::vector<int>(parts, 0));
  for (std::size_t j = 0; j + 1 < parts; j++) {
    t[j][o[j]] = 1;
    for (std::size_t k = j + 1; k < parts; k++) t[j][o[k]] = -1;
  }
  return t;
}

// The pivot basis with `part` first and the others in their order (Hron, Filzmoser and Thompson 2012,
// (x_l, x_1, …, x_{l−1}, x_{l+1}, …, x_D)): its first coordinate is part l against the geometric mean of
// the rest.
inline Basis pivot_basis(std::size_t parts, std::size_t part = 0) {
  require(parts >= 2, "pivot_basis", "at least two parts");
  require(part < parts, "pivot_basis", "the pivot is one of the parts");
  std::vector<std::size_t> o{part};
  for (std::size_t i = 0; i < parts; i++)
    if (i != part) o.push_back(i);
  return basis(pivot_partition(parts, o));
}

// ilr(x) = Ψ ln x in the given basis.
inline std::vector<double> ilr(std::span<const double> x, const Basis& b) {
  require(x.size() == b.parts, "ilr", "the composition has the basis's parts");
  const auto z = clr(x);
  return calx::linalg::multiply(b.contrast, z);
}

// ilr(x) in pivot coordinates of the parts in order.
inline std::vector<double> ilr(std::span<const double> x) { return ilr(x, pivot_basis(x.size())); }

// ilr⁻¹(z) = C(exp(Ψᵀ z)).
inline std::vector<double> ilr_inverse(std::span<const double> z, const Basis& b, double total = 1) {
  require(z.size() + 1 == b.parts, "ilr_inverse", "D − 1 coordinates for a basis of D parts");
  for (double v : z) require(std::isfinite(v), "ilr_inverse", "finite coordinates");
  return clr_inverse(calx::linalg::multiply_transpose(b.contrast, z), total);
}

// ilr⁻¹(z) in pivot coordinates of D = z.size() + 1 parts in order.
inline std::vector<double> ilr_inverse(std::span<const double> z, double total = 1) { return ilr_inverse(z, pivot_basis(z.size() + 1), total); }

// The ilr coordinates of each row of a sample (n × D) → n × (D − 1).
inline Rectangular ilr(const Rectangular& X, const Basis& b) {
  require(X.cols == b.parts, "ilr", "the sample's columns are the basis's parts");
  Rectangular Z(X.rows, b.parts - 1);
  for (std::size_t i = 0; i < X.rows; i++) {
    const auto row = X.row(i);
    const auto z = ilr(row, b);
    for (std::size_t k = 0; k + 1 < b.parts; k++) Z(i, k) = z[k];
  }
  return Z;
}

// ilr⁻¹ of each row (n × (D − 1)) → n × D, each closed to `total`.
inline Rectangular ilr_inverse(const Rectangular& Z, const Basis& b, double total = 1) {
  require(Z.cols + 1 == b.parts, "ilr_inverse", "D − 1 coordinates for a basis of D parts");
  Rectangular X(Z.rows, b.parts);
  for (std::size_t i = 0; i < Z.rows; i++) {
    const auto row = Z.row(i);
    const auto x = ilr_inverse(row, b, total);
    for (std::size_t j = 0; j < b.parts; j++) X(i, j) = x[j];
  }
  return X;
}

// The orthogonal matrix Ψ_to Ψ_fromᵀ ((D − 1) × (D − 1)) that carries ilr coordinates in `from` to
// those in `to`: ilr_to(x) = change_of_basis(from, to) · ilr_from(x).
inline Rectangular change_of_basis(const Basis& from, const Basis& to) {
  require(from.parts == to.parts, "change_of_basis", "two bases of the same parts");
  return to.contrast * from.contrast.transpose();
}

// The balance √(rs/(r + s)) ln(g(x_plus)/g(x_minus)) between two disjoint, non-empty groups of parts.
inline double balance(std::span<const double> x, std::span<const std::size_t> plus, std::span<const std::size_t> minus) {
  detail::require_positive(x, "balance");
  require(!plus.empty() && !minus.empty(), "balance", "two non-empty groups of parts");
  std::vector<char> mark(x.size(), 0);
  double lp = 0, lm = 0;
  for (std::size_t i : plus) {
    require(i < x.size() && !mark[i], "balance", "the groups are disjoint sets of the parts");
    mark[i] = 1;
    lp += std::log(x[i]);
  }
  for (std::size_t i : minus) {
    require(i < x.size() && !mark[i], "balance", "the groups are disjoint sets of the parts");
    mark[i] = 1;
    lm += std::log(x[i]);
  }
  const double r = static_cast<double>(plus.size()), s = static_cast<double>(minus.size());
  return std::sqrt(r * s / (r + s)) * (lp / r - lm / s);
}

}  // namespace tyche::compositions
