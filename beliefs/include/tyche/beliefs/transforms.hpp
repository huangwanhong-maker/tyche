// A mass function's belief, plausibility and commonality on every subset, and the Möbius transforms back
// (decision 0023).
//
// Bel and m determine each other: b = ζm with b(A) = Σ_{B ⊆ A} m(B), and m(A) = Σ_{B ⊆ A} (−1)^{|A∖B|} b(B), the
// Möbius inversion on the lattice of subsets (Shafer 1976, A Mathematical Theory of Evidence, ch. 2; Rota 1964,
// Z. Wahrscheinlichkeitstheorie 2, 340–368); likewise Q(A) = Σ_{B ⊇ A} m(B) and m(A) = Σ_{B ⊇ A} (−1)^{|B∖A|} Q(B) (Shafer 1976, ch. 2). Both are
// computed in O(n 2ⁿ) additions, one coordinate at a time (the fast Möbius transform; Kennes 1992, IEEE Trans.
// Systems, Man and Cybernetics 22, 201–223), by morphe's subset transforms; the superset ones by complementing
// the index. Tables are indexed by the subset's mask; frames of at most 20 elements.
//
//   to_vector(m), belief_function(m), plausibility_function(m), commonality_function(m)
//   from_vector, from_belief, from_plausibility, from_commonality   the mass function back, validated
//
// Going back, the transform's rounding (below 4n ε times the largest value) is set to zero, as are masses less
// than the tolerance below zero; a mass further below zero means the table is not a belief (plausibility,
// commonality) function and is refused, naming the set. For a belief function given with m(∅) allowed,
// m(∅) = 1 − Bel(Ω); from a plausibility function, Bel(A) = Pl(Ω) − Pl(Ω ∖ A).

#pragma once

#include <tyche/beliefs/common.hpp>
#include <tyche/beliefs/mass.hpp>

#include <morphe/combinatorics/mobius.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace tyche::beliefs {

inline std::vector<double> to_vector(const Mass& m) {
  detail::check_dense(m.frame);
  std::vector<double> v(std::size_t{1} << m.frame, 0.0);
  for (const auto& f : m.focal) v[f.set] = f.mass;
  return v;
}

namespace detail {
inline std::vector<double> superset_zeta(std::vector<double> g) {
  std::reverse(g.begin(), g.end());  // index S ↦ Ω ∖ S
  g = morphe::combinatorics::subset_zeta(std::move(g));
  std::reverse(g.begin(), g.end());
  return g;
}
inline std::vector<double> superset_mobius(std::vector<double> g) {
  std::reverse(g.begin(), g.end());
  g = morphe::combinatorics::subset_mobius(std::move(g));
  std::reverse(g.begin(), g.end());
  return g;
}
inline std::string mask_text(Set A, std::size_t frame) {
  std::string s = "{";
  bool first = true;
  for (std::size_t i = 0; i < frame; i++)
    if (A >> i & 1) {
      s += (first ? "" : ", ") + std::to_string(i);
      first = false;
    }
  return s + "}";
}
}  // namespace detail

// Bel(A) for every A (Bel(∅) = 0).
inline std::vector<double> belief_function(const Mass& m) {
  auto b = morphe::combinatorics::subset_zeta(to_vector(m));
  const double e = m.conflict();
  for (double& x : b) x -= e;
  b[0] = 0;
  return b;
}

// Pl(A) for every A (Pl(∅) = 0).
inline std::vector<double> plausibility_function(const Mass& m) {
  const auto b = morphe::combinatorics::subset_zeta(to_vector(m));
  const std::size_t N = b.size();
  const double total = b[N - 1];
  std::vector<double> pl(N);
  for (std::size_t A = 0; A < N; A++) pl[A] = total - b[(N - 1) ^ A];
  pl[0] = 0;
  return pl;
}

// Q(A) for every A (Q(∅) = Σ m = 1).
inline std::vector<double> commonality_function(const Mass& m) { return detail::superset_zeta(to_vector(m)); }

// A mass function from its values on every subset, rounding noise and tolerated negatives set to zero.
inline Mass from_vector(std::size_t frame, std::span<const double> v, const MassOptions& o = {}) {
  detail::check_frame(frame);
  detail::check_dense(frame);
  detail::require(v.size() == (std::size_t{1} << frame), "a table of 2ⁿ values");
  double scale = 1;
  for (double x : v) {
    detail::require(std::isfinite(x), "a value is not finite");
    scale = std::max(scale, std::fabs(x));
  }
  const double noise = 4.0 * static_cast<double>(frame) * std::numeric_limits<double>::epsilon() * scale;
  std::vector<Focal> f;
  for (std::size_t A = 0; A < v.size(); A++) {
    const double x = v[A];
    detail::require(x >= -o.tolerance, "m(" + detail::mask_text(static_cast<Set>(A), frame) + ") = " + std::to_string(x) +
                                           " < 0: the table is not of a belief function");
    if (x > noise) f.push_back({static_cast<Set>(A), x});
  }
  return mass(frame, f, o);
}

inline Mass from_belief(std::size_t frame, std::span<const double> bel, const MassOptions& o = {}) {
  detail::check_frame(frame);
  detail::check_dense(frame);
  detail::require(bel.size() == (std::size_t{1} << frame), "a table of 2ⁿ values");
  detail::require(std::fabs(bel[0]) <= o.tolerance, "Bel(∅) = " + std::to_string(bel[0]) + ", not 0");
  std::vector<double> b(bel.begin(), bel.end());
  b[0] = 0;
  auto m = morphe::combinatorics::subset_mobius(std::move(b));
  m[0] = o.allow_empty ? std::max(0.0, 1 - bel.back()) : 0.0;
  return from_vector(frame, m, o);
}

inline Mass from_plausibility(std::size_t frame, std::span<const double> pl, const MassOptions& o = {}) {
  detail::check_frame(frame);
  detail::check_dense(frame);
  detail::require(pl.size() == (std::size_t{1} << frame), "a table of 2ⁿ values");
  const std::size_t N = pl.size();
  std::vector<double> bel(N);
  for (std::size_t A = 0; A < N; A++) bel[A] = pl[N - 1] - pl[(N - 1) ^ A];
  bel[0] = 0;
  return from_belief(frame, bel, o);
}

inline Mass from_commonality(std::size_t frame, std::span<const double> q, const MassOptions& o = {}) {
  detail::check_frame(frame);
  detail::check_dense(frame);
  detail::require(q.size() == (std::size_t{1} << frame), "a table of 2ⁿ values");
  return from_vector(frame, detail::superset_mobius(std::vector<double>(q.begin(), q.end())), o);
}

}  // namespace tyche::beliefs
