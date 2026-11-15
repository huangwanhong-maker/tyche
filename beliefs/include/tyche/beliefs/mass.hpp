// Mass functions on a finite frame, and the belief, plausibility and commonality they define (decision 0023).
//
// A basic probability assignment, or mass function, m: 2^Ω → [0, 1] with Σ_A m(A) = 1 commits m(A) of one's belief
// to A exactly and to no smaller set (Shafer 1976, A Mathematical Theory of Evidence, ch. 2; after Dempster 1967,
// Ann. Math. Statist. 38, 325–339). The sets of positive mass are its focal sets. Shafer asks m(∅) = 0 (a normal
// mass function); Smets's transferable belief model lets m(∅) > 0 stand for conflict or an open world (Smets and
// Kennes 1994, Artificial Intelligence 66, 191–234), and such functions are accepted where asked (allow_empty).
//
//   belief        Bel(A) = Σ_{∅ ≠ B ⊆ A} m(B)       what the evidence implies of A
//   plausibility  Pl(A)  = Σ_{B ∩ A ≠ ∅} m(B)       what it leaves possible; Pl(A) = Bel(Ω) − Bel(Ω ∖ A)
//   commonality   Q(A)   = Σ_{B ⊇ A} m(B)            Q(∅) = 1
//   implicability b(A)   = Σ_{B ⊆ A} m(B) = Bel(A) + m(∅)  (Smets 1993, Int. J. Approx. Reasoning 9, 1–35)
// For a normal mass function Bel(A) ≤ Pl(A) and Bel(A) + Pl(Ω ∖ A) = 1; in general the sum is 1 − m(∅).
//
// Kinds: Bayesian (every focal set a singleton: Bel = Pl, a probability); consonant (the focal sets nested: Pl is a
// possibility measure, Pl(A) = max_{ω ∈ A} Pl({ω}); Shafer 1976, on consonant support functions; Dubois and Prade
// 1988, Possibility Theory); vacuous (m(Ω) = 1, total ignorance);
// categorical (m(A) = 1); simple support (m(A) = s, m(Ω) = 1 − s).
//
// A mass function is validated when made: a frame of 1 to 30 elements, sets within it, masses finite and not
// negative, summing to one within the tolerance, m(∅) = 0 unless allowed. A set given twice has its masses summed;
// sets of zero mass are dropped. The masses are kept as given, not renormalised.

#pragma once

#include <tyche/beliefs/common.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tyche::beliefs {

struct Focal {
  Set set = 0;
  double mass = 0;
};

struct MassOptions {
  double tolerance = 1e-9;   // how far the masses may sum from one
  bool allow_empty = false;  // whether m(∅) > 0 is accepted (the transferable belief model)
};

struct Mass {
  std::size_t frame = 0;     // the number of elements of Ω
  std::vector<Focal> focal;  // the focal sets, each once, in increasing order of their masks, with positive masses

  // m(A), zero when A is not focal.
  double operator()(Set A) const {
    const auto it = std::lower_bound(focal.begin(), focal.end(), A, [](const Focal& f, Set s) { return f.set < s; });
    return it != focal.end() && it->set == A ? it->mass : 0.0;
  }
  // m(∅), the mass of conflict.
  double conflict() const { return (*this)(0); }
};

namespace detail {

inline double neumaier_add(double& s, double& c, double x) {
  const double t = s + x;
  c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
  s = t;
  return s + c;
}

inline double sum_of(const std::vector<Focal>& f) {
  double s = 0, c = 0;
  for (const auto& x : f) neumaier_add(s, c, x.mass);
  return s + c;
}

// Merges, drops zeros and sorts, without validation (for results the library computes).
inline Mass assemble(std::size_t frame, const std::map<Set, double>& acc) {
  Mass m;
  m.frame = frame;
  for (const auto& [A, x] : acc)
    if (x > 0) m.focal.push_back({A, x});
  return m;
}

inline void check_set(Set A, std::size_t frame) { require(within(A, frame), "a set outside the frame of " + std::to_string(frame) + " elements"); }

}  // namespace detail

inline Mass mass(std::size_t frame, const std::vector<Focal>& entries, const MassOptions& o = {}) {
  detail::check_frame(frame);
  std::map<Set, double> acc;
  for (const auto& f : entries) {
    detail::check_set(f.set, frame);
    detail::require(std::isfinite(f.mass), "a mass is not finite");
    detail::require(f.mass >= 0, "a mass is negative");
    acc[f.set] += f.mass;
  }
  Mass m = detail::assemble(frame, acc);
  detail::require(o.allow_empty || m.conflict() == 0, "m(∅) = " + std::to_string(m.conflict()) + " > 0: a normal mass function gives the empty set nothing (allow_empty admits it)");
  const double s = detail::sum_of(m.focal);
  detail::require(std::fabs(s - 1) <= o.tolerance, "the masses sum to " + std::to_string(s) + ", not to 1");
  return m;
}

inline Mass vacuous(std::size_t frame) { return Mass{frame, {{full(frame), 1.0}}}; }

inline Mass categorical(std::size_t frame, Set A) {
  detail::check_frame(frame);
  detail::check_set(A, frame);
  detail::require(A != 0, "a categorical mass function on the empty set is not normal");
  return Mass{frame, {{A, 1.0}}};
}

// m(A) = s, m(Ω) = 1 − s.
inline Mass simple_support(std::size_t frame, Set A, double s) {
  detail::require(s >= 0 && s <= 1, "a degree of support lies in [0, 1]");
  detail::check_frame(frame);
  detail::check_set(A, frame);
  detail::require(A != 0, "simple support of the empty set");
  return mass(frame, {{A, s}, {full(frame), 1 - s}});
}

// The Bayesian mass function of a probability vector, one entry per element.
inline Mass bayesian(std::span<const double> p, double tolerance = 1e-9) {
  std::vector<Focal> f;
  for (std::size_t i = 0; i < p.size(); i++) f.push_back({singleton(i), p[i]});
  return mass(p.size(), f, {tolerance, false});
}

inline double belief(const Mass& m, Set A) {
  detail::check_set(A, m.frame);
  double s = 0, c = 0;
  for (const auto& f : m.focal)
    if (f.set != 0 && (f.set & ~A) == 0) detail::neumaier_add(s, c, f.mass);
  return s + c;
}

inline double plausibility(const Mass& m, Set A) {
  detail::check_set(A, m.frame);
  double s = 0, c = 0;
  for (const auto& f : m.focal)
    if (f.set & A) detail::neumaier_add(s, c, f.mass);
  return s + c;
}

inline double commonality(const Mass& m, Set A) {
  detail::check_set(A, m.frame);
  double s = 0, c = 0;
  for (const auto& f : m.focal)
    if ((A & ~f.set) == 0) detail::neumaier_add(s, c, f.mass);
  return s + c;
}

inline double implicability(const Mass& m, Set A) { return belief(m, A) + m.conflict(); }

inline bool is_normal(const Mass& m) { return m.conflict() == 0; }

inline bool is_bayesian(const Mass& m) {
  for (const auto& f : m.focal)
    if (cardinality(f.set) != 1) return false;
  return true;
}

inline bool is_consonant(const Mass& m) {
  if (!is_normal(m)) return false;
  std::vector<Set> sets;
  for (const auto& f : m.focal) sets.push_back(f.set);
  std::sort(sets.begin(), sets.end(), [](Set a, Set b) { return cardinality(a) < cardinality(b); });
  for (std::size_t i = 1; i < sets.size(); i++)
    if ((sets[i - 1] & ~sets[i]) != 0) return false;
  return true;
}

inline bool is_vacuous(const Mass& m) { return m.focal.size() == 1 && m.focal[0].set == full(m.frame); }

}  // namespace tyche::beliefs
