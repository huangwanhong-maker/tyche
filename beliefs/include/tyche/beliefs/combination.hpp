// The combination of independent bodies of evidence, discounting and conditioning (decision 0023).
//
// Two mass functions m₁, m₂ on one frame, from distinct sources, combine by intersecting their focal sets:
//   conjunctive    (m₁ ∩ m₂)(A) = Σ_{B ∩ C = A} m₁(B) m₂(C), the unnormalised rule of the transferable belief
//                  model (Smets and Kennes 1994, Artificial Intelligence 66, 191–234), whose m(∅) is the conflict
//                  K = Σ_{B ∩ C = ∅} m₁(B) m₂(C)
//   Dempster       (m₁ ⊕ m₂)(A) = (m₁ ∩ m₂)(A) / (1 − K) for A ≠ ∅, and 0 at ∅ (Dempster 1967, Ann. Math. Statist.
//                  38, 325–339; Shafer 1976, A Mathematical Theory of Evidence, ch. 3); undefined, and refused, at
//                  total conflict K = 1, when no focal set of one meets a focal set of the other. Zadeh (1984, AI
//                  Magazine 5(3), 81–83) showed how the normalisation can give full belief to what both sources
//                  thought almost impossible; the conflict is reported so that a caller can see it
//   Yager          the conflict given to Ω instead of normalised away (Yager 1987, Information Sciences 41, 93–137)
//   disjunctive    (m₁ ∪ m₂)(A) = Σ_{B ∪ C = A} m₁(B) m₂(C), for when only one source is known to be reliable
//                  (Smets 1993, Int. J. Approx. Reasoning 9, 1–35)
// The conjunctive and disjunctive rules and Dempster's are commutative and associative; Yager's is commutative
// but not associative. Dempster's rule of several mass functions is the conjunctive combination of all,
// normalised once, its conflict the m(∅) of that combination.
//
//   discount(m, α)   m^α(A) = α m(A) for A ≠ Ω, m^α(Ω) = α m(Ω) + 1 − α: a source believed reliable with
//                    probability α (Shafer 1976; Smets 1993); α = 1 leaves m, α = 0 gives the vacuous function
//   normalise(m)     m(A) / (1 − m(∅)) for A ≠ ∅, refused when m(∅) = 1
//   condition(m, B)  Dempster's conditioning, m ⊕ m_B with m_B(B) = 1: Bel(A | B) = (Bel(A ∪ B̄) − Bel(B̄)) /
//                    (1 − Bel(B̄)); refused when Pl(B) = 0
//
// Masses are products and sums of the inputs; the normalising constant is computed as the sum of the masses that
// meet (not as 1 − K), so that it keeps its accuracy when the conflict is near one.

#pragma once

#include <tyche/beliefs/common.hpp>
#include <tyche/beliefs/mass.hpp>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace tyche::beliefs {

struct Combined {
  Mass mass;          // the combination
  double conflict = 0;  // K, the mass the conjunctive combination gives the empty set
};

namespace detail {
inline void same_frame(const Mass& a, const Mass& b) {
  require(a.frame == b.frame && a.frame >= 1, "the mass functions are on frames of " + std::to_string(a.frame) + " and " + std::to_string(b.frame) + " elements");
}
}  // namespace detail

inline Mass conjunctive(const Mass& a, const Mass& b) {
  detail::same_frame(a, b);
  std::map<Set, double> acc;
  for (const auto& f : a.focal)
    for (const auto& g : b.focal) acc[f.set & g.set] += f.mass * g.mass;
  return detail::assemble(a.frame, acc);
}

inline Mass conjunctive(const std::vector<Mass>& ms) {
  detail::require(!ms.empty(), "no mass function to combine");
  Mass r = ms[0];
  for (std::size_t i = 1; i < ms.size(); i++) r = conjunctive(r, ms[i]);
  return r;
}

inline Mass disjunctive(const Mass& a, const Mass& b) {
  detail::same_frame(a, b);
  std::map<Set, double> acc;
  for (const auto& f : a.focal)
    for (const auto& g : b.focal) acc[f.set | g.set] += f.mass * g.mass;
  return detail::assemble(a.frame, acc);
}

namespace detail {
// Dempster's normalisation of a conjunctive combination, with its conflict.
inline Combined normalised(const Mass& c, const char* what) {
  std::map<Set, double> acc;
  double s = 0, comp = 0;
  for (const auto& f : c.focal)
    if (f.set != 0) {
      acc[f.set] = f.mass;
      neumaier_add(s, comp, f.mass);
    }
  require(!acc.empty(), std::string(what) + ": total conflict (K = 1): no focal set of one meets a focal set of the other, and Dempster's rule is undefined");
  const double keep = s + comp;
  for (auto& [A, x] : acc) x /= keep;
  return Combined{assemble(c.frame, acc), c.conflict()};
}
}  // namespace detail

inline Combined dempster(const Mass& a, const Mass& b) { return detail::normalised(conjunctive(a, b), "dempster"); }

inline Combined dempster(const std::vector<Mass>& ms) { return detail::normalised(conjunctive(ms), "dempster"); }

inline Combined yager(const Mass& a, const Mass& b) {
  const Mass c = conjunctive(a, b);
  std::map<Set, double> acc;
  for (const auto& f : c.focal)
    if (f.set != 0) acc[f.set] += f.mass;
  acc[full(c.frame)] += c.conflict();
  return Combined{detail::assemble(c.frame, acc), c.conflict()};
}

inline Mass normalise(const Mass& m) { return detail::normalised(m, "normalise").mass; }

inline Mass discount(const Mass& m, double reliability) {
  detail::require(reliability >= 0 && reliability <= 1, "a reliability lies in [0, 1]");
  std::map<Set, double> acc;
  for (const auto& f : m.focal) acc[f.set] += reliability * f.mass;
  acc[full(m.frame)] += 1 - reliability;
  return detail::assemble(m.frame, acc);
}

inline Mass condition(const Mass& m, Set event) {
  detail::check_set(event, m.frame);
  detail::require(plausibility(m, event) > 0, "conditioning on an event of plausibility zero");
  return dempster(m, Mass{m.frame, {{event, 1.0}}}).mass;
}

}  // namespace tyche::beliefs
