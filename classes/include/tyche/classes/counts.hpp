// Samples of classes: individuals counted by class, or classes detected in sampling units.
//
// Two kinds of sample are read (Colwell et al. 2012, J. Plant Ecol. 5: 3–21, §2; Chao et al. 2014,
// Ecol. Monogr. 84: 45–67, "Sampling models and data"):
//
//   abundance   n individuals drawn at random from a population, each recorded by its class. X_i is
//               the count of class i, n = Σ X_i, and f_k = #{i : X_i = k} are the frequency counts:
//               f₁ the singletons, f₂ the doubletons, S_obs = Σ_{k≥1} f_k the classes observed. The
//               model is multinomial: n draws with replacement from a law p on S classes.
//   incidence   T sampling units (plots, quadrats, traps, days), each recording which classes it
//               holds. Y_i is the number of units in which class i was detected, U = Σ Y_i the
//               incidences, and Q_k = #{i : Y_i = k}: Q₁ the uniques, Q₂ the duplicates. The model is
//               Bernoulli: class i is detected in each unit independently with probability π_i.
//
// Counts are given as doubles (std::span<const double>, the library's plain interface) and must be
// non-negative integers below 2⁵³; a zero is a class of a list that was not seen, and is dropped.

#pragma once

#include <calx/linalg/rectangular.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::classes {

inline void require(bool ok, const char* where, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::classes::") + where + ": " + what);
}

// A count given as a double, checked to be a whole number.
inline std::int64_t whole(double x, const char* where) {
  require(std::isfinite(x) && x >= 0 && x == std::floor(x) && x < 9007199254740992.0, where, "counts are non-negative integers");
  return static_cast<std::int64_t>(x);
}

struct Abundances {
  std::vector<std::int64_t> counts;  // X_i > 0 of the observed classes, in the order given
  std::int64_t n = 0;                // individuals, Σ X_i
  std::vector<std::int64_t> f;       // f[k] = #{i : X_i = k}, k = 0, …, max X_i (f[0] = 0)

  std::size_t observed() const { return counts.size(); }
  std::int64_t frequency(std::int64_t k) const { return k >= 0 && k < static_cast<std::int64_t>(f.size()) ? f[static_cast<std::size_t>(k)] : 0; }
  std::int64_t singletons() const { return frequency(1); }
  std::int64_t doubletons() const { return frequency(2); }
};

inline Abundances abundances(std::span<const double> counts) {
  Abundances a;
  std::int64_t largest = 0;
  for (double x : counts) {
    const std::int64_t c = whole(x, "abundances");
    if (c == 0) continue;
    a.counts.push_back(c);
    a.n += c;
    if (c > largest) largest = c;
  }
  require(a.n > 0, "abundances", "a sample holds at least one individual");
  a.f.assign(static_cast<std::size_t>(largest) + 1, 0);
  for (std::int64_t c : a.counts) a.f[static_cast<std::size_t>(c)]++;
  return a;
}

struct Incidences {
  std::int64_t units = 0;            // T
  std::vector<std::int64_t> counts;  // Y_i > 0 of the detected classes, in the order given
  std::int64_t total = 0;            // U = Σ Y_i
  std::vector<std::int64_t> Q;       // Q[k] = #{i : Y_i = k}, k = 0, …, T (Q[0] = 0)

  std::size_t observed() const { return counts.size(); }
  std::int64_t frequency(std::int64_t k) const { return k >= 0 && k < static_cast<std::int64_t>(Q.size()) ? Q[static_cast<std::size_t>(k)] : 0; }
  std::int64_t uniques() const { return frequency(1); }
  std::int64_t duplicates() const { return frequency(2); }
};

// From the incidence frequencies Y_i (each at most T) and the number of units T.
inline Incidences incidences(std::span<const double> frequencies, std::int64_t units) {
  require(units >= 1, "incidences", "at least one sampling unit");
  Incidences s;
  s.units = units;
  s.Q.assign(static_cast<std::size_t>(units) + 1, 0);
  for (double x : frequencies) {
    const std::int64_t y = whole(x, "incidences");
    if (y == 0) continue;
    require(y <= units, "incidences", "a class is detected in at most every unit");
    s.counts.push_back(y);
    s.total += y;
    s.Q[static_cast<std::size_t>(y)]++;
  }
  require(s.total > 0, "incidences", "at least one detection");
  return s;
}

// From a table of detections: a row for each unit, a column for each class; an entry above zero
// (a count, a cover, a one) is a detection.
inline Incidences incidences(const calx::linalg::Rectangular& detections) {
  std::vector<double> y(detections.cols, 0.0);
  for (std::size_t u = 0; u < detections.rows; u++)
    for (std::size_t i = 0; i < detections.cols; i++) {
      const double v = detections(u, i);
      require(std::isfinite(v) && v >= 0, "incidences", "detections are non-negative");
      if (v > 0) y[i] += 1;
    }
  return incidences(y, static_cast<std::int64_t>(detections.rows));
}

}  // namespace tyche::classes
