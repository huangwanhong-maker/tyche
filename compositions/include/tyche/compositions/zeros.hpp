// Zeros in compositions: multiplicative replacement of rounded zeros, and the refusal of essential ones.
//
// A zero part has no log-ratio, so nothing in Aitchison's geometry is defined at it. Martín-Fernández,
// Barceló-Vidal and Pawlowsky-Glahn (2003, Math. Geol. 35: 253–278) distinguish two kinds:
//
//   ROUNDED zeros   the part is present but below what was recorded (a detection limit, a resolution:
//                   an activity of a few seconds in a diary kept by the quarter hour). The value is
//                   missing but known to be small, and may be imputed.
//   ESSENTIAL zeros the part is truly absent (structural: no paid work for a person without a job).
//                   Nothing is imputed; the composition belongs to a subcomposition of the other parts,
//                   or the zero is modelled apart (for example a two-part model of presence and amount).
//
// MULTIPLICATIVE REPLACEMENT (Martín-Fernández et al. 2003). For x closed to a total c, with Z the set
// of its zero parts and δ_j > 0 the value imputed for each j ∈ Z,
//
//   r_j = δ_j                              for j ∈ Z,
//   r_j = (1 − Σ_{k∈Z} δ_k / c) · x_j      otherwise.
//
// r is closed to c, and the ratios among the non-zero parts are kept exactly (r_i/r_j = x_i/x_j), so
// every subcomposition of the parts that were present is unchanged: the property for which they
// prefer it to additive replacement. The δ are either given directly, or as a fraction of each part's
// detection limit; in their study the distortion was least with δ near 65% of the limit, the default
// fraction here (a rule of thumb, not an optimum for every sample).
//
// Refusals, each with its reason: a zero in a part marked essential; δ_j ≤ 0; Σ_{k∈Z} δ_k ≥ c (the
// non-zero parts would be scaled to nothing or below); a composition with no non-zero part.
//
// UNITS. x is read in its own units and closed to `total` (by default its own sum, so that minutes stay
// minutes); δ and detection limits are in the units of that closure.

#pragma once

#include <tyche/compositions/simplex.hpp>

#include <calx/linalg/rectangular.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace tyche::compositions {

struct ZeroOptions {
  // One of the two ways to give δ: detection limits (one per part, in the closure's units), δ_j being
  // `fraction` of the limit of each zero part; or, when no limits are given, one `delta` for every zero.
  std::vector<double> detection_limits;
  double fraction = 0.65;
  double delta = std::numeric_limits<double>::quiet_NaN();
  // The total to close to; NaN: the composition's own sum.
  double total = std::numeric_limits<double>::quiet_NaN();
  // Parts whose zeros are essential (one flag per part, or empty): a zero there is refused.
  std::vector<bool> essential;
};

// The composition with its rounded zeros replaced multiplicatively; a composition without zeros comes
// back closed to the total and otherwise unchanged.
inline std::vector<double> replace_zeros(std::span<const double> x, const ZeroOptions& o) {
  require(x.size() >= 2, "replace_zeros", "a composition has at least two parts");
  require(o.essential.empty() || o.essential.size() == x.size(), "replace_zeros", "one essential flag per part, or none");
  require(o.detection_limits.empty() || o.detection_limits.size() == x.size(), "replace_zeros", "one detection limit per part, or none");
  double sum = 0;
  for (double v : x) {
    require(std::isfinite(v) && v >= 0, "replace_zeros", "parts are non-negative and finite");
    sum += v;
  }
  require(sum > 0, "replace_zeros", "at least one part is not zero");
  const double c = std::isnan(o.total) ? sum : o.total;
  require(std::isfinite(c) && c > 0, "replace_zeros", "the total is positive and finite");
  auto r = closure(x, c);
  double imputed = 0;
  std::vector<double> d(x.size(), 0.0);
  for (std::size_t j = 0; j < x.size(); j++) {
    if (x[j] != 0) continue;
    require(o.essential.empty() || !o.essential[j], "replace_zeros",
            "a zero in a part marked essential is a true absence and is not imputed: analyse the subcomposition of the parts present, or "
            "model the zero apart (Martín-Fernández et al. 2003)");
    if (!o.detection_limits.empty()) {
      require(std::isfinite(o.detection_limits[j]) && o.detection_limits[j] > 0, "replace_zeros", "detection limits are positive and finite");
      require(std::isfinite(o.fraction) && o.fraction > 0, "replace_zeros", "the fraction of the detection limit is positive");
      d[j] = o.fraction * o.detection_limits[j];
    } else {
      require(!std::isnan(o.delta), "replace_zeros", "a zero part needs a value to impute: give detection limits or delta");
      require(std::isfinite(o.delta) && o.delta > 0, "replace_zeros", "the imputed value delta is positive and finite");
      d[j] = o.delta;
    }
    imputed += d[j];
  }
  if (imputed == 0) return r;
  require(imputed < c, "replace_zeros", "the imputed values together reach the total: the non-zero parts would vanish; lower delta");
  const double scale = 1 - imputed / c;
  for (std::size_t j = 0; j < x.size(); j++) r[j] = x[j] == 0 ? d[j] : r[j] * scale;
  return r;
}

// Each row of a sample (n × D) replaced as above, with the same options.
inline Rectangular replace_zeros(const Rectangular& X, const ZeroOptions& o) {
  Rectangular Y(X.rows, X.cols);
  for (std::size_t i = 0; i < X.rows; i++) {
    const auto row = X.row(i);
    const auto y = replace_zeros(row, o);
    for (std::size_t j = 0; j < X.cols; j++) Y(i, j) = y[j];
  }
  return Y;
}

// Whether a composition (or sample) holds any zero part.
inline bool has_zeros(std::span<const double> x) {
  for (double v : x)
    if (v == 0) return true;
  return false;
}

}  // namespace tyche::compositions
