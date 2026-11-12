// Classes detected in sampling units (incidence data, counts.hpp): how many there are, and what fewer
// or more units would find.
//
// T units each record which classes they hold; a class detected in Y_i units has incidence frequency
// Y_i, Q_k classes are detected in exactly k units, and U = Σ Y_i. The model is Bernoulli: class i
// turns up in each unit independently with probability π_i (Colwell et al. 2012, J. Plant Ecol. 5:
// 3–21, §2.2; Chao et al. 2014, Ecol. Monogr. 84: 45–67). It is the capture–recapture model with the
// units as occasions, for which Chao (1987, Biometrics 43: 783–791) gave the lower bound Q₁²/(2Q₂).
//
//   Chao2   Ŝ = S_obs + Q̂₀, Q̂₀ = A Q₁²/(2Q₂) if Q₂ > 0 and A Q₁(Q₁ − 1)/2 if Q₂ = 0, A = (T − 1)/T
//           (the classic form; Chao 2005 and Chao et al. 2014); or A Q₁(Q₁ − 1)/(2(Q₂ + 1)) for every
//           Q₂ (`Chao1Form::bias_corrected`). Its variance and log-normal interval are Chao1's
//           (richness.hpp) with the Q_k for the f_k and A = (T − 1)/T; when Q̂₀ = 0 the Poisson variance
//           of S_obs is conditioned on U, the analogue of n.
//   x       the detection ratio T Q̂₀/(T Q̂₀ + Q₁) of the classic Q̂₀: (T − 1)Q₁/((T − 1)Q₁ + 2Q₂) if
//           Q₂ > 0, (T − 1)(Q₁ − 1)/((T − 1)(Q₁ − 1) + 2) if Q₂ = 0 < Q₁, 0 if Q₁ = 0.
//   Ĉ_T     1 − (Q₁/U) x, the estimated share of the incidence probability Σ π_i that belongs to the
//           detected classes (Chao et al. 2014, the incidence analogue of Chao and Jost's eq. 4a).
//
// Along the number of units t (Colwell et al. 2012; Chao et al. 2014):
//
//   t ≤ T   Ŝ_t = Σ_i [1 − C(T − Y_i, t)/C(T, t)], exactly the mean number of classes over all C(T, t)
//           sets of t of the units (the sample-based rarefaction curve, Mao Tau: Colwell, Mao and
//           Chang 2004, Ecology 85: 2717–2727);
//   t > T   Ŝ_{T+t*} = S_obs + Q̂₀ (1 − x^{t*});
//   coverage   t < T: 1 − Σ_i (Y_i/U) C(T − Y_i, t)/C(T − 1, t); t = T: Ĉ_T; t > T: 1 − (Q₁/U) x^{t*+1}.
//
// Intervals by the incidence bootstrap assemblage of Chao et al. (2014): detection probabilities
// π̂_i = (Y_i/T)[1 − λ̂(1 − Y_i/T)^T] for the detected classes, λ̂ = (U/T)(1 − Ĉ_T)/Σ_i (Y_i/T)(1 − Y_i/T)^T,
// and ⌈Q̂₀⌉ undetected classes each with (U/T)(1 − Ĉ_T)/⌈Q̂₀⌉ (at most 1), so that the expected
// incidences are U; a replicate draws each class's Y_i* ~ Binomial(T, π̂_i) from calx's streams
// (replicate r on the identity (·, r, 1)). Standard errors and intervals as for abundance
// (rarefaction.hpp).
//
// Orders 1 and 2 of incidence data (Chao et al. 2014's incidence-based Shannon and Simpson diversity)
// are not written.

#pragma once

#include <tyche/classes/rarefaction.hpp>
#include <tyche/classes/richness.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tyche::classes {

inline double undetected(const Incidences& s, Chao1Form form = Chao1Form::classic) {
  const double T = static_cast<double>(s.units);
  return detail::undetected(static_cast<double>(s.uniques()), static_cast<double>(s.duplicates()), (T - 1) / T, form);
}

inline double detection_ratio(const Incidences& s) {
  return detail::detection_ratio(static_cast<double>(s.uniques()), static_cast<double>(s.duplicates()), static_cast<double>(s.units));
}

// Ĉ_T = 1 − (Q₁/U) x.
inline double coverage(const Incidences& s) {
  return 1 - static_cast<double>(s.uniques()) / static_cast<double>(s.total) * detection_ratio(s);
}

inline RichnessEstimate chao2(const Incidences& s, double level = 0.95, Chao1Form form = Chao1Form::classic) {
  const double T = static_cast<double>(s.units);
  return detail::chao(s.Q, static_cast<double>(s.observed()), (T - 1) / T, static_cast<double>(s.total), level, form);
}

// The richness expected in t units: interpolated for t ≤ T (a whole number), extrapolated above.
inline double richness_at(const Incidences& s, double t) {
  const double T = static_cast<double>(s.units);
  require(t >= 0, "richness_at", "t ≥ 0");
  if (t <= T) {
    const std::int64_t ti = detail::whole_size(t, "richness_at");
    double r = 0;
    for (std::size_t k = 1; k < s.Q.size(); k++)
      if (s.Q[k]) r += static_cast<double>(s.Q[k]) * (1 - detail::missed(s.units, static_cast<std::int64_t>(k), ti));
    return r;
  }
  const double q0 = undetected(s);
  return q0 > 0 ? static_cast<double>(s.observed()) + q0 * (1 - std::pow(detection_ratio(s), t - T)) : static_cast<double>(s.observed());
}

// The coverage expected in t units.
inline double coverage_at(const Incidences& s, double t) {
  const double T = static_cast<double>(s.units), U = static_cast<double>(s.total);
  require(t >= 0, "coverage_at", "t ≥ 0");
  if (t < T) {
    const std::int64_t ti = detail::whole_size(t, "coverage_at");
    double miss = 0;
    for (std::size_t k = 1; k < s.Q.size(); k++)
      if (s.Q[k]) miss += static_cast<double>(s.Q[k]) * static_cast<double>(k) / U * detail::missed(s.units - 1, static_cast<std::int64_t>(k) - 1, ti);
    return 1 - miss;
  }
  return 1 - static_cast<double>(s.uniques()) / U * std::pow(detection_ratio(s), t - T + 1);
}

// The detection probabilities of the incidence bootstrap assemblage: the detected classes in the
// sample's order, then ⌈Q̂₀⌉ undetected ones.
inline std::vector<double> bootstrap_detection(const Incidences& s) {
  const double T = static_cast<double>(s.units), U = static_cast<double>(s.total);
  const double deficit = (U / T) * (1 - coverage(s)), q0 = std::ceil(undetected(s) - 1e-12);
  double denom = 0;
  for (std::int64_t y : s.counts) {
    const double p = static_cast<double>(y) / T;
    denom += p * std::pow(1 - p, T);
  }
  const double lambda = deficit > 0 && denom > 0 ? deficit / denom : 0.0;
  std::vector<double> pi;
  for (std::int64_t y : s.counts) {
    const double p = static_cast<double>(y) / T;
    pi.push_back(p * (1 - lambda * std::pow(1 - p, T)));
  }
  if (q0 >= 1 && deficit > 0)
    for (double j = 0; j < q0; j++) pi.push_back(std::min(1.0, deficit / q0));
  return pi;
}

struct IncidenceCurveOptions {
  std::vector<double> sizes;  // whole numbers of units ≥ 1; empty: default_sizes(T, endpoint, knots)
  double endpoint = 0;        // 0: 2T
  std::size_t knots = 40;
  std::size_t replicates = 50;
  double level = 0.95;
  std::uint64_t seed = 20261112;
};

// The richness (order 0) and coverage along the number of units, with bootstrap intervals.
inline std::vector<CurvePoint> richness_curve(const Incidences& s, const IncidenceCurveOptions& o = {}) {
  const double Td = static_cast<double>(s.units);
  auto sizes = o.sizes.empty() ? default_sizes(s.units, o.endpoint, o.knots) : o.sizes;
  for (double t : sizes) require(std::isfinite(t) && t >= 1 && t == std::floor(t), "richness_curve", "sizes are whole numbers ≥ 1");
  std::sort(sizes.begin(), sizes.end());
  const std::size_t M = sizes.size();
  auto evaluate = [&](const Incidences& a, std::vector<double>& est, std::vector<double>& cov) {
    est.assign(M, 0.0);
    cov.assign(M, 0.0);
    for (std::size_t j = 0; j < M; j++) est[j] = richness_at(a, sizes[j]), cov[j] = coverage_at(a, sizes[j]);
  };
  std::vector<double> est, cov, se(M, 0.0), cse(M, 0.0);
  evaluate(s, est, cov);
  if (o.replicates >= 2) {
    const auto pi = bootstrap_detection(s);
    std::vector<double> sum(M, 0.0), sum2(M, 0.0), csum(M, 0.0), csum2(M, 0.0), e, c, y(pi.size());
    std::size_t used = 0;
    for (std::size_t r = 0; r < o.replicates; r++) {
      calx::random::Stream rng(o.seed, 0x636c6173u, static_cast<std::uint32_t>(r), 1);
      for (std::size_t i = 0; i < pi.size(); i++) y[i] = static_cast<double>(rng.binomial(s.units, pi[i]));
      bool any = false;
      for (double v : y) any = any || v > 0;
      if (!any) continue;  // a replicate that detects nothing (possible only for tiny samples) is skipped
      evaluate(incidences(y, s.units), e, c);
      used++;
      for (std::size_t k = 0; k < M; k++) sum[k] += e[k], sum2[k] += e[k] * e[k], csum[k] += c[k], csum2[k] += c[k] * c[k];
    }
    if (used >= 2) {
      const double B = static_cast<double>(used);
      auto sd = [&](double a, double b) { return std::sqrt(std::max(0.0, (b - a * a / B) / (B - 1))); };
      for (std::size_t k = 0; k < M; k++) se[k] = sd(sum[k], sum2[k]), cse[k] = sd(csum[k], csum2[k]);
    }
  }
  const double z = detail::normal_quantile_two_sided(o.level);
  std::vector<CurvePoint> out;
  for (std::size_t j = 0; j < M; j++) {
    CurvePoint p;
    p.size = sizes[j];
    p.method = sizes[j] < Td ? Method::interpolated : sizes[j] > Td ? Method::extrapolated : Method::observed;
    p.order = 0;
    p.estimate = est[j];
    p.se = se[j];
    p.lower = p.estimate - z * p.se;
    p.upper = p.estimate + z * p.se;
    p.coverage = cov[j];
    p.coverage_se = cse[j];
    p.coverage_lower = std::clamp(p.coverage - z * p.coverage_se, 0.0, 1.0);
    p.coverage_upper = std::clamp(p.coverage + z * p.coverage_se, 0.0, 1.0);
    out.push_back(p);
  }
  return out;
}

}  // namespace tyche::classes
