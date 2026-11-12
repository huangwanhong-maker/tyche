// How many classes a population holds, and how much of it a sample has seen: estimates from a
// sample of individuals (abundance data, counts.hpp).
//
// Undetected classes. The rare classes carry the information about the unseen: Chao (1987,
// Biometrics 43: 783–791) bounds the expected number of undetected classes below by f₁²/(2f₂), which
// is nearly unbiased when the rarest classes are about equally rare. With the factor A = (n − 1)/n
// that removes its bias at small n (Chao 2005, Encyclopedia of Statistical Sciences, 2nd ed., "Species
// estimation and applications"), two forms are given:
//
//   `Chao1Form::classic`         f̂₀ = A f₁²/(2f₂) if f₂ > 0, and A f₁(f₁ − 1)/2 if f₂ = 0 (the form of
//                                Chao et al. 2014, Ecol. Monogr. 84: 45–67, and of iNEXT);
//   `Chao1Form::bias_corrected`  f̂₀ = A f₁(f₁ − 1)/(2(f₂ + 1)) for every f₂ (Chao 2005; EstimateS's
//                                "bias-corrected Chao1"; scikit-bio's without the factor A).
//
// Chao1 is Ŝ = S_obs + f̂₀.
//
// Its variance is Chao's (1987, §2) delta method with the multinomial covariances of the frequency
// counts among Ŝ cells, Cov(f_i, f_j) = f_i(δ_ij − f_j/Ŝ): with a_k = ∂Ŝ/∂f_k = 1 + d_k (d_k = 0 for
// k ≥ 3), D = Σ d_k f_k and T = f̂₀,
//
//   var Ŝ = Σ_k a_k² f_k − (Σ_k a_k f_k)²/Ŝ = Σ_k d_k² f_k + T − (D − T)²/Ŝ,
//
// the second form free of cancellation. For the classic form with f₂ > 0, Ŝ is homogeneous of degree
// one in (f₁, f₂), D = T, and this is Chao's closed form f₂[A r²/2 + A² r³ + A² r⁴/4], r = f₁/f₂;
// with f₂ = 0 it is A f₁(f₁ − 1)/2 + A² f₁(2f₁ − 1)²/4 − A² f₁⁴/(4Ŝ). (EstimateS's variance of the
// bias-corrected form drops the term (D − T)²/Ŝ.) The interval is Chao's (1987) for T = Ŝ − S_obs
// taken as log-normal, since Ŝ ≥ S_obs and its law is skewed:
//
//   K = exp(z √ln(1 + var/T²)),   [S_obs + T/K, S_obs + T K],   z the normal quantile of the level.
//
// When nothing is estimated undetected (f₁ = 0, or f₁ = 1 and f₂ = 0), T = 0 and that interval
// collapses. Then, under Poisson sampling (each class seen X ~ Poisson(λ), λ estimated by its count)
// conditioned on n, var S_obs = Σ_k f_k(e^{−k} − e^{−2k}) − (Σ_k k e^{−k} f_k)²/n; P = Σ_k f_k e^{−k}/S_obs
// estimates the chance that a class is missed, and the interval is (S_obs ± z √var)/(1 − P), its lower
// end not below S_obs (the form of iNEXT's ChaoRichness for this case).
//
// Coverage. The sample coverage C_n is the total probability of the classes the sample holds;
// 1 − C_n is the chance that one more individual belongs to a new class. Turing's estimate is
// 1 − f₁/n (Good 1953, Biometrika 40: 237–264, after Turing); Chao and Jost (2012, Ecology 93: 2533–2547,
// eq. 4a) correct it with the doubletons:
//
//   Ĉ_n = 1 − (f₁/n) x,   x = (n − 1) f₁ / ((n − 1) f₁ + 2 f₂),
//
// and x is the detection ratio n f̂₀/(n f̂₀ + f₁) of the classic f̂₀. With f₂ = 0 the classic f̂₀ gives
// x = (n − 1)(f₁ − 1)/((n − 1)(f₁ − 1) + 2), as Chao et al. (2014) and iNEXT use; x = 0 when f₁ = 0.
// The same x governs extrapolation (rarefaction.hpp) and the estimate of entropy (asymptotic.hpp).
//
// ACE, the abundance-based coverage estimator (Chao and Lee 1992, J. Amer. Statist. Assoc. 87:
// 210–217; Chao, Ma and Yang 1993, Biometrika 80: 193–201), splits the classes at a threshold κ
// (10 by default; Chao et al. 2000, Statist. Sinica 10: 227–246): S_abund classes with more than κ
// individuals are taken as seen, and the S_rare others, holding N_rare individuals, are scaled by
// their coverage C_ace = 1 − f₁/N_rare and corrected for their heterogeneity,
//
//   γ̂² = max[(S_rare/C_ace) Σ_{k≤κ} k(k − 1) f_k / (N_rare(N_rare − 1)) − 1, 0],
//   Ŝ_ace = S_abund + S_rare/C_ace + (f₁/C_ace) γ̂²,
//
// in the form of EstimateS's manual, which scikit-bio also follows. It is undefined when every rare
// class is a singleton (C_ace = 0).
//
// Validity. All are asymptotic in n with S fixed; Chao1 and ACE estimate lower bounds that become
// sharp as the rare classes become alike, and say nothing of classes too rare to have been seen even
// once in any likely sample.

#pragma once

#include <tyche/classes/counts.hpp>
#include <tyche/probability/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace tyche::classes {

enum class Chao1Form { classic, bias_corrected };

struct RichnessEstimate {
  double estimate = 0;    // Ŝ
  double observed = 0;    // S_obs
  double undetected = 0;  // f̂₀ = Ŝ − S_obs
  double variance = 0, se = 0;
  double lower = 0, upper = 0, level = 0.95;
  bool lognormal = true;  // false: nothing estimated undetected; the interval from the variance of S_obs
};

namespace detail {

inline double normal_quantile_two_sided(double level) {
  require(level > 0 && level < 1, "interval", "the level lies in (0, 1)");
  return tyche::probability::Normal::standard_quantile(1 - (1 - level) / 2);
}

// f̂₀ from the singletons f1, doubletons f2 and the factor A.
inline double undetected(double f1, double f2, double A, Chao1Form form) {
  if (form == Chao1Form::classic && f2 > 0) return A * f1 * f1 / (2 * f2);
  return A * f1 * (f1 - 1) / (2 * (f2 + 1));
}

// x = n f̂₀ / (n f̂₀ + f₁) of the classic f̂₀, with x = 0 when f̂₀ = 0; `size` is n (or T).
inline double detection_ratio(double f1, double f2, double size) {
  if (f1 <= 0) return 0;
  if (f2 > 0) return (size - 1) * f1 / ((size - 1) * f1 + 2 * f2);
  return (size - 1) * (f1 - 1) / ((size - 1) * (f1 - 1) + 2);
}

// Chao's estimate, variance and interval; `f` the frequency counts (f[k] for k ≥ 1), `total` the n
// (or U) on which the Poisson variance of S_obs is conditioned.
inline RichnessEstimate chao(std::span<const std::int64_t> f, double sobs, double A, double total, double level, Chao1Form form) {
  const double f1 = f.size() > 1 ? static_cast<double>(f[1]) : 0, f2 = f.size() > 2 ? static_cast<double>(f[2]) : 0;
  RichnessEstimate r;
  r.observed = sobs;
  r.level = level;
  const double z = normal_quantile_two_sided(level);
  double T, d1, d2;
  if (form == Chao1Form::classic && f2 > 0) {
    T = A * f1 * f1 / (2 * f2);
    d1 = A * f1 / f2;
    d2 = -A * f1 * f1 / (2 * f2 * f2);
  } else {
    T = A * f1 * (f1 - 1) / (2 * (f2 + 1));
    d1 = A * (2 * f1 - 1) / (2 * (f2 + 1));
    d2 = -A * f1 * (f1 - 1) / (2 * (f2 + 1) * (f2 + 1));
  }
  r.undetected = T;
  r.estimate = sobs + T;
  if (T > 0) {
    const double D = d1 * f1 + d2 * f2, E = d1 * d1 * f1 + d2 * d2 * f2;
    r.variance = std::max(0.0, E + T - (D - T) * (D - T) / r.estimate);
    const double K = std::exp(z * std::sqrt(std::log1p(r.variance / (T * T))));
    r.lower = sobs + T / K;
    r.upper = sobs + T * K;
    r.lognormal = true;
  } else {
    double v = 0, m = 0, P = 0;
    for (std::size_t k = 1; k < f.size(); k++) {
      if (f[k] == 0) continue;
      const double e = std::exp(-static_cast<double>(k)), fk = static_cast<double>(f[k]);
      v += fk * (e - e * e);
      m += static_cast<double>(k) * e * fk;
      P += fk * e;
    }
    r.variance = std::max(0.0, v - m * m / total);
    P /= sobs;
    const double s = z * std::sqrt(r.variance);
    r.lower = std::max(sobs, (sobs - s) / (1 - P));
    r.upper = (sobs + s) / (1 - P);
    r.lognormal = false;
  }
  r.se = std::sqrt(r.variance);
  return r;
}

}  // namespace detail

// f̂₀, the estimated number of undetected classes.
inline double undetected(const Abundances& s, Chao1Form form = Chao1Form::classic) {
  const double n = static_cast<double>(s.n);
  return detail::undetected(static_cast<double>(s.singletons()), static_cast<double>(s.doubletons()), (n - 1) / n, form);
}

// x, the detection ratio of the classic f̂₀.
inline double detection_ratio(const Abundances& s) {
  return detail::detection_ratio(static_cast<double>(s.singletons()), static_cast<double>(s.doubletons()), static_cast<double>(s.n));
}

// Ĉ_n = 1 − (f₁/n) x.
inline double coverage(const Abundances& s) {
  return 1 - static_cast<double>(s.singletons()) / static_cast<double>(s.n) * detection_ratio(s);
}
inline double coverage(std::span<const double> counts) { return coverage(abundances(counts)); }

// Turing's coverage 1 − f₁/n.
inline double turing_coverage(const Abundances& s) { return 1 - static_cast<double>(s.singletons()) / static_cast<double>(s.n); }

// Chao1 with its variance and interval.
inline RichnessEstimate chao1(const Abundances& s, double level = 0.95, Chao1Form form = Chao1Form::classic) {
  const double n = static_cast<double>(s.n);
  return detail::chao(s.f, static_cast<double>(s.observed()), (n - 1) / n, n, level, form);
}
inline RichnessEstimate chao1(std::span<const double> counts, double level = 0.95, Chao1Form form = Chao1Form::classic) {
  return chao1(abundances(counts), level, form);
}

struct AceEstimate {
  double estimate = 0;  // Ŝ_ace; NaN when undefined
  double observed = 0;
  double rare = 0, abundant = 0;  // S_rare, S_abund
  double rare_individuals = 0;    // N_rare
  double coverage_rare = 0;       // C_ace
  double cv_squared = 0;          // γ̂²
  std::int64_t threshold = 10;    // κ
  bool defined = true;            // false when every rare class is a singleton
};

inline AceEstimate ace(const Abundances& s, std::int64_t threshold = 10) {
  require(threshold >= 1, "ace", "the threshold is at least one");
  AceEstimate a;
  a.threshold = threshold;
  a.observed = static_cast<double>(s.observed());
  double sum_kk = 0;
  for (std::size_t k = 1; k < s.f.size(); k++) {
    const double fk = static_cast<double>(s.f[k]), kd = static_cast<double>(k);
    if (static_cast<std::int64_t>(k) <= threshold) {
      a.rare += fk;
      a.rare_individuals += kd * fk;
      sum_kk += kd * (kd - 1) * fk;
    } else {
      a.abundant += fk;
    }
  }
  if (a.rare == 0) {
    a.estimate = a.abundant;
    a.coverage_rare = 1;
    return a;
  }
  const double f1 = static_cast<double>(s.singletons());
  a.coverage_rare = 1 - f1 / a.rare_individuals;
  if (!(a.coverage_rare > 0)) {
    a.defined = false;
    a.estimate = std::numeric_limits<double>::quiet_NaN();
    return a;
  }
  a.cv_squared = std::max(a.rare / a.coverage_rare * sum_kk / (a.rare_individuals * (a.rare_individuals - 1)) - 1, 0.0);
  a.estimate = a.abundant + a.rare / a.coverage_rare + f1 / a.coverage_rare * a.cv_squared;
  return a;
}
inline AceEstimate ace(std::span<const double> counts, std::int64_t threshold = 10) { return ace(abundances(counts), threshold); }

}  // namespace tyche::classes
