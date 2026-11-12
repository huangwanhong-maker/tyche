// The diversity of the whole population, estimated from a sample of individuals: the asymptotes of
// the curves of rarefaction.hpp, for the orders q = 0, 1, 2.
//
//   q = 0   Chao1 (richness.hpp, the classic form).
//   q = 1   exp(Ĥ), Ĥ the estimator of Shannon's entropy of Chao, Wang and Jost (2013, Methods Ecol.
//           Evol. 4: 1091–1100, the estimator they propose, by the discovery rates of new classes):
//
//             Ĥ = Σ_{1≤X_i≤n−1} (X_i/n) Σ_{k=X_i}^{n−1} 1/k
//                 + (f₁/n)(1 − A)^{1−n} [−ln A − Σ_{r=1}^{n−1} (1 − A)^r / r],
//
//           A = 2f₂/((n − 1)f₁ + 2f₂) if f₂ > 0, 2/((n − 1)(f₁ − 1) + 2) if f₂ = 0 < f₁, and 1 if
//           f₁ = f₂ = 0; 1 − A is the detection ratio x of richness.hpp. It rests on the expansion
//           H = Σ_{k≥1} (1/k) Σ_i p_i(1 − p_i)^k: its first n − 1 terms have the unbiased estimates
//           Σ_i (X_i/n) C(n − X_i, k)/C(n − 1, k), whose weighted sum is the first line (an identity,
//           Σ_{k=1}^{n−1} (1/k) C(n − X, k)/C(n − 1, k) = Σ_{k=X}^{n−1} 1/k, held by the checks), and the
//           tail is estimated from the singletons and doubletons. Its bias is small for n of a few
//           tens and up, and much smaller than that of the plug-in −Σ (X_i/n) ln(X_i/n).
//   q = 2   n(n − 1) / Σ X_i(X_i − 1), the reciprocal of the unbiased estimate Σ X_i(X_i − 1)/(n(n − 1))
//           of Σ p_i²; +∞ when every class is a singleton.
//
// Numerics. The inner sums of the first line are ψ(n) − ψ(X_i) (calx's digamma, absolute error about
// 10⁻¹⁵). The second line is (f₁/n) Σ_{j≥1} x^j/(n − 1 + j), x = 1 − A, since −ln A = Σ_{r≥1} x^r/r; it
// is summed as that convergent series when n(1 − x) ≥ 1 (at most about 40/(1 − x) ≤ 40n terms), and
// from the bracket with a compensated sum of n − 1 terms otherwise, where x^{1−n} < e and the bracket
// loses at most a few digits to cancellation.

#pragma once

#include <tyche/classes/richness.hpp>

#include <calx/special_functions/gamma.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace tyche::classes {

namespace detail {

// Σ_{j≥1} x^j / (c + j) for 0 ≤ x < 1 and c ≥ 0, with n = c + 1.
inline double singleton_tail(double x, double n) {
  if (x <= 0) return 0;
  const double c = n - 1;
  if (n * (1 - x) >= 1) {
    double s = 0, t = x;
    for (double j = 1;; j++) {
      const double term = t / (c + j);
      s += term;
      if (term <= 1e-17 * (1 - x) * s) break;  // the rest is below term/(1 − x)
      t *= x;
    }
    return s;
  }
  // x^{−c} [−ln(1 − x) − Σ_{r=1}^{c} x^r / r], compensated.
  double s = 0, comp = 0, t = 1;
  for (double r = 1; r <= c; r++) {
    t *= x;
    const double y = -t / r, u = s + y;
    comp += std::fabs(s) >= std::fabs(y) ? (s - u) + y : (y - u) + s;
    s = u;
  }
  return std::exp(-c * std::log(x)) * ((-std::log1p(-x)) + (s + comp));
}

}  // namespace detail

// Chao, Wang and Jost's estimate of Shannon's entropy, in nats.
inline double entropy_estimate(const Abundances& s) {
  const double n = static_cast<double>(s.n);
  const double psi_n = calx::special_functions::digamma(n);
  double h = 0;
  for (std::size_t k = 1; k < s.f.size(); k++) {
    if (s.f[k] == 0 || static_cast<std::int64_t>(k) >= s.n) continue;
    const double kd = static_cast<double>(k);
    h += static_cast<double>(s.f[k]) * (kd / n) * (psi_n - calx::special_functions::digamma(kd));
  }
  const double f1 = static_cast<double>(s.singletons());
  if (f1 > 0) h += f1 / n * detail::singleton_tail(detection_ratio(s), n);
  return h;
}
inline double entropy_estimate(std::span<const double> counts) { return entropy_estimate(abundances(counts)); }

// −Σ (X_i/n) ln(X_i/n), the entropy of the sample's own law.
inline double observed_entropy(const Abundances& s) {
  const double n = static_cast<double>(s.n);
  double h = 0;
  for (std::size_t k = 1; k < s.f.size(); k++)
    if (s.f[k] > 0) {
      const double p = static_cast<double>(k) / n;
      h -= static_cast<double>(s.f[k]) * p * std::log(p);
    }
  return h;
}

// Σ X_i(X_i − 1) / (n(n − 1)), the unbiased estimate of Σ p_i²; NaN when n = 1.
inline double concentration_estimate(const Abundances& s) {
  const double n = static_cast<double>(s.n);
  if (s.n < 2) return std::numeric_limits<double>::quiet_NaN();
  double t = 0;
  for (std::size_t k = 2; k < s.f.size(); k++) t += static_cast<double>(s.f[k]) * static_cast<double>(k) * static_cast<double>(k - 1);
  return t / (n * (n - 1));
}

struct Asymptotic {
  double observed[3] = {0, 0, 0};   // ^0D, ^1D, ^2D of the sample's own law
  double estimate[3] = {0, 0, 0};   // the asymptotic estimates of orders 0, 1, 2
  double entropy = 0;               // Ĥ (Chao, Wang and Jost), nats
  double entropy_observed = 0;      // the plug-in entropy
  double coverage = 0;              // Ĉ_n
  RichnessEstimate richness;        // Chao1, with its interval
};

inline Asymptotic asymptotic_diversity(const Abundances& s, double level = 0.95) {
  Asymptotic a;
  const double n = static_cast<double>(s.n);
  double sq = 0;
  for (std::size_t k = 1; k < s.f.size(); k++) sq += static_cast<double>(s.f[k]) * static_cast<double>(k) * static_cast<double>(k);
  a.entropy_observed = observed_entropy(s);
  a.observed[0] = static_cast<double>(s.observed());
  a.observed[1] = std::exp(a.entropy_observed);
  a.observed[2] = n * n / sq;
  a.richness = chao1(s, level);
  a.entropy = entropy_estimate(s);
  a.estimate[0] = a.richness.estimate;
  a.estimate[1] = std::exp(a.entropy);
  const double c = concentration_estimate(s);
  a.estimate[2] = c > 0 ? 1 / c : std::numeric_limits<double>::infinity();
  a.coverage = coverage(s);
  return a;
}
inline Asymptotic asymptotic_diversity(std::span<const double> counts, double level = 0.95) { return asymptotic_diversity(abundances(counts), level); }

}  // namespace tyche::classes
