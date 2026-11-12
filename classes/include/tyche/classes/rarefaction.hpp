// What a smaller or a larger sample would find: rarefaction and extrapolation of Hill numbers, and of
// coverage, along the size m of a sample of individuals.
//
// Diversity seen in a sample grows with its size, so two samples are compared at a common size or,
// better, at a common coverage (Chao and Jost 2012, Ecology 93: 2533–2547). Chao et al. (2014, Ecol.
// Monogr. 84: 45–67, §"Rarefaction and extrapolation", the formulas also of iNEXT) join the curve
// below the sample's size (interpolation) to the curve above it (extrapolation) at the sample itself.
//
// Interpolation, m ≤ n. A subsample of m of the n individuals drawn without replacement holds a
// class of X_i individuals k times with the hypergeometric probability C(X_i, k)C(n − X_i, m − k)/C(n, m),
// so the expected number of classes with exactly k individuals is
//
//   f̂_k(m) = Σ_i C(X_i, k) C(n − X_i, m − k) / C(n, m),
//
// and the Hill number at m is taken from these expected counts:
//
//   q = 0   Ŝ_m = Σ_i [1 − C(n − X_i, m)/C(n, m)]   (Hurlbert 1971, Ecology 52: 577–586; exactly the
//           mean number of classes over all C(n, m) subsamples);
//   q = 1   exp(−Σ_k (k/m) ln(k/m) f̂_k(m))         (the exponential of the subsamples' mean entropy);
//   q = 2   1 / [1/m + ((m − 1)/m) Σ X_i(X_i − 1)/(n(n − 1))]
//                                                 (the reciprocal of their mean Simpson concentration);
//   q       (Σ_k (k/m)^q f̂_k(m))^{1/(1−q)} for any other finite q.
//
// Each is a function of the subsamples' mean of a sample statistic (richness, entropy, concentration,
// Σ(k/m)^q), so the checks hold them to the mean over every subsample enumerated. Ŝ_m and the mean
// concentration are moreover unbiased for the expectations under the population law of a sample of m,
// Σ_i[1 − (1 − p_i)^m] and 1/m + ((m − 1)/m)Σ p_i², since a subsample of a multinomial sample is one.
// At m = n every formula returns the sample's own Hill number.
//
// Extrapolation, m = n + m* ≥ n, for q = 0, 1, 2:
//
//   q = 0   Ŝ_{n+m*} = S_obs + f̂₀ [1 − (1 − f₁/(n f̂₀ + f₁))^{m*}] = S_obs + f̂₀ (1 − x^{m*})
//           (Shen, Chao and Lin 2003, Ecology 84: 798–804; Chao et al. 2014), f̂₀ the classic Chao1
//           term and x the detection ratio (richness.hpp);
//   q = 1   exp[(n/m) Ĥ_obs + (m*/m) Ĥ], the entropies of the sample and of the population
//           (asymptotic.hpp) weighted by the shares of the sample in the larger one (Chao et al. 2014);
//   q = 2   1 / [1/m + ((m − 1)/m) Σ X_i(X_i − 1)/(n(n − 1))], the interpolation formula continued,
//           which estimates without bias the expected Simpson concentration of a sample of m.
//
// Each tends to its asymptote (asymptotic.hpp) as m → ∞ and equals the sample's value at m* = 0, so
// the joined curve is continuous at n. Chao et al. (2014) advise extrapolating richness no further
// than two or three times n (its bias grows beyond); orders 1 and 2 extrapolate further reliably.
// Under heterogeneous rare classes the q = 0 extrapolation inherits Chao1's downward bias (the checks
// measure 2% at 2n on a geometric law).
//
// Coverage at m (Chao and Jost 2012; Chao et al. 2014):
//
//   m < n   Ĉ_m = 1 − Σ_i (X_i/n) C(n − X_i, m)/C(n − 1, m), unbiased for 1 − Σ p_i(1 − p_i)^m, the
//           expected coverage of a sample of m (the chance that the (m + 1)-th individual drawn
//           belongs to a class among the first m); Ĉ_{n−1} = 1 − f₁/n, Turing's;
//   m = n   Ĉ_n = 1 − (f₁/n) x (richness.hpp);
//   m > n   Ĉ_{n+m*} = 1 − (f₁/n) x^{m*+1}.
//
// `size_at_coverage` inverts it: the least whole m ≤ n with Ĉ_m ≥ C, or above n the real m solving
// the extrapolated form.
//
// Intervals by Chao et al.'s (2014; Chao and Jost 2012, Appendix) bootstrap assemblage: the observed
// classes with probabilities p̂_i = (X_i/n)[1 − λ̂ (1 − X_i/n)^n], λ̂ = (1 − Ĉ_n)/Σ_i (X_i/n)(1 − X_i/n)^n,
// which lowers the probabilities of the rare observed classes to make room for the undetected, and
// ⌈f̂₀⌉ undetected classes sharing 1 − Ĉ_n equally. Samples of n are drawn from it (multinomially, by
// calx's streams under a seed, replicate r on the stream identity (·, r, 0), so a run repeats itself
// exactly), each is estimated at every size, and the standard error is the replicates' standard
// deviation (divisor B − 1); the interval is estimate ± z·se (iNEXT's), coverage's clipped to [0, 1].
//
// Accuracy. Hypergeometric terms are summed from the mode outward by the ratio of successive terms
// and normalised by their sum, so they are exact to rounding; C(n − X, m)/C(n, m) is a product for
// X ≤ 64 and from log Γ above, where its relative error is about ε·ln Γ(n), some 10⁻¹⁰ at n ~ 10⁵.

#pragma once

#include <tyche/classes/asymptotic.hpp>
#include <tyche/classes/richness.hpp>

#include <calx/random/multinomial.hpp>
#include <calx/random/stream.hpp>
#include <calx/special_functions/gamma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace tyche::classes {

namespace detail {

// C(n − k, m)/C(n, m): the chance that a subsample of m of n misses all k marked individuals.
inline double missed(std::int64_t n, std::int64_t k, std::int64_t m) {
  if (m == 0 || k == 0) return 1;
  if (k > n - m) return 0;
  if (k <= 64) {
    double p = 1;
    for (std::int64_t j = 0; j < k; j++) p *= static_cast<double>(n - m - j) / static_cast<double>(n - j);
    return p;
  }
  using calx::special_functions::log_gamma;
  const double nd = static_cast<double>(n), kd = static_cast<double>(k), md = static_cast<double>(m);
  return std::exp(log_gamma(nd - kd + 1) - log_gamma(nd - kd - md + 1) - log_gamma(nd + 1) + log_gamma(nd - md + 1));
}

// visit(k, P(K = k)) for K the number of a class's X individuals among m drawn without replacement
// from n, over every k whose probability is above 10⁻¹⁸ of the mode's, normalised to sum to one.
template <class Visit>
void hypergeometric(std::int64_t n, std::int64_t X, std::int64_t m, Visit&& visit) {
  const std::int64_t lo = std::max<std::int64_t>(0, m - (n - X)), hi = std::min(X, m);
  if (lo == hi) { visit(lo, 1.0); return; }
  std::int64_t mode = static_cast<std::int64_t>(std::floor(static_cast<double>(m + 1) * static_cast<double>(X + 1) / static_cast<double>(n + 2)));
  mode = std::clamp(mode, lo, hi);
  std::vector<double> up{1.0}, down;  // up[j] = P(mode + j)/P(mode), down[j] = P(mode − 1 − j)/P(mode)
  const double nd = static_cast<double>(n), Xd = static_cast<double>(X), md = static_cast<double>(m);
  double t = 1;
  for (std::int64_t k = mode; k < hi; k++) {
    const double kd = static_cast<double>(k);
    t *= (Xd - kd) * (md - kd) / ((kd + 1) * (nd - Xd - md + kd + 1));
    if (t < 1e-18) break;
    up.push_back(t);
  }
  t = 1;
  for (std::int64_t k = mode; k > lo; k--) {
    const double kd = static_cast<double>(k);
    t *= kd * (nd - Xd - md + kd) / ((Xd - kd + 1) * (md - kd + 1));
    if (t < 1e-18) break;
    down.push_back(t);
  }
  double s = 0;
  for (std::size_t j = down.size(); j-- > 0;) s += down[j];
  for (double v : up) s += v;
  for (std::size_t j = down.size(); j-- > 0;) visit(mode - 1 - static_cast<std::int64_t>(j), down[j] / s);
  for (std::size_t j = 0; j < up.size(); j++) visit(mode + static_cast<std::int64_t>(j), up[j] / s);
}

// What the evaluations at many sizes share.
struct Summary {
  std::int64_t n = 0;
  double sobs = 0, f1 = 0, f0 = 0, x = 0;
  double entropy_observed = 0, entropy = 0;  // Ĥ_obs, Ĥ (Chao, Wang and Jost)
  double concentration = 0;                  // Σ X(X − 1)/(n(n − 1)), 0 when n = 1
  explicit Summary(const Abundances& s)
      : n(s.n), sobs(static_cast<double>(s.observed())), f1(static_cast<double>(s.singletons())), f0(undetected(s)),
        x(detection_ratio(s)), entropy_observed(observed_entropy(s)), entropy(entropy_estimate(s)) {
    const double c = concentration_estimate(s);
    concentration = std::isfinite(c) ? c : 0.0;
  }
};

inline std::int64_t whole_size(double m, const char* where) {
  require(std::isfinite(m) && m >= 0 && m == std::floor(m), where, "a size below the sample's is a whole number");
  return static_cast<std::int64_t>(m);
}

inline double interpolate(const Abundances& s, double q, std::int64_t m) {
  const std::int64_t n = s.n;
  require(m >= 1 && m <= n, "interpolated", "1 ≤ m ≤ n");
  require(q >= 0 && std::isfinite(q), "interpolated", "a finite order q ≥ 0");
  const double md = static_cast<double>(m);
  if (q == 0) {
    double r = 0;
    for (std::size_t k = 1; k < s.f.size(); k++)
      if (s.f[k]) r += static_cast<double>(s.f[k]) * (1 - missed(n, static_cast<std::int64_t>(k), m));
    return r;
  }
  if (q == 2) {
    if (m == 1) return 1;
    const double c = concentration_estimate(s);
    return 1 / (1 / md + (md - 1) / md * c);
  }
  // From the expected frequency counts, class by class of equal count.
  double acc = 0;
  for (std::size_t X = 1; X < s.f.size(); X++) {
    if (!s.f[X]) continue;
    double a = 0;
    hypergeometric(n, static_cast<std::int64_t>(X), m, [&](std::int64_t k, double p) {
      if (k == 0) return;
      const double share = static_cast<double>(k) / md;
      a += p * (q == 1 ? -share * std::log(share) : std::pow(share, q));
    });
    acc += static_cast<double>(s.f[X]) * a;
  }
  if (q == 1) return std::exp(acc);
  return std::pow(acc, 1 / (1 - q));
}

inline double extrapolate(const Summary& u, int q, double m) {
  const double n = static_cast<double>(u.n);
  require(m >= n, "extrapolated", "m ≥ n");
  const double extra = m - n;
  switch (q) {
    case 0: return u.f0 > 0 ? u.sobs + u.f0 * (1 - std::pow(u.x, extra)) : u.sobs;
    case 1: return std::exp(n / m * u.entropy_observed + extra / m * u.entropy);
    case 2: return 1 / (1 / m + (m - 1) / m * u.concentration);
    default: require(false, "extrapolated", "extrapolation is for the orders 0, 1 and 2");
  }
  return 0;
}

inline double coverage_at(const Abundances& s, const Summary& u, double m) {
  const double n = static_cast<double>(s.n);
  require(m >= 0, "coverage_at", "m ≥ 0");
  if (m < n) {
    const std::int64_t mi = whole_size(m, "coverage_at");
    double miss = 0;
    for (std::size_t k = 1; k < s.f.size(); k++)
      if (s.f[k]) miss += static_cast<double>(s.f[k]) * static_cast<double>(k) / n * missed(s.n - 1, static_cast<std::int64_t>(k) - 1, mi);
    return 1 - miss;
  }
  return 1 - u.f1 / n * std::pow(u.x, m - n + 1);
}

inline Abundances from_counts(const std::vector<std::int64_t>& counts) {
  Abundances a;
  std::int64_t largest = 0;
  for (std::int64_t c : counts) {
    if (c <= 0) continue;
    a.counts.push_back(c);
    a.n += c;
    largest = std::max(largest, c);
  }
  a.f.assign(static_cast<std::size_t>(largest) + 1, 0);
  for (std::int64_t c : a.counts) a.f[static_cast<std::size_t>(c)]++;
  return a;
}

}  // namespace detail

// f̂_k(m) for k = 0, …, m (f̂_0(m) the observed classes a subsample of m misses), m ≤ n.
inline std::vector<double> expected_frequencies(const Abundances& s, std::int64_t m) {
  require(m >= 0 && m <= s.n, "expected_frequencies", "0 ≤ m ≤ n");
  std::vector<double> f(static_cast<std::size_t>(m) + 1, 0.0);
  for (std::size_t X = 1; X < s.f.size(); X++) {
    if (!s.f[X]) continue;
    const double w = static_cast<double>(s.f[X]);
    detail::hypergeometric(s.n, static_cast<std::int64_t>(X), m, [&](std::int64_t k, double p) { f[static_cast<std::size_t>(k)] += w * p; });
  }
  return f;
}

// The rarefied Hill number of order q (finite, ≥ 0) at a whole size 1 ≤ m ≤ n.
inline double interpolated(const Abundances& s, double q, std::int64_t m) { return detail::interpolate(s, q, m); }

// The extrapolated Hill number of order q ∈ {0, 1, 2} at a size m ≥ n (real).
inline double extrapolated(const Abundances& s, int q, double m) { return detail::extrapolate(detail::Summary(s), q, m); }

// The Hill number of order q expected at size m: interpolated for m ≤ n (a whole number), extrapolated
// above (q ∈ {0, 1, 2}).
inline double estimate_at(const Abundances& s, double q, double m) {
  if (m <= static_cast<double>(s.n)) return detail::interpolate(s, q, detail::whole_size(m, "estimate_at"));
  require(q == 0 || q == 1 || q == 2, "estimate_at", "extrapolation is for the orders 0, 1 and 2");
  return detail::extrapolate(detail::Summary(s), static_cast<int>(q), m);
}

// The coverage expected at size m ≥ 0 (a whole number below n).
inline double coverage_at(const Abundances& s, double m) { return detail::coverage_at(s, detail::Summary(s), m); }

// The size at which the expected coverage reaches C ∈ [0, 1]: the least whole m ≤ n with Ĉ_m ≥ C, or
// the real m > n of the extrapolated coverage; +∞ when C is beyond every size's.
inline double size_at_coverage(const Abundances& s, double C) {
  require(C >= 0 && C <= 1, "size_at_coverage", "a coverage in [0, 1]");
  const detail::Summary u(s);
  if (C <= 0) return 0;
  const double cn = detail::coverage_at(s, u, static_cast<double>(s.n));
  if (C <= cn) {
    std::int64_t lo = 0, hi = s.n;  // Ĉ_lo < C ≤ Ĉ_hi
    while (hi - lo > 1) {
      const std::int64_t mid = lo + (hi - lo) / 2;
      if (detail::coverage_at(s, u, static_cast<double>(mid)) >= C) hi = mid;
      else lo = mid;
    }
    return static_cast<double>(hi);
  }
  if (!(u.x > 0) || !(u.x < 1) || C >= 1) return std::numeric_limits<double>::infinity();
  const double n = static_cast<double>(s.n);
  return n + std::log((1 - C) * n / u.f1) / std::log(u.x) - 1;
}

// The bootstrap assemblage: a law on the S_obs observed classes (in the sample's order) followed by
// ⌈f̂₀⌉ undetected ones.
inline std::vector<double> bootstrap_assemblage(const Abundances& s) {
  const double n = static_cast<double>(s.n);
  const double deficit = 1 - coverage(s), f0 = std::ceil(undetected(s) - 1e-12);
  double denom = 0;
  for (std::int64_t X : s.counts) {
    const double p = static_cast<double>(X) / n;
    denom += p * std::pow(1 - p, n);
  }
  const double lambda = deficit > 0 && denom > 0 ? deficit / denom : 0.0;
  std::vector<double> law;
  law.reserve(s.observed() + static_cast<std::size_t>(f0));
  for (std::int64_t X : s.counts) {
    const double p = static_cast<double>(X) / n;
    law.push_back(p * (1 - lambda * std::pow(1 - p, n)));
  }
  if (f0 >= 1 && deficit > 0)
    for (double j = 0; j < f0; j++) law.push_back(deficit / f0);
  return law;
}

enum class Method { interpolated, observed, extrapolated };

struct CurvePoint {
  double size = 0;  // m
  Method method = Method::observed;
  double order = 0;  // q
  double estimate = 0, se = 0, lower = 0, upper = 0;
  double coverage = 0, coverage_se = 0, coverage_lower = 0, coverage_upper = 0;
};

struct CurveOptions {
  std::vector<double> orders{0, 1, 2};  // any finite q ≥ 0 below n; 0, 1, 2 above it
  std::vector<double> sizes;            // whole sizes ≥ 1; empty: default_sizes(n, endpoint, knots)
  double endpoint = 0;                  // 0: 2n
  std::size_t knots = 40;
  std::size_t replicates = 50;  // bootstrap replicates; 0 for no intervals
  double level = 0.95;
  std::uint64_t seed = 20261112;
};

// About `knots` whole sizes from 1 to `endpoint`, half below n and half above it, n always among them
// (after iNEXT's choice).
inline std::vector<double> default_sizes(std::int64_t n, double endpoint, std::size_t knots) {
  require(n >= 1, "default_sizes", "n ≥ 1");
  const double nd = static_cast<double>(n), end = endpoint > 0 ? endpoint : 2 * nd;
  const std::size_t half = std::max<std::size_t>(knots / 2, 1);
  std::vector<double> m;
  for (std::size_t i = 0; i < half; i++) {
    const double v = half == 1 ? 1 : 1 + (nd - 2) * static_cast<double>(i) / static_cast<double>(half - 1);
    if (v >= 1 && v < nd) m.push_back(std::round(v));
  }
  m.push_back(nd);
  if (end > nd)
    for (std::size_t i = 0; i < half; i++) {
      const double v = half == 1 ? end : nd + 1 + (end - nd - 1) * static_cast<double>(i) / static_cast<double>(half - 1);
      m.push_back(std::round(v));
    }
  std::sort(m.begin(), m.end());
  m.erase(std::unique(m.begin(), m.end()), m.end());
  return m;
}

// The rarefaction–extrapolation curves of the given orders, with coverage, and bootstrap intervals.
// Points are ordered by order, then by size.
inline std::vector<CurvePoint> curve(const Abundances& s, const CurveOptions& o = {}) {
  const double nd = static_cast<double>(s.n);
  auto sizes = o.sizes.empty() ? default_sizes(s.n, o.endpoint, o.knots) : o.sizes;
  for (double m : sizes) {
    require(std::isfinite(m) && m >= 1 && m == std::floor(m), "curve", "sizes are whole numbers ≥ 1");
    if (m > nd)
      for (double q : o.orders) require(q == 0 || q == 1 || q == 2, "curve", "extrapolation is for the orders 0, 1 and 2");
  }
  for (double q : o.orders) require(q >= 0 && std::isfinite(q), "curve", "finite orders q ≥ 0");
  std::sort(sizes.begin(), sizes.end());

  const std::size_t Q = o.orders.size(), M = sizes.size();
  // Estimates of a sample at every (order, size), and coverage at every size.
  auto evaluate = [&](const Abundances& a, std::vector<double>& est, std::vector<double>& cov) {
    const detail::Summary u(a);
    est.assign(Q * M, 0.0);
    cov.assign(M, 0.0);
    for (std::size_t j = 0; j < M; j++) {
      const double m = sizes[j];
      cov[j] = detail::coverage_at(a, u, m);
      for (std::size_t i = 0; i < Q; i++)
        est[i * M + j] = m <= nd ? detail::interpolate(a, o.orders[i], static_cast<std::int64_t>(m)) : detail::extrapolate(u, static_cast<int>(o.orders[i]), m);
    }
  };
  std::vector<double> est, cov;
  evaluate(s, est, cov);

  std::vector<double> se(Q * M, 0.0), cse(M, 0.0);
  if (o.replicates >= 2) {
    const auto law = bootstrap_assemblage(s);
    std::vector<double> sum(Q * M, 0.0), sum2(Q * M, 0.0), csum(M, 0.0), csum2(M, 0.0), e, c;
    for (std::size_t r = 0; r < o.replicates; r++) {
      calx::random::Stream rng(o.seed, 0x636c6173u, static_cast<std::uint32_t>(r), 0);
      const auto draw = calx::random::multinomial(rng, s.n, law);
      evaluate(detail::from_counts(draw), e, c);
      for (std::size_t k = 0; k < Q * M; k++) sum[k] += e[k], sum2[k] += e[k] * e[k];
      for (std::size_t k = 0; k < M; k++) csum[k] += c[k], csum2[k] += c[k] * c[k];
    }
    const double B = static_cast<double>(o.replicates);
    auto sd = [&](double a, double b) { return std::sqrt(std::max(0.0, (b - a * a / B) / (B - 1))); };
    for (std::size_t k = 0; k < Q * M; k++) se[k] = sd(sum[k], sum2[k]);
    for (std::size_t k = 0; k < M; k++) cse[k] = sd(csum[k], csum2[k]);
  }
  const double z = detail::normal_quantile_two_sided(o.level);
  std::vector<CurvePoint> out;
  out.reserve(Q * M);
  for (std::size_t i = 0; i < Q; i++)
    for (std::size_t j = 0; j < M; j++) {
      CurvePoint p;
      p.size = sizes[j];
      p.method = sizes[j] < nd ? Method::interpolated : sizes[j] > nd ? Method::extrapolated : Method::observed;
      p.order = o.orders[i];
      p.estimate = est[i * M + j];
      p.se = se[i * M + j];
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
inline std::vector<CurvePoint> curve(std::span<const double> counts, const CurveOptions& o = {}) { return curve(abundances(counts), o); }

}  // namespace tyche::classes
