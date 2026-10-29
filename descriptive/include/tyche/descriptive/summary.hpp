// Descriptive statistics of samples: moments, quantiles, weighted moments, covariance and
// correlation matrices, and a summary.
//
// Accuracy. Sums are compensated (Neumaier 1974, ZAMM 54, 39–51: error bounded by about 2ε|Σx|
// independently of n), the mean is corrected by a second pass (x̄ + Σ(x − x̄)/n), and the sum of
// squared deviations is the corrected two-pass formula Σ(x − x̄)² − (Σ(x − x̄))²/n (Chan, Golub and
// LeVeque 1983, "Algorithms for computing the sample variance", Amer. Statist. 37, 242–247,
// eq. 1.7), whose relative error is about nε + n²ε²κ², κ the condition number √(1 + x̄²/s²) of
// the data. On NIST's NumAcc4 (1001 values 10⁷ + 0.1, 0.2, 0.3; κ ≈ 10⁸) the standard deviation
// keeps the 8 digits the input's decimal rounding leaves (checks); the one-pass textbook formula
// Σx² − n x̄² keeps none.
//
// Definitions.
// - variance(x, ddof): Σ(x − x̄)² / (n − ddof); ddof = 1 is the unbiased sample variance, 0 the
//   moment.
// - skewness: g₁ = m₃ / m₂^{3/2}, or with bias_corrected G₁ = g₁ √(n(n − 1)) / (n − 2); kurtosis:
//   g₂ = m₄ / m₂² − 3 (excess), or G₂ = ((n + 1) g₂ + 6)(n − 1) / ((n − 2)(n − 3)), mᵣ the r-th
//   central moment with divisor n (Joanes and Gill 1998, "Comparing measures of sample skewness and
//   kurtosis", The Statistician 47, 183–189: g₁, G₁ are SAS's and Excel's, b₁ = g₁((n − 1)/n)^{3/2}
//   is Minitab's).
// - quantile(x, p, type): the nine sample quantiles of Hyndman and Fan (1996), "Sample quantiles in
//   statistical packages", Amer. Statist. 50, 361–365: types 1–3 discontinuous (1 the inverse of
//   the empirical distribution function, 2 averaging at its jumps, 3 the nearest even order
//   statistic), types 4–9 piecewise linear through (p_k, x_(k)) with p_k = (k − α)/(n + 1 − α − β)
//   for (α, β) = (0, 1), (½, ½), (0, 0), (1, 1), (⅓, ⅓), (⅜, ⅜). Type 7 (R's and NumPy's default)
//   is the default; Hyndman and Fan recommend type 8. Computed as R's quantile() computes them,
//   with its fuzz of 4ε in deciding which order statistic p falls on.
// - weighted_variance: frequency weights (an observation of weight w stands for w equal ones)
//   divide by Σw − ddof; reliability weights (proportional to precision) by V₁ − V₂/V₁ with
//   V₁ = Σw, V₂ = Σw², which is unbiased for independent observations (Gentleman's formula; GSL's
//   gsl_stats_wvariance).
// - weighted_quantile: the lower inverse of the weighted empirical distribution function, the least
//   x_(k) whose cumulative weight reaches p Σw (type 1 when the weights are equal).
// - covariance_matrix(data), correlation_matrix(data): data has one row per observation and one
//   column per variable; divisor n − ddof; correlations have an exact unit diagonal.

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::descriptive {

using calx::linalg::Matrix;
using calx::linalg::Rectangular;

namespace detail {

inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::descriptive: ") + what);
}

// Neumaier's compensated sum.
template <class F>
inline double compensated_sum(std::size_t n, F&& term) {
  double s = 0, c = 0;
  for (std::size_t i = 0; i < n; i++) {
    const double v = term(i);
    const double t = s + v;
    c += std::fabs(s) >= std::fabs(v) ? (s - t) + v : (v - t) + s;
    s = t;
  }
  return s + c;
}

// Σ(x − m)² − (Σ(x − m))²/n, with m the corrected mean (Chan, Golub and LeVeque, eq. 1.7).
struct Centred {
  double mean = 0, ss = 0;
};
inline Centred centred(std::span<const double> x) {
  const std::size_t n = x.size();
  require(n > 0, "a non-empty sample");
  double m = compensated_sum(n, [&](std::size_t i) { return x[i]; }) / static_cast<double>(n);
  const double d = compensated_sum(n, [&](std::size_t i) { return x[i] - m; });
  m += d / static_cast<double>(n);
  const double s1 = compensated_sum(n, [&](std::size_t i) { return x[i] - m; });
  const double s2 = compensated_sum(n, [&](std::size_t i) { return (x[i] - m) * (x[i] - m); });
  return {m, std::max(0.0, s2 - s1 * s1 / static_cast<double>(n))};
}

}  // namespace detail

inline double sum(std::span<const double> x) { return detail::compensated_sum(x.size(), [&](std::size_t i) { return x[i]; }); }

inline double mean(std::span<const double> x) { return detail::centred(x).mean; }

inline double variance(std::span<const double> x, double ddof = 1) {
  detail::require(static_cast<double>(x.size()) > ddof, "more observations than ddof");
  return detail::centred(x).ss / (static_cast<double>(x.size()) - ddof);
}

inline double standard_deviation(std::span<const double> x, double ddof = 1) { return std::sqrt(variance(x, ddof)); }

// The central moment of order r with divisor n.
inline double central_moment(std::span<const double> x, int r) {
  const double m = mean(x);
  return detail::compensated_sum(x.size(), [&](std::size_t i) { return std::pow(x[i] - m, r); }) / static_cast<double>(x.size());
}

inline double skewness(std::span<const double> x, bool bias_corrected = false) {
  const double n = static_cast<double>(x.size());
  detail::require(n >= (bias_corrected ? 3 : 1), "skewness needs n ≥ 3 when bias-corrected");
  const double m2 = detail::centred(x).ss / n, m3 = central_moment(x, 3);
  const double g1 = m3 / std::pow(m2, 1.5);
  return bias_corrected ? g1 * std::sqrt(n * (n - 1)) / (n - 2) : g1;
}

// Excess kurtosis g₂ (or G₂); excess = false adds 3 back.
inline double kurtosis(std::span<const double> x, bool excess = true, bool bias_corrected = false) {
  const double n = static_cast<double>(x.size());
  detail::require(n >= (bias_corrected ? 4 : 1), "kurtosis needs n ≥ 4 when bias-corrected");
  const double m2 = detail::centred(x).ss / n, m4 = central_moment(x, 4);
  double k = m4 / (m2 * m2) - 3;
  if (bias_corrected) k = ((n + 1) * k + 6) * (n - 1) / ((n - 2) * (n - 3));
  return excess ? k : k + 3;
}

// The Hyndman–Fan quantile of type 1–9 of an already sorted sample.
inline double quantile_sorted(std::span<const double> s, double p, int type = 7) {
  const std::size_t n = s.size();
  detail::require(n > 0, "a non-empty sample");
  detail::require(p >= 0 && p <= 1, "a probability in [0, 1]");
  detail::require(type >= 1 && type <= 9, "a quantile type from 1 to 9");
  const double dn = static_cast<double>(n), fuzz = 4 * std::numeric_limits<double>::epsilon();
  double nppm, h;
  long long j;
  if (type <= 3) {
    nppm = type == 3 ? dn * p - 0.5 : dn * p;
    j = static_cast<long long>(std::floor(nppm + fuzz));
    const bool past = nppm > static_cast<double>(j);
    if (type == 1) h = past ? 1 : 0;                  // the next order statistic once past x_(j)
    else if (type == 2) h = past ? 1 : 0.5;           // the average of the two at a jump
    else h = (nppm != static_cast<double>(j) || j % 2 != 0) ? 1 : 0;  // the nearest even order statistic
  } else {
    static constexpr double alpha[10] = {0, 0, 0, 0, 0, 0.5, 0, 1, 1.0 / 3, 3.0 / 8};
    const double a = alpha[type], b = a == 0 && type == 4 ? 1.0 : a;  // type 4: (α, β) = (0, 1)
    nppm = a + p * (dn + 1 - a - b);
    j = static_cast<long long>(std::floor(nppm + fuzz));
    h = nppm - static_cast<double>(j);
    if (std::fabs(h) < fuzz) h = 0;
  }
  // x_j with x_0 = x_{−1} = x_1 and x_{n+1} = x_{n+2} = x_n (1-based), as R pads.
  auto at = [&](long long k) { return s[static_cast<std::size_t>(std::clamp<long long>(k, 1, static_cast<long long>(n)) - 1)]; };
  const double lo = at(j), hi = at(j + 1);
  if (h == 0) return lo;
  if (h == 1) return hi;
  return lo == hi ? lo : (1 - h) * lo + h * hi;
}

inline double quantile(std::span<const double> x, double p, int type = 7) {
  std::vector<double> s(x.begin(), x.end());
  std::sort(s.begin(), s.end());
  return quantile_sorted(s, p, type);
}

inline std::vector<double> quantiles(std::span<const double> x, std::span<const double> ps, int type = 7) {
  std::vector<double> s(x.begin(), x.end());
  std::sort(s.begin(), s.end());
  std::vector<double> out;
  out.reserve(ps.size());
  for (double p : ps) out.push_back(quantile_sorted(s, p, type));
  return out;
}

inline double median(std::span<const double> x) { return quantile(x, 0.5, 7); }

inline double interquartile_range(std::span<const double> x, int type = 7) {
  const double ps[2] = {0.25, 0.75};
  const auto q = quantiles(x, ps, type);
  return q[1] - q[0];
}

// ---- weighted ----
enum class Weights { frequency, reliability };

inline double weighted_mean(std::span<const double> x, std::span<const double> w) {
  detail::require(x.size() == w.size() && !x.empty(), "one weight per observation");
  const double sw = detail::compensated_sum(w.size(), [&](std::size_t i) {
    detail::require(w[i] >= 0, "non-negative weights");
    return w[i];
  });
  detail::require(sw > 0, "a positive total weight");
  double m = detail::compensated_sum(x.size(), [&](std::size_t i) { return w[i] * x[i]; }) / sw;
  m += detail::compensated_sum(x.size(), [&](std::size_t i) { return w[i] * (x[i] - m); }) / sw;
  return m;
}

inline double weighted_variance(std::span<const double> x, std::span<const double> w, Weights kind = Weights::frequency, double ddof = 1) {
  const double m = weighted_mean(x, w);
  const double v1 = detail::compensated_sum(w.size(), [&](std::size_t i) { return w[i]; });
  const double s1 = detail::compensated_sum(x.size(), [&](std::size_t i) { return w[i] * (x[i] - m); });
  const double ss = detail::compensated_sum(x.size(), [&](std::size_t i) { return w[i] * (x[i] - m) * (x[i] - m); }) - s1 * s1 / v1;
  if (kind == Weights::frequency) {
    detail::require(v1 > ddof, "a total weight above ddof");
    return ss / (v1 - ddof);
  }
  const double v2 = detail::compensated_sum(w.size(), [&](std::size_t i) { return w[i] * w[i]; });
  detail::require(v1 * v1 > v2, "at least two observations of positive weight");
  return ss / (v1 - v2 / v1);
}

inline double weighted_standard_deviation(std::span<const double> x, std::span<const double> w, Weights kind = Weights::frequency, double ddof = 1) {
  return std::sqrt(weighted_variance(x, w, kind, ddof));
}

inline double weighted_quantile(std::span<const double> x, std::span<const double> w, double p) {
  detail::require(x.size() == w.size() && !x.empty(), "one weight per observation");
  detail::require(p >= 0 && p <= 1, "a probability in [0, 1]");
  std::vector<std::size_t> order(x.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  const double total = detail::compensated_sum(w.size(), [&](std::size_t i) { return w[i]; });
  const double target = p * total * (1 - 4 * std::numeric_limits<double>::epsilon());
  double cum = 0;
  for (std::size_t k : order) {
    cum += w[k];
    if (w[k] > 0 && cum >= target) return x[k];
  }
  return x[order.back()];
}

// ---- two variables, and matrices ----
inline double covariance(std::span<const double> x, std::span<const double> y, double ddof = 1) {
  detail::require(x.size() == y.size() && static_cast<double>(x.size()) > ddof, "two samples of one length, more than ddof");
  const double mx = mean(x), my = mean(y);
  const double n = static_cast<double>(x.size());
  const double sx = detail::compensated_sum(x.size(), [&](std::size_t i) { return x[i] - mx; });
  const double sy = detail::compensated_sum(y.size(), [&](std::size_t i) { return y[i] - my; });
  const double sxy = detail::compensated_sum(x.size(), [&](std::size_t i) { return (x[i] - mx) * (y[i] - my); });
  return (sxy - sx * sy / n) / (n - ddof);
}

inline double correlation(std::span<const double> x, std::span<const double> y) {
  return covariance(x, y, 1) / std::sqrt(variance(x, 1) * variance(y, 1));
}

inline Matrix covariance_matrix(const Rectangular& data, double ddof = 1) {
  const std::size_t n = data.rows, p = data.cols;
  detail::require(static_cast<double>(n) > ddof, "more observations than ddof");
  std::vector<std::vector<double>> cols(p);
  for (std::size_t j = 0; j < p; j++) cols[j] = data.column(j);
  Matrix C(p);
  for (std::size_t a = 0; a < p; a++)
    for (std::size_t b = 0; b <= a; b++) C(a, b) = C(b, a) = a == b ? variance(cols[a], ddof) : covariance(cols[a], cols[b], ddof);
  return C;
}

inline Matrix correlation_matrix(const Rectangular& data) {
  Matrix C = covariance_matrix(data, 1);
  const std::size_t p = C.n;
  std::vector<double> sd(p);
  for (std::size_t j = 0; j < p; j++) sd[j] = std::sqrt(C(j, j));
  for (std::size_t a = 0; a < p; a++)
    for (std::size_t b = 0; b < p; b++) C(a, b) = a == b ? 1.0 : C(a, b) / (sd[a] * sd[b]);
  return C;
}

// ---- a summary ----
struct Summary {
  std::size_t n = 0;
  double mean = 0, variance = 0, standard_deviation = 0;  // ddof = 1
  double min = 0, q1 = 0, median = 0, q3 = 0, max = 0;   // quartiles of type 7
  double skewness = 0, kurtosis = 0;                       // g₁ and excess g₂
};

inline Summary summarise(std::span<const double> x) {
  Summary s;
  s.n = x.size();
  detail::require(s.n >= 2, "a summary needs two observations");
  const auto c = detail::centred(x);
  s.mean = c.mean;
  s.variance = c.ss / static_cast<double>(s.n - 1);
  s.standard_deviation = std::sqrt(s.variance);
  std::vector<double> sorted(x.begin(), x.end());
  std::sort(sorted.begin(), sorted.end());
  s.min = sorted.front();
  s.max = sorted.back();
  s.q1 = quantile_sorted(sorted, 0.25);
  s.median = quantile_sorted(sorted, 0.5);
  s.q3 = quantile_sorted(sorted, 0.75);
  s.skewness = skewness(x);
  s.kurtosis = kurtosis(x);
  return s;
}

}  // namespace tyche::descriptive
