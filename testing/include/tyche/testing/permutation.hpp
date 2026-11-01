// Permutation tests and the beta approximation of the permutation law of a minimum p-value.
//
// - The permutation p-value (b + 1)/(B + 1), b of B random permutations at least as extreme as the
//   observation: exact (never anti-conservative) because the observation is itself one draw of the
//   permutation law, and never 0 (Phipson and Smyth 2010, Stat. Appl. Genet. Mol. Biol. 9, art. 39,
//   eq. 1; Davison and Hinkley 1997, Bootstrap Methods and their Application, §4.3).
// - A family of m correlated tests (the variants near a gene, say) is summarised by its least
//   p-value, whose null law depends on the correlation. Under m independent tests it is
//   1 − (1 − p)^m, the law of Beta(1, m); FastQTL (Ongen, Buil, Brown, Dermitzakis and Delaneau
//   2016, Bioinformatics 32, 1479–1485, §2.2) fits Beta(a, b) to the minima of a few thousand
//   permutations by maximum likelihood and reads the observed minimum's adjusted p-value from it,
//   F_Beta(p_min; a, b): b plays the effective number of independent tests and a absorbs the
//   departure from Beta(1, ·). This resolves adjusted p-values far below 1/B. fit_beta maximises
//   ℓ(a, b) = (a − 1) Σ log pᵢ + (b − 1) Σ log(1 − pᵢ) − N log B(a, b), which is concave in (a, b) (an
//   exponential family in its natural parameters), by Newton's method with the trigamma Hessian
//   from the method-of-moments start (a = μ(μ(1 − μ)/v − 1), b = a(1/μ − 1)), halving steps that
//   leave the positive quadrant or lose likelihood.
// - The effective degrees of freedom. The nominal p-values of permuted correlations are computed
//   with the t law on ν = n − 2 − k degrees of freedom (k covariates), which is exact for Gaussian
//   residuals; FastQTL and tensorQTL (Taylor-Weiner et al. 2019, Genome Biology 20, 228; its
//   fit_beta_parameters and df_cost) re-estimate ν as the value at which the moment estimate of a is
//   1, so that the minima of the permutations are Beta(1, b) on that scale. effective_df finds that
//   root by Brent's method; correlation_p_value(r², ν) is the two-sided p-value of t = √(ν r²/(1 − r²)).
//
// Limits: the beta approximation is an approximation (checked against direct permutations at the
// accuracy their number allows); minima must lie in (0, 1). Complexity: O(N) per Newton step.

#pragma once

#include <tyche/probability/distributions.hpp>

#include <calx/solvers/roots.hpp>
#include <calx/special_functions/gamma.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::testing {

// (b + 1)/(B + 1): `extreme` of `permutations` draws at least as extreme as the observation.
inline double permutation_p_value(std::size_t extreme, std::size_t permutations) {
  if (extreme > permutations) throw std::invalid_argument("tyche::testing::permutation_p_value: at most B extreme draws of B");
  return (static_cast<double>(extreme) + 1) / (static_cast<double>(permutations) + 1);
}

// The two-sided p-value of a squared correlation r² on ν degrees of freedom: t = √(ν r²/(1 − r²)).
inline double correlation_p_value(double r2, double df) {
  if (!(r2 >= 0 && r2 <= 1) || !(df > 0)) throw std::invalid_argument("tyche::testing::correlation_p_value: r² in [0, 1] and ν > 0");
  if (r2 >= 1) return 0;
  // P(|T| > t) = I_{ν/(ν + t²)}(ν/2, 1/2), with ν/(ν + t²) = 1 − r².
  return calx::special_functions::beta_i(df / 2, 0.5, 1 - r2);
}

struct BetaFit {
  double a = 1, b = 1;          // the shape parameters
  double log_likelihood = 0;
  double moment_a = 1, moment_b = 1;  // the method-of-moments start
  int iterations = 0;
  bool converged = false;
  double cdf(double p) const { return probability::Beta{a, b}.cdf(p); }
};

namespace detail {

inline double beta_log_likelihood(double a, double b, double slog, double slog1m, double N) {
  return (a - 1) * slog + (b - 1) * slog1m - N * calx::special_functions::log_beta(a, b);
}

}  // namespace detail

inline BetaFit fit_beta(std::span<const double> p, double tolerance = 1e-10, int max_iterations = 200) {
  const std::size_t N = p.size();
  if (N < 2) throw std::invalid_argument("tyche::testing::fit_beta: at least two values");
  double mean = 0, slog = 0, slog1m = 0;
  for (const double x : p) {
    if (!(x > 0 && x < 1)) throw std::invalid_argument("tyche::testing::fit_beta: values in (0, 1)");
    mean += x;
    slog += std::log(x);
    slog1m += std::log1p(-x);
  }
  const double dN = static_cast<double>(N);
  mean /= dN;
  double var = 0;
  for (const double x : p) var += (x - mean) * (x - mean);
  var /= dN;  // numpy's var, as tensorQTL's moment start
  BetaFit f;
  f.moment_a = var > 0 ? mean * (mean * (1 - mean) / var - 1) : 1;
  f.moment_b = f.moment_a * (1 / mean - 1);
  if (!(f.moment_a > 0 && f.moment_b > 0)) f.moment_a = f.moment_b = 1;
  double a = f.moment_a, b = f.moment_b;
  double ll = detail::beta_log_likelihood(a, b, slog, slog1m, dN);
  namespace sf = calx::special_functions;
  for (int it = 1; it <= max_iterations; it++) {
    f.iterations = it;
    const double pab = sf::digamma(a + b), tab = sf::trigamma(a + b);
    const double ga = slog - dN * (sf::digamma(a) - pab), gb = slog1m - dN * (sf::digamma(b) - pab);
    // −H = N [[ψ′(a) − ψ′(a+b), −ψ′(a+b)], [−ψ′(a+b), ψ′(b) − ψ′(a+b)]], positive definite.
    const double haa = dN * (sf::trigamma(a) - tab), hbb = dN * (sf::trigamma(b) - tab), hab = -dN * tab;
    const double det = haa * hbb - hab * hab;
    double da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det;
    double step = 1, nll = -INFINITY, na = a, nb = b;
    for (int h = 0; h < 60; h++) {
      na = a + step * da;
      nb = b + step * db;
      if (na > 0 && nb > 0) {
        nll = detail::beta_log_likelihood(na, nb, slog, slog1m, dN);
        if (nll >= ll - 1e-12 * std::fabs(ll)) break;
      }
      step /= 2;
    }
    if (!(na > 0 && nb > 0)) break;
    const bool small = std::fabs(na - a) <= tolerance * std::max(1.0, a) && std::fabs(nb - b) <= tolerance * std::max(1.0, b);
    a = na;
    b = nb;
    ll = std::max(ll, nll);
    if (small) {
      f.converged = true;
      break;
    }
  }
  f.a = a;
  f.b = b;
  f.log_likelihood = detail::beta_log_likelihood(a, b, slog, slog1m, dN);
  return f;
}

// The degrees of freedom ν at which the permutations' least p-values, correlation_p_value(r²ᵢ, ν),
// have moment estimate a = 1 (tensorQTL's df_cost); `start` is the nominal ν. Returns `start` when
// no root lies in [start/50, 50 start].
inline double effective_df(std::span<const double> r2, double start) {
  if (r2.size() < 2 || !(start > 0)) throw std::invalid_argument("tyche::testing::effective_df: at least two values and ν > 0");
  const auto cost = [&](double log_df) {
    const double df = std::exp(log_df);
    double mean = 0, m2 = 0;
    for (const double x : r2) mean += correlation_p_value(x, df);
    mean /= static_cast<double>(r2.size());
    for (const double x : r2) {
      const double d = correlation_p_value(x, df) - mean;
      m2 += d * d;
    }
    const double var = m2 / static_cast<double>(r2.size());
    return mean * (mean * (1 - mean) / var - 1) - 1;
  };
  const double lo = std::log(start / 50), hi = std::log(start * 50);
  const double clo = cost(lo), chi = cost(hi);
  if (!std::isfinite(clo) || !std::isfinite(chi) || (clo > 0) == (chi > 0)) return start;
  return std::exp(calx::solvers::brent(cost, lo, hi, 1e-10).x);
}

}  // namespace tyche::testing
