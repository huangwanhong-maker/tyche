// Finite mixtures of binomials with known scales, fitted by expectation–maximisation, the number of
// components chosen by an information criterion.
//
// THE MODEL. Observation i is a count kᵢ of nᵢ trials whose probability of success is cᵢ θ_z, with a
// known scale cᵢ > 0 and the parameter θ_z of an unobserved component z ∈ {1, …, K} drawn with
// probability π_z:
//     P(kᵢ) = Σ_j π_j C(nᵢ, kᵢ) (cᵢ θ_j)^{kᵢ} (1 − cᵢ θ_j)^{nᵢ − kᵢ},   0 < θ_j ≤ θ_max,  cᵢ θ_max < 1.
// With every cᵢ = 1 it is the plain binomial mixture; the scales let each observation see the
// component's parameter through its own thinning (a detection probability, the fraction of a
// sample's molecules a parameter stands for), which is how cancer cell fractions are clustered
// from read counts (PyClone, Roth et al. 2014, Nature Methods 11, 396–398, without its copy-number
// states; DPClust's binomial mixture, Nik-Zainal et al. 2012, Cell 149, 994–1007).
//
// THE ALGORITHM (Dempster, Laird and Rubin 1977, JRSS B 39, 1–38; McLachlan and Peel 2000, Finite
// Mixture Models, §2.8). E-step: responsibilities r_ij ∝ π_j Bin(kᵢ; nᵢ, cᵢθ_j), normalised in log
// space. M-step: π_j = Σᵢ r_ij / N, and θ_j the maximiser on (0, θ_max] of
//     g_j(θ) = Σᵢ r_ij [kᵢ log(cᵢ θ) + (nᵢ − kᵢ) log(1 − cᵢ θ)],
// which is strictly concave (each term is, where kᵢ > 0 or nᵢ > kᵢ), with g′(θ) = Σᵢ r_ij [kᵢ/θ −
// (nᵢ − kᵢ) cᵢ/(1 − cᵢθ)] decreasing from +∞: its root by calx's Brent when it lies below the bound,
// the bound otherwise; with equal scales c, θ_j = Σ r k / (c Σ r n) exactly. Each iteration does
// not decrease the log-likelihood (the EM inequality; checked), and the iteration stops when it
// rises by less than tolerance · (1 + |ℓ|). Starts: θ at the quantiles (j − ½)/K of the
// observations' own estimates kᵢ/(nᵢcᵢ) (held to the bound), equal weights; then `restarts` starts
// at K of the observations' estimates drawn without replacement from a calx stream keyed by the
// seed and K; the fit of greatest likelihood is kept, so the result is a function of the data and
// the seed.
//
// THE CHOICE OF K. BIC = −2ℓ + (2K − 1) log N and AIC = −2ℓ + 2(2K − 1), the K parameters θ and the
// K − 1 free weights counted (the glossary's convention: the parameters estimated); the binomial
// coefficients are included in ℓ. select_binomial_mixture fits K = 1, …, K_max and returns the least
// BIC. Components whose weight falls below 1/(10N) are kept (they cost their parameters in BIC).
//
// ACCURACY AND LIMITS. EM finds a local maximum; the restarts make a global one likely, not
// certain. The likelihood of a mixture is invariant under permuting the components, so components
// are returned by decreasing θ. BIC's consistency for the order of a mixture holds under
// regularity conditions that mixtures violate at the boundary (Keribin 2000, Sankhyā A 62, 49–66,
// proves it for bounded parameters); a component of tiny weight or two of nearly equal θ is the
// usual failure. A Dirichlet-process mixture (PyClone-VI, Gillis and Roth 2020, BMC Bioinformatics
// 21, 571) is the Bayesian alternative; not written.

#pragma once

#include <calx/random/stream.hpp>
#include <calx/solvers/roots.hpp>
#include <calx/special_functions/gamma.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::multivariate {

struct BinomialMixtureOptions {
  std::size_t components = 1;
  double theta_max = 1;          // the upper bound of every θ (with cᵢ θ_max < 1 for every i)
  std::size_t max_iterations = 1000;
  double tolerance = 1e-10;
  std::size_t restarts = 10;
  std::uint64_t seed = 1;
};

struct BinomialMixtureFit {
  std::vector<double> weights, theta;               // by decreasing θ
  std::vector<std::vector<double>> responsibility;  // N × K
  std::vector<std::size_t> assignment;              // the component of greatest responsibility
  double log_likelihood = -std::numeric_limits<double>::infinity();
  double bic = std::numeric_limits<double>::infinity(), aic = std::numeric_limits<double>::infinity();
  std::size_t iterations = 0;
  bool converged = false;
  bool monotone = true;                             // ℓ never decreased by more than rounding
};

namespace mixture_detail {

inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::multivariate::binomial_mixture: ") + what);
}

inline double log_binomial(double k, double n, double p) {
  const double lc = calx::special_functions::log_choose(n, k);
  const double a = k > 0 ? k * std::log(p) : 0, b = n - k > 0 ? (n - k) * std::log1p(-p) : 0;
  return lc + a + b;
}

struct Data {
  std::span<const double> k, n, c;
  double upper = 1;  // the effective bound of θ
};

// One EM run from the given start.
inline BinomialMixtureFit run(const Data& d, std::vector<double> theta, std::vector<double> weight, const BinomialMixtureOptions& o) {
  const std::size_t N = d.k.size(), K = theta.size();
  BinomialMixtureFit f;
  std::vector<std::vector<double>> r(N, std::vector<double>(K));
  double previous = -std::numeric_limits<double>::infinity();
  const auto e_step = [&]() {
    double ll = 0;
    for (std::size_t i = 0; i < N; i++) {
      double top = -std::numeric_limits<double>::infinity();
      for (std::size_t j = 0; j < K; j++) {
        r[i][j] = weight[j] > 0 ? std::log(weight[j]) + log_binomial(d.k[i], d.n[i], d.c[i] * theta[j]) : -std::numeric_limits<double>::infinity();
        top = std::max(top, r[i][j]);
      }
      double s = 0;
      for (std::size_t j = 0; j < K; j++) s += std::exp(r[i][j] - top);
      ll += top + std::log(s);
      for (std::size_t j = 0; j < K; j++) r[i][j] = std::exp(r[i][j] - top) / s;
    }
    return ll;
  };
  double ll = e_step();
  for (std::size_t it = 1; it <= o.max_iterations; it++) {
    // M-step.
    for (std::size_t j = 0; j < K; j++) {
      double w = 0;
      for (std::size_t i = 0; i < N; i++) w += r[i][j];
      weight[j] = w / static_cast<double>(N);
      if (w <= 0) continue;
      const auto score = [&](double t) {
        double g = 0;
        for (std::size_t i = 0; i < N; i++) {
          if (r[i][j] == 0) continue;
          g += r[i][j] * (d.k[i] / t - (d.n[i] - d.k[i]) * d.c[i] / (1 - d.c[i] * t));
        }
        return g;
      };
      double sk = 0;
      for (std::size_t i = 0; i < N; i++) sk += r[i][j] * d.k[i];
      if (sk <= 0) { theta[j] = d.upper * 1e-6; continue; }
      if (score(d.upper) >= 0) { theta[j] = d.upper; continue; }
      double lo = d.upper * 1e-9;
      while (score(lo) <= 0 && lo > 1e-300) lo *= 1e-3;
      theta[j] = calx::solvers::brent(score, lo, d.upper, 1e-15).x;
    }
    const double next = e_step();
    if (next < ll - 1e-9 * (1 + std::fabs(ll))) f.monotone = false;
    previous = ll;
    ll = next;
    f.iterations = it;
    if (std::fabs(ll - previous) <= o.tolerance * (1 + std::fabs(ll))) { f.converged = true; break; }
  }
  // Components by decreasing θ.
  std::vector<std::size_t> order(K);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return theta[a] > theta[b]; });
  f.theta.resize(K);
  f.weights.resize(K);
  for (std::size_t j = 0; j < K; j++) {
    f.theta[j] = theta[order[j]];
    f.weights[j] = weight[order[j]];
  }
  f.responsibility.assign(N, std::vector<double>(K));
  f.assignment.assign(N, 0);
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < K; j++) f.responsibility[i][j] = r[i][order[j]];
    f.assignment[i] = static_cast<std::size_t>(std::max_element(f.responsibility[i].begin(), f.responsibility[i].end()) - f.responsibility[i].begin());
  }
  f.log_likelihood = ll;
  const double p = static_cast<double>(2 * K - 1);
  f.bic = -2 * ll + p * std::log(static_cast<double>(N));
  f.aic = -2 * ll + 2 * p;
  return f;
}

}  // namespace mixture_detail

// The mixture of `options.components` components fitted to (kᵢ, nᵢ, cᵢ).
inline BinomialMixtureFit fit_binomial_mixture(std::span<const double> k, std::span<const double> n, std::span<const double> c, const BinomialMixtureOptions& o = {}) {
  using mixture_detail::require;
  const std::size_t N = k.size(), K = o.components;
  require(N > 0 && n.size() == N && c.size() == N, "one count, one number of trials and one scale per observation");
  require(K >= 1, "at least one component");
  require(o.theta_max > 0, "a positive bound for θ");
  double cmax = 0;
  for (std::size_t i = 0; i < N; i++) {
    require(n[i] >= 0 && k[i] >= 0 && k[i] <= n[i] && c[i] > 0 && std::isfinite(c[i]), "0 ≤ k ≤ n and c > 0");
    cmax = std::max(cmax, c[i]);
  }
  mixture_detail::Data d{k, n, c, std::min(o.theta_max, (1 - 1e-12) / cmax)};
  // The observations' own estimates.
  std::vector<double> own(N);
  for (std::size_t i = 0; i < N; i++) own[i] = std::clamp(n[i] > 0 ? k[i] / (n[i] * c[i]) : 0.0, d.upper * 1e-6, d.upper);
  std::vector<double> sorted = own;
  std::sort(sorted.begin(), sorted.end());
  std::vector<double> theta(K), weight(K, 1.0 / static_cast<double>(K));
  for (std::size_t j = 0; j < K; j++) {
    const double q = (static_cast<double>(j) + 0.5) / static_cast<double>(K) * static_cast<double>(N - 1);
    theta[j] = sorted[static_cast<std::size_t>(std::lround(q))];
  }
  BinomialMixtureFit best = mixture_detail::run(d, theta, weight, o);
  calx::random::Stream r(o.seed, 0x424d4958u /* "BMIX" */, static_cast<std::uint32_t>(K), 0);
  for (std::size_t s = 0; s < o.restarts && K > 1; s++) {
    std::vector<std::size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    for (std::size_t j = 0; j < std::min(K, N); j++) std::swap(idx[j], idx[j + r.below(N - j)]);
    for (std::size_t j = 0; j < K; j++) theta[j] = own[idx[j % N]];
    std::fill(weight.begin(), weight.end(), 1.0 / static_cast<double>(K));
    BinomialMixtureFit f = mixture_detail::run(d, theta, weight, o);
    if (f.log_likelihood > best.log_likelihood + 1e-9 * (1 + std::fabs(best.log_likelihood))) best = std::move(f);
  }
  return best;
}

struct BinomialMixtureSelection {
  std::vector<BinomialMixtureFit> fits;  // K = 1, …, K_max
  std::size_t best = 0;                  // the index of the least BIC
};

inline BinomialMixtureSelection select_binomial_mixture(std::span<const double> k, std::span<const double> n, std::span<const double> c, std::size_t max_components,
                                                        BinomialMixtureOptions o = {}) {
  mixture_detail::require(max_components >= 1, "at least one component");
  BinomialMixtureSelection s;
  for (std::size_t K = 1; K <= std::min(max_components, k.size()); K++) {
    o.components = K;
    s.fits.push_back(fit_binomial_mixture(k, n, c, o));
    if (s.fits.back().bic < s.fits[s.best].bic) s.best = s.fits.size() - 1;
  }
  return s;
}

}  // namespace tyche::multivariate
