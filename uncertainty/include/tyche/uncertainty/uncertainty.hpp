// The uncertainty of a model's output: how it follows from the uncertainty of its inputs, and which inputs it answers to most
// (decision 0022). The model is any function f: ℝᵈ → ℝ; each input is given by its law's quantile function u ↦ x on (0, 1), so
// that any of tyche's laws (or a table's empirical quantiles) can stand for it, and the inputs are independent.
//
//   monte_carlo(n, d), latin_hypercube(n, d)   designs on (0, 1)ᵈ: independent uniforms; or n strata in each coordinate, each
//                                              met once, in a random pairing (McKay, Beckman and Conover 1979, Technometrics
//                                              21: 239–245), whose estimates of a mean have no larger variance for monotone f
//   propagate(f, inputs, n, design)            the output's sample: its mean, standard deviation and quantiles
//   delta_method(f, x₀, Σ)                     the first-order law: mean f(x₀), variance ∇fᵀ Σ ∇f with the gradient by central
//                                              differences (GUM, JCGM 100:2008, eq. 13); exact for an affine f, an approximation
//                                              otherwise, which the Monte Carlo propagation checks
//   morris(f, inputs, r, p)                    elementary effects along r trajectories on a grid of p levels u_k = (k + ½)/p,
//                                              each step a jump of p/2 levels (half the inputs' probability scale): μ, μ* = mean
//                                              |EE| and σ of each input's effects (Morris 1991, Technometrics 33: 161–174;
//                                              Campolongo, Cariboni and Saltelli 2007, Environ. Model. Softw. 22: 1509–1518)
//   sobol(f, inputs, N)                        variance-based indices from two independent designs A, B and the d hybrids A_B^i:
//                                              first order S_i = mean f(B)(f(A_B^i) − f(A)) / V (Saltelli et al. 2010, Comput.
//                                              Phys. Commun. 181: 259–270, Table 2 (b)), total S_Ti = mean (f(A) − f(A_B^i))² / 2V
//                                              (Jansen 1999), with bootstrap intervals; N (d + 2) evaluations
//
// Accuracy: Monte Carlo, so errors of order N^(−1/2), which the bootstrap intervals measure. The indices assume independent
// inputs; with dependent inputs they are not defined in this form.

#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::uncertainty {

using calx::linalg::Rectangular;
using Model = std::function<double(std::span<const double>)>;
using Quantile = std::function<double(double)>;

namespace detail {
inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::uncertainty: ") + what);
}
inline std::vector<double> at(const std::vector<Quantile>& in, std::span<const double> u) {
  std::vector<double> x(in.size());
  for (std::size_t i = 0; i < in.size(); i++) x[i] = in[i](u[i]);
  return x;
}
inline double quantile_of(std::vector<double> v, double p) {  // Hyndman and Fan's type 7
  std::sort(v.begin(), v.end());
  const double pos = p * static_cast<double>(v.size() - 1);
  const std::size_t i = static_cast<std::size_t>(pos);
  const double f = pos - static_cast<double>(i);
  return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v.back();
}
}  // namespace detail

// ---- designs ----

inline Rectangular monte_carlo(std::size_t n, std::size_t d, std::uint64_t seed = 1) {
  calx::random::Stream rng(seed, 0, 3, 0);
  Rectangular U(n, d);
  for (double& v : U.a) v = rng.uniform();
  return U;
}

inline Rectangular latin_hypercube(std::size_t n, std::size_t d, std::uint64_t seed = 1) {
  detail::require(n >= 1 && d >= 1, "latin_hypercube: n ≥ 1 and d ≥ 1");
  calx::random::Stream rng(seed, 0, 4, 0);
  Rectangular U(n, d);
  std::vector<std::size_t> perm(n);
  for (std::size_t j = 0; j < d; j++) {
    std::iota(perm.begin(), perm.end(), 0);
    for (std::size_t i = n; i-- > 1;) std::swap(perm[i], perm[static_cast<std::size_t>(rng.uniform() * static_cast<double>(i + 1)) % (i + 1)]);
    for (std::size_t i = 0; i < n; i++) U(i, j) = (static_cast<double>(perm[i]) + rng.uniform()) / static_cast<double>(n);
  }
  return U;
}

// ---- propagation ----

struct Propagated {
  double mean = 0, sd = 0;
  double lower = 0, median = 0, upper = 0;  // the 2.5%, 50% and 97.5% quantiles
  std::vector<double> sample;
};

inline Propagated propagate(const Model& f, const std::vector<Quantile>& inputs, std::size_t n = 10000, bool latin = true, std::uint64_t seed = 1) {
  const std::size_t d = inputs.size();
  detail::require(d >= 1 && n >= 10, "propagate: at least one input and ten draws");
  const Rectangular U = latin ? latin_hypercube(n, d, seed) : monte_carlo(n, d, seed);
  Propagated P;
  for (std::size_t i = 0; i < n; i++) P.sample.push_back(f(detail::at(inputs, U.row(i))));
  for (double v : P.sample) P.mean += v / static_cast<double>(n);
  for (double v : P.sample) P.sd += (v - P.mean) * (v - P.mean);
  P.sd = std::sqrt(P.sd / static_cast<double>(n - 1));
  P.lower = detail::quantile_of(P.sample, 0.025), P.median = detail::quantile_of(P.sample, 0.5), P.upper = detail::quantile_of(P.sample, 0.975);
  return P;
}

struct Delta {
  double mean = 0, sd = 0;
  std::vector<double> gradient;
  std::vector<double> share;  // each input's share of the first-order variance when Σ is diagonal
};

inline Delta delta_method(const Model& f, const std::vector<double>& x0, const calx::linalg::Matrix& cov) {
  const std::size_t d = x0.size();
  detail::require(cov.n == d, "delta_method: Σ of the inputs' order");
  Delta D;
  D.mean = f(x0);
  std::vector<double> y = x0;
  for (std::size_t i = 0; i < d; i++) {
    const double h = 1e-5 * std::max(1.0, std::fabs(x0[i]));
    y[i] = x0[i] + h;
    const double up = f(y);
    y[i] = x0[i] - h;
    const double down = f(y);
    y[i] = x0[i];
    D.gradient.push_back((up - down) / (2 * h));
  }
  double var = 0;
  for (std::size_t i = 0; i < d; i++)
    for (std::size_t j = 0; j < d; j++) var += D.gradient[i] * cov(i, j) * D.gradient[j];
  D.sd = std::sqrt(std::max(var, 0.0));
  for (std::size_t i = 0; i < d; i++) D.share.push_back(var > 0 ? D.gradient[i] * D.gradient[i] * cov(i, i) / var : 0.0);
  return D;
}

// ---- Morris's elementary effects ----

struct Morris {
  std::vector<double> mu, mu_star, sigma;  // per input, in output units per half the input's probability scale
  std::size_t trajectories = 0, levels = 0, evaluations = 0;
};

inline Morris morris(const Model& f, const std::vector<Quantile>& inputs, std::size_t r = 20, std::size_t p = 4, std::uint64_t seed = 1) {
  const std::size_t d = inputs.size();
  detail::require(d >= 1 && r >= 2 && p >= 2 && p % 2 == 0, "morris: inputs, at least two trajectories, and an even number of levels");
  calx::random::Stream rng(seed, 0, 5, 0);
  const std::size_t jump = p / 2;
  auto level = [&](std::size_t k) { return (static_cast<double>(k) + 0.5) / static_cast<double>(p); };
  const double du = static_cast<double>(jump) / static_cast<double>(p);
  std::vector<std::vector<double>> ee(d);
  Morris M;
  M.trajectories = r, M.levels = p;
  for (std::size_t t = 0; t < r; t++) {
    std::vector<std::size_t> k(d), order(d);
    for (std::size_t i = 0; i < d; i++) k[i] = static_cast<std::size_t>(rng.uniform() * static_cast<double>(p)) % p;
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t i = d; i-- > 1;) std::swap(order[i], order[static_cast<std::size_t>(rng.uniform() * static_cast<double>(i + 1)) % (i + 1)]);
    std::vector<double> u(d);
    for (std::size_t i = 0; i < d; i++) u[i] = level(k[i]);
    double fx = f(detail::at(inputs, u));
    M.evaluations++;
    for (std::size_t i : order) {
      const bool up = k[i] + jump < p && (k[i] < jump || rng.uniform() < 0.5);
      const std::size_t next = up ? k[i] + jump : k[i] - jump;
      std::vector<double> v = u;
      v[i] = level(next);
      const double fy = f(detail::at(inputs, v));
      M.evaluations++;
      ee[i].push_back((fy - fx) / (up ? du : -du));
      k[i] = next, u = v, fx = fy;
    }
  }
  for (std::size_t i = 0; i < d; i++) {
    double m = 0, ms = 0;
    for (double e : ee[i]) m += e, ms += std::fabs(e);
    m /= static_cast<double>(r), ms /= static_cast<double>(r);
    double s2 = 0;
    for (double e : ee[i]) s2 += (e - m) * (e - m);
    M.mu.push_back(m), M.mu_star.push_back(ms), M.sigma.push_back(std::sqrt(s2 / static_cast<double>(r - 1)));
  }
  return M;
}

// ---- Sobol' indices ----

struct Sobol {
  std::vector<double> first, total;              // per input
  std::vector<double> first_lower, first_upper;  // bootstrap 95% intervals
  std::vector<double> total_lower, total_upper;
  double mean = 0, variance = 0;
  std::size_t evaluations = 0;
};

inline Sobol sobol(const Model& f, const std::vector<Quantile>& inputs, std::size_t N = 4096, std::uint64_t seed = 1, std::size_t bootstrap = 200) {
  const std::size_t d = inputs.size();
  detail::require(d >= 1 && N >= 64, "sobol: inputs, and N ≥ 64");
  const Rectangular U = monte_carlo(N, 2 * d, seed);
  std::vector<double> fA(N), fB(N);
  std::vector<std::vector<double>> fAB(d, std::vector<double>(N));
  for (std::size_t n = 0; n < N; n++) {
    std::vector<double> a(d), b(d);
    for (std::size_t i = 0; i < d; i++) a[i] = U(n, i), b[i] = U(n, d + i);
    fA[n] = f(detail::at(inputs, a)), fB[n] = f(detail::at(inputs, b));
    for (std::size_t i = 0; i < d; i++) {
      std::vector<double> ab = a;
      ab[i] = b[i];
      fAB[i][n] = f(detail::at(inputs, ab));
    }
  }
  Sobol S;
  S.evaluations = N * (d + 2);
  auto estimate = [&](const std::vector<std::size_t>& idx, std::vector<double>& first, std::vector<double>& total, double& mean, double& var) {
    const double m = static_cast<double>(idx.size());
    mean = 0;
    for (std::size_t n : idx) mean += (fA[n] + fB[n]) / (2 * m);
    var = 0;
    for (std::size_t n : idx) var += ((fA[n] - mean) * (fA[n] - mean) + (fB[n] - mean) * (fB[n] - mean)) / (2 * m - 1);
    first.assign(d, 0.0), total.assign(d, 0.0);
    for (std::size_t i = 0; i < d; i++) {
      double s = 0, t = 0;
      for (std::size_t n : idx) {
        s += fB[n] * (fAB[i][n] - fA[n]);
        t += (fA[n] - fAB[i][n]) * (fA[n] - fAB[i][n]);
      }
      first[i] = var > 0 ? s / m / var : 0.0;
      total[i] = var > 0 ? t / (2 * m) / var : 0.0;
    }
  };
  std::vector<std::size_t> all(N);
  std::iota(all.begin(), all.end(), 0);
  estimate(all, S.first, S.total, S.mean, S.variance);
  calx::random::Stream rng(seed, 0, 6, 0);
  std::vector<std::vector<double>> bf(d), bt(d);
  for (std::size_t b = 0; b < bootstrap; b++) {
    std::vector<std::size_t> idx(N);
    for (auto& v : idx) v = static_cast<std::size_t>(rng.uniform() * static_cast<double>(N)) % N;
    std::vector<double> fi, ti;
    double mm, vv;
    estimate(idx, fi, ti, mm, vv);
    for (std::size_t i = 0; i < d; i++) bf[i].push_back(fi[i]), bt[i].push_back(ti[i]);
  }
  for (std::size_t i = 0; i < d; i++) {
    S.first_lower.push_back(bootstrap ? detail::quantile_of(bf[i], 0.025) : NAN), S.first_upper.push_back(bootstrap ? detail::quantile_of(bf[i], 0.975) : NAN);
    S.total_lower.push_back(bootstrap ? detail::quantile_of(bt[i], 0.025) : NAN), S.total_upper.push_back(bootstrap ? detail::quantile_of(bt[i], 0.975) : NAN);
  }
  return S;
}

}  // namespace tyche::uncertainty
