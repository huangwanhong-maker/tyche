// Latent variables: what a few unobserved quantities explain of many observed ones.
//
// Factor analysis. p observed variables with covariance Σ are modelled as Σ = ΛΛᵀ + Ψ: k common factors, uncorrelated and
// of unit variance, with loadings Λ (p × k), and a unique variance ψⱼ for each variable (Lawley and Maxwell 1971, Factor
// Analysis as a Statistical Method). Given a sample covariance S of n observations, the maximum-likelihood estimate
// minimises the discrepancy F = log|Σ| + tr(Σ⁻¹S) − log|S| − p, which is zero exactly when Σ = S. It is found here by the
// EM algorithm of Rubin and Thayer (1982, EM algorithms for ML factor analysis, Psychometrika 47, 69–76): with
// β = ΛᵀΣ⁻¹ and C = I − βΛ + βSβᵀ, the step is Λ ← SβᵀC⁻¹, Ψ ← diag(S − ΛβS); the likelihood never decreases. A unique
// variance that falls to the floor (10⁻⁶ of its variable's variance) is a Heywood case, reported, not hidden. The
// likelihood-ratio statistic (n − 1 − (2p + 5)/6 − 2k/3)·F with Bartlett's correction is χ² on ((p − k)² − p − k)/2 degrees
// of freedom (Lawley and Maxwell 1971, §4.6). Loadings are identified only up to rotation; varimax rotates them, with
// Kaiser's normalisation, to make each factor's squared loadings as unequal as possible: pairs of factors rotated by the
// angle in closed form, until no rotation changes them (Kaiser 1958, The varimax criterion for analytic rotation in factor
// analysis, Psychometrika 23, 187–200). Rotation leaves ΛΛᵀ, and so the fit, unchanged.
//
// Latent classes. J binary items are independent given a class among K, the class drawn with probabilities π (Lazarsfeld
// and Henry 1968, Latent Structure Analysis; Goodman 1974, Biometrika 61, 215–231). The EM algorithm alternates the
// posterior of each person's class and the weighted proportions; the log-likelihood never decreases; a missing response
// (−1) contributes nothing. Starts are drawn from a stream the caller gives, the best of several kept, since the likelihood
// has several maxima; classes are identified only up to their labels, and are returned in decreasing order of π.

#pragma once

#include <calx/linalg/cholesky.hpp>
#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigen.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::latent {

using calx::linalg::Matrix;

// ---- Factor analysis ----------------------------------------------------------------------------------------------------------

struct FactorFit {
  std::vector<std::vector<double>> loadings;  // [variable][factor]
  std::vector<double> uniqueness;             // ψⱼ
  std::vector<double> communality;            // Σ_k λⱼₖ²
  double discrepancy = 0;                     // F at the estimate
  double chi2 = std::numeric_limits<double>::quiet_NaN(), df = 0;  // the likelihood-ratio test, when n is given
  std::vector<double> trace;                  // F at each iteration
  std::vector<std::size_t> heywood;           // variables whose unique variance reached its floor
  std::size_t iterations = 0;
  bool converged = false;
};

namespace detail {

inline double log_det_spd(const Matrix& A) {
  const auto c = calx::linalg::cholesky(A);
  if (!c.positive_definite) throw std::invalid_argument("tyche::latent: a covariance that is not positive definite");
  double s = 0;
  for (std::size_t i = 0; i < A.n; i++) s += 2 * std::log(c.l(i, i));
  return s;
}

inline Matrix inverse_spd(const Matrix& A) { return calx::linalg::solve(A, Matrix::identity(A.n)); }

inline Matrix implied(const std::vector<std::vector<double>>& L, const std::vector<double>& psi) {
  const std::size_t p = L.size(), k = L.empty() ? 0 : L[0].size();
  Matrix S(p);
  for (std::size_t i = 0; i < p; i++)
    for (std::size_t j = 0; j < p; j++) {
      double v = 0;
      for (std::size_t f = 0; f < k; f++) v += L[i][f] * L[j][f];
      S(i, j) = v + (i == j ? psi[i] : 0);
    }
  return S;
}

inline double discrepancy(const Matrix& sigma, const Matrix& S, double logdet_s) {
  const Matrix inv = inverse_spd(sigma);
  double tr = 0;
  for (std::size_t i = 0; i < S.n; i++)
    for (std::size_t j = 0; j < S.n; j++) tr += inv(i, j) * S(j, i);
  return log_det_spd(sigma) + tr - logdet_s - static_cast<double>(S.n);
}

}  // namespace detail

// The discrepancy F of loadings and unique variances from a sample covariance.
inline double discrepancy(const Matrix& S, const std::vector<std::vector<double>>& loadings, const std::vector<double>& uniqueness) {
  return detail::discrepancy(detail::implied(loadings, uniqueness), S, detail::log_det_spd(S));
}

// k factors by maximum likelihood from a covariance (or correlation) matrix S of n observations (n = 0: no test).
inline FactorFit factor_analysis(const Matrix& S, std::size_t k, std::size_t n = 0, std::size_t most = 20000, double tolerance = 1e-12) {
  const std::size_t p = S.n;
  if (k == 0 || 2 * k >= p) throw std::invalid_argument("tyche::latent::factor_analysis: between one factor and fewer than half the variables");
  const double logdet_s = detail::log_det_spd(S);
  // Start: principal axes of S with a common unique variance of half each variable's variance.
  std::vector<double> psi(p);
  for (std::size_t j = 0; j < p; j++) psi[j] = 0.5 * S(j, j);
  const auto eig = calx::linalg::symmetric_eigen(S);
  std::vector<std::vector<double>> L(p, std::vector<double>(k, 0));
  for (std::size_t f = 0; f < k; f++) {
    const std::size_t c = p - 1 - f;  // the largest eigenvalues last
    const double scale = std::sqrt(std::max(eig.values[c] * 0.5, 1e-6));
    for (std::size_t j = 0; j < p; j++) L[j][f] = eig.vectors(j, c) * scale;
  }
  FactorFit fit;
  double previous = std::numeric_limits<double>::infinity();
  for (fit.iterations = 0; fit.iterations < most; fit.iterations++) {
    const Matrix sigma = detail::implied(L, psi);
    const double F = detail::discrepancy(sigma, S, logdet_s);
    fit.trace.push_back(F);
    if (previous - F < tolerance) {
      fit.converged = true;
      break;
    }
    previous = F;
    const Matrix inv = detail::inverse_spd(sigma);
    // β = Λᵀ Σ⁻¹ (k × p); βS (k × p); C = I − βΛ + βSβᵀ (k × k).
    std::vector<std::vector<double>> beta(k, std::vector<double>(p, 0)), bs(k, std::vector<double>(p, 0));
    for (std::size_t f = 0; f < k; f++)
      for (std::size_t j = 0; j < p; j++)
        for (std::size_t l = 0; l < p; l++) beta[f][j] += L[l][f] * inv(l, j);
    for (std::size_t f = 0; f < k; f++)
      for (std::size_t j = 0; j < p; j++)
        for (std::size_t l = 0; l < p; l++) bs[f][j] += beta[f][l] * S(l, j);
    Matrix C = Matrix::identity(k);
    for (std::size_t a = 0; a < k; a++)
      for (std::size_t b = 0; b < k; b++) {
        double bl = 0, bsb = 0;
        for (std::size_t j = 0; j < p; j++) bl += beta[a][j] * L[j][b], bsb += bs[a][j] * beta[b][j];
        C(a, b) += bsb - bl;
      }
    const Matrix Ci = calx::linalg::solve(C, Matrix::identity(k));
    std::vector<std::vector<double>> next(p, std::vector<double>(k, 0));
    for (std::size_t j = 0; j < p; j++)
      for (std::size_t f = 0; f < k; f++)
        for (std::size_t g = 0; g < k; g++) next[j][f] += bs[g][j] * Ci(g, f);  // (Sβᵀ)_{jg} = (βS)_{gj}, S symmetric
    for (std::size_t j = 0; j < p; j++) {
      double v = S(j, j);
      for (std::size_t f = 0; f < k; f++) v -= next[j][f] * bs[f][j];
      psi[j] = std::max(v, 1e-6 * S(j, j));
    }
    L = std::move(next);
  }
  fit.loadings = L;
  fit.uniqueness = psi;
  for (std::size_t j = 0; j < p; j++) {
    double c = 0;
    for (std::size_t f = 0; f < k; f++) c += L[j][f] * L[j][f];
    fit.communality.push_back(c);
    if (psi[j] <= 1e-6 * S(j, j) * (1 + 1e-9)) fit.heywood.push_back(j);
  }
  fit.discrepancy = fit.trace.back();
  const double pp = static_cast<double>(p), kk = static_cast<double>(k);
  fit.df = ((pp - kk) * (pp - kk) - pp - kk) / 2;
  if (n > 0) fit.chi2 = (static_cast<double>(n) - 1 - (2 * pp + 5) / 6 - 2 * kk / 3) * fit.discrepancy;
  return fit;
}

// The varimax criterion of loadings normalised by their communalities.
inline double varimax_criterion(const std::vector<std::vector<double>>& L) {
  const std::size_t p = L.size(), k = L.empty() ? 0 : L[0].size();
  double v = 0;
  for (std::size_t f = 0; f < k; f++) {
    double s2 = 0, s4 = 0;
    for (std::size_t j = 0; j < p; j++) {
      double h = 0;
      for (std::size_t g = 0; g < k; g++) h += L[j][g] * L[j][g];
      const double x = h > 0 ? L[j][f] * L[j][f] / h : 0;
      s2 += x, s4 += x * x;
    }
    v += s4 / static_cast<double>(p) - (s2 / static_cast<double>(p)) * (s2 / static_cast<double>(p));
  }
  return v;
}

// Loadings rotated by varimax with Kaiser's normalisation.
inline std::vector<std::vector<double>> varimax(std::vector<std::vector<double>> L, std::size_t most = 1000, double tolerance = 1e-12) {
  const std::size_t p = L.size(), k = L.empty() ? 0 : L[0].size();
  if (k < 2) return L;
  std::vector<double> h(p);
  for (std::size_t j = 0; j < p; j++) {
    for (std::size_t f = 0; f < k; f++) h[j] += L[j][f] * L[j][f];
    h[j] = std::sqrt(h[j]);
    if (h[j] > 0)
      for (std::size_t f = 0; f < k; f++) L[j][f] /= h[j];
  }
  const double n = static_cast<double>(p);
  for (std::size_t sweep = 0; sweep < most; sweep++) {
    double largest = 0;
    for (std::size_t a = 0; a < k; a++)
      for (std::size_t b = a + 1; b < k; b++) {
        double A = 0, B = 0, C = 0, D = 0;
        for (std::size_t j = 0; j < p; j++) {
          const double x = L[j][a], y = L[j][b], u = x * x - y * y, v = 2 * x * y;
          A += u, B += v, C += u * u - v * v, D += 2 * u * v;
        }
        const double phi = 0.25 * std::atan2(D - 2 * A * B / n, C - (A * A - B * B) / n);
        largest = std::max(largest, std::abs(phi));
        const double c = std::cos(phi), s = std::sin(phi);
        for (std::size_t j = 0; j < p; j++) {
          const double x = L[j][a], y = L[j][b];
          L[j][a] = c * x + s * y;
          L[j][b] = -s * x + c * y;
        }
      }
    if (largest < tolerance) break;
  }
  for (std::size_t j = 0; j < p; j++)
    for (std::size_t f = 0; f < k; f++) L[j][f] *= h[j];
  return L;
}

// ---- Latent classes ------------------------------------------------------------------------------------------------------------------

struct ClassFit {
  std::vector<double> prior;                     // π, decreasing
  std::vector<std::vector<double>> probability;  // [class][item]: P(item = 1 | class)
  std::vector<std::vector<double>> posterior;    // [person][class]
  double log_likelihood = -std::numeric_limits<double>::infinity();
  double bic = 0;
  std::vector<double> trace;                     // of the best start
  std::size_t iterations = 0;
  bool converged = false;
};

namespace detail {
inline ClassFit em(const std::vector<std::vector<int>>& x, std::size_t K, calx::random::Stream& rng, std::size_t most, double tolerance) {
  const std::size_t n = x.size(), J = x[0].size();
  ClassFit f;
  f.prior.assign(K, 1.0 / static_cast<double>(K));
  f.probability.assign(K, std::vector<double>(J));
  for (auto& row : f.probability)
    for (auto& v : row) v = 0.2 + 0.6 * rng.uniform();
  f.posterior.assign(n, std::vector<double>(K));
  double previous = -std::numeric_limits<double>::infinity();
  for (f.iterations = 0; f.iterations < most; f.iterations++) {
    double ll = 0;
    for (std::size_t i = 0; i < n; i++) {
      std::vector<double> lp(K);
      for (std::size_t c = 0; c < K; c++) {
        lp[c] = std::log(f.prior[c]);
        for (std::size_t j = 0; j < J; j++)
          if (x[i][j] >= 0) lp[c] += std::log(x[i][j] ? f.probability[c][j] : 1 - f.probability[c][j]);
      }
      const double top = *std::max_element(lp.begin(), lp.end());
      double z = 0;
      for (std::size_t c = 0; c < K; c++) z += std::exp(lp[c] - top);
      for (std::size_t c = 0; c < K; c++) f.posterior[i][c] = std::exp(lp[c] - top) / z;
      ll += top + std::log(z);
    }
    f.trace.push_back(ll);
    f.log_likelihood = ll;
    if (ll - previous < tolerance * (1 + std::abs(ll))) {
      f.converged = true;
      break;
    }
    previous = ll;
    for (std::size_t c = 0; c < K; c++) {
      double w = 0;
      for (std::size_t i = 0; i < n; i++) w += f.posterior[i][c];
      f.prior[c] = std::max(w / static_cast<double>(n), 1e-300);
      for (std::size_t j = 0; j < J; j++) {
        double seen = 0, right = 0;
        for (std::size_t i = 0; i < n; i++)
          if (x[i][j] >= 0) seen += f.posterior[i][c], right += f.posterior[i][c] * x[i][j];
        f.probability[c][j] = std::clamp(seen > 0 ? right / seen : 0.5, 1e-10, 1 - 1e-10);
      }
    }
  }
  return f;
}
}  // namespace detail

inline ClassFit latent_classes(const std::vector<std::vector<int>>& x, std::size_t K, calx::random::Stream& rng, std::size_t starts = 10, std::size_t most = 5000,
                               double tolerance = 1e-12) {
  if (x.empty() || x[0].empty() || K == 0) throw std::invalid_argument("tyche::latent::latent_classes: responses and at least one class");
  for (const auto& row : x) {
    if (row.size() != x[0].size()) throw std::invalid_argument("tyche::latent::latent_classes: every person has a response, or −1, for every item");
    for (const int v : row)
      if (v != 0 && v != 1 && v != -1) throw std::invalid_argument("tyche::latent::latent_classes: responses are 1, 0 or −1");
  }
  ClassFit best;
  for (std::size_t s = 0; s < std::max<std::size_t>(starts, 1); s++) {
    auto f = detail::em(x, K, rng, most, tolerance);
    if (f.log_likelihood > best.log_likelihood) best = std::move(f);
  }
  // Classes in decreasing order of π.
  std::vector<std::size_t> order(K);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return best.prior[a] > best.prior[b]; });
  ClassFit sorted = best;
  for (std::size_t c = 0; c < K; c++) {
    sorted.prior[c] = best.prior[order[c]];
    sorted.probability[c] = best.probability[order[c]];
    for (std::size_t i = 0; i < x.size(); i++) sorted.posterior[i][c] = best.posterior[i][order[c]];
  }
  const double parameters = static_cast<double>(K - 1 + K * x[0].size());
  sorted.bic = -2 * sorted.log_likelihood + parameters * std::log(static_cast<double>(x.size()));
  return sorted;
}

}  // namespace tyche::latent
