// Gaussian conditioning: the posterior of a Gaussian random field given noisy observations of linear
// functionals of it, extended one observation at a time.
//
// The model. A random field f has a Gaussian law with a known mean and covariance. Each observation is
// yᵢ = Lᵢ f + εᵢ, a linear functional of the field (its value at a point, a component of its gradient, an
// average) plus noise εᵢ ~ N(0, rᵢ), independent of f and of each other. With C = K + R, Kᵢⱼ = Cov(Lᵢ f, Lⱼ f)
// and R = diag(rᵢ), and the innovations dᵢ = yᵢ − E[yᵢ], the law of any further functional q = Q f given the
// observations is Gaussian with
//
//   E[q | y] = E[q] + kᵀ C⁻¹ d,   Var[q | y] = k_qq − kᵀ C⁻¹ k,   Cov[q, q′ | y] = k_qq′ − kᵀ C⁻¹ k′,
//
// kᵢ = Cov(q, Lᵢ f) (Rasmussen and Williams 2006, Gaussian Processes for Machine Learning, eqs. 2.23–2.24;
// the same formulas are simple kriging, Cressie 1993, Statistics for Spatial Data, §3.4.1, and the analysis
// step of optimal interpolation, Kalnay 2003, §5.4). Nothing here names the field: the caller supplies the
// covariances, from any kernel (physis analysis gives a potential field's).
//
// The computation. C = L Lᵀ by Cholesky's factorisation, kept as its rows; observing one more functional
// adds a row: L l = c (c its covariances with the earlier observations) by forward substitution, and the
// diagonal √(c_nn + r_n − |l|²). The whitened innovations z = L⁻¹ d grow the same way, zₙ = (dₙ − l · z)/Lₙₙ.
// An observation costs O(n²), against O(n³) to factorise afresh; the factor is the batch factorisation's,
// row by row rather than column by column, the same to rounding. A query costs O(n²):
// kᵀ C⁻¹ d = (L⁻¹ k) · z, kᵀ C⁻¹ k = |L⁻¹ k|²; the mean alone, by the weights C⁻¹ d, O(n).
//
// What follows from the model, and is checked: the posterior does not depend on the order of the
// observations; the whitened innovations z are independent standard normals when the model is the truth,
// so Σ zᵢ²/n → 1 tests its calibration; and the log marginal likelihood is
// log p(y) = −½ |z|² − Σ log Lᵢᵢ − (n/2) log 2π.
//
// Accuracy. Cholesky's factorisation is backward stable (Higham 2002, Theorem 10.3); the posterior variance
// is a difference, k_qq − |L⁻¹k|², and where the data determine q almost exactly it loses the digits of
// k_qq/Var[q | y]: it is clamped at zero, and a noise variance of zero on observations that nearly repeat
// one another makes C singular, which is refused (a pivot that is not positive).

#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::spatial {

class GaussianConditioning {
 public:
  // The number of observations conditioned on.
  std::size_t size() const { return z_.size(); }

  // One more observation: its prior covariances with the earlier ones (size() values, in their order), its
  // prior variance, its noise variance and its innovation y − E[y].
  void observe(std::span<const double> covariances, double variance, double noise, double innovation) {
    const std::size_t n = size();
    if (covariances.size() != n) throw std::invalid_argument("tyche::spatial::GaussianConditioning::observe: one covariance for each earlier observation");
    if (!(noise >= 0) || !std::isfinite(variance) || !std::isfinite(innovation)) throw std::invalid_argument("tyche::spatial::GaussianConditioning::observe: a finite variance and innovation, a noise variance of zero or more");
    std::vector<double> l = forward(covariances);
    double d = variance + noise;
    for (double x : l) d -= x * x;
    if (!(d > 0)) throw std::domain_error("tyche::spatial::GaussianConditioning::observe: the observations' covariance is not positive definite (an observation the earlier ones determine, without noise)");
    const double diag = std::sqrt(d);
    double zn = innovation;
    for (std::size_t k = 0; k < n; k++) zn -= l[k] * z_[k];
    zn /= diag;
    l.push_back(diag);
    rows_.push_back(std::move(l));
    z_.push_back(zn);
    log_det_half_ += std::log(diag);
    weights_.clear();
  }

  struct Posterior {
    double mean = 0, variance = 0;
    double sd() const { return std::sqrt(variance); }
  };

  // The posterior of a query: its prior covariances with the observations (size() values), its prior mean
  // and its prior variance.
  Posterior predict(std::span<const double> covariances, double prior_mean, double prior_variance) const {
    const std::vector<double> w = forward(covariances);
    Posterior p;
    p.mean = prior_mean;
    p.variance = prior_variance;
    for (std::size_t k = 0; k < w.size(); k++) p.mean += w[k] * z_[k], p.variance -= w[k] * w[k];
    if (p.variance < 0) p.variance = 0;
    return p;
  }

  // The posterior covariance of two queries.
  double covariance(std::span<const double> a, std::span<const double> b, double prior_covariance) const {
    const std::vector<double> wa = forward(a), wb = forward(b);
    double c = prior_covariance;
    for (std::size_t k = 0; k < wa.size(); k++) c -= wa[k] * wb[k];
    return c;
  }

  // C⁻¹ d, the weights by which the posterior mean is kᵀ (C⁻¹ d): computed once after each observation.
  const std::vector<double>& weights() const {
    if (weights_.size() != size()) {
      weights_ = z_;
      for (std::size_t i = size(); i-- > 0;) {
        for (std::size_t k = i + 1; k < size(); k++) weights_[i] -= rows_[k][i] * weights_[k];
        weights_[i] /= rows_[i][i];
      }
    }
    return weights_;
  }

  // The posterior mean alone, by the weights: O(n) a query.
  double mean(std::span<const double> covariances, double prior_mean) const {
    const auto& w = weights();
    if (covariances.size() != w.size()) throw std::invalid_argument("tyche::spatial::GaussianConditioning::mean: one covariance for each observation");
    double m = prior_mean;
    for (std::size_t k = 0; k < w.size(); k++) m += covariances[k] * w[k];
    return m;
  }

  // L⁻¹ k for a query's covariances k: the posterior of several queries at once is then
  // E = E₀ + (L⁻¹k)·z and Cov(a, b) = k_ab − (L⁻¹k_a)·(L⁻¹k_b), one substitution each.
  std::vector<double> whiten(std::span<const double> covariances) const { return forward(covariances); }

  // The whitened innovations z = L⁻¹ d: independent standard normals under the model.
  const std::vector<double>& whitened() const { return z_; }

  // log p(y) = −½ |z|² − ½ log det C − (n/2) log 2π.
  double log_likelihood() const {
    double q = 0;
    for (double x : z_) q += x * x;
    return -0.5 * q - log_det_half_ - 0.5 * double(size()) * std::log(2 * std::numbers::pi);
  }

 private:
  // L⁻¹ c by forward substitution.
  std::vector<double> forward(std::span<const double> c) const {
    const std::size_t n = size();
    if (c.size() != n) throw std::invalid_argument("tyche::spatial::GaussianConditioning: one covariance for each observation");
    std::vector<double> w(c.begin(), c.end());
    for (std::size_t i = 0; i < n; i++) {
      const auto& row = rows_[i];
      double s = w[i];
      for (std::size_t k = 0; k < i; k++) s -= row[k] * w[k];
      w[i] = s / row[i];
    }
    return w;
  }

  std::vector<std::vector<double>> rows_;  // row i of L: i + 1 entries, the last its diagonal
  std::vector<double> z_;                  // L⁻¹ d
  double log_det_half_ = 0;                // Σ log Lᵢᵢ = ½ log det C
  mutable std::vector<double> weights_;    // C⁻¹ d, when computed
};

}  // namespace tyche::spatial
