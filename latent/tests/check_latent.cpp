// Checks of latent-variable models (latent.hpp): factor analysis fitted to a covariance that two factors generate exactly
// recovers it (discrepancy zero, ΛΛᵀ and Ψ as they were), its EM never raising the discrepancy; varimax returns a simple
// structure that was rotated away, without changing ΛΛᵀ; on a sample, the true number of factors is not rejected and one
// factor fewer is; latent classes: one class is the model of independence exactly, and three classes that simulated 3,000
// persons are recovered, the log-likelihood never falling, the posteriors summing to one, and BIC preferring three to two.

#include <tyche/latent/latent.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace tyche::latent;

int main() {
  calx::verification::Suite s;

  // ---- Factor analysis ---------------------------------------------------------------------------------------------------------------
  const std::vector<std::vector<double>> L0 = {{0.9, 0}, {0.8, 0}, {0.7, 0}, {0, 0.85}, {0, 0.75}, {0, 0.6}};
  const std::vector<double> psi0 = {0.19, 0.36, 0.51, 0.2775, 0.4375, 0.64};
  Matrix S(6);
  for (std::size_t i = 0; i < 6; i++)
    for (std::size_t j = 0; j < 6; j++) S(i, j) = L0[i][0] * L0[j][0] + L0[i][1] * L0[j][1] + (i == j ? psi0[i] : 0);
  const auto f = factor_analysis(S, 2);
  double worst = 0, worst_psi = 0;
  for (std::size_t i = 0; i < 6; i++) {
    worst_psi = std::max(worst_psi, std::abs(f.uniqueness[i] - psi0[i]));
    for (std::size_t j = 0; j < 6; j++) {
      const double fitted = f.loadings[i][0] * f.loadings[j][0] + f.loadings[i][1] * f.loadings[j][1];
      worst = std::max(worst, std::abs(fitted - (L0[i][0] * L0[j][0] + L0[i][1] * L0[j][1])));
    }
  }
  s.require("on a covariance two factors generate exactly, the EM converges", f.converged, std::to_string(f.iterations) + " iterations");
  s.check("and the discrepancy falls to zero", f.discrepancy, 1e-8);
  s.check("ΛΛᵀ is recovered", worst, 1e-4);
  s.check("and so are the unique variances", worst_psi, 1e-4);
  bool monotone = true;
  for (std::size_t t = 1; t < f.trace.size(); t++) monotone = monotone && f.trace[t] <= f.trace[t - 1] + 1e-12;
  s.require("the discrepancy never rises from one iteration to the next", monotone);
  s.require("the degrees of freedom of two factors among six variables are ((6 − 2)² − 6 − 2)/2 = 4", f.df == 4);

  // Varimax: the simple structure rotated by 30° and rotated back.
  const double c = std::cos(0.5236), sn = std::sin(0.5236);
  std::vector<std::vector<double>> turned(6, std::vector<double>(2));
  for (std::size_t j = 0; j < 6; j++) turned[j] = {c * L0[j][0] - sn * L0[j][1], sn * L0[j][0] + c * L0[j][1]};
  const auto back = varimax(turned);
  double gap = 0, kept = 0;
  for (std::size_t j = 0; j < 6; j++) {
    // Up to the order and signs of the columns.
    const double straight = std::abs(std::abs(back[j][0]) - L0[j][0]) + std::abs(std::abs(back[j][1]) - L0[j][1]);
    const double swapped = std::abs(std::abs(back[j][1]) - L0[j][0]) + std::abs(std::abs(back[j][0]) - L0[j][1]);
    gap = std::max(gap, std::min(straight, swapped));
    for (std::size_t i = 0; i < 6; i++)
      kept = std::max(kept, std::abs(back[i][0] * back[j][0] + back[i][1] * back[j][1] - (turned[i][0] * turned[j][0] + turned[i][1] * turned[j][1])));
  }
  s.check("varimax finds the simple structure that a rotation of 30° hid", gap, 1e-8);
  s.check("without changing ΛΛᵀ", kept, 1e-12);
  s.require("and raises the varimax criterion", varimax_criterion(back) > varimax_criterion(turned));

  // On a sample of 2,000.
  calx::random::Stream rng(20261115, 51, 0, 0);
  const std::size_t n = 2000;
  std::vector<std::vector<double>> data(n, std::vector<double>(6));
  for (auto& row : data) {
    const double f1 = rng.normal(), f2 = rng.normal();
    for (std::size_t j = 0; j < 6; j++) row[j] = L0[j][0] * f1 + L0[j][1] * f2 + std::sqrt(psi0[j]) * rng.normal();
  }
  Matrix Sn(6);
  std::vector<double> mean(6, 0);
  for (const auto& row : data)
    for (std::size_t j = 0; j < 6; j++) mean[j] += row[j] / n;
  for (const auto& row : data)
    for (std::size_t i = 0; i < 6; i++)
      for (std::size_t j = 0; j < 6; j++) Sn(i, j) += (row[i] - mean[i]) * (row[j] - mean[j]) / (n - 1);
  const auto two = factor_analysis(Sn, 2, n), one = factor_analysis(Sn, 1, n);
  s.require("on a sample of 2,000, two factors are not rejected at the 0.1% level (χ²₄ < 18.47)", two.chi2 < 18.47, std::to_string(two.chi2));
  s.require("and one factor is (χ²₉ far above 27.88)", one.chi2 > 27.88 && one.df == 9, std::to_string(one.chi2));

  // ---- Latent classes -------------------------------------------------------------------------------------------------------------------
  const std::vector<double> prior = {0.5, 0.3, 0.2};
  const std::vector<std::vector<double>> profile = {{0.9, 0.85, 0.8, 0.9, 0.2, 0.1, 0.15, 0.1},
                                                    {0.1, 0.2, 0.15, 0.1, 0.9, 0.85, 0.8, 0.9},
                                                    {0.85, 0.9, 0.1, 0.15, 0.85, 0.9, 0.1, 0.2}};
  std::vector<std::vector<int>> x(3000, std::vector<int>(8));
  for (auto& row : x) {
    const double u = rng.uniform();
    const std::size_t k = u < 0.5 ? 0 : u < 0.8 ? 1 : 2;
    for (std::size_t j = 0; j < 8; j++) row[j] = rng.bernoulli(profile[k][j]);
  }
  const auto c1 = latent_classes(x, 1, rng, 1);
  double independence = 0, worst_mean = 0;
  for (std::size_t j = 0; j < 8; j++) {
    double m = 0;
    for (const auto& row : x) m += row[j];
    m /= x.size();
    worst_mean = std::max(worst_mean, std::abs(c1.probability[0][j] - m));
    for (const auto& row : x) independence += std::log(row[j] ? m : 1 - m);
  }
  s.check("one class is independence: its probabilities are the items' proportions", worst_mean, 1e-12);
  s.check("and its log-likelihood is the independence model's", std::abs(c1.log_likelihood - independence) / std::abs(independence), 1e-12);
  const auto c3 = latent_classes(x, 3, rng, 10);
  double wp = 0, wq = 0;
  for (std::size_t k = 0; k < 3; k++) {
    wp = std::max(wp, std::abs(c3.prior[k] - prior[k]));
    for (std::size_t j = 0; j < 8; j++) wq = std::max(wq, std::abs(c3.probability[k][j] - profile[k][j]));
  }
  s.check("three classes recovered from 3,000 persons: their shares within 0.04", wp, 0.04);
  s.check("and their profiles within 0.08", wq, 0.08);
  bool rising = true, sums = true;
  for (std::size_t t = 1; t < c3.trace.size(); t++) rising = rising && c3.trace[t] >= c3.trace[t - 1] - 1e-9;
  for (const auto& row : c3.posterior) sums = sums && std::abs(row[0] + row[1] + row[2] - 1) < 1e-12;
  s.require("the log-likelihood never falls, and every person's posterior sums to one", rising && sums);
  const auto c2 = latent_classes(x, 2, rng, 10);
  s.require("BIC prefers three classes to two", c3.bic < c2.bic, std::to_string(c3.bic) + " against " + std::to_string(c2.bic));
  bool refused = false;
  try { (void)factor_analysis(S, 3); } catch (const std::invalid_argument&) { refused = true; }
  s.require("as many factors as half the variables are refused", refused);
  return s.done();
}
