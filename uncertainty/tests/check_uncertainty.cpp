// Checks of tyche::uncertainty: a Latin hypercube meets every stratum once in every coordinate; a linear model's propagated mean and
// spread, and the delta method's, which is exact for it; the delta method on a product against its closed form, and the Monte Carlo
// spread against the exact one, which the first order misses by σ₁²σ₂²; Morris's elementary effects of a linear model, exact; Sobol'
// indices of a linear model (c_i²/Σc²) and of Ishigami's function, whose indices are known in closed form (Ishigami and Homma 1990;
// Sobol' and Levitan 1999), with the bootstrap interval holding the truth.

#include <tyche/probability/distributions.hpp>
#include <tyche/uncertainty/uncertainty.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

using namespace tyche::uncertainty;

int main() {
  calx::verification::Suite s;

  {
    const auto U = latin_hypercube(50, 3, 7);
    bool every = true;
    for (std::size_t j = 0; j < 3; j++) {
      std::vector<int> hit(50, 0);
      for (std::size_t i = 0; i < 50; i++) hit[static_cast<std::size_t>(U(i, j) * 50)]++;
      every = every && std::all_of(hit.begin(), hit.end(), [](int h) { return h == 1; });
    }
    s.require("Latin hypercube: every coordinate meets each of its 50 strata exactly once", every);
  }
  const tyche::probability::Normal X1{1, 1}, X2{2, 0.5};
  const std::vector<Quantile> normals{[&](double u) { return X1.quantile(u); }, [&](double u) { return X2.quantile(u); }};
  {
    const auto P = propagate([](std::span<const double> x) { return 2 * x[0] + 3 * x[1]; }, normals, 20000);
    s.check("propagate: 2X₁ + 3X₂ has mean 8", std::fabs(P.mean - 8), 0.02);
    s.check("propagate: and standard deviation √(4 + 2.25) = 2.5, within 2%", std::fabs(P.sd / 2.5 - 1), 0.02);
    calx::linalg::Matrix cov(2);
    cov(0, 0) = 1, cov(1, 1) = 0.25;
    const auto D = delta_method([](std::span<const double> x) { return 2 * x[0] + 3 * x[1]; }, {1, 2}, cov);
    s.check("delta method: exact for a linear model (mean 8, sd 2.5)", std::fabs(D.mean - 8) + std::fabs(D.sd - 2.5), 1e-8);
    const auto Dp = delta_method([](std::span<const double> x) { return x[0] * x[1]; }, {1, 2}, cov);
    s.check("delta method on X₁X₂: variance μ₂²σ₁² + μ₁²σ₂² = 4.25", std::fabs(Dp.sd * Dp.sd - 4.25), 1e-8);
    const auto Pp = propagate([](std::span<const double> x) { return x[0] * x[1]; }, normals, 40000);
    s.check("propagate X₁X₂: the exact sd √(4.25 + σ₁²σ₂²) = √4.5, which the first order misses", std::fabs(Pp.sd / std::sqrt(4.5) - 1), 0.02);
  }
  {
    const std::vector<double> c{4, -2, 1, 0};
    const std::vector<Quantile> uniform(4, [](double u) { return u; });
    const auto f = [&](std::span<const double> x) { double y = 0; for (std::size_t i = 0; i < 4; i++) y += c[i] * x[i]; return y; };
    const auto M = morris(f, uniform, 30, 4, 3);
    double worst = 0;
    for (std::size_t i = 0; i < 4; i++) worst = std::max({worst, std::fabs(M.mu_star[i] - std::fabs(c[i])), M.sigma[i]});
    s.check("Morris: a linear model's elementary effects are its coefficients, μ* = |c_i| and σ = 0", worst, 1e-12);
    s.require("Morris: r (d + 1) evaluations", M.evaluations == 30 * 5);
    const auto S = sobol(f, uniform, 20000, 11, 100);
    double sum = 0, wf = 0, wt = 0;
    for (double v : c) sum += v * v;
    for (std::size_t i = 0; i < 4; i++) wf = std::max(wf, std::fabs(S.first[i] - c[i] * c[i] / sum)), wt = std::max(wt, std::fabs(S.total[i] - c[i] * c[i] / sum));
    s.check("Sobol': a linear model's first-order indices are c_i²/Σc²", wf, 0.03);
    s.check("Sobol': and its total indices equal them (no interaction)", wt, 0.03);
    s.require("Sobol': N (d + 2) evaluations", S.evaluations == 20000 * 6);
  }
  {
    const double a = 7, b = 0.1, pi4 = std::pow(std::numbers::pi, 4), pi8 = pi4 * pi4;
    const double V1 = 0.5 * (1 + b * pi4 / 5) * (1 + b * pi4 / 5), V2 = a * a / 8, V13 = 8 * b * b * pi8 / 225, V = V1 + V2 + V13;
    const std::vector<Quantile> box(3, [](double u) { return -std::numbers::pi + 2 * std::numbers::pi * u; });
    const auto ishigami = [=](std::span<const double> x) { return std::sin(x[0]) + a * std::sin(x[1]) * std::sin(x[1]) + b * std::pow(x[2], 4) * std::sin(x[0]); };
    const auto S = sobol(ishigami, box, 16384, 5, 200);
    const double first[3] = {V1 / V, V2 / V, 0}, total[3] = {(V1 + V13) / V, V2 / V, V13 / V};
    double wf = 0, wt = 0;
    for (int i = 0; i < 3; i++) wf = std::max(wf, std::fabs(S.first[i] - first[i])), wt = std::max(wt, std::fabs(S.total[i] - total[i]));
    s.check("Sobol', Ishigami: first-order indices 0.3139, 0.4424, 0", wf, 0.03);
    s.check("Sobol', Ishigami: total indices 0.5576, 0.4424, 0.2437 (x₃ acts only with x₁)", wt, 0.03);
    s.check("Sobol', Ishigami: the variance 13.8446", std::fabs(S.variance / V - 1), 0.03);
    s.require("Sobol', Ishigami: the bootstrap interval of S₁ holds the truth", S.first_lower[0] <= first[0] && first[0] <= S.first_upper[0]);
  }
  return s.done();
}
