// Checks of tyche::bayes's conjugate models: the posteriors by hand; the beta–binomial's marginal likelihood summing to one over
// k; the gamma–Poisson's marginal likelihood against a numerical integral over the rate; the normal–inverse-gamma's marginal
// likelihood of one observation equal to its prior predictive Student t, and of two observations equal to the product of the
// sequential predictives; the Dirichlet's posterior mean and its marginal likelihood of a sequence as the product of predictives.

#include <tyche/bayes/conjugate.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <numbers>
#include <vector>

using namespace tyche::bayes;
using calx::verification::relative;

int main() {
  calx::verification::Suite s;

  {
    const auto r = beta_binomial(2, 3, 7, 10);
    s.check("beta–binomial: Beta(2, 3) and 7 of 10 give Beta(9, 6), mean 0.6", std::fabs(r.a - 9) + std::fabs(r.b - 6) + std::fabs(r.mean - 0.6), 1e-15);
    double total = 0;
    for (int k = 0; k <= 10; k++) total += std::exp(beta_binomial(2, 3, k, 10).log_marginal);
    s.check("beta–binomial: the marginal likelihoods of k = 0 … 10 sum to one", std::fabs(total - 1), 1e-13);
    s.check("beta–binomial: a uniform prior gives every k the probability 1/(n + 1)", std::fabs(std::exp(beta_binomial(1, 1, 4, 10).log_marginal) - 1.0 / 11), 1e-14);
  }
  {
    const std::vector<double> y{3, 5, 2}, t{1.0, 2.0, 0.5};
    const auto r = gamma_poisson(2, 1, y, t);
    s.check("gamma–Poisson: Gamma(2, 1) and 10 events over 3.5 give Gamma(12, 4.5)", std::fabs(r.shape - 12) + std::fabs(r.rate - 4.5), 1e-15);
    // ∫ Π Poisson(y_i | λ t_i) Gamma(λ | 2, 1) dλ by the trapezoid rule on a fine grid.
    double integral = 0;
    const double h = 1e-4;
    for (double lam = h / 2; lam < 40; lam += h) {
      double lp = std::log(lam) - lam;  // Gamma(2, 1): λ e^{−λ} / Γ(2)
      for (std::size_t i = 0; i < y.size(); i++) lp += y[i] * std::log(lam * t[i]) - lam * t[i] - std::lgamma(y[i] + 1);
      integral += std::exp(lp) * h;
    }
    s.check("gamma–Poisson: the marginal likelihood against the integral over the rate", relative(std::exp(r.log_marginal), integral), 1e-6);
  }
  {
    const std::vector<double> x{1.2, 0.8, 1.9, 1.1};
    const auto r = normal_known_variance(0, 2, x, 1);
    // precision 1/4 + 4 = 4.25; mean (0 + 5)/4.25.
    s.check("normal with known variance: precision 1/τ₀² + n/σ², mean Σx/σ² over it", std::fabs(r.mean - 5 / 4.25) + std::fabs(r.sd - std::sqrt(1 / 4.25)), 1e-15);
  }
  {
    const double mu0 = 1, kappa0 = 2, alpha0 = 3, beta0 = 4;
    const std::vector<double> one{2.5};
    const auto r1 = normal_inverse_gamma(mu0, kappa0, alpha0, beta0, one);
    // The prior predictive of one observation is t_{2α₀}(μ₀, β₀(κ₀ + 1)/(α₀κ₀)).
    const auto prior = normal_inverse_gamma(mu0, kappa0, alpha0, beta0, std::vector<double>{});
    const double sc = prior.predictive_scale, z = (2.5 - mu0) / sc;
    s.check("normal–inverse-gamma: p(x) of one observation is the prior predictive Student t", relative(std::exp(r1.log_marginal), prior.mean_law.pdf(z) / sc), 1e-13);
    const std::vector<double> two{2.5, 0.4};
    const auto r2 = normal_inverse_gamma(mu0, kappa0, alpha0, beta0, two);
    const double z2 = (0.4 - r1.mu) / r1.predictive_scale;
    s.check("normal–inverse-gamma: p(x₁, x₂) = p(x₁) p(x₂ | x₁), the sequential predictive",
            relative(std::exp(r2.log_marginal), std::exp(r1.log_marginal) * r1.mean_law.pdf(z2) / r1.predictive_scale), 1e-13);
    s.check("normal–inverse-gamma: κₙ = κ₀ + n, μₙ = (κ₀μ₀ + n x̄)/κₙ, αₙ = α₀ + n/2", std::fabs(r2.kappa - 4) + std::fabs(r2.mu - (2 + 2.9) / 4) + std::fabs(r2.alpha - 4), 1e-15);
  }
  {
    const std::vector<double> a{1, 2, 3}, n{4, 0, 1};
    const auto r = dirichlet_multinomial(a, n);
    s.check("Dirichlet: the posterior mean (α + n)/Σ", std::fabs(r.mean[0] - 5.0 / 11) + std::fabs(r.mean[2] - 4.0 / 11), 1e-15);
    // The sequence A, A, A, A, C as a product of predictives: (1/6)(2/7)(3/8)(4/9)(3/10).
    s.check("Dirichlet: the marginal likelihood of a sequence is the product of its predictives", relative(std::exp(r.log_marginal), (1.0 / 6) * (2.0 / 7) * (3.0 / 8) * (4.0 / 9) * (3.0 / 10)), 1e-13);
  }
  return s.done();
}
