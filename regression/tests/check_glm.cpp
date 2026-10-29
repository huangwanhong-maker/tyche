// Checks of tyche/regression's generalised linear models.
//
// - Dobson's (1990, p. 93) Poisson example, the one R's ?glm prints: counts 18 17 15 20 10 20 25
//   13 12 by outcome and treatment. The model is the log-linear independence model, so its MLE
//   is closed-form (fitted = row total × column total / grand total): β = (log 21, log 40/63,
//   log 47/63, 0, 0), standard errors √(1/63 + 1/50 − 1/150), √(1/63 + 1/40), √(1/63 + 1/47),
//   √(2/50) twice; and R prints residual deviance 5.1291 on 4 df, null deviance 10.5814 on 8 df,
//   AIC 56.761.
// - Bliss's (1935) beetle mortality data as Dobson and Barnett (2008, An Introduction to
//   Generalized Linear Models, 3rd ed., §7.3) fit them: logit b = (−60.72, 34.27), standard errors
//   (5.18, 2.91), deviance 11.23; probit (−34.94, 19.73), (2.65, 1.49), 10.12; at their printed
//   precision, and against statsmodels 0.15.0 (GLM, Binomial; computed 2026-10-24) to 10⁻⁷ as a
//   cross-check of the digits beyond.
// - The gaussian family with the identity link is least squares: NIST's Norris data against ols().
// - Poisson regression's score equations Σ xᵢ(yᵢ − μᵢ) = 0; an offset's scale moving only the
//   intercept; PPML (Santos Silva and Tenreyro 2006) recovering a gravity equation's elasticities
//   under heteroskedastic multiplicative errors, where least squares on logarithms is biased;
//   clusters of one observation each reproducing hc1.

#include <tyche/regression/glm.hpp>
#include <tyche/regression/linear.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace tyche::regression;
using calx::verification::relative;

int main() {
  calx::verification::Suite s;

  // ---- Dobson's Poisson example ----
  {
    const std::vector<double> counts{18, 17, 15, 20, 10, 20, 25, 13, 12};
    Rectangular X(9, 5);
    for (std::size_t i = 0; i < 9; i++) {
      const std::size_t outcome = i % 3, treatment = i / 3;
      X(i, 0) = 1;
      X(i, 1) = outcome == 1;
      X(i, 2) = outcome == 2;
      X(i, 3) = treatment == 1;
      X(i, 4) = treatment == 2;
    }
    GlmOptions o;
    o.family = Family::poisson;
    const GlmFit g = glm(X, counts, o);
    s.require("poisson (Dobson): IRLS converges", g.converged, std::to_string(g.iterations) + " iterations");
    const double want_b[5] = {std::log(21.0), std::log(40.0 / 63), std::log(47.0 / 63), 0, 0};
    const double want_se[5] = {std::sqrt(1.0 / 63 + 1.0 / 50 - 1.0 / 150), std::sqrt(1.0 / 63 + 1.0 / 40), std::sqrt(1.0 / 63 + 1.0 / 47), std::sqrt(0.04), std::sqrt(0.04)};
    double eb = 0, es = 0;
    for (int j = 0; j < 5; j++) {
      eb = std::max(eb, std::fabs(g.beta[j] - want_b[j]));
      es = std::max(es, relative(g.se[j], want_se[j]));
    }
    s.check("poisson (Dobson): β in closed form, the independence model's", eb, 1e-12);
    s.check("poisson (Dobson): standard errors in closed form, relative", es, 1e-11);
    s.check("poisson (Dobson): residual deviance 5.1291 (R)", g.deviance - 5.1291, 5e-5);
    s.check("poisson (Dobson): null deviance 10.5814 (R)", g.null_deviance - 10.5814, 5e-5);
    s.check("poisson (Dobson): AIC 56.761 (R)", g.aic - 56.761, 5e-4);
    s.require("poisson (Dobson): z statistics against the normal", !g.uses_t && g.df_residual == 4);
    double score = 0;
    for (std::size_t j = 0; j < 5; j++) {
      double v = 0;
      for (std::size_t i = 0; i < 9; i++) v += X(i, j) * (counts[i] - g.fitted[i]);
      score = std::max(score, std::fabs(v));
    }
    s.check("poisson: the score equations Σ xᵢ(yᵢ − μᵢ) = 0 at the MLE", score, 1e-9);
  }

  // ---- Bliss's beetles (Dobson and Barnett 2008, §7.3) ----
  {
    const std::vector<double> dose{1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839};
    const std::vector<double> n{59, 60, 62, 56, 63, 59, 62, 60}, dead{6, 13, 18, 28, 52, 53, 61, 60};
    std::vector<double> prop(8);
    for (int i = 0; i < 8; i++) prop[i] = dead[i] / n[i];
    const Rectangular X = design_matrix({dose});
    GlmOptions o;
    o.family = Family::binomial;
    o.weights = n;
    const GlmFit lg = glm(X, prop, o);
    // A value as printed to two decimals: round(100 v)/100, a correctly rounded quotient, so that
    // it equals the literal exactly.
    auto at = [](double v, double) { return std::round(v * 100) / 100; };
    s.require("binomial logit (beetles): b = (−60.72, 34.27) as Dobson and Barnett print", at(lg.beta[0], 0.01) == -60.72 && at(lg.beta[1], 0.01) == 34.27);
    s.require("binomial logit: standard errors (5.18, 2.91)", at(lg.se[0], 0.01) == 5.18 && at(lg.se[1], 0.01) == 2.91);
    s.require("binomial logit: deviance 11.23", at(lg.deviance, 0.01) == 11.23);
    s.check("binomial logit: β against statsmodels (cross-check), relative",
            std::max(relative(lg.beta[0], -60.71745456163593), relative(lg.beta[1], 34.270325734147264)), 1e-8);
    s.check("binomial logit: standard errors against statsmodels, relative", std::max(relative(lg.se[0], 5.180711461309134), relative(lg.se[1], 2.9121400694821045)), 1e-7);
    s.check("binomial logit: deviance 11.232231097419 and null deviance 284.202449480831 (statsmodels)",
            std::max(relative(lg.deviance, 11.232231097419255), relative(lg.null_deviance, 284.20244948083143)), 1e-9);
    s.check("binomial logit: log-likelihood with the binomial coefficients, −18.7151346573 (statsmodels)", relative(lg.log_likelihood, -18.71513465725598), 1e-9);
    s.check("binomial logit: AIC = −2ℓ + 2r = 41.4302693145", relative(lg.aic, 41.43026931451196), 1e-9);
    s.check("binomial logit: McFadden's pseudo-R² = 1 − ℓ/ℓ₀", lg.pseudo_r_squared - (1 - lg.log_likelihood / lg.null_log_likelihood), 1e-15);
    o.link = Link::probit;
    const GlmFit pr = glm(X, prop, o);
    s.require("binomial probit: b = (−34.94, 19.73), se (2.65, 1.49), deviance 10.12 as printed",
              at(pr.beta[0], 0.01) == -34.94 && at(pr.beta[1], 0.01) == 19.73 && at(pr.se[0], 0.01) == 2.65 && at(pr.se[1], 0.01) == 1.49 && at(pr.deviance, 0.01) == 10.12);
    s.check("binomial probit: β against statsmodels, relative", std::max(relative(pr.beta[0], -34.935258899178), relative(pr.beta[1], 19.727934211322342)), 1e-8);
    s.check("binomial probit: standard errors (expected information) against statsmodels", std::max(relative(pr.se[0], 2.6479177986178843), relative(pr.se[1], 1.4872350409027706)), 1e-7);
    s.check("binomial probit: deviance 10.1197581130", relative(pr.deviance, 10.119758113001495), 1e-9);
  }

  // ---- gaussian identity = least squares (NIST Norris) ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/nist-norris.dat");
    std::vector<double> x, y;
    std::string line;
    for (int i = 1; std::getline(in, line); i++) {
      if (i < 61) continue;
      double a, b;
      if (std::sscanf(line.c_str(), "%lf %lf", &a, &b) == 2) {
        y.push_back(a);
        x.push_back(b);
      }
    }
    const Rectangular X = design_matrix({x});
    const GlmFit g = glm(X, y);
    const LinearFit f = ols(X, y);
    s.check("gaussian identity: β = least squares (Norris)", std::max(relative(g.beta[0], f.beta[0]), relative(g.beta[1], f.beta[1])), 1e-12);
    s.check("gaussian identity: se with φ̂ = RSS/(n − r), as ols", std::max(relative(g.se[0], f.se[0]), relative(g.se[1], f.se[1])), 1e-12);
    s.check("gaussian identity: deviance = RSS", relative(g.deviance, f.rss), 1e-12);
    s.check("gaussian identity: the log-likelihood of ols", relative(g.log_likelihood, f.log_likelihood), 1e-12);
    s.require("gaussian identity: t statistics on n − r degrees of freedom", g.uses_t && g.df_inference == 34);
  }

  // ---- PPML: a gravity equation with heteroskedastic multiplicative errors ----
  {
    calx::random::Stream rng(20261024, 41, 0, 0);
    const std::size_t n = 4000;
    std::vector<double> ld(n), lgdp(n), y(n), logy(n), exposure(n), loge(n);
    const double b0 = 2.0, b_dist = -1.0, b_gdp = 0.8;
    for (std::size_t i = 0; i < n; i++) {
      ld[i] = rng.uniform(0, 3);
      lgdp[i] = rng.normal();
      const double mu = std::exp(b0 + b_dist * ld[i] + b_gdp * lgdp[i]);
      // η log-normal with E η = 1 and Var η = 1/μ: Var(y|x) ∝ μ (SST 2006, case 2).
      const double s2 = std::log1p(1 / mu);
      const double eta = std::exp(rng.normal(-0.5 * s2, std::sqrt(s2)));
      y[i] = mu * eta;
      logy[i] = std::log(y[i]);
    }
    const Rectangular X = design_matrix({ld, lgdp});
    GlmOptions o;
    o.family = Family::poisson;
    o.covariance = GlmCovariance::hc0;
    const GlmFit g = glm(X, y, o);
    const double zp = std::max(std::fabs(g.beta[1] - b_dist) / g.se[1], std::fabs(g.beta[2] - b_gdp) / g.se[2]);
    s.check("ppml: the elasticities recovered within 4 robust standard errors (largest |β̂ − β|/se)", zp, 4);
    OlsOptions oo;
    oo.covariance = Covariance::hc1;
    const LinearFit f = ols(X, logy, oo);
    s.require("ppml: least squares on logarithms is biased here (its distance elasticity > 6 se off; Santos Silva and Tenreyro 2006)",
              std::fabs(f.beta[1] - b_dist) / f.se[1] > 6, std::to_string(f.beta[1]));
    // An offset: exposure scaled by 2 moves only the intercept, by −log 2.
    std::vector<double> off1(n), off2(n);
    for (std::size_t i = 0; i < n; i++) {
      off1[i] = 0.1 * static_cast<double>(i % 7);
      off2[i] = off1[i] + std::log(2.0);
    }
    o.offset = off1;
    const GlmFit a = glm(X, y, o);
    o.offset = off2;
    const GlmFit b = glm(X, y, o);
    s.check("poisson offset: doubling every exposure moves the intercept by −log 2", b.beta[0] - a.beta[0] + std::log(2.0), 1e-10);
    s.check("poisson offset: and leaves the slopes", std::fabs(b.beta[1] - a.beta[1]) + std::fabs(b.beta[2] - a.beta[2]), 1e-10);
    // Clusters of one reproduce hc1.
    GlmOptions oc;
    oc.family = Family::poisson;
    oc.covariance = GlmCovariance::cluster;
    for (std::size_t i = 0; i < n; i++) oc.clusters.push_back(static_cast<std::int64_t>(i));
    GlmOptions oh;
    oh.family = Family::poisson;
    oh.covariance = GlmCovariance::hc1;
    s.check("glm cluster: one observation per cluster reproduces hc1", relative(glm(X, y, oc).covariance(1, 1), glm(X, y, oh).covariance(1, 1)), 1e-12);
  }
  return s.done();
}
