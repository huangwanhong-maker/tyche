// Checks of tyche/regression/variance_components.hpp.
//
// - Against R's regress 1.3.22 (fixtures/variance_components.txt, made by
//   make_variance_components_fixtures.py with R 4.4.3), on seeded genotypes (200 samples, 400 sites) and
//   their standardised Gram matrices: one kernel (y1, σ_g² = σ_e² = 0.5 simulated) by the spectral
//   REML and by AI-REML; two kernels and the residual (y2) by AI-REML. The components equal regress's
//   REML estimates, the fixed effects its GLS estimates, ℓ_R at the estimate R's own evaluation of
//   Harville's restricted likelihood there, and the standard errors regress's (inverse expected
//   information; AI⁻¹ estimates the same, so within a few per cent).
// - The spectral REML's estimate is the maximum of its own profile of ℓ_R over h² ∈ [0, 1), and AI-REML
//   reaches the same point from GCTA's start.
// - Recovery of the truth: over 60 simulated samples (n = 300, m = 600 sites, h² = 0.4, the genetic
//   values g = Zβ with β ~ N(0, h²/m), so Var g = h² K exactly), REML's ĥ² has mean 0.4 within its
//   standard error, and its reported standard error matches the spread of the estimates; Haseman and
//   Elston's ĥ² too, its model-based standard error (the Gaussian quadratic form's) matching its spread,
//   its jackknife's exceeding it (conservative for this nearly degenerate U-statistic), and REML the
//   more precise.

#include <tyche/regression/variance_components.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tr = tyche::regression;
using calx::linalg::Matrix;
using calx::linalg::Rectangular;

namespace {

std::map<std::string, std::vector<double>> read_fixture(const std::string& path) {
  std::map<std::string, std::vector<double>> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto colon = line.find(" :");
    if (colon == std::string::npos) continue;
    std::istringstream values(line.substr(colon + 2));
    std::vector<double> v;
    std::string token;
    while (values >> token) {
      try {
        v.push_back(std::stod(token));
      } catch (...) {
        break;
      }
    }
    out[line.substr(0, colon)] = v;
  }
  return out;
}

// The standardised Gram matrix of the sites [first, last) of an n × m dosage matrix (row-major).
Matrix gram(const std::vector<double>& G, std::size_t n, std::size_t m, std::size_t first, std::size_t last) {
  Matrix K(n);
  for (std::size_t j = first; j < last; j++) {
    double p = 0;
    for (std::size_t i = 0; i < n; i++) p += G[i * m + j];
    p /= 2.0 * static_cast<double>(n);
    const double sd = std::sqrt(2 * p * (1 - p));
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; i++) z[i] = (G[i * m + j] - 2 * p) / sd;
    for (std::size_t a = 0; a < n; a++)
      for (std::size_t b = 0; b <= a; b++) K(a, b) += z[a] * z[b];
  }
  const double c = static_cast<double>(last - first);
  for (std::size_t a = 0; a < n; a++)
    for (std::size_t b = 0; b <= a; b++) K(b, a) = K(a, b) = K(a, b) / c;
  return K;
}

double rel(double a, double b) { return std::fabs(a - b) / std::fabs(b); }

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto F = read_fixture(std::string(TYCHE_FIXTURES) + "/variance_components.txt");
  s.require("the fixture of regress's fits is read", F.count("G") && F.count("one_sigma") && F.count("two_sigma"));
  const auto n = static_cast<std::size_t>(F.at("dims")[0]), m = static_cast<std::size_t>(F.at("dims")[1]);
  const auto& G = F.at("G");
  const auto& x = F.at("x");
  const Rectangular X = tr::design_matrix({x}, true);
  const Matrix K = gram(G, n, m, 0, m), K1 = gram(G, n, m, 0, 200), K2 = gram(G, n, m, 200, m);

  // ---- one kernel ----
  {
    const auto& ref = F.at("one_sigma");
    const auto& cov = F.at("one_sigma_cov");
    const auto sp = tr::reml_spectral(X, F.at("y1"), K);
    s.check("spectral REML: σ_g² equals regress's (relative)", rel(sp.sigma2[0], ref[0]), 1e-6);
    s.check("and σ_e²", rel(sp.sigma2[1], ref[1]), 1e-6);
    s.check("ℓ_R at the estimate equals R's evaluation of Harville's restricted likelihood there", std::fabs(sp.log_likelihood - F.at("one_reml")[0]), 1e-7);
    s.check("the GLS fixed effects equal regress's", std::max(rel(sp.beta[0], F.at("one_beta")[0]), rel(sp.beta[1], F.at("one_beta")[1])), 1e-6);
    s.check("the standard error of σ_g² (AI) is regress's (expected information) within 5%", rel(sp.se[0], std::sqrt(cov[0])), 0.05);
    s.check("and of σ_e²", rel(sp.se[1], std::sqrt(cov[3])), 0.05);
    double best = -INFINITY;
    for (const double l : sp.profile_log_likelihood) best = std::max(best, l);
    s.require("the estimate is the maximum of ℓ_R's profile over h²", sp.log_likelihood >= best - 1e-12);
    tr::RemlOptions tight;
    tight.tolerance = 1e-10;
    const auto ai = tr::reml(X, F.at("y1"), {K}, tight);
    s.check("AI-REML from GCTA's start, run to a gain of 10⁻¹⁰, reaches the spectral estimate: h² (absolute)", std::fabs(ai.h2 - sp.h2), 1e-6);
    s.check("its σ_g² equals regress's", rel(ai.sigma2[0], ref[0]), 1e-4);
    s.check("its ℓ_R equals the spectral one's", std::fabs(ai.log_likelihood - sp.log_likelihood), 1e-6);
    s.check("its standard error of h² equals the spectral one's", rel(ai.h2_se, sp.h2_se), 1e-3);
    s.require("and it converged", ai.converged, "iterations " + std::to_string(ai.iterations));
  }
  // ---- two kernels ----
  {
    const auto& ref = F.at("two_sigma");
    const auto& cov = F.at("two_sigma_cov");
    const auto ai = tr::reml(X, F.at("y2"), {K1, K2});
    double worst = 0, worst_se = 0;
    for (std::size_t i = 0; i < 3; i++) {
      worst = std::max(worst, rel(ai.sigma2[i], ref[i]));
      worst_se = std::max(worst_se, rel(ai.se[i], std::sqrt(cov[i * 3 + i])));
    }
    s.check("AI-REML with two kernels: σ₁², σ₂², σ_e² equal regress's (largest relative difference)", worst, 1e-4);
    s.check("ℓ_R at the estimate equals R's evaluation", std::fabs(ai.log_likelihood - F.at("two_reml")[0]), 1e-5);
    s.check("the fixed effects equal regress's", std::max(rel(ai.beta[0], F.at("two_beta")[0]), rel(ai.beta[1], F.at("two_beta")[1])), 1e-5);
    s.check("the standard errors are regress's within 5%", worst_se, 0.05);
  }
  // ---- recovery of a simulated h² ----
  {
    const std::size_t R = 60, nn = 300, mm = 600;
    const double h2 = 0.4;
    double sum = 0, sum2 = 0, se_sum = 0, he_sum = 0, he_sum2 = 0, he_se = 0, he_jk = 0;
    for (std::size_t r = 0; r < R; r++) {
      calx::random::Stream g(41, 0x52454d4c, static_cast<std::uint32_t>(r), 0);
      std::vector<double> Gs(nn * mm);
      std::vector<double> f(mm);
      for (double& v : f) v = g.uniform(0.05, 0.5);
      for (std::size_t i = 0; i < nn; i++)
        for (std::size_t j = 0; j < mm; j++) Gs[i * mm + j] = static_cast<double>(g.binomial(2, f[j]));
      bool ok = true;
      for (std::size_t j = 0; j < mm && ok; j++) {
        double c = 0;
        for (std::size_t i = 0; i < nn; i++) c += Gs[i * mm + j];
        ok = c > 0 && c < 2.0 * nn;
      }
      if (!ok) continue;
      const Matrix Ks = gram(Gs, nn, mm, 0, mm);
      // g = Zβ, β ~ N(0, h²/m): Var g = h² ZZᵀ/m = h² K.
      std::vector<double> y(nn, 0.0), cov(nn);
      for (std::size_t j = 0; j < mm; j++) {
        double p = 0;
        for (std::size_t i = 0; i < nn; i++) p += Gs[i * mm + j];
        p /= 2.0 * nn;
        const double sd = std::sqrt(2 * p * (1 - p)), beta = g.normal() * std::sqrt(h2 / mm);
        for (std::size_t i = 0; i < nn; i++) y[i] += beta * (Gs[i * mm + j] - 2 * p) / sd;
      }
      for (std::size_t i = 0; i < nn; i++) {
        cov[i] = g.normal();
        y[i] += 2 + 0.3 * cov[i] + g.normal() * std::sqrt(1 - h2);
      }
      const Rectangular Xs = tr::design_matrix({cov}, true);
      const auto e = tr::reml_spectral(Xs, y, Ks);
      sum += e.h2;
      sum2 += e.h2 * e.h2;
      se_sum += e.h2_se;
      const auto he = tr::haseman_elston(Ks, y, Xs);
      he_sum += he.h2;
      he_sum2 += he.h2 * he.h2;
      he_se += he.se;
      he_jk += he.se_jackknife;
    }
    const double mean = sum / R, sd = std::sqrt(sum2 / R - mean * mean), mse = se_sum / R;
    const double hm = he_sum / R, hsd = std::sqrt(he_sum2 / R - hm * hm), hse = he_se / R, hjk = he_jk / R;
    s.check("REML recovers h² = 0.4 over 60 samples: (mean ĥ² − 0.4)/(sd/√60)", (mean - h2) / (sd / std::sqrt(static_cast<double>(R))), 3, "SE");
    s.check("REML's standard error matches the spread of its estimates: mean SE/sd − 1", mse / sd - 1, 0.3);
    s.check("Haseman–Elston recovers h² too: (mean − 0.4)/(sd/√60)", (hm - h2) / (hsd / std::sqrt(static_cast<double>(R))), 3, "SE");
    s.check("its model-based standard error (the delta method on the Gaussian quadratic form) matches its spread: mean SE/sd − 1", hse / hsd - 1, 0.3);
    s.require("its jackknife standard error is conservative (mean " + calx::verification::exponential(hjk) + " ≥ sd " + calx::verification::exponential(hsd) + ")", hjk >= hsd);
    s.require("and REML is the more precise (sd " + calx::verification::exponential(sd) + " against " + calx::verification::exponential(hsd) + ")", sd <= hsd * 1.05);
  }
  return s.done();
}
