// Checks of tyche/testing/permutation.hpp.
//
// - The permutation p-value (b + 1)/(B + 1) in closed form, and its size: under the null, P(p ≤ α) ≤ α
//   exactly (α = 0.05, 2000 observations of uniform rank).
// - The correlation's p-value against Student's t on ν degrees of freedom (tyche's StudentT), and both
//   against SciPy's t.sf where the tail is small and ν large (to 10⁻¹⁷³).
// - The beta fit against SciPy (fixtures/permutation.txt, made by make_permutation_fixtures.py with SciPy
//   1.18.1): scipy.stats.beta.fit(x, floc = 0, fscale = 1) on 2000 draws of Beta(0.9, 37); the score
//   equations at the estimate; and the closed form of independent tests: the least of m uniforms is
//   Beta(1, m), recovered from 4000 minima (m = 50).
// - The effective degrees of freedom against tensorQTL's df_cost solved by SciPy on 1000 seeded maxima
//   of 40 correlated squared correlations.
// - The beta approximation against direct permutations: for correlated tests (40 AR(1) predictors,
//   ρ = 0.8, n = 120), the adjusted p-value of observed minima from 1000 permutations equals the
//   fraction of 20 000 further permutations whose minimum is smaller, within binomial error, over a
//   range of p from 10⁻³ to 0.5; and under the null the adjusted p-values are uniform (size at 0.05
//   and 0.2 over 300 null phenotypes).

#include <tyche/probability/distributions.hpp>
#include <tyche/testing/permutation.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tt = tyche::testing;

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
    double x;
    while (values >> x) v.push_back(x);
    out[line.substr(0, colon)] = v;
  }
  return out;
}

// A set of m AR(1)-correlated predictors over n samples, standardised (mean 0, Σ x² = n).
std::vector<double> correlated_predictors(std::size_t n, std::size_t m, double rho, std::uint64_t seed) {
  std::vector<double> g(n * m);
  calx::random::Stream r(seed, 0x50524544, 0, 0);
  for (std::size_t i = 0; i < n; i++) {
    double prev = r.normal();
    g[i * m] = prev;
    for (std::size_t j = 1; j < m; j++) g[i * m + j] = prev = rho * prev + std::sqrt(1 - rho * rho) * r.normal();
  }
  for (std::size_t j = 0; j < m; j++) {
    double mean = 0, ss = 0;
    for (std::size_t i = 0; i < n; i++) mean += g[i * m + j];
    mean /= static_cast<double>(n);
    for (std::size_t i = 0; i < n; i++) ss += (g[i * m + j] - mean) * (g[i * m + j] - mean);
    const double sd = std::sqrt(ss / static_cast<double>(n));
    for (std::size_t i = 0; i < n; i++) g[i * m + j] = (g[i * m + j] - mean) / sd;
  }
  return g;
}

// The largest squared correlation between y (standardised in place) and the predictors.
double max_r2(const std::vector<double>& g, std::size_t n, std::size_t m, const std::vector<double>& y) {
  double best = 0;
  for (std::size_t j = 0; j < m; j++) {
    double s = 0;
    for (std::size_t i = 0; i < n; i++) s += g[i * m + j] * y[i];
    const double r = s / static_cast<double>(n);
    best = std::max(best, r * r);
  }
  return best;
}

void standardise(std::vector<double>& y) {
  double mean = 0, ss = 0;
  for (const double v : y) mean += v;
  mean /= static_cast<double>(y.size());
  for (const double v : y) ss += (v - mean) * (v - mean);
  const double sd = std::sqrt(ss / static_cast<double>(y.size()));
  for (double& v : y) v = (v - mean) / sd;
}

void shuffle(std::vector<double>& y, calx::random::Stream& r) {
  for (std::size_t i = y.size(); i > 1; i--) std::swap(y[i - 1], y[r.below(i)]);
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto F = read_fixture(std::string(TYCHE_FIXTURES) + "/permutation.txt");
  s.require("the fixture of SciPy's values is read", F.count("beta_sample") && F.at("beta_sample").size() == 2000 && F.count("r2_sample"));

  // ---- permutation p-values ----
  s.require("(b + 1)/(B + 1): 0 of 999 gives 1/1000, 999 of 999 gives 1", tt::permutation_p_value(0, 999) == 0.001 && tt::permutation_p_value(999, 999) == 1);
  {
    // An observation exchangeable with B = 19 null draws has a uniform rank; P(p ≤ 0.05) = 1/20 exactly.
    calx::random::Stream r(3, 0x5045524d, 0, 0);
    std::size_t hits = 0;
    const std::size_t N = 20000;
    for (std::size_t k = 0; k < N; k++) {
      const double obs = r.uniform();
      std::size_t b = 0;
      for (int t = 0; t < 19; t++) b += r.uniform() >= obs;
      hits += tt::permutation_p_value(b, 19) <= 0.05;
    }
    s.check("its size: P(p ≤ 0.05) − 0.05 under the null (B = 19, 20000 observations; SE 0.0015)", static_cast<double>(hits) / N - 0.05, 0.005);
  }
  {
    double worst = 0;
    for (const double df : {5.0, 30.0, 96.0})
      for (const double t : {0.3, 1.5, 4.0, 9.0}) {
        const double r2 = t * t / (t * t + df);
        const double ref = 2 * tyche::probability::StudentT{df}.survival(t);
        worst = std::max(worst, std::fabs(tt::correlation_p_value(r2, df) - ref) / ref);
      }
    s.check("the correlation's p-value is Student's two-sided tail at t = √(ν r²/(1 − r²)) (largest relative difference)", worst, 1e-12);
    // Student's upper tail keeps its relative accuracy where it is small and ν large (tyche's StudentT,
    // corrected with this module: it had switched formulas at t² = ν and lost tails below 10⁻¹⁶).
    double worst_tail = 0, worst_corr = 0;
    for (int k = 0; k < 4; k++) {
      const auto& v = F.at("student_tail" + std::to_string(k));
      const double df = v[0], t = v[1], ref = v[2];
      worst_tail = std::max(worst_tail, std::fabs(tyche::probability::StudentT{df}.survival(t) - ref) / ref);
      worst_corr = std::max(worst_corr, std::fabs(tt::correlation_p_value(t * t / (t * t + df), df) / 2 - ref) / ref);
    }
    s.check("Student's upper tail at (ν, t) = (96, 9), (3000, 10), (3000, 30), (500, 4) equals SciPy's down to 10⁻¹⁷³ (relative)", worst_tail, 1e-12);
    s.check("and so does the correlation's p-value there", worst_corr, 1e-12);
  }

  // ---- the beta fit ----
  {
    const auto& x = F.at("beta_sample");
    const auto f = tt::fit_beta(x);
    const auto& ref = F.at("beta_fit");
    s.check("the beta's first shape equals SciPy's maximum-likelihood fit (relative)", std::fabs(f.a - ref[0]) / ref[0], 1e-7);
    s.check("and its second shape", std::fabs(f.b - ref[1]) / ref[1], 1e-7);
    double sl = 0, sl1 = 0;
    for (const double v : x) {
      sl += std::log(v);
      sl1 += std::log1p(-v);
    }
    const double N = static_cast<double>(x.size());
    namespace sf = calx::special_functions;
    const double ga = sl - N * (sf::digamma(f.a) - sf::digamma(f.a + f.b)), gb = sl1 - N * (sf::digamma(f.b) - sf::digamma(f.a + f.b));
    s.check("the score equations vanish at the estimate (|∂ℓ/∂a| + |∂ℓ/∂b|)", std::fabs(ga) + std::fabs(gb), 1e-6);
    s.require("Newton's method converged", f.converged);
  }
  {
    // The least of m independent uniforms is Beta(1, m).
    const std::size_t m = 50, N = 4000;
    std::vector<double> minima(N);
    calx::random::Stream r(5, 0x4d494e53, 0, 0);
    for (std::size_t k = 0; k < N; k++) {
      double low = 1;
      for (std::size_t j = 0; j < m; j++) low = std::min(low, r.uniform());
      minima[k] = low;
    }
    const auto f = tt::fit_beta(minima);
    // Fisher information of Beta(1, 50) per draw: the asymptotic SEs of â and b̂ are about 0.019 and 1.2.
    s.check("the least of 50 independent uniforms: â − 1 (4000 minima; SE about 0.02)", f.a - 1, 0.08);
    s.check("and b̂/50 − 1 (SE about 0.024)", f.b / 50 - 1, 0.1);
  }

  // ---- the effective degrees of freedom ----
  {
    const double df = tt::effective_df(F.at("r2_sample"), 96);
    s.check("the effective degrees of freedom equal tensorQTL's df_cost root by SciPy (relative)", std::fabs(df - F.at("effective_df")[0]) / F.at("effective_df")[0], 1e-6);
  }

  // ---- the beta approximation against direct permutations ----
  {
    const std::size_t n = 120, m = 40, B = 1000, B2 = 20000;
    const auto g = correlated_predictors(n, m, 0.8, 11);
    // Observed phenotypes with signals of graded strength on predictor 17, so that p spans 10⁻³ … 0.5.
    std::size_t within = 0, cases = 0;
    double worst_z = 0;
    for (std::size_t c = 0; c < 8; c++) {
      calx::random::Stream r(13, 0x4f425356, static_cast<std::uint32_t>(c), 0);
      std::vector<double> y(n);
      const double beta = 0.04 * static_cast<double>(c);
      for (std::size_t i = 0; i < n; i++) y[i] = beta * g[i * m + 17] + r.normal();
      standardise(y);
      const double obs = max_r2(g, n, m, y);
      std::vector<double> perm_p(B);
      std::vector<double> z = y;
      for (std::size_t b = 0; b < B; b++) {
        shuffle(z, r);
        perm_p[b] = tt::correlation_p_value(max_r2(g, n, m, z), static_cast<double>(n - 2));
      }
      const auto fit = tt::fit_beta(perm_p);
      const double adjusted = fit.cdf(tt::correlation_p_value(obs, static_cast<double>(n - 2)));
      std::size_t more = 0;
      for (std::size_t b = 0; b < B2; b++) {
        shuffle(z, r);
        more += max_r2(g, n, m, z) >= obs;
      }
      const double direct = (static_cast<double>(more) + 1) / (static_cast<double>(B2) + 1);
      // The fit itself is uncertain from B = 1000 draws; allow the direct estimate's binomial error and
      // a relative 25% for the fit's.
      const double se = std::sqrt(direct * (1 - direct) / B2) + 0.25 * direct;
      const double zscore = std::fabs(adjusted - direct) / se;
      worst_z = std::max(worst_z, zscore);
      within += zscore <= 2;
      cases++;
    }
    s.check("the beta-adjusted p of 8 observed minima equals 20000 direct permutations' within their error (largest |difference|/error)", worst_z, 2.5);
    s.require("at every strength of signal from p ≈ 0.5 down to ≈ 10⁻³", within + 1 >= cases);
  }
  {
    // Calibration under the null: 300 phenotypes, 500 permutations each.
    const std::size_t n = 80, m = 30, B = 500, R = 300;
    const auto g = correlated_predictors(n, m, 0.7, 17);
    std::size_t below05 = 0, below20 = 0;
    for (std::size_t c = 0; c < R; c++) {
      calx::random::Stream r(19, 0x4e554c4c, static_cast<std::uint32_t>(c), 0);
      std::vector<double> y(n);
      for (double& v : y) v = r.normal();
      standardise(y);
      const double obs = max_r2(g, n, m, y);
      std::vector<double> perm_p(B);
      std::vector<double> z = y;
      for (std::size_t b = 0; b < B; b++) {
        shuffle(z, r);
        perm_p[b] = tt::correlation_p_value(max_r2(g, n, m, z), static_cast<double>(n - 2));
      }
      const double adjusted = tt::fit_beta(perm_p).cdf(tt::correlation_p_value(obs, static_cast<double>(n - 2)));
      below05 += adjusted <= 0.05;
      below20 += adjusted <= 0.2;
    }
    s.check("under the null the adjusted p-values are uniform: P(p ≤ 0.05) − 0.05 (300 phenotypes; SE 0.013)", static_cast<double>(below05) / R - 0.05, 0.04);
    s.check("and P(p ≤ 0.2) − 0.2 (SE 0.023)", static_cast<double>(below20) / R - 0.2, 0.07);
  }
  return s.done();
}
