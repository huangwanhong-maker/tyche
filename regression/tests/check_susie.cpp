// Checks of tyche/regression/susie.hpp.
//
// - Against the authors' R package susieR 0.14.2 (fixtures/susie.txt, made by make_susie_fixtures.py
//   with R 4.4.3), on seeded genotype-like data (400 samples, 80 columns in strong linkage, three
//   non-zero effects): susie() with its defaults (L = 10), with the EM step for the prior variances
//   (L = 5), and the single-effect regression with fixed variances (L = 1); susie_suff_stat on the
//   centred data's XᵀX, Xᵀy and yᵀy; susie_rss on z-scores and cor(X). In each: α, μ and μ² of every
//   effect, the prior variances, the residual variance, the ELBO at every iteration and their number,
//   the PIPs, the effects' log Bayes factors, the credible sets (which effects, which members, their
//   purity and coverage), and for susie() the intercept, the coefficients and the fitted values.
// - Closed forms: the single-effect regression's posterior is exact, equal to the enumeration of its p
//   one-column models, whose marginal likelihoods are written by the matrix determinant lemma
//   (log N(y; 0, σ²I + σ₀² x xᵀ) − log N(y; 0, σ²I)) and whose posterior means by Sherman–Morrison.
// - The ELBO never falls from one iteration to the next.
// - Calibration: over 300 simulated regions (n = 300, p = 60 in linkage, one to three effects of 0.4–0.7
//   per allele against a unit noise), the
//   95% credible sets contain an effect at least 90% of the time (the coverage the authors report,
//   §5.1 of Wang et al. 2020, is near 0.95), and the PIPs are calibrated in bins.

#include <tyche/regression/susie.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace tr = tyche::regression;

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

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return INFINITY;
  double w = 0;
  for (std::size_t i = 0; i < a.size(); i++) w = std::max(w, std::fabs(a[i] - b[i]));
  return w;
}

double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b, double floor) {
  if (a.size() != b.size()) return INFINITY;
  double w = 0;
  for (std::size_t i = 0; i < a.size(); i++) w = std::max(w, std::fabs(a[i] - b[i]) / std::max(floor, std::fabs(b[i])));
  return w;
}

// Compares a fit with susieR's under the fixture's tag.
void compare(calx::verification::Suite& s, const std::map<std::string, std::vector<double>>& F, const std::string& tag, const std::string& what,
             const tr::SusieFit& f) {
  const auto get = [&](const std::string& k) { return F.at(tag + "_" + k); };
  const auto niter = static_cast<int>(get("niter")[0]);
  s.require(what + ": the same number of iterations as susieR (" + std::to_string(niter) + ")", f.iterations == niter,
            "tyche " + std::to_string(f.iterations));
  s.check(what + ": every effect's α equals susieR's (largest difference)", max_abs_diff(f.alpha, get("alpha")), 1e-6);
  s.check(what + ": μ (largest difference relative to max(|μ|, 10⁻³))", max_rel_diff(f.mu, get("mu"), 1e-3), 1e-5);
  s.check(what + ": μ²", max_rel_diff(f.mu2, get("mu2"), 1e-6), 1e-5);
  s.check(what + ": the prior variances σ₀ₗ²", max_rel_diff(f.prior_variance, get("V"), 1e-6), 1e-5);
  s.check(what + ": the residual variance", std::fabs(f.residual_variance - get("sigma2")[0]) / get("sigma2")[0], 1e-9);
  // susieR's elbo vector starts with −Inf before the first iteration.
  std::vector<double> elbo = get("elbo");
  if (!elbo.empty() && !std::isfinite(elbo[0])) elbo.erase(elbo.begin());
  s.check(what + ": the ELBO at every iteration", max_rel_diff(f.elbo, elbo, 1), 1e-9);
  s.check(what + ": the PIPs", max_abs_diff(f.pip, get("pip")), 1e-6);
  s.check(what + ": the effects' log Bayes factors", max_rel_diff(f.lbf, get("lbf"), 1), 1e-6);
  const auto count = static_cast<std::size_t>(get("cs_count")[0]);
  bool same = f.sets.size() == count;
  for (std::size_t k = 0; same && k < count; k++) {
    std::vector<double> mine;
    for (const auto j : f.sets[k].variables) mine.push_back(static_cast<double>(j));
    std::vector<double> ref = get("cs" + std::to_string(k));
    std::sort(mine.begin(), mine.end());
    std::sort(ref.begin(), ref.end());
    same = mine == ref && static_cast<double>(f.sets[k].effect) == get("cs_index")[k] &&
           std::fabs(f.sets[k].min_abs_corr - get("cs_min_abs_corr")[k]) < 1e-9 && std::fabs(f.sets[k].mean_abs_corr - get("cs_mean_abs_corr")[k]) < 1e-9 &&
           std::fabs(f.sets[k].coverage - get("cs_coverage")[k]) < 1e-6;
  }
  s.require(what + ": the same " + std::to_string(count) + " credible sets, effects, members, purity and coverage", same,
            "tyche found " + std::to_string(f.sets.size()));
}

// Genotype-like columns: dosages from two thresholded AR(1) Gaussian haplotypes per sample.
calx::linalg::Rectangular genotypes(std::size_t n, std::size_t p, double rho, calx::random::Stream& r) {
  calx::linalg::Rectangular X(n, p);
  std::vector<double> f(p);
  for (double& v : f) v = r.uniform(0.1, 0.5);
  for (int copy = 0; copy < 2; copy++)
    for (std::size_t i = 0; i < n; i++) {
      double z = r.normal();
      for (std::size_t j = 0; j < p; j++) {
        if (j) z = rho * z + std::sqrt(1 - rho * rho) * r.normal();
        // P(z > t) = f: t = Φ⁻¹(1 − f), by a rational approximation good to 10⁻³ (enough to set frequencies).
        const double q = 1 - f[j], tq = std::sqrt(-2 * std::log(std::min(q, 1 - q)));
        double t = tq - (2.515517 + 0.802853 * tq + 0.010328 * tq * tq) / (1 + 1.432788 * tq + 0.189269 * tq * tq + 0.001308 * tq * tq * tq);
        if (q < 0.5) t = -t;
        X(i, j) += z > t ? 1 : 0;
      }
    }
  return X;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto F = read_fixture(std::string(TYCHE_FIXTURES) + "/susie.txt");
  s.require("the fixture of susieR's fits is read", F.count("dims") && F.count("ind_alpha") && F.count("rss_alpha"));
  const auto n = static_cast<std::size_t>(F.at("dims")[0]), p = static_cast<std::size_t>(F.at("dims")[1]);
  calx::linalg::Rectangular X(n, p);
  X.a = F.at("X");
  const std::vector<double>& y = F.at("y");

  // ---- against susieR ----
  {
    tr::SusieOptions o;
    const auto f = tr::susie(X, y, o);
    compare(s, F, "ind", "susie(X, y, L = 10)", f);
    s.check("susie: the intercept equals susieR's", std::fabs(f.intercept - F.at("ind_intercept")[0]), 1e-7);
    s.check("susie: the coefficients on X's scale", max_rel_diff(f.coef, F.at("ind_coef"), 1e-3), 1e-5);
    s.check("susie: the fitted values", max_abs_diff(f.fitted, F.at("ind_fitted")), 1e-7);
    bool rising = true;
    for (std::size_t i = 1; i < f.elbo.size(); i++) rising = rising && f.elbo[i] >= f.elbo[i - 1] - 1e-9;
    s.require("the ELBO never falls from one iteration to the next", rising);
  }
  {
    tr::SusieOptions o;
    o.effects = 5;
    o.estimate_prior_variance = tr::PriorVariance::em;
    compare(s, F, "em", "susie(L = 5, EM prior variances)", tr::susie(X, y, o));
  }
  {
    double vy = 0, m = 0;
    for (const double v : y) m += v;
    m /= static_cast<double>(n);
    for (const double v : y) vy += (v - m) * (v - m);
    vy /= static_cast<double>(n - 1);
    tr::SusieOptions o;
    o.effects = 1;
    o.scaled_prior_variance = 0.1;
    o.residual_variance = 0.9 * vy;
    o.estimate_prior_variance = tr::PriorVariance::fixed;
    o.estimate_residual_variance = false;
    const auto f = tr::susie(X, y, o);
    compare(s, F, "ser", "the single-effect regression (L = 1, variances fixed)", f);
    // Exact enumeration of the p one-column models, on the standardised columns.
    const double s2 = 0.9 * vy, v0 = 0.1 * vy;
    std::vector<double> logml(p), postmean(p);
    for (std::size_t j = 0; j < p; j++) {
      double mx = 0;
      for (std::size_t i = 0; i < n; i++) mx += X(i, j);
      mx /= static_cast<double>(n);
      double sxx = 0;
      for (std::size_t i = 0; i < n; i++) sxx += (X(i, j) - mx) * (X(i, j) - mx);
      const double sd = std::sqrt(sxx / static_cast<double>(n - 1));
      double d = 0, xy = 0;
      for (std::size_t i = 0; i < n; i++) {
        const double x = (X(i, j) - mx) / sd;
        d += x * x;
        xy += x * (y[i] - m);
      }
      // log N(y; 0, s²I + v₀xxᵀ) − log N(y; 0, s²I) = −½ log(1 + v₀d/s²) + ½ v₀(xᵀy)²/(s²(s² + v₀d)).
      logml[j] = -0.5 * std::log1p(v0 * d / s2) + 0.5 * v0 * xy * xy / (s2 * (s2 + v0 * d));
      postmean[j] = v0 * xy / (s2 + v0 * d);
    }
    const double top = *std::max_element(logml.begin(), logml.end());
    double z = 0;
    for (const double l : logml) z += std::exp(l - top);
    double worst_alpha = 0, worst_mu = 0;
    for (std::size_t j = 0; j < p; j++) {
      worst_alpha = std::max(worst_alpha, std::fabs(f.alpha[j] - std::exp(logml[j] - top) / z));
      worst_mu = std::max(worst_mu, std::fabs(f.mu[j] - postmean[j]) / std::max(1e-3, std::fabs(postmean[j])));
    }
    s.check("the SER's α equal the enumeration of the 80 one-column models (determinant lemma)", worst_alpha, 1e-12);
    s.check("and its posterior means Sherman–Morrison's", worst_mu, 1e-11);
    s.check("its log Bayes factor is log Σ_j π_j BF_j of the enumeration", std::fabs(f.lbf[0] - (top + std::log(z * (1.0 / static_cast<double>(p) + o.prior_weight_guard)))), 1e-10);
  }
  {
    // Sufficient statistics of the centred data.
    calx::linalg::Matrix XtX(p);
    std::vector<double> mx(p, 0.0), Xty(p, 0.0);
    double my = 0;
    for (const double v : y) my += v;
    my /= static_cast<double>(n);
    for (std::size_t j = 0; j < p; j++) {
      for (std::size_t i = 0; i < n; i++) mx[j] += X(i, j);
      mx[j] /= static_cast<double>(n);
    }
    double yty = 0;
    for (std::size_t i = 0; i < n; i++) {
      yty += (y[i] - my) * (y[i] - my);
      for (std::size_t j = 0; j < p; j++) {
        Xty[j] += (X(i, j) - mx[j]) * (y[i] - my);
        for (std::size_t k = 0; k < p; k++) XtX(j, k) += (X(i, j) - mx[j]) * (X(i, k) - mx[k]);
      }
    }
    compare(s, F, "ss", "susie_suff_stat", tr::susie_suff_stat(XtX, Xty, yty, n, {}));
    // RSS from susieR's z-scores and cor(X).
    calx::linalg::Matrix R(p);
    for (std::size_t j = 0; j < p; j++)
      for (std::size_t k = 0; k < p; k++) R(j, k) = XtX(j, k) / std::sqrt(XtX(j, j) * XtX(k, k));
    tr::SusieOptions o;
    o.estimate_residual_variance = false;
    compare(s, F, "rss", "susie_rss (z-scores and R, σ² fixed)", tr::susie_rss(F.at("rss_z"), R, n, o));
  }

  // ---- calibration over simulated regions ----
  {
    const std::size_t N = 300, nn = 300, pp = 60;
    std::size_t sets = 0, covering = 0, effects = 0, found = 0;
    std::vector<double> bin_pip(5, 0.0), bin_truth(5, 0.0), bin_count(5, 0.0);
    for (std::size_t r = 0; r < N; r++) {
      calx::random::Stream g(2026, 0x53555349, static_cast<std::uint32_t>(r), 0);
      const auto Xr = genotypes(nn, pp, 0.95, g);
      const std::size_t k = 1 + g.below(3);
      std::vector<double> b(pp, 0.0);
      std::vector<std::size_t> causal;
      while (causal.size() < k) {
        const std::size_t j = g.below(pp);
        if (std::find(causal.begin(), causal.end(), j) == causal.end()) causal.push_back(j);
      }
      for (const auto j : causal) b[j] = (g.uniform() < 0.5 ? -1 : 1) * g.uniform(0.4, 0.7);
      std::vector<double> yr(nn);
      for (std::size_t i = 0; i < nn; i++) {
        double v = 0;
        for (std::size_t j = 0; j < pp; j++) v += Xr(i, j) * b[j];
        yr[i] = v + g.normal();
      }
      tr::SusieOptions o;
      o.effects = 5;
      const auto f = tr::susie(Xr, yr, o);
      for (const auto& cs : f.sets) {
        sets++;
        bool hit = false;
        for (const auto j : cs.variables) hit = hit || b[j] != 0;
        covering += hit;
      }
      effects += k;
      for (const auto j : causal) {
        bool in = false;
        for (const auto& cs : f.sets) in = in || std::find(cs.variables.begin(), cs.variables.end(), j) != cs.variables.end();
        found += in;
      }
      for (std::size_t j = 0; j < pp; j++) {
        const auto bin = std::min<std::size_t>(4, static_cast<std::size_t>(f.pip[j] * 5));
        bin_pip[bin] += f.pip[j];
        bin_truth[bin] += b[j] != 0;
        bin_count[bin]++;
      }
    }
    const double coverage = static_cast<double>(covering) / static_cast<double>(sets);
    s.check("95% credible sets over 300 simulated regions cover an effect: 0.95 − coverage (" + std::to_string(sets) + " sets)", 0.95 - coverage, 0.05);
    s.require("and they hold most of the effects (" + std::to_string(found) + " of " + std::to_string(effects) + ")", found * 2 > effects);
    // PIPs in the top bin [0.8, 1]: the fraction of causal columns there is near the mean PIP.
    s.check("PIPs of at least 0.8 are calibrated: fraction causal − mean PIP", bin_truth[4] / bin_count[4] - bin_pip[4] / bin_count[4], 0.06);
    s.check("PIPs below 0.2 too", bin_truth[0] / bin_count[0] - bin_pip[0] / bin_count[0], 0.02);
  }
  return s.done();
}
