// Checks of tyche/testing/multiple.hpp and tyche/regression/smoothing_spline.hpp.
//
// - Against R (fixtures/multiple.txt, made by make_multiple_fixtures.py with R 4.4.3 and qvalue 2.38.0):
//   p.adjust's Bonferroni, Holm and Benjamini–Hochberg on 1000 seeded p-values (800 null, 200 from
//   N(3, 1) statistics); qvalue's π̂₀(λ) on its grid, its smoothed π̂₀ and q-values (the default
//   smoother: the spline at the degrees of freedom R's smooth.spline reached, and at df = 3 exactly,
//   which R matches only to its search's tolerance; both to the effect of R's penalty weight, below), the bootstrap's π̂₀ and q-values, and λ = 0.5's;
//   smooth.spline(df = 5) on 25 irregular points, at the knots and between them, which differs from the
//   exact natural spline by R's penalty weight 0.3330 for 1/3 (its sgram); against an independent exact
//   B-spline solution (scipy, in the fixture script) the spline agrees to 10⁻⁹.
// - Closed forms: Holm and BH by hand on five p-values; the smoothing spline's degrees of freedom tend
//   to 2 as λ grows (the least-squares line, reproduced) and to n as λ falls (interpolation); a line is
//   reproduced at every λ.
// - Error rates by simulation: BH at q = 0.1 keeps the false discovery proportion's mean at most
//   π₀ q (200 replicates of 1000 tests with π₀ = 0.8); q-values at 0.1 near it, and more powerful.

#include <tyche/regression/smoothing_spline.hpp>
#include <tyche/testing/multiple.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tt = tyche::testing;
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

double worst_relative(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return INFINITY;
  double w = 0;
  for (std::size_t i = 0; i < a.size(); i++) w = std::max(w, std::fabs(a[i] - b[i]) / std::max(1e-300, std::fabs(b[i])));
  return w;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto F = read_fixture(std::string(TYCHE_FIXTURES) + "/multiple.txt");
  s.require("the fixture of R's values is read", F.count("p") && F.at("p").size() == 1000 && F.count("smoother_q"));
  const std::vector<double>& p = F.at("p");

  // ---- adjusted p-values against R's p.adjust ----
  s.check("Bonferroni equals R's p.adjust (largest relative difference)", worst_relative(tt::bonferroni(p), F.at("bonferroni")), 1e-14);
  s.check("Holm's step-down equals R's p.adjust", worst_relative(tt::holm(p), F.at("holm")), 1e-14);
  s.check("Benjamini–Hochberg's step-up equals R's p.adjust", worst_relative(tt::benjamini_hochberg(p), F.at("bh")), 1e-14);
  {
    const std::vector<double> q{0.01, 0.04, 0.03, 0.005, 0.5};
    // By hand: sorted 0.005, 0.01, 0.03, 0.04, 0.5. Holm: 5·0.005 = 0.025, 4·0.01 = 0.04, 3·0.03 = 0.09,
    // 2·0.04 = 0.08 → 0.09 (monotone), 0.5. BH: 0.5·5/5 = 0.5, 0.04·5/4 = 0.05, 0.03·5/3 = 0.05,
    // 0.01·5/2 = 0.025, 0.005·5 = 0.025.
    const auto h = tt::holm(q), b = tt::benjamini_hochberg(q);
    s.require("Holm by hand on five p-values", std::fabs(h[0] - 0.04) < 1e-15 && std::fabs(h[1] - 0.09) < 1e-15 && std::fabs(h[2] - 0.09) < 1e-15 &&
                                                  std::fabs(h[3] - 0.025) < 1e-15 && std::fabs(h[4] - 0.5) < 1e-15);
    s.require("Benjamini–Hochberg by hand on five p-values", std::fabs(b[0] - 0.025) < 1e-15 && std::fabs(b[1] - 0.05) < 1e-15 &&
                                                                std::fabs(b[2] - 0.05) < 1e-15 && std::fabs(b[3] - 0.025) < 1e-15 && std::fabs(b[4] - 0.5) < 1e-15);
  }

  // ---- Storey's π₀ and q-values against R's qvalue ----
  {
    const auto e = tt::pi0_estimate(p);
    s.check("π̂₀(λ) on qvalue's grid equals pi0est's pi0.lambda", worst_relative(e.by_lambda, F.at("smoother_pi0_lambda")), 1e-14);
    // R's smooth.spline reached df = 3.00033 (its search's tolerance); at that df the spline is R's.
    tt::Pi0Options o;
    o.smooth_df = F.at("smoother_spline_df")[0];
    const auto r = tt::pi0_estimate(p, o);
    s.check("the smoother's curve at R's own df is pi0est's pi0.smooth (to the effect of R's penalty weight)", worst_relative(r.smoothed, F.at("smoother_pi0_smooth")), 1e-6);
    s.check("and its π̂₀ is pi0est's (smoother, the default)", std::fabs(r.pi0 - F.at("smoother_pi0")[0]), 1e-6);
    s.check("π̂₀ at df = 3 exactly is R's within the tolerance of its df search", std::fabs(e.pi0 - F.at("smoother_pi0")[0]), 2e-4);
    const auto q = tt::qvalues(p, o);
    s.check("q-values are qvalue's (smoother)", worst_relative(q.q, F.at("smoother_q")), 1e-6);
    tt::Pi0Options b;
    b.method = tt::Pi0Method::bootstrap;
    s.check("the bootstrap's π̂₀ equals pi0est's (closed form of Storey, Taylor and Siegmund)", std::fabs(tt::pi0_estimate(p, b).pi0 - F.at("bootstrap_pi0")[0]), 1e-14);
    s.check("and its q-values qvalue's", worst_relative(tt::qvalues(p, b).q, F.at("bootstrap_q")), 1e-13);
    tt::Pi0Options f;
    f.method = tt::Pi0Method::fixed;
    s.check("λ = 0.5: π̂₀ = #{p ≥ ½}/(m/2) equals pi0est's", std::fabs(tt::pi0_estimate(p, f).pi0 - F.at("fixed_pi0")[0]), 1e-14);
    s.check("and its q-values qvalue's", worst_relative(tt::qvalues(p, f).q, F.at("fixed_q")), 1e-13);
    s.require("π₀ = 1 gives Benjamini and Hochberg's adjustment", worst_relative(tt::qvalues(p, {}, 1.0).q, tt::benjamini_hochberg(p)) == 0);
  }

  // ---- the smoothing spline ----
  {
    const auto& x = F.at("spline_x");
    const auto& y = F.at("spline_y");
    const auto sp = tr::smoothing_spline(x, y, F.at("spline_df5_df")[0]);
    s.check("the smoothing spline at df = 5.0006 on 25 irregular points equals an independent B-spline solution (largest relative difference)",
            worst_relative(sp.fitted, F.at("spline_df5_exact")), 1e-9);
    // R's smooth.spline integrates its penalty with 0.3330 for 1/3 (its sgram): the B-spline solution with
    // that weight is R's to 4·10⁻¹¹, and the exact spline departs from R's by about 2·10⁻⁴.
    s.check("R's smooth.spline differs from it only by R's penalty weight 0.3330 (the B-spline solution with it, against R)",
            worst_relative(F.at("spline_df5_sgram333"), F.at("spline_df5_fit")), 1e-9);
    s.check("so the spline is R's to the effect of that weight", worst_relative(sp.fitted, F.at("spline_df5_fit")), 5e-4);
    const std::vector<double> between{sp(0.5), sp(2.5), sp(5.5), sp(9.5)};
    s.check("and so are its values between the knots and beyond the last (linear)", worst_relative(between, F.at("spline_df5_between")), 5e-4);
    s.check("its degrees of freedom are those asked", std::fabs(sp.df - F.at("spline_df5_df")[0]), 1e-9);
    const auto stiff = tr::smoothing_spline_lambda(x, y, 1e12);
    const auto loose = tr::smoothing_spline_lambda(x, y, 1e-12);
    s.check("as λ → ∞ the degrees of freedom tend to 2", stiff.df - 2, 1e-6);
    s.check("as λ → 0 they tend to n (interpolation)", static_cast<double>(x.size()) - loose.df, 1e-4);
    // A straight line is not penalised: reproduced at every λ.
    std::vector<double> line(x.size());
    for (std::size_t i = 0; i < x.size(); i++) line[i] = 2 - 0.3 * x[i];
    const auto l = tr::smoothing_spline_lambda(x, line, 10.0);
    double worst = 0;
    for (std::size_t i = 0; i < x.size(); i++) worst = std::max(worst, std::fabs(l.fitted[i] - line[i]));
    s.check("a straight line is reproduced whatever λ (largest error)", worst, 1e-11);
    // The least-squares line as λ → ∞.
    double mx = 0, my = 0;
    for (std::size_t i = 0; i < x.size(); i++) { mx += x[i]; my += y[i]; }
    mx /= static_cast<double>(x.size()); my /= static_cast<double>(x.size());
    double sxy = 0, sxx = 0;
    for (std::size_t i = 0; i < x.size(); i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
    worst = 0;
    for (std::size_t i = 0; i < x.size(); i++) worst = std::max(worst, std::fabs(stiff.fitted[i] - (my + sxy / sxx * (x[i] - mx))));
    s.check("and the stiff spline is the least-squares line", worst, 1e-6);
  }

  // ---- the false discovery rate by simulation ----
  {
    const std::size_t R = 200, m = 1000, m0 = 800;
    double fdp_bh = 0, fdp_q = 0, power_bh = 0, power_q = 0;
    for (std::size_t r = 0; r < R; r++) {
      calx::random::Stream g(7, 0x46445253, static_cast<std::uint32_t>(r), 0);
      std::vector<double> pv(m);
      for (std::size_t i = 0; i < m; i++) pv[i] = i < m0 ? g.uniform() : 0.5 * std::erfc((g.normal() + 2.5) / std::sqrt(2.0));
      const auto b = tt::benjamini_hochberg(pv);
      const auto q = tt::qvalues(pv).q;
      double fb = 0, tb = 0, fq = 0, tq = 0;
      for (std::size_t i = 0; i < m; i++) {
        if (b[i] <= 0.1) (i < m0 ? fb : tb) += 1;
        if (q[i] <= 0.1) (i < m0 ? fq : tq) += 1;
      }
      fdp_bh += fb / std::max(1.0, fb + tb);
      fdp_q += fq / std::max(1.0, fq + tq);
      power_bh += tb / static_cast<double>(m - m0);
      power_q += tq / static_cast<double>(m - m0);
    }
    fdp_bh /= R; fdp_q /= R; power_bh /= R; power_q /= R;
    // The FDR of BH is π₀ q = 0.08 exactly for independent tests (Benjamini and Hochberg 1995, Theorem 1;
    // equality, Benjamini and Yekutieli 2001); the mean of 200 FDPs has standard error about 0.002.
    s.check("Benjamini–Hochberg at q = 0.1: mean false discovery proportion − π₀q (200 × 1000 tests, π₀ = 0.8)", fdp_bh - 0.08, 0.01);
    s.check("q-values at 0.1: mean false discovery proportion − 0.1 (Storey's estimate makes it nearer q)", fdp_q - 0.1, 0.015);
    s.require("and q-values find more than BH (power " + calx::verification::exponential(power_q) + " against " + calx::verification::exponential(power_bh) + ")", power_q > power_bh);
  }
  return s.done();
}
