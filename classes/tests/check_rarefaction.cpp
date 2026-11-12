// Checks of tyche/classes/rarefaction against brute force, exact expectations and seeded simulations.
//
// The interpolated Hill numbers and expected frequency counts are held to the mean over every
// subsample of every size, enumerated individual by individual on two small samples. The rarefied
// richness and the interpolated coverage are held to be exactly unbiased for their population
// expectations by enumerating every multinomial sample of a small law. The joined curve returns the
// sample's own value at n from both sides, its increments on either side agree, and it tends to the
// asymptotic estimates. Coverage is held to Turing's at n − 1 and to Simpson's at 1, inverted by
// size_at_coverage, and monotone. Seeded samples from a known law show the extrapolations near what
// samples of 2n find, and the bootstrap intervals covering it.

#include <tyche/classes/asymptotic.hpp>
#include <tyche/classes/diversity.hpp>
#include <tyche/classes/rarefaction.hpp>

#include <calx/random/multinomial.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <vector>

using namespace tyche::classes;

namespace {

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

std::vector<double> to_double(const std::vector<std::int64_t>& v) { return std::vector<double>(v.begin(), v.end()); }

// Every subsample of the individuals: their classes `label` (one entry per individual).
struct Means {
  std::vector<double> richness, entropy, concentration, power_half, power_three;  // by m
  std::vector<std::vector<double>> frequencies;                                    // [m][k]
};

Means enumerate(const std::vector<int>& label, int classes) {
  const int n = static_cast<int>(label.size());
  Means r;
  r.richness.assign(n + 1, 0), r.entropy.assign(n + 1, 0), r.concentration.assign(n + 1, 0), r.power_half.assign(n + 1, 0), r.power_three.assign(n + 1, 0);
  r.frequencies.assign(n + 1, std::vector<double>(n + 1, 0.0));
  std::vector<double> count(n + 1, 0.0);
  for (unsigned mask = 1; mask < (1u << n); mask++) {
    std::vector<int> k(classes, 0);
    int m = 0;
    for (int i = 0; i < n; i++)
      if (mask >> i & 1) k[label[i]]++, m++;
    count[m]++;
    double rich = 0, h = 0, c = 0, ph = 0, p3 = 0;
    for (int x : k) {
      r.frequencies[m][x]++;
      if (!x) continue;
      const double p = static_cast<double>(x) / m;
      rich++, h -= p * std::log(p), c += p * p, ph += std::sqrt(p), p3 += p * p * p;
    }
    r.richness[m] += rich, r.entropy[m] += h, r.concentration[m] += c, r.power_half[m] += ph, r.power_three[m] += p3;
  }
  for (int m = 1; m <= n; m++) {
    r.richness[m] /= count[m], r.entropy[m] /= count[m], r.concentration[m] /= count[m], r.power_half[m] /= count[m], r.power_three[m] /= count[m];
    for (double& f : r.frequencies[m]) f /= count[m];
  }
  return r;
}

// Every composition of n into the law's classes, with its multinomial probability.
void compositions(const std::vector<double>& p, int n, const std::function<void(const std::vector<double>&, double)>& visit) {
  std::vector<double> x(p.size(), 0);
  std::function<void(std::size_t, int, double, double)> rec = [&](std::size_t i, int left, double logp, double logfact) {
    if (i + 1 == p.size()) {
      x[i] = left;
      const double lp = logp + left * std::log(p[i]) - std::lgamma(left + 1.0);
      visit(x, std::exp(logfact + lp));
      return;
    }
    for (int k = 0; k <= left; k++) {
      x[i] = k;
      rec(i + 1, left - k, logp + k * std::log(p[i]) - std::lgamma(k + 1.0), logfact);
    }
  };
  rec(0, n, 0, std::lgamma(n + 1.0));
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- interpolation against every subsample ----
  for (const auto& label : {std::vector<int>{0, 0, 0, 1, 1, 2, 3}, std::vector<int>{0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 5, 6}}) {
    int classes = 0;
    for (int c : label) classes = std::max(classes, c + 1);
    std::vector<double> counts(classes, 0.0);
    for (int c : label) counts[c]++;
    const auto a = abundances(counts);
    const auto e = enumerate(label, classes);
    const int n = static_cast<int>(label.size());
    double g0 = 0, g1 = 0, g2 = 0, gh = 0, g3 = 0, gf = 0;
    for (int m = 1; m <= n; m++) {
      g0 = std::max(g0, std::fabs(interpolated(a, 0, m) - e.richness[m]));
      g1 = std::max(g1, std::fabs(std::log(interpolated(a, 1, m)) - e.entropy[m]));
      g2 = std::max(g2, std::fabs(1 / interpolated(a, 2, m) - e.concentration[m]));
      gh = std::max(gh, std::fabs(std::pow(interpolated(a, 0.5, m), 0.5) - e.power_half[m]));
      g3 = std::max(g3, std::fabs(std::pow(interpolated(a, 3, m), -2) - e.power_three[m]));
      const auto f = expected_frequencies(a, m);
      for (int k = 1; k <= m; k++) gf = std::max(gf, std::fabs(f[k] - e.frequencies[m][k]));
    }
    const std::string tag = " (n = " + std::to_string(n) + ", every subsample of every size)";
    s.check("interpolated q = 0: the mean richness of the subsamples" + tag, g0, 1e-12);
    s.check("interpolated q = 1: the log is the subsamples' mean entropy" + tag, g1, 1e-12);
    s.check("interpolated q = 2: the reciprocal is the subsamples' mean concentration" + tag, g2, 1e-12);
    s.check("interpolated q = ½: from the subsamples' mean Σ (k/m)^½" + tag, gh, 1e-12);
    s.check("interpolated q = 3: from the subsamples' mean Σ (k/m)³" + tag, g3, 1e-12);
    s.check("expected frequency counts f̂_k(m): the subsamples' mean f_k" + tag, gf, 1e-12);
  }

  // ---- exact unbiasedness, every multinomial sample of a small law ----
  {
    const std::vector<double> p{0.5, 0.3, 0.15, 0.05};
    const int n = 7;
    std::vector<double> rich(n, 0.0), cov(n, 0.0);
    double conc = 0, total = 0;
    compositions(p, n, [&](const std::vector<double>& x, double prob) {
      total += prob;
      const auto a = abundances(x);
      for (int m = 1; m < n; m++) rich[m] += prob * interpolated(a, 0, m);
      for (int m = 0; m < n; m++) cov[m] += prob * coverage_at(a, m);
      conc += prob * concentration_estimate(a);
    });
    double gr = 0, gc = 0;
    for (int m = 1; m < n; m++) {
      double t = 0;
      for (double q : p) t += 1 - std::pow(1 - q, m);
      gr = std::max(gr, std::fabs(rich[m] - t));
    }
    for (int m = 0; m < n; m++) {
      double t = 1;
      for (double q : p) t -= q * std::pow(1 - q, m);
      gc = std::max(gc, std::fabs(cov[m] - t));
    }
    double sq = 0;
    for (double q : p) sq += q * q;
    s.check("enumeration: the multinomial probabilities sum to one", total - 1, 1e-14);
    s.check("unbiased: E Ŝ_m = Σ [1 − (1 − p_i)^m], m < n (every sample of 7 from a law on 4)", gr, 1e-14);
    s.check("unbiased: E Ĉ_m = 1 − Σ p_i(1 − p_i)^m, m < n", gc, 1e-14);
    s.check("unbiased: E Σ X(X − 1)/(n(n − 1)) = Σ p²", conc - sq, 1e-15);
  }

  // ---- the joined curve at n, and its asymptotes ----
  const std::vector<double> sample{1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 7, 9, 12, 20, 31};
  const auto a = abundances(sample);
  const double n = static_cast<double>(a.n);
  {
    const auto p = law_of(sample);
    double at_n = 0, ex_n = 0, near = 0;
    for (double q : {0.0, 0.5, 1.0, 2.0, 3.0}) at_n = std::max(at_n, std::fabs(interpolated(a, q, a.n) - hill(p, q)));
    for (int q : {0, 1, 2}) {
      ex_n = std::max(ex_n, std::fabs(extrapolated(a, q, n) - hill(p, q)));
      near = std::max(near, std::fabs(estimate_at(a, q, n + 1e-9) - hill(p, q)));
    }
    s.check("at m = n the interpolation returns the sample's own Hill number (q = 0, ½, 1, 2, 3)", at_n, 1e-12);
    s.check("at m = n the extrapolation returns it too (q = 0, 1, 2)", ex_n, 1e-13);
    s.check("the extrapolation is continuous at n (m = n + 10⁻⁹)", near, 1e-8);
    double ratio = 0;
    for (int q : {0, 1, 2}) {
      const double below = estimate_at(a, q, n) - estimate_at(a, q, n - 1), above = estimate_at(a, q, n + 1) - estimate_at(a, q, n);
      ratio = std::max(ratio, std::fabs(std::log(above / below)));
    }
    s.check("the increments on either side of n agree within a factor 1.5 (|ln ratio|, q = 0, 1, 2)", ratio, std::log(1.5));
    s.check("q = 0 increment below n is f₁/n", (estimate_at(a, 0, n) - estimate_at(a, 0, n - 1)) - 6 / n, 1e-12);
    const auto as = asymptotic_diversity(a);
    const double far = 1e10 * n;
    s.check("extrapolation q = 0 tends to Chao1 (relative, m = 10¹⁰ n)", extrapolated(a, 0, far) / as.estimate[0] - 1, 1e-9);
    s.check("extrapolation q = 1 tends to exp Ĥ", extrapolated(a, 1, far) / as.estimate[1] - 1, 1e-9);
    s.check("extrapolation q = 2 tends to n(n − 1)/Σ X(X − 1)", extrapolated(a, 2, far) / as.estimate[2] - 1, 1e-9);
    double rise = 0;
    for (double m = 2; m <= 3 * n; m++) rise = std::min(rise, estimate_at(a, 0, m) - estimate_at(a, 0, m - 1));
    s.require("q = 0 never falls along the curve (m = 1 … 3n)", rise >= -1e-12);
  }

  // ---- coverage along the curve ----
  {
    s.check("coverage at 0 is 0", coverage_at(a, 0), 1e-15);
    s.check("coverage at n − 1 is Turing's 1 − f₁/n", coverage_at(a, n - 1) - (1 - 6 / n), 1e-14);
    s.check("coverage at 1 is Σ X(X − 1)/(n(n − 1))", coverage_at(a, 1) - concentration_estimate(a), 1e-14);
    s.check("coverage at n is Ĉ_n", coverage_at(a, n) - coverage(a), 0);
    double fall = 0;
    for (double m = 1; m <= 4 * n; m++) fall = std::min(fall, coverage_at(a, m) - coverage_at(a, m - 1));
    s.require("coverage never falls (m = 0 … 4n)", fall >= -1e-14);
    double inv_lo = 0, inv_ex = 0;
    for (double C : {0.2, 0.5, 0.7, 0.8}) {
      const double m = size_at_coverage(a, C);
      if (coverage_at(a, m) < C || (m >= 1 && coverage_at(a, m - 1) >= C)) inv_lo = 1;
    }
    for (double C : {0.97, 0.99, 0.999}) inv_ex = std::max(inv_ex, std::fabs(coverage_at(a, size_at_coverage(a, C)) - C));
    s.require("size at coverage below Ĉ_n: the least whole m with Ĉ_m ≥ C", inv_lo == 0 && coverage(a) > 0.8);
    s.check("size at coverage above Ĉ_n: the extrapolated coverage there is C", inv_ex, 1e-12);
    s.require("size at coverage 1 with singletons: unbounded", std::isinf(size_at_coverage(a, 1)));
  }

  // ---- the bootstrap assemblage and the curve ----
  {
    const auto law = bootstrap_assemblage(a);
    double total = 0, undetected_mass = 0, worst = 0;
    for (std::size_t i = 0; i < law.size(); i++) {
      total += law[i];
      if (i >= a.observed()) undetected_mass += law[i];
      else worst = std::max(worst, law[i] - static_cast<double>(a.counts[i]) / n);
    }
    s.check("bootstrap assemblage: a law", total - 1, 1e-14);
    s.require("bootstrap assemblage: S_obs + ⌈f̂₀⌉ classes", law.size() == a.observed() + static_cast<std::size_t>(std::ceil(undetected(a))));
    s.check("bootstrap assemblage: the undetected hold 1 − Ĉ_n", undetected_mass - (1 - coverage(a)), 1e-14);
    s.require("bootstrap assemblage: observed classes lowered, never raised, never negative", worst <= 1e-16 && *std::min_element(law.begin(), law.end()) >= 0);

    CurveOptions o;
    o.replicates = 40;
    const auto c1 = curve(a, o), c2 = curve(a, o);
    o.seed = 7;
    const auto c3 = curve(a, o);
    bool same = c1.size() == c2.size(), differ = false, observed_ok = true, positive = true;
    for (std::size_t i = 0; i < c1.size(); i++) {
      same = same && c1[i].se == c2[i].se && c1[i].estimate == c2[i].estimate;
      differ = differ || c1[i].se != c3[i].se;
      if (c1[i].method == Method::observed) observed_ok = observed_ok && std::fabs(c1[i].estimate - hill(law_of(sample), c1[i].order)) < 1e-12;
      if (c1[i].method == Method::extrapolated) positive = positive && c1[i].se > 0 && c1[i].lower < c1[i].estimate && c1[i].estimate < c1[i].upper;
    }
    s.require("curve: the same seed gives the same intervals; another seed others", same && differ);
    s.require("curve: at m = n the point is the sample itself", observed_ok);
    s.require("curve: extrapolated points carry positive standard errors", positive);
    const auto sizes = default_sizes(a.n, 0, 40);
    s.require("default sizes: from 1 to 2n, n among them, increasing", sizes.front() == 1 && sizes.back() == 2 * n && std::find(sizes.begin(), sizes.end(), n) != sizes.end() && std::is_sorted(sizes.begin(), sizes.end()));
    s.require("refused: extrapolating order 3; a fractional size below n", throws([&] { estimate_at(a, 3, 2 * n); }) && throws([&] { estimate_at(a, 0, 3.5); }));
  }

  // ---- seeded samples from a known law ----
  {
    const std::size_t S = 300;
    std::vector<double> p(S);
    double total = 0;
    for (std::size_t i = 0; i < S; i++) total += (p[i] = std::exp(-static_cast<double>(i) / 60.0));
    for (double& x : p) x /= total;
    const std::int64_t n0 = 800;
    const double m2 = 2.0 * static_cast<double>(n0), half = static_cast<double>(n0 / 2);
    calx::random::Stream rng(20261112, 4, 0, 0);
    // What samples of 2n and n/2 find, by their law: richness exactly, Simpson exactly, entropy by simulation.
    double s2 = 0, sh = 0, sq = 0;
    for (double x : p) s2 += 1 - std::pow(1 - x, m2), sh += 1 - std::pow(1 - x, half), sq += x * x;
    const double d2_2 = 1 / (1 / m2 + (m2 - 1) / m2 * sq), d2_h = 1 / (1 / half + (half - 1) / half * sq);
    double h2 = 0, hh = 0;
    for (int r = 0; r < 4000; r++) {
      h2 += observed_entropy(abundances(to_double(calx::random::multinomial(rng, static_cast<std::int64_t>(m2), p)))) / 4000;
      hh += observed_entropy(abundances(to_double(calx::random::multinomial(rng, n0 / 2, p)))) / 4000;
    }
    const double truth[2][3] = {{sh, std::exp(hh), d2_h}, {s2, std::exp(h2), d2_2}};
    CurveOptions o;
    o.sizes = {half, m2};
    o.replicates = 60;
    const int R = 120;
    double mean[2][3] = {{0, 0, 0}, {0, 0, 0}};
    int covered[2][3] = {{0, 0, 0}, {0, 0, 0}};
    for (int r = 0; r < R; r++) {
      o.seed = 1000 + static_cast<std::uint64_t>(r);
      const auto c = curve(abundances(to_double(calx::random::multinomial(rng, n0, p))), o);
      for (const auto& pt : c) {
        const int j = pt.size > static_cast<double>(n0) ? 1 : 0, q = static_cast<int>(pt.order);
        mean[j][q] += pt.estimate / R;
        covered[j][q] += pt.lower <= truth[j][q] && truth[j][q] <= pt.upper;
      }
    }
    const double bias_limit[2][3] = {{0.005, 0.005, 0.005}, {0.04, 0.01, 0.01}};
    for (int j = 0; j < 2; j++)
      for (int q = 0; q < 3; q++) {
        const std::string where = std::string(j ? "extrapolated to 2n" : "interpolated at n/2") + ", q = " + std::to_string(q);
        s.check("simulation (n = 800 from 300 classes): " + where + ", mean relative error against what such samples find", mean[j][q] / truth[j][q] - 1, bias_limit[j][q]);
        s.check("simulation: " + where + ", bootstrap 95% intervals miss in at most 15% of 120 samples (share missed − 0.05)", std::max(0.0, 1 - covered[j][q] / static_cast<double>(R) - 0.05), 0.10);
      }
  }

  // ---- a large sample runs ----
  {
    calx::random::Stream rng(20261112, 5, 0, 0);
    std::vector<double> p(2500);
    double total = 0;
    for (std::size_t i = 0; i < p.size(); i++) total += (p[i] = 1 / std::pow(static_cast<double>(i + 1), 1.1));
    for (double& x : p) x /= total;
    const auto big = abundances(to_double(calx::random::multinomial(rng, 82000, p)));
    CurveOptions o;
    o.replicates = 20;
    const auto c = curve(big, o);
    bool finite = true;
    for (const auto& pt : c) finite = finite && std::isfinite(pt.estimate) && std::isfinite(pt.se) && std::isfinite(pt.coverage);
    s.require("a sample of 82 000 individuals: curves of orders 0, 1, 2 at 40 sizes with 20 replicates, all finite", finite);
  }

  return s.done();
}
