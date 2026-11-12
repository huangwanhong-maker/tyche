// Checks of tyche/classes/incidence against hand-worked examples, brute force and seeded simulations.
//
// Chao2 is worked by hand with its variance; the interpolated richness is held to the mean over every
// set of t units, enumerated; the curve returns S_obs at T from both sides and tends to Chao2; the
// coverage is held to its forms at T − 1 and T and to monotonicity; the bootstrap assemblage keeps the
// expected incidences. Seeded samples from a Bernoulli model with known detection probabilities show
// the rarefied richness unbiased, the extrapolation and Chao2 near the truth, and Ĉ_T near the
// sample's coverage.

#include <tyche/classes/incidence.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <vector>

using namespace tyche::classes;
using calx::linalg::Rectangular;

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

// The mean richness over every set of t of the units of a detection table.
std::vector<double> mean_over_unit_sets(const Rectangular& d) {
  const std::size_t T = d.rows;
  std::vector<double> sum(T + 1, 0.0), count(T + 1, 0.0);
  for (unsigned mask = 1; mask < (1u << T); mask++) {
    int t = 0;
    for (std::size_t u = 0; u < T; u++) t += mask >> u & 1;
    double rich = 0;
    for (std::size_t i = 0; i < d.cols; i++) {
      bool seen = false;
      for (std::size_t u = 0; u < T; u++) seen = seen || ((mask >> u & 1) && d(u, i) > 0);
      rich += seen;
    }
    sum[t] += rich, count[t]++;
  }
  for (std::size_t t = 1; t <= T; t++) sum[t] /= count[t];
  return sum;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261112, 6, 0, 0);

  // ---- a detection table by hand: 5 units, 7 classes ----
  Rectangular d(5, 7);
  const int table[5][7] = {{1, 1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 0, 0, 0}, {1, 0, 0, 0, 1, 0, 0}, {1, 1, 1, 0, 0, 0, 0}};
  for (int u = 0; u < 5; u++)
    for (int i = 0; i < 7; i++) d(u, i) = table[u][i] * (1 + i);  // counts or covers: any positive entry is a detection
  const auto s5 = incidences(d);
  {
    // Y = (5, 3, 2, 1, 1, 1, 0): T 5, U 13, S_obs 6, Q1 3, Q2 1, Q3 1, Q5 1
    s.require("incidences: T = 5, U = 13, S_obs = 6, Q₁ = 3, Q₂ = Q₃ = Q₅ = 1", s5.units == 5 && s5.total == 13 && s5.observed() == 6 && s5.uniques() == 3 && s5.duplicates() == 1 && s5.frequency(3) == 1 && s5.frequency(5) == 1);
    const std::vector<double> too_many{6};
    s.require("incidences: refused when a class is in more units than there are", throws([&] { incidences(too_many, 5); }));
    const double A = 0.8, T0 = A * 9 / 2, r = 3;
    const auto e = chao2(s5);
    s.check("Chao2: 6 + (4/5)·3²/(2·1) = 9.6", e.estimate - (6 + T0), 1e-14);
    s.check("Chao2: the variance Q₂[A r²/2 + A² r³ + A² r⁴/4], r = Q₁/Q₂", e.variance - (A * r * r / 2 + A * A * r * r * r + A * A * r * r * r * r / 4), 1e-12);
    s.check("Chao2 bias-corrected: 6 + (4/5)·3·2/(2·2)", chao2(s5, 0.95, Chao1Form::bias_corrected).estimate - (6 + A * 1.5), 1e-14);
    const double x = 4.0 * 3 / (4.0 * 3 + 2);
    s.check("incidence coverage: Ĉ_T = 1 − (Q₁/U)·(T − 1)Q₁/((T − 1)Q₁ + 2Q₂)", coverage(s5) - (1 - 3.0 / 13 * x), 1e-15);
    s.check("incidence coverage at T − 1 is 1 − Q₁/U", coverage_at(s5, 4) - (1 - 3.0 / 13), 1e-15);
    s.check("incidence coverage at T is Ĉ_T", coverage_at(s5, 5) - coverage(s5), 0);
    s.check("incidence coverage at 0 is 0", coverage_at(s5, 0), 1e-15);
  }

  // ---- interpolation against every set of units ----
  {
    Rectangular big(9, 16);
    for (std::size_t u = 0; u < big.rows; u++)
      for (std::size_t i = 0; i < big.cols; i++) big(u, i) = rng.uniform() < 0.6 / (1 + 0.4 * static_cast<double>(i)) ? 1 : 0;
    for (const auto* table : {&d, &big}) {
      const auto si = incidences(*table);
      const auto mean = mean_over_unit_sets(*table);
      double gap = 0;
      for (std::size_t t = 1; t <= table->rows; t++) gap = std::max(gap, std::fabs(richness_at(si, static_cast<double>(t)) - mean[t]));
      s.check("interpolated richness: the mean over every set of t units (T = " + std::to_string(table->rows) + ")", gap, 1e-12);
    }
    const double T = 5;
    s.check("at T the richness is S_obs", richness_at(s5, T) - 6, 1e-14);
    s.check("the extrapolation is continuous at T", richness_at(s5, T + 1e-9) - 6, 1e-8);
    s.check("the extrapolation tends to Chao2", richness_at(s5, 1e10) - chao2(s5).estimate, 1e-9);
    double fall = 0;
    for (double t = 1; t <= 20; t++) fall = std::min({fall, richness_at(s5, t) - richness_at(s5, t - 1), coverage_at(s5, t) - coverage_at(s5, t - 1)});
    s.require("richness and coverage never fall along t", fall >= -1e-14);
  }

  // ---- the bootstrap assemblage and the curve ----
  {
    const auto pi = bootstrap_detection(s5);
    double expected = 0, worst = 0;
    for (std::size_t i = 0; i < pi.size(); i++) {
      expected += 5 * pi[i];
      if (i < s5.observed()) worst = std::max(worst, pi[i] - static_cast<double>(s5.counts[i]) / 5);
    }
    s.check("incidence bootstrap: the expected incidences are U", expected - 13, 1e-13);
    s.require("incidence bootstrap: S_obs + ⌈Q̂₀⌉ classes, detected ones lowered", pi.size() == 6 + static_cast<std::size_t>(std::ceil(undetected(s5))) && worst <= 1e-16);
    IncidenceCurveOptions o;
    o.replicates = 40;
    const auto c1 = richness_curve(s5, o), c2 = richness_curve(s5, o);
    bool same = true, observed = false;
    for (std::size_t i = 0; i < c1.size(); i++) {
      same = same && c1[i].se == c2[i].se;
      if (c1[i].method == Method::observed) observed = std::fabs(c1[i].estimate - 6) < 1e-14 && c1[i].se > 0;
    }
    s.require("incidence curve: repeatable by seed; at T the point is S_obs, with a positive standard error", same && observed);
  }

  // ---- seeded samples from a Bernoulli model ----
  {
    const std::size_t S = 200;
    std::vector<double> pi(S);
    for (std::size_t i = 0; i < S; i++) pi[i] = 0.5 * std::exp(-static_cast<double>(i) / 40.0);
    auto sample = [&](std::int64_t T) {
      std::vector<double> y(S);
      for (std::size_t i = 0; i < S; i++) y[i] = static_cast<double>(rng.binomial(T, pi[i]));
      return y;
    };
    const std::int64_t T = 40;
    double ehalf = 0, e2 = 0, total_pi = 0;
    for (double p : pi) ehalf += 1 - std::pow(1 - p, T / 2), e2 += 1 - std::pow(1 - p, 2 * T), total_pi += p;
    double mean_half = 0, mean_2 = 0, cov_err = 0;
    const int R = 300;
    for (int r = 0; r < R; r++) {
      const auto y = sample(T);
      const auto si = incidences(y, T);
      mean_half += richness_at(si, static_cast<double>(T / 2)) / R;
      mean_2 += richness_at(si, static_cast<double>(2 * T)) / R;
      double seen = 0;
      for (std::size_t i = 0; i < S; i++)
        if (y[i] > 0) seen += pi[i];
      cov_err += (coverage(si) - seen / total_pi) / R;
    }
    s.check("simulation (T = 40, 200 classes): rarefied richness at T/2 unbiased (relative mean error)", mean_half / ehalf - 1, 0.003);
    s.check("simulation: extrapolated richness at 2T against what 2T units find (relative)", mean_2 / e2 - 1, 0.03);
    s.check("simulation: Ĉ_T's mean error against the sample's coverage", cov_err, 0.005);
    double chao = 0;
    for (int r = 0; r < 30; r++) chao += chao2(incidences(sample(400), 400)).estimate / 30;
    s.check("simulation (T = 400): Chao2 within 2% of S = 200", chao / S - 1, 0.02);
  }

  return s.done();
}
