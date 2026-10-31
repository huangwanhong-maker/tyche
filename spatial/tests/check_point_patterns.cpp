// Checks of tyche/spatial/point_patterns.hpp: the co-location quotient and Clark and Evans's ratio.
//
// - The co-location quotient by hand: six points on a line, marks A B B A A B, with a tie at the nearest
//   distance shared: CLQ_{A→A} = 0, CLQ_{A→B} = 5/3, CLQ_{B→A} = 5/6, CLQ_{B→B} = 5/4 (Leslie and
//   Kronenfeld's eqs. 1–4 worked by hand); the same within a radius of 1.6, where one point has no
//   neighbour and leaves N_A; invariant under translation.
// - Its invariants: Σ_B CLQ_{A→B} N'_B/(N − 1) = 1 for every A (random patterns, four marks, by nearest
//   neighbours and within a radius); under random labelling the permutation null's mean is 1 (exactly in
//   expectation); the upper p-value under random labelling is uniform: P(p ≤ 0.05) ≈ 0.05 (400 labellings).
// - Attraction found: B placed beside A gives CLQ_{A→B} far above 1, p_upper at its least 1/(1 + perms).
// - Strata kept: a pattern whose halves differ in composition (a common terrain) has CLQ_{A→A} above 1,
//   significant against the unstratified null and not against the stratified one, whose mean is the
//   observed segregation; and the same null whatever values label the strata.
// - Clark and Evans: a square lattice has r̄ = spacing exactly, R = 2 against the uncorrected expectation
//   and r̄ / E_Donnelly in closed form; a hexagonal lattice R = 2√2/3^{1/4} = 2.1491 uncorrected; Poisson
//   points (1000 patterns of 100) give mean R = 1 and reject at α = 0.05 at the rate α (the test's size,
//   with Donnelly's corrections); clustered points (a Thomas process) R ≪ 1 and p tiny.

#include <tyche/spatial/point_patterns.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace sp = tyche::spatial;

int main() {
  calx::verification::Suite s;

  // ---- by hand ----
  {
    std::vector<double> x{0, 1, 2, 5, 9, 10.5}, y(6, 0.0);
    const std::vector<std::size_t> mark{0, 1, 1, 0, 0, 1};
    const auto c = sp::colocation(x, y, mark, 2, {}, 0, 99, 7);
    const double expect[4] = {0, 5.0 / 3, 5.0 / 6, 5.0 / 4};
    double worst = 0;
    for (int i = 0; i < 4; i++) worst = std::max(worst, std::fabs(c.clq[i] - expect[i]));
    s.check("colocation: six points by hand, a tie shared (CLQ_AA, AB, BA, BB = 0, 5/3, 5/6, 5/4)", worst, 1e-15);
    s.require("colocation: N_A per mark", c.count[0] == 3 && c.count[1] == 3 && c.permutations == 99);
    const auto r = sp::colocation(x, y, mark, 2, {}, 1.6, 99, 7);
    worst = 0;
    for (int i = 0; i < 4; i++) worst = std::max(worst, std::fabs(r.clq[i] - expect[i]));
    s.check("colocation: within a radius of 1.6 (the point at 5 has no neighbour and leaves N_A: the same values)", worst, 1e-15);
    for (auto& v : x) v += 1000;
    for (auto& v : y) v -= 250;
    const auto t = sp::colocation(x, y, mark, 2, {}, 0, 99, 7);
    worst = 0;
    for (int i = 0; i < 4; i++) worst = std::max(worst, std::fabs(t.clq[i] - expect[i]));
    s.check("colocation: invariant under translation", worst, 1e-15);
  }

  // ---- invariants, and the null under random labelling ----
  {
    const std::size_t N = 300, marks = 4;
    calx::random::Stream z(51, 0, 0, 0);
    std::vector<double> x(N), y(N);
    std::vector<std::size_t> mark(N);
    for (std::size_t i = 0; i < N; i++) {
      x[i] = z.uniform(0, 100);
      y[i] = z.uniform(0, 60);
      mark[i] = i % 7 == 0 ? 3 : z.below(3);
    }
    for (const double radius : {0.0, 6.0}) {
      const auto c = sp::colocation(x, y, mark, marks, {}, radius, 499, 3);
      double worst = 0, worst_null = 0;
      for (std::size_t A = 0; A < marks; A++) {
        double sum = 0;
        for (std::size_t B = 0; B < marks; B++) {
          const double nb = static_cast<double>(c.count[B]) - (A == B ? 1.0 : 0.0);
          sum += c.clq[A * marks + B] * nb / static_cast<double>(N - 1);
          if (radius == 0) worst_null = std::max(worst_null, std::fabs(c.null_mean[A * marks + B] - 1) / (c.null_sd[A * marks + B] / std::sqrt(499.0)));
        }
        worst = std::max(worst, std::fabs(sum - 1));
      }
      const std::string what = radius == 0 ? " (nearest neighbours)" : " (within a radius)";
      s.check("colocation: Σ_B CLQ_{A→B} N'_B/(N − 1) = 1 for every A" + what, worst, 1e-12);
      if (radius == 0) s.check("colocation: the permutation null's mean is 1 under random labelling (in its standard errors)", worst_null, 4.0);
    }
    // P(p_upper ≤ 0.05) under random labelling.
    double rejected = 0;
    const int R = 400;
    for (int r = 0; r < R; r++) {
      calx::random::Stream zr(52, static_cast<std::uint32_t>(r), 0, 0);
      std::vector<std::size_t> m(N);
      for (auto& v : m) v = zr.below(2);
      const auto c = sp::colocation(x, y, m, 2, {}, 0, 99, 100 + static_cast<std::uint64_t>(r));
      rejected += c.p_upper[0 * 2 + 1] <= 0.05 ? 1 : 0;
    }
    s.check("colocation: under random labelling the upper p-value of CLQ_{A→B} is at most 0.05 at the rate 0.05 (400 labellings)", rejected / R - 0.05, 0.03);
  }

  // ---- attraction ----
  {
    calx::random::Stream z(53, 0, 0, 0);
    std::vector<double> x, y;
    std::vector<std::size_t> mark;
    for (int i = 0; i < 60; i++) {
      const double ax = z.uniform(0, 100), ay = z.uniform(0, 100);
      x.push_back(ax), y.push_back(ay), mark.push_back(0);
      x.push_back(ax + z.uniform(-0.5, 0.5)), y.push_back(ay + z.uniform(-0.5, 0.5)), mark.push_back(1);  // a B beside each A
      x.push_back(z.uniform(0, 100)), y.push_back(z.uniform(0, 100)), mark.push_back(2);
    }
    const auto c = sp::colocation(x, y, mark, 3, {}, 0, 999, 5);
    s.require("colocation: B beside A: CLQ_{A→B} far above 1", c.clq[0 * 3 + 1] > 2.5, "CLQ " + std::to_string(c.clq[1]));
    s.check("colocation: with p_upper at its least, 1/(1 + 999)", c.p_upper[0 * 3 + 1] - 1.0 / 1000, 1e-15);
    s.require("colocation: and C, placed at random, not attracted to A (p_upper > 0.05)", c.p_upper[0 * 3 + 2] > 0.05, "p " + std::to_string(c.p_upper[2]));
  }

  // ---- strata: a common terrain ----
  {
    calx::random::Stream z(54, 0, 0, 0);
    std::vector<double> x, y;
    std::vector<std::size_t> mark, stratum;
    for (int i = 0; i < 400; i++) {
      const bool left = i < 200;
      x.push_back(left ? z.uniform(0, 50) : z.uniform(50, 100));
      y.push_back(z.uniform(0, 50));
      mark.push_back(z.uniform() < (left ? 0.8 : 0.2) ? 0 : 1);  // A common on the left, rare on the right
      stratum.push_back(left ? 0 : 1);
    }
    const auto plain = sp::colocation(x, y, mark, 2, {}, 0, 999, 9);
    const auto strat = sp::colocation(x, y, mark, 2, stratum, 0, 999, 9);
    s.require("colocation: the terrain makes A gather about A (CLQ_AA > 1.2), significant against random labelling", plain.clq[0] > 1.2 && plain.p_upper[0] < 0.01,
              "CLQ " + std::to_string(plain.clq[0]) + ", p " + std::to_string(plain.p_upper[0]));
    s.require("colocation: not against the stratified null, which keeps each half's composition (p_upper > 0.05)", strat.p_upper[0] > 0.05, "p " + std::to_string(strat.p_upper[0]));
    s.check("colocation: the stratified null's mean is the terrain's segregation (CLQ_AA − null mean, in null sd)", (strat.clq[0] - strat.null_mean[0]) / strat.null_sd[0], 2.5);
    s.require("colocation: while the unstratified null's mean is 1", std::fabs(plain.null_mean[0] - 1) < 0.02, "mean " + std::to_string(plain.null_mean[0]));
    std::vector<std::size_t> labels(stratum.size());
    for (std::size_t i = 0; i < labels.size(); i++) labels[i] = stratum[i] == 0 ? 4000000007u : 3u;  // labels of any value
    const auto relabelled = sp::colocation(x, y, mark, 2, labels, 0, 999, 9);
    s.check("colocation: strata named by any labels give the same null (difference of the null means, in their standard errors)",
            (relabelled.null_mean[0] - strat.null_mean[0]) / (strat.null_sd[0] * std::sqrt(2.0 / 999)), 4.0);
  }

  // ---- Clark and Evans ----
  {
    const int m = 30;
    std::vector<double> x, y;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < m; j++) {
        x.push_back(i + 0.5);
        y.push_back(j + 0.5);
      }
    const auto ce = sp::clark_evans(x, y, 0, m, 0, m);
    const double n = m * m, A = n, P = 4.0 * m;
    const double donnelly = 0.5 * std::sqrt(A / n) + (0.0514 + 0.041 / std::sqrt(n)) * P / n;
    s.check("clark_evans: a square lattice has r̄ = its spacing", ce.observed - 1, 1e-15);
    s.check("clark_evans: R = 2 against the uncorrected 0.5/√(n/A)", ce.observed / (0.5 / std::sqrt(n / A)) - 2, 1e-14);
    s.check("clark_evans: R = r̄ / E r̄ with Donnelly's expectation 0.5√(A/n) + (0.0514 + 0.041/√n) P/n", ce.ratio - 1 / donnelly, 1e-14);
    s.require("clark_evans: regular: z large, p tiny", ce.z > 20 && ce.p < 1e-50);

    // A hexagonal lattice: rows √3/2 apart, alternate rows shifted by a half.
    std::vector<double> hx, hy;
    const int cols = 40, rows = 46;
    const double h = std::sqrt(3.0) / 2;
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++) {
        hx.push_back(c + 0.25 + 0.5 * (r % 2));
        hy.push_back((r + 0.5) * h);
      }
    const auto hex = sp::clark_evans(hx, hy, 0, cols, 0, rows * h);
    const double hn = static_cast<double>(hx.size()), hA = cols * rows * h;
    s.check("clark_evans: a hexagonal lattice has R = 2√2/3^{1/4} = 2.1491 uncorrected, the greatest", hex.observed / (0.5 / std::sqrt(hn / hA)) - 2 * std::numbers::sqrt2 / std::pow(3.0, 0.25), 1e-12);
  }
  {
    double mean = 0, rejected = 0;
    const int R = 1000;
    for (int r = 0; r < R; r++) {
      calx::random::Stream z(55, static_cast<std::uint32_t>(r), 0, 0);
      std::vector<double> x(100), y(100);
      for (int i = 0; i < 100; i++) {
        x[i] = z.uniform(0, 2);
        y[i] = z.uniform(0, 1);
      }
      const auto ce = sp::clark_evans(x, y, 0, 2, 0, 1);
      mean += ce.ratio / R;
      rejected += ce.p < 0.05 ? 1 : 0;
    }
    s.check("clark_evans: Poisson points in a 2 × 1 rectangle have mean R = 1 with Donnelly's correction (1000 patterns of 100)", mean - 1, 0.01);
    s.check("clark_evans: and reject at α = 0.05 at the rate α", rejected / R - 0.05, 0.02);

    calx::random::Stream z(56, 0, 0, 0);
    std::vector<double> x, y;
    for (int p = 0; p < 20; p++) {
      const double px = z.uniform(0.1, 1.9), py = z.uniform(0.1, 0.9);
      for (int k = 0; k < 5; k++) {
        x.push_back(std::clamp(px + z.normal(0, 0.02), 0.0, 2.0));
        y.push_back(std::clamp(py + z.normal(0, 0.02), 0.0, 1.0));
      }
    }
    const auto ce = sp::clark_evans(x, y, 0, 2, 0, 1);
    s.require("clark_evans: clustered points (a Thomas process) have R < 0.6 and p < 10⁻⁶", ce.ratio < 0.6 && ce.p < 1e-6, "R " + std::to_string(ce.ratio) + ", p " + std::to_string(ce.p));
  }
  return s.done();
}
