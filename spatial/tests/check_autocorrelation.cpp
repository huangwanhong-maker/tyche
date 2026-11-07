// Checks of tyche/spatial/autocorrelation.hpp: Moran's I, global and local, and the weights of nearest neighbours.
//
// - Against PySAL's esda on a rook lattice (binary and row-standardised, symmetric W) and on forty points with their four
//   nearest neighbours (row-standardised, unsymmetric W): I, its expectation, both variances, z-scores and p-values, and
//   each place's local I and quadrant (fixtures/make_autocorrelation_fixtures.py).
// - Closed forms: a checkerboard on a rook lattice has I = −1 exactly; a lattice split in two halves has I near +1; the
//   sum of the local Moran's I over the places is (n − 1) I for row-standardised W.
// - Invariance: I is unchanged by adding a constant to the values and by scaling them.
// - The nearest neighbours of forty points agree with libpysal's KNN.
// - The test's size: under random assignment the permutation test rejects at about α (values that are noise, 300 draws), and
//   its null's mean is −1/(n − 1). Power: a smooth surface gives p at its least, 1/(1 + permutations).
// - A place with no neighbour stands with quadrant 0 and no p-value; values that do not vary are refused.

#include <tyche/spatial/autocorrelation.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixtures/autocorrelation_fixtures.hpp"

namespace sp = tyche::spatial;

namespace {

sp::SpatialWeights weights_of(const fixtures::Case& c) {
  std::vector<std::vector<sp::WeightEntry>> rows;
  for (const auto& r : c.rows) {
    std::vector<sp::WeightEntry> row;
    for (const auto& [j, w] : r) row.push_back({j, w});
    rows.push_back(std::move(row));
  }
  return sp::weights_from_rows(rows, false);
}

void hold(calx::verification::Suite& suite, const std::string& name, const fixtures::Case& c) {
  const auto W = weights_of(c);
  const auto M = sp::morans_i(c.values, W);
  suite.check(name + ": I against esda", M.I - c.I, 1e-12);
  suite.check(name + ": its expectation", M.expected - c.expected, 1e-14);
  suite.check(name + ": its variance for normal values", M.variance_normal - c.variance_normal, 1e-12);
  suite.check(name + ": its variance under random assignment", M.variance_randomisation - c.variance_randomisation, 1e-12);
  suite.check(name + ": z for normal values", M.z_normal - c.z_normal, 1e-10);
  suite.check(name + ": z under random assignment", M.z_randomisation - c.z_randomisation, 1e-10);
  suite.check(name + ": p for normal values", M.p_normal - c.p_normal, 1e-10);
  suite.check(name + ": p under random assignment", M.p_randomisation - c.p_randomisation, 1e-10);
  if (!c.local.empty()) {
    const auto L = sp::local_morans_i(c.values, W, 0);
    double most = 0;
    bool same = L.I.size() == c.local.size();
    for (std::size_t i = 0; same && i < c.local.size(); i++) {
      most = std::max(most, std::abs(L.I[i] - c.local[i]));
      same = L.quadrant[i] == c.quadrant[i];
    }
    suite.check(name + ": each place's local I against esda", most, 1e-11);
    suite.require(name + ": each place's quadrant", same);
  }
}

// The rook lattice of rows × columns, each row's neighbours weighing `weight`, or standardised.
std::vector<std::vector<sp::WeightEntry>> rook(std::size_t rows, std::size_t columns) {
  std::vector<std::vector<sp::WeightEntry>> out(rows * columns);
  for (std::size_t r = 0; r < rows; r++)
    for (std::size_t c = 0; c < columns; c++) {
      auto& row = out[r * columns + c];
      if (r > 0) row.push_back({(r - 1) * columns + c, 1.0});
      if (r + 1 < rows) row.push_back({(r + 1) * columns + c, 1.0});
      if (c > 0) row.push_back({r * columns + c - 1, 1.0});
      if (c + 1 < columns) row.push_back({r * columns + c + 1, 1.0});
    }
  return out;
}

}  // namespace

int main() {
  calx::verification::Suite suite;

  hold(suite, "rook lattice, binary", fixtures::lattice_binary);
  hold(suite, "rook lattice, standardised", fixtures::lattice_standardised);
  hold(suite, "four nearest neighbours", fixtures::knn_standardised);

  // ---- nearest neighbours against libpysal ----
  {
    const auto& c = fixtures::knn_standardised;
    const auto W = sp::k_nearest_weights(c.x, c.y, 4);
    bool same = W.size() == c.rows.size();
    for (std::size_t i = 0; same && i < c.rows.size(); i++) {
      std::map<std::size_t, double> theirs;
      for (const auto& [j, w] : c.rows[i]) theirs[j] = w;
      for (std::size_t e = W.row_start[i]; e < W.row_start[i + 1]; e++) same = same && theirs.count(W.column[e]) && std::abs(theirs[W.column[e]] - W.weight[e]) < 1e-14;
      same = same && W.row_start[i + 1] - W.row_start[i] == theirs.size();
    }
    suite.require("the four nearest of forty points, as libpysal finds them", same);
  }

  // ---- neighbours on the sphere: across the antimeridian, where degrees would mislead ----
  {
    const double deg = std::acos(-1.0) / 180.0;
    const std::vector<double> lon{179.5, -179.5, 170.0, -170.0, 0.0}, lat{0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> x, y, z;
    for (std::size_t i = 0; i < lon.size(); i++) {
      x.push_back(std::cos(lat[i] * deg) * std::cos(lon[i] * deg));
      y.push_back(std::cos(lat[i] * deg) * std::sin(lon[i] * deg));
      z.push_back(std::sin(lat[i] * deg));
    }
    const auto W = sp::k_nearest_weights(x, y, z, 1);
    suite.require("on the sphere, the nearest across the antimeridian is one degree away, not 359", W.column[W.row_start[0]] == 1 && W.column[W.row_start[1]] == 0 && W.column[W.row_start[2]] == 0 && W.column[W.row_start[3]] == 1);
  }

  // ---- closed forms ----
  {
    const auto rows = rook(6, 8);
    std::vector<double> checker, halves;
    for (std::size_t r = 0; r < 6; r++)
      for (std::size_t c = 0; c < 8; c++) {
        checker.push_back((r + c) % 2 ? 1.0 : -1.0);
        halves.push_back(c < 4 ? 1.0 : -1.0);
      }
    const auto W = sp::weights_from_rows(rows, true), B = sp::weights_from_rows(rows, false);
    suite.check("a checkerboard, I = −1 exactly (row-standardised)", sp::morans_i(checker, W).I + 1.0, 1e-13);
    suite.check("a checkerboard, I = −1 exactly (binary)", sp::morans_i(checker, B).I + 1.0, 1e-13);
    // Only the 6 edges across the middle of 82 disagree: I = (82 − 2·6)/82 on binary weights.
    suite.check("two halves, I = 1 − 2·6/82 by counting the edges across (binary)", sp::morans_i(halves, B).I - (1.0 - 12.0 / 82.0), 1e-13);
    calx::random::Stream rng(7, 1, 0, 0);
    std::vector<double> noise;
    for (std::size_t i = 0; i < 48; i++) noise.push_back(rng.uniform());
    const auto M = sp::morans_i(noise, W), shifted = [&] {
      auto v = noise;
      for (double& x : v) x = 3.5 * x - 120.0;
      return sp::morans_i(v, W);
    }();
    suite.check("I is unchanged by adding a constant and scaling", M.I - shifted.I, 1e-12);
    const auto L = sp::local_morans_i(noise, W, 0);
    double sum = 0;
    for (double v : L.I) sum += v;
    suite.check("the local I sum to (n − 1) I on row-standardised weights", sum - 47.0 * M.I, 1e-11);
  }

  // ---- the permutation test: its size, and its power ----
  {
    const auto W = sp::weights_from_rows(rook(7, 7), true);
    calx::random::Stream rng(11, 2, 0, 0);
    int rejected = 0;
    const int draws = 300;
    for (int d = 0; d < draws; d++) {
      std::vector<double> v;
      for (std::size_t i = 0; i < 49; i++) v.push_back(rng.uniform());
      // One-sided in the direction observed: rejecting at 0.025 is a two-sided test at 0.05.
      if (sp::morans_i(v, W, 199, static_cast<std::uint64_t>(d + 1)).p_permutation <= 0.025) rejected++;
    }
    const double rate = static_cast<double>(rejected) / draws, se = std::sqrt(0.05 * 0.95 / draws);
    suite.require("under random assignment the test rejects at about 5% (" + std::to_string(rate) + ")", std::abs(rate - 0.05) < 4 * se);
    std::vector<double> surface;
    for (std::size_t r = 0; r < 7; r++)
      for (std::size_t c = 0; c < 7; c++) surface.push_back(static_cast<double>(r) + 0.5 * static_cast<double>(c));
    const auto S = sp::morans_i(surface, W, 499, 3);
    suite.check("a smooth surface: p at its least, 1/500", S.p_permutation - 1.0 / 500.0, 1e-15);
    suite.require("a smooth surface: I well above its expectation", S.I > 0.6 && S.z_randomisation > 5);

    // A hot spot among noise-free cold: high among high is found, with a small p-value, at its centre.
    std::vector<double> spot(49, 0.0);
    for (std::size_t r = 2; r <= 4; r++)
      for (std::size_t c = 2; c <= 4; c++) spot[r * 7 + c] = 10.0;
    const auto H = sp::local_morans_i(spot, W, 999, 5);
    suite.require("a hot spot: its centre is high among high", H.quadrant[3 * 7 + 3] == 1);
    suite.require("a hot spot: its centre's p-value is small", H.p[3 * 7 + 3] <= 0.01);
    suite.require("a hot spot: far from it, low among low", H.quadrant[0] == 3);
  }

  // ---- what is refused, and what stands apart ----
  {
    std::vector<std::vector<sp::WeightEntry>> rows = rook(3, 3);
    rows.push_back({});  // a tenth place with no neighbour
    const auto W = sp::weights_from_rows(rows, true);
    std::vector<double> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const auto L = sp::local_morans_i(v, W, 19);
    suite.require("a place with no neighbour: quadrant 0 and no p-value", L.quadrant[9] == 0 && std::isnan(L.p[9]) && L.I[9] == 0.0);
    std::vector<double> flat(10, 2.0);
    bool refused = false;
    try {
      sp::morans_i(flat, W);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    suite.require("values that do not vary are refused", refused);
    refused = false;
    try {
      std::vector<std::vector<sp::WeightEntry>> self{{{0, 1.0}}};
      sp::weights_from_rows(self);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    suite.require("a place is not its own neighbour", refused);
  }

  return suite.done();
}
