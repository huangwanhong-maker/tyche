// Checks of tyche/spatial/support.hpp: a quantity carried from one set of zones to another.
//
// - Areal weighting of an extensive quantity against values by hand, the sources' totals kept when the targets cover them, and a
//   source partly outside the targets giving only its share inside; of an intensive quantity, the measure-weighted mean.
// - Dasymetric carriage: the amounts follow the ancillary amounts, the totals are kept, and a source with no ancillary amount falls
//   back to measure and is counted.
// - Class densities: the densities of three classes recovered exactly from sources that mix them, and a dasymetric carriage with
//   them reproducing a truth that is uniform within classes, where areal weighting does not.
// - Pycnophylactic interpolation: every zone's total kept to rounding; one zone gives a flat surface; on a gradient that crosses the
//   zones' borders it errs a third as much as a uniform density within each zone; a surface symmetric under reflection when its zones are.

#include <tyche/spatial/support.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <vector>

using namespace tyche::spatial;

int main() {
  calx::verification::Suite s;

  // ---- areal weighting ----
  {
    // Two sources (areas 4 and 6), three targets; source 0 splits 1 and 3, source 1 splits 2, 2 and 2.
    const std::vector<Part> parts = {{0, 0, 1, 0}, {0, 1, 3, 0}, {1, 1, 2, 0}, {1, 2, 4, 0}};
    const std::vector<double> y = {40, 60};
    const Carried c = areal_extensive(3, parts, y);
    s.check("target 0 receives a quarter of source 0", c.values[0] - 10, 1e-12);
    s.check("target 1 receives three quarters of source 0 and a third of source 1", c.values[1] - 50, 1e-12);
    s.check("target 2 receives two thirds of source 1", c.values[2] - 40, 1e-12);
    s.check("the sources' totals are kept", c.values[0] + c.values[1] + c.values[2] - 100, 1e-12);
    const std::vector<double> whole = {8, 6};
    const Carried part = areal_extensive(3, parts, y, whole);
    s.check("a source half outside the targets gives half its amount", part.given[0] - 20, 1e-12);
    const std::vector<double> rate = {0.1, 0.4};
    const Carried r = areal_intensive(3, parts, rate);
    s.check("an intensive quantity is the measure-weighted mean: (3·0.1 + 2·0.4)/5", r.values[1] - 0.22, 1e-12);
  }

  // ---- dasymetric ----
  {
    const std::vector<Part> parts = {{0, 0, 1, 30}, {0, 1, 3, 10}, {1, 1, 2, 0}, {1, 2, 4, 0}};
    const std::vector<double> y = {40, 60};
    const Carried c = dasymetric(3, parts, y);
    s.check("the amount follows the ancillary amount: 3/4 of source 0 to target 0", c.values[0] - 30, 1e-12);
    s.require("a source with no ancillary amount falls back to measure, and is counted", c.fell_back == 1 && std::abs(c.values[2] - 40) < 1e-12);
    s.check("the totals are kept", c.values[0] + c.values[1] + c.values[2] - 100, 1e-12);
  }

  // ---- class densities ----
  {
    // Truth: classes of density 10, 2 and 0 (dense housing, sparse housing, parks). Four sources mix them.
    const double d[3] = {10, 2, 0};
    Rectangular A(4, 3);
    const double areas[4][3] = {{1, 2, 0}, {0, 3, 1}, {2, 1, 2}, {0.5, 0.5, 3}};
    std::vector<double> y(4, 0.0);
    for (int sidx = 0; sidx < 4; sidx++)
      for (int k = 0; k < 3; k++) A(sidx, k) = areas[sidx][k], y[sidx] += d[k] * areas[sidx][k];
    const ClassDensities cd = class_densities(A, y);
    s.check("the densities of three classes are recovered", std::abs(cd.density[0] - 10) + std::abs(cd.density[1] - 2) + std::abs(cd.density[2]), 1e-9);
    s.check("... and explain the totals wholly", cd.explained - 1, 1e-12);
    // Source 2 splits into a target holding its dense class and one holding its sparse class and park.
    const std::vector<Part> parts = {{2, 0, 2, 2 * cd.density[0]}, {2, 1, 3, 1 * cd.density[1] + 2 * cd.density[2]}};
    const std::vector<double> ys = {0, 0, y[2], 0};
    const Carried c = dasymetric(2, parts, ys);
    s.check("with them a dasymetric carriage gives the truth, 20 and 2", std::abs(c.values[0] - 20) + std::abs(c.values[1] - 2), 1e-9);
    const Carried a = areal_extensive(2, parts, ys);
    s.require("where areal weighting does not (8.8 and 13.2)", std::abs(a.values[0] - 8.8) < 1e-9);
  }

  // ---- pycnophylactic ----
  {
    const std::size_t R = 40, C = 40;
    ZoneGrid g{R, C, std::vector<int>(R * C)};
    std::vector<double> truth(R * C), totals(4, 0.0);
    for (std::size_t r = 0; r < R; r++)
      for (std::size_t c = 0; c < C; c++) {
        const int z = static_cast<int>((r >= R / 2) * 2 + (c >= C / 2));
        g.zone[r * C + c] = z;
        truth[r * C + c] = 1 + 0.1 * static_cast<double>(r) + 0.2 * static_cast<double>(c);
        totals[static_cast<std::size_t>(z)] += truth[r * C + c];
      }
    const Surface sf = pycnophylactic(g, totals);
    std::vector<double> sum(4, 0.0);
    for (std::size_t i = 0; i < R * C; i++) sum[static_cast<std::size_t>(g.zone[i])] += sf.values[i];
    double worst = 0;
    for (int z = 0; z < 4; z++) worst = std::max(worst, std::abs(sum[z] - totals[z]) / totals[z]);
    s.check("every zone's total is kept", worst, 1e-12);
    s.require("the sweeps converge", sf.converged, std::to_string(sf.sweeps));
    double e_pyc = 0, e_flat = 0;
    for (std::size_t i = 0; i < R * C; i++) {
      const double flat = totals[static_cast<std::size_t>(g.zone[i])] / (R * C / 4.0);
      e_pyc += (sf.values[i] - truth[i]) * (sf.values[i] - truth[i]);
      e_flat += (flat - truth[i]) * (flat - truth[i]);
    }
    s.require("on a gradient across the zones it errs far less than a uniform density in each zone", e_pyc < 0.5 * e_flat,
              std::to_string(std::sqrt(e_pyc / e_flat)));
    ZoneGrid one{10, 10, std::vector<int>(100, 0)};
    const std::vector<double> t1 = {50};
    const Surface flat = pycnophylactic(one, t1);
    double spread = 0;
    for (double v : flat.values) spread = std::max(spread, std::abs(v - 0.5));
    s.check("one zone gives a flat surface", spread, 1e-12);
    // Two zones side by side with equal totals: the surface is symmetric under reflection.
    ZoneGrid two{6, 10, std::vector<int>(60)};
    for (std::size_t r = 0; r < 6; r++)
      for (std::size_t c = 0; c < 10; c++) two.zone[r * 10 + c] = c < 5 ? 0 : 1;
    const std::vector<double> t2 = {30, 90};
    const Surface sy = pycnophylactic(two, t2);
    double asym = 0;
    for (std::size_t r = 0; r < 6; r++)
      for (std::size_t c = 0; c < 10; c++) asym = std::max(asym, std::abs(sy.values[r * 10 + c] - sy.values[(5 - r) * 10 + c]));
    s.check("zones symmetric under reflection give a symmetric surface", asym, 1e-9);
    bool rising = true;
    for (std::size_t c = 0; c + 1 < 10; c++) rising = rising && sy.values[2 * 10 + c] <= sy.values[2 * 10 + c + 1] + 1e-12;
    s.require("... rising from the sparser zone into the denser one", rising);
  }

  return s.done();
}
