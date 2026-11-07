// Checks of annual losses and exceedance curves (exceedance.hpp).
//
// - The average annual loss of a catalogue is its sum over its length, exactly.
// - Occurrence curve: maxima 100, 90, ..., 10 over 100 years: the k-th is at return period 100 / k; between them, linear in log T.
// - A law known in closed form: L(T) = a ln T has E[L] = a over p in (0, 1); the trapezoids converge to it as the grid is refined, and
//   their error falls by a factor near four when the spacing halves in log T.
// - What is assumed outside the curve is reported apart: `below`, `beyond`, and a curve held at its last loss.
// - A catalogue drawn from an exponential law: the area under its occurrence curve approaches the mean of the years' maxima.
// - Refusals: unordered return periods, a return period beyond the record.

#include <tyche/descriptive/exceedance.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace td = tyche::descriptive;

int main() {
  calx::verification::Suite s;
  {
    s.check("the average annual loss is the sum over the years", td::average_annual_loss({1000, 2000, 7000}, 100) - 100, 0);
  }
  {
    std::vector<double> maxima;
    for (int k = 1; k <= 10; k++) maxima.push_back(110 - 10.0 * k);
    const auto C = td::occurrence_curve(maxima, 100, {100, 50, 25, 10});
    s.check("the largest of 100 years is the 100-year loss", C[0] - 100, 1e-12);
    s.check("the second is the 50-year loss", C[1] - 90, 1e-12);
    s.check("the fourth is the 25-year loss", C[2] - 70, 1e-12);
    s.check("the tenth is the 10-year loss", C[3] - 10, 1e-12);
    const double T = std::sqrt(100.0 * 50.0);
    s.check("between two ranks it is linear in the logarithm of the return period", td::occurrence_curve(maxima, 100, {T})[0] - 95, 1e-12);
  }
  {
    const auto area = [](double step) {
      std::vector<double> T, L;
      for (double x = step; x <= 14 + 1e-9; x += step) { T.push_back(std::exp(x)); L.push_back(std::log(T.back())); }
      return td::curve_area(T, L, 0.0, true).total();
    };
    const double a1 = area(0.2), a2 = area(0.1), a3 = area(0.05);
    s.check("L(T) = ln T has a mean of 1: the trapezoids reach it", a3 - 1, 2e-3);
    const double rate = (a1 - 1) / (a2 - 1);
    s.require("and their error falls near fourfold when the spacing halves", rate > 3 && rate < 5, std::to_string(rate));
    s.check("and again", (a2 - 1) / (a3 - 1) - 4, 1);
  }
  {
    const auto A = td::curve_area({2, 10}, {10, 50}, 0, true);
    s.check("the part within the curve", A.within - (0.5 - 0.1) * 30, 1e-12);
    s.check("the part below its first point falls to the loss at p = 1", A.below - 0.5 * 5, 1e-12);
    s.check("the part beyond its last holds the last loss", A.beyond - 5, 1e-12);
    s.check("and without it nothing is counted beyond", td::curve_area({2, 10}, {10, 50}, 0, false).beyond, 0);
  }
  {
    calx::random::Stream rng(20261107, 0x45584345, 0, 0);
    const int years = 200000;
    std::vector<double> maxima;
    maxima.reserve(years);
    double mean = 0;
    for (int y = 0; y < years; y++) { const double u = rng.uniform(); const double v = -std::log(1 - u) * 1000; maxima.push_back(v); mean += v / years; }
    std::vector<double> T;
    for (double t = 1.02; t < 1e5; t *= 1.08) T.push_back(t);
    const auto L = td::occurrence_curve(maxima, years, T);
    const double got = td::curve_area(T, L, 0.0, true).total();
    s.check("a catalogue's area under its occurrence curve approaches the mean of its years' maxima", got / mean - 1, 2e-2);
  }
  {
    bool refused = false;
    try { td::curve_area({10, 5}, {1, 2}); } catch (const std::invalid_argument&) { refused = true; }
    s.require("return periods out of order are refused", refused);
    refused = false;
    try { td::occurrence_curve({1, 2, 3}, 10, {100}); } catch (const std::invalid_argument&) { refused = true; }
    s.require("a return period beyond the record is refused", refused);
  }
  return s.done();
}
