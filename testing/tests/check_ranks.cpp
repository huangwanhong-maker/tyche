// Checks of tyche/testing/ranks: Mann and Whitney's U worked by hand (x = 1, 3, 5 against y = 2, 3: U = 3.5, AUC 7/12, the variance
// with a tie of two (6/12)(6 − 6/20) = 2.85 and z = 0.5/√2.85); U against the count of pairs on random samples with ties; perfect
// separation; alike samples; the statistic's size near its level under the null; NaNs left out; refusals.

#include <tyche/testing/ranks.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <vector>

using namespace tyche::testing;

int main() {
  calx::verification::Suite s;
  {
    const std::vector<double> x{1, 3, 5}, y{2, 3};
    const auto R = mann_whitney(x, y);
    s.check("U by hand: x = 1, 3, 5 against y = 2, 3: U = 3.5, AUC 7/12", std::fabs(R.u - 3.5) + std::fabs(R.auc - 7.0 / 12), 1e-15);
    s.check("... the tie-corrected variance (6/12)(6 − 6/20) = 2.85 and z = 0.5/√2.85", std::fabs(R.statistic - 0.5 / std::sqrt(2.85)), 1e-14);
  }
  {
    calx::random::Stream g(20261123, 21, 22, 23);
    double worst = 0;
    for (int t = 0; t < 200; t++) {
      std::vector<double> x(1 + g.below(12)), y(1 + g.below(12));
      for (auto& v : x) v = static_cast<double>(g.below(6));
      for (auto& v : y) v = static_cast<double>(g.below(6));
      double u = 0;
      for (double a : x)
        for (double b : y) u += a > b ? 1 : a == b ? 0.5 : 0;
      worst = std::max(worst, std::fabs(mann_whitney(x, y).u - u));
    }
    s.check("U equals the count of pairs with ties as halves, on 200 random samples with many ties", worst, 1e-12);
  }
  {
    const auto P = mann_whitney(std::vector<double>{5, 6, 7}, std::vector<double>{1, 2, 3, 4});
    const auto A = mann_whitney(std::vector<double>{2, 2, 2}, std::vector<double>{2, 2});
    s.require("every positive above every negative: AUC 1; alike samples: AUC 1/2", P.auc == 1 && A.auc == 0.5);
  }
  {
    calx::random::Stream g(20261123, 24, 25, 26);
    int rejected = 0;
    const int runs = 2000;
    for (int t = 0; t < runs; t++) {
      std::vector<double> x(30), y(40);
      for (auto& v : x) v = g.normal();
      for (auto& v : y) v = g.normal();
      rejected += mann_whitney(x, y).p_value < 0.05;
    }
    s.check("under the null, 30 against 40 normal values, the test rejects near 5% at 5% (2,000 runs)", rejected / static_cast<double>(runs) - 0.05, 0.015);
  }
  {
    const double nan = std::nan("");
    const auto R = mann_whitney(std::vector<double>{1, nan, 3}, std::vector<double>{2});
    s.require("NaNs are left out: two values against one", R.n1 == 2 && R.n2 == 1 && R.u == 1);
    bool refused = false;
    try {
      mann_whitney(std::vector<double>{}, std::vector<double>{1});
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    s.require("an empty sample is refused", refused);
  }
  return s.done();
}
