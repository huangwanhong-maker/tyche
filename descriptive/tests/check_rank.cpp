// Checks of ranks and rank correlation: mid-ranks, Spearman's ρ against its closed form without ties
// and by hand with them, Kendall's τ_b against the O(n²) count of pairs, invariance under monotone
// maps.

#include <tyche/descriptive/rank.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <vector>

using namespace tyche::descriptive;

namespace {

// τ_b by counting every pair (Kendall 1945).
double tau_b_pairs(const std::vector<double>& x, const std::vector<double>& y) {
  double c = 0, d = 0, tx = 0, ty = 0;
  const std::size_t n = x.size();
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++) {
      const double a = x[i] - x[j], b = y[i] - y[j];
      if (a == 0 && b == 0) { tx++; ty++; continue; }
      if (a == 0) { tx++; continue; }
      if (b == 0) { ty++; continue; }
      (a * b > 0 ? c : d) += 1;
    }
  const double n0 = n * (n - 1) / 2.0;
  return (c - d) / std::sqrt((n0 - tx) * (n0 - ty));
}

}  // namespace

int main() {
  calx::verification::Suite s;
  {
    const std::vector<double> x = {10, 20, 20, 30, 20, 5};
    const auto r = ranks(x);
    s.require("mid-ranks: 5 → 1, 10 → 2, the three 20s → 4, 30 → 6", r == std::vector<double>{2, 4, 4, 6, 4, 1});
  }
  {
    // Without ties, ρ = 1 − 6 Σ d² / (n (n² − 1)).
    const std::vector<double> x = {1, 2, 3, 4, 5, 6, 7}, y = {2, 1, 4, 3, 7, 5, 6};
    double d2 = 0;
    for (std::size_t i = 0; i < x.size(); i++) d2 += (x[i] - y[i]) * (x[i] - y[i]);
    s.check("Spearman's ρ without ties against 1 − 6Σd²/(n(n² − 1))", spearman(x, y) - (1 - 6 * d2 / (7.0 * 48)), 1e-15);
    std::vector<double> ex(x.size()), cube(y.size());
    for (std::size_t i = 0; i < x.size(); i++) { ex[i] = std::exp(x[i]); cube[i] = y[i] * y[i] * y[i]; }
    s.check("ρ and τ_b are unchanged by increasing maps", std::fabs(spearman(ex, cube) - spearman(x, y)) + std::fabs(kendall_tau_b(ex, cube) - kendall_tau_b(x, y)), 1e-15);
    s.check("τ_b without ties: (n_c − n_d)/21 by counting pairs", kendall_tau_b(x, y) - tau_b_pairs(x, y), 1e-15);
  }
  {
    // With ties: x = (1, 2, 2, 3), y = (1, 3, 2, 2). Mid-ranks rx = (1, 2.5, 2.5, 4), ry = (1, 4, 2.5, 2.5);
    // centred (−1.5, 0, 0, 1.5) and (−1.5, 1.5, 0, 0): ρ = 2.25 / √(4.5 · 4.5) = 0.5.
    const std::vector<double> x = {1, 2, 2, 3}, y = {1, 3, 2, 2};
    s.check("Spearman's ρ with ties, by hand: 1/2", spearman(x, y) - 0.5, 1e-15);
    // Pairs: (1,2) c, (1,3) c, (1,4) c, (2,3) tied in x, (2,4) d, (3,4) tied in y: τ_b = (3 − 1)/√(5 · 5).
    s.check("Kendall's τ_b with ties, by hand: 2/5", kendall_tau_b(x, y) - 0.4, 1e-15);
  }
  {
    calx::random::Stream r(5, 0x52414e4b, 0, 0);
    double worst = 0;
    for (int trial = 0; trial < 200; trial++) {
      const std::size_t n = 2 + r.below(60);
      std::vector<double> x(n), y(n);
      for (std::size_t i = 0; i < n; i++) {
        x[i] = static_cast<double>(r.below(8));  // many ties
        y[i] = static_cast<double>(r.below(5)) + 0.5 * x[i];
      }
      const double a = kendall_tau_b(x, y), b = tau_b_pairs(x, y);
      if (std::isnan(a) && std::isnan(b)) continue;
      worst = std::max(worst, std::fabs(a - b));
    }
    s.check("Knight's τ_b against the count of pairs on 200 samples with ties", worst, 1e-12);
  }
  return s.done();
}
