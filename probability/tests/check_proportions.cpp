// Checks of intervals for a proportion (proportions.hpp): Wilson's bounds as the roots of the score equation, its closed forms at
// none and all, a value worked by hand; the Clopper–Pearson bounds as the points where the binomial's tails hold α/2 each, its
// closed forms at none and all; both intervals holding the estimate; exact coverage of both over a
// grid of true proportions, Clopper–Pearson's never under its level; and what is refused.

#include <tyche/probability/proportions.hpp>

#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

using namespace tyche::probability;

int main() {
  calx::verification::Suite s;
  const double z = Normal{}.quantile(0.975);

  // ---- Wilson ---------------------------------------------------------------------------------------------------------------------------
  {
    double worst = 0;
    for (std::int64_t n : {1, 5, 12, 40, 400})
      for (std::int64_t k = 1; k < n; k++) {
        const auto [lo, hi] = wilson_interval(k, n);
        const double p = static_cast<double>(k) / static_cast<double>(n);
        for (const double b : {lo, hi}) worst = std::max(worst, std::abs(std::abs(p - b) - z * std::sqrt(b * (1 - b) / static_cast<double>(n))));
      }
    s.check("Wilson's bounds are the roots of the score equation |k/n − p| = z √(p(1 − p)/n) (every k of five sample sizes)", worst, 1e-13);
    const auto none = wilson_interval(0, 10), all = wilson_interval(10, 10);
    s.check("none of ten: from 0 to z²/(n + z²) = 0.2775", std::abs(none.second - z * z / (10 + z * z)) + none.first, 1e-14);
    s.check("all of ten: from n/(n + z²) to 1", std::abs(all.first - 10 / (10 + z * z)) + std::abs(all.second - 1), 1e-14);
    // 7 of 20: p̂ = 0.35, z = 1.959964: centre (0.35 + 0.096036)/1.192073 = 0.374169; half 1.959964 × √(0.011375 + 0.002401)/1.192073
    // = 0.192978; the interval 0.1812 to 0.5671.
    const auto seven = wilson_interval(7, 20);
    s.check("seven of twenty by hand: 0.1812", std::abs(seven.first - 0.181192), 5e-6);
    s.check("to 0.5671", std::abs(seven.second - 0.567147), 5e-6);
    const auto wide = wilson_interval(7, 20, 0.99), narrow = wilson_interval(7, 20, 0.5);
    s.require("a greater confidence gives a wider interval", wide.first < seven.first && wide.second > seven.second && narrow.first > seven.first && narrow.second < seven.second);
  }

  // ---- Clopper and Pearson -------------------------------------------------------------------------------------------------------------
  {
    double worst = 0;
    bool wider = true;
    for (std::int64_t n : {1, 5, 12, 40, 400})
      for (std::int64_t k = 0; k <= n; k++) {
        const auto [lo, hi] = clopper_pearson_interval(k, n);
        if (k > 0) worst = std::max(worst, std::abs(Binomial{n, lo}.survival(k - 1) - 0.025));  // P(X ≥ k | lo) = α/2
        if (k < n) worst = std::max(worst, std::abs(Binomial{n, hi}.cdf(k) - 0.025));           // P(X ≤ k | hi) = α/2
        const auto w = wilson_interval(k, n);
        const double est = static_cast<double>(k) / static_cast<double>(n);
        wider = wider && lo <= est && est <= hi && w.first <= est && est <= w.second && 0 <= lo && hi <= 1;
      }
    s.check("the exact bounds are where the binomial's tails hold α/2 each (every k of five sample sizes)", worst, 1e-9);
    s.require("both intervals hold the estimate k/n, within the unit interval", wider);
    const auto none = clopper_pearson_interval(0, 10), all = clopper_pearson_interval(10, 10);
    s.check("none of ten: from 0 to 1 − 0.025^(1/10) = 0.3085", std::abs(none.second - (1 - std::pow(0.025, 0.1))) + none.first, 1e-10);
    s.check("all of ten: from 0.025^(1/10) to 1", std::abs(all.first - std::pow(0.025, 0.1)) + std::abs(all.second - 1), 1e-10);
  }

  // ---- Coverage, exactly ------------------------------------------------------------------------------------------------------------------
  {
    // The probability, at a true p, that the interval from the count holds p: the sum of the binomial's mass over the counts whose
    // interval does.
    double least_exact = 1, mean_wilson = 0, least_wilson = 1;
    int points = 0;
    for (const std::int64_t n : {10, 30, 100})
      for (double p = 0.02; p < 0.99; p += 0.03) {
        double exact = 0, wilson = 0;
        for (std::int64_t k = 0; k <= n; k++) {
          const double mass = Binomial{n, p}.pmf(k);
          const auto c = clopper_pearson_interval(k, n);
          const auto w = wilson_interval(k, n);
          if (c.first <= p && p <= c.second) exact += mass;
          if (w.first <= p && p <= w.second) wilson += mass;
        }
        least_exact = std::min(least_exact, exact), least_wilson = std::min(least_wilson, wilson);
        mean_wilson += wilson, points++;
      }
    s.require("the exact interval covers the true proportion with probability 0.95 at least, at every proportion and size tried", least_exact >= 0.95 - 1e-9);
    s.check("Wilson's covers it with 0.95 on average, within a point and a half", std::abs(mean_wilson / points - 0.95), 0.015);
    s.require("and somewhere with less than 0.95, as a score interval does, though never under 0.8 at the points tried", least_wilson > 0.8 && least_wilson < 0.95);
  }

  bool r1 = false, r2 = false, r3 = false;
  try { (void)wilson_interval(3, 0); } catch (const std::invalid_argument&) { r1 = true; }
  try { (void)clopper_pearson_interval(11, 10); } catch (const std::invalid_argument&) { r2 = true; }
  try { (void)wilson_interval(3, 10, 1.0); } catch (const std::invalid_argument&) { r3 = true; }
  s.require("no trials, more successes than trials, and a confidence of one are refused", r1 && r2 && r3);
  return s.done();
}
