// Checks of layers of a loss and of basis risk (layers.hpp).
//
// - A tower splits every loss into parts that sum to it exactly, and its layers' expected losses sum to the average annual loss.
// - An exponential law of mean μ: E L_[a, a+l] = μ (e^(−a/μ) − e^(−(a+l)/μ)); from 400,000 simulated years, within 1%.
// - Per event, the annual layer loss is compound Poisson: its variance Σ x² / years, met by simulating years of a Poisson count of
//   events drawn from the catalogue.
// - From a curve: the layer from 0 without limit is curve_area's total exactly; a tower's layers sum to it; on the curve of an
//   exponential law, L(T) = μ ln T, the layer converges to the closed form as the grid of return periods is refined.
// - Basis risk by hand on three samples; a payout equal to the loss has none.
// - A loss made by a linear payout on an index is recovered by fit_payout: the trigger, the exhaustion and no error.
// - Refusals: a negative loss, a layer of no limit, a tower that does not begin at 0, more yearly losses than years.

#include <tyche/descriptive/exceedance.hpp>
#include <tyche/descriptive/layers.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace td = tyche::descriptive;

namespace {
bool throws(auto&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}
}  // namespace

int main() {
  calx::verification::Suite s;
  const double inf = std::numeric_limits<double>::infinity();
  {
    calx::random::Stream rng(29, 1, 2, 3);
    std::vector<double> losses;
    for (int i = 0; i < 2000; i++) losses.push_back(rng.lognormal(10, 1.5));
    const std::vector<double> at{0, 5e4, 2e5, 1e6};
    const auto parts = td::split(losses, at);
    double worst = 0;
    for (std::size_t i = 0; i < losses.size(); i++) {
      double sum = 0;
      for (double p : parts[i]) sum += p;
      worst = std::max(worst, std::fabs(sum - losses[i]) / losses[i]);
    }
    s.check("a tower of layers splits every loss into parts that sum to it", worst, 1e-15);
    double expected = 0;
    for (std::size_t k = 0; k < at.size(); k++)
      expected += td::layer_statistics(losses, 5000, {at[k], k + 1 < at.size() ? at[k + 1] - at[k] : inf}).expected;
    s.check("its layers' expected losses sum to the average annual loss", expected / td::average_annual_loss(losses, 5000) - 1, 1e-13);
  }
  {
    const double mu = 1e6, a = 5e5, l = 1.5e6;
    calx::random::Stream rng(31, 4, 5, 6);
    std::vector<double> years;
    for (int i = 0; i < 400000; i++) years.push_back(mu * rng.exponential());
    const auto L = td::layer_statistics(years, 400000, {a, l}, td::Record::per_year);
    const double want = mu * (std::exp(-a / mu) - std::exp(-(a + l) / mu));
    s.check("an exponential law: the layer's expected loss is μ (e^(−a/μ) − e^(−(a+l)/μ)), from 400,000 years", L.expected / want - 1, 0.01);
    s.check("and the share of years it attaches is e^(−a/μ)", L.attachment_rate - std::exp(-a / mu), 0.003);
    s.check("and the share it exhausts is e^(−(a+l)/μ)", L.exhaustion_rate - std::exp(-(a + l) / mu), 0.003);
  }
  {
    calx::random::Stream rng(37, 7, 8, 9);
    std::vector<double> catalogue;
    for (int i = 0; i < 300; i++) catalogue.push_back(rng.lognormal(12, 1));
    const double years = 1000, a = 2e5, l = 4e5;
    const auto L = td::layer_statistics(catalogue, years, {a, l}, td::Record::per_event);
    double sum2 = 0;
    for (double v : catalogue) sum2 += std::pow(td::layer_loss(v, a, l), 2);
    s.check("per event, the variance a year is Σ x² over the years", L.sd * L.sd / (sum2 / years) - 1, 1e-14);
    const double rate = catalogue.size() / years;
    double m = 0, m2 = 0;
    const int Y = 200000;
    for (int y = 0; y < Y; y++) {
      const auto k = rng.poisson(rate);
      double total = 0;
      for (std::int64_t j = 0; j < k; j++) total += td::layer_loss(catalogue[rng.below(catalogue.size())], a, l);
      m += total;
      m2 += total * total;
    }
    m /= Y;
    const double var = m2 / Y - m * m;
    s.check("simulated Poisson years of the catalogue have that mean", m / L.expected - 1, 0.02);
    s.check("and that variance", var / (L.sd * L.sd) - 1, 0.03);
  }
  {
    const std::vector<double> T{2, 5, 10, 20, 50, 100, 250, 500, 1000}, Lc{0, 1e5, 3e5, 6e5, 1.1e6, 1.6e6, 2.4e6, 3e6, 3.7e6};
    const double area = td::curve_area(T, Lc, 0, true).total();
    s.check("from a curve, the layer from 0 without limit is the curve's area", td::layer_from_curve(T, Lc, {0, inf}) / area - 1, 1e-14);
    const std::vector<double> at{0, 2e5, 1e6, 2.5e6};
    double sum = 0;
    for (std::size_t k = 0; k < at.size(); k++) sum += td::layer_from_curve(T, Lc, {at[k], k + 1 < at.size() ? at[k + 1] - at[k] : inf});
    s.check("a tower's layers from a curve sum to its area", sum / area - 1, 1e-14);
    const double mu = 1e6, a = 1e6, l = 2e6, want = mu * (std::exp(-a / mu) - std::exp(-(a + l) / mu));
    const auto layer = [&](double step) {
      std::vector<double> RT, RL;
      for (double x = std::log(1.01); x <= std::log(1e5) + 1e-9; x += step) {
        RT.push_back(std::exp(x));
        RL.push_back(mu * x);
      }
      return td::layer_from_curve(RT, RL, {a, l}, 0, true);
    };
    const double e1 = std::fabs(layer(0.2) - want), e2 = std::fabs(layer(0.05) - want);
    s.check("on the curve L(T) = μ ln T the layer approaches μ (e^(−a/μ) − e^(−(a+l)/μ)) as the grid is refined", e2 / want, 2e-3);
    s.require("and its error falls as the spacing in log T falls", e2 < e1 / 4);
  }
  {
    const auto B = td::basis_risk({0, 5, 10}, {10, 0, 10});
    s.require("basis risk by hand: |P − C| sums to 15, the shortfall 10, the excess 5, one loss missed and one payout unearned",
              B.absolute == 15 && B.shortfall == 10 && B.excess == 5 && B.missed == 1 && B.unearned == 1 && B.paid == 15 && B.covered == 20);
    const auto P = td::basis_risk({1, 2, 3, 7}, {1, 2, 3, 7}, {0.1, 0.2, 0.3, 0.4});
    s.require("a payout equal to the loss has no basis risk and a correlation of one", P.absolute == 0 && P.rmse == 0 && std::fabs(P.correlation - 1) < 1e-12);
  }
  {
    std::vector<double> index, covered;
    const td::Payout truth{30, 70, 100};
    for (int i = 0; i < 100; i++) {
      index.push_back(i);
      covered.push_back(truth(i));
    }
    const auto F = td::fit_payout(index, covered, 100, {}, 99);
    s.require("a loss made by a linear payout on its index is recovered: trigger 30, exhaustion 70, no error",
              F.payout.trigger == 30 && F.payout.exhaustion == 70 && F.risk.rmse < 1e-12 && F.tried > 1000);
  }
  {
    s.require("refusals: a negative loss, a layer of no limit, a tower not from 0, more yearly losses than years",
              throws([] { td::layer_statistics({-1}, 10, {0, 1}); }) && throws([] { td::layer_statistics({1}, 10, {0, 0}); }) &&
                  throws([] { td::split({1}, {5, 10}); }) && throws([] { td::layer_statistics({1, 2, 3}, 2, {0, 1}, td::Record::per_year); }));
  }
  return s.done();
}
