// Checks of tyche/time_series/changepoint.hpp: change points by optimal partitioning, PELT, binary,
// wild binary and circular binary segmentation.
//
// - The costs in closed form: each model's cost of a segment against its −2 log-likelihood computed
//   directly (up to the stated terms in the observations alone); the cost inequality pruning needs.
// - The noise variance from the MAD of first differences, on iid normal noise with mean shifts.
// - Exactness by brute force: on 300 short series (n ≤ 12), every segmentation enumerated, for every
//   model, the penalties BIC, MBIC and manual and minimum lengths 1–3: optimal partitioning and PELT
//   reach the least objective and return the same segmentation.
// - PELT equals optimal partitioning on 200 longer series (n up to 600), every model, minimum lengths
//   1–6: the same change points and the same objective, and PELT evaluates fewer costs.
// - ruptures 1.1.10 (cross-check, fixtures/changepoint-reference.txt): PELT with the l2 and normal
//   costs, and binary segmentation, give ruptures' breakpoints.
// - Planted changes recovered: changes in mean (σ² estimated), in variance, in mean and variance, in a
//   Poisson rate with exposures; a joint segmentation of two series; binary and wild binary
//   segmentation; WBS finding the short middle segments that binary segmentation's first step misses.
// - CBS: the maximal statistic by branch and bound equal to the brute-force maximum over all arcs, and
//   the threshold search's decision equal to brute force; the t statistic in closed form; the
//   early-rejection boundary's hypergeometric levels and its exact added size; the size of the first
//   split's test under the null; planted arcs recovered; pruning and undoing; determinism.
// - Speed for n = 10⁵ (reported).

#include <tyche/time_series/changepoint.hpp>

#include <calx/random/stream.hpp>
#include <calx/special_functions/gamma.hpp>
#include <calx/verification/check.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace ts = tyche::time_series;

namespace {

std::vector<double> planted(calx::random::Stream& z, std::size_t n, const std::vector<std::size_t>& cps, const std::vector<double>& means, const std::vector<double>& sds) {
  std::vector<double> y(n);
  std::size_t k = 0;
  for (std::size_t i = 0; i < n; i++) {
    while (k < cps.size() && i >= cps[k]) k++;
    y[i] = means[k] + sds[k] * z.normal();
  }
  return y;
}

// The objective of a segmentation computed directly from the cost.
double objective_of(const ts::SegmentCost& c, const ts::PenaltyTerms& p, const std::vector<std::size_t>& cps) {
  double s = 0;
  std::size_t b = 0;
  for (std::size_t k = 0; k <= cps.size(); k++) {
    const std::size_t e = k < cps.size() ? cps[k] : c.size();
    s += c(b, e) + (p.per_segment != 0 ? p.per_segment * std::log(static_cast<double>(e - b) / static_cast<double>(p.n)) : 0.0);
    b = e;
  }
  return s + p.per_changepoint * static_cast<double>(cps.size());
}

// The brute-force maximum of |Z| over all admissible arcs of y.
ts::ArcStatistic brute_arc(const std::vector<double>& y, std::size_t w) {
  const std::size_t m = y.size();
  double mean = 0;
  for (const double v : y) mean += v;
  mean /= static_cast<double>(m);
  std::vector<double> S(m + 1, 0.0);
  for (std::size_t i = 0; i < m; i++) S[i + 1] = S[i] + (y[i] - mean);
  ts::ArcStatistic best;
  double top = -1;
  auto start_ok = [&](std::size_t i) { return i == 0 || (i >= w && i + w <= m); };
  auto end_ok = [&](std::size_t j) { return j == m || (j >= w && j + w <= m); };
  for (std::size_t i = 0; i < m; i++)
    for (std::size_t j = i + 1; j <= m; j++) {
      const std::size_t k = j - i;
      if (!start_ok(i) || !end_ok(j) || k < w || k + w > m) continue;
      const double z = (S[j] - S[i]) / std::sqrt(static_cast<double>(k) * static_cast<double>(m - k) / static_cast<double>(m));
      if (std::fabs(z) > top) {
        top = std::fabs(z);
        best.begin = i;
        best.end = j;
        best.z = z;
      }
    }
  return best;
}

double seconds_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

}  // namespace

int main() {
  calx::verification::Suite s;
  const double two_pi = 2 * std::numbers::pi;

  // ---- the costs in closed form ----
  {
    calx::random::Stream z(1, 1, 0, 0);
    std::vector<double> y(40), w(40), e(40), cnt(40);
    for (std::size_t i = 0; i < 40; i++) {
      y[i] = 1.5 + z.normal();
      w[i] = 0.5 + z.uniform();
      e[i] = 0.3 + 2 * z.uniform();
      cnt[i] = static_cast<double>(z.poisson(3 * e[i]));
    }
    const std::size_t a = 7, b = 31;
    const double len = static_cast<double>(b - a);
    double worst = 0, wmean = 0, wvar = 0, wmv = 0, wpois = 0;
    {  // mean, weighted, σ² = 2.5
      ts::CostOptions o;
      o.variance = 2.5;
      o.weights = w;
      const ts::SegmentCost c(y, o);
      double W = 0, A = 0;
      for (std::size_t i = a; i < b; i++) W += w[i], A += w[i] * y[i];
      const double m = A / W;
      double rss = 0;
      for (std::size_t i = a; i < b; i++) rss += w[i] * (y[i] - m) * (y[i] - m);
      wmean = std::max({std::fabs(c(a, b) - rss / 2.5), std::fabs(c.segment(a, b).mean - m)});
    }
    {  // mean_variance: −2ℓ at the MLE
      ts::CostOptions o;
      o.model = ts::SegmentModel::mean_variance;
      const ts::SegmentCost c(y, o);
      double m = 0;
      for (std::size_t i = a; i < b; i++) m += y[i] / len;
      double rss = 0;
      for (std::size_t i = a; i < b; i++) rss += (y[i] - m) * (y[i] - m);
      const double s2 = rss / len;
      double ll = 0;
      for (std::size_t i = a; i < b; i++) ll += -0.5 * std::log(two_pi * s2) - (y[i] - m) * (y[i] - m) / (2 * s2);
      wmv = std::fabs(c(a, b) + 2 * ll);
    }
    {  // variance with known mean 1.5
      ts::CostOptions o;
      o.model = ts::SegmentModel::variance;
      o.mean = 1.5;
      const ts::SegmentCost c(y, o);
      double q = 0;
      for (std::size_t i = a; i < b; i++) q += (y[i] - 1.5) * (y[i] - 1.5);
      const double s2 = q / len;
      double ll = 0;
      for (std::size_t i = a; i < b; i++) ll += -0.5 * std::log(two_pi * s2) - (y[i] - 1.5) * (y[i] - 1.5) / (2 * s2);
      wvar = std::fabs(c(a, b) + 2 * ll);
    }
    {  // poisson with exposures: C = −2ℓ + 2Σ(yᵢ log eᵢ − log yᵢ!)
      ts::CostOptions o;
      o.model = ts::SegmentModel::poisson;
      o.exposure = e;
      const ts::SegmentCost c(cnt, o);
      double Y = 0, E = 0;
      for (std::size_t i = a; i < b; i++) Y += cnt[i], E += e[i];
      const double lam = Y / E;
      double ll = 0, extra = 0;
      for (std::size_t i = a; i < b; i++) {
        ll += cnt[i] * std::log(lam * e[i]) - lam * e[i] - calx::special_functions::log_gamma(cnt[i] + 1);
        extra += cnt[i] * std::log(e[i]) - calx::special_functions::log_gamma(cnt[i] + 1);
      }
      wpois = std::max(std::fabs(c(a, b) - (-2 * ll + 2 * extra)), std::fabs(c.segment(a, b).mean - lam));
    }
    worst = std::max({wmean, wvar, wmv, wpois});
    s.check("costs: mean (weighted, σ² given) as Σ w(y − ȳ_w)²/σ², and its weighted mean", wmean, 1e-10);
    s.check("costs: variance (known mean) as −2ℓ at the MLE", wvar, 1e-9);
    s.check("costs: mean_variance as −2ℓ at the MLE", wmv, 1e-9);
    s.check("costs: poisson with exposures as −2ℓ + 2Σ(yᵢ log eᵢ − log yᵢ!), and its rate Σy/Σe", wpois, 1e-9);
    (void)worst;
    // The inequality C(a, t) + C(t, b) ≤ C(a, b) for every model and split.
    double violation = 0;
    for (const auto model : {ts::SegmentModel::mean, ts::SegmentModel::variance, ts::SegmentModel::mean_variance, ts::SegmentModel::poisson}) {
      ts::CostOptions o;
      o.model = model;
      const ts::SegmentCost c(model == ts::SegmentModel::poisson ? cnt : y, o);
      for (std::size_t aa = 0; aa < 40; aa += 3)
        for (std::size_t bb = aa + 4; bb <= 40; bb += 3)
          for (std::size_t t = aa + 2; t + 2 <= bb; t++) violation = std::max(violation, c(aa, t) + c(t, bb) - c(aa, bb));
    }
    s.check("costs: a split never raises the cost (the inequality PELT's pruning needs)", std::max(0.0, violation), 1e-9);
  }

  // ---- the noise variance ----
  {
    calx::random::Stream z(2, 0, 0, 0);
    const auto y = planted(z, 20000, {5000, 9000, 15000}, {0, 3, -2, 1}, {2, 2, 2, 2});
    s.check("robust_variance: σ² = 4 from the MAD of first differences despite three shifts (relative)", ts::robust_variance(y) / 4 - 1, 0.03);
    std::vector<double> w(20000, 4.0), yw(20000);
    for (std::size_t i = 0; i < 20000; i++) yw[i] = z.normal(0, 1);  // Var = 1 = σ²/w with σ² = 4
    s.check("robust_variance: with precisions wᵢ = 4, σ² = 4 (relative)", ts::robust_variance(yw, w) / 4 - 1, 0.03);
  }

  // ---- exactness against brute force ----
  {
    calx::random::Stream z(3, 0, 0, 0);
    std::size_t cases = 0, wrong_op = 0, wrong_pelt = 0, differ = 0;
    double gap = 0;
    for (int rep = 0; rep < 300; rep++) {
      const std::size_t n = 4 + z.below(9);  // 4..12
      std::vector<double> y(n), cnt(n);
      const double jump = z.uniform(0, 3);
      for (std::size_t i = 0; i < n; i++) {
        y[i] = (i > n / 2 ? jump : 0.0) + z.normal() * (i % 3 == 0 ? 1.5 : 1.0);
        cnt[i] = static_cast<double>(z.poisson(i > n / 3 ? 6.0 : 2.0));
      }
      const auto model = static_cast<ts::SegmentModel>(rep % 4);
      ts::CostOptions co;
      co.model = model;
      if (model == ts::SegmentModel::mean) co.variance = 0.8;
      const auto& data = model == ts::SegmentModel::poisson ? cnt : y;
      const ts::SegmentCost c(data, co);
      for (const auto pen : {ts::Penalty::bic, ts::Penalty::mbic, ts::Penalty::manual}) {
        const auto terms = ts::penalty_terms(pen, n, c.parameters(), 1.7);
        for (std::size_t ms = std::max<std::size_t>(1, c.min_size()); ms <= 3; ms++) {
          if (n < ms) continue;
          // Enumerate all subsets of {1..n−1} as change points with every segment ≥ ms.
          double best = std::numeric_limits<double>::infinity();
          for (std::uint32_t mask = 0; mask < (1u << (n - 1)); mask++) {
            std::vector<std::size_t> cps;
            for (std::size_t t = 1; t < n; t++)
              if (mask & (1u << (t - 1))) cps.push_back(t);
            bool ok = true;
            std::size_t b = 0;
            for (std::size_t k = 0; k <= cps.size() && ok; k++) {
              const std::size_t e = k < cps.size() ? cps[k] : n;
              ok = e - b >= ms;
              b = e;
            }
            if (ok) best = std::min(best, objective_of(c, terms, cps));
          }
          const auto op = ts::optimal_partitioning(c, terms, ms);
          const auto pe = ts::pelt(c, terms, ms);
          cases++;
          const double tol = 1e-9 * (1 + std::fabs(best));
          if (std::fabs(op.objective - best) > tol) wrong_op++;
          if (std::fabs(pe.objective - best) > tol) wrong_pelt++;
          if (op.changepoints != pe.changepoints) differ++;
          gap = std::max(gap, std::fabs(objective_of(c, terms, op.changepoints) - op.objective));
        }
      }
    }
    s.require("brute force: optimal partitioning reaches the least objective over all segmentations (" + std::to_string(cases) + " cases)", wrong_op == 0, std::to_string(wrong_op) + " wrong");
    s.require("brute force: PELT reaches the least objective", wrong_pelt == 0, std::to_string(wrong_pelt) + " wrong");
    s.require("brute force: PELT and optimal partitioning return the same segmentation", differ == 0, std::to_string(differ) + " differ");
    s.check("brute force: the objective reported is the segmentation's", gap, 1e-9);
  }

  // ---- PELT equals optimal partitioning on longer series ----
  {
    calx::random::Stream z(4, 0, 0, 0);
    std::size_t differ = 0, more = 0, eval_op = 0, eval_pelt = 0;
    double worst = 0;
    for (int rep = 0; rep < 200; rep++) {
      const std::size_t n = 100 + z.below(501);
      const std::size_t m = z.below(8);
      std::vector<std::size_t> cps;
      for (std::size_t k = 0; k < m; k++) cps.push_back(10 + z.below(n - 20));
      std::sort(cps.begin(), cps.end());
      std::vector<double> means, sds;
      for (std::size_t k = 0; k <= m; k++) means.push_back(z.normal(0, 1.5)), sds.push_back(std::exp(z.normal(0, 0.4)));
      auto y = planted(z, n, cps, means, sds);
      const auto model = static_cast<ts::SegmentModel>(rep % 4);
      if (model == ts::SegmentModel::poisson)
        for (auto& v : y) v = static_cast<double>(z.poisson(std::exp(0.5 * v + 1)));
      if (model == ts::SegmentModel::variance)
        for (std::size_t i = 0; i < n; i++) y[i] -= means[std::upper_bound(cps.begin(), cps.end(), i) - cps.begin()];
      ts::ChangepointOptions o;
      o.cost.model = model;
      o.penalty = rep % 3 == 0 ? ts::Penalty::mbic : rep % 3 == 1 ? ts::Penalty::bic : ts::Penalty::aic;
      o.min_size = 1 + z.below(6);
      const auto a = ts::optimal_partitioning(y, o);
      const auto b = ts::pelt(y, o);
      if (a.changepoints != b.changepoints) differ++;
      worst = std::max(worst, std::fabs(a.objective - b.objective) / (1 + std::fabs(a.objective)));
      if (b.evaluations > a.evaluations) more++;
      eval_op += a.evaluations;
      eval_pelt += b.evaluations;
    }
    s.require("PELT = optimal partitioning on 200 series (n ≤ 600, four models, three penalties, minimum lengths 1–6)", differ == 0, std::to_string(differ) + " differ");
    s.check("PELT = optimal partitioning: the objectives (relative)", worst, 1e-12);
    s.require("PELT never evaluates more segment costs than optimal partitioning, and fewer in all (" + std::to_string(eval_pelt) + " against " + std::to_string(eval_op) + ")",
              more == 0 && eval_pelt < eval_op, std::to_string(more) + " evaluated more");
  }

  // ---- ruptures (cross-check) ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/changepoint-reference.txt");
    s.require("ruptures fixtures: the reference file opens", static_cast<bool>(in));
    std::string line;
    std::size_t cases = 0, agree = 0;
    std::string failures;
    while (std::getline(in, line)) {
      if (line.rfind("case ", 0) != 0) continue;
      std::istringstream h(line.substr(5));
      std::string algo, cost;
      double pen;
      std::size_t ms, n;
      h >> algo >> cost >> pen >> ms >> n;
      std::string ys, bs;
      std::getline(in, ys);
      std::getline(in, bs);
      std::vector<double> y;
      std::istringstream yi(ys);
      for (double v; yi >> v;) y.push_back(v);
      std::vector<std::size_t> bk;
      std::istringstream bi(bs);
      for (std::size_t v; bi >> v;) bk.push_back(v);
      bk.pop_back();  // ruptures ends with n
      ts::ChangepointOptions o;
      o.penalty = ts::Penalty::manual;
      o.beta = pen;
      o.min_size = ms;
      if (cost == "l2") o.cost.variance = 1;
      else o.cost.model = ts::SegmentModel::mean_variance;
      const auto got = algo == "pelt" ? ts::pelt(y, o) : ts::binary_segmentation(y, o);
      cases++;
      if (got.changepoints == bk) agree++;
      else failures += " " + algo + "/" + cost + "/n" + std::to_string(n);
    }
    s.require("ruptures 1.1.10: PELT (l2, normal) and binary segmentation (l2) give its breakpoints (" + std::to_string(agree) + " of " + std::to_string(cases) + ")",
              cases == 40 && agree == cases, failures);
  }

  // ---- planted changes recovered ----
  {
    calx::random::Stream z(5, 0, 0, 0);
    const std::vector<std::size_t> truth{300, 550, 900, 1400};
    auto near = [&](const std::vector<std::size_t>& got, std::size_t slack) {
      if (got.size() != truth.size()) return false;
      for (std::size_t k = 0; k < truth.size(); k++)
        if ((got[k] > truth[k] ? got[k] - truth[k] : truth[k] - got[k]) > slack) return false;
      return true;
    };
    auto join = [](const std::vector<std::size_t>& v) {
      std::string t;
      for (const auto x : v) t += " " + std::to_string(x);
      return t;
    };
    const auto ym = planted(z, 2000, truth, {0, 1.2, -0.4, 0.9, 0}, {0.5, 0.5, 0.5, 0.5, 0.5});
    const auto pm = ts::pelt(ym);
    s.require("planted means: PELT with BIC (σ² estimated) finds the four changes within 3", near(pm.changepoints, 3), join(pm.changepoints));
    s.check("planted means: σ² estimated (relative to 0.25; the MAD's standard error at n = 2000 is about 5%)", pm.variance / 0.25 - 1, 0.15);
    const auto pmb = ts::pelt(ym, ts::ChangepointOptions{{}, ts::Penalty::mbic});
    s.require("planted means: PELT with MBIC finds the same four", near(pmb.changepoints, 3), join(pmb.changepoints));
    const auto bs = ts::binary_segmentation(ym);
    auto covers = [&](const std::vector<std::size_t>& got, std::size_t slack) {
      for (const std::size_t t : truth)
        if (std::none_of(got.begin(), got.end(), [&](std::size_t g) { return (g > t ? g - t : t - g) <= slack; })) return false;
      return true;
    };
    s.require("planted means: binary segmentation finds each change within 3 (greedy, it may add one)", covers(bs.changepoints, 3) && bs.changepoints.size() <= 5, join(bs.changepoints));
    s.require("binary segmentation: its objective is no lower than PELT's (PELT is exact)", bs.objective >= pm.objective - 1e-9 * std::fabs(pm.objective));
    s.require("binary segmentation: one gain per change point, each above β", bs.scores.size() == bs.changepoints.size() &&
              std::all_of(bs.scores.begin(), bs.scores.end(), [&](double g) { return g > bs.penalty; }));
    ts::ChangepointOptions two;
    two.max_changepoints = 2;
    s.require("binary segmentation: max_changepoints stops it", ts::binary_segmentation(ym, two).changepoints.size() == 2);
    const auto wb = ts::wild_binary_segmentation(ym);
    s.require("planted means: wild binary segmentation finds them", near(wb.changepoints, 3), join(wb.changepoints));
    const auto wb2 = ts::wild_binary_segmentation(ym);
    s.require("wild binary segmentation: the same seed gives the same answer", wb.changepoints == wb2.changepoints && wb.scores == wb2.scores);

    const auto yv = planted(z, 2000, truth, {0, 0, 0, 0, 0}, {1, 3, 0.7, 2, 1});
    ts::ChangepointOptions ov;
    ov.cost.model = ts::SegmentModel::variance;
    ov.min_size = 5;
    const auto pv = ts::pelt(yv, ov);
    s.require("planted variances: PELT (variance, known mean) finds the four changes within 25", near(pv.changepoints, 25), join(pv.changepoints));
    const auto ymv = planted(z, 2000, truth, {0, 1, 1, -1, 0}, {1, 1, 2.5, 2.5, 0.6});
    ts::ChangepointOptions omv;
    omv.cost.model = ts::SegmentModel::mean_variance;
    omv.min_size = 5;
    const auto pmv = ts::pelt(ymv, omv);
    s.require("planted means and variances: PELT (mean_variance) finds the four within 25", near(pmv.changepoints, 25), join(pmv.changepoints));
    // Poisson: counts with exposures varying along the series (expected counts), the rate ×1.5, ×0.5, ×1.
    std::vector<double> cnt(2000), ex(2000);
    const std::vector<double> rate{1, 1.5, 0.5, 1.5, 1};
    for (std::size_t i = 0; i < 2000; i++) {
      ex[i] = 30 * (1 + 0.4 * std::sin(i / 37.0));
      const std::size_t k = std::upper_bound(truth.begin(), truth.end(), i) - truth.begin();
      cnt[i] = static_cast<double>(z.poisson(rate[k] * ex[i]));
    }
    ts::ChangepointOptions op;
    op.cost.model = ts::SegmentModel::poisson;
    op.cost.exposure = ex;
    const auto pp = ts::pelt(cnt, op);
    s.require("planted Poisson rates with exposures: PELT finds the four changes within 3", near(pp.changepoints, 3), join(pp.changepoints));
    double rate_err = 0;
    for (std::size_t k = 0; k < pp.segments.size() && pp.segments.size() == 5; k++) rate_err = std::max(rate_err, std::fabs(pp.segments[k].mean / rate[k] - 1));
    s.check("planted Poisson rates: the segments' rates λ̂ (relative)", rate_err, 0.03);
    // A joint segmentation of two series sharing the change points.
    const auto y1 = planted(z, 2000, truth, {0, 1, 1, 0, 1}, {1, 1, 1, 1, 1});
    const auto y2 = planted(z, 2000, truth, {0, 0, -1, -1, 0}, {1, 1, 1, 1, 1});
    ts::CostOptions unit;
    unit.variance = 1;
    const ts::SumCost joint({ts::SegmentCost(y1, unit), ts::SegmentCost(y2, unit)});
    const auto pj = ts::pelt(joint, ts::penalty_terms(ts::Penalty::bic, 2000, joint.parameters()), 1);
    s.require("joint segmentation: two series, each with only some of the changes, give all four within 20", near(pj.changepoints, 20), join(pj.changepoints));
    const auto jt = ts::penalty_terms(ts::Penalty::bic, 2000, joint.parameters());
    double at_truth = 0;
    std::size_t lo = 0;
    for (std::size_t k = 0; k <= truth.size(); k++) {
      const std::size_t hi = k < truth.size() ? truth[k] : 2000;
      at_truth += joint(lo, hi);
      lo = hi;
    }
    at_truth += jt.per_changepoint * static_cast<double>(truth.size());
    s.require("joint segmentation: the optimum's objective is no higher than the truth's", pj.objective <= at_truth + 1e-9 * std::fabs(at_truth));
  }
  {
    // Wild binary segmentation where binary segmentation's first step goes astray: the short middle
    // segments of Fearnhead and Fryzlewicz's (2024, Figure 5) kind, changes at 80, 100, 120 of 200.
    calx::random::Stream z(6, 0, 0, 0);
    std::size_t wbs_right = 0, bs_right = 0;
    for (int rep = 0; rep < 50; rep++) {
      const auto y = planted(z, 200, {80, 100, 120}, {0, 3, -3, 0}, {1.2, 1.2, 1.2, 1.2});
      ts::ChangepointOptions o;
      o.cost.variance = 1.44;
      auto ok = [](const std::vector<std::size_t>& c) {
        if (c.size() != 3) return false;
        const std::size_t t[3] = {80, 100, 120};
        for (int k = 0; k < 3; k++)
          if ((c[k] > t[k] ? c[k] - t[k] : t[k] - c[k]) > 3) return false;
        return true;
      };
      ts::WbsOptions wo;
      wo.seed = static_cast<std::uint64_t>(rep + 1);
      if (ok(ts::wild_binary_segmentation(y, o, wo).changepoints)) wbs_right++;
      if (ok(ts::binary_segmentation(y, o).changepoints)) bs_right++;
    }
    s.require("short middle segments (80, 100, 120 of 200): WBS right in ≥ 45 of 50", wbs_right >= 45, std::to_string(wbs_right) + " (binary segmentation " + std::to_string(bs_right) + ")");
    // The threshold rule: Fryzlewicz's C = 1.3, a gain of 2 C² log n.
    const auto y = planted(z, 200, {80, 100, 120}, {0, 3, -3, 0}, {1.2, 1.2, 1.2, 1.2});
    ts::ChangepointOptions o;
    o.cost.variance = 1.44;
    ts::WbsOptions wo;
    wo.threshold = 2 * 1.3 * 1.3 * std::log(200.0);
    const auto wt = ts::wild_binary_segmentation(y, o, wo);
    s.require("WBS with Fryzlewicz's threshold: three change points, every gain above it", wt.changepoints.size() == 3 &&
              std::all_of(wt.scores.begin(), wt.scores.end(), [&](double g) { return g >= wo.threshold; }));
  }

  // ---- circular binary segmentation ----
  {
    calx::random::Stream z(7, 0, 0, 0);
    std::size_t differ = 0, cases = 0, decisions_wrong = 0;
    for (int rep = 0; rep < 400; rep++) {
      const std::size_t m = 4 + z.below(rep < 300 ? 60 : 300);
      const std::size_t w = 1 + z.below(3);
      if (m < 2 * w + 1) continue;
      std::vector<double> y(m);
      const std::size_t a = z.below(m), b = std::min(m, a + 1 + z.below(m));
      for (std::size_t i = 0; i < m; i++) y[i] = z.normal() + (i >= a && i < b ? z.uniform(0, 2) : 0.0);
      const auto fast = ts::cbs_statistic(y, w);
      const auto slow = brute_arc(y, w);
      cases++;
      // An arc [0, j) and its complement [j, m) are one split: compare the change points they make.
      auto cuts = [&](const ts::ArcStatistic& a) {
        std::vector<std::size_t> c;
        if (a.begin > 0) c.push_back(a.begin);
        if (a.end < m) c.push_back(a.end);
        return c;
      };
      if (cuts(fast) != cuts(slow) || std::fabs(std::fabs(fast.z) - std::fabs(slow.z)) > 1e-9 * (1 + std::fabs(slow.z))) differ++;
      // The threshold search's decision, at a threshold just above and just below the maximum.
      double tss = 0;
      const auto S = ts::changepoint_detail::centred_partial_sums(y, tss);
      for (const double f : {0.999, 1.001, 0.7}) {
        ts::changepoint_detail::ArcSearch search(S, w);
        const auto hit = search.run(std::fabs(slow.z) * f);
        const bool says = std::fabs(hit.z) >= std::fabs(slow.z) * f;
        if (says != (f < 1)) decisions_wrong++;
      }
    }
    s.require("CBS: the branch-and-bound maximum equals the brute-force maximum over all arcs (" + std::to_string(cases) + " series, m ≤ 303, widths 1–3)", differ == 0,
              std::to_string(differ) + " differ");
    s.require("CBS: the threshold search decides as brute force does (thresholds 0.7, 0.999, 1.001 of the maximum)", decisions_wrong == 0, std::to_string(decisions_wrong) + " wrong");
    // The t statistic in closed form: Welch-free pooled two-sample t of the arc against the rest.
    std::vector<double> y(50);
    for (std::size_t i = 0; i < 50; i++) y[i] = z.normal() + (i >= 12 && i < 30 ? 1.5 : 0.0);
    const auto a = ts::cbs_statistic(y, 2);
    double m1 = 0, m2 = 0;
    const double k = static_cast<double>(a.end - a.begin), r = 50 - k;
    for (std::size_t i = 0; i < 50; i++) (i >= a.begin && i < a.end ? m1 : m2) += y[i];
    m1 /= k;
    m2 /= r;
    double ss = 0;
    for (std::size_t i = 0; i < 50; i++) ss += std::pow(y[i] - (i >= a.begin && i < a.end ? m1 : m2), 2);
    const double t = (m1 - m2) / std::sqrt(ss / 48 * (1 / k + 1 / r));
    s.check("CBS: t of the maximising arc equals the pooled two-sample t", a.t - t, 1e-10);
    const auto ab = brute_arc(y, 2);
    s.require("CBS: the maximising arc near the planted [12, 30), and brute force's", a.begin == ab.begin && a.end == ab.end && a.begin >= 10 && a.begin <= 14 && a.end >= 27 && a.end <= 34,
              std::to_string(a.begin) + ", " + std::to_string(a.end));
  }
  {
    // The early-rejection boundary: b_i the least b with P(H ≤ i − 1) ≤ η 2^{−i}.
    const std::size_t B = 999, r_star = 50;  // α = 0.05: reject iff 1 + R ≤ 50
    const double eta = 0.05;
    const auto bd = ts::changepoint_detail::cbs_boundary(B, r_star, eta);
    using calx::special_functions::log_choose;
    auto cdf = [&](std::size_t x, std::size_t b) {
      double p = 0;
      for (std::size_t h = 0; h <= x && h <= b; h++)
        if (b - h <= B - r_star) p += std::exp(log_choose(r_star, h) + log_choose(B - r_star, b - h) - log_choose(B, b));
      return p;
    };
    bool ok = !bd.empty();
    for (std::size_t i = 0; i < bd.size() && ok; i++) {
      const double level = eta * std::ldexp(1.0, -static_cast<int>(i + 1));
      ok = cdf(i, bd[i]) <= level && cdf(i, bd[i] - 1) > level;
    }
    s.require("CBS boundary: each b_i is the least b whose hypergeometric level is ≤ η 2^{−i} (" + std::to_string(bd.size()) + " boundaries, b_1 = " +
                  (bd.empty() ? std::string("-") : std::to_string(bd[0])) + ")", ok);
    // The early rule's added size under the null, exactly: Σ_{r ≥ r*} P(early | r)/(B + 1), P(early | r)
    // by dynamic programming over the order of the r exceedances among the B permutations.
    double added = 0, worst_given_r = 0;
    for (std::size_t r = r_star; r <= B; r++) {
      std::vector<double> p(r_star + 1, 0.0), q(r_star + 1, 0.0);
      p[0] = 1;  // the distribution of R(b) on the paths not yet stopped (R < r*)
      double early = 0;
      std::size_t next = 0;
      for (std::size_t b = 0; b < bd.back(); b++) {
        std::fill(q.begin(), q.end(), 0.0);
        for (std::size_t x = 0; x < r_star; x++) {
          if (p[x] == 0) continue;
          const double up = static_cast<double>(r - x) / static_cast<double>(B - b);
          q[x + 1] += p[x] * up;
          q[x] += p[x] * (1 - up);
        }
        q[r_star] = 0;  // curtailed: stopped, not significant
        std::swap(p, q);
        while (next < bd.size() && bd[next] == b + 1) {
          for (std::size_t x = 0; x <= next && x < r_star; x++) {
            early += p[x];
            p[x] = 0;
          }
          next++;
        }
      }
      added += early / static_cast<double>(B + 1);
      worst_given_r = std::max(worst_given_r, early);
      if (early < 1e-14 && r > r_star + 50) break;
    }
    s.check("CBS boundary: given any full record that does not reject, early rejection has probability ≤ η", std::max(0.0, worst_given_r - eta), 0);
    s.check("CBS boundary: the exact size added under the null at α = 0.05, B = 999, η = 0.05", added, 0.005);
    std::printf("      (added size %.3g; largest P(early | R(B) = r) %.3g)\n", added, worst_given_r);
  }
  {
    // The size of the first split's permutation test under the null: iid normal series.
    calx::random::Stream z(8, 0, 0, 0);
    std::size_t rejections = 0;
    const std::size_t series = 300;
    for (std::size_t rep = 0; rep < series; rep++) {
      std::vector<double> y(80);
      for (auto& v : y) v = z.normal();
      ts::CbsOptions o;
      o.alpha = 0.05;
      o.permutations = 199;
      o.eta = 0;
      o.seed = rep + 1;
      if (!ts::circular_binary_segmentation(y, o).changepoints.empty()) rejections++;
    }
    const double rate = static_cast<double>(rejections) / static_cast<double>(series);
    s.check("CBS: the first split's size under the null, α = 0.05 (300 series of 80; ±3 standard errors)", rate - 0.05, 3 * std::sqrt(0.05 * 0.95 / series));
  }
  {
    // Planted arcs: a copy-number-like profile.
    calx::random::Stream z(9, 0, 0, 0);
    const std::vector<std::size_t> truth{200, 260, 700, 1100, 1130};
    const auto y = planted(z, 1500, truth, {0, 0.8, 0, -0.6, 0.9, 0}, {0.25, 0.25, 0.25, 0.25, 0.25, 0.25});
    const auto c = ts::circular_binary_segmentation(y);
    bool ok = c.changepoints.size() == truth.size();
    for (std::size_t k = 0; ok && k < truth.size(); k++) ok = (c.changepoints[k] > truth[k] ? c.changepoints[k] - truth[k] : truth[k] - c.changepoints[k]) <= 2;
    std::string got;
    for (const auto t : c.changepoints) got += " " + std::to_string(t);
    s.require("CBS: the five planted changes of a copy-number-like profile within 2 (α = 0.01, B = 10⁴)", ok, got);
    s.require("CBS: every p-value ≤ α, one score and p-value per change point", c.p_values.size() == c.changepoints.size() && c.scores.size() == c.changepoints.size() &&
              std::all_of(c.p_values.begin(), c.p_values.end(), [](double p) { return p <= 0.01; }));
    const auto c2 = ts::circular_binary_segmentation(y);
    s.require("CBS: the same seed gives the same segmentation", c.changepoints == c2.changepoints && c.p_values == c2.p_values);
    // Pruning and undoing: noise at a looser level makes spurious splits; pruning and undo_sd remove them.
    std::vector<double> yn(1500);
    for (std::size_t i = 0; i < 1500; i++) yn[i] = y[i] + (i % 97 == 0 ? 2.0 : 0.0);  // isolated outliers
    ts::CbsOptions loose;
    loose.alpha = 0.05;
    const auto many = ts::circular_binary_segmentation(yn, loose);
    loose.prune = 0.05;
    const auto pruned = ts::circular_binary_segmentation(yn, loose);
    // undo_sd: a real shift of 1.2σ over 300 observations is significant, and undone at 2σ.
    std::vector<double> ys = y;
    for (std::size_t i = 300; i < 600; i++) ys[i] += 0.3;
    ts::CbsOptions plain;
    const auto kept = ts::circular_binary_segmentation(ys, plain);
    plain.undo_sd = 2;
    plain.variance = 0.0625;
    const auto undone = ts::circular_binary_segmentation(ys, plain);
    s.require("CBS pruning (γ = 0.05): fewer change points than unpruned, the planted five kept", pruned.changepoints.size() < many.changepoints.size() &&
              pruned.changepoints.size() >= 5, std::to_string(many.changepoints.size()) + " → " + std::to_string(pruned.changepoints.size()));
    s.require("CBS undo_sd = 2: a significant shift of 1.2σ found, then undone, the five planted kept", kept.changepoints.size() == 7 && undone.changepoints.size() == 5,
              std::to_string(kept.changepoints.size()) + " → " + std::to_string(undone.changepoints.size()));
  }

  // ---- speed for n = 10⁵ ----
  {
    calx::random::Stream z(10, 0, 0, 0);
    std::vector<std::size_t> truth;
    for (std::size_t t = 2000; t < 100000; t += 2000 + z.below(2000)) truth.push_back(t);
    std::vector<double> means, sds;
    for (std::size_t k = 0; k <= truth.size(); k++) means.push_back(k % 2 ? z.uniform(-1, 1) : 0.0), sds.push_back(0.3);
    const auto y = planted(z, 100000, truth, means, sds);
    auto t0 = std::chrono::steady_clock::now();
    const auto p = ts::pelt(y);
    const double tp = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    const auto c = ts::circular_binary_segmentation(y);
    const double tc = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    const auto b = ts::binary_segmentation(y);
    const double tb = seconds_since(t0);
    std::printf("      n = 100000, %zu planted changes: PELT %zu found in %.3f s (%zu cost evaluations); CBS %zu in %.2f s; binary segmentation %zu in %.3f s\n",
                truth.size(), p.changepoints.size(), tp, p.evaluations, c.changepoints.size(), tc, b.changepoints.size(), tb);
    s.check("speed: PELT on n = 10⁵ (seconds)", tp, 5);
    s.check("speed: CBS on n = 10⁵ (seconds)", tc, 300);
  }
  return s.done();
}
