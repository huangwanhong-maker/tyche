// Checks of information bought under a budget (budget.hpp): the worth of two tests on separate decisions is the sum of
// their EVSIs, held by forming the product problem outright (product states, product acts, added utilities, product signals) and
// analysing it as one; three sites inspected by hand; the set bought against every subset within the budget; costs charged
// against what a test gains; equal costs; and what is refused.

#include <tyche/decision/budget.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace tyche::decision;

namespace {

Informable random_problem(calx::random::Stream& rng, const std::string& name) {
  Informable q;
  q.name = name;
  const std::size_t S = 2 + static_cast<std::size_t>(rng.uniform(0, 2)), A = 2 + static_cast<std::size_t>(rng.uniform(0, 2)), Y = 2 + static_cast<std::size_t>(rng.uniform(0, 2));
  q.table.u.assign(A, std::vector<double>(S));
  for (auto& row : q.table.u)
    for (auto& x : row) x = rng.uniform(-10, 10);
  double total = 0;
  q.prior.resize(S);
  for (auto& p : q.prior) total += p = rng.uniform(0.1, 1);
  for (auto& p : q.prior) p /= total;
  q.test.likelihood.assign(S, std::vector<double>(Y));
  for (auto& row : q.test.likelihood) {
    double t = 0;
    for (auto& x : row) t += x = rng.uniform(0.05, 1);
    for (auto& x : row) x /= t;
  }
  q.cost = rng.uniform(0.2, 3);
  return q;
}

// A site that is or is not a hazard (prior p); to act costs `act` and averts a harm `harm`; an inspection finds a hazard with
// probability `sens` and clears a sound site with probability `spec`.
Informable site(const std::string& name, double p, double harm, double act, double sens, double spec, double cost) {
  Informable q;
  q.name = name;
  q.table.acts = {"act", "leave"};
  q.table.states = {"hazard", "sound"};
  q.table.u = {{-act, -act}, {-harm, 0}};
  q.prior = {p, 1 - p};
  q.test.signals = {"found", "clear"};
  q.test.likelihood = {{sens, 1 - sens}, {1 - spec, spec}};
  q.cost = cost;
  return q;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- The worths of tests on separate decisions add ---------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261116, 31, 0, 0);
    double worst = 0, worst_one = 0;
    for (int trial = 0; trial < 200; trial++) {
      const auto a = random_problem(rng, "a"), b = random_problem(rng, "b");
      const double ea = evsi(a.table, a.prior, a.test).evsi, eb = evsi(b.table, b.prior, b.test).evsi;
      // The two decisions as one: states (s, t), acts (x, y), utility u(x, s) + v(y, t), and both signals seen, or only a's.
      const std::size_t Sa = a.prior.size(), Sb = b.prior.size(), Aa = a.table.u.size(), Ab = b.table.u.size(), Ya = a.test.likelihood[0].size(), Yb = b.test.likelihood[0].size();
      Table joint;
      joint.u.assign(Aa * Ab, std::vector<double>(Sa * Sb));
      std::vector<double> prior(Sa * Sb);
      Test both, only_a;
      both.likelihood.assign(Sa * Sb, std::vector<double>(Ya * Yb));
      only_a.likelihood.assign(Sa * Sb, std::vector<double>(Ya));
      for (std::size_t i = 0; i < Sa; i++)
        for (std::size_t j = 0; j < Sb; j++) {
          prior[i * Sb + j] = a.prior[i] * b.prior[j];
          for (std::size_t x = 0; x < Aa; x++)
            for (std::size_t y = 0; y < Ab; y++) joint.u[x * Ab + y][i * Sb + j] = a.table.u[x][i] + b.table.u[y][j];
          for (std::size_t g = 0; g < Ya; g++) {
            only_a.likelihood[i * Sb + j][g] = a.test.likelihood[i][g];
            for (std::size_t h = 0; h < Yb; h++) both.likelihood[i * Sb + j][g * Yb + h] = a.test.likelihood[i][g] * b.test.likelihood[j][h];
          }
        }
      worst = std::max(worst, std::abs(evsi(joint, prior, both).evsi - (ea + eb)));
      worst_one = std::max(worst_one, std::abs(evsi(joint, prior, only_a).evsi - ea));
    }
    s.check("two tests on separate decisions, analysed as one product problem, are worth the sum of their EVSIs (200 random pairs)", worst, 1e-11);
    s.check("and one of them alone, in the product problem, is worth what it is worth on its own decision", worst_one, 1e-11);
  }

  // ---- Three sites by hand -------------------------------------------------------------------------------------------------------------
  {
    // Acting costs 10 and averts a harm of 100. An inspection finds a hazard with probability 0.9 and clears a sound site with 0.8.
    // Site A, prior 0.5: now act (−10 against −50). P(found) = 0.45 + 0.1 = 0.55: act, −10. P(clear) = 0.45, P(hazard | clear) =
    // 0.05 / 0.45 = 1/9: leaving costs 100/9 = 11.1, so act, −10. The inspection changes nothing: EVSI 0.
    // Site B, prior 0.1: now act (−10) or leave (−10): −10. P(found) = 0.09 + 0.18 = 0.27, P(hazard | found) = 1/3: act, −10.
    // P(clear) = 0.73, P(hazard | clear) = 0.01 / 0.73: leave, −100 × 0.01 / 0.73. With: 0.27 × (−10) + (−1) = −3.7. EVSI 6.3.
    // Site C, prior 0.02: now leave, −2. P(found) = 0.018 + 0.196 = 0.214, P(hazard | found) = 0.018 / 0.214: leaving costs 8.41,
    // less than 10: leave. Nothing changes: EVSI 0.
    const std::vector<Informable> sites = {site("A", 0.5, 100, 10, 0.9, 0.8, 1), site("B", 0.1, 100, 10, 0.9, 0.8, 1), site("C", 0.02, 100, 10, 0.9, 0.8, 1)};
    const auto one = buy(sites, 1);
    s.check("an inspection that could not change the act is worth nothing: the site already sure to be acted on", std::abs(one.analysis[0].evsi), 1e-12);
    s.check("the doubtful site's inspection is worth 6.3", std::abs(one.analysis[1].evsi - 6.3), 1e-12);
    s.check("and nothing at the site too unlikely for a finding to move", std::abs(one.analysis[2].evsi), 1e-12);
    s.require("with one inspection to spend it goes to the doubtful site, not the likeliest", one.bought == std::vector<std::size_t>{1} && one.exact && std::abs(one.gained - 6.3) < 1e-12 && one.spent == 1);
    s.check("the three decisions are worth −10, −10 and −2 with nothing inspected", std::abs(one.without - (-22.0)), 1e-12);
    const auto all = buy(sites, 3);
    s.require("with budget for all three, the two that are worth nothing are still not run", all.bought == std::vector<std::size_t>{1} && all.spent == 1);
    const auto dear = buy(std::vector<Informable>{site("B", 0.1, 100, 10, 0.9, 0.8, 7)}, 10, true);
    s.require("charged against what it gains, an inspection that costs more than it is worth is not run, whatever the budget", dear.bought.empty() && dear.gained == 0 && std::abs(dear.worth[0] - (6.3 - 7)) < 1e-12);
    const auto cheap = buy(std::vector<Informable>{site("B", 0.1, 100, 10, 0.9, 0.8, 2)}, 10, true);
    s.check("and one that costs less is run for the difference, 6.3 − 2", std::abs(cheap.gained - 4.3), 1e-12);
  }

  // ---- The set bought against every subset -------------------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261116, 32, 0, 0);
    double worst = 0, equal = 0;
    bool within = true;
    for (int trial = 0; trial < 100; trial++) {
      const std::size_t n = 2 + static_cast<std::size_t>(rng.uniform(0, 9));
      std::vector<Informable> qs;
      double total = 0;
      for (std::size_t i = 0; i < n; i++) qs.push_back(random_problem(rng, std::to_string(i))), total += qs.back().cost;
      const double budget = rng.uniform(0, total);
      const bool charged = trial % 2 == 1;
      const auto got = buy(qs, budget, charged);
      double best = 0;
      for (std::size_t mask = 0; mask < (std::size_t{1} << n); mask++) {
        double value = 0, cost = 0;
        for (std::size_t i = 0; i < n; i++)
          if (mask >> i & 1) value += got.worth[i], cost += qs[i].cost;
        if (cost <= budget) best = std::max(best, value);
      }
      worst = std::max(worst, std::abs(got.gained - best));
      within = within && got.spent <= budget && got.exact && got.bound == got.gained;
      // Equal costs: the k tests of greatest worth, k what the budget runs to.
      for (auto& q : qs) q.cost = 1;
      const std::size_t k = static_cast<std::size_t>(rng.uniform(0, static_cast<double>(n) + 1));
      const auto top = buy(qs, static_cast<double>(k));
      std::vector<double> w = top.worth;
      std::sort(w.rbegin(), w.rend());
      double sum = 0;
      for (std::size_t i = 0; i < std::min(k, n); i++) sum += std::max(0.0, w[i]);
      equal = std::max(equal, std::abs(top.gained - sum));
    }
    s.check("the tests bought gain what the best subset within the budget gains, by every subset (100 problems of up to 10 tests, costs charged in half)", worst, 1e-10);
    s.require("what is spent is within the budget, and the answer is exact", within);
    s.check("with equal costs they are the tests of greatest worth, as many as the budget runs to", equal, 1e-10);
  }

  // ---- What is refused -----------------------------------------------------------------------------------------------------------------
  bool r1 = false, r2 = false;
  try { (void)buy(std::vector<Informable>{site("A", 0.5, 100, 10, 0.9, 0.8, 1)}, -1); } catch (const std::invalid_argument&) { r1 = true; }
  try { (void)buy(std::vector<Informable>{site("A", 0.5, 100, 10, 0.9, 0.8, -1)}, 1); } catch (const std::invalid_argument&) { r2 = true; }
  s.require("a negative budget and a negative cost are refused", r1 && r2);
  const auto none = buy(std::vector<Informable>{}, 5);
  s.require("no decision at all buys nothing", none.bought.empty() && none.gained == 0 && none.without == 0);
  return s.done();
}
