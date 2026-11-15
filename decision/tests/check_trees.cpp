// Checks of tyche's decision trees, value of information and lotteries (trees.hpp, value.hpp, risk.hpp): folding back against
// every strategy on a wildcatter's tree and on random trees; EVPI and EVSI's bounds, their extreme tests, and EVSI computed a
// second way by the tree of deciding after the test; the normal formulas against quadrature; stochastic dominance against the
// utilities that characterise it; certainty equivalents.

#include <tyche/decision/risk.hpp>
#include <tyche/decision/trees.hpp>
#include <tyche/decision/value.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

using namespace tyche::decision;

namespace {

double best_by_enumeration(const Tree& t) {
  double best = -INFINITY;
  for (const auto& st : strategies(t)) best = std::max(best, evaluate(t, st));
  return best;
}

// A random tree of the given depth: decision and chance nodes alternating at random, payoffs on every node.
std::size_t grow(Tree& t, calx::random::Stream& rng, int depth) {
  Node n;
  n.payoff = std::round(20 * rng.uniform() - 10);
  if (depth == 0 || rng.uniform() < 0.2) return t.add(n);
  n.kind = rng.uniform() < 0.5 ? NodeKind::decision : NodeKind::chance;
  const std::size_t k = 2 + rng.below(2);
  const std::size_t v = t.add(n);
  double z = 0;
  std::vector<double> w;
  for (std::size_t i = 0; i < k; i++) {
    const std::size_t c = grow(t, rng, depth - 1);
    t.nodes[v].children.push_back(c);
    w.push_back(rng.uniform() + 0.1);
    z += w.back();
  }
  if (t.nodes[v].kind == NodeKind::chance)
    for (double x : w) t.nodes[v].probabilities.push_back(x / z);
  return v;
}

double numerical_evpi(double mean, double sd) {
  const int n = 200000;
  const double a = mean - 12 * sd, b = mean + 12 * sd, h = (b - a) / n;
  double sum = 0;
  for (int i = 0; i <= n; i++) {
    const double x = a + i * h, w = (i == 0 || i == n) ? 0.5 : 1;
    sum += w * std::max(x, 0.0) * std::exp(-0.5 * (x - mean) * (x - mean) / (sd * sd)) / (sd * std::sqrt(2 * std::numbers::pi));
  }
  return sum * h - std::max(mean, 0.0);
}

double eu(const Lottery& l, const std::function<double(double)>& u) {
  double e = 0;
  for (std::size_t i = 0; i < l.outcomes.size(); i++) e += l.probabilities[i] * u(l.outcomes[i]);
  return e;
}

Lottery random_lottery(calx::random::Stream& rng) {
  Lottery l;
  const std::size_t k = 2 + rng.below(4);
  double z = 0;
  for (std::size_t i = 0; i < k; i++) {
    l.outcomes.push_back(static_cast<double>(rng.below(10)));
    l.probabilities.push_back(rng.uniform());
    z += l.probabilities.back();
  }
  for (auto& p : l.probabilities) p /= z;
  return l;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- A wildcatter's tree ---------------------------------------------------------------------------------------------------------
  {
    // Drill now, test first (cost 10), or sell the rights (90). The well is dry, wet or soaking: 0, 120, 270 net of drilling,
    // with probabilities 0.5, 0.3, 0.2. The test says closed or open structure.
    Tree t;
    const auto leaf = [&](double v) { Node n; n.payoff = v; return t.add(n); };
    const auto well = [&](double pd, double pw, double ps) {
      Node c;
      c.kind = NodeKind::chance;
      c.children = {leaf(0), leaf(120), leaf(270)};
      c.probabilities = {pd, pw, ps};
      return t.add(c);
    };
    const auto choose = [&](std::size_t drill) {
      Node d;
      d.kind = NodeKind::decision;
      d.children = {drill, leaf(90)};
      return t.add(d);
    };
    // P(closed) = 0.5·0.8 + 0.3·0.4 + 0.2·0.1 = 0.54; the posteriors after each signal.
    const double pc = 0.5 * 0.8 + 0.3 * 0.4 + 0.2 * 0.1;
    const std::size_t after_closed = choose(well(0.5 * 0.8 / pc, 0.3 * 0.4 / pc, 0.2 * 0.1 / pc));
    const std::size_t after_open = choose(well(0.5 * 0.2 / (1 - pc), 0.3 * 0.6 / (1 - pc), 0.2 * 0.9 / (1 - pc)));
    Node test;
    test.kind = NodeKind::chance;
    test.payoff = -10;
    test.children = {after_closed, after_open};
    test.probabilities = {pc, 1 - pc};
    const std::size_t tv = t.add(test);
    Node root;
    root.kind = NodeKind::decision;
    root.children = {well(0.5, 0.3, 0.2), tv, leaf(90)};
    t.root = t.add(root);
    const auto f = fold_back(t);
    // Drill now: 0.3·120 + 0.2·270 = 90; sell: 90; test: 0.54·max(…, 90) + 0.46·max(…, 90) − 10.
    const double closed_drill = (0.3 * 0.4 * 120 + 0.2 * 0.1 * 270) / pc, open_drill = (0.3 * 0.6 * 120 + 0.2 * 0.9 * 270) / (1 - pc);
    const double with_test = pc * std::max(closed_drill, 90.0) + (1 - pc) * std::max(open_drill, 90.0) - 10;
    s.check("the wildcatter's tree folds back to the value worked by hand", std::abs(f.value[t.root] - std::max({90.0, with_test})), 1e-12);
    s.require("its best plan is to test, sell on a closed structure and drill on an open one", f.best.at(t.root) == 1 && f.best.at(after_closed) == 1 && f.best.at(after_open) == 0);
    s.check("folding back equals the best of its 12 strategies, each evaluated alone", std::abs(f.value[t.root] - best_by_enumeration(t)), 1e-12);
  }
  {
    calx::random::Stream rng(20261115, 2, 0, 0);
    double worst = 0;
    std::size_t trees = 0;
    for (int trial = 0; trial < 300; trial++) {
      Tree t;
      t.root = grow(t, rng, 4);
      try {
        if (strategies(t, 4096).empty()) continue;
      } catch (const std::invalid_argument&) {
        continue;  // too many strategies to enumerate
      }
      const auto f = fold_back(t);
      worst = std::max({worst, std::abs(f.value[t.root] - best_by_enumeration(t)), std::abs(evaluate(t, f.best) - f.value[t.root])});
      trees++;
    }
    s.check("on " + std::to_string(trees) + " random trees, folding back equals the best strategy by enumeration, and its own strategy earns it", worst, 1e-9);
  }
  {
    Tree t;
    Node a;
    a.kind = NodeKind::chance;
    a.children = {0};
    a.probabilities = {1};
    t.nodes = {a};
    bool refused = false;
    try { validate(t); } catch (const std::invalid_argument&) { refused = true; }
    s.require("a node reached twice (a cycle) is refused: a tree, not a graph", refused);
  }

  // ---- The value of information ------------------------------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261115, 3, 0, 0);
    std::size_t negative = 0, above = 0, perfect_differs = 0, useless_differs = 0, tree_differs = 0;
    for (int trial = 0; trial < 300; trial++) {
      const std::size_t A = 2 + rng.below(3), S = 2 + rng.below(3), Y = 2 + rng.below(3);
      Table t;
      t.u.assign(A, std::vector<double>(S));
      for (auto& row : t.u)
        for (auto& x : row) x = std::round(20 * rng.uniform());
      std::vector<double> p(S);
      double z = 0;
      for (auto& x : p) z += x = rng.uniform() + 0.05;
      for (auto& x : p) x /= z;
      Test test;
      test.likelihood.assign(S, std::vector<double>(Y));
      for (auto& row : test.likelihood) {
        double zz = 0;
        for (auto& x : row) zz += x = rng.uniform();
        for (auto& x : row) x /= zz;
      }
      const double vpi = evpi(t, p), vsi = evsi(t, p, test).evsi;
      negative += vsi < -1e-12 || vpi < -1e-12;
      above += vsi > vpi + 1e-12;
      // A test that names the state is perfect; one whose signal is the same in every state is useless.
      Test perfect;
      perfect.likelihood.assign(S, std::vector<double>(S, 0.0));
      for (std::size_t st = 0; st < S; st++) perfect.likelihood[st][st] = 1;
      perfect_differs += std::abs(evsi(t, p, perfect).evsi - vpi) > 1e-12;
      Test useless;
      useless.likelihood.assign(S, test.likelihood[0]);
      useless_differs += std::abs(evsi(t, p, useless).evsi) > 1e-12;
      // EVSI a second way: fold back the tree of deciding after the test.
      const auto tree = informed_tree(t, p, test.likelihood);
      const auto e = expectations(t, p);
      tree_differs += std::abs(fold_back(tree).value[tree.root] - *std::max_element(e.begin(), e.end()) - vsi) > 1e-9;
    }
    s.require("on 300 random problems the value of information is never negative (Good 1967)", negative == 0);
    s.require("and no test is worth more than perfect information", above == 0);
    s.require("a test that reveals the state is worth exactly EVPI, and one independent of the state nothing", perfect_differs == 0 && useless_differs == 0);
    s.require("EVSI by preposterior analysis equals EVSI by folding back the tree of deciding after the test", tree_differs == 0);
  }
  {
    // By hand: invest (u = 100 if up, −50 if down) or not (0); P(up) = 0.4. A forecast right 80% of the time.
    const Table t{{"invest", "abstain"}, {"up", "down"}, {{100, -50}, {0, 0}}};
    const std::vector<double> p = {0.4, 0.6};
    s.check("EVPI = 0.4 × 100 − max(10, 0) = 30", std::abs(evpi(t, p) - 30), 1e-12);
    const auto pp = evsi(t, p, Test{{"up", "down"}, {{0.8, 0.2}, {0.2, 0.8}}});
    // P(says up) = 0.44; after it, E[invest] = (0.32·100 − 0.12·50)/0.44 = 26/0.44; after "down", E = (8 − 24)/0.56 < 0.
    s.check("EVSI of an 80% forecast = 0.44 × 26/0.44 − 10 = 16", std::abs(pp.evsi - 16), 1e-12);
    s.require("and it says: invest on \"up\", abstain on \"down\"", pp.act == std::vector<std::size_t>{0, 1});
  }
  {
    double worst = 0;
    for (const double mean : {-3.0, -1.0, -0.2, 0.0, 0.4, 1.5, 4.0})
      for (const double sd : {0.5, 1.0, 2.0}) worst = std::max(worst, std::abs(evpi_normal(mean, sd) - numerical_evpi(mean, sd)));
    s.check("for a normal net benefit, EVPI = σ φ(|μ|/σ) − |μ| Φ(−|μ|/σ) equals E max(B, 0) − max(μ, 0) by quadrature, on 21 cases", worst, 1e-7);
    s.check("at μ = 0, EVPI = σ/√(2π)", std::abs(evpi_normal(0, 2) - 2 / std::sqrt(2 * std::numbers::pi)), 1e-15);
    bool rising = true, below = true;
    double previous = 0;
    for (const double n : {1.0, 4.0, 16.0, 64.0, 256.0, 1024.0}) {
      const double v = evsi_normal(0.3, 1.0, 9.0, n);
      rising = rising && v > previous;
      below = below && v < evpi_normal(0.3, 1.0);
      previous = v;
    }
    s.require("a study's value rises with its size and stays below EVPI", rising && below);
    s.check("and tends to it", std::abs(evsi_normal(0.3, 1.0, 9.0, 1e9) - evpi_normal(0.3, 1.0)), 1e-7);
  }

  // ---- Lotteries -----------------------------------------------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261115, 4, 0, 0);
    std::size_t fsd = 0, ssd = 0, fsd_broken = 0, ssd_broken = 0, fsd_missed = 0, ssd_missed = 0;
    for (int trial = 0; trial < 3000; trial++) {
      const Lottery a = random_lottery(rng), b = random_lottery(rng);
      // First order is characterised by the step utilities 1[x > t]; second order by the ramps min(x, t).
      bool all_steps = true, all_ramps = true;
      for (int t = -1; t <= 10; t++) {
        all_steps = all_steps && eu(a, [t](double x) { return x > t ? 1.0 : 0.0; }) >= eu(b, [t](double x) { return x > t ? 1.0 : 0.0; }) - 1e-12;
        all_ramps = all_ramps && eu(a, [t](double x) { return std::min(x, double(t)); }) >= eu(b, [t](double x) { return std::min(x, double(t)); }) - 1e-12;
      }
      const bool differ = std::abs(expectation(a) - expectation(b)) > 1e-12 || !std::equal(a.outcomes.begin(), a.outcomes.end(), b.outcomes.begin(), b.outcomes.end());
      if (first_order_dominates(a, b)) {
        fsd++;
        // Every increasing utility, sampled: sums of steps and ramps with positive weights.
        for (int k = 0; k < 20; k++) {
          const double c1 = rng.uniform(), c2 = rng.uniform(), t1 = 10 * rng.uniform(), t2 = 10 * rng.uniform();
          const auto u = [&](double x) { return c1 * std::max(0.0, x - t1) + c2 * std::min(x, t2) + (x > t1 ? 0.5 : 0); };
          if (eu(a, u) < eu(b, u) - 1e-12) fsd_broken++;
        }
      } else if (all_steps && differ && !first_order_dominates(b, a)) {
        // Every step agrees but no dominance found: only possible when the distributions are equal.
        bool equal = true;
        for (int t = -1; t <= 10; t++) equal = equal && std::abs(eu(a, [t](double x) { return x > t ? 1.0 : 0.0; }) - eu(b, [t](double x) { return x > t ? 1.0 : 0.0; })) < 1e-12;
        fsd_missed += !equal;
      }
      if (second_order_dominates(a, b)) {
        ssd++;
        for (int k = 0; k < 20; k++) {
          const double c1 = rng.uniform(), c2 = rng.uniform(), t1 = 10 * rng.uniform(), t2 = 10 * rng.uniform();
          const auto u = [&](double x) { return c1 * std::min(x, t1) + c2 * std::min(x, t2) + 0.1 * std::log(1 + std::max(x, 0.0)); };
          if (eu(a, u) < eu(b, u) - 1e-12) ssd_broken++;
        }
      } else if (all_ramps && !second_order_dominates(b, a)) {
        bool equal = true;
        for (int t = -1; t <= 10; t++) equal = equal && std::abs(eu(a, [t](double x) { return std::min(x, double(t)); }) - eu(b, [t](double x) { return std::min(x, double(t)); })) < 1e-12;
        ssd_missed += !equal;
      }
    }
    s.require("first-order dominance: every sampled increasing utility prefers the dominant lottery (" + std::to_string(fsd) + " pairs)", fsd > 100 && fsd_broken == 0);
    s.require("and it is found whenever every step utility agrees, distributions apart", fsd_missed == 0);
    s.require("second-order dominance: every sampled increasing concave utility prefers the dominant lottery (" + std::to_string(ssd) + " pairs)", ssd > fsd && ssd_broken == 0);
    s.require("and it is found whenever every ramp utility agrees", ssd_missed == 0);
  }
  {
    const Lottery coin{{0, 100}, {0.5, 0.5}};
    s.require("a risk-averse certainty equivalent lies below the mean, and tends to it as aversion vanishes",
              certainty_equivalent_cara(coin, 0.05) < 50 && std::abs(certainty_equivalent_cara(coin, 1e-9) - 50) < 1e-5 && certainty_equivalent_cara(coin, 0) == 50);
    const Lottery positive{{1, 4}, {0.5, 0.5}};
    s.check("constant relative risk aversion 1 gives the geometric mean, 2", std::abs(certainty_equivalent_crra(positive, 1) - 2), 1e-12);
    s.check("and 0 the arithmetic mean, 2.5", std::abs(certainty_equivalent_crra(positive, 0) - 2.5), 1e-12);
    // A binomial of 2000 steps standardised to mean 10 and sd 2 is nearly normal: CE ≈ μ − aσ²/2 = 10 − 0.5·0.5·4 = 9.
    Lottery normal;
    const int n = 2000;
    double logc = 0;
    for (int k = 0; k <= n; k++) {
      if (k > 0) logc += std::log(double(n - k + 1)) - std::log(double(k));
      normal.outcomes.push_back(10 + 2 * (k - n / 2.0) / std::sqrt(n / 4.0));
      normal.probabilities.push_back(std::exp(logc - n * std::log(2.0)));
    }
    s.check("for a nearly normal lottery, CARA's certainty equivalent is μ − aσ²/2 (Arrow–Pratt), here 9", std::abs(certainty_equivalent_cara(normal, 0.5) - 9), 2e-3);
  }

  return s.done();
}
