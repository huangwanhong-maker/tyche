// Checks of tyche::graphical's belief propagation: on seeded random polytrees, with and without hard and virtual
// evidence, the marginals equal variable elimination's and are reported exact; on a pairwise chain given as a
// factor graph, the beliefs equal the normalised sum-product; on the diamond A → B, A → C, B → D, C → D, loopy
// propagation is exact with no evidence (the messages toward parents being uniform) and, given D, converges close
// to the exact marginals and is reported approximate, damping reaching the same fixed point; likewise on the Asia
// network given a positive X-ray and dyspnoea; evidence on A (on Smoking in Asia) cuts the cycle and makes the
// result exact; a run stopped early reports its last change; impossible evidence and a damping out of range are
// refused.

#include <tyche/graphical/graphical.hpp>

#include "brute_force.hpp"

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace tyche::graphical;

namespace {

double max_diff(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); i++)
    for (std::size_t j = 0; j < a[i].size(); j++) m = std::max(m, std::fabs(a[i][j] - b[i][j]));
  return m;
}

template <class F>
std::string refusal(F f) {
  try {
    f();
  } catch (const Refusal& e) {
    return e.what();
  } catch (...) {
    return "(another exception)";
  }
  return "";
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- polytrees: exact ----
  {
    calx::random::Stream rng(23, 0x67726170, 2, 0);
    double d0 = 0, d1 = 0;
    bool all_exact = true;
    std::size_t rounds = 0;
    for (int t = 0; t < 40; t++) {
      const auto ns = brute::random_polytree(rng, 12, 3);
      const Network net = network(ns);
      const auto bp = propagate(net);
      d0 = std::max(d0, max_diff(bp.marginals, marginals(net)));
      all_exact = all_exact && bp.tree && bp.exact && bp.converged;
      rounds = std::max(rounds, bp.iterations);
      Evidence e;
      const std::size_t a = rng.below(12), b = (a + 1 + rng.below(11)) % 12, c = rng.below(12);
      e.observed[a] = rng.below(net.cardinality(a));
      e.observed[b] = rng.below(net.cardinality(b));
      std::vector<double> lam(net.cardinality(c));
      for (double& q : lam) q = 0.1 + rng.uniform();
      e.likelihood[c] = lam;
      if (evidence_probability(net, e) <= 0) continue;
      const auto be = propagate(net, e);
      d1 = std::max(d1, max_diff(be.marginals, marginals(net, e)));
      all_exact = all_exact && be.exact;
    }
    s.check("polytrees: belief propagation equals elimination without evidence", d0, 1e-12);
    s.check("polytrees: belief propagation equals elimination with hard and virtual evidence", d1, 1e-12);
    s.require("polytrees: every result is reported a converged tree, exact", all_exact);
    s.require("polytrees: the messages settle within the diameter of the factor graph and one round more", rounds <= 2 * 12 + 1, std::to_string(rounds));
  }

  // ---- a factor graph given directly: a pairwise chain ----
  {
    const std::vector<std::size_t> cards{2, 3, 2, 2, 3};
    std::vector<Factor> fs;
    calx::random::Stream rng(23, 0x67726170, 3, 0);
    for (std::size_t v = 0; v < cards.size(); v++) {
      std::vector<double> u(cards[v]);
      for (double& x : u) x = 0.2 + rng.uniform();
      fs.push_back(factor({v}, {cards[v]}, u));
      if (v + 1 < cards.size()) {
        std::vector<double> w(cards[v] * cards[v + 1]);
        for (double& x : w) x = 0.2 + rng.uniform();
        fs.push_back(factor({v, v + 1}, {cards[v], cards[v + 1]}, w));
      }
    }
    const auto bp = propagate(cards, fs);
    double d = 0;
    for (std::size_t v = 0; v < cards.size(); v++) {
      const auto m = normalise(sum_product(fs, {v})).values;
      for (std::size_t q = 0; q < m.size(); q++) d = std::max(d, std::fabs(m[q] - bp.marginals[v][q]));
    }
    s.check("chain factor graph: the beliefs equal the normalised sums of the product", d, 1e-14);
    s.require("chain factor graph: reported exact", bp.exact);
  }

  // ---- loopy: the diamond ----
  {
    const std::vector<std::string> yn{"yes", "no"};
    const std::vector<Node> diamond{
        {"A", yn, {}, {0.3, 0.7}},
        {"B", yn, {"A"}, {0.8, 0.2, 0.3, 0.7}},
        {"C", yn, {"A"}, {0.6, 0.4, 0.1, 0.9}},
        {"D", yn, {"B", "C"}, {0.9, 0.1, 0.6, 0.4, 0.5, 0.5, 0.2, 0.8}},
    };
    const Network net = network(diamond);
    const auto exact = marginals(net);
    const auto bp = propagate(net);
    s.require("diamond: loopy propagation converges", bp.converged, "last change " + std::to_string(bp.last_change));
    s.require("diamond: reported approximate, the factor graph having a cycle", !bp.tree && !bp.exact);
    // With nothing observed every message toward a parent is uniform (Σ_x P(x | pa) = 1), so no information goes
    // round the cycle and even loopy propagation gives the exact marginals; the flag still says approximate,
    // since it is decided by the graph.
    s.check("diamond: with no evidence the messages toward parents are uniform, and loopy marginals are exact", max_diff(bp.marginals, exact), 1e-15);
    const Evidence ea = evidence(net, {{"A", "no"}});
    const auto bc = propagate(net, ea);
    s.require("diamond: evidence on A cuts the cycle, and the result is exact", bc.tree && bc.exact);
    s.check("diamond: given A, propagation equals elimination", max_diff(bc.marginals, marginals(net, ea)), 1e-14);
    const Evidence ed = evidence(net, {{"D", "yes"}});
    const auto bdd = propagate(net, ed);
    const double off = max_diff(bdd.marginals, marginals(net, ed));
    s.require("diamond: given D, loopy propagation converges and is reported approximate", bdd.converged && !bdd.tree && !bdd.exact);
    s.check("diamond: given D, the loopy marginals are close to the exact ones", off, 0.05);
    s.require("diamond: given D, the loopy marginals are not the exact ones", off > 1e-6, std::to_string(off));
    PropagationOptions damped;
    damped.damping = 0.5;
    const auto bd = propagate(net, ed, damped);
    s.check("diamond: given D, damping reaches the same fixed point", max_diff(bd.marginals, bdd.marginals), 1e-10);
    s.require("diamond: damping takes more rounds", bd.iterations > bdd.iterations, std::to_string(bd.iterations) + " against " + std::to_string(bdd.iterations));
    PropagationOptions few;
    few.max_iterations = 2;
    few.tolerance = 0;
    const auto bf = propagate(net, ed, few);
    s.require("diamond: stopped after two rounds, not converged, its last change reported", !bf.converged && bf.iterations == 2 && bf.last_change > 0);
  }

  // ---- loopy: Asia ----
  {
    const std::vector<std::string> yn{"yes", "no"};
    const std::vector<Node> nodes{
        {"Asia", yn, {}, {0.01, 0.99}},
        {"Tuberculosis", yn, {"Asia"}, {0.05, 0.95, 0.01, 0.99}},
        {"Smoking", yn, {}, {0.5, 0.5}},
        {"LungCancer", yn, {"Smoking"}, {0.1, 0.9, 0.01, 0.99}},
        {"Bronchitis", yn, {"Smoking"}, {0.6, 0.4, 0.3, 0.7}},
        {"Either", yn, {"LungCancer", "Tuberculosis"}, {1, 0, 1, 0, 1, 0, 0, 1}},
        {"XRay", yn, {"Either"}, {0.98, 0.02, 0.05, 0.95}},
        {"Dyspnoea", yn, {"Bronchitis", "Either"}, {0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9}},
    };
    const Network asia = network(nodes);
    const Evidence e = evidence(asia, {{"XRay", "yes"}, {"Dyspnoea", "yes"}});
    const auto bp = propagate(asia, e);
    s.require("Asia: loopy propagation converges and is reported approximate", bp.converged && !bp.exact);
    s.check("Asia: loopy marginals given a positive X-ray and dyspnoea close to the exact ones", max_diff(bp.marginals, marginals(asia, e)), 0.05);
    const Evidence es = evidence(asia, {{"Smoking", "yes"}, {"XRay", "yes"}});
    const auto bs = propagate(asia, es);
    s.require("Asia: evidence on Smoking cuts the cycle, and the result is exact", bs.exact);
    s.check("Asia: given Smoking, propagation equals elimination", max_diff(bs.marginals, marginals(asia, es)), 1e-14);
    const Evidence impossible = evidence(asia, {{"Either", "no"}, {"Tuberculosis", "yes"}, {"Smoking", "yes"}});
    s.require("Asia: impossible evidence is refused", refusal([&] { propagate(asia, impossible); }).find("impossible") != std::string::npos);
    PropagationOptions bad;
    bad.damping = 1;
    s.require("propagation: a damping of one is refused", !refusal([&] { propagate(asia, e, bad); }).empty());
  }
  return s.done();
}
