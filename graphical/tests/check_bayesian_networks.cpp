// Checks of tyche::graphical's factors, networks and variable elimination: the factor operations against entries
// worked by hand; the refusals of a network (a cycle, a row not summing to one, an unknown parent, a table of the
// wrong size, a negative entry, a repeated name); the Asia network of Lauritzen and Spiegelhalter (1988), its
// marginals with and without evidence, joints, likelihood ratios, virtual evidence and most probable explanation
// against enumeration of its joint law (tests/brute_force.hpp, written without the library's factors); impossible
// evidence refused; the probability of evidence below the smallest double kept in logarithms; and seeded random
// networks against enumeration.

#include <tyche/graphical/graphical.hpp>

#include "brute_force.hpp"

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

using namespace tyche::graphical;
using calx::verification::relative;

namespace {

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

bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

double max_diff(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); i++)
    for (std::size_t j = 0; j < a[i].size(); j++) m = std::max(m, std::fabs(a[i][j] - b[i][j]));
  return m;
}

// The Asia network (Lauritzen and Spiegelhalter 1988, J. R. Statist. Soc. B 50, 157–224, §2 and Fig. 1), its
// tables as the paper gives them; listed out of topological order on purpose.
std::vector<Node> asia_nodes() {
  const std::vector<std::string> yn{"yes", "no"};
  return {
      {"Dyspnoea", yn, {"Bronchitis", "Either"}, {0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9}},
      {"XRay", yn, {"Either"}, {0.98, 0.02, 0.05, 0.95}},
      {"Either", yn, {"LungCancer", "Tuberculosis"}, {1, 0, 1, 0, 1, 0, 0, 1}},
      {"Asia", yn, {}, {0.01, 0.99}},
      {"Tuberculosis", yn, {"Asia"}, {0.05, 0.95, 0.01, 0.99}},
      {"Smoking", yn, {}, {0.5, 0.5}},
      {"LungCancer", yn, {"Smoking"}, {0.1, 0.9, 0.01, 0.99}},
      {"Bronchitis", yn, {"Smoking"}, {0.6, 0.4, 0.3, 0.7}},
  };
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- factors ----
  {
    const Factor phi = factor({0, 1}, {2, 2}, {1, 2, 3, 4});        // φ(a, b), b fastest
    const Factor psi = factor({1, 2}, {2, 3}, {1, 2, 3, 4, 5, 6});  // ψ(b, c), c fastest
    const Factor pr = product(phi, psi);
    double err = (pr.scope == std::vector<std::size_t>{0, 1, 2}) ? 0 : 1;
    for (std::size_t a = 0; a < 2; a++)
      for (std::size_t b = 0; b < 2; b++)
        for (std::size_t c = 0; c < 3; c++) err += std::fabs(pr.values[a * 6 + b * 3 + c] - phi.values[a * 2 + b] * psi.values[b * 3 + c]);
    s.check("factor: the product φ(a, b)ψ(b, c) over (a, b, c), entry by entry", err, 0);
    const Factor sb = sum_out(pr, 1);  // Σ_b φψ, over (a, c)
    err = 0;
    for (std::size_t a = 0; a < 2; a++)
      for (std::size_t c = 0; c < 3; c++) err += std::fabs(sb.values[a * 3 + c] - (pr.values[a * 6 + c] + pr.values[a * 6 + 3 + c]));
    s.check("factor: summing out the middle variable", err, 0);
    const Factor mb = max_out(pr, 2);
    err = 0;
    for (std::size_t a = 0; a < 2; a++)
      for (std::size_t b = 0; b < 2; b++) err += std::fabs(mb.values[a * 2 + b] - std::max({pr.values[a * 6 + b * 3], pr.values[a * 6 + b * 3 + 1], pr.values[a * 6 + b * 3 + 2]}));
    s.check("factor: maximising out the last variable", err, 0);
    const Factor rd = reduce(pr, 1, 1);
    err = (rd.scope == std::vector<std::size_t>{0, 2}) ? 0 : 1;
    for (std::size_t a = 0; a < 2; a++)
      for (std::size_t c = 0; c < 3; c++) err += std::fabs(rd.values[a * 3 + c] - pr.values[a * 6 + 3 + c]);
    s.check("factor: reduction to b = 1", err, 0);
    const Factor ro = reorder(pr, {2, 0, 1});
    err = 0;
    for (std::size_t a = 0; a < 2; a++)
      for (std::size_t b = 0; b < 2; b++)
        for (std::size_t c = 0; c < 3; c++) err += std::fabs(ro.values[c * 4 + a * 2 + b] - pr.values[a * 6 + b * 3 + c]);
    err += calx::verification::max_difference(reorder(ro, {0, 1, 2}).values, pr.values);
    s.check("factor: reordering the scope, and back", err, 0);
    const Factor mg = marginalise(pr, {2, 0});
    err = 0;
    for (std::size_t c = 0; c < 3; c++)
      for (std::size_t a = 0; a < 2; a++) err += std::fabs(mg.values[c * 2 + a] - sb.values[a * 3 + c]);
    s.check("factor: marginalising to (c, a)", err, 0);
    s.check("factor: a normalised factor sums to one", std::fabs(total(normalise(pr)) - 1), 1e-15);
    s.require("factor: normalising a zero factor is refused", contains(refusal([] { normalise(factor({0}, {2}, {0, 0})); }), "not a positive"));
    s.require("factor: a negative value is refused", !refusal([] { factor({0}, {2}, {0.5, -0.1}); }).empty());
  }

  // ---- networks and their refusals ----
  const auto nodes = asia_nodes();
  const Network asia = network(nodes);
  {
    std::vector<std::size_t> place(asia.size());
    for (std::size_t i = 0; i < asia.order.size(); i++) place[asia.order[i]] = i;
    bool ok = asia.order.size() == asia.size();
    for (std::size_t v = 0; v < asia.size(); v++)
      for (std::size_t p : asia.parents[v]) ok = ok && place[p] < place[v];
    s.require("network: the order found puts every variable after its parents", ok);
    const std::vector<std::string> yn{"yes", "no"};
    const auto cyc = refusal([&] { network({{"A", yn, {"C"}, {0.5, 0.5, 0.5, 0.5}}, {"B", yn, {"A"}, {0.5, 0.5, 0.5, 0.5}}, {"C", yn, {"B"}, {0.5, 0.5, 0.5, 0.5}}}); });
    s.require("network: a cycle among the parents is refused and named", contains(cyc, "cycle") && contains(cyc, "A") && contains(cyc, "→"), cyc);
    auto bad = nodes;
    bad[1].table = {0.98, 0.02, 0.05, 0.9};
    const auto row = refusal([&] { network(bad); });
    s.require("network: a row not summing to one is refused, the row named", contains(row, "XRay") && contains(row, "Either = no") && contains(row, "summing"), row);
    bad = nodes;
    bad[0].parents = {"Bronchitis", "Eithr"};
    s.require("network: an unknown parent is refused", contains(refusal([&] { network(bad); }), "Eithr"));
    bad = nodes;
    bad[1].table = {0.98, 0.02};
    s.require("network: a table of the wrong size is refused", contains(refusal([&] { network(bad); }), "entries"));
    bad = nodes;
    bad[5].table = {1.5, -0.5};
    s.require("network: a negative entry is refused", contains(refusal([&] { network(bad); }), "negative"));
    bad = nodes;
    bad[7].name = "Smoking";
    s.require("network: two variables of one name are refused", contains(refusal([&] { network(bad); }), "two variables"));
    s.require("network: an unknown state is refused", contains(refusal([&] { evidence(asia, {{"XRay", "maybe"}}); }), "maybe"));
  }

  // ---- Asia against enumeration ----
  const brute::Joint J(nodes);
  const std::size_t D = asia.index("Dyspnoea"), X = asia.index("XRay"), E = asia.index("Either"), A = asia.index("Asia"), T = asia.index("Tuberculosis"),
                    L = asia.index("LungCancer");
  {
    s.check("Asia: P(Tuberculosis = yes) = 0.01 × 0.05 + 0.99 × 0.01", std::fabs(marginal(asia, T)[0] - (0.01 * 0.05 + 0.99 * 0.01)), 1e-16);
    s.check("Asia: every marginal without evidence against enumeration", max_diff(marginals(asia), J.marginals({})), 1e-15);
    const Evidence e = evidence(asia, {{"XRay", "yes"}, {"Dyspnoea", "yes"}});
    s.check("Asia: every marginal given a positive X-ray and dyspnoea, against enumeration", max_diff(marginals(asia, e), J.marginals(e)), 1e-15);
    s.check("Asia: P(positive X-ray, dyspnoea) against enumeration", relative(evidence_probability(asia, e), J.evidence(e)), 1e-14);
    const Evidence e2 = evidence(asia, {{"Asia", "yes"}, {"Dyspnoea", "yes"}, {"Smoking", "no"}});
    s.check("Asia: every marginal given a visit to Asia, dyspnoea and no smoking, against enumeration", max_diff(marginals(asia, e2), J.marginals(e2)), 1e-15);
    EliminationOptions md{Heuristic::min_degree, {}}, given{Heuristic::min_fill, {0, 1, 2, 3, 4, 5, 6, 7}};
    s.check("Asia: min-degree and a given order agree with min-fill", std::max(max_diff(marginals(asia, e, md), marginals(asia, e)), max_diff(marginals(asia, e, given), marginals(asia, e))), 1e-15);
    const Factor lt = joint(asia, {L, T}, e);
    s.check("Asia: the joint of lung cancer and tuberculosis given the evidence, against enumeration", calx::verification::max_difference(lt.values, J.pair(L, T, e)), 1e-15);
    s.check("Asia: P(Q, e) sums to P(e)", relative(total(joint_with_evidence(asia, {L, T}, e)), J.evidence(e)), 1e-14);
    const Factor xq = joint(asia, {X, L}, e);
    s.check("Asia: a queried variable that is observed has its observed state with probability one", std::fabs(xq.values[0] + xq.values[1] - 1) + xq.values[2] + xq.values[3], 1e-15);

    // P(e | T = yes) and P(e | T = no) by enumeration.
    double eh = 0, en = 0, ph = 0;
    J.each([&](const std::vector<std::size_t>& x) {
      const double p = J.p(x);
      (x[T] == 0 ? eh : en) += p * brute::Joint::weight(e, x);
      if (x[T] == 0) ph += p;
    });
    const auto lr = likelihood_ratio(asia, T, 0, e);
    s.check("Asia: P(e | tuberculosis) against enumeration", relative(lr.given_h, eh / ph), 1e-13);
    s.check("Asia: P(e | no tuberculosis) against enumeration", relative(lr.given_not_h, en / (1 - ph)), 1e-13);
    s.check("Asia: the likelihood ratio against enumeration", relative(lr.ratio, (eh / ph) / (en / (1 - ph))), 1e-13);
    s.check("Asia: posterior odds = likelihood ratio × prior odds", relative(lr.posterior / (1 - lr.posterior), lr.ratio * lr.prior / (1 - lr.prior)), 1e-13);
    s.check("Asia: log ratio = log of the ratio", std::fabs(lr.log_ratio - std::log(lr.ratio)), 1e-13);

    Evidence soft = e2;
    soft.likelihood[X] = {0.8, 0.3};
    s.check("Asia: virtual evidence on the X-ray, every marginal against enumeration", max_diff(marginals(asia, soft), J.marginals(soft)), 1e-15);
    s.check("Asia: virtual evidence, P(e) against enumeration", relative(evidence_probability(asia, soft), J.evidence(soft)), 1e-14);

    const auto mpe = most_probable_explanation(asia, e);
    s.check("Asia: the most probable explanation's probability against enumeration", relative(mpe.probability, J.most_probable(e)), 1e-14);
    s.check("Asia: the explanation found has that probability", relative(J.p(mpe.states) * brute::Joint::weight(e, mpe.states), J.most_probable(e)), 1e-14);
    const auto mpe0 = most_probable_explanation(asia);
    s.check("Asia: the most probable explanation without evidence", relative(J.p(mpe0.states), J.most_probable({})), 1e-14);

    const Evidence impossible = evidence(asia, {{"Either", "no"}, {"Tuberculosis", "yes"}});
    s.check("Asia: evidence contradicting a deterministic table has probability exactly zero", evidence_probability(asia, impossible), 0);
    s.require("Asia: a posterior on impossible evidence is refused", contains(refusal([&] { marginal(asia, D, impossible); }), "probability zero"));
    s.require("Asia: an explanation of impossible evidence is refused", contains(refusal([&] { most_probable_explanation(asia, impossible); }), "probability zero"));
    s.require("Asia: a likelihood ratio on impossible evidence is refused", contains(refusal([&] { likelihood_ratio(asia, L, 0, impossible); }), "zero under both"));
    const auto inf = likelihood_ratio(asia, E, 0, evidence(asia, {{"Tuberculosis", "yes"}}));
    s.require("Asia: evidence excluding only the alternative gives an infinite ratio", std::isinf(inf.ratio) && inf.ratio > 0 && inf.posterior == 1);
    (void)A;
  }

  // ---- the probability of evidence below the smallest double ----
  {
    const std::size_t n = 1100;
    std::vector<Node> chain;
    for (std::size_t i = 0; i < n; i++) {
      Node node{"C" + std::to_string(i), {"a", "b"}, {}, {0.5, 0.5}};
      if (i > 0) {
        node.parents = {"C" + std::to_string(i - 1)};
        node.table = {0.5, 0.5, 0.5, 0.5};
      }
      chain.push_back(node);
    }
    const Network net = network(chain);
    Evidence e;
    for (std::size_t i = 0; i < n; i++)
      if (i != n / 2) e.observed[i] = i % 2;
    const double expected = -static_cast<double>(n - 1) * std::numbers::ln2;
    s.check("chain of 1100: log P(e) = −1099 log 2 although P(e) underflows", relative(log_evidence_probability(net, e), expected), 1e-13);
    s.check("chain of 1100: the one unobserved variable is even", std::fabs(marginal(net, n / 2, e)[0] - 0.5), 1e-15);
  }

  // ---- random networks against enumeration ----
  {
    calx::random::Stream rng(23, 0x67726170, 1, 0);
    double dm = 0, dp = 0, dx = 0, dl = 0;
    for (int t = 0; t < 30; t++) {
      const auto ns = brute::random_network(rng, 7, 3, 3);
      const Network net = network(ns);
      const brute::Joint Jr(ns);
      Evidence e;
      const std::size_t a = rng.below(7), b = (a + 1 + rng.below(6)) % 7, c = rng.below(7);
      e.observed[a] = rng.below(net.cardinality(a));
      e.observed[b] = rng.below(net.cardinality(b));
      std::vector<double> lam(net.cardinality(c));
      for (double& q : lam) q = rng.uniform();
      e.likelihood[c] = lam;
      if (Jr.evidence(e) <= 0) continue;
      dm = std::max(dm, max_diff(marginals(net, e), Jr.marginals(e)));
      dp = std::max(dp, relative(evidence_probability(net, e), Jr.evidence(e)));
      dx = std::max(dx, relative(most_probable_explanation(net, e).probability, Jr.most_probable(e)));
      // H: the first unobserved variable lies in its first states but one.
      std::size_t h = 0;
      while (h == a || h == b) h++;
      std::vector<std::size_t> S;
      for (std::size_t q = 0; q + 1 < net.cardinality(h); q++) S.push_back(q);
      double fh = 0, fn = 0, ph = 0;
      Jr.each([&](const std::vector<std::size_t>& x) {
        const double p = Jr.p(x), w = p * brute::Joint::weight(e, x);
        const bool in = x[h] + 1 < net.cardinality(h);
        (in ? fh : fn) += w;
        if (in) ph += p;
      });
      dl = std::max(dl, relative(likelihood_ratio(net, h, S, e).ratio, (fh / ph) / (fn / (1 - ph))));
    }
    s.check("random networks: every marginal given hard and virtual evidence, against enumeration", dm, 1e-14);
    s.check("random networks: P(e) against enumeration", dp, 1e-13);
    s.check("random networks: the most probable explanation against enumeration", dx, 1e-13);
    s.check("random networks: the likelihood ratio of a set of states against enumeration", dl, 1e-12);
  }
  {
    // Asia's moral graph has the chordless cycle Smoking–LungCancer–Either–Bronchitis, so no order is narrower
    // than 2; the other variables are simplicial or joined to it by one edge, and one fill edge across the cycle
    // gives 2.
    std::vector<std::size_t> all;
    for (std::size_t v = 0; v < asia.size(); v++) all.push_back(v);
    const Plan p = plan(asia.tables, all);
    s.require("Asia: min-fill finds an order of induced width 2", p.width == 2 && p.order.size() == asia.size(), std::to_string(p.width));
    // Either first: its neighbours LungCancer, Tuberculosis, XRay, Bronchitis and Dyspnoea are joined, width 5.
    std::vector<std::size_t> either_first{asia.index("Either")};
    for (std::size_t v = 0; v < asia.size(); v++)
      if (v != either_first[0]) either_first.push_back(v);
    const Plan q = plan(asia.tables, all, Heuristic::min_fill, either_first);
    s.require("Asia: a given order is followed and its width computed", q.order == either_first && q.width == 5, std::to_string(q.width));
  }
  return s.done();
}
