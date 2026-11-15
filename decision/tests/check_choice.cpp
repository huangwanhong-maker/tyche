// Checks of tyche's decisions in one step (choice.hpp): five criteria under ignorance that choose five different acts on one
// table, worked by hand; regret and dominance; Ellsberg's urn, where maxmin expected utility makes the choices no single
// probability makes; credal sets from bounds; and Troffaes's inclusions among the rules for credal sets on random tables, with
// E-admissibility's linear programme held to the acts that are best under sampled probabilities.

#include <tyche/decision/choice.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace tyche::decision;

namespace {

bool subset(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
  return std::all_of(a.begin(), a.end(), [&](std::size_t x) { return std::find(b.begin(), b.end(), x) != b.end(); });
}

std::string show(const std::vector<std::size_t>& v) {
  std::string s = "{";
  for (std::size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + std::to_string(v[i]);
  return s + "}";
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- Ignorance: five criteria, five acts ---------------------------------------------------------------------------------------
  {
    // Found by search so that no act is dominated and each criterion has one choice, its own.
    const Table t{{"a0", "a1", "a2", "a3", "a4"}, {"s1", "s2", "s3", "s4"}, {{5, 2, 6, 6}, {1, 0, 8, 9}, {4, 2, 3, 7}, {5, 3, 5, 3}, {6, 6, 1, 8}}};
    s.require("no act of the table is dominated", admissible(t).size() == 5);
    s.require("maximin takes a3, whose worst outcome, 3, is best", maximin(t).chosen == std::vector<std::size_t>{3} && maximin(t).value[3] == 3);
    s.require("maximax takes a1, whose best outcome, 9, is best", maximax(t).chosen == std::vector<std::size_t>{1});
    s.require("Hurwicz at α = 0.6 takes a2: 0.6 × 2 + 0.4 × 7 = 4.0 against 3.8 at most for the others", hurwicz(t, 0.6).chosen == std::vector<std::size_t>{2} && std::abs(hurwicz(t, 0.6).value[2] - 4.0) < 1e-15);
    s.require("Laplace takes a4, of mean 21/4", laplace(t).chosen == std::vector<std::size_t>{4} && std::abs(laplace(t).value[4] - 5.25) < 1e-15);
    const auto r = regret(t);
    s.require("the regret table: a0's regrets are 1, 4, 2, 3, against the best of each state, 6, 6, 8, 9", r[0] == std::vector<double>{1, 4, 2, 3});
    s.require("minimax regret takes a0, whose greatest regret, 4, is least", minimax_regret(t).chosen == std::vector<std::size_t>{0} && !minimax_regret(t).higher_is_better);
    s.require("Hurwicz at α = 1 is maximin and at α = 0 maximax", hurwicz(t, 1).chosen == maximin(t).chosen && hurwicz(t, 0).chosen == maximax(t).chosen);
  }
  {
    const Table t{{}, {}, {{3, 3}, {2, 3}, {4, 1}}};
    s.require("weak dominance: (3, 3) dominates (2, 3) weakly and not strictly; (4, 1) is not dominated", dominates(t, 0, 1) && !dominates(t, 0, 1, true) && !dominates(t, 0, 2) && admissible(t) == std::vector<std::size_t>{0, 2});
    s.require("a tie is a set: two acts of one best value are both chosen", maximin(Table{{}, {}, {{1, 2}, {2, 1}}}).chosen.size() == 2);
  }

  // ---- Risk ----------------------------------------------------------------------------------------------------------------------
  {
    const Table t{{}, {}, {{10, 0}, {4, 4}}};
    const std::vector<double> p = {0.5, 0.5}, q = {0.3, 0.7};
    s.require("expected utility at even odds takes the gamble (5 against 4), and at 0.3 the sure thing", expected_utility(t, p).chosen == std::vector<std::size_t>{0} && expected_utility(t, q).chosen == std::vector<std::size_t>{1});
  }

  // ---- Ellsberg's urn ----------------------------------------------------------------------------------------------------------
  {
    // 30 red balls and 60 black or yellow in unknown proportion. States R, B, Y; P(R) = 1/3, P(B) anywhere in [0, 2/3].
    const Table t{{"red", "black", "red or yellow", "black or yellow"}, {"R", "B", "Y"}, {{1, 0, 0}, {0, 1, 0}, {1, 0, 1}, {0, 1, 1}}};
    const Credal c{{{1.0 / 3, 0, 2.0 / 3}, {1.0 / 3, 2.0 / 3, 0}}};
    const Table first{{}, {}, {t.u[0], t.u[1]}}, second{{}, {}, {t.u[2], t.u[3]}};
    s.require("maxmin expected utility prefers betting on red to betting on black", gamma_maximin(first, c).chosen == std::vector<std::size_t>{0});
    s.require("and betting on black or yellow to betting on red or yellow", gamma_maximin(second, c).chosen == std::vector<std::size_t>{1});
    bool any = false;
    for (int i = 0; i <= 600; i++) {
      const double pb = (2.0 / 3) * i / 600.0;
      const std::vector<double> p = {1.0 / 3, pb, 2.0 / 3 - pb};
      const auto e = expectations(t, p);
      any = any || (e[0] > e[1] && e[3] > e[2]);
    }
    s.require("no single probability in the set makes both choices (601 probabilities tried): Ellsberg's pattern is not expected utility's", !any);
    s.require("between red and black the set alone does not choose: each is maximal and E-admissible, and Γ-maximin settles it", maximal(first, c) == std::vector<std::size_t>{0, 1} && e_admissible(first, c) == std::vector<std::size_t>{0, 1});
    s.require("among all four bets, red is not maximal: black or yellow betters it under every probability of the set", maximal(t, c) == std::vector<std::size_t>{1, 2, 3});
  }

  // ---- Credal sets from bounds -------------------------------------------------------------------------------------------------
  {
    const std::vector<double> lo = {0.1, 0.2, 0.1}, hi = {0.5, 0.6, 0.4};
    const auto c = credal_from_bounds(lo, hi);
    bool ok = c.points.size() == 6;
    for (const auto& p : c.points) ok = ok && std::abs(p[0] + p[1] + p[2] - 1) < 1e-12;
    s.require("the bounds [0.1, 0.5], [0.2, 0.6], [0.1, 0.4] give a credal set of six vertices, each a probability", ok, std::to_string(c.points.size()));
    bool refused = false;
    try { (void)credal_from_bounds(std::vector<double>{0.6, 0.6}, std::vector<double>{0.7, 0.7}); } catch (const std::invalid_argument&) { refused = true; }
    s.require("bounds that admit no probability are refused", refused);
    const std::vector<double> f = {1, 0, 0};
    s.require("the lower and upper probability of the first state are its bounds", std::abs(lower_expectation(c, f) - 0.1) < 1e-12 && std::abs(upper_expectation(c, f) - 0.5) < 1e-12);
  }

  // ---- Troffaes's inclusions on random tables ----------------------------------------------------------------------------------
  {
    calx::random::Stream rng(20261115, 1, 0, 0);
    std::size_t broken = 0, empty = 0, unsound = 0, single_differ = 0, tables = 0;
    std::string first;
    for (int trial = 0; trial < 400; trial++) {
      const std::size_t A = 2 + rng.below(4), S = 2 + rng.below(3), K = 1 + rng.below(4);
      Table t;
      t.u.assign(A, std::vector<double>(S));
      for (auto& row : t.u)
        for (auto& x : row) x = std::round(10 * rng.uniform()) / 2;  // halves, so that ties occur
      Credal c;
      for (std::size_t k = 0; k < K; k++) {
        std::vector<double> p(S);
        double z = 0;
        for (auto& x : p) z += x = rng.uniform();
        for (auto& x : p) x /= z;
        c.points.push_back(p);
      }
      tables++;
      const auto gmin = gamma_maximin(t, c).chosen, gmax = gamma_maximax(t, c).chosen;
      const auto e = e_admissible(t, c), m = maximal(t, c), iu = interval_undominated(t, c);
      const bool ok = subset(gmin, m) && subset(gmax, e) && subset(e, m) && subset(m, iu);
      if (!ok && first.empty()) first = "Γ-maximin " + show(gmin) + ", Γ-maximax " + show(gmax) + ", E " + show(e) + ", maximal " + show(m) + ", interval " + show(iu);
      broken += !ok;
      empty += e.empty();
      // Every act best under some sampled probability of the set is E-admissible.
      for (int j = 0; j < 60; j++) {
        std::vector<double> lam(K), p(S, 0.0);
        double z = 0;
        for (auto& x : lam) z += x = rng.uniform();
        for (std::size_t k = 0; k < K; k++)
          for (std::size_t st = 0; st < S; st++) p[st] += lam[k] / z * c.points[k][st];
        double zp = 0;
        for (double x : p) zp += x;
        for (auto& x : p) x /= zp;
        if (!subset(expected_utility(t, p).chosen, e)) unsound++;
      }
      if (K == 1) {
        // With one probability, E-admissibility and maximality are expected utility's choice.
        const auto eu = expected_utility(t, c.points[0]).chosen;
        if (e != eu || m != eu) single_differ++;
      }
    }
    s.require("on 400 random tables and credal sets: Γ-maximin ⊆ maximal, Γ-maximax ⊆ E-admissible ⊆ maximal ⊆ interval-undominated", broken == 0, first);
    s.require("the E-admissible acts are never none", empty == 0);
    s.require("every act best under a sampled probability of the set is found E-admissible by the linear programme (24,000 probabilities)", unsound == 0);
    s.require("with a single probability, E-admissibility and maximality choose what expected utility chooses", single_differ == 0);
  }

  // ---- Refusals ----------------------------------------------------------------------------------------------------------------
  {
    bool a = false, b = false, c = false;
    try { (void)expected_utility(Table{{}, {}, {{1, 2}}}, std::vector<double>{0.5, 0.6}); } catch (const std::invalid_argument&) { a = true; }
    try { (void)maximin(Table{{}, {}, {{1, 2}, {3}}}); } catch (const std::invalid_argument&) { b = true; }
    try { (void)hurwicz(Table{{}, {}, {{1, 2}}}, 1.5); } catch (const std::invalid_argument&) { c = true; }
    s.require("a law that does not sum to 1, a ragged table and an α outside [0, 1] are refused", a && b && c);
  }

  return s.done();
}
