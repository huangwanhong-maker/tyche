// Checks of tyche/census: the Price decomposition of a census record, in exact rationals (morphe's) unless said.
//
// A panel of three firms worked by hand in every form (one exits, one enters): the incumbent form's between −1/5, within 1/2, cross
// 1/10, entry 4/5, exit 0; the survivors' selection −1/4 and transmission 3/4, Price's W̄ ΔZ_C = Cov + E as 4/3 · 1/2 = −1/3 + 1, entry
// 7/10; the average form's −9/50, 11/20, 17/25 and 3/20; every form summing to ΔZ = 6/5. Fisher's theorem: the trait the fitness, copied
// exactly, Δē = Var(e)/ē. The replicator step and neutral selection. Melitz and Polanec's closed form: every character growing by γ, entry
// credited 0, s_N γ/2 and s_N γ, exit 0, s_X γ/2 and 0 by the survivors', average and incumbent forms. Two levels worked by hand: between
// −3/4, within −1/4. Outside proposals credited to their source. On 200 random records with several parents, parts that arrive, zero
// weights after and two levels of groups: every form an identity, selection by level summing to selection, Price's fitness form exact, the
// credits to operators, sources and reasons summing to their terms, the second moment's identity; records composed, the totals adding over
// steps and the selection terms not; a step composed with an identity changing nothing; doubles agreeing with rationals. Refusals.

#include <tyche/census/price.hpp>

#include <morphe/arithmetic/rational.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace tyche::census;
using Q = morphe::arithmetic::Rational;

namespace {

template <class F>
bool refused(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

Q q(long long n, long long d = 1) { return Q(morphe::arithmetic::Integer(n), morphe::arithmetic::Integer(d)); }

template <class T>
T sum(const std::map<std::string, T>& m) {
  T s{};
  for (const auto& [k, v] : m) s += v;
  return s;
}
template <class T>
T sum(const std::vector<T>& v) {
  T s{};
  for (const auto& x : v) s += x;
  return s;
}

// The hand panel: before A (2, 1), B (1, 4), C (1, 2); after A (3, 2), B (1, 4), D (1, 6); C exits, D enters.
Record<Q> panel() {
  Census<Q> b{{q(2), q(1), q(1)}, {q(1), q(4), q(2)}, {}}, a{{q(3), q(1), q(1)}, {q(2), q(4), q(6)}, {}};
  return by_identity<Q>({"A", "B", "C"}, b, {"A", "B", "D"}, a, "new firm", "closed");
}

// A random record: every member before has a positive weight; members after descend from one or two parents, wholly or in part, or
// arrive from one of three sources; some members after weigh nothing; two levels of groups.
Record<Q> random_record(calx::random::Stream& g) {
  const std::size_t n = 2 + g.below(5), m = 2 + g.below(5);
  Record<Q> r;
  for (std::size_t i = 0; i < n; i++) {
    r.before.weight.push_back(q(1 + static_cast<long long>(g.below(6)), 1 + static_cast<long long>(g.below(3))));
    r.before.trait.push_back(q(static_cast<long long>(g.below(11)) - 5, 1 + static_cast<long long>(g.below(3))));
    r.before.levels.push_back({g.below(2) ? "A" : "B", g.below(2) ? "x" : "y"});
    r.exit_reason.push_back(g.below(2) ? "failed" : "retired");
  }
  const char* sources[] = {"outside", "model", "person"};
  const char* ops[] = {"copy", "mutate", "cross"};
  bool any = false;
  for (std::size_t j = 0; j < m; j++) {
    r.after.weight.push_back(q(static_cast<long long>(g.below(5)), 1 + static_cast<long long>(g.below(2))));
    r.after.trait.push_back(q(static_cast<long long>(g.below(13)) - 6, 1 + static_cast<long long>(g.below(4))));
    std::vector<Link<Q>> L;
    std::vector<Arrival<Q>> A;
    const auto kind = g.below(6);
    if (kind == 0) {
      A.push_back({sources[g.below(3)], q(1)});
    } else {
      const std::size_t k = 1 + g.below(2);
      const Q p = kind == 1 ? q(1, 2) : q(1);
      for (std::size_t t = 0; t < k; t++) L.push_back({static_cast<std::size_t>(g.below(n)), p / Q(static_cast<long long>(k))});
      if (kind == 1) A.push_back({sources[g.below(3)], q(1, 2)});
      any = true;
    }
    r.parents.push_back(L), r.arrivals.push_back(A), r.made_by.push_back(ops[g.below(3)]);
  }
  if (!any || sum(r.after.weight) == Q(0)) {  // at least one descended member with weight, so that the survivors' form applies
    r.parents[0] = {{0, q(1)}}, r.arrivals[0] = {}, r.after.weight[0] = q(1);
  }
  // Make sure the descended members do not all weigh nothing.
  bool weighty = false;
  for (std::size_t j = 0; j < m; j++)
    if (!r.parents[j].empty() && Q(0) < r.after.weight[j]) weighty = true;
  if (!weighty) r.parents[0] = {{0, q(1)}}, r.arrivals[0] = {}, r.after.weight[0] = q(1);
  return r;
}

Record<double> to_double(const Record<Q>& r) {
  Record<double> d;
  for (const auto& x : r.before.weight) d.before.weight.push_back(x.to_double());
  for (const auto& x : r.before.trait) d.before.trait.push_back(x.to_double());
  for (const auto& x : r.after.weight) d.after.weight.push_back(x.to_double());
  for (const auto& x : r.after.trait) d.after.trait.push_back(x.to_double());
  d.before.levels = r.before.levels, d.made_by = r.made_by, d.exit_reason = r.exit_reason;
  for (const auto& L : r.parents) {
    d.parents.emplace_back();
    for (const auto& l : L) d.parents.back().push_back({l.parent, l.share.to_double()});
  }
  for (const auto& A : r.arrivals) {
    d.arrivals.emplace_back();
    for (const auto& a : A) d.arrivals.back().push_back({a.source, a.share.to_double()});
  }
  return d;
}

// The identity record of a census: every member its own copy.
template <class T>
Record<T> identity(const Census<T>& c) {
  Record<T> r;
  r.before = c, r.after = c, r.after.levels.clear();
  for (std::size_t i = 0; i < c.size(); i++) r.parents.push_back({{i, T(1)}}), r.arrivals.push_back({}), r.made_by.push_back("copy");
  r.exit_reason.assign(c.size(), "exit");
  return r;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- The hand panel ----------------------------------------------------------------------------------------------------------
  {
    const auto r = panel();
    const auto I = decompose(r, Form::incumbent), S = decompose(r, Form::survivors), A = decompose(r, Form::average);
    s.require("a panel by hand: Z 2, Z' 16/5, ΔZ 6/5", I.before_mean == q(2) && I.after_mean == q(16, 5) && I.total == q(6, 5));
    s.require("... the incumbent form (FHK): between −1/5, within 1/2, cross 1/10, entry 4/5, exit 0",
              I.selection == q(-1, 5) && I.within == q(1, 2) && I.cross == q(1, 10) && I.entry == q(4, 5) && I.exit == q(0) && I.reference == q(2));
    s.require("... the survivors' form: Z_C 2, Z'_C 5/2, selection −1/4, transmission 3/4, entry 7/10, exit 0",
              S.continuing_change == q(1, 2) && S.selection == q(-1, 4) && S.transmission == q(3, 4) && S.entry == q(7, 10) && S.exit == q(0));
    s.require("... Price's equation: W̄ 4/3, Cov −1/3, E(wΔz) 1, W̄ ΔZ_C = Cov + E",
              S.mean_fitness == q(4, 3) && S.covariance == q(-1, 3) && S.expectation == q(1) && S.mean_fitness * S.continuing_change == S.covariance + S.expectation);
    s.require("... the average form (Griliches–Regev): R 13/5, selection −9/50, within 11/20, entry 17/25, exit 3/20",
              A.reference == q(13, 5) && A.selection == q(-9, 50) && A.within == q(11, 20) && A.entry == q(17, 25) && A.exit == q(3, 20));
    bool all = true;
    for (const auto& D : {I, S, A, decompose(r, Form::declared, q(-7, 3))}) all = all && D.selection + D.transmission + D.entry + D.exit == D.total;
    s.require("... every form, and a declared reference of −7/3, sums to ΔZ", all);
    s.require("... credited: entry to the new firm, exit to closing", I.entry_by_source.at("new firm") == I.entry && I.exit_by_reason.at("closed") == I.exit);
  }

  // ---- Fisher's theorem, the replicator and neutrality ---------------------------------------------------------------------------
  {
    // Fitness 1, 3/2, 3 on weights 2, 1, 1, the trait the fitness and copied exactly.
    std::vector<Q> x{q(2), q(1), q(1)}, w{q(1), q(3, 2), q(3)};
    Record<Q> r;
    for (int i = 0; i < 3; i++) {
      r.before.weight.push_back(x[i]), r.before.trait.push_back(w[i]);
      r.after.weight.push_back(x[i] * w[i]), r.after.trait.push_back(w[i]);
      r.parents.push_back({{static_cast<std::size_t>(i), q(1)}}), r.arrivals.push_back({}), r.made_by.push_back("copy");
    }
    const auto D = decompose(r);
    Q X = sum(x), mean{}, var{};
    for (int i = 0; i < 3; i++) mean += x[i] / X * w[i];
    for (int i = 0; i < 3; i++) var += x[i] / X * (w[i] - mean) * (w[i] - mean);
    s.require("Fisher's theorem: the trait the fitness, copied exactly, Δē = Var(e)/ē, all of it selection", D.total == var / mean && D.transmission == Q(0) && D.selection == D.total);
    bool rep = true;
    Q Xp = sum(r.after.weight);
    for (int i = 0; i < 3; i++) rep = rep && r.after.weight[i] / Xp == x[i] / X * w[i] / mean;
    s.require("... the shares after are the replicator's step s_i f_i / f̄, and W̄ is f̄", rep && D.mean_fitness == mean);
    for (int i = 0; i < 3; i++) r.after.weight[i] = x[i] * q(5, 4), r.after.trait[i] = w[i] + q(1, 7);
    const auto N = decompose(r);
    s.require("neutral selection: every fitness 5/4, the covariance and the selection nought, the change all transmission (1/7)",
              N.covariance == Q(0) && N.selection == Q(0) && N.transmission == q(1, 7) && N.total == q(1, 7));
  }

  // ---- Melitz and Polanec's representative firm ----------------------------------------------------------------------------------
  {
    // Before: survivors S1 (3), S2 (2) and an exiter X (1), every character φ = 2; after: S1 (4), S2 (2) and an entrant N (2), every
    // character φ + γ with γ = 1/3. s_N = 2/8, s_X = 1/6.
    const Q phi = q(2), gam = q(1, 3);
    Census<Q> b{{q(3), q(2), q(1)}, {phi, phi, phi}, {}}, a{{q(4), q(2), q(2)}, {phi + gam, phi + gam, phi + gam}, {}};
    const auto r = by_identity<Q>({"S1", "S2", "X"}, b, {"S1", "S2", "N"}, a);
    const auto S = decompose(r, Form::survivors), A = decompose(r, Form::average), I = decompose(r, Form::incumbent);
    const Q sN = q(1, 4), sX = q(1, 6);
    s.require("Melitz and Polanec's closed form: every character grows by γ; entry 0, s_N γ/2, s_N γ by the survivors', average and incumbent forms",
              S.entry == Q(0) && A.entry == sN * gam / q(2) && I.entry == sN * gam);
    s.require("... exit 0, s_X γ/2, 0", S.exit == Q(0) && A.exit == sX * gam / q(2) && I.exit == Q(0));
    s.require("... the shares entering and leaving read back", S.entering_share == sN && S.exiting_share == sX);
  }

  // ---- Two levels by hand ----------------------------------------------------------------------------------------------------------
  {
    Record<Q> r;
    r.before = {{q(1), q(1), q(2)}, {q(0), q(2), q(4)}, {{"g1"}, {"g1"}, {"g2"}}};
    r.after = {{q(2), q(1), q(1)}, {q(0), q(2), q(4)}, {}};
    for (std::size_t i = 0; i < 3; i++) r.parents.push_back({{i, q(1)}}), r.arrivals.push_back({}), r.made_by.push_back("copy");
    const auto D = decompose(r);
    s.require("two levels by hand: ΔZ −1, all selection: between groups −3/4, within them −1/4",
              D.total == q(-1) && D.selection_by_level.size() == 2 && D.selection_by_level[0] == q(-3, 4) && D.selection_by_level[1] == q(-1, 4));
  }

  // ---- Outside proposals in a search -----------------------------------------------------------------------------------------------
  {
    Record<Q> r;
    r.before = {{q(1), q(1)}, {q(1), q(2)}, {}};
    r.after = {{q(1), q(1), q(1)}, {q(3), q(2), q(5)}, {}};
    r.parents = {{{0, q(1)}}, {{1, q(1)}}, {}};
    r.arrivals = {{}, {}, {{"model", q(1)}}};
    r.made_by = {"mutate", "copy", ""};
    const auto D = decompose(r);
    s.require("a search's proposal from a model: its entry 1/3 · (5 − 5/2) = 5/6; transmission by mutation 1, by copying 0",
              D.entry_by_source.at("model") == q(5, 6) && D.transmission_by_operator.at("mutate") == q(1) && D.transmission_by_operator.at("copy") == Q(0));
  }

  // ---- Random records ----------------------------------------------------------------------------------------------------------
  {
    calx::random::Stream g(20261123, 1, 2, 3);
    bool identity_ok = true, levels_ok = true, price_ok = true, credits_ok = true, moment_ok = true, doubles_ok = true;
    bool totals_add = true, selection_differs = false, identity_composed = true;
    int composed = 0, with_identity = 0;
    for (int t = 0; t < 200; t++) {
      const auto r = random_record(g);
      for (Form f : {Form::survivors, Form::incumbent, Form::average, Form::declared}) {
        const auto D = decompose(r, f, q(3, 7));
        identity_ok = identity_ok && D.selection + D.transmission + D.entry + D.exit == D.total && D.total == D.after_mean - D.before_mean;
        credits_ok = credits_ok && sum(D.entry_by_source) == D.entry && sum(D.exit_by_reason) == D.exit;
        if (f == Form::survivors) {
          levels_ok = levels_ok && sum(D.selection_by_level) == D.selection && D.selection_by_level.size() == 3;
          price_ok = price_ok && D.mean_fitness * D.continuing_change == D.covariance + D.expectation && D.selection + D.transmission == D.continuing_change;
          credits_ok = credits_ok && sum(D.transmission_by_operator) == D.transmission;
          const auto d = decompose(to_double(r), f);
          doubles_ok = doubles_ok && std::fabs(d.selection - D.selection.to_double()) < 1e-12 && std::fabs(d.entry - D.entry.to_double()) < 1e-12 &&
                       std::fabs(d.transmission - D.transmission.to_double()) < 1e-12;
        } else if (f == Form::average) {
          credits_ok = credits_ok && sum(D.transmission_by_operator) == D.within && D.cross == Q(0);
        } else {
          credits_ok = credits_ok && sum(D.transmission_by_operator) == D.within + D.cross;
        }
      }
      const auto M = second_moment(r);
      moment_ok = moment_ok && M.change == M.of_square.total - M.mean_shift;
      // A second step from this record's census after, composed into one span.
      auto r2 = random_record(g);
      r2.before = r.after, r2.before.levels.clear();
      for (auto& L : r2.parents)
        for (auto& l : L) l.parent = l.parent % r.after.size();
      r2.exit_reason.assign(r.after.size(), "failed later");
      bool r2_ok = true;
      try {
        validate(r2);
        bool weighty = false;
        for (std::size_t j = 0; j < r2.after.size(); j++)
          for (const auto& l : r2.parents[j])
            if (Q(0) < r.after.weight[l.parent] && Q(0) < r2.after.weight[j]) weighty = true;
        r2_ok = weighty;
      } catch (const std::invalid_argument&) {
        r2_ok = false;
      }
      if (!r2_ok) continue;
      try {
        const auto c = compose(r, r2);
        const auto D1 = decompose(r, Form::incumbent), D2 = decompose(r2, Form::incumbent), Dc = decompose(c, Form::incumbent);
        totals_add = totals_add && Dc.total == D1.total + D2.total;
        composed++;
        if (Dc.selection != D1.selection + D2.selection) selection_differs = true;
        const auto ci = compose(r, identity(r.after));
        const auto Di = decompose(ci, Form::incumbent);
        identity_composed = identity_composed && Di.selection == D1.selection && Di.entry == D1.entry && Di.exit == D1.exit && Di.transmission == D1.transmission;
        with_identity++;
      } catch (const std::invalid_argument& e) {
        // An ancestor of zero weight in the middle census cannot pass on a lineage; such pairs are not composed.
        if (std::string(e.what()).find("positive weight") == std::string::npos) totals_add = false;
      }
    }
    s.require("200 random records: every form an identity, summing to Z' − Z", identity_ok);
    s.require("... selection by level (two levels and within) summing to selection", levels_ok);
    s.require("... Price's equation in its fitness form exact, and selection + transmission the survivors' change", price_ok);
    s.require("... credits to operators, sources and reasons summing to their terms", credits_ok);
    s.require("... the second moment: the change of variance is the change of the mean square less the shift of the mean squared", moment_ok);
    s.require("... doubles agreeing with exact rationals to 1e-12", doubles_ok);
    s.require("... two steps composed: the totals add over steps", totals_add && composed >= 50, std::to_string(composed) + " spans");
    s.require("... and the selection terms of the steps do not add to the span's (Holm 2014, §4)", selection_differs);
    s.require("... a step composed with an identity changes no term", identity_composed && with_identity >= 20, std::to_string(with_identity) + " spans");
  }

  // ---- Refusals ------------------------------------------------------------------------------------------------------------------
  {
    auto r = panel();
    auto bad = r;
    bad.parents[0][0].share = q(1, 2);
    s.require("a member whose links and arrivals do not sum to one is refused", refused([&] { decompose(bad); }));
    bad = r, bad.before.weight[0] = q(-1);
    s.require("a negative weight is refused", refused([&] { decompose(bad); }));
    bad = r, bad.parents[0][0].parent = 7;
    s.require("a parent outside the census is refused", refused([&] { decompose(bad); }));
    bad = r, bad.before.weight[0] = q(0);
    s.require("an ancestor of zero weight with a descendant is refused", refused([&] { decompose(bad); }));
    bad = r;
    for (std::size_t j = 0; j < bad.after.size(); j++) bad.parents[j] = {}, bad.arrivals[j] = {{"entry", q(1)}};
    s.require("the survivors' form with no member continuing is refused; the incumbent form still answers",
              refused([&] { decompose(bad, Form::survivors); }) && decompose(bad, Form::incumbent).entry == decompose(bad, Form::incumbent).total);
    s.require("two ids alike in a census are refused", refused([&] { by_identity<Q>({"A", "A"}, Census<Q>{{q(1), q(1)}, {q(1), q(1)}, {}}, {"A"}, Census<Q>{{q(1)}, {q(1)}, {}}); }));
  }
  return s.done();
}
