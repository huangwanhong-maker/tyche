// Checks of finite POMDPs (decision/pomdp.hpp, alpha.hpp, point_based.hpp, simulate.hpp).
//
// The reference throughout is the exact lookahead on the tree of beliefs, which shares no code with the
// α-vectors: V_h(b) = max_a [ρ(b, a) + γ Σ_o P(o | b, a) V_{h−1}(τ(b, a, o))], V_0 = 0.
// - Bayes's rule on the tiger problem in closed form; refusals of laws that are not laws.
// - Exact value iteration by incremental pruning equals the lookahead at 300 beliefs, for every horizon
//   to 5 on the tiger problem (discounted and not) and to 4 on two random POMDPs; every pruned vector has
//   a witness, and pruning changes no value.
// - The tiger at discount 0.75 (pomdp.org's tiger.aaai): HSVI's interval of width 10⁻⁶ contains the published
//   V*(½) = 1.933439, and PBVI reaches it.
// - V* bracketed independently at every belief tried by the lookahead of depth 4 on the blind bound (below) and on
//   the fast informed bound (above): QMDP ≥ FIB ≥ those brackets ≥ the blind bound.
// - PBVI's, Perseus's and HSVI's lower bounds lie below the upper bracket, HSVI's upper bound above the lower;
//   HSVI closes at b₀ (to 10⁻³ on the tiger, 10⁻² on the random POMDP); PBVI and Perseus come within 1 % of its
//   lower bound there.
// - HSVI's policy, run for 10 000 episodes, earns its interval within four standard errors.

#include <tyche/decision/alpha.hpp>
#include <tyche/decision/point_based.hpp>
#include <tyche/decision/pomdp.hpp>
#include <tyche/decision/simulate.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace td = tyche::decision;

namespace {

// The tiger problem (Kaelbling, Littman and Cassandra 1998, §5): the tiger behind the left door (state 0)
// or the right (1); listen (0) for −1, hearing it on the correct side with probability 0.85; open the
// left door (1) or the right (2) for −100 with the tiger behind it and +10 without, after which the
// problem starts again (the tiger placed afresh, nothing heard).
td::Pomdp tiger(double gamma) {
  td::Pomdp P(2, 3, 2, gamma);
  for (std::size_t s = 0; s < 2; s++) {
    P.transition(0, s, s) = 1;
    P.observe(0, s, s) = 0.85, P.observe(0, s, 1 - s) = 0.15;
    P.reward(s, 0) = -1;
    for (std::size_t a = 1; a <= 2; a++) {
      P.transition(a, s, 0) = P.transition(a, s, 1) = 0.5;
      P.observe(a, s, 0) = P.observe(a, s, 1) = 0.5;
    }
  }
  P.reward(0, 1) = -100, P.reward(1, 1) = 10;  // open left: the tiger is there in state 0
  P.reward(0, 2) = 10, P.reward(1, 2) = -100;
  return P;
}

std::vector<double> random_law(calx::random::Stream& rng, std::size_t n) {
  std::vector<double> p(n);
  double t = 0;
  for (auto& x : p) t += (x = rng.exponential());
  for (auto& x : p) x /= t;
  return p;
}

td::Pomdp random_pomdp(std::uint32_t id, std::size_t S, std::size_t A, std::size_t O, double gamma) {
  calx::random::Stream rng(20261104, 0x72706f6du, id, 0);  // "rpom"
  td::Pomdp P(S, A, O, gamma);
  for (std::size_t a = 0; a < A; a++)
    for (std::size_t s = 0; s < S; s++) {
      const auto t = random_law(rng, S), z = random_law(rng, O);
      for (std::size_t k = 0; k < S; k++) P.transition(a, s, k) = t[k];
      for (std::size_t o = 0; o < O; o++) P.observe(a, s, o) = z[o];
      P.reward(s, a) = rng.uniform(-1, 1);
    }
  P.initial = random_law(rng, S);
  return P;
}

std::vector<td::Belief> beliefs(std::size_t S, int n, std::uint32_t id) {
  calx::random::Stream rng(20261104, 0x62656c66u, id, 0);  // "belf"
  std::vector<td::Belief> B;
  for (std::size_t s = 0; s < S; s++) {
    td::Belief e(S, 0.0);
    e[s] = 1;
    B.push_back(e);
  }
  while (static_cast<int>(B.size()) < n) B.push_back(random_law(rng, S));
  return B;
}

double zero(std::span<const double>) { return 0; }

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- the model and Bayes's rule -----------------------------------------------------------------------------------
  const auto T95 = tiger(0.95);
  td::validate(T95);
  {
    const td::Belief half{0.5, 0.5};
    const auto u1 = td::update(T95, half, 0, 0);
    const auto u2 = td::update(T95, u1.belief, 0, 0);
    const double two = 0.85 * 0.85 / (0.85 * 0.85 + 0.15 * 0.15);
    s.check("listening once from ½: the tiger on the side heard with 0.85, heard with probability ½",
            std::max(std::fabs(u1.belief[0] - 0.85), std::fabs(u1.probability - 0.5)), 1e-15);
    s.check("listening twice, hearing the left both times: 0.85² / (0.85² + 0.15²)", std::fabs(u2.belief[0] - two), 1e-15);
    const auto u3 = td::update(T95, u2.belief, 1, 1);
    s.check("opening a door starts again: the belief ½ whatever is heard", std::fabs(u3.belief[0] - 0.5), 1e-15);
    int refused = 0;
    auto bad = T95;
    bad.transition(0, 0, 0) = 0.9;
    try { td::validate(bad); } catch (const std::invalid_argument&) { refused++; }
    bad = T95, bad.discount = 1.5;
    try { td::validate(bad); } catch (const std::invalid_argument&) { refused++; }
    bad = T95, bad.initial = {0.7, 0.7};
    try { td::validate(bad); } catch (const std::invalid_argument&) { refused++; }
    td::Pomdp sure(2, 1, 2, 0.9);
    sure.transition(0, 0, 0) = sure.transition(0, 1, 1) = 1, sure.observe(0, 0, 0) = sure.observe(0, 1, 1) = 1;
    try { td::update(sure, td::Belief{1, 0}, 0, 1); } catch (const std::invalid_argument&) { refused++; }
    s.require("refusals: a transition row that is not a law, γ > 1, an initial belief that is not a law, an impossible observation", refused == 4, std::to_string(refused));
  }

  // ---- exact value iteration against the lookahead --------------------------------------------------------------------
  struct Case { std::string name; td::Pomdp P; int horizon; };
  const std::vector<Case> cases = {{"the tiger, γ = 0.95", T95, 5},
                                   {"the tiger undiscounted", tiger(1.0), 5},
                                   {"a random POMDP of 3 states, 2 actions, 2 observations", random_pomdp(1, 3, 2, 2, 0.9), 4},
                                   {"a random POMDP of 4 states, 3 actions, 3 observations", random_pomdp(2, 4, 3, 3, 0.9), 4}};
  for (std::size_t c = 0; c < cases.size(); c++) {
    const auto& [name, P, H] = cases[c];
    const auto V = td::exact_value_iteration(P, H);
    const auto B = beliefs(P.states, 300, static_cast<std::uint32_t>(c));
    double worst = 0;
    for (int h = 1; h <= H; h++)
      for (const auto& b : B) worst = std::max(worst, std::fabs(V[h - 1].value(b) - td::lookahead(P, b, h, zero)));
    std::string sizes;
    for (const auto& v : V) sizes += std::to_string(v.size()) + " ";
    s.check("exact value iteration equals the lookahead at every horizon: " + name + " (vectors " + sizes + ")", worst, 1e-9);
  }
  {
    // Pruning: every vector kept has a witness; the unpruned cross sum gives the same values.
    const auto P = random_pomdp(2, 4, 3, 3, 0.9);
    const auto V2 = td::exact_value_iteration(P, 2).back();
    bool witnessed = true;
    for (std::size_t i = 0; i < V2.size(); i++) {
      std::vector<const std::vector<double>*> others;
      for (std::size_t j = 0; j < V2.size(); j++)
        if (j != i) others.push_back(&V2.vectors[j].values);
      witnessed = witnessed && td::detail::witness_margin(V2.vectors[i].values, others) > 0;
    }
    s.require("every vector kept by pruning is the strict maximum at some belief", witnessed);
    const auto V1 = td::exact_value_iteration(P, 1).back();
    std::vector<td::AlphaVector> raw;  // Γ_2 by the full cross sum, unpruned
    for (std::size_t a = 0; a < P.actions; a++) {
      std::vector<std::vector<double>> W{std::vector<double>(P.states, 0.0)};
      for (std::size_t o = 0; o < P.observations; o++) {
        std::vector<std::vector<double>> next;
        for (const auto& w : W)
          for (const auto& v : V1.vectors) {
            auto g = td::detail::project(P, a, o, v.values, 1.0 / P.observations);
            for (std::size_t k = 0; k < P.states; k++) g[k] += w[k];
            next.push_back(g);
          }
        W.swap(next);
      }
      for (auto& w : W) raw.push_back({w, a});
    }
    const td::ValueFunction full{raw};
    double worst = 0;
    for (const auto& b : beliefs(P.states, 300, 9)) worst = std::max(worst, std::fabs(full.value(b) - V2.value(b)));
    s.check("pruning changes no value (" + std::to_string(raw.size()) + " vectors to " + std::to_string(V2.size()) + ")", worst, 1e-12);
  }

  // ---- bounds, point-based solvers and HSVI against V* ----------------------------------------------------------------
  // ---- the tiger at discount 0.75: the published value --------------------------------------------------------------
  {
    // pomdp.org's tiger.aaai.POMDP (Cassandra, Kaelbling and Littman 1994), discount 0.75: V*(½) = 1.933439, as the R
    // package pomdp documents its solution (a secondary source).
    const auto T75 = tiger(0.75);
    const auto hs = td::hsvi(T75, {.epsilon = 1e-6});
    s.require("the tiger at discount 0.75: HSVI's interval for V*(½), of width 10⁻⁶, contains the published 1.933439",
              hs.converged && hs.lower_value <= 1.933439 + 5e-7 && hs.upper_value >= 1.933439 - 5e-7,
              "[" + std::to_string(hs.lower_value) + ", " + std::to_string(hs.upper_value) + "]");
    s.check("... and PBVI's lower bound reaches it", std::fabs(td::pbvi(T75).lower.value(T75.initial) - 1.933439), 1e-6);
  }

  // ---- bounds, point-based solvers and HSVI against V* ----------------------------------------------------------------
  // V* at a belief is bracketed independently by the exact lookahead of depth 4 with the blind bound at its leaves
  // (below) and the fast informed bound (above).
  const std::vector<Case> solved = {{"the tiger, γ = 0.95", T95, 4}, {"a random POMDP of 4 states, 3 actions, 3 observations, γ = 0.9", random_pomdp(2, 4, 3, 3, 0.9), 4}};
  for (std::size_t c = 0; c < solved.size(); c++) {
    const auto& [name, P, depth] = solved[c];
    const auto Q = td::qmdp(P), F = td::fast_informed_bound(P), L = td::blind_lower_bound(P);
    const auto below_v = [&](std::span<const double> b) { return td::lookahead(P, b, depth, [&](std::span<const double> x) { return L.value(x); }); };
    const auto above_v = [&](std::span<const double> b) { return td::lookahead(P, b, depth, [&](std::span<const double> x) { return F.value(x); }); };
    const auto B = beliefs(P.states, 60, 20 + static_cast<std::uint32_t>(c));
    bool order = true;
    for (const auto& b : B) order = order && Q.value(b) >= F.value(b) - 1e-9 && F.value(b) >= above_v(b) - 1e-9 && above_v(b) >= below_v(b) - 1e-9 && below_v(b) >= L.value(b) - 1e-9;
    s.require("QMDP ≥ FIB ≥ the lookahead on FIB ≥ the lookahead on the blind bound ≥ the blind bound, at 60 beliefs: " + name, order);

    const double eps = c == 0 ? 1e-3 : 1e-2;  // the random POMDP's values are small: 10⁻² is 4 % of V*(b₀)
    const auto hs = td::hsvi(P, {.epsilon = eps});
    const auto pb = td::pbvi(P), pe = td::perseus(P, {.beliefs = 200});
    bool below = true, above = true;
    for (const auto& b : B) {
      const double up = above_v(b), lo = below_v(b);
      below = below && pb.lower.value(b) <= up + 1e-9 && pe.lower.value(b) <= up + 1e-9 && hs.lower.value(b) <= up + 1e-9;
      above = above && hs.upper.value(b) >= lo - 1e-9;
    }
    s.require("PBVI's, Perseus's and HSVI's lower bounds lie below V*'s upper bracket everywhere tried: " + name, below);
    s.require("HSVI's upper bound lies above V*'s lower bracket everywhere tried: " + name, above);
    s.require("HSVI closes at b₀ to " + std::to_string(eps) + ": " + name, hs.converged && hs.upper_value - hs.lower_value <= eps,
              "[" + std::to_string(hs.lower_value) + ", " + std::to_string(hs.upper_value) + "] after " + std::to_string(hs.trials) + " trials");
    const double vstar = hs.lower_value, scale = 1 + std::fabs(vstar);
    s.check("PBVI within 1 % of HSVI's lower bound at b₀: " + name + " (" + std::to_string(pb.beliefs.size()) + " beliefs)", (vstar - pb.lower.value(P.initial)) / scale, 0.01);
    s.check("Perseus within 1 % of HSVI's lower bound at b₀: " + name + " (" + std::to_string(pe.iterations) + " stages)", (vstar - pe.lower.value(P.initial)) / scale, 0.01);

    // HSVI's policy, simulated: within its interval, four standard errors and the truncation.
    const auto E = td::evaluate(P, [&](std::span<const double> b) { return hs.lower.action(b); }, 10000, 300);
    s.require("HSVI's policy earns V*(b₀) within four standard errors: " + name,
              E.mean >= hs.lower_value - 4 * E.error - E.truncation - 1e-3 && E.mean <= hs.upper_value + 4 * E.error + E.truncation,
              std::to_string(E.mean) + " ± " + std::to_string(E.error) + " against [" + std::to_string(hs.lower_value) + ", " + std::to_string(hs.upper_value) + "]");
  }
  return s.done();
}
