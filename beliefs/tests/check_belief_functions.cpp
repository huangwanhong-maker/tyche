// Checks of tyche::beliefs: the refusals of malformed mass functions; on seeded random mass functions over a frame
// of five, Bel ≤ Pl, Bel(A) + Pl(Ω ∖ A) = 1, the point and tabulated functions alike, the commonality by its
// definition, and the Möbius transforms inverting exactly; Dempster's rule against its definition summed over all
// pairs of subsets, commutative and associative, the conjunctive rule's commonality the product of commonalities
// and the disjunctive rule's implicability the product of implicabilities; on Bayesian masses Dempster's rule is
// Bayes's normalised product; Zadeh's example, its conflict and result, under Dempster's and Yager's rules; total
// conflict refused; discounting at α = 1 and α = 0; conditioning against the formula of conditional belief;
// consonance and the contour; the pignistic law summing to one and lying in the credal set; the Choquet lower and
// upper expectations against a linear programme over the credal set (calx's simplex method) and against the laws
// induced by all orderings of the frame.

#include <tyche/beliefs/beliefs.hpp>

#include <calx/optimisation/linear_programming.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

using namespace tyche::beliefs;
using calx::verification::max_difference;
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

// A random normal mass function: up to `most` distinct non-empty focal sets with Dirichlet(1) masses.
Mass random_mass(calx::random::Stream& rng, std::size_t frame, std::size_t most) {
  const std::size_t k = 1 + rng.below(most);
  std::vector<Focal> f;
  double s = 0;
  for (std::size_t i = 0; i < k; i++) {
    const Set A = static_cast<Set>(1 + rng.below((std::size_t{1} << frame) - 1));
    const double x = rng.gamma(1.0);
    f.push_back({A, x});
    s += x;
  }
  for (auto& x : f) x.mass /= s;
  return mass(frame, f);
}

// The laws induced by every ordering of the frame: p(σ_k) = Bel({σ_1, …, σ_k}) − Bel({σ_1, …, σ_{k−1}}).
std::vector<std::vector<double>> vertices(const Mass& m) {
  std::vector<std::size_t> sigma(m.frame);
  std::iota(sigma.begin(), sigma.end(), 0);
  std::vector<std::vector<double>> out;
  do {
    std::vector<double> p(m.frame);
    Set before = 0;
    double bb = 0;
    for (std::size_t k = 0; k < m.frame; k++) {
      const Set now = before | singleton(sigma[k]);
      const double bn = belief(m, now);
      p[sigma[k]] = bn - bb;
      before = now;
      bb = bn;
    }
    out.push_back(p);
  } while (std::next_permutation(sigma.begin(), sigma.end()));
  return out;
}

// min (or max) of Σ f_i p_i over {p ≥ 0, Σ p = 1, Σ_{i ∈ A} p_i ≥ Bel(A) for every non-empty proper A}.
double credal_lp(const Mass& m, const std::vector<double>& f, bool maximise) {
  const std::size_t n = m.frame, N = std::size_t{1} << n;
  calx::optimisation::LinearProgram lp;
  for (double x : f) lp.c.push_back(maximise ? -x : x);
  std::vector<std::vector<double>> rows;
  for (std::size_t A = 1; A + 1 < N; A++) {
    std::vector<double> r(n, 0.0);
    for (std::size_t i = 0; i < n; i++)
      if (A >> i & 1) r[i] = -1;
    rows.push_back(r);
    lp.b_ub.push_back(-belief(m, static_cast<Set>(A)));
  }
  lp.a_ub = calx::linalg::Rectangular::from_rows(rows);
  lp.a_eq = calx::linalg::Rectangular::from_rows({std::vector<double>(n, 1.0)});
  lp.b_eq = {1.0};
  const auto r = calx::optimisation::simplex(lp);
  if (r.status != calx::optimisation::LpStatus::optimal) return NAN;
  return maximise ? -r.value : r.value;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- refusals ----
  s.require("mass: masses not summing to one are refused", contains(refusal([] { mass(3, {{1, 0.5}, {2, 0.4}}); }), "sum to"));
  s.require("mass: a negative mass is refused", contains(refusal([] { mass(3, {{1, 1.2}, {2, -0.2}}); }), "negative"));
  s.require("mass: a set outside the frame is refused", contains(refusal([] { mass(3, {{8, 1.0}}); }), "outside"));
  s.require("mass: m(∅) > 0 is refused unless allowed", contains(refusal([] { mass(3, {{0, 0.1}, {7, 0.9}}); }), "m(∅)") &&
                                                            refusal([] { mass(3, {{0, 0.1}, {7, 0.9}}, {1e-9, true}); }).empty());
  s.require("mass: a frame of 31 elements is refused", !refusal([] { vacuous(31); }).empty());
  s.require("mass: the subsets of a frame of 21 elements are not tabulated", contains(refusal([] { belief_function(vacuous(21)); }), "not tabulated"));
  {
    // Bel({0}) = Bel({1}) = 0.5 and Bel({0, 1}) = 0.6 would need m({0, 1}) = −0.4.
    const std::vector<double> bel{0, 0.5, 0.5, 0.6};
    s.require("transforms: a table that is not a belief function is refused, the set named", contains(refusal([&] { from_belief(2, bel, {1e-9, true}); }), "m({0, 1})"));
  }

  // ---- random mass functions ----
  calx::random::Stream rng(23, 0x62656c66, 1, 0);
  const std::size_t n = 5, N = std::size_t{1} << n;
  double bel_pl = 0, sum_one = 0, point = 0, comm = 0, mob_b = 0, mob_q = 0, mob_pl = 0;
  double comm_d = 0, assoc_d = 0, def_d = 0, many_d = 0, k_d = 0, q_prod = 0, b_prod = 0, assoc_u = 0, comm_u = 0;
  double pig_sum = 0, cond_d = 0;
  bool pig_in = true;
  double low_lp = 0, up_lp = 0, low_v = 0, up_v = 0, bounds_v = 0;
  for (int t = 0; t < 60; t++) {
    const Mass m1 = random_mass(rng, n, 8), m2 = random_mass(rng, n, 8), m3 = random_mass(rng, n, 8);
    const auto b = belief_function(m1), p = plausibility_function(m1), q = commonality_function(m1);
    const Set W = full(n);
    for (std::size_t A = 0; A < N; A++) {
      bel_pl = std::max(bel_pl, b[A] - p[A]);
      sum_one = std::max(sum_one, std::fabs(b[A] + p[W ^ A] - 1));
      point = std::max({point, std::fabs(b[A] - belief(m1, static_cast<Set>(A))), std::fabs(p[A] - plausibility(m1, static_cast<Set>(A))),
                        std::fabs(q[A] - commonality(m1, static_cast<Set>(A)))});
      double qa = 0;
      for (std::size_t B = 0; B < N; B++)
        if ((A & ~B) == 0) qa += m1(static_cast<Set>(B));
      comm = std::max(comm, std::fabs(q[A] - qa));
    }
    const auto v1 = to_vector(m1);
    mob_b = std::max(mob_b, max_difference(to_vector(from_belief(n, b)), v1));
    mob_q = std::max(mob_q, max_difference(to_vector(from_commonality(n, q)), v1));
    mob_pl = std::max(mob_pl, max_difference(to_vector(from_plausibility(n, p)), v1));

    // Dempster's rule only where the two are not in total conflict.
    if (std::all_of(m1.focal.begin(), m1.focal.end(), [&](const Focal& f) { return std::all_of(m2.focal.begin(), m2.focal.end(), [&](const Focal& g) { return (f.set & g.set) == 0; }); }))
      continue;
    const auto d12 = dempster(m1, m2), d21 = dempster(m2, m1);
    comm_d = std::max(comm_d, max_difference(to_vector(d12.mass), to_vector(d21.mass)));
    // By definition: Σ over every pair of subsets.
    const auto v2 = to_vector(m2);
    std::vector<double> def(N, 0.0);
    double K = 0;
    for (std::size_t A = 0; A < N; A++)
      for (std::size_t B = 0; B < N; B++) {
        if ((A & B) == 0) K += v1[A] * v2[B];
        else def[A & B] += v1[A] * v2[B];
      }
    for (double& x : def) x /= 1 - K;
    def_d = std::max(def_d, max_difference(to_vector(d12.mass), def));
    k_d = std::max({k_d, std::fabs(d12.conflict - K), std::fabs(conjunctive(m1, m2).conflict() - K)});
    try {
      const auto left = dempster(d12.mass, m3).mass, right = dempster(m1, dempster(m2, m3).mass).mass;
      assoc_d = std::max(assoc_d, max_difference(to_vector(left), to_vector(right)));
      many_d = std::max(many_d, max_difference(to_vector(dempster(std::vector<Mass>{m1, m2, m3}).mass), to_vector(left)));
    } catch (const Refusal&) {
    }
    const auto q1 = commonality_function(m1), q2 = commonality_function(m2), q12 = commonality_function(conjunctive(m1, m2));
    for (std::size_t A = 0; A < N; A++) q_prod = std::max(q_prod, std::fabs(q12[A] - q1[A] * q2[A]));
    const auto b12 = morphe::combinatorics::subset_zeta(to_vector(disjunctive(m1, m2)));
    const auto b1 = morphe::combinatorics::subset_zeta(v1), b2 = morphe::combinatorics::subset_zeta(v2);
    for (std::size_t A = 0; A < N; A++) b_prod = std::max(b_prod, std::fabs(b12[A] - b1[A] * b2[A]));
    comm_u = std::max(comm_u, max_difference(to_vector(disjunctive(m1, m2)), to_vector(disjunctive(m2, m1))));
    assoc_u = std::max(assoc_u, max_difference(to_vector(disjunctive(disjunctive(m1, m2), m3)), to_vector(disjunctive(m1, disjunctive(m2, m3)))));

    const auto bet = pignistic(m1);
    pig_sum = std::max(pig_sum, std::fabs(std::accumulate(bet.begin(), bet.end(), 0.0) - 1));
    pig_in = pig_in && in_credal_set(m1, bet, 1e-12);

    // Conditioning: Bel(A | B) = (Bel(A ∪ B̄) − Bel(B̄)) / (1 − Bel(B̄)).
    const Set B = static_cast<Set>(1 + rng.below(N - 1));
    if (plausibility(m1, B) > 1e-9) {
      const Mass c = condition(m1, B);
      const double bb = belief(m1, W ^ B);
      for (std::size_t A = 1; A < N; A++)
        cond_d = std::max(cond_d, std::fabs(belief(c, static_cast<Set>(A)) - (belief(m1, static_cast<Set>(A) | (W ^ B)) - bb) / (1 - bb)));
    }

    std::vector<double> f(n);
    for (double& x : f) x = rng.normal();
    const double lo = lower_expectation(m1, f), hi = upper_expectation(m1, f);
    low_lp = std::max(low_lp, std::fabs(lo - credal_lp(m1, f, false)));
    up_lp = std::max(up_lp, std::fabs(hi - credal_lp(m1, f, true)));
    double vmin = INFINITY, vmax = -INFINITY;
    const Set A = static_cast<Set>(1 + rng.below(N - 1));
    double pmin = INFINITY, pmax = -INFINITY;
    for (const auto& v : vertices(m1)) {
      double e = 0, pa = 0;
      for (std::size_t i = 0; i < n; i++) {
        e += f[i] * v[i];
        if (A >> i & 1) pa += v[i];
      }
      vmin = std::min(vmin, e);
      vmax = std::max(vmax, e);
      pmin = std::min(pmin, pa);
      pmax = std::max(pmax, pa);
    }
    low_v = std::max(low_v, std::fabs(lo - vmin));
    up_v = std::max(up_v, std::fabs(hi - vmax));
    const auto iv = probability_bounds(m1, A);
    bounds_v = std::max({bounds_v, std::fabs(iv.lower - pmin), std::fabs(iv.upper - pmax)});
  }
  s.check("random: Bel(A) ≤ Pl(A) for every A (largest excess)", std::max(bel_pl, 0.0), 0);
  s.check("random: Bel(A) + Pl(Ω ∖ A) = 1", sum_one, 1e-15);
  s.check("random: the point functions equal the tabulated ones", point, 1e-15);
  s.check("random: the commonality by its definition", comm, 1e-15);
  s.check("random: the Möbius transform of Bel inverts the belief function", mob_b, 1e-15);
  s.check("random: the Möbius transform of Q inverts the commonality", mob_q, 1e-15);
  s.check("random: m recovered from Pl", mob_pl, 1e-15);
  s.check("random: Dempster's rule against its definition over all pairs of subsets", def_d, 1e-15);
  s.check("random: the conflict K of Dempster's rule and of the conjunctive rule", k_d, 1e-15);
  // The products are summed in another order, so the two sides may differ in the last place.
  s.check("random: Dempster's rule is commutative", comm_d, 1e-15);
  s.check("random: Dempster's rule is associative", assoc_d, 1e-15);
  s.check("random: Dempster's rule of three equals the rule applied twice", many_d, 1e-15);
  s.check("random: the conjunctive rule multiplies commonalities", q_prod, 1e-15);
  s.check("random: the disjunctive rule multiplies implicabilities", b_prod, 1e-15);
  s.check("random: the disjunctive rule is commutative", comm_u, 1e-15);
  s.check("random: the disjunctive rule is associative", assoc_u, 1e-15);
  s.check("random: the pignistic law sums to one", pig_sum, 1e-15);
  s.require("random: the pignistic law lies in the credal set", pig_in);
  s.check("random: conditioning against Bel(A | B) = (Bel(A ∪ B̄) − Bel(B̄)) / (1 − Bel(B̄))", cond_d, 1e-14);
  s.check("random: the lower expectation equals the linear programme's minimum over the credal set", low_lp, 1e-12);
  s.check("random: the upper expectation equals the linear programme's maximum over the credal set", up_lp, 1e-12);
  s.check("random: the lower expectation equals the least expectation over the laws of the orderings", low_v, 1e-15);
  s.check("random: the upper expectation equals the greatest expectation over the laws of the orderings", up_v, 1e-15);
  s.check("random: [Bel(A), Pl(A)] are the least and greatest P(A) over the laws of the orderings", bounds_v, 1e-15);

  // ---- Bayesian masses ----
  {
    const std::vector<double> p{0.1, 0.2, 0.3, 0.4}, q{0.4, 0.3, 0.2, 0.1};
    const Mass a = bayesian(p), b = bayesian(q);
    const auto d = dempster(a, b);
    double z = 0;
    for (std::size_t i = 0; i < 4; i++) z += p[i] * q[i];
    double err = 0;
    for (std::size_t i = 0; i < 4; i++) err = std::max(err, std::fabs(d.mass(singleton(i)) - p[i] * q[i] / z));
    s.check("Bayesian: Dempster's rule is the normalised product p q / Σ p q", err, 1e-16);
    s.check("Bayesian: the conflict is 1 − Σ p q", std::fabs(d.conflict - (1 - z)), 1e-16);
    s.require("Bayesian: recognised as Bayesian, and their combination too", is_bayesian(a) && is_bayesian(d.mass) && !is_bayesian(vacuous(4)));
    s.check("Bayesian: Bel = Pl = P, and the pignistic law is p", std::max(std::fabs(belief(a, 5) - 0.4), max_difference(pignistic(a), p)) + std::fabs(plausibility(a, 5) - 0.4), 1e-16);
  }

  // ---- Zadeh's example ----
  {
    // Two physicians: one says meningitis 0.99, concussion 0.01; the other tumour 0.99, concussion 0.01.
    const Frame fr = frame({"meningitis", "concussion", "tumour"});
    const Mass d1 = mass(3, {{fr.set({"meningitis"}), 0.99}, {fr.set({"concussion"}), 0.01}});
    const Mass d2 = mass(3, {{fr.set({"tumour"}), 0.99}, {fr.set({"concussion"}), 0.01}});
    const auto c = dempster(d1, d2);
    s.check("Zadeh: the conflict is 1 − 0.01 × 0.01", std::fabs(c.conflict - (1 - 0.01 * 0.01)), 1e-15);
    s.require("Zadeh: Dempster's rule gives concussion, which both thought nearly impossible, full belief",
              c.mass.focal.size() == 1 && c.mass.focal[0].set == fr.set({"concussion"}) && c.mass.focal[0].mass == 1);
    const auto y = yager(d1, d2);
    s.check("Zadeh: Yager's rule gives concussion 0.0001 and the frame the conflict", std::fabs(y.mass(fr.set({"concussion"})) - 1e-4) + std::fabs(y.mass(fr.all()) - (1 - 1e-4)), 1e-15);
    const auto u = conjunctive(d1, d2);
    s.check("Zadeh: the conjunctive rule keeps the conflict as m(∅)", std::fabs(u.conflict() - (1 - 1e-4)), 1e-15);
    s.check("Zadeh: unnormalised, Bel(A) + Pl(Ω ∖ A) = 1 − m(∅)", std::fabs(belief(u, fr.set({"concussion"})) + plausibility(u, fr.set({"meningitis", "tumour"})) - 1e-4), 1e-15);
    s.require("Zadeh: credal operations on the unnormalised result are refused", contains(refusal([&] { lower_expectation(u, std::vector<double>{1, 2, 3}); }), "normalise"));
    s.check("Zadeh: normalising the conjunctive result is Dempster's rule", std::fabs(normalise(u)(fr.set({"concussion"})) - 1), 1e-15);
    s.require("Zadeh: the names of a set", fr.names(fr.set({"tumour", "meningitis"})) == std::vector<std::string>{"meningitis", "tumour"});
  }
  s.require("Dempster: total conflict is refused", contains(refusal([] { dempster(categorical(3, 1), categorical(3, 2)); }), "total conflict"));
  s.require("Dempster: conditioning on an event of plausibility zero is refused", contains(refusal([] { condition(categorical(3, 1), 6); }), "plausibility zero"));

  // ---- discounting ----
  {
    calx::random::Stream r2(23, 0x62656c66, 2, 0);
    const Mass m = random_mass(r2, 4, 6);
    s.check("discounting: α = 1 leaves the mass function", max_difference(to_vector(discount(m, 1)), to_vector(m)), 0);
    s.require("discounting: α = 0 gives the vacuous function", is_vacuous(discount(m, 0)));
    const Mass d = discount(m, 0.7);
    double err = 0;
    for (Set A = 1; A < 16; A++) err = std::max(err, std::fabs(d(A) - (0.7 * m(A) + (A == 15 ? 0.3 : 0.0))));
    s.check("discounting: α = 0.7 scales every mass and gives the frame 0.3", err, 1e-16);
  }

  // ---- consonance ----
  {
    const Mass c = mass(4, {{0b0001, 0.5}, {0b0011, 0.2}, {0b1011, 0.3}});
    const auto pl = contour(c);
    double err = 0;
    for (Set A = 1; A < 16; A++) {
      double mx = 0;
      for (std::size_t i = 0; i < 4; i++)
        if (A >> i & 1) mx = std::max(mx, pl[i]);
      err = std::max(err, std::fabs(plausibility(c, A) - mx));
    }
    s.require("consonant: nested focal sets are consonant; crossing ones are not", is_consonant(c) && !is_consonant(mass(4, {{0b0011, 0.5}, {0b0110, 0.5}})));
    s.check("consonant: Pl(A) is the greatest plausibility of A's elements", err, 1e-16);
  }
  return s.done();
}
