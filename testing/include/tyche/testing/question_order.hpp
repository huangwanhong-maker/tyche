// The order of two questions: classical and quantum-like models of sequential judgement, nested, and the QQ equality
// (decision 0032).
//
// Two binary questions A and B are asked of two groups, one in the order A then B, the other B then A; each group's answers
// fall in four cells (yes–yes, yes–no, no–yes, no–no, in the order asked). Three families of models:
//   saturated    each order its own law of the four cells (6 parameters);
//   order-free   one joint law of the two answers whatever the order (3 parameters): no order effect;
//   projective   the answers as projections of one state onto two bases of a real plane (Busemeyer and Bruza 2012, Quantum
//                Models of Cognition and Decision, ch. 4): a state at angle θ, A's "yes" along 0 and B's "yes" along φ, so that
//                in the order A then B, P(yes, yes) = cos²θ cos²φ, P(yes, no) = cos²θ sin²φ, P(no, yes) = sin²θ sin²φ,
//                P(no, no) = sin²θ cos²φ, and in the order B then A the same with θ − φ in place of θ (2 parameters).
// The QQ equality (Wang and Busemeyer 2013, "A quantum question order model supported by empirical tests of an a priori and
// precise prediction", Top. Cogn. Sci. 5, 689–710): any model in which each answer is a projection, in any dimension, makes the
// probability of answering the two questions differently the same in both orders, q = P_AB(yes, no) + P_AB(no, yes) −
// P_BA(yes, no) − P_BA(no, yes) = 0; it needs no parameter. The order-free model satisfies it too, so the equality alone does
// not tell the projective model from the absence of an order effect; what tells them apart is an order effect (the order-free
// model rejected) with the equality holding (Wang, Solloway, Shiffrin and Busemeyer 2014, Proc. Natl Acad. Sci. 111,
// 9431–9436, found it so in 70 national surveys). A classical model in which the first answer moves the second's probability
// by amounts that differ between the questions generally violates it; a symmetric anchoring (the second answer copying the first
// with one probability, else drawn from its own marginal) does not, and the checks show both.
//
// Tests, each between nested models by the likelihood ratio referred to χ²: order-free against saturated (3 degrees of
// freedom: is there an order effect?); q = 0 against saturated (1: the equality; the constrained fit in closed form, a common
// probability of differing answers, the cells within agreement and within disagreement split as observed); projective
// against q = 0 (3: is the plane enough?). And the equality by its direct z-test, q̂ over its standard error.
// Not rejecting the equality is not evidence for it: that needs an equivalence test, two one-sided tests that |q| < Δ for a
// declared margin Δ (Schuirmann 1987, "A comparison of the two one-sided tests procedure and the power approach for assessing
// the equivalence of average bioavailability", J. Pharmacokinet. Biopharm. 15, 657–680), each at level α, the equality shown
// within Δ when both reject.
#pragma once

#include <tyche/probability/distributions.hpp>
#include <tyche/probability/p_values.hpp>

#include <calx/optimisation/nelder_mead.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::testing {

// Counts in the order asked: ab = (A, B) and ba = (B, A), each yes–yes, yes–no, no–yes, no–no.
struct OrderCounts {
  std::array<double, 4> ab{}, ba{};
};

struct Laws {
  std::array<double, 4> ab{}, ba{};  // cell probabilities, in the order asked
};

struct Fit {
  std::string model;
  std::size_t parameters = 0;
  Laws laws;
  double log_likelihood = 0, aic = 0;
  std::vector<double> estimates;  // the model's own parameters (θ and φ for the projective model)
};

struct QQ {
  double q = 0, se = 0, z = 0, p = 1;
};

struct Equivalence {
  double margin = 0;
  double lower_z = 0, upper_z = 0;  // (q + margin)/se and (margin - q)/se
  double p = 1;                     // the larger of the two one-sided p-values
  bool shown = false;
};

struct Nested {
  double statistic = 0, df = 0, p = 1;
};

struct OrderAnalysis {
  QQ qq;
  Fit saturated, order_free, equality, projective;
  Nested order_effect, equality_holds, plane_enough;
  Equivalence equivalence;  // when a margin is declared
  std::string reading;
};

namespace detail {
inline double total(const std::array<double, 4>& c) { return c[0] + c[1] + c[2] + c[3]; }
inline double loglik(const OrderCounts& n, const Laws& L) {
  double s = 0;
  for (int i = 0; i < 4; i++) {
    if (n.ab[i] > 0) s += n.ab[i] * std::log(std::max(L.ab[i], 1e-300));
    if (n.ba[i] > 0) s += n.ba[i] * std::log(std::max(L.ba[i], 1e-300));
  }
  return s;
}
inline Fit finish(std::string name, std::size_t k, const OrderCounts& n, const Laws& L, std::vector<double> est = {}) {
  Fit f;
  f.model = std::move(name), f.parameters = k, f.laws = L, f.estimates = std::move(est);
  f.log_likelihood = loglik(n, L);
  f.aic = 2 * static_cast<double>(k) - 2 * f.log_likelihood;
  return f;
}
inline Nested lr(const Fit& small, const Fit& big) {
  Nested r;
  r.df = static_cast<double>(big.parameters) - static_cast<double>(small.parameters);
  r.statistic = std::max(0.0, 2 * (big.log_likelihood - small.log_likelihood));
  r.p = r.df > 0 ? tyche::probability::ChiSquared{r.df}.survival(r.statistic) : 1;
  return r;
}
}  // namespace detail

// The projective model's laws at (θ, φ).
inline Laws projective_laws(double theta, double phi) {
  const double c = std::cos(theta) * std::cos(theta), s = 1 - c, cp = std::cos(phi) * std::cos(phi), sp = 1 - cp;
  const double c2 = std::cos(theta - phi) * std::cos(theta - phi), s2 = 1 - c2;
  return {{c * cp, c * sp, s * sp, s * cp}, {c2 * cp, c2 * sp, s2 * sp, s2 * cp}};
}

inline QQ qq_test(const OrderCounts& n) {
  const double Na = detail::total(n.ab), Nb = detail::total(n.ba);
  if (!(Na > 0) || !(Nb > 0)) throw std::invalid_argument("tyche::qq_test: answers in both orders");
  const double p1 = (n.ab[1] + n.ab[2]) / Na, p2 = (n.ba[1] + n.ba[2]) / Nb;
  QQ r;
  r.q = p1 - p2;
  r.se = std::sqrt(p1 * (1 - p1) / Na + p2 * (1 - p2) / Nb);
  r.z = r.se > 0 ? r.q / r.se : 0;
  r.p = r.se > 0 ? tyche::probability::two_sided_normal(r.z) : (r.q == 0 ? 1 : 0);
  return r;
}

// The equality shown within a margin by two one-sided tests at level alpha.
inline Equivalence qq_equivalence(const OrderCounts& n, double margin, double alpha = 0.05) {
  if (!(margin > 0)) throw std::invalid_argument("tyche::qq_equivalence: a positive margin");
  const QQ t = qq_test(n);
  Equivalence e;
  e.margin = margin;
  if (!(t.se > 0)) {
    e.shown = std::fabs(t.q) < margin;
    e.p = e.shown ? 0 : 1;
    return e;
  }
  e.lower_z = (t.q + margin) / t.se, e.upper_z = (margin - t.q) / t.se;
  const tyche::probability::Normal N{0, 1};
  e.p = std::max(N.survival(e.lower_z), N.survival(e.upper_z));
  e.shown = e.p < alpha;
  return e;
}

inline OrderAnalysis analyse_order(const OrderCounts& n, double level = 0.05, double margin = 0) {
  const double Na = detail::total(n.ab), Nb = detail::total(n.ba);
  if (!(Na > 0) || !(Nb > 0)) throw std::invalid_argument("tyche::analyse_order: answers in both orders");
  OrderAnalysis R;
  R.qq = qq_test(n);
  Laws sat;
  for (int i = 0; i < 4; i++) sat.ab[i] = n.ab[i] / Na, sat.ba[i] = n.ba[i] / Nb;
  R.saturated = detail::finish("saturated", 6, n, sat);
  // Order-free: one joint law of (A, B); in the order B then A the cells (B, A) = yy, yn, ny, nn are (A, B) = yy, ny, yn, nn.
  std::array<double, 4> j{n.ab[0] + n.ba[0], n.ab[1] + n.ba[2], n.ab[2] + n.ba[1], n.ab[3] + n.ba[3]};
  for (double& v : j) v /= Na + Nb;
  R.order_free = detail::finish("order-free", 3, n, {{j[0], j[1], j[2], j[3]}, {j[0], j[2], j[1], j[3]}});
  // q = 0: a common probability of differing answers; within agreement and within disagreement the cells split as observed.
  {
    const double da = n.ab[1] + n.ab[2], db = n.ba[1] + n.ba[2], d = (da + db) / (Na + Nb);
    auto split = [d](const std::array<double, 4>& c) {
      const double agree = c[0] + c[3], differ = c[1] + c[2];
      std::array<double, 4> p{};
      p[0] = agree > 0 ? (1 - d) * c[0] / agree : (1 - d) / 2;
      p[3] = agree > 0 ? (1 - d) * c[3] / agree : (1 - d) / 2;
      p[1] = differ > 0 ? d * c[1] / differ : d / 2;
      p[2] = differ > 0 ? d * c[2] / differ : d / 2;
      return p;
    };
    R.equality = detail::finish("q = 0", 5, n, {split(n.ab), split(n.ba)});
  }
  // Projective: θ, φ by maximum likelihood, from the best of a grid, refined by Nelder and Mead.
  {
    auto nll = [&](std::span<const double> x) { return -detail::loglik(n, projective_laws(x[0], x[1])); };
    double best = INFINITY;
    std::vector<double> x0{0, 0};
    const int G = 60;
    for (int a = 0; a < G; a++)
      for (int b = 0; b < G; b++) {
        const std::vector<double> x{std::numbers::pi * (a + 0.5) / G, std::numbers::pi * (b + 0.5) / G};
        const double v = nll(x);
        if (v < best) best = v, x0 = x;
      }
    calx::optimisation::NelderMeadOptions o;
    const auto r = calx::optimisation::nelder_mead(nll, x0, o);
    const std::vector<double> x = r.value < best ? r.x : x0;
    R.projective = detail::finish("projective (a real plane)", 2, n, projective_laws(x[0], x[1]), x);
  }
  R.order_effect = detail::lr(R.order_free, R.saturated);
  R.equality_holds = detail::lr(R.equality, R.saturated);
  R.plane_enough = detail::lr(R.projective, R.equality);
  if (margin > 0) R.equivalence = qq_equivalence(n, margin, level);
  const bool effect = R.order_effect.p < level, holds = R.equality_holds.p >= level;
  const std::string within = margin > 0 ? (R.equivalence.shown ? ", and shown within a margin of " + std::to_string(margin) : ", though not shown within a margin of " + std::to_string(margin)) : "";
  R.reading = !effect ? (holds ? "no order effect shown; the equality not rejected" + within + ", as the order-free model also says" : "no order effect shown, yet the equality rejected")
                      : (holds ? "an order effect with the equality not rejected" + within + ": the order-free model fails where projective models do not"
                               : "an order effect that breaks the equality: no projective model of these answers");
  return R;
}

}  // namespace tyche::testing
