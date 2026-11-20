// Checks of tyche/testing/question_order: counts made exactly by a projective model give q = 0, a plane that fits as well as
// the saturated model, and an order effect, read as such; its angles reproduce its laws; counts made by one joint law in both
// orders show no order effect and keep the equality; a classical Markov update of the first answer, asymmetric between the
// questions, breaks the equality; the constrained fit keeps q = 0 exactly; and over 1,000 surveys drawn from a projective model the z-test of the
// equality rejects about 5% at the 5% level.

#include <tyche/testing/question_order.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>

using namespace tyche::testing;

namespace {
OrderCounts expected(const Laws& L, double n) {
  OrderCounts c;
  for (int i = 0; i < 4; i++) c.ab[i] = n * L.ab[i], c.ba[i] = n * L.ba[i];
  return c;
}
std::size_t draw(const std::array<double, 4>& p, calx::random::Stream& r) {
  double u = r.uniform(), s = 0;
  for (std::size_t i = 0; i < 3; i++)
    if ((s += p[i]) > u) return i;
  return 3;
}
}  // namespace

int main() {
  calx::verification::Suite s;
  {
    const Laws L = projective_laws(0.7, 1.1);
    const auto A = analyse_order(expected(L, 1000));
    s.check("a projective model's counts: q = 0 exactly", A.qq.q, 1e-12);
    s.check("... the plane fits them as well as the saturated model", A.saturated.log_likelihood - A.projective.log_likelihood, 1e-6);
    double worst = 0;
    for (int i = 0; i < 4; i++) worst = std::max({worst, std::fabs(A.projective.laws.ab[i] - L.ab[i]), std::fabs(A.projective.laws.ba[i] - L.ba[i])});
    s.check("... its fitted angles reproduce the laws (the angles themselves are known only up to reflections)", worst, 1e-6);
    s.require("... an order effect with the equality holding, read as such", A.order_effect.p < 1e-6 && A.equality_holds.p > 0.99 && A.reading.find("projective models do not") != std::string::npos,
              A.reading);
  }
  {
    const std::array<double, 4> j{0.4, 0.1, 0.2, 0.3};  // (A, B) = yy, yn, ny, nn
    OrderCounts c;
    for (int i = 0; i < 4; i++) c.ab[i] = 800 * j[i];
    c.ba = {800 * j[0], 800 * j[2], 800 * j[1], 800 * j[3]};  // (B, A)
    const auto A = analyse_order(c);
    s.check("one joint law in both orders: no order effect (the likelihood ratio is zero)", A.order_effect.statistic, 1e-9);
    s.check("... and the equality holds", A.qq.q, 1e-12);
    s.require("... read as no order effect", A.reading.find("no order effect shown; the equality not rejected") == 0, A.reading);
  }
  {
    // A Markov update of the first answer: in the order A then B, P(A yes) = 0.6, P(B yes | A yes) = 0.8, P(B yes | A no) = 0.3;
    // in the order B then A, P(B yes) = 0.4, P(A yes | B yes) = 0.7, P(A yes | B no) = 0.5. Differing answers: 0.24 against 0.42.
    auto markov = [](double first, double yes_yes, double yes_no) {
      return std::array<double, 4>{first * yes_yes, first * (1 - yes_yes), (1 - first) * yes_no, (1 - first) * (1 - yes_no)};
    };
    OrderCounts c;
    const auto ab = markov(0.6, 0.8, 0.3), ba = markov(0.4, 0.7, 0.5);
    for (int i = 0; i < 4; i++) c.ab[i] = 1500 * ab[i], c.ba[i] = 1500 * ba[i];
    const auto A = analyse_order(c);
    s.require("a classical Markov update of the first answer breaks the equality (q = −0.18): rejected, and read so", A.equality_holds.p < 1e-3 && A.reading.find("breaks the equality") != std::string::npos,
              A.reading + ", q = " + std::to_string(A.qq.q));
    const auto& E = A.equality.laws;
    s.check("the fit under q = 0 keeps it exactly", (E.ab[1] + E.ab[2]) - (E.ba[1] + E.ba[2]), 1e-12);
    // A symmetric anchoring (the second answer copies the first with probability 0.3, else is drawn from its own marginal) keeps
    // the equality: differing answers are 0.7 (f(1 − s) + (1 − f)s) in either order, so the equality alone singles out nothing.
    auto anchored = [](double f, double sm, double cp) {
      return std::array<double, 4>{f * (cp + (1 - cp) * sm), f * (1 - cp) * (1 - sm), (1 - f) * (1 - cp) * sm, (1 - f) * (cp + (1 - cp) * (1 - sm))};
    };
    OrderCounts d;
    const auto ab2 = anchored(0.6, 0.4, 0.3), ba2 = anchored(0.4, 0.6, 0.3);
    for (int i = 0; i < 4; i++) d.ab[i] = 1500 * ab2[i], d.ba[i] = 1500 * ba2[i];
    s.check("a symmetric anchoring keeps the equality exactly, though it is classical", qq_test(d).q, 1e-12);
  }
  {
    const Laws L = projective_laws(0.4, 0.9);
    calx::random::Stream r(77, 1, 0, 0);
    int rejected = 0;
    const int surveys = 1000;
    for (int k = 0; k < surveys; k++) {
      OrderCounts c;
      for (int i = 0; i < 500; i++) c.ab[draw(L.ab, r)] += 1, c.ba[draw(L.ba, r)] += 1;
      rejected += qq_test(c).p < 0.05;
    }
    s.check("1,000 surveys of 500 per order from a projective model: the equality rejected about 5% of the time", rejected / 1000.0 - 0.05, 0.018);
    // Equivalence within 0.05: with 2,000 per order it is shown; with 100 per order it is not, though the equality is true.
    OrderCounts big, small;
    for (int i = 0; i < 2000; i++) big.ab[draw(L.ab, r)] += 1, big.ba[draw(L.ba, r)] += 1;
    for (int i = 0; i < 100; i++) small.ab[draw(L.ab, r)] += 1, small.ba[draw(L.ba, r)] += 1;
    s.require("with 2,000 per order the equality is shown within 0.05 by two one-sided tests", qq_equivalence(big, 0.05).shown);
    s.require("with 100 per order it is not: not rejecting the equality is not evidence for it", !qq_equivalence(small, 0.05).shown);
    const auto A = analyse_order(big, 0.05, 0.05);
    s.require("... and the reading says which", A.reading.find("shown within") != std::string::npos, A.reading);
  }
  return s.done();
}
