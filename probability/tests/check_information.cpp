// Checks of tyche/probability/information against identities and closed forms.
//
// The chain rule of relative entropy through a record is held on random finite laws and kernels
// (an identity, to rounding), with its limits: a record that shows everything hides nothing, one
// that shows nothing hides everything, and a garbling of a record shows no more than it. The
// rate at which two jump processes part is held to its closed form on the two-state chain,
// 2p(1 − p)[c ln(c/c′) − c + c′], and to the relative entropy of the transition laws over a short
// time, computed from the matrix exponentials.

#include <tyche/probability/information.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace tyche::probability;
using calx::linalg::Matrix;

namespace {

std::vector<double> random_law(calx::random::Stream& rng, std::size_t n) {
  std::vector<double> p(n);
  double total = 0;
  for (double& x : p) total += (x = 0.05 + rng.uniform());
  for (double& x : p) x /= total;
  return p;
}

Kernel random_kernel(calx::random::Stream& rng, std::size_t n, std::size_t m) {
  Kernel K(n, m);
  for (std::size_t i = 0; i < n; i++) {
    const auto row = random_law(rng, m);
    for (std::size_t y = 0; y < m; y++) K(i, y) = row[y];
  }
  return K;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261112, 1, 0, 0);

  // ---- entropy and relative entropy ----
  {
    const std::vector<double> u(8, 0.125), p{0.5, 0.25, 0.125, 0.125, 0, 0, 0, 0};
    s.check("entropy: the uniform law on eight outcomes holds ln 8", entropy(u) - std::log(8.0), 1e-15);
    s.check("entropy: (1/2, 1/4, 1/8, 1/8) holds 7/4 bits", entropy(p) / std::log(2.0) - 1.75, 1e-15);
    s.check("relative entropy: against the uniform law, ln n − H", relative_entropy(p, u) - (std::log(8.0) - entropy(p)), 1e-15);
    s.require("relative entropy: infinite where the first law is not absolutely continuous with respect to the second", std::isinf(relative_entropy(u, p)));
    s.check("relative entropy: of Bernoulli(0.9) from Bernoulli(0.1), 0.8 ln 9", bernoulli_relative_entropy(0.9, 0.1) - 0.8 * std::log(9.0), 1e-15);
  }

  // ---- the chain rule through a record ----
  {
    double worst = 0, least = 1e9, garble = -1e9, joint = 0;
    for (int trial = 0; trial < 200; trial++) {
      const std::size_t n = 3 + rng.below(6), m = 2 + rng.below(4);
      const auto p = random_law(rng, n), q = random_law(rng, n);
      const auto K = random_kernel(rng, n, m);
      const auto d = discrepancy(p, q, K);
      worst = std::max(worst, std::fabs(d.total - d.visible - d.hidden));
      least = std::min({least, d.visible, d.hidden});
      const auto G = random_kernel(rng, m, 2 + rng.below(3));
      garble = std::max(garble, discrepancy(p, q, garbled(K, G)).visible - d.visible);
      // The joint law of outcome and record, when the two candidates also record differently.
      const auto K2 = random_kernel(rng, n, m);
      std::vector<double> jp(n * m), jq(n * m);
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t y = 0; y < m; y++) jp[i * m + y] = p[i] * K(i, y), jq[i * m + y] = q[i] * K2(i, y);
      joint = std::max(joint, std::fabs(relative_entropy(jp, jq) - d.total - kernel_discrepancy(p, K, K2)));
    }
    s.check("chain rule: total = visible + hidden on 200 random laws and kernels", worst, 1e-13);
    s.require("chain rule: neither part is negative", least >= -1e-15);
    s.require("data processing: a garbled record shows no more", garble <= 1e-13);
    s.check("joint law: D(p ⊗ K ‖ q ⊗ K′) = D(p ‖ q) + E_p D(K ‖ K′)", joint, 1e-13);

    const auto p = random_law(rng, 6), q = random_law(rng, 6);
    Kernel all(6, 6), none(6, 1);
    for (std::size_t i = 0; i < 6; i++) all(i, i) = 1, none(i, 0) = 1;
    s.check("a record that shows the outcome hides nothing", discrepancy(p, q, all).hidden, 1e-15);
    s.check("a record that shows nothing hides everything", discrepancy(p, q, none).hidden - relative_entropy(p, q), 1e-15);
    const std::vector<std::size_t> cell{0, 0, 1, 1, 2, 2};
    const auto d = discrepancy(p, q, coarsening(cell));
    std::vector<double> cp(3, 0.0), cq(3, 0.0);
    for (std::size_t i = 0; i < 6; i++) cp[cell[i]] += p[i], cq[cell[i]] += q[i];
    s.check("a coarsening shows the relative entropy of the cells' laws", d.visible - relative_entropy(cp, cq), 1e-15);
  }

  // ---- two procedures with one law of outcomes ----
  {
    // Two parties are each heard or not, then a fair lottery awards. The first procedure hears each with probability 0.9;
    // the second hears one with 0.9 and the other with 0.1. Histories: (heard 1, heard 2, award), sixteen... eight of them.
    std::vector<double> p(8), q(8);
    std::vector<std::size_t> award(8);
    for (std::size_t h = 0; h < 8; h++) {
      const bool a = h & 1, b = h & 2;
      p[h] = (a ? 0.9 : 0.1) * (b ? 0.9 : 0.1) * 0.5;
      q[h] = (a ? 0.9 : 0.1) * (b ? 0.1 : 0.9) * 0.5;
      award[h] = (h >> 2) & 1;
    }
    const auto d = discrepancy(p, q, coarsening(award));
    s.check("outcome-equivalent procedures: nothing of their difference is visible in the awards", d.visible, 1e-15);
    s.check("outcome-equivalent procedures: all of it, 0.8 ln 9 = 1.7578 nats, is hidden", d.hidden - 0.8 * std::log(9.0), 1e-14);
  }

  // ---- jump processes ----
  {
    const double p1 = 0.4, c = 1, c2 = 4;
    Matrix k(2), k2(2);
    k(0, 1) = c * p1, k(1, 0) = c * (1 - p1);
    k2(0, 1) = c2 * p1, k2(1, 0) = c2 * (1 - p1);
    const double closed = 2 * p1 * (1 - p1) * (c * std::log(c / c2) - c + c2);
    s.check("two-state chain: the rate 2p(1 − p)[c ln(c/c′) − c + c′] = 0.7746", path_discrepancy_rate(k, k2) - closed, 1e-14);
    s.check("two-state chain: the value 0.7746", closed - 0.7746, 5e-5);
    const auto pi = physis::statistical_mechanics::stationary(k), pi2 = physis::statistical_mechanics::stationary(k2);
    s.check("two-state chain: the stationary laws coincide, so a cross-section shows nothing", relative_entropy(pi, pi2), 1e-15);
    s.check("jump processes: equal rates part at rate zero", path_discrepancy_rate(k, k), 1e-15);
    Matrix closedOff = k2;
    closedOff(0, 1) = 0;
    s.require("jump processes: a jump the second process cannot make parts them at once", std::isinf(path_discrepancy_rate(k, closedOff)));

    // Over a short time h from the stationary law, the paths are told apart by their jumps alone: the relative entropy of the laws
    // of (start, end) is the rate times h to first order, on a chain of four states.
    Matrix a(4), b(4);
    for (std::size_t i = 0; i < 4; i++)
      for (std::size_t j = 0; j < 4; j++)
        if (i != j) a(i, j) = 0.2 + rng.uniform(), b(i, j) = 0.2 + rng.uniform();
    const auto st = physis::statistical_mechanics::stationary(a);
    const double h = 1e-4;
    std::vector<double> ja(16), jb(16);
    for (std::size_t i = 0; i < 4; i++) {
      std::vector<double> e(4, 0.0);
      e[i] = 1;
      const auto ra = physis::statistical_mechanics::evolve(a, e, h), rb = physis::statistical_mechanics::evolve(b, e, h);
      for (std::size_t j = 0; j < 4; j++) ja[i * 4 + j] = st[i] * ra[j], jb[i * 4 + j] = st[i] * rb[j];
    }
    s.check("jump processes: over a short time the laws of start and end part at the rate, to first order", relative_entropy(ja, jb) / h / path_discrepancy_rate(a, b) - 1, 2e-3);
  }

  // ---- what is refused ----
  {
    bool threw = false;
    const std::vector<double> p{0.5, 0.5}, bad{0.7, 0.7};
    Kernel K(2, 2);
    K(0, 0) = K(1, 1) = 1;
    try { (void)discrepancy(p, bad, K); } catch (const std::invalid_argument&) { threw = true; }
    s.require("a law that does not sum to one is refused", threw);
    threw = false;
    Kernel wrong(2, 2);
    wrong(0, 0) = 1, wrong(1, 0) = 0.4;
    try { (void)discrepancy(p, p, wrong); } catch (const std::invalid_argument&) { threw = true; }
    s.require("a kernel whose rows are not laws is refused", threw);
  }
  return s.done();
}
