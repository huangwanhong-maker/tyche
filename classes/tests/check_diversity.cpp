// Checks of tyche/classes/diversity and partition against closed forms and identities.
//
// Hill numbers are held to a uniform law (^qD = D for every q), to a law of dyadic shares worked by
// hand, to their limits at q = 1 and q = ∞, and to their monotonicity in q on random laws.
// Leinster and Cobbold's diversity is held to the Hill number when Z = I, to the groups' Hill number
// when Z is a block of ones within groups, to 1 when every class is alike, to monotonicity in q, and to
// never exceeding the Hill number. The partition is held to beta = 1 for identical units and N for
// units that share nothing, to 1 ≤ beta ≤ N on random tables, to Jost's q = 1 beta with unequal
// weights as the exponential of a mutual information, and the overlaps to Sørensen's, Jaccard's,
// Horn's and Morisita and Horn's indices worked by hand for two units, and to C_qN = U_{(2−q)N}.

#include <tyche/classes/diversity.hpp>
#include <tyche/classes/partition.hpp>
#include <tyche/probability/information.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace tyche::classes;
using calx::linalg::Matrix;
using calx::linalg::Rectangular;

namespace {

const double inf = std::numeric_limits<double>::infinity();

std::vector<double> random_law(calx::random::Stream& rng, std::size_t n, bool zeros) {
  std::vector<double> p(n);
  double total = 0;
  for (double& x : p) {
    x = rng.exponential();
    if (zeros && rng.uniform() < 0.2) x = 0;
    total += x;
  }
  if (total == 0) p[0] = total = 1;
  for (double& x : p) x /= total;
  return p;
}

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261112, 2, 0, 0);
  const std::vector<double> orders{0, 0.25, 0.5, 0.9, 0.999999, 1, 1.000001, 1.1, 1.5, 2, 3, 7, 50, inf};

  // ---- Hill numbers ----
  {
    const std::vector<double> u(7, 1.0 / 7);
    double worst = 0;
    for (double q : orders) worst = std::max(worst, std::fabs(hill(u, q) - 7));
    s.check("Hill: a uniform law on 7 classes is worth 7 classes at every order", worst, 1e-13);

    const std::vector<double> p{0.5, 0.25, 0.125, 0.125, 0};
    s.check("Hill: ^0D of (1/2, 1/4, 1/8, 1/8, 0) counts its 4 classes", hill(p, 0) - 4, 0);
    s.check("Hill: ^1D = 2^{7/4}, the entropy 7/4 bits", hill(p, 1) - std::pow(2.0, 1.75), 1e-14);
    s.check("Hill: ^2D = 1/0.34375", hill(p, 2) - 1 / 0.34375, 1e-14);
    s.check("Hill: ^3D = 0.14453125^{-1/2}", hill(p, 3) - std::pow(0.14453125, -0.5), 1e-14);
    s.check("Hill: ^∞D = 1/max p = 2", hill(p, inf) - 2, 1e-15);
    s.check("Hill: continuous at q = 1 (q = 1 ± 10⁻⁹)", std::max(std::fabs(hill(p, 1 - 1e-9) - hill(p, 1)), std::fabs(hill(p, 1 + 1e-9) - hill(p, 1))), 1e-8);
    s.check("Hill: tends to ^∞D (q = 10⁶)", hill(p, 1e6) - 2, 1e-5);
    s.check("Rényi: H_2 = −ln Σ p²", renyi(p, 2) + std::log(0.34375), 1e-15);
    s.check("Rényi: H_1 is Shannon's entropy", renyi(p, 1) - tyche::probability::entropy(p), 1e-15);

    double rise = -inf, renyi_gap = 0, profile_gap = 0, unit_gap = 0;
    for (int trial = 0; trial < 100; trial++) {
      const auto law = random_law(rng, 2 + rng.below(30), true);
      const auto profile = hill_profile(law, orders);
      for (std::size_t k = 0; k < orders.size(); k++) {
        profile_gap = std::max(profile_gap, std::fabs(profile[k] - hill(law, orders[k])));
        renyi_gap = std::max(renyi_gap, std::fabs(std::log(profile[k]) - renyi(law, orders[k])));
        if (k > 0) rise = std::max(rise, (profile[k] - profile[k - 1]) / profile[k - 1]);
        unit_gap = std::max(unit_gap, std::max(0.0, 1 - profile[k]));
      }
    }
    s.require("Hill: non-increasing in q on 100 random laws (14 orders from 0 to ∞)", rise <= 1e-12);
    s.require("Hill: never below one class", unit_gap <= 1e-12);
    s.check("Rényi: ln of the Hill number", renyi_gap, 1e-12);
    s.check("profile: the Hill number at each order", profile_gap, 0);
  }

  // ---- Leinster and Cobbold ----
  {
    double identity = 0, ones = 0, blocks = 0, rise = -inf, above = -inf;
    for (int trial = 0; trial < 60; trial++) {
      const std::size_t n = 2 + rng.below(12);
      const auto p = random_law(rng, n, true);
      const Matrix I = Matrix::identity(n);
      Matrix J(n);
      for (double& v : J.a) v = 1;
      // groups: class i in group i % g
      const std::size_t g = 1 + rng.below(n);
      Matrix B(n);
      std::vector<double> group(g, 0.0);
      for (std::size_t i = 0; i < n; i++) {
        group[i % g] += p[i];
        for (std::size_t j = 0; j < n; j++) B(i, j) = (i % g == j % g) ? 1 : 0;
      }
      // a similarity from random points on a line
      std::vector<double> x(n);
      for (double& v : x) v = rng.uniform(0, 3);
      Matrix D(n);
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++) D(i, j) = std::fabs(x[i] - x[j]);
      const Matrix Z = similarity_from_distances(D, 1.3);
      const auto prof = similarity_profile(p, Z, orders);
      for (std::size_t k = 0; k < orders.size(); k++) {
        const double q = orders[k];
        identity = std::max(identity, std::fabs(similarity_diversity(p, I, q) - hill(p, q)) / hill(p, q));
        ones = std::max(ones, std::fabs(similarity_diversity(p, J, q) - 1));
        blocks = std::max(blocks, std::fabs(similarity_diversity(p, B, q) - hill(group, q)) / hill(group, q));
        if (k > 0) rise = std::max(rise, (prof[k] - prof[k - 1]) / prof[k - 1]);
        above = std::max(above, (prof[k] - hill(p, q)) / hill(p, q));
      }
    }
    s.check("Leinster–Cobbold: Z = I gives the Hill number (60 laws × 14 orders)", identity, 1e-12);
    s.check("Leinster–Cobbold: Z of ones (every class alike) gives 1", ones, 1e-12);
    s.check("Leinster–Cobbold: Z of ones within groups gives the groups' Hill number", blocks, 1e-12);
    s.require("Leinster–Cobbold: non-increasing in q", rise <= 1e-12);
    s.require("Leinster–Cobbold: similarity never raises diversity above Hill's", above <= 1e-12);
    const std::vector<double> p{0.5, 0.5};
    Matrix Z(2);
    Z(0, 0) = Z(1, 1) = 1, Z(0, 1) = Z(1, 0) = 0.5;
    double half = 0;
    for (double q : orders) half = std::max(half, std::fabs(similarity_diversity(p, Z, q) - 1 / 0.75));
    s.check("Leinster–Cobbold: two equal classes half alike are worth 1/0.75 at every order", half, 1e-14);
  }

  // ---- refusals ----
  {
    const std::vector<double> bad{0.5, 0.6}, good{0.5, 0.5};
    Matrix Z = Matrix::identity(2);
    Z(1, 1) = 0.9;
    s.require("refused: a law that does not sum to one", throws([&] { hill(bad, 2); }));
    s.require("refused: a negative order", throws([&] { hill(good, -1); }));
    s.require("refused: a class not wholly similar to itself", throws([&] { similarity_diversity(good, Z, 1); }));
  }

  // ---- the partition ----
  {
    const std::vector<double> part_orders{0, 0.5, 1, 2, 3, inf};
    // identical units, of unequal size
    Rectangular same(3, 4);
    const double row[4] = {5, 3, 1, 1};
    for (std::size_t j = 0; j < 3; j++)
      for (std::size_t i = 0; i < 4; i++) same(j, i) = row[i] * static_cast<double>(j + 1);
    // units that share no class, of unequal size
    Rectangular apart(3, 6);
    apart(0, 0) = 4, apart(0, 1) = 1, apart(1, 2) = 7, apart(1, 3) = 2, apart(2, 4) = 1, apart(2, 5) = 9;
    // units of one size and one law
    Rectangular twin(3, 4);
    for (std::size_t j = 0; j < 3; j++)
      for (std::size_t i = 0; i < 4; i++) twin(j, i) = row[i];
    const std::vector<double> sizes{1.0 / 6, 2.0 / 6, 3.0 / 6};
    double one = 0, one_size = 0, scaled = 0, n_equal = 0, n_size = 0, overlaps_one = 0, overlaps_zero = 0;
    for (double q : part_orders) {
      for (auto w : {Weighting::equal, Weighting::size}) {
        const auto a = partition(w == Weighting::equal ? same : twin, q, w);
        (w == Weighting::equal ? one : one_size) = std::max(w == Weighting::equal ? one : one_size, std::fabs(a.beta - 1));
        if (std::isfinite(q)) overlaps_one = std::max({overlaps_one, std::fabs(a.overlap.sorensen - 1), std::fabs(a.overlap.jaccard - 1)});
        const auto b = partition(apart, q, w);
        (w == Weighting::equal ? n_equal : n_size) = std::max(w == Weighting::equal ? n_equal : n_size, std::fabs(b.beta - 3));
        if (std::isfinite(q)) overlaps_zero = std::max({overlaps_zero, std::fabs(b.overlap.sorensen), std::fabs(b.overlap.jaccard)});
      }
      // one law at sizes 1 : 2 : 3, weighted by size: the cells' Hill number is ^qD(w) ^qD(p), so beta = N / ^qD(w)
      scaled = std::max(scaled, std::fabs(partition(same, q, Weighting::size).beta - 3 / hill(sizes, q)));
    }
    s.check("partition: units of one law give beta = 1 equally weighted, whatever their sizes (q = 0, ½, 1, 2, 3, ∞)", one, 1e-13);
    s.check("partition: units of one law and one size give beta = 1 weighted by size", one_size, 1e-13);
    s.check("partition: units of one law at sizes w, weighted by size, give beta = N / ^qD(w)", scaled, 1e-13);
    s.check("partition: units sharing no class, equally weighted, give beta = N = 3", n_equal, 1e-13);
    s.check("partition: units sharing no class, weighted by size, give beta = N = 3", n_size, 1e-13);
    s.check("overlap: 1 for identical units", overlaps_one, 1e-13);
    s.check("overlap: 0 for units sharing no class", overlaps_zero, 1e-13);

    double below = 0, above = 0, size_vs_equal = 0, mutual = 0;
    for (int trial = 0; trial < 80; trial++) {
      const std::size_t N = 2 + rng.below(5), S = 2 + rng.below(10);
      Rectangular z(N, S), level(N, S);
      for (std::size_t j = 0; j < N; j++) {
        const auto pj = random_law(rng, S, true);
        const double size = 1 + 20 * rng.uniform();
        for (std::size_t i = 0; i < S; i++) z(j, i) = size * pj[i], level(j, i) = pj[i];
      }
      for (double q : part_orders) {
        for (auto w : {Weighting::equal, Weighting::size}) {
          const auto r = partition(z, q, w);
          below = std::max(below, 1 - r.beta);
          above = std::max(above, r.beta - static_cast<double>(N));
        }
        // rows of equal totals: weighting by size is weighting equally
        size_vs_equal = std::max(size_vs_equal, std::fabs(partition(level, q, Weighting::size).beta - partition(level, q, Weighting::equal).beta));
      }
      // Jost's q = 1 with unequal weights: beta = exp I(unit; class) in the joint law w_j p_ij
      std::vector<double> w(N), joint(N * S), wl(N);
      double wt = 0;
      for (double& x : w) wt += (x = 0.2 + rng.uniform());
      for (std::size_t j = 0; j < N; j++) wl[j] = w[j] / wt;
      std::vector<double> pooled(S, 0.0);
      for (std::size_t j = 0; j < N; j++)
        for (std::size_t i = 0; i < S; i++) {
          joint[j * S + i] = wl[j] * level(j, i);
          pooled[i] += joint[j * S + i];
        }
      const double I = tyche::probability::entropy(wl) + tyche::probability::entropy(pooled) - tyche::probability::entropy(joint);
      mutual = std::max(mutual, std::fabs(std::log(partition(z, 1, w).beta) - I));
    }
    s.require("partition: beta ≥ 1 on 80 random tables (both weightings, 6 orders)", below <= 1e-12);
    s.require("partition: beta ≤ N", above <= 1e-12);
    s.check("partition: with equal totals, size weights are equal weights", size_vs_equal, 1e-12);
    s.check("partition: Jost's q = 1 beta with unequal weights is exp I(unit; class)", mutual, 1e-13);
    Rectangular z(2, 2);
    z(0, 0) = 1, z(0, 1) = 2, z(1, 0) = 3, z(1, 1) = 1;
    const std::vector<double> uneven{0.3, 0.7}, even{2, 2};
    s.require("partition: Jost's alpha with unequal weights refused for q = 2", throws([&] { partition(z, 2, uneven); }));
    s.check("partition: given equal weights are equal weighting", partition(z, 2, even).beta - partition(z, 2, Weighting::equal).beta, 1e-15);
  }

  // ---- overlaps for two units, by hand ----
  {
    // unit 1: classes a, b, c, d; unit 2: c, d, e. Shared 2, Sørensen 2·2/(4 + 3) = 4/7, Jaccard 2/5.
    Rectangular z(2, 5);
    const double u1[5] = {1, 2, 3, 4, 0}, u2[5] = {0, 0, 5, 1, 4};
    for (std::size_t i = 0; i < 5; i++) z(0, i) = u1[i], z(1, i) = u2[i];
    const auto r0 = partition(z, 0);
    s.check("overlap q = 0: Sørensen's index 2a/(2a + b + c) = 4/7", r0.overlap.sorensen - 4.0 / 7, 1e-14);
    s.check("overlap q = 0: Jaccard's index a/(a + b + c) = 2/5", r0.overlap.jaccard - 0.4, 1e-14);
    s.check("overlap q = 0: 1 − C_0N is Whittaker's (β − 1)/(N − 1)", r0.overlap.sorensen_dissimilarity - (r0.beta - 1), 1e-14);
    const auto r2 = partition(z, 2);
    double p11 = 0, p22 = 0, p12 = 0;
    for (std::size_t i = 0; i < 5; i++) {
      const double a = u1[i] / 10, b = u2[i] / 10;
      p11 += a * a, p22 += b * b, p12 += a * b;
    }
    s.check("overlap q = 2: Morisita and Horn's 2Σ p₁p₂/(Σ p₁² + Σ p₂²)", r2.overlap.sorensen - 2 * p12 / (p11 + p22), 1e-14);
    const auto r1 = partition(z, 1);
    s.check("overlap q = 1: Horn's 1 − ln β / ln 2", r1.overlap.sorensen - (1 - std::log(r1.beta) / std::log(2.0)), 1e-15);
    double dual = 0, cont = 0;
    for (int t = 0; t < 200; t++) {
      const std::size_t N = 2 + rng.below(8);
      const double beta = 1 + (static_cast<double>(N) - 1) * rng.uniform(), q = 2 * rng.uniform();
      dual = std::max(dual, std::fabs(overlap(beta, N, q).sorensen - overlap(beta, N, 2 - q).jaccard));
      cont = std::max(cont, std::fabs(overlap(beta, N, 1 + 1e-9).sorensen - overlap(beta, N, 1).sorensen));
    }
    s.check("overlap: C_qN = U_{(2−q)N} (200 random β, N, q)", dual, 1e-12);
    s.check("overlap: continuous at q = 1", cont, 1e-8);
  }

  return s.done();
}
