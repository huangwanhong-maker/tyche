// Checks of tyche/networks/qap and null_models: QAP correlation and MRQAP, and conditional uniform graphs.
//
// QAP correlation: r is Pearson's correlation over the off-diagonal cells; its exact p-values equal those
// counted by relabelling a 5-node matrix by each of the 120 permutations and recomputing r from scratch;
// random relabellings approach them; symmetric data give the same r and exact p from the pairs alone; under
// independence the p-values are near uniform. MRQAP: the coefficients are least squares over the cells; the
// double semi-partialling statistic at the identity is the full model's t (Frisch, Waugh and Lovell); exact
// Y-permutation p-values equal a brute-force count; a real effect is found and an absent one is not, and
// the false-positive rate of an irrelevant matrix correlated with a relevant one stays near its level.
// Null models: G(n, m) has m ties and is uniform over the 20 graphs of 3 ties on 4 nodes; double-edge swaps
// keep every degree, directed and undirected, and reach the 70 two-regular graphs on 6 nodes uniformly
// (10 are two triangles); CUG tests are reproducible by seed, and place a ring lattice's transitivity far
// above both nulls and the reciprocity ρ of random digraphs at 0.

#include <tyche/networks/descriptive.hpp>
#include <tyche/networks/network.hpp>
#include <tyche/networks/null_models.hpp>
#include <tyche/networks/qap.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace tyche::networks;
using calx::linalg::Rectangular;

namespace {

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

Matrix random_matrix(calx::random::Stream& rng, std::size_t n, bool symmetric) {
  Matrix A(n);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = symmetric ? i + 1 : 0; j < n; j++) {
      if (i == j) continue;
      A(i, j) = rng.normal();
      if (symmetric) A(j, i) = A(i, j);
    }
  return A;
}

// Pearson's correlation over the off-diagonal cells, from scratch.
double cell_correlation(const Matrix& A, const Matrix& B) {
  const std::size_t n = A.n;
  double m = 0, ma = 0, mb = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++)
      if (i != j) ma += A(i, j), mb += B(i, j), m += 1;
  ma /= m, mb /= m;
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < n; j++)
      if (i != j) sab += (A(i, j) - ma) * (B(i, j) - mb), saa += (A(i, j) - ma) * (A(i, j) - ma), sbb += (B(i, j) - mb) * (B(i, j) - mb);
  return sab / std::sqrt(saa * sbb);
}

Matrix relabel(const Matrix& B, const std::vector<std::size_t>& p) {
  Matrix C(B.n);
  for (std::size_t i = 0; i < B.n; i++)
    for (std::size_t j = 0; j < B.n; j++) C(i, j) = B(p[i], p[j]);
  return C;
}

bool same_degrees(const Network& a, const Network& b) {
  const auto da = degrees(a), db = degrees(b);
  return da.out == db.out && da.in == db.in;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261113, 0x6e657477, 103, 0);

  // ---- QAP correlation ----
  {
    const auto A = random_matrix(rng, 5, false);
    Matrix B = random_matrix(rng, 5, false);
    for (std::size_t i = 0; i < 5; i++)
      for (std::size_t j = 0; j < 5; j++) B(i, j) += 0.8 * A(i, j);
    QapOptions ex;
    ex.exact = true;
    const auto q = qap_correlation(A, B, ex);
    s.check("QAP: r is Pearson's correlation over the off-diagonal cells", q.r - cell_correlation(A, B), 1e-14);
    std::vector<std::size_t> p(5);
    std::iota(p.begin(), p.end(), std::size_t{0});
    double ge = 0, le = 0, two = 0, count = 0;
    do {
      const double r = cell_correlation(A, relabel(B, p));
      ge += r >= q.r - 1e-12, le += r <= q.r + 1e-12, two += std::fabs(r) >= std::fabs(q.r) - 1e-12, count += 1;
    } while (std::next_permutation(p.begin(), p.end()));
    s.require("QAP exact: 120 relabellings of a 5-node matrix", q.permutations == 120 && count == 120 && q.exact);
    s.check("QAP exact p-values (greater, less, two-sided) equal a count over every relabelling, r recomputed from scratch",
            std::fabs(q.p_greater - ge / 120) + std::fabs(q.p_less - le / 120) + std::fabs(q.p_two_sided - two / 120), 1e-15);
    s.require("QAP exact: the observed relabelling counted (p ≥ 1/120)", q.p_greater >= 1.0 / 120 - 1e-15);
    QapOptions many;
    many.permutations = 20000;
    const auto qr = qap_correlation(A, B, many);
    s.check("QAP: 20 000 random relabellings approach the exact p-value", qr.p_greater - q.p_greater, 0.01);
    const auto qr2 = qap_correlation(A, B, many);
    QapOptions other = many;
    other.seed = 7;
    s.require("QAP: the same seed gives the same p-value, another seed another", qr2.p_greater == qr.p_greater && qap_correlation(A, B, other).p_greater != qr.p_greater);
    const auto S = random_matrix(rng, 6, true);
    Matrix T = random_matrix(rng, 6, true);
    for (std::size_t i = 0; i < 6; i++)
      for (std::size_t j = 0; j < 6; j++) T(i, j) += 0.5 * S(i, j);
    QapOptions sym;
    sym.exact = true;
    sym.symmetric = true;
    const auto qs = qap_correlation(S, T, sym), qf = qap_correlation(S, T, ex);
    s.check("symmetric data: the pairs i < j give the same r and exact p-value as every cell", std::fabs(qs.r - qf.r) + std::fabs(qs.p_two_sided - qf.p_two_sided), 1e-14);
    std::size_t rejected = 0;
    const int R = 200;
    QapOptions few;
    few.permutations = 199;
    for (int r = 0; r < R; r++) {
      few.seed = 1000 + static_cast<std::uint64_t>(r);
      if (qap_correlation(random_matrix(rng, 10, false), random_matrix(rng, 10, false), few).p_two_sided <= 0.05) rejected++;
    }
    s.check("QAP under independence: the share of p ≤ 0.05 in 200 tests, less 0.05", static_cast<double>(rejected) / R - 0.05, 0.04);
    s.require("QAP: a constant matrix is refused", throws([&] { qap_correlation(A, Matrix(5), ex); }));
    s.require("QAP: an exact test of eleven actors is refused", throws([&] { qap_correlation(random_matrix(rng, 11, false), random_matrix(rng, 11, false), QapOptions{999, 1, true, false}); }));
  }

  // ---- MRQAP ----
  {
    const std::size_t n = 12;
    const auto X1 = random_matrix(rng, n, false), Z = random_matrix(rng, n, false), E = random_matrix(rng, n, false);
    Matrix X2(n), Y(n);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < n; j++) {
        if (i == j) continue;
        X2(i, j) = 0.6 * X1(i, j) + 0.8 * Z(i, j);  // correlated with X₁, no effect on Y
        Y(i, j) = 1 + 0.5 * X1(i, j) + E(i, j);
      }
    MrqapOptions o;
    o.permutations = 499;
    const auto m = mrqap(Y, {X1, X2}, o);
    {
      // Least squares over the cells, by hand through tyche's ols.
      Rectangular D(n * (n - 1), 3);
      std::vector<double> y;
      std::size_t r = 0;
      for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < n; j++)
          if (i != j) D(r, 0) = 1, D(r, 1) = X1(i, j), D(r, 2) = X2(i, j), y.push_back(Y(i, j)), r++;
      const auto f = tyche::regression::ols(D, y);
      s.check("MRQAP: the coefficients are least squares over the n(n − 1) cells", std::fabs(f.beta[1] - m.fit.beta[1]) + std::fabs(f.beta[2] - m.fit.beta[2]), 1e-13);
    }
    std::vector<std::size_t> id(n);
    std::iota(id.begin(), id.end(), std::size_t{0});
    s.check("double semi-partialling at the identity gives the full model's t (Frisch, Waugh and Lovell), X₁",
            dsp_statistic(Y, {X1, X2}, 1, id) - m.fit.t[1], 1e-9 * (1 + std::fabs(m.fit.t[1])));
    s.check("and X₂", dsp_statistic(Y, {X1, X2}, 2, id) - m.fit.t[2], 1e-9 * (1 + std::fabs(m.fit.t[2])));
    s.require("MRQAP (double semi-partialling): the effect of X₁ found (p ≤ 0.01)", m.p_two_sided[1] <= 0.01);
    s.require("MRQAP: no p-value for the intercept", std::isnan(m.p_two_sided[0]));
    MrqapOptions yp = o;
    yp.method = MrqapNull::y_permutation;
    const auto my = mrqap(Y, {X1, X2}, yp);
    s.require("MRQAP (Y-permutation): the effect of X₁ found (p ≤ 0.01)", my.p_two_sided[1] <= 0.01);
    // Exact Y-permutation p-values on five actors, against a count over every relabelling.
    const auto a1 = random_matrix(rng, 5, false), a2 = random_matrix(rng, 5, false), ay = random_matrix(rng, 5, false);
    MrqapOptions exy;
    exy.method = MrqapNull::y_permutation;
    exy.exact = true;
    const auto me = mrqap(ay, {a1, a2}, exy);
    std::vector<std::size_t> p(5);
    std::iota(p.begin(), p.end(), std::size_t{0});
    double two1 = 0, two2 = 0;
    do {
      const auto f = mrqap(relabel(ay, p), {a1, a2}, MrqapOptions{MrqapNull::y_permutation, true, 0, 1, false, false}).fit;
      two1 += std::fabs(f.t[1]) >= std::fabs(me.fit.t[1]) - 1e-10 * (1 + std::fabs(me.fit.t[1]));
      two2 += std::fabs(f.t[2]) >= std::fabs(me.fit.t[2]) - 1e-10 * (1 + std::fabs(me.fit.t[2]));
    } while (std::next_permutation(p.begin(), p.end()));
    s.check("MRQAP exact (Y-permutation, five actors): p-values equal a count over the 120 relabellings", std::fabs(me.p_two_sided[1] - two1 / 120) + std::fabs(me.p_two_sided[2] - two2 / 120), 1e-15);
    MrqapOptions exd;
    exd.exact = true;
    const auto md = mrqap(ay, {a1, a2}, exd);
    double d1 = 0;
    std::iota(p.begin(), p.end(), std::size_t{0});
    do d1 += std::fabs(dsp_statistic(ay, {a1, a2}, 1, p)) >= std::fabs(md.fit.t[1]) - 1e-10 * (1 + std::fabs(md.fit.t[1]));
    while (std::next_permutation(p.begin(), p.end()));
    s.check("MRQAP exact (double semi-partialling, five actors): the p-value equals a count of dsp_statistic over the 120 relabellings", md.p_two_sided[1] - d1 / 120, 1e-15);
    // The false-positive rate of X₂ (correlated with X₁, no effect) under double semi-partialling.
    std::size_t false_pos = 0;
    const int R = 100;
    MrqapOptions cal;
    cal.permutations = 99;
    for (int r = 0; r < R; r++) {
      const auto x1 = random_matrix(rng, 10, false), z = random_matrix(rng, 10, false), e = random_matrix(rng, 10, false);
      Matrix x2(10), y(10);
      for (std::size_t i = 0; i < 10; i++)
        for (std::size_t j = 0; j < 10; j++)
          if (i != j) x2(i, j) = 0.7 * x1(i, j) + 0.7 * z(i, j), y(i, j) = x1(i, j) + e(i, j);
      cal.seed = 5000 + static_cast<std::uint64_t>(r);
      if (mrqap(y, {x1, x2}, cal).p_two_sided[2] <= 0.05) false_pos++;
    }
    s.check("MRQAP (double semi-partialling): the false-positive rate of an irrelevant, correlated matrix in 100 tests, less 0.05", static_cast<double>(false_pos) / R - 0.05, 0.06);
  }

  // ---- null models ----
  {
    bool ties_ok = true;
    for (int t = 0; t < 20; t++) {
      const std::size_t n = 3 + rng.below(20);
      const bool directed = t % 2;
      const std::size_t N = directed ? n * (n - 1) : n * (n - 1) / 2;
      const std::size_t m = rng.below(N + 1);
      const auto g = uniform_network(n, m, directed, rng);
      ties_ok = ties_ok && g.size() == m && g.directed == directed;
    }
    s.require("G(n, m): exactly m ties, both below and above half of the pairs (20 draws)", ties_ok);
    std::map<std::vector<Tie>, int> freq;
    const int draws = 20000;
    for (int r = 0; r < draws; r++) freq[uniform_network(4, 3, false, rng).ties]++;
    int lo = draws, hi = 0;
    for (const auto& [k, c] : freq) lo = std::min(lo, c), hi = std::max(hi, c);
    s.require("G(4, 3) undirected: all 20 graphs drawn", freq.size() == 20);
    s.check("G(4, 3): each graph's share within 0.2 of 1/20 (relative), 20 000 draws", std::max(std::fabs(lo / 1000.0 - 1), std::fabs(hi / 1000.0 - 1)), 0.2);
    bool kept = true;
    std::size_t accepted_total = 0;
    for (int t = 0; t < 20; t++) {
      std::vector<Tie> tt;
      const std::size_t n = 10 + rng.below(20);
      const bool directed = t % 2;
      for (std::size_t a = 0; a < n; a++)
        for (std::size_t b = directed ? 0 : a + 1; b < n; b++)
          if (a != b && rng.uniform() < 0.2) tt.push_back({a, b});
      const auto g = network(n, tt, directed);
      std::size_t acc = 0;
      const auto h = degree_preserving(g, 20 * g.size(), rng, &acc);
      kept = kept && same_degrees(g, h) && h.size() == g.size();
      accepted_total += acc;
    }
    s.require("double-edge swaps keep every degree (in and out when directed) and the number of ties (20 networks)", kept && accepted_total > 0);
    // Two-regular graphs on six labelled nodes: 60 six-cycles and 10 pairs of triangles.
    std::vector<Tie> hex;
    for (std::size_t a = 0; a < 6; a++) hex.push_back({a, (a + 1) % 6});
    const auto cyc = network(6, hex, false);
    std::set<std::vector<Tie>> seen;
    double triangles_pair = 0;
    const int D = 6000;
    for (int r = 0; r < D; r++) {
      calx::random::Stream st(77, 0x6e657477, 104, static_cast<std::uint32_t>(r));
      const auto h = degree_preserving(cyc, 300, st);
      seen.insert(h.ties);
      if (transitivity(h) == 1) triangles_pair += 1;
    }
    s.require("swaps from a six-cycle reach all 70 two-regular graphs on six nodes", seen.size() == 70);
    s.check("and draw two triangles with probability 10/70 (6000 draws)", triangles_pair / D - 1.0 / 7, 0.025);
  }

  // ---- CUG tests ----
  {
    // A ring lattice: each of 30 nodes tied to its two neighbours on each side (transitivity 1/2).
    std::vector<Tie> ring;
    for (std::size_t a = 0; a < 30; a++) ring.push_back({a, (a + 1) % 30}), ring.push_back({a, (a + 2) % 30});
    const auto lat = network(30, ring, false);
    s.check("a ring lattice of degree 4: transitivity 1/2", transitivity(lat) - 0.5, 1e-15);
    CugOptions o;
    o.draws = 199;
    const auto c1 = cug_test(lat, [](const Network& g) { return transitivity(g); }, o);
    s.require("CUG (ties): the lattice's transitivity above every draw, p = 1/200", c1.p_greater == 1.0 / 200 && c1.observed > *std::max_element(c1.values.begin(), c1.values.end()));
    s.check("CUG (ties): the mean transitivity of G(30, 60) near its density 60/435", c1.mean - 60.0 / 435, 0.03);
    o.conditioning = Conditioning::degrees;
    const auto c2 = cug_test(lat, [](const Network& g) { return transitivity(g); }, o);
    s.require("CUG (degrees): the lattice's transitivity above every draw, 600 swaps attempted per draw", c2.p_greater == 1.0 / 200 && c2.swaps == 600);
    const auto c3 = cug_test(lat, [](const Network& g) { return transitivity(g); }, o);
    s.require("CUG: reproducible by seed", c3.values == c2.values);
    // ρ of random digraphs: 0 on average.
    calx::random::Stream st(3, 0x6e657477, 105, 0);
    const auto dg = uniform_network(40, 300, true, st);
    CugOptions od;
    od.draws = 400;
    const auto c4 = cug_test(dg, [](const Network& g) { return reciprocity(g).garlaschelli_loffredo; }, od);
    s.check("CUG (ties): Garlaschelli and Loffredo's ρ of random digraphs averages 0", c4.mean, 0.01);
    s.require("CUG: p-values in (0, 1], summing to at least 1", c4.p_greater > 0 && c4.p_less > 0 && c4.p_greater + c4.p_less >= 1);
  }

  return s.done();
}
