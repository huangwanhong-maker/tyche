// Checks of tyche/multivariate's dissimilarities, principal coordinates, PERMANOVA, PERMDISP and the
// Mantel test against hand-worked values, identities and enumeration.
//
// The dissimilarities are worked by hand on two samples; symmetry, positivity and the triangle
// inequality are held where they hold (Jaccard, Hellinger, chord, Euclidean, on every triple of 25
// samples), and Bray–Curtis is shown to break it. Principal coordinates of Euclidean distances
// reproduce the distances and the configuration's centred Gram matrix (so the configuration up to
// rotation); Bray–Curtis gives negative eigenvalues, which Lingoes's and Cailliez's corrections
// remove, Cailliez's constant being the least that does. PERMANOVA's F is held to the one-way ANOVA F
// for Euclidean distances in one and in three dimensions, its exact p-value to an independent
// enumeration of every relabelling, and the random p-value to the exact one; PERMDISP's distances to
// the centroids are Euclidean distances to group means; the Mantel r is Pearson's on the pairs, and
// its exact p-value is held to an enumeration of every relabelling.

#include <tyche/multivariate/dissimilarity.hpp>
#include <tyche/multivariate/mantel.hpp>
#include <tyche/multivariate/pcoa.hpp>
#include <tyche/multivariate/permanova.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace tyche::multivariate;
using calx::linalg::Matrix;
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

Matrix euclidean(const Rectangular& x) { return dissimilarities(x, Dissimilarity::euclidean); }

double min_eigenvalue(const Matrix& d) {
  const auto p = pcoa(d);
  return p.eigenvalues.back();
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261112, 7, 0, 0);

  // ---- dissimilarities by hand ----
  {
    const std::vector<double> x{1, 0, 3, 2}, y{0, 4, 1, 2};
    s.check("Bray–Curtis: (1 + 4 + 2 + 0)/(6 + 7) = 7/13", dissimilarity(x, y, Dissimilarity::bray_curtis) - 7.0 / 13, 1e-15);
    s.check("Jaccard: shared 2 of 4 present, 1 − 2/4", dissimilarity(x, y, Dissimilarity::jaccard) - 0.5, 1e-15);
    s.check("Sørensen: 1 − 2·2/(2·2 + 2) = 1/3", dissimilarity(x, y, Dissimilarity::sorensen) - 1.0 / 3, 1e-15);
    double h = 0, c = 0, nx = std::sqrt(14.0), ny = std::sqrt(21.0);
    for (int i = 0; i < 4; i++) {
      h += std::pow(std::sqrt(x[i] / 6) - std::sqrt(y[i] / 7), 2);
      c += std::pow(x[i] / nx - y[i] / ny, 2);
    }
    s.check("Hellinger: √Σ(√(x/6) − √(y/7))²", dissimilarity(x, y, Dissimilarity::hellinger) - std::sqrt(h), 1e-15);
    s.check("chord: √(2(1 − cos θ)), cos θ = 7/√(14·21)", dissimilarity(x, y, Dissimilarity::chord) - std::sqrt(2 * (1 - 7 / std::sqrt(14.0 * 21))), 1e-15);
    s.check("chord: the Euclidean distance of the normalised samples", dissimilarity(x, y, Dissimilarity::chord) - std::sqrt(c), 1e-15);
    s.check("Euclidean: √(1 + 16 + 4)", dissimilarity(x, y, Dissimilarity::euclidean) - std::sqrt(21.0), 1e-15);
    const std::vector<double> empty{0, 0, 0, 0};
    s.require("empty samples: Bray–Curtis and Jaccard 0 between two, 1 against one; Hellinger refused",
              dissimilarity(empty, empty, Dissimilarity::bray_curtis) == 0 && dissimilarity(empty, x, Dissimilarity::bray_curtis) == 1 && dissimilarity(empty, empty, Dissimilarity::jaccard) == 0 &&
                  throws([&] { dissimilarity(empty, x, Dissimilarity::hellinger); }));
    // Bray–Curtis breaks the triangle inequality: (1, 0), (1, 1), (0, 1).
    const std::vector<double> a{1, 0}, b{1, 1}, cc{0, 1};
    s.require("Bray–Curtis is semimetric: d(a, c) = 1 > d(a, b) + d(b, c) = 2/3 for (1, 0), (1, 1), (0, 1)",
              dissimilarity(a, cc, Dissimilarity::bray_curtis) > dissimilarity(a, b, Dissimilarity::bray_curtis) + dissimilarity(b, cc, Dissimilarity::bray_curtis));
  }

  // ---- axioms on random samples ----
  {
    Rectangular t(25, 9), presence(25, 9);
    for (std::size_t i = 0; i < t.rows; i++) {
      double row = 0;
      for (std::size_t j = 0; j < t.cols; j++) {
        t(i, j) = rng.uniform() < 0.35 ? 0 : std::floor(20 * rng.uniform() * rng.uniform());
        row += t(i, j);
      }
      if (row == 0) t(i, 0) = 1;
      for (std::size_t j = 0; j < t.cols; j++) presence(i, j) = t(i, j) > 0;
    }
    bool symmetric = true;
    double triangle = 0;
    for (auto kind : {Dissimilarity::bray_curtis, Dissimilarity::jaccard, Dissimilarity::sorensen, Dissimilarity::hellinger, Dissimilarity::chord, Dissimilarity::euclidean}) {
      const auto d = dissimilarities(t, kind);
      for (std::size_t i = 0; i < d.n; i++)
        for (std::size_t j = 0; j < d.n; j++) symmetric = symmetric && d(i, j) == d(j, i) && d(i, j) >= 0 && (i != j || d(i, j) == 0);
      if (kind == Dissimilarity::bray_curtis || kind == Dissimilarity::sorensen) continue;
      for (std::size_t i = 0; i < d.n; i++)
        for (std::size_t j = 0; j < d.n; j++)
          for (std::size_t k = 0; k < d.n; k++) triangle = std::max(triangle, d(i, k) - d(i, j) - d(j, k));
    }
    s.require("every dissimilarity: symmetric, non-negative, zero on the diagonal (25 samples)", symmetric);
    s.require("Jaccard, Hellinger, chord, Euclidean: the triangle inequality on every triple", triangle <= 1e-14);
    const auto bc = dissimilarities(presence, Dissimilarity::bray_curtis), so = dissimilarities(t, Dissimilarity::sorensen);
    double gap = 0, hmax = 0;
    for (std::size_t k = 0; k < bc.a.size(); k++) gap = std::max(gap, std::fabs(bc.a[k] - so.a[k]));
    for (double v : dissimilarities(t, Dissimilarity::hellinger).a) hmax = std::max(hmax, v);
    s.check("Sørensen is Bray–Curtis on presence and absence", gap, 1e-15);
    s.require("Hellinger distances at most √2", hmax <= std::sqrt(2.0) + 1e-15);

    // ---- principal coordinates ----
    {
      Rectangular x(12, 3);
      for (double& v : x.a) v = rng.normal();
      const auto d = euclidean(x);
      const auto p = pcoa(d);
      double dist = 0, gram = 0;
      std::vector<double> mean(3, 0.0);
      for (std::size_t i = 0; i < 12; i++)
        for (std::size_t k = 0; k < 3; k++) mean[k] += x(i, k) / 12;
      for (std::size_t i = 0; i < 12; i++)
        for (std::size_t j = 0; j < 12; j++) {
          double dd = 0, gx = 0, gp = 0;
          for (std::size_t k = 0; k < p.coordinates.cols; k++) dd += std::pow(p.coordinates(i, k) - p.coordinates(j, k), 2), gp += p.coordinates(i, k) * p.coordinates(j, k);
          for (std::size_t k = 0; k < 3; k++) gx += (x(i, k) - mean[k]) * (x(j, k) - mean[k]);
          dist = std::max(dist, std::fabs(std::sqrt(dd) - d(i, j)));
          gram = std::max(gram, std::fabs(gx - gp));
        }
      s.require("PCoA of Euclidean distances in 3 dimensions: three positive axes, no negative eigenvalue", p.coordinates.cols == 3 && p.negative == 0);
      s.check("PCoA: the coordinates reproduce the distances", dist, 1e-12);
      s.check("PCoA: their Gram matrix is the configuration's centred Gram matrix (the configuration up to rotation)", gram, 1e-12);
      double sum = 0;
      for (double v : p.explained) sum += v;
      s.check("PCoA: the shares explained sum to one", sum - 1, 1e-14);
      s.require("PCoA: eigenvalues descending", std::is_sorted(p.eigenvalues.rbegin(), p.eigenvalues.rend()));
    }
    {
      const auto d = dissimilarities(t, Dissimilarity::bray_curtis);
      const auto p = pcoa(d);
      s.require("PCoA of Bray–Curtis: negative eigenvalues, reported", p.negative < 0);
      const auto l = pcoa(d, Correction::lingoes), c = pcoa(d, Correction::cailliez);
      const double top_l = l.eigenvalues.front(), top_c = c.eigenvalues.front();
      s.require("Lingoes's correction: no negative eigenvalue remains", l.negative == 0 && l.eigenvalues.back() >= -1e-10 * top_l && l.constant > 0);
      s.check("Lingoes's constant is −λ_min", l.constant + p.eigenvalues.back(), 1e-14);
      s.require("Cailliez's correction: no negative eigenvalue remains", c.negative == 0 && c.eigenvalues.back() >= -1e-10 * top_c && c.constant > 0);
      Matrix less(d.n);
      for (std::size_t i = 0; i < d.n; i++)
        for (std::size_t j = 0; j < d.n; j++) less(i, j) = i == j ? 0 : d(i, j) + 0.98 * c.constant;
      s.require("Cailliez's constant is the least: d + 0.98c still has a negative eigenvalue", min_eigenvalue(less) < 0);
      double rep = 0;
      for (std::size_t i = 0; i < d.n; i++)
        for (std::size_t j = 0; j < d.n; j++) {
          double dd = 0;
          for (std::size_t k = 0; k < c.coordinates.cols; k++) dd += std::pow(c.coordinates(i, k) - c.coordinates(j, k), 2);
          rep = std::max(rep, std::fabs(std::sqrt(dd) - c.distances(i, j)));
        }
      s.check("Cailliez: the coordinates reproduce the corrected distances d + c", rep, 1e-9);
    }
  }

  // ---- PERMANOVA ----
  {
    // one dimension: F is the one-way ANOVA F
    const std::vector<double> v{1.0, 2.5, 1.7, 3.1, 4.0, 3.6, 5.2, 2.2, 6.1, 5.5, 4.8};
    const std::vector<std::size_t> g{0, 0, 0, 1, 1, 1, 2, 0, 2, 2, 1};
    Rectangular x(v.size(), 1);
    for (std::size_t i = 0; i < v.size(); i++) x(i, 0) = v[i];
    const auto r = permanova(euclidean(x), g, 199);
    std::vector<double> sum(3, 0.0), cnt(3, 0.0);
    double all = 0;
    for (std::size_t i = 0; i < v.size(); i++) sum[g[i]] += v[i], cnt[g[i]]++, all += v[i];
    double ssa = 0, ssw = 0;
    for (int k = 0; k < 3; k++) ssa += cnt[k] * std::pow(sum[k] / cnt[k] - all / 11, 2);
    for (std::size_t i = 0; i < v.size(); i++) ssw += std::pow(v[i] - sum[g[i]] / cnt[g[i]], 2);
    s.check("PERMANOVA, Euclidean in one dimension: F is the one-way ANOVA F", r.F - (ssa / 2) / (ssw / 8), 1e-12);
    s.check("PERMANOVA: SS_W is the within-group sum of squares", r.ss_within - ssw, 1e-12);
    s.check("PERMANOVA: R² = SS_A/SS_T", r.r_squared - ssa / (ssa + ssw), 1e-14);
    // three dimensions: the trace F
    Rectangular y(15, 3);
    std::vector<std::size_t> h(15);
    for (std::size_t i = 0; i < 15; i++) {
      h[i] = i % 3;
      for (std::size_t k = 0; k < 3; k++) y(i, k) = rng.normal() + 0.8 * static_cast<double>(h[i] == k);
    }
    const auto r3 = permanova(euclidean(y), h, 99);
    double sa = 0, sw = 0;
    for (std::size_t k = 0; k < 3; k++) {
      std::vector<double> gm(3, 0.0);
      double m = 0;
      for (std::size_t i = 0; i < 15; i++) gm[h[i]] += y(i, k) / 5, m += y(i, k) / 15;
      for (int q = 0; q < 3; q++) sa += 5 * std::pow(gm[q] - m, 2);
      for (std::size_t i = 0; i < 15; i++) sw += std::pow(y(i, k) - gm[h[i]], 2);
    }
    s.check("PERMANOVA, Euclidean in three dimensions: F is the trace F", r3.F - (sa / 2) / (sw / 12), 1e-12);

    // exact p-values by an independent enumeration: 3 + 3 samples (20 relabellings), 2 + 2 + 2 (90)
    const auto d = dissimilarities([&] {
      Rectangular t(6, 4);
      for (double& u : t.a) u = std::floor(10 * rng.uniform());
      for (std::size_t i = 0; i < 6; i++) t(i, i % 4) += 3;
      return t;
    }(), Dissimilarity::bray_curtis);
    const std::vector<std::size_t> two{0, 1, 0, 1, 0, 1}, three{2, 0, 1, 2, 0, 1};
    for (const auto* lab : {&two, &three}) {
      const auto e = permanova_exact(d, *lab);
      // enumerate every assignment of the labels' multiset by brute force over all 6! orders, counting distinct ones
      std::vector<std::size_t> idx(6);
      std::iota(idx.begin(), idx.end(), std::size_t{0});
      std::vector<std::vector<std::size_t>> seen;
      std::size_t ge = 0;
      do {
        std::vector<std::size_t> l(6);
        for (std::size_t i = 0; i < 6; i++) l[i] = (*lab)[idx[i]];
        if (std::find(seen.begin(), seen.end(), l) != seen.end()) continue;
        seen.push_back(l);
        ge += permanova(d, l, 0).F >= e.F * (1 - 1e-10);
      } while (std::next_permutation(idx.begin(), idx.end()));
      const std::string tag = lab == &two ? " (3 + 3, 20 relabellings)" : " (2 + 2 + 2, 90 relabellings)";
      s.require("PERMANOVA exact: every distinct relabelling enumerated" + tag, e.permutations == seen.size());
      s.check("PERMANOVA exact: the p-value counted independently" + tag, e.p_value - static_cast<double>(ge) / static_cast<double>(seen.size()), 1e-15);
      const auto mc = permanova(d, *lab, 20000, 11);
      s.check("PERMANOVA: 20 000 random relabellings approach the exact p-value" + tag, mc.p_value - e.p_value, 0.012);
    }
    const auto a1 = permanova(d, three, 499, 5), a2 = permanova(d, three, 499, 5);
    s.require("PERMANOVA: repeatable by seed; p ≥ 1/(B + 1)", a1.p_value == a2.p_value && a1.p_value >= 1.0 / 500);
    s.require("refused: a single group; groups as many as samples", throws([&] { permanova(d, std::vector<std::size_t>(6, 0)); }) && throws([&] { permanova(d, std::vector<std::size_t>{0, 1, 2, 3, 4, 5}); }));

    // separated groups are found
    Rectangular w(30, 2);
    std::vector<std::size_t> wg(30);
    for (std::size_t i = 0; i < 30; i++) {
      wg[i] = i / 10;
      w(i, 0) = rng.normal() + 2.0 * static_cast<double>(wg[i]);
      w(i, 1) = rng.normal();
    }
    s.require("PERMANOVA: three groups two standard deviations apart, p = 1/(B + 1)", permanova(euclidean(w), wg, 999).p_value == 1.0 / 1000);
  }

  // ---- PERMDISP ----
  {
    Rectangular x(24, 2);
    std::vector<std::size_t> g(24);
    for (std::size_t i = 0; i < 24; i++) {
      g[i] = i < 12 ? 4 : 9;
      const double spread = g[i] == 4 ? 0.5 : 3.0;
      x(i, 0) = spread * rng.normal();
      x(i, 1) = spread * rng.normal() + 1;
    }
    const auto d = euclidean(x);
    const auto r = dispersion(d, g, 999);
    double worst = 0;
    for (std::size_t i = 0; i < 24; i++) {
      double m0 = 0, m1 = 0;
      for (std::size_t j = 0; j < 24; j++)
        if (g[j] == g[i]) m0 += x(j, 0) / 12, m1 += x(j, 1) / 12;
      worst = std::max(worst, std::fabs(r.distances[i] - std::hypot(x(i, 0) - m0, x(i, 1) - m1)));
    }
    s.check("PERMDISP: for Euclidean distances, z_i is the distance to the group's mean", worst, 1e-12);
    s.require("PERMDISP: groups labelled 4 and 9 kept in order, means of z", r.labels.size() == 2 && r.labels[0] == 4 && r.labels[1] == 9 && r.group_mean[0] < r.group_mean[1]);
    s.require("PERMDISP: spreads 0.5 and 3 told apart, p = 1/(B + 1)", r.p_value == 1.0 / 1000);
    // PCoA in full gives the same distances for Bray–Curtis, imaginary axes included
    Rectangular t(10, 5);
    for (double& u : t.a) u = std::floor(8 * rng.uniform());
    for (std::size_t i = 0; i < 10; i++) t(i, i % 5) += 2;
    const auto bc = dissimilarities(t, Dissimilarity::bray_curtis);
    const std::vector<std::size_t> grp{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    const auto z = centroid_distances(bc, grp);
    const auto G = gower_centred(bc);
    const auto e = calx::linalg::symmetric_eigen(G);
    double gap = 0;
    for (std::size_t i = 0; i < 10; i++) {
      double z2 = 0;
      for (std::size_t k = 0; k < 10; k++) {
        double c = 0;
        for (std::size_t j = 0; j < 10; j++)
          if (grp[j] == grp[i]) c += e.vectors(j, k) / 5;
        z2 += e.values[k] * std::pow(e.vectors(i, k) - c, 2);  // real axes add, imaginary subtract
      }
      gap = std::max(gap, std::fabs(z[i] - std::sqrt(std::fabs(z2))));
    }
    s.check("PERMDISP for Bray–Curtis: the distances to centroids over every principal axis, imaginary ones subtracted", gap, 1e-12);
  }

  // ---- the Mantel test ----
  {
    Rectangular x(7, 2);
    for (double& v : x.a) v = rng.normal();
    const auto A = euclidean(x);
    Matrix B(7);
    for (std::size_t i = 0; i < 7; i++)
      for (std::size_t j = 0; j < 7; j++) B(i, j) = i == j ? 0 : A(i, j) + 0.6 * std::fabs(std::sin(static_cast<double>(3 * i + 5 * j + 1)) + std::sin(static_cast<double>(3 * j + 5 * i + 1)));
    // Pearson's r on the pairs, by hand
    std::vector<double> a, b;
    for (std::size_t i = 0; i < 7; i++)
      for (std::size_t j = i + 1; j < 7; j++) a.push_back(A(i, j)), b.push_back(B(i, j));
    const double ma = std::accumulate(a.begin(), a.end(), 0.0) / 21, mb = std::accumulate(b.begin(), b.end(), 0.0) / 21;
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t k = 0; k < 21; k++) sab += (a[k] - ma) * (b[k] - mb), saa += (a[k] - ma) * (a[k] - ma), sbb += (b[k] - mb) * (b[k] - mb);
    const auto e = mantel_exact(A, B);
    s.check("Mantel: r is Pearson's correlation of the 21 pairs", e.r - sab / std::sqrt(saa * sbb), 1e-14);
    // the exact p by enumerating 7! relabellings independently
    std::vector<std::size_t> p(7);
    std::iota(p.begin(), p.end(), std::size_t{0});
    std::size_t ge = 0, total = 0;
    do {
      double num = 0;
      std::size_t k = 0;
      for (std::size_t i = 0; i < 7; i++)
        for (std::size_t j = i + 1; j < 7; j++) num += (a[k++] - ma) * (B(p[i], p[j]) - mb);
      ge += num / std::sqrt(saa * sbb) >= e.r - 1e-12;
      total++;
    } while (std::next_permutation(p.begin(), p.end()));
    s.require("Mantel exact: 5040 relabellings", e.permutations == 5040 && total == 5040);
    s.check("Mantel exact: the p-value counted independently", e.p_value - static_cast<double>(ge) / 5040, 1e-15);
    s.check("Mantel: 20 000 random relabellings approach the exact p-value", mantel(A, B, 20000, 3).p_value - e.p_value, 0.01);
    const auto self = mantel(A, A, 999);
    s.require("Mantel: a matrix against itself, r = 1 and p small", std::fabs(self.r - 1) < 1e-14 && self.p_value < 0.01);
  }

  return s.done();
}
