// Latent positions of a network, with the group under which they are identified (decision 0033).
//
// A random dot product graph gives each node a position x_i ∈ ℝ^d and joins i and j with probability x_iᵀ x_j; the generalised
// one, with signature (p, q), with probability x_iᵀ I_{p,q} x_j, I_{p,q} = diag(1 … 1, −1 … −1), which also holds networks that
// join unlike nodes more than like ones (Rubin-Delanchy, Cape, Tang and Priebe 2022, "A statistical interpretation of spectral
// embedding: the generalised random dot product graph", J. R. Stat. Soc. B 84, 1446–1473). The adjacency spectral embedding
// estimates the positions from the d eigenvalues of the adjacency matrix largest in magnitude, X̂ = U_d |Λ_d|^{1/2}, consistently
// as the network grows (Sussman, Tang, Fishkind and Priebe 2012, J. Am. Stat. Assoc. 107, 1119–1128; Athreya et al. 2018,
// "Statistical inference on random dot product graphs: a survey", J. Mach. Learn. Res. 18(226)); the signature is read from the
// signs of the kept eigenvalues.
//
// The positions are identified only up to a group: the orthogonal group O(d) for the random dot product graph, the indefinite
// orthogonal group O(p, q) for the generalised one, since x ↦ x W with W I_{p,q} Wᵀ = I_{p,q} leaves every probability as it was.
// Only what the group leaves unchanged may be reported: the probabilities x_iᵀ I_{p,q} x_j always; distances between positions
// under O(d), but not under O(p, q), where a hyperbolic rotation moves them. The group is declared with every embedding, and a
// statistic is tested against it by applying random elements of the group before it is reported.
//
// Embeddings of two networks of the same nodes are compared after aligning one to the other by the orthogonal Procrustes rotation,
// W = argmin ‖X W − Y‖ over O(d), the polar factor of XᵀY (Schönemann 1966, "A generalized solution of the orthogonal Procrustes
// problem", Psychometrika 31, 1–10).
//
// The dimension may be chosen by the profile likelihood of the ordered magnitudes of the eigenvalues (Zhu and Ghodsi 2006,
// "Automatic dimensionality selection from the scree plot via the use of profile likelihood", Comput. Stat. Data Anal. 51,
// 918–930): two Gaussian groups with a common variance, the split of greatest likelihood. *Caveat.* Where one leading eigenvalue
// dominates the others, the common variance it inflates makes the split stop early (at one dimension for leading eigenvalues of
// 384 and 96 above a bulk near 30, in the checks); the scree should be looked at.
//
// A bipartite incidence (rows and columns, as in which places hold which capabilities) orders its rows by the second eigenvector
// of the row-to-row walk D_r⁻¹ M D_c⁻¹ Mᵀ; Mealy, Farmer and Teytelboym (2019, "Interpreting economic complexity", Sci. Adv. 5,
// eaau1705) show the economic complexity index of Hidalgo and Hausmann (2009) to be this ordering, the relaxation of a normalised
// cut, and nothing more: the eigenvector of the spectral embedding of the rows' similarity.
#pragma once

#include <calx/linalg/dense.hpp>
#include <calx/linalg/eigen.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::networks {

using Positions = std::vector<std::vector<double>>;

struct Embedding {
  std::size_t p = 0, q = 0;          // the signature: positive and negative eigenvalues kept
  Positions x;                       // a row of d = p + q coordinates per node, the positive coordinates first
  std::vector<double> eigenvalues;   // the kept eigenvalues, in the order of the coordinates
  std::string group() const { return q == 0 ? "O(" + std::to_string(p) + ")" : "O(" + std::to_string(p) + "," + std::to_string(q) + ")"; }
  // x_iᵀ I_{p,q} x_j, identified whatever the group.
  double probability(std::size_t i, std::size_t j) const {
    double s = 0;
    for (std::size_t k = 0; k < p + q; k++) s += (k < p ? 1 : -1) * x[i][k] * x[j][k];
    return s;
  }
};

// The magnitudes of the eigenvalues of a symmetric matrix, largest first (for the scree).
inline std::vector<double> scree(const calx::linalg::Matrix& A) {
  const auto E = calx::linalg::symmetric_eigen(A);
  std::vector<double> m;
  for (double v : E.values) m.push_back(std::fabs(v));
  std::sort(m.begin(), m.end(), std::greater<>());
  return m;
}

// Zhu and Ghodsi's choice: the q maximising the profile likelihood of two Gaussian groups (the first q, the rest), common variance.
inline std::size_t profile_dimension(const std::vector<double>& magnitudes, std::size_t most = 0) {
  const std::size_t m = most ? std::min(most + 1, magnitudes.size()) : magnitudes.size();
  if (m < 3) throw std::invalid_argument("tyche::networks::profile_dimension: at least three eigenvalues");
  std::size_t best = 1;
  double top = -INFINITY;
  for (std::size_t q = 1; q < m; q++) {
    double m1 = 0, m2 = 0;
    for (std::size_t i = 0; i < q; i++) m1 += magnitudes[i] / q;
    for (std::size_t i = q; i < m; i++) m2 += magnitudes[i] / (m - q);
    double ss = 0;
    for (std::size_t i = 0; i < m; i++) ss += std::pow(magnitudes[i] - (i < q ? m1 : m2), 2);
    const double var = std::max(ss / static_cast<double>(m), 1e-300);
    const double ll = -0.5 * static_cast<double>(m) * std::log(2 * 3.141592653589793 * var) - ss / (2 * var);
    if (ll > top) top = ll, best = q;
  }
  return best;
}

// The adjacency spectral embedding in d dimensions; with `generalised`, the d eigenvalues largest in magnitude, signs and all;
// without, the d largest (a random dot product graph).
inline Embedding spectral_embedding(const calx::linalg::Matrix& A, std::size_t d, bool generalised = true) {
  const std::size_t n = A.n;
  if (d == 0 || d > n) throw std::invalid_argument("tyche::networks::spectral_embedding: a dimension between 1 and the number of nodes");
  const auto E = calx::linalg::symmetric_eigen(A);
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  if (generalised) std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return std::fabs(E.values[a]) > std::fabs(E.values[b]); });
  else std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return E.values[a] > E.values[b]; });
  std::vector<std::size_t> kept(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(d));
  std::stable_sort(kept.begin(), kept.end(), [&](std::size_t a, std::size_t b) { return (E.values[a] > 0) > (E.values[b] > 0); });
  Embedding out;
  for (std::size_t k : kept) (E.values[k] > 0 || !generalised ? out.p : out.q)++;
  out.x.assign(n, std::vector<double>(d));
  for (std::size_t c = 0; c < d; c++) {
    const std::size_t k = kept[c];
    out.eigenvalues.push_back(E.values[k]);
    const double s = std::sqrt(std::fabs(E.values[k]));
    for (std::size_t i = 0; i < n; i++) out.x[i][c] = E.vectors(i, k) * s;
  }
  return out;
}

inline Positions act(const Positions& x, const calx::linalg::Matrix& W) {
  Positions y(x.size(), std::vector<double>(W.n, 0));
  for (std::size_t i = 0; i < x.size(); i++)
    for (std::size_t c = 0; c < W.n; c++)
      for (std::size_t k = 0; k < W.n; k++) y[i][c] += x[i][k] * W(k, c);
  return y;
}

// A random element of O(d): Gram and Schmidt on Gaussian columns.
inline calx::linalg::Matrix random_orthogonal(std::size_t d, calx::random::Stream& r) {
  calx::linalg::Matrix Q(d);
  for (std::size_t c = 0; c < d; c++) {
    std::vector<double> v(d);
    for (double& e : v) e = r.normal();
    for (std::size_t b = 0; b < c; b++) {
      double dot = 0;
      for (std::size_t i = 0; i < d; i++) dot += v[i] * Q(i, b);
      for (std::size_t i = 0; i < d; i++) v[i] -= dot * Q(i, b);
    }
    double norm = 0;
    for (double e : v) norm += e * e;
    norm = std::sqrt(norm);
    for (std::size_t i = 0; i < d; i++) Q(i, c) = v[i] / norm;
  }
  return Q;
}

// A random element of the group an embedding is identified under: orthogonal within each sign's block, and for a signature with
// both signs a hyperbolic rotation of rapidity ρ ~ N(0, 1) mixing the first positive and the first negative coordinate.
inline calx::linalg::Matrix random_element(std::size_t p, std::size_t q, calx::random::Stream& r) {
  const std::size_t d = p + q;
  calx::linalg::Matrix W(d);
  const auto P = random_orthogonal(p, r);
  for (std::size_t i = 0; i < p; i++)
    for (std::size_t j = 0; j < p; j++) W(i, j) = P(i, j);
  if (q > 0) {
    const auto Q = random_orthogonal(q, r);
    for (std::size_t i = 0; i < q; i++)
      for (std::size_t j = 0; j < q; j++) W(p + i, p + j) = Q(i, j);
    const double rho = r.normal(), c = std::cosh(rho), s = std::sinh(rho);
    calx::linalg::Matrix B = calx::linalg::Matrix::identity(d);
    B(0, 0) = c, B(0, p) = s, B(p, 0) = s, B(p, p) = c;
    W = W * B;
  }
  return W;
}

struct Identified {
  bool identified = false;
  double largest_change = 0;  // relative to the statistic's size
  std::string group;
};

// Whether a statistic of the positions is identified: unchanged (to `tolerance`, relatively) under `trials` random elements of the
// embedding's group.
inline Identified identified(const Embedding& e, const std::function<double(const Positions&)>& statistic, calx::random::Stream& r, int trials = 20, double tolerance = 1e-9) {
  Identified out;
  out.group = e.group();
  const double base = statistic(e.x), scale = std::max(1.0, std::fabs(base));
  for (int t = 0; t < trials; t++) {
    const double v = statistic(act(e.x, random_element(e.p, e.q, r)));
    out.largest_change = std::max(out.largest_change, std::fabs(v - base) / scale);
  }
  out.identified = out.largest_change <= tolerance;
  return out;
}

// The orthogonal W minimising ‖X W − Y‖: the polar factor of XᵀY, M (MᵀM)^{−1/2}.
inline calx::linalg::Matrix procrustes(const Positions& X, const Positions& Y) {
  if (X.empty() || X.size() != Y.size()) throw std::invalid_argument("tyche::networks::procrustes: two sets of positions of the same nodes");
  const std::size_t d = X[0].size();
  calx::linalg::Matrix M(d);
  for (std::size_t i = 0; i < X.size(); i++)
    for (std::size_t a = 0; a < d; a++)
      for (std::size_t b = 0; b < d; b++) M(a, b) += X[i][a] * Y[i][b];
  calx::linalg::Matrix MtM(d);
  for (std::size_t a = 0; a < d; a++)
    for (std::size_t b = 0; b < d; b++)
      for (std::size_t k = 0; k < d; k++) MtM(a, b) += M(k, a) * M(k, b);
  const auto E = calx::linalg::symmetric_eigen(MtM);
  calx::linalg::Matrix S(d);  // (MᵀM)^{−1/2}
  for (std::size_t a = 0; a < d; a++)
    for (std::size_t b = 0; b < d; b++)
      for (std::size_t k = 0; k < d; k++) {
        if (!(E.values[k] > 1e-300)) throw std::invalid_argument("tyche::networks::procrustes: XᵀY is singular, the alignment is not unique");
        S(a, b) += E.vectors(a, k) * E.vectors(b, k) / std::sqrt(E.values[k]);
      }
  return M * S;
}

// The rows of a bipartite incidence ordered by the second eigenvector of the row-to-row walk, signed to rise with the rows' degree.
inline std::vector<double> bipartite_ordering(const calx::linalg::Rectangular& M) {
  const std::size_t R = M.rows, C = M.cols;
  std::vector<double> dr(R, 0), dc(C, 0);
  for (std::size_t i = 0; i < R; i++)
    for (std::size_t j = 0; j < C; j++) dr[i] += M(i, j), dc[j] += M(i, j);
  for (double v : dr)
    if (!(v > 0)) throw std::invalid_argument("tyche::networks::bipartite_ordering: every row needs an entry");
  // The symmetric form D_r^{−1/2} M D_c^{−1} Mᵀ D_r^{−1/2}, similar to the walk.
  calx::linalg::Matrix S(R);
  for (std::size_t a = 0; a < R; a++)
    for (std::size_t b = 0; b < R; b++) {
      double s = 0;
      for (std::size_t j = 0; j < C; j++)
        if (dc[j] > 0) s += M(a, j) * M(b, j) / dc[j];
      S(a, b) = s / std::sqrt(dr[a] * dr[b]);
    }
  const auto E = calx::linalg::symmetric_eigen(S);  // ascending: the second largest is at R − 2
  std::vector<double> v(R);
  for (std::size_t i = 0; i < R; i++) v[i] = E.vectors(i, R - 2) / std::sqrt(dr[i]);
  double cov = 0, md = 0, mv = 0;
  for (std::size_t i = 0; i < R; i++) md += dr[i] / R, mv += v[i] / R;
  for (std::size_t i = 0; i < R; i++) cov += (dr[i] - md) * (v[i] - mv);
  if (cov < 0)
    for (double& e : v) e = -e;
  return v;
}

}  // namespace tyche::networks
