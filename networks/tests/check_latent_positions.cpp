// Checks of tyche/networks/latent: the adjacency spectral embedding of a random dot product graph recovers its positions after
// Procrustes alignment, more nearly as the network grows (the error about halving when the nodes are quadrupled); the
// probabilities are identified under the declared group, and so are distances under O(2); the profile likelihood chooses the
// true dimension where the leading eigenvalues are comparable and stops early where one dominates; a disassortative graph has
// the signature (1, 1), its probabilities identified and its distances not, which the test of identification says; Procrustes
// recovers a known rotation exactly; a nested incidence's rows are ordered by their diversity.

#include <tyche/networks/latent.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>

using namespace tyche::networks;

namespace {

// Two clusters of positions about (0.6, 0.2) and (0.2, 0.6), and the adjacency they make.
struct Graph {
  Positions x;
  calx::linalg::Matrix A;
};
Graph rdpg(std::size_t n, std::uint64_t seed, double hi = 0.6, double lo = 0.2) {
  calx::random::Stream r(seed, 3, 0, 0);
  Graph g;
  g.A = calx::linalg::Matrix(n);
  for (std::size_t i = 0; i < n; i++) g.x.push_back(i % 2 ? std::vector<double>{hi + r.uniform(-0.05, 0.05), lo + r.uniform(-0.05, 0.05)}
                                                          : std::vector<double>{lo + r.uniform(-0.05, 0.05), hi + r.uniform(-0.05, 0.05)});
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = i + 1; j < n; j++)
      if (r.uniform() < g.x[i][0] * g.x[j][0] + g.x[i][1] * g.x[j][1]) g.A(i, j) = g.A(j, i) = 1;
  return g;
}

double rms_after_alignment(const Positions& est, const Positions& truth) {
  const auto W = procrustes(est, truth);
  const auto a = act(est, W);
  double s = 0;
  for (std::size_t i = 0; i < a.size(); i++) s += std::pow(a[i][0] - truth[i][0], 2) + std::pow(a[i][1] - truth[i][1], 2);
  return std::sqrt(s / static_cast<double>(a.size()));
}

double distance01(const Positions& x) { return std::hypot(x[0][0] - x[1][0], x[0][1] - x[1][1]); }
double probability01(const Embedding& e, const Positions& x) {
  double s = 0;
  for (std::size_t k = 0; k < e.p + e.q; k++) s += (k < e.p ? 1 : -1) * x[0][k] * x[1][k];
  return s;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream r(5, 9, 0, 0);
  {
    const auto small = rdpg(300, 1), big = rdpg(1200, 2);
    const auto es = spectral_embedding(small.A, 2, false), eb = spectral_embedding(big.A, 2, false);
    const double a = rms_after_alignment(es.x, small.x), b = rms_after_alignment(eb.x, big.x);
    s.check("the spectral embedding of 1,200 nodes recovers their positions after alignment (root-mean-square error)", b, 0.08);
    s.require("... about twice as nearly as 300 nodes do (the error falls as 1/√n)", a / b > 1.6, std::to_string(a) + " against " + std::to_string(b));
    s.require("a random dot product graph is identified under O(2)", eb.group() == "O(2)");
    const auto P = identified(eb, [&](const Positions& x) { return probability01(eb, x); }, r);
    const auto D = identified(eb, distance01, r);
    s.require("... its probabilities are identified", P.identified);
    s.require("... and so are distances between positions", D.identified);
    // Its leading eigenvalues are about 384 and 96: one dominates, and the common-variance split stops at one (the caveat).
    s.require("where one leading eigenvalue dominates, the profile likelihood stops at one dimension (its caveat, said)", profile_dimension(scree(big.A), 20) == 1);
    // Clusters about (0.7, 0.1) and (0.1, 0.7): leading eigenvalues about 384 and 216, comparable, and the choice is two.
    const auto even = rdpg(1200, 3, 0.7, 0.1);
    s.require("where the leading eigenvalues are comparable it chooses the true two dimensions", profile_dimension(scree(even.A), 20) == 2);
  }
  {
    // Two blocks joined more across than within: B = [[0.1, 0.5], [0.5, 0.1]], eigenvalues 0.6 and −0.4.
    const std::size_t n = 600;
    calx::random::Stream g(8, 4, 0, 0);
    calx::linalg::Matrix A(n);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = i + 1; j < n; j++)
        if (g.uniform() < ((i % 2) == (j % 2) ? 0.1 : 0.5)) A(i, j) = A(j, i) = 1;
    const auto e = spectral_embedding(A, 2, true);
    s.require("a disassortative two-block graph has the signature (1, 1), identified under O(1,1)", e.p == 1 && e.q == 1 && e.group() == "O(1,1)");
    const auto P = identified(e, [&](const Positions& x) { return probability01(e, x); }, r);
    const auto D = identified(e, distance01, r);
    s.require("... its probabilities are identified", P.identified);
    s.require("... its distances are not: a hyperbolic rotation moves them, and the test says so", !D.identified && D.largest_change > 1e-3, std::to_string(D.largest_change));
    double within = 0, across = 0;
    for (std::size_t i = 0; i < 40; i++) within += e.probability(i, i + 2) / 40, across += e.probability(i, i + 1) / 40;
    s.check("... and its estimated probabilities are the blocks' (0.1 within, 0.5 across)", std::fabs(within - 0.1) + std::fabs(across - 0.5), 0.05);
  }
  {
    Positions X;
    calx::random::Stream g(12, 5, 0, 0);
    for (int i = 0; i < 30; i++) X.push_back({g.normal(), g.normal(), g.normal()});
    const auto R = random_orthogonal(3, g);
    const auto W = procrustes(X, act(X, R));
    double worst = 0;
    for (std::size_t i = 0; i < 3; i++)
      for (std::size_t j = 0; j < 3; j++) worst = std::max(worst, std::fabs(W(i, j) - R(i, j)));
    s.check("Procrustes recovers a known rotation exactly", worst, 1e-12);
  }
  {
    // A nested incidence: row i holds the first 2(i + 1) of 16 columns.
    calx::linalg::Rectangular M(8, 16);
    for (std::size_t i = 0; i < 8; i++)
      for (std::size_t j = 0; j < 2 * (i + 1); j++) M(i, j) = 1;
    const auto v = bipartite_ordering(M);
    bool rising = true;
    for (std::size_t i = 1; i < v.size(); i++) rising = rising && v[i] > v[i - 1];
    s.require("a nested incidence's rows are ordered by their diversity (the complexity index as a normalised cut)", rising);
  }
  return s.done();
}
