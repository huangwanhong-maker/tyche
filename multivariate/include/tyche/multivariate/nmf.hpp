// Non-negative matrix factorisation V ≈ W H, and the stability of a factorisation's rank.
//
// THE PROBLEM. V is m × n with entries ≥ 0 (in the analysis of mutations: m mutation types by n
// genomes); W (m × k) and H (k × n) are non-negative, k the rank. Two objectives (Lee and Seung 2001,
// "Algorithms for non-negative matrix factorization", NIPS 13, 556–562, eqs. 2–3):
//     frobenius:         ‖V − WH‖²_F = Σ (V_ij − (WH)_ij)²;
//     kullback_leibler:  D(V‖WH) = Σ [V_ij log(V_ij/(WH)_ij) − V_ij + (WH)_ij]   (0 log 0 = 0),
// the generalised divergence, the −log-likelihood of Poisson counts V_ij with means (WH)_ij up to a
// term in V alone.
//
// THE ALGORITHM. Lee and Seung's multiplicative updates, alternately for H and W:
//     frobenius (Theorem 1, eq. 4):  H ← H ⊙ (WᵀV) ⊘ (WᵀWH),       W ← W ⊙ (VHᵀ) ⊘ (WHHᵀ);
//     KL (Theorem 2, eq. 5):         H_aμ ← H_aμ Σ_i W_ia V_iμ/(WH)_iμ / Σ_k W_ka,
//                                    W_ia ← W_ia Σ_μ H_aμ V_iμ/(WH)_iμ / Σ_ν H_aν.
// Each objective is non-increasing under its updates (the theorems, by auxiliary functions); the
// iterates approach a stationary point, slowly (linearly at best), and the factorisation is unique at
// most up to a permutation and a positive scaling of the k components (more only for "separable" or
// otherwise sufficiently sparse factors). An entry whose denominator is 0 is left as it is (its
// numerator is 0 then too, or the entry already 0); (WH)_iμ is kept ≥ 10⁻³⁰⁰ in the KL ratios. The
// factors start from a calx stream (entries uniform on (0, 1) times √(mean V / k), the scale of
// scikit-learn's random start), with several restarts; the best final objective wins. It stops when
// the objective's relative decrease over check_every iterations is below tolerance (after
// min_iterations), or at max_iterations. With normalise, W's columns are scaled to sum to 1 and H's
// rows by the inverse (WH unchanged): signatures as distributions over the m types.
//
// THE STABILITY OF A RANK (Alexandrov, Nik-Zainal, Wedge, Campbell and Stratton 2013, "Deciphering
// signatures of mutational processes operative in human cancer", Cell Reports 3, 246–259, Experimental
// Procedures, steps 2–6; Islam et al. 2022, "Uncovering novel mutational signatures by de novo
// extraction with SigProfilerExtractor", Cell Genomics 2, 100179, STAR Methods, modules 2–4). For a
// rank k: R replicates, each a factorisation of V resampled (multinomially within each column keeping
// its total, Alexandrov et al.'s Monte Carlo bootstrap; or each entry Poisson with mean V_ij, Islam et
// al.'s) and optionally normalised (log2: each column times log₂(total)/total; hundred_x: a column
// whose total exceeds 100 m scaled to 100 m; Islam et al.'s GMM normalisation is not written). The R·k
// signatures (W's columns, each summing to 1) are clustered into k clusters with exactly one signature
// of each replicate in each: every replicate's signatures are matched to the centroids by the
// assignment of greatest total cosine similarity (calx's Hungarian method; SigProfilerExtractor's
// choice), the centroids are the means of their clusters, and the two steps repeat until the average
// silhouette changes by less than 10⁻¹² (ibid.); the centroids start from the replicate of least
// objective (SigProfilerExtractor starts from random centroids: a departure, for determinism). The
// stability of a cluster is the mean silhouette width of its members (Rousseeuw 1987, J. Comput. Appl.
// Math. 20, 53–65) under the cosine distance 1 − cos; the average and the least over the clusters are
// the rank's average and minimum stability (SigProfilerExtractor calls a WGS solution stable at
// ≥ 0.80 and ≥ 0.20). The consensus signatures W̄ are the centroids, renormalised; the exposures H̄
// are the non-negative least squares of each column of V on W̄; the reconstruction error is
// ‖V − W̄H̄‖²_F (Alexandrov et al., step 6), with ‖V − W̄H̄‖_F/‖V‖_F beside it. For k = 1 there is no
// other cluster; a member's silhouette is then 1 − a(i), a(i) its mean distance to the others, which
// is the silhouette with the nearest other cluster at the greatest cosine distance, 1.
//
// LIMITS. The multiplicative updates find local minima; restarts and the stability analysis are the
// guard. Ranks with unstable clusters (low silhouettes) mean that the data do not determine k
// signatures; the choice of the rank is the caller's, from the stabilities and errors by rank.

#pragma once

#include <tyche/regression/nnls.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/rectangular.hpp>
#include <calx/optimisation/assignment.hpp>
#include <calx/random/multinomial.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::multivariate {

using calx::linalg::Matrix;
using calx::linalg::Rectangular;

enum class NmfLoss { frobenius, kullback_leibler };

struct NmfOptions {
  NmfLoss loss = NmfLoss::kullback_leibler;
  std::size_t max_iterations = 100000;  // updates of both factors
  std::size_t min_iterations = 0;
  std::size_t check_every = 10;         // iterations between checks of the objective
  double tolerance = 1e-10;             // on the relative decrease of the objective between checks
  std::size_t restarts = 1;             // random starts; the best final objective wins
  std::uint64_t seed = 1;
  bool normalise = true;                // W's columns sum to 1, H's rows rescaled
};

struct Nmf {
  Rectangular W;                        // m × k
  Rectangular H;                        // k × n
  double objective = 0;                 // ‖V − WH‖²_F or D(V‖WH) at the end
  std::size_t iterations = 0;
  bool converged = false;
  std::size_t restart = 0;              // the restart that won
  std::vector<double> restart_objectives;
  std::vector<double> trace;            // the winning run's objective at each check (non-increasing)
};

namespace nmf_detail {

inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::multivariate::nmf: ") + what);
}

inline Rectangular product(const Rectangular& W, const Rectangular& H) { return W * H; }

inline void check_nonnegative(const Rectangular& V) {
  for (const double v : V.a) require(v >= 0 && std::isfinite(v), "V non-negative and finite");
}

}  // namespace nmf_detail

// The objective of a factorisation.
inline double nmf_objective(const Rectangular& V, const Rectangular& W, const Rectangular& H, NmfLoss loss) {
  nmf_detail::require(W.rows == V.rows && H.cols == V.cols && W.cols == H.rows, "shapes V (m × n), W (m × k), H (k × n)");
  const Rectangular WH = W * H;
  double s = 0;
  for (std::size_t q = 0; q < V.a.size(); q++) {
    const double v = V.a[q], x = WH.a[q];
    if (loss == NmfLoss::frobenius) s += (v - x) * (v - x);
    else s += (v > 0 ? v * std::log(v / std::max(x, 1e-300)) : 0.0) - v + x;
  }
  return s;
}

// Scale W's columns to sum to 1 and H's rows inversely.
inline void normalise_factors(Rectangular& W, Rectangular& H) {
  for (std::size_t a = 0; a < W.cols; a++) {
    double s = 0;
    for (std::size_t i = 0; i < W.rows; i++) s += W(i, a);
    if (!(s > 0)) continue;
    for (std::size_t i = 0; i < W.rows; i++) W(i, a) /= s;
    for (std::size_t j = 0; j < H.cols; j++) H(a, j) *= s;
  }
}

// The multiplicative updates from given factors.
inline Nmf nmf_from(const Rectangular& V, Rectangular W, Rectangular H, const NmfOptions& o = {}) {
  using namespace nmf_detail;
  check_nonnegative(V);
  const std::size_t m = V.rows, n = V.cols, k = W.cols;
  require(W.rows == m && H.cols == n && H.rows == k && k >= 1, "shapes V (m × n), W (m × k), H (k × n)");
  for (const double w : W.a) require(w >= 0 && std::isfinite(w), "W non-negative");
  for (const double h : H.a) require(h >= 0 && std::isfinite(h), "H non-negative");
  Nmf r;
  double last = nmf_objective(V, W, H, o.loss);
  r.trace.push_back(last);
  const std::size_t every = std::max<std::size_t>(o.check_every, 1);
  Rectangular WH, num, den;
  for (std::size_t it = 1; it <= o.max_iterations; it++) {
    if (o.loss == NmfLoss::frobenius) {
      // H ← H ⊙ (WᵀV) ⊘ (WᵀW H).
      const Rectangular Wt = W.transpose();
      num = Wt * V;
      den = (Wt * W) * H;
      for (std::size_t q = 0; q < H.a.size(); q++)
        if (den.a[q] > 0) H.a[q] *= num.a[q] / den.a[q];
      // W ← W ⊙ (V Hᵀ) ⊘ (W H Hᵀ).
      const Rectangular Ht = H.transpose();
      num = V * Ht;
      den = W * (H * Ht);
      for (std::size_t q = 0; q < W.a.size(); q++)
        if (den.a[q] > 0) W.a[q] *= num.a[q] / den.a[q];
    } else {
      // H_aμ ← H_aμ Σ_i W_ia V_iμ/(WH)_iμ / Σ_i W_ia.
      WH = W * H;
      Rectangular Q(m, n);
      for (std::size_t q = 0; q < Q.a.size(); q++) Q.a[q] = V.a[q] > 0 ? V.a[q] / std::max(WH.a[q], 1e-300) : 0.0;
      num = W.transpose() * Q;
      for (std::size_t a = 0; a < k; a++) {
        double ws = 0;
        for (std::size_t i = 0; i < m; i++) ws += W(i, a);
        if (!(ws > 0)) continue;
        for (std::size_t j = 0; j < n; j++) H(a, j) *= num(a, j) / ws;
      }
      // W_ia ← W_ia Σ_μ H_aμ V_iμ/(WH)_iμ / Σ_μ H_aμ.
      WH = W * H;
      for (std::size_t q = 0; q < Q.a.size(); q++) Q.a[q] = V.a[q] > 0 ? V.a[q] / std::max(WH.a[q], 1e-300) : 0.0;
      num = Q * H.transpose();
      for (std::size_t a = 0; a < k; a++) {
        double hs = 0;
        for (std::size_t j = 0; j < n; j++) hs += H(a, j);
        if (!(hs > 0)) continue;
        for (std::size_t i = 0; i < m; i++) W(i, a) *= num(i, a) / hs;
      }
    }
    r.iterations = it;
    if (it % every == 0 || it == o.max_iterations) {
      const double now = nmf_objective(V, W, H, o.loss);
      r.trace.push_back(now);
      const double drop = (last - now) / std::max(std::fabs(last), 1e-300);
      last = now;
      if (it >= o.min_iterations && drop <= o.tolerance) {
        r.converged = true;
        break;
      }
    }
  }
  if (o.normalise) normalise_factors(W, H);
  r.objective = nmf_objective(V, W, H, o.loss);
  r.W = std::move(W);
  r.H = std::move(H);
  return r;
}

// The factorisation of rank k from random starts (the best of o.restarts).
inline Nmf nmf(const Rectangular& V, std::size_t rank, const NmfOptions& o = {}) {
  using namespace nmf_detail;
  check_nonnegative(V);
  require(rank >= 1, "rank ≥ 1");
  const std::size_t m = V.rows, n = V.cols;
  double mean = 0;
  for (const double v : V.a) mean += v;
  mean /= static_cast<double>(std::max<std::size_t>(V.a.size(), 1));
  const double scale = std::sqrt(std::max(mean, 1e-300) / static_cast<double>(rank));
  Nmf best;
  bool have = false;
  std::vector<double> objectives;
  for (std::size_t s = 0; s < std::max<std::size_t>(o.restarts, 1); s++) {
    calx::random::Stream rng(o.seed, 0x4e4d4621u, static_cast<std::uint32_t>(s), static_cast<std::uint32_t>(rank));  // "NMF!"
    Rectangular W(m, rank), H(rank, n);
    for (double& w : W.a) w = scale * rng.uniform();
    for (double& h : H.a) h = scale * rng.uniform();
    Nmf r = nmf_from(V, std::move(W), std::move(H), o);
    objectives.push_back(r.objective);
    if (!have || r.objective < best.objective) {
      best = std::move(r);
      best.restart = s;
      have = true;
    }
  }
  best.restart_objectives = std::move(objectives);
  return best;
}

// ---------------------------------------------------------------------------------------------
// Similarities and silhouettes.

inline double cosine_similarity(std::span<const double> a, std::span<const double> b) {
  nmf_detail::require(a.size() == b.size(), "vectors of one length");
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size(); i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0;
}

// Silhouette widths s(i) = (b(i) − a(i))/max(a(i), b(i)) (Rousseeuw 1987) of points with the given
// labels (0..L−1) and symmetric distances; a(i) the mean distance to the rest of i's cluster, b(i) the
// least mean distance to another cluster (with one cluster, b(i) = far). A point alone in its cluster
// has s = 0.
inline std::vector<double> silhouette(const Matrix& distance, std::span<const std::size_t> label, double far = 1.0) {
  const std::size_t N = distance.n;
  nmf_detail::require(label.size() == N, "one label per point");
  std::size_t L = 0;
  for (const std::size_t l : label) L = std::max(L, l + 1);
  std::vector<double> s(N, 0.0), sum(L), count(L);
  for (std::size_t i = 0; i < N; i++) {
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(count.begin(), count.end(), 0.0);
    for (std::size_t j = 0; j < N; j++) {
      if (j == i) continue;
      sum[label[j]] += distance(i, j);
      count[label[j]] += 1;
    }
    if (count[label[i]] == 0) continue;
    const double a = sum[label[i]] / count[label[i]];
    double b = std::numeric_limits<double>::infinity();
    for (std::size_t l = 0; l < L; l++)
      if (l != label[i] && count[l] > 0) b = std::min(b, sum[l] / count[l]);
    if (!std::isfinite(b)) b = far;
    const double d = std::max(a, b);
    s[i] = d > 0 ? (b - a) / d : 0.0;
  }
  return s;
}

// ---------------------------------------------------------------------------------------------
// The stability of a rank.

enum class Resampling { none, multinomial, poisson };
enum class Normalisation { none, log2, hundred_x };

struct StabilityOptions {
  std::size_t replicates = 100;              // R
  Resampling resampling = Resampling::poisson;
  Normalisation normalisation = Normalisation::none;
  NmfOptions nmf;                            // each replicate's factorisation (best of nmf.restarts)
  std::uint64_t seed = 1;                    // of the resampling (nmf.seed is mixed with the replicate)
  std::size_t max_rounds = 100;              // of the clustering
  double average_threshold = 0.80;           // stable: average stability ≥ this
  double minimum_threshold = 0.20;           // and every cluster's ≥ this
};

struct RankStability {
  std::size_t rank = 0;
  Rectangular signatures;                    // W̄ (m × k), columns summing to 1, the most stable first
  Rectangular exposures;                     // H̄ (k × n), NNLS of V's columns on W̄
  std::vector<double> stability;             // each cluster's mean silhouette (W̄'s order)
  std::vector<double> spread;                // each cluster's mean cosine distance to its centroid
  double average_stability = 0, minimum_stability = 0;
  double frobenius_error = 0;                // ‖V − W̄H̄‖²_F
  double relative_error = 0;                 // ‖V − W̄H̄‖_F / ‖V‖_F
  double mean_objective = 0;                 // the replicates' final objectives, averaged
  bool stable = false;
  std::size_t rounds = 0;                    // of the clustering
  std::vector<std::vector<std::size_t>> cluster;  // cluster[r][a]: replicate r's signature a in W̄'s column
};

namespace nmf_detail {

inline Rectangular resample(const Rectangular& V, Resampling how, calx::random::Stream& rng) {
  Rectangular out = V;
  if (how == Resampling::poisson) {
    for (double& v : out.a) v = static_cast<double>(rng.poisson(v));
  } else if (how == Resampling::multinomial) {
    for (std::size_t j = 0; j < V.cols; j++) {
      const auto col = V.column(j);
      double total = 0;
      for (const double v : col) total += v;
      if (!(total > 0)) continue;
      const auto k = calx::random::multinomial(rng, static_cast<std::int64_t>(std::llround(total)), col);
      for (std::size_t i = 0; i < V.rows; i++) out(i, j) = static_cast<double>(k[i]);
    }
  }
  return out;
}

inline void normalise_columns(Rectangular& V, Normalisation how) {
  if (how == Normalisation::none) return;
  const double cap = 100.0 * static_cast<double>(V.rows);
  for (std::size_t j = 0; j < V.cols; j++) {
    double total = 0;
    for (std::size_t i = 0; i < V.rows; i++) total += V(i, j);
    if (!(total > 0)) continue;
    double f = 1;
    if (how == Normalisation::log2) f = std::log2(total) / total;
    else if (total > cap) f = cap / total;
    for (std::size_t i = 0; i < V.rows; i++) V(i, j) *= f;
  }
}

}  // namespace nmf_detail

inline RankStability nmf_stability(const Rectangular& V, std::size_t rank, const StabilityOptions& o = {}) {
  using namespace nmf_detail;
  check_nonnegative(V);
  require(rank >= 1, "rank ≥ 1");
  require(o.replicates >= 1, "at least one replicate");
  const std::size_t m = V.rows, n = V.cols, R = o.replicates, k = rank;
  RankStability out;
  out.rank = k;
  // The replicates.
  std::vector<Rectangular> sig(R);
  std::vector<double> obj(R);
  for (std::size_t r = 0; r < R; r++) {
    calx::random::Stream rng(o.seed, 0x5245534du, static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(k));  // "RESM"
    Rectangular Vr = resample(V, o.resampling, rng);
    normalise_columns(Vr, o.normalisation);
    NmfOptions no = o.nmf;
    no.normalise = true;
    no.seed = o.nmf.seed * 0x9E3779B97F4A7C15ull + r;
    const Nmf f = nmf(Vr, k, no);
    sig[r] = f.W;
    obj[r] = f.objective;
    out.mean_objective += f.objective / static_cast<double>(R);
  }
  // Centroids from the replicate of least objective.
  const std::size_t start = static_cast<std::size_t>(std::min_element(obj.begin(), obj.end()) - obj.begin());
  Rectangular centroid = sig[start];
  out.cluster.assign(R, std::vector<std::size_t>(k, 0));
  auto column = [](const Rectangular& A, std::size_t a) { return A.column(a); };
  std::vector<std::size_t> label(R * k);
  // The cosine distances between all the replicates' signatures (fixed; only the labels change).
  std::vector<std::vector<double>> pts;
  for (std::size_t r = 0; r < R; r++)
    for (std::size_t a = 0; a < k; a++) pts.push_back(column(sig[r], a));
  Matrix dist(R * k);
  for (std::size_t p = 0; p < R * k; p++)
    for (std::size_t q = p + 1; q < R * k; q++) dist(p, q) = dist(q, p) = std::max(0.0, 1 - cosine_similarity(pts[p], pts[q]));
  double previous = std::numeric_limits<double>::quiet_NaN(), average = 0;
  std::vector<double> sil;
  for (std::size_t round = 0; round < std::max<std::size_t>(o.max_rounds, 1); round++) {
    out.rounds = round + 1;
    // Match each replicate's signatures to the centroids.
    for (std::size_t r = 0; r < R; r++) {
      Rectangular cost(k, k);
      for (std::size_t a = 0; a < k; a++)
        for (std::size_t c = 0; c < k; c++) cost(a, c) = 1 - cosine_similarity(column(sig[r], a), column(centroid, c));
      const auto as = calx::optimisation::assignment(cost);
      for (std::size_t a = 0; a < k; a++) out.cluster[r][a] = as.column[a];
    }
    // New centroids: the clusters' means, renormalised.
    Rectangular next(m, k);
    for (std::size_t r = 0; r < R; r++)
      for (std::size_t a = 0; a < k; a++)
        for (std::size_t i = 0; i < m; i++) next(i, out.cluster[r][a]) += sig[r](i, a) / static_cast<double>(R);
    Rectangular empty(k, 0);
    normalise_factors(next, empty);
    centroid = next;
    // Silhouettes under the cosine distance.
    for (std::size_t r = 0; r < R; r++)
      for (std::size_t a = 0; a < k; a++) label[r * k + a] = out.cluster[r][a];
    sil = silhouette(dist, label, 1.0);
    average = 0;
    for (const double v : sil) average += v / static_cast<double>(sil.size());
    if (std::isfinite(previous) && std::fabs(average - previous) <= 1e-12) break;
    previous = average;
  }
  // Per cluster: stability and spread.
  std::vector<double> stab(k, 0.0), spread(k, 0.0);
  for (std::size_t r = 0; r < R; r++)
    for (std::size_t a = 0; a < k; a++) {
      const std::size_t c = out.cluster[r][a];
      stab[c] += sil[r * k + a] / static_cast<double>(R);
      spread[c] += (1 - cosine_similarity(column(sig[r], a), column(centroid, c))) / static_cast<double>(R);
    }
  // Order the clusters by stability, the most stable first.
  std::vector<std::size_t> order(k);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return stab[a] > stab[b]; });
  std::vector<std::size_t> place(k);
  for (std::size_t q = 0; q < k; q++) place[order[q]] = q;
  out.signatures = Rectangular(m, k);
  for (std::size_t q = 0; q < k; q++) {
    for (std::size_t i = 0; i < m; i++) out.signatures(i, q) = centroid(i, order[q]);
    out.stability.push_back(stab[order[q]]);
    out.spread.push_back(spread[order[q]]);
  }
  for (auto& row : out.cluster)
    for (auto& c : row) c = place[c];
  out.average_stability = 0;
  out.minimum_stability = std::numeric_limits<double>::infinity();
  for (const double s : out.stability) {
    out.average_stability += s / static_cast<double>(k);
    out.minimum_stability = std::min(out.minimum_stability, s);
  }
  out.stable = out.average_stability >= o.average_threshold && out.minimum_stability >= o.minimum_threshold;
  // Exposures and the reconstruction error against V itself.
  out.exposures = regression::nnls_columns(out.signatures, V);
  const Rectangular fit = out.signatures * out.exposures;
  double err = 0, norm = 0;
  for (std::size_t q = 0; q < V.a.size(); q++) {
    err += (V.a[q] - fit.a[q]) * (V.a[q] - fit.a[q]);
    norm += V.a[q] * V.a[q];
  }
  out.frobenius_error = err;
  out.relative_error = norm > 0 ? std::sqrt(err / norm) : 0.0;
  (void)n;
  return out;
}

// The stabilities of the ranks rank_min..rank_max.
inline std::vector<RankStability> nmf_rank_stability(const Rectangular& V, std::size_t rank_min, std::size_t rank_max, const StabilityOptions& o = {}) {
  nmf_detail::require(rank_min >= 1 && rank_min <= rank_max, "1 ≤ rank_min ≤ rank_max");
  std::vector<RankStability> out;
  for (std::size_t k = rank_min; k <= rank_max; k++) out.push_back(nmf_stability(V, k, o));
  return out;
}

}  // namespace tyche::multivariate
