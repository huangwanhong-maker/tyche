// Checks of tyche/multivariate/nmf.hpp: non-negative matrix factorisation by Lee and Seung's
// multiplicative updates, and the stability of a rank.
//
// - Lee and Seung's (2001) Theorems 1 and 2: ‖V − WH‖²_F and D(V‖WH) never increase along the
//   updates, on random data of several shapes (every check of the objective).
// - One update of each rule against the formulas written out entry by entry.
// - Exact recovery: a planted separable factorisation (anchor rows in W, anchor columns in H), V = WH,
//   recovered up to permutation and scale (cosines > 0.999 after matching by calx's assignment), the
//   objective near 0, for both objectives.
// - normalise: W's columns sum to 1 and WH is unchanged.
// - Silhouettes against a hand computation (Rousseeuw 1987's definitions).
// - Stability: Poisson counts of planted signatures; at the true rank the clusters are tight (average
//   stability > 0.9), the consensus signatures match the planted ones (cosine > 0.99); one rank more is
//   less stable; the relative error falls to the noise's level at the true rank; multinomial
//   resampling keeps the columns' totals; the same seed gives the same answer.

#include <tyche/multivariate/nmf.hpp>

#include <calx/optimisation/assignment.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace mv = tyche::multivariate;
using calx::linalg::Rectangular;

namespace {

// The planted factors: W (m × k) with anchor rows, columns summing to 1; H (k × n) with anchor columns.
void planted(calx::random::Stream& z, std::size_t m, std::size_t k, std::size_t n, double scale, Rectangular& W, Rectangular& H) {
  W = Rectangular(m, k);
  H = Rectangular(k, n);
  for (std::size_t i = 0; i < m; i++)
    for (std::size_t a = 0; a < k; a++) W(i, a) = i < k ? (i == a ? 1.0 : 0.0) : (z.uniform() < 0.5 ? z.uniform() : 0.0);
  for (std::size_t a = 0; a < k; a++)
    for (std::size_t j = 0; j < n; j++) H(a, j) = j < k ? (j == a ? scale : 0.0) : (z.uniform() < 0.6 ? scale * z.uniform() : 0.0);
  mv::normalise_factors(W, H);
}

// The best matching of found columns to planted ones by cosine: the least cosine over the pairs.
double matched_cosine(const Rectangular& found, const Rectangular& truth) {
  const std::size_t k = truth.cols;
  Rectangular cost(k, k);
  for (std::size_t a = 0; a < k; a++)
    for (std::size_t b = 0; b < k; b++) cost(a, b) = -mv::cosine_similarity(found.column(a), truth.column(b));
  const auto as = calx::optimisation::assignment(cost);
  double least = 1;
  for (std::size_t a = 0; a < k; a++) least = std::min(least, -cost(a, as.column[a]));
  return least;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- the theorems: monotone objectives ----
  {
    calx::random::Stream z(1, 0, 0, 0);
    double rise_f = 0, rise_k = 0;
    for (int rep = 0; rep < 12; rep++) {
      const std::size_t m = 5 + z.below(30), n = 5 + z.below(40), k = 1 + z.below(5);
      Rectangular V(m, n);
      for (auto& v : V.a) v = z.uniform() < 0.2 ? 0.0 : static_cast<double>(z.poisson(5));
      for (const auto loss : {mv::NmfLoss::frobenius, mv::NmfLoss::kullback_leibler}) {
        mv::NmfOptions o;
        o.loss = loss;
        o.max_iterations = 2000;
        o.check_every = 1;
        o.tolerance = 0;
        o.seed = static_cast<std::uint64_t>(rep + 1);
        const auto f = mv::nmf(V, k, o);
        for (std::size_t t = 1; t < f.trace.size(); t++) {
          const double rise = (f.trace[t] - f.trace[t - 1]) / std::max(f.trace[t - 1], 1e-300);
          (loss == mv::NmfLoss::frobenius ? rise_f : rise_k) = std::max(loss == mv::NmfLoss::frobenius ? rise_f : rise_k, rise);
        }
      }
    }
    s.check("Lee and Seung's Theorem 1: ‖V − WH‖²_F never increases (largest relative rise over 12 × 2000 updates)", std::max(0.0, rise_f), 1e-12);
    s.check("Lee and Seung's Theorem 2: D(V‖WH) never increases", std::max(0.0, rise_k), 1e-12);
  }

  // ---- one update against the formulas ----
  {
    calx::random::Stream z(2, 0, 0, 0);
    const std::size_t m = 4, n = 5, k = 2;
    Rectangular V(m, n), W(m, k), H(k, n);
    for (auto& v : V.a) v = z.uniform(0, 3);
    for (auto& v : W.a) v = z.uniform(0.1, 1);
    for (auto& v : H.a) v = z.uniform(0.1, 1);
    double worst = 0;
    for (const auto loss : {mv::NmfLoss::frobenius, mv::NmfLoss::kullback_leibler}) {
      Rectangular W1 = W, H1 = H;
      auto wh = [&](std::size_t i, std::size_t j) {
        double v = 0;
        for (std::size_t a = 0; a < k; a++) v += W1(i, a) * H1(a, j);
        return v;
      };
      // H first, then W with the new H (the order of the implementation).
      Rectangular Hn = H1;
      for (std::size_t a = 0; a < k; a++)
        for (std::size_t j = 0; j < n; j++) {
          double num = 0, den = 0;
          if (loss == mv::NmfLoss::frobenius) {
            for (std::size_t i = 0; i < m; i++) num += W1(i, a) * V(i, j), den += W1(i, a) * wh(i, j);
          } else {
            for (std::size_t i = 0; i < m; i++) num += W1(i, a) * V(i, j) / wh(i, j), den += W1(i, a);
          }
          Hn(a, j) = H1(a, j) * num / den;
        }
      H1 = Hn;
      Rectangular Wn = W1;
      for (std::size_t i = 0; i < m; i++)
        for (std::size_t a = 0; a < k; a++) {
          double num = 0, den = 0;
          if (loss == mv::NmfLoss::frobenius) {
            for (std::size_t j = 0; j < n; j++) num += V(i, j) * H1(a, j), den += wh(i, j) * H1(a, j);
          } else {
            for (std::size_t j = 0; j < n; j++) num += H1(a, j) * V(i, j) / wh(i, j), den += H1(a, j);
          }
          Wn(i, a) = W1(i, a) * num / den;
        }
      mv::NmfOptions o;
      o.loss = loss;
      o.max_iterations = 1;
      o.normalise = false;
      const auto f = mv::nmf_from(V, W, H, o);
      for (std::size_t q = 0; q < Wn.a.size(); q++) worst = std::max(worst, std::fabs(f.W.a[q] - Wn.a[q]));
      for (std::size_t q = 0; q < H1.a.size(); q++) worst = std::max(worst, std::fabs(f.H.a[q] - H1.a[q]));
    }
    s.check("one update of each rule equals Lee and Seung's eqs. 4 and 5 written out entry by entry", worst, 1e-14);
  }

  // ---- exact recovery of a planted factorisation ----
  {
    calx::random::Stream z(3, 0, 0, 0);
    Rectangular W, H;
    planted(z, 30, 4, 60, 100, W, H);
    const Rectangular V = W * H;
    for (const auto loss : {mv::NmfLoss::frobenius, mv::NmfLoss::kullback_leibler}) {
      mv::NmfOptions o;
      o.loss = loss;
      o.restarts = 5;
      o.max_iterations = 200000;
      o.tolerance = 1e-14;
      const auto f = mv::nmf(V, 4, o);
      double vnorm = 0;
      for (const double v : V.a) vnorm += loss == mv::NmfLoss::frobenius ? v * v : v;
      const std::string name = loss == mv::NmfLoss::frobenius ? "Frobenius" : "Kullback–Leibler";
      s.check("planted separable V = WH (30 × 60, rank 4), " + name + ": the objective relative to V's size", f.objective / vnorm, 1e-6);
      s.check("planted separable V = WH, " + name + ": W recovered up to permutation (1 − least matched cosine)", 1 - matched_cosine(f.W, W), 1e-3);
      double colsum = 0;
      for (std::size_t a = 0; a < 4; a++) {
        double t = 0;
        for (std::size_t i = 0; i < 30; i++) t += f.W(i, a);
        colsum = std::max(colsum, std::fabs(t - 1));
      }
      s.check("normalise: W's columns sum to 1 (" + name + ")", colsum, 1e-12);
    }
    // normalise leaves WH unchanged.
    Rectangular W2 = W, H2 = H;
    for (auto& v : W2.a) v *= 3;
    for (auto& v : H2.a) v /= 3;
    const Rectangular before = W2 * H2;
    mv::normalise_factors(W2, H2);
    const Rectangular after = W2 * H2;
    double d = 0;
    for (std::size_t q = 0; q < before.a.size(); q++) d = std::max(d, std::fabs(before.a[q] - after.a[q]) / (1 + std::fabs(before.a[q])));
    s.check("normalise_factors: WH unchanged", d, 1e-13);
  }

  // ---- silhouettes by hand ----
  {
    // Four points on a line at 0, 1, 5, 7: clusters {0, 1} and {5, 7}.
    const double x[4] = {0, 1, 5, 7};
    calx::linalg::Matrix D(4);
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) D(i, j) = std::fabs(x[i] - x[j]);
    const std::vector<std::size_t> label{0, 0, 1, 1};
    const auto sv = mv::silhouette(D, label);
    // Point 0: a = 1, b = (5 + 7)/2 = 6 → 5/6; point 1: a = 1, b = 5 → 4/5; point 2: a = 2, b = 4.5 →
    // 2.5/4.5; point 3: a = 2, b = 6.5 → 4.5/6.5.
    const double want[4] = {5.0 / 6, 4.0 / 5, 2.5 / 4.5, 4.5 / 6.5};
    double d = 0;
    for (int i = 0; i < 4; i++) d = std::max(d, std::fabs(sv[i] - want[i]));
    s.check("silhouette: four points on a line, by hand (Rousseeuw 1987)", d, 1e-15);
  }

  // ---- the stability of a rank ----
  {
    calx::random::Stream z(4, 0, 0, 0);
    Rectangular W, H;
    planted(z, 24, 3, 40, 400, W, H);
    const Rectangular mean = W * H;
    Rectangular V(mean.rows, mean.cols);
    for (std::size_t q = 0; q < V.a.size(); q++) V.a[q] = static_cast<double>(z.poisson(mean.a[q]));
    mv::StabilityOptions o;
    o.replicates = 30;
    o.nmf.max_iterations = 20000;
    o.nmf.tolerance = 1e-9;
    o.nmf.restarts = 2;
    const auto all = mv::nmf_rank_stability(V, 2, 5, o);
    const auto& k3 = all[1];
    s.check("stability: at the true rank 3 the clusters are tight (1 − average stability)", 1 - k3.average_stability, 0.1);
    s.check("stability: the consensus signatures match the planted ones (1 − least matched cosine)", 1 - matched_cosine(k3.signatures, W), 0.01);
    s.require("stability: rank 5 is less stable than rank 3", all[3].average_stability < k3.average_stability,
              std::to_string(all[3].average_stability) + " < " + std::to_string(k3.average_stability));
    s.require("stability: the relative error falls from rank 2 to rank 3, little after", all[0].relative_error > k3.relative_error &&
              all[2].relative_error > 0.8 * k3.relative_error, std::to_string(all[0].relative_error) + ", " + std::to_string(k3.relative_error) + ", " + std::to_string(all[2].relative_error));
    s.require("stability: rank 3 called stable (≥ 0.80 on average, ≥ 0.20 for every cluster)", k3.stable);
    s.require("stability: clusters ordered by stability, one signature of each replicate in each", std::is_sorted(k3.stability.rbegin(), k3.stability.rend()) &&
              std::all_of(k3.cluster.begin(), k3.cluster.end(), [](const std::vector<std::size_t>& c) {
                std::vector<std::size_t> t = c;
                std::sort(t.begin(), t.end());
                return t == std::vector<std::size_t>{0, 1, 2};
              }));
    const auto again = mv::nmf_stability(V, 3, o);
    s.require("stability: the same seeds give the same answer", again.signatures.a == k3.signatures.a && again.stability == k3.stability);
    mv::StabilityOptions om = o;
    om.resampling = mv::Resampling::multinomial;
    om.replicates = 10;
    const auto km = mv::nmf_stability(V, 3, om);
    s.check("stability with multinomial resampling (Alexandrov et al. 2013): signatures matched (1 − least cosine)", 1 - matched_cosine(km.signatures, W), 0.01);
    calx::random::Stream r(5, 0, 0, 0);
    const auto Vr = mv::nmf_detail::resample(V, mv::Resampling::multinomial, r);
    double total_gap = 0;
    for (std::size_t j = 0; j < V.cols; j++) {
      double a = 0, b = 0;
      for (std::size_t i = 0; i < V.rows; i++) a += V(i, j), b += Vr(i, j);
      total_gap = std::max(total_gap, std::fabs(a - b));
    }
    s.check("multinomial resampling keeps each column's total", total_gap, 0);
  }
  return s.done();
}
