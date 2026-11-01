// Change points in a sequence of observations: optimal partitioning, PELT, binary, wild binary and
// circular binary segmentation.
//
// THE PROBLEM. Observations y₀, …, y_{n−1} in their order (along time, or along a genome) are cut by
// change points 0 < τ₁ < ⋯ < τ_m < n into m + 1 segments [τ_k, τ_{k+1}) (half-open, 0-based; τ₀ = 0,
// τ_{m+1} = n), within each of which a model with its own parameters holds. A segment's cost C(a, b)
// is twice the negative maximised log-likelihood of y_a, …, y_{b−1}, up to terms that do not depend on
// the segmentation; a segmentation's objective is (Killick, Fearnhead and Eckley 2012, "Optimal
// detection of changepoints with a linear computational cost", JASA 107, 1590–1598, eq. 3)
//     Σ_{k=0}^{m} [C(τ_k, τ_{k+1}) + c log((τ_{k+1} − τ_k)/n)] + β m,
// with β the penalty per change point and c the weight of the segment lengths (0 except for MBIC).
//
// THE COSTS (SegmentModel; every one from prefix sums, O(1) per segment):
// - mean: Gaussian with a change in mean and a variance σ² known or estimated once for the whole
//   series: C = Σ wᵢ (yᵢ − ȳ_w)² / σ², with precision weights wᵢ (Var yᵢ = σ²/wᵢ; all 1 by default) and
//   ȳ_w the weighted mean of the segment. σ² not given is robust_variance(y, w): the MAD of the first
//   differences (Hampel 1974; Fryzlewicz 2014, Ann. Statist. 42, 2243–2281, §3; Fearnhead and
//   Fryzlewicz 2024, arXiv:2405.06796, §7), consistent when changes are rare and the noise is iid
//   Gaussian, an underestimate under positive autocorrelation or heavy tails (ibid.).
// - variance: Gaussian of known mean μ with a change in variance: C = ℓ log(2π σ̂²) + Σ(y − μ)²/σ̂²,
//   σ̂² = max(Σ(y − μ)²/ℓ, floor), ℓ the segment's length.
// - mean_variance: Gaussian with a change in mean and variance: C = ℓ log(2π σ̂²) + RSS/σ̂²,
//   σ̂² = max(RSS/ℓ, floor), RSS = Σ(y − ȳ)²; segments of at least two observations.
// The floor makes each cost the exact −2 log-likelihood maximised over σ² ≥ floor, so that a constant
// segment costs a finite amount and the costs keep the inequality pruning needs (below).
// - poisson: counts yᵢ ≥ 0 (not necessarily integers) with E yᵢ = λ eᵢ, eᵢ > 0 an exposure (expected
//   counts, all 1 by default): C = 2 (Y − Y log(Y/E)) with Y = Σ y, E = Σ e over the segment
//   (0 when Y = 0), which is −2ℓ at λ̂ = Y/E plus 2 Σ (yᵢ log eᵢ − log yᵢ!), a sum over observations.
// SumCost adds the costs of several series observed at the same positions (joint segmentation, e.g. of
// read depth and allele fractions); any type with size() and operator()(begin, end) is a cost
// (the concept SegmentCostFunction), and the searches below are templates over it.
//
// THE PENALTIES (Penalty; on the scale of −2 log-likelihood, d the parameters of a segment):
// - bic (SIC): β = (d + 1) log n, the location and the d parameters each counted (Yao 1988, Statist.
//   Probab. Lett. 6, 181–189; Fearnhead and Fryzlewicz 2024, eq. 12: SIC(N) = (n/2) log σ̂² + N log n
//   for a change in mean, i.e. β = 2 log n); aic: β = 2(d + 1); hannan_quinn: β = 2(d + 1) log log n;
// - mbic: Zhang and Siegmund's (2007, Biometrics 63, 22–32, Theorem 1; restated by Fearnhead and
//   Fryzlewicz 2024, eq. 22) modified BIC, −2 × ½ [3m log n + Σ_k log((τ_{k+1} − τ_k)/n)] for a change
//   in mean: β = 3 log n and c = 1. For d > 1 this header uses β = (d + 2) log n and c = d, the same
//   Laplace approximation with d parameters per segment (a generalisation, not Zhang and Siegmund's);
// - manual: β given; none: β = 0 (with binary segmentation, the solution path to max_changepoints).
//
// THE SEARCHES.
// - optimal_partitioning: the exact minimiser by F(t) = min_s F(s) + C(s, t) + β, F(0) = −β (Jackson
//   et al. 2005, IEEE Signal Process. Lett. 12, 105–108; Killick et al. 2012, Algorithm 1). O(n²).
// - pelt: the same minimiser with the candidates s pruned once F(s) + C(s, t) > F(t) + K (Killick et
//   al. 2012, Theorem 3.1 and Algorithm 2), valid because every cost here satisfies
//   C(a, t) + C(t, b) ≤ C(a, b) (a split never lowers a maximised likelihood) and log q is subadditive on
//   (0, 1]. With a minimum segment length ℓ_min, t can be the last change before T only for
//   T ≥ t + ℓ_min, so a candidate found dominated at t is removed only from T = t + ℓ_min on: the
//   departure from Algorithm 2 that keeps PELT exact with ℓ_min > 1. K = 10⁻⁹ (1 + |F(t)|) absorbs the
//   rounding of the cost inequality, so that pruning never removes a candidate that ties the optimum:
//   pelt returns the very segmentation optimal_partitioning returns (both take the earliest of tied
//   last change points). Expected O(n) when the number of change points grows with n (Killick et
//   al. 2012, Theorem 3.2); O(n²) at worst.
// - binary_segmentation: the best single split of a segment, accepted while its gain
//   C(a, b) − C(a, t) − C(t, b) (with the segment-length terms) exceeds β (Killick et al. 2012, eq. 2;
//   Scott and Knott 1974, Biometrics 30, 507–512; Vostrikova 1981, Soviet Math. Dokl. 24, 55–59),
//   the segments taken greatest gain first, up to max_changepoints. Not exact; O(n log n).
// - wild_binary_segmentation: Fryzlewicz's (2014) random intervals, the full one among them; the
//   solution path of the WBS.NONREC procedure (Fearnhead and Fryzlewicz 2024, §4.2), each candidate
//   the best split of the interval of greatest gain left, intervals containing it removed; then either
//   the gains above a threshold, or the prefix of the path minimising the penalised objective (the
//   approach of Fryzlewicz's strengthened SIC with the penalty chosen here). For the mean model the gain
//   is the squared CUSUM over σ², so Fryzlewicz's threshold C √(2 log n) σ is a gain of 2C² log n.
//
// CIRCULAR BINARY SEGMENTATION (Olshen, Venkatraman, Lucito and Wigler 2004, Biostatistics 5,
// 557–572; Venkatraman and Olshen 2007, Bioinformatics 23, 657–663). A segment of m observations is
// joined end to end into a circle; an arc [i, j) against its complement is tested by the two-sample
// t statistic, and the segment is split where T = max |t_ij| is significant: once (i = 0 or j = m) or
// twice. With S̃ the partial sums of the segment's centred values and k = j − i,
//     Z_ij = (S̃_j − S̃_i) / √(k (m − k)/m),   t_ij = Z_ij / √((TSS − Z_ij²)/(m − 2)),
// TSS the segment's total sum of squares: Z_ij² is the reduction of the residual sum of squares the arc
// makes, and t is increasing in |Z|, so the maximal t is found by maximising |Z|. Its reference
// distribution is that of the same statistic over permutations of the segment (seeded calx streams
// keyed by the segment's bounds, so that the result does not depend on the order of the recursion);
// the p-value is (1 + R)/(1 + B), R of B permutations reaching the observed maximum (Phipson and
// Smyth 2010, Stat. Appl. Genet. Mol. Biol. 9, 39). Three accelerations, all exact or with a stated
// error:
// - the maximum, and in a permutation the question whether any arc reaches the observed one, by
//   branch and bound on a tree of index ranges: a pair of ranges bounds |S̃_j − S̃_i| by their extremes
//   and the denominator by its least value over the arc lengths they allow (√(k(m − k)/m) is concave);
//   only pairs whose bound reaches the best (or the observed) value are refined. Exact.
// - curtailment (Besag and Clifford 1991, Biometrika 78, 301–304): the permutations stop, not
//   significant, as soon as R exceeds what (1 + R)/(1 + B) ≤ α allows. Exact: the decision is the full
//   test's.
// - early rejection (after Venkatraman and Olshen 2007's sequential boundary): after b_i permutations
//   with fewer than i exceedances the split is declared significant, b_i the least b for which a
//   hypergeometric draw of b of the B permutations, r* = ⌊α(B + 1)⌋ of them exceedances (the least
//   count the full test does not reject), shows fewer than i of them with probability ≤ η 2^{−i}. Given
//   the full record, the early rule overturns a non-rejection with probability below η; under the null
//   the added size is (1/(B + 1)) Σ_{r ≥ r*} P(early | r), far below η (checks).
// Splits are refused that would leave a segment shorter than min_size (DNAcopy's min.width). Then,
// optionally: pruning (Olshen et al. 2004): of the K change points found, the fewest j whose best
// subset (by dynamic programming) has a residual sum of squares SS_j with SS_j/SS_K − 1 < γ; and the
// undoing of splits whose neighbouring means differ by less than undo_sd σ̂ (σ̂ from robust_variance;
// DNAcopy uses a trimmed standard deviation), the closest pair first.
//
// LIMITS. The penalised searches are exact for their objective; the objective is a model choice whose
// consistency needs the assumptions of its source (independent observations, the model's family). The
// CBS p-values are exact permutation p-values for the first split; after it, the levels of later tests
// are conditional (Olshen et al. 2004). Weights in CBS, the hybrid tail approximation of Venkatraman
// and Olshen (2007), and the smoothing of outliers are not written.

#pragma once

#include <tyche/descriptive/summary.hpp>

#include <calx/random/stream.hpp>
#include <calx/special_functions/gamma.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyche::time_series {

// ---------------------------------------------------------------------------------------------
// Models, segments and segmentations.

enum class SegmentModel { mean, variance, mean_variance, poisson };

struct CostOptions {
  SegmentModel model = SegmentModel::mean;
  double variance = 0;           // mean: σ² > 0, or ≤ 0 to estimate it by robust_variance(y, weights)
  double mean = 0;               // variance: the known mean μ
  std::vector<double> weights;   // mean: precisions wᵢ > 0 (Var yᵢ = σ²/wᵢ); empty: all 1
  std::vector<double> exposure;  // poisson: eᵢ > 0 with E yᵢ = λ eᵢ; empty: all 1
  double variance_floor = 0;     // variance, mean_variance: σ̂² ≥ floor; ≤ 0: 10⁻¹⁰ × the series' variance
};

struct Segment {
  std::size_t begin = 0, end = 0;  // the observations [begin, end), 0-based
  double mean = 0;                 // ȳ_w (mean, mean_variance), μ (variance), λ̂ = Σy/Σe (poisson)
  double variance = 0;             // σ² (mean), σ̂² (variance, mean_variance; floored), λ̂ (poisson)
  double cost = 0;                 // C(begin, end)
  std::size_t size() const { return end - begin; }
};

struct Segmentation {
  std::vector<std::size_t> changepoints;  // τ₁ < ⋯ < τ_m, each the first observation of a new segment
  std::vector<Segment> segments;          // the m + 1 segments, covering [0, n) in order
  double cost = 0;                        // Σ C over the segments
  double penalty = 0;                     // β, per change point
  double objective = 0;                   // Σ C + c Σ log(ℓ_k/n) + β m
  double variance = 0;                    // the mean model's σ² (given or estimated); 0 for the others
  std::vector<double> scores;             // per change point: the gain (BS, WBS), |t| (CBS); empty otherwise
  std::vector<double> p_values;           // per change point: CBS's permutation p-value of its split
  std::size_t evaluations = 0;            // segment costs (or arc bounds, for CBS) evaluated: the work done
};

// ---------------------------------------------------------------------------------------------
// The noise variance.

// σ̂² = (1.4826 MAD(dᵢ))², dᵢ = (y_{i+1} − yᵢ)/√(1/wᵢ + 1/w_{i+1}) (= Δy/√2 for unit weights); MAD the
// median absolute deviation from the median, 1.4826 = 1/Φ⁻¹(3/4). If the MAD is 0 (data on a lattice),
// the mean square of the dᵢ; if that is 0 too (a constant series), 1.
inline double robust_variance(std::span<const double> y, std::span<const double> weights = {}) {
  const std::size_t n = y.size();
  if (!weights.empty() && weights.size() != n) throw std::invalid_argument("tyche::time_series::robust_variance: one weight per observation");
  if (n < 2) return 1;
  std::vector<double> d(n - 1);
  for (std::size_t i = 0; i + 1 < n; i++) {
    const double s = weights.empty() ? std::numbers::sqrt2 : std::sqrt(1 / weights[i] + 1 / weights[i + 1]);
    d[i] = (y[i + 1] - y[i]) / s;
  }
  const double med = descriptive::median(d);
  std::vector<double> a(d.size());
  for (std::size_t i = 0; i < d.size(); i++) a[i] = std::fabs(d[i] - med);
  const double sigma = 1.482602218505602 * descriptive::median(a);
  if (sigma > 0) return sigma * sigma;
  double ms = 0;
  for (const double v : d) ms += v * v;
  ms /= static_cast<double>(d.size());
  return ms > 0 ? ms : 1.0;
}

// ---------------------------------------------------------------------------------------------
// Segment costs.

namespace changepoint_detail {

// Prefix sums P[t] = Σ_{i<t} f(i) by Neumaier's compensated summation, each stored rounded once.
template <class F>
inline std::vector<double> prefix(std::size_t n, F&& f) {
  std::vector<double> p(n + 1, 0.0);
  double s = 0, c = 0;
  for (std::size_t i = 0; i < n; i++) {
    const double v = f(i);
    const double t = s + v;
    c += std::fabs(s) >= std::fabs(v) ? (s - t) + v : (v - t) + s;
    s = t;
    p[i + 1] = s + c;
  }
  return p;
}

inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("tyche::time_series::changepoint: ") + what);
}

}  // namespace changepoint_detail

class SegmentCost {
 public:
  SegmentCost() = default;

  explicit SegmentCost(std::span<const double> y, const CostOptions& o = {}) : model_(o.model), n_(y.size()) {
    changepoint_detail::require(o.weights.empty() || o.weights.size() == n_, "one weight per observation");
    changepoint_detail::require(o.exposure.empty() || o.exposure.size() == n_, "one exposure per observation");
    changepoint_detail::require(o.weights.empty() || o.model == SegmentModel::mean, "weights belong to the mean model");
    changepoint_detail::require(o.exposure.empty() || o.model == SegmentModel::poisson, "exposures belong to the poisson model");
    for (std::size_t i = 0; i < n_; i++) changepoint_detail::require(std::isfinite(y[i]), "finite observations");
    for (const double w : o.weights) changepoint_detail::require(w > 0 && std::isfinite(w), "positive finite weights");
    for (const double e : o.exposure) changepoint_detail::require(e > 0 && std::isfinite(e), "positive finite exposures");
    auto w = [&](std::size_t i) { return o.weights.empty() ? 1.0 : o.weights[i]; };
    switch (model_) {
      case SegmentModel::mean: {
        sigma2_ = o.variance > 0 ? o.variance : robust_variance(y, o.weights);
        const auto W = changepoint_detail::prefix(n_, w);
        const auto A = changepoint_detail::prefix(n_, [&](std::size_t i) { return w(i) * y[i]; });
        shift_ = n_ ? A[n_] / W[n_] : 0.0;
        w_ = W;
        a_ = changepoint_detail::prefix(n_, [&](std::size_t i) { return w(i) * (y[i] - shift_); });
        q_ = changepoint_detail::prefix(n_, [&](std::size_t i) { return w(i) * (y[i] - shift_) * (y[i] - shift_); });
        break;
      }
      case SegmentModel::variance:
      case SegmentModel::mean_variance: {
        shift_ = model_ == SegmentModel::variance ? o.mean : (n_ ? descriptive::mean(y) : 0.0);
        a_ = changepoint_detail::prefix(n_, [&](std::size_t i) { return y[i] - shift_; });
        q_ = changepoint_detail::prefix(n_, [&](std::size_t i) { return (y[i] - shift_) * (y[i] - shift_); });
        if (o.variance_floor > 0) {
          floor_ = o.variance_floor;
        } else {
          const double v = n_ > 1 ? descriptive::variance(y, 0) : 0.0;
          floor_ = v > 0 ? 1e-10 * v : 1e-300;
        }
        break;
      }
      case SegmentModel::poisson: {
        for (std::size_t i = 0; i < n_; i++) changepoint_detail::require(y[i] >= 0, "poisson counts are non-negative");
        a_ = changepoint_detail::prefix(n_, [&](std::size_t i) { return y[i]; });
        w_ = changepoint_detail::prefix(n_, [&](std::size_t i) { return o.exposure.empty() ? 1.0 : o.exposure[i]; });
        break;
      }
    }
  }

  std::size_t size() const { return n_; }
  SegmentModel model() const { return model_; }
  // The shortest segment of finite cost: 2 for mean_variance, 1 for the others.
  std::size_t min_size() const { return model_ == SegmentModel::mean_variance ? 2 : 1; }
  // d, the parameters of a segment: 2 for mean_variance, 1 for the others.
  std::size_t parameters() const { return model_ == SegmentModel::mean_variance ? 2 : 1; }
  // The mean model's σ² (given or estimated); 0 for the others.
  double variance() const { return model_ == SegmentModel::mean ? sigma2_ : 0.0; }

  // C(begin, end), begin < end ≤ n.
  double operator()(std::size_t b, std::size_t e) const {
    const double len = static_cast<double>(e - b);
    switch (model_) {
      case SegmentModel::mean: {
        const double W = w_[e] - w_[b], A = a_[e] - a_[b];
        return std::max(0.0, q_[e] - q_[b] - A * A / W) / sigma2_;
      }
      case SegmentModel::variance: {
        const double Q = std::max(0.0, q_[e] - q_[b]);
        const double s2 = std::max(Q / len, floor_);
        return len * std::log(2 * std::numbers::pi * s2) + Q / s2;
      }
      case SegmentModel::mean_variance: {
        const double A = a_[e] - a_[b];
        const double rss = std::max(0.0, q_[e] - q_[b] - A * A / len);
        const double s2 = std::max(rss / len, floor_);
        return len * std::log(2 * std::numbers::pi * s2) + rss / s2;
      }
      case SegmentModel::poisson: {
        const double Y = std::max(0.0, a_[e] - a_[b]), E = w_[e] - w_[b];
        return Y > 0 ? 2 * (Y - Y * std::log(Y / E)) : 0.0;
      }
    }
    return 0;
  }

  // The segment [begin, end) with its estimates and cost.
  Segment segment(std::size_t b, std::size_t e) const {
    Segment s;
    s.begin = b;
    s.end = e;
    s.cost = (*this)(b, e);
    const double len = static_cast<double>(e - b);
    switch (model_) {
      case SegmentModel::mean: {
        const double W = w_[e] - w_[b];
        s.mean = shift_ + (a_[e] - a_[b]) / W;
        s.variance = sigma2_;
        break;
      }
      case SegmentModel::variance:
        s.mean = shift_;
        s.variance = std::max(std::max(0.0, q_[e] - q_[b]) / len, floor_);
        break;
      case SegmentModel::mean_variance: {
        const double A = a_[e] - a_[b];
        s.mean = shift_ + A / len;
        s.variance = std::max(std::max(0.0, q_[e] - q_[b] - A * A / len) / len, floor_);
        break;
      }
      case SegmentModel::poisson:
        s.mean = std::max(0.0, a_[e] - a_[b]) / (w_[e] - w_[b]);
        s.variance = s.mean;
        break;
    }
    return s;
  }

 private:
  SegmentModel model_ = SegmentModel::mean;
  std::size_t n_ = 0;
  double sigma2_ = 1, shift_ = 0, floor_ = 0;
  std::vector<double> w_, a_, q_;  // prefix sums: weights or exposures; first and second moments
};

// The sum of the costs of several series at the same positions: one segmentation for all of them.
class SumCost {
 public:
  SumCost() = default;
  explicit SumCost(std::vector<SegmentCost> parts) : parts_(std::move(parts)) {
    for (const auto& p : parts_) changepoint_detail::require(p.size() == parts_.front().size(), "the series of a SumCost have one length");
  }
  std::size_t size() const { return parts_.empty() ? 0 : parts_.front().size(); }
  std::size_t min_size() const {
    std::size_t m = 1;
    for (const auto& p : parts_) m = std::max(m, p.min_size());
    return m;
  }
  std::size_t parameters() const {
    std::size_t d = 0;
    for (const auto& p : parts_) d += p.parameters();
    return d;
  }
  double operator()(std::size_t b, std::size_t e) const {
    double c = 0;
    for (const auto& p : parts_) c += p(b, e);
    return c;
  }
  const std::vector<SegmentCost>& parts() const { return parts_; }

 private:
  std::vector<SegmentCost> parts_;
};

template <class Cost>
concept SegmentCostFunction = requires(const Cost& c, std::size_t b, std::size_t e) {
  { c.size() } -> std::convertible_to<std::size_t>;
  { c(b, e) } -> std::convertible_to<double>;
};

// ---------------------------------------------------------------------------------------------
// Penalties.

enum class Penalty { bic, mbic, aic, hannan_quinn, manual, none };

struct PenaltyTerms {
  double per_changepoint = 0;  // β
  double per_segment = 0;      // c: each segment of length ℓ adds c log(ℓ/n) (MBIC); 0 otherwise
  std::size_t n = 0;           // the series' length in log(ℓ/n); 0: the cost's size
};

// The terms of a penalty for a series of n observations whose segments have d parameters; manual is
// β for Penalty::manual.
inline PenaltyTerms penalty_terms(Penalty p, std::size_t n, std::size_t d, double manual = 0) {
  const double ln = std::log(static_cast<double>(std::max<std::size_t>(n, 2))), dd = static_cast<double>(d);
  PenaltyTerms t;
  t.n = n;
  switch (p) {
    case Penalty::bic: t.per_changepoint = (dd + 1) * ln; break;
    case Penalty::mbic: t.per_changepoint = (dd + 2) * ln; t.per_segment = dd; break;
    case Penalty::aic: t.per_changepoint = 2 * (dd + 1); break;
    case Penalty::hannan_quinn: t.per_changepoint = 2 * (dd + 1) * std::log(std::max(ln, 1.0)); break;
    case Penalty::manual:
      changepoint_detail::require(manual >= 0 && std::isfinite(manual), "a manual penalty is finite and ≥ 0");
      t.per_changepoint = manual;
      break;
    case Penalty::none: break;
  }
  return t;
}

// ---------------------------------------------------------------------------------------------
// The generic searches, on any cost.

namespace changepoint_detail {

template <class Cost>
inline double penalised(const Cost& cost, const PenaltyTerms& p, std::size_t b, std::size_t e) {
  double v = cost(b, e);
  if (p.per_segment != 0) v += p.per_segment * std::log(static_cast<double>(e - b) / static_cast<double>(p.n));
  return v;
}

// The segmentation with the given change points: segments with their costs (estimates NaN), the
// objective under the penalty.
template <class Cost>
inline Segmentation finish(const Cost& cost, std::vector<std::size_t> cps, const PenaltyTerms& p) {
  Segmentation s;
  const std::size_t n = cost.size();
  s.changepoints = std::move(cps);
  s.penalty = p.per_changepoint;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::size_t b = 0;
  double objective = 0;
  for (std::size_t k = 0; k <= s.changepoints.size(); k++) {
    const std::size_t e = k < s.changepoints.size() ? s.changepoints[k] : n;
    Segment g;
    g.begin = b;
    g.end = e;
    g.mean = g.variance = nan;
    g.cost = cost(b, e);
    s.cost += g.cost;
    objective += penalised(cost, p, b, e);
    s.segments.push_back(g);
    b = e;
  }
  s.objective = objective + p.per_changepoint * static_cast<double>(s.changepoints.size());
  return s;
}

inline std::size_t effective_min_size(std::size_t requested, std::size_t model_min) { return std::max<std::size_t>({requested, model_min, 1}); }

// Optimal partitioning (prune = false) or PELT (prune = true).
template <class Cost>
inline Segmentation partition(const Cost& cost, PenaltyTerms pen, std::size_t ms, bool prune) {
  const std::size_t n = cost.size();
  if (pen.n == 0) pen.n = n;
  ms = std::max<std::size_t>(ms, 1);
  if (n == 0) return {};
  require(n >= ms, "the series is shorter than the minimum segment");
  const double beta = pen.per_changepoint, inf = std::numeric_limits<double>::infinity();
  std::vector<double> F(n + 1, inf);
  std::vector<std::size_t> last(n + 1, 0);
  F[0] = -beta;
  std::size_t evaluations = 0;
  std::vector<std::size_t> cand{0};
  std::vector<std::size_t> expiry(n + 1, std::numeric_limits<std::size_t>::max());
  std::vector<double> value;
  for (std::size_t t = ms; t <= n; t++) {
    double best = inf;
    std::size_t arg = 0;
    if (!prune) {
      for (std::size_t s = 0; s + ms <= t; s = s == 0 ? ms : s + 1) {
        if (!(F[s] < inf)) continue;
        const double v = F[s] + penalised(cost, pen, s, t) + beta;
        evaluations++;
        if (v < best) {
          best = v;
          arg = s;
        }
      }
    } else {
      std::size_t keep = 0;
      value.resize(cand.size());
      for (std::size_t idx = 0; idx < cand.size(); idx++) {
        const std::size_t s = cand[idx];
        if (expiry[s] <= t) continue;
        cand[keep] = s;
        if (s + ms > t) {
          value[keep++] = inf;
          continue;
        }
        const double v = F[s] + penalised(cost, pen, s, t) + beta;
        evaluations++;
        value[keep++] = v;
        if (v < best) {
          best = v;
          arg = s;
        }
      }
      cand.resize(keep);
      if (best < inf) {
        const double K = 1e-9 * (1 + std::fabs(best));
        for (std::size_t idx = 0; idx < keep; idx++)
          if (value[idx] < inf && value[idx] - beta > best + K) expiry[cand[idx]] = std::min(expiry[cand[idx]], t + ms);
      }
    }
    F[t] = best;
    last[t] = arg;
    if (prune && best < inf) cand.push_back(t);
  }
  std::vector<std::size_t> cps;
  for (std::size_t t = n; t > 0;) {
    const std::size_t s = last[t];
    if (s > 0) cps.push_back(s);
    t = s;
  }
  std::reverse(cps.begin(), cps.end());
  Segmentation out = finish(cost, std::move(cps), pen);
  out.objective = F[n];
  out.evaluations = evaluations;
  return out;
}

// The best split of [b, e) into [b, t) and [t, e), both at least ms long: (t, gain); gain −∞ if none.
template <class Cost>
inline std::pair<std::size_t, double> best_split(const Cost& cost, const PenaltyTerms& pen, std::size_t b, std::size_t e, std::size_t ms, bool with_lengths, std::size_t& evaluations) {
  std::pair<std::size_t, double> best{0, -std::numeric_limits<double>::infinity()};
  if (e - b < 2 * ms) return best;
  PenaltyTerms p = pen;
  if (!with_lengths) p.per_segment = 0;
  const double whole = penalised(cost, p, b, e);
  for (std::size_t t = b + ms; t + ms <= e; t++) {
    const double g = whole - penalised(cost, p, b, t) - penalised(cost, p, t, e);
    evaluations += 2;
    if (g > best.second) best = {t, g};
  }
  return best;
}

template <class Cost>
inline Segmentation finish_scored(const Cost& cost, std::vector<std::pair<std::size_t, double>> found, const PenaltyTerms& pen) {
  std::sort(found.begin(), found.end());
  std::vector<std::size_t> cps;
  std::vector<double> scores;
  for (const auto& [t, g] : found) {
    cps.push_back(t);
    scores.push_back(g);
  }
  Segmentation s = finish(cost, std::move(cps), pen);
  s.scores = std::move(scores);
  return s;
}

}  // namespace changepoint_detail

// The exact minimiser of the penalised objective by optimal partitioning, O(n²).
template <SegmentCostFunction Cost>
inline Segmentation optimal_partitioning(const Cost& cost, const PenaltyTerms& penalty, std::size_t min_size) {
  return changepoint_detail::partition(cost, penalty, min_size, false);
}

// The same minimiser by PELT.
template <SegmentCostFunction Cost>
inline Segmentation pelt(const Cost& cost, const PenaltyTerms& penalty, std::size_t min_size) {
  return changepoint_detail::partition(cost, penalty, min_size, true);
}

// Binary segmentation: splits accepted, greatest gain first, while the gain exceeds β, up to
// max_changepoints.
template <SegmentCostFunction Cost>
inline Segmentation binary_segmentation(const Cost& cost, PenaltyTerms pen, std::size_t min_size,
                                        std::size_t max_changepoints = std::numeric_limits<std::size_t>::max()) {
  const std::size_t n = cost.size();
  if (pen.n == 0) pen.n = n;
  const std::size_t ms = std::max<std::size_t>(min_size, 1);
  if (n == 0) return {};
  struct Split {
    double gain;
    std::size_t b, e, t;
    bool operator<(const Split& o) const { return gain < o.gain || (gain == o.gain && t > o.t); }
  };
  std::size_t evaluations = 0;
  std::priority_queue<Split> queue;
  auto offer = [&](std::size_t b, std::size_t e) {
    const auto [t, g] = changepoint_detail::best_split(cost, pen, b, e, ms, true, evaluations);
    if (g > -std::numeric_limits<double>::infinity()) queue.push({g, b, e, t});
  };
  offer(0, n);
  std::vector<std::pair<std::size_t, double>> found;
  while (!queue.empty() && found.size() < max_changepoints) {
    const Split s = queue.top();
    queue.pop();
    if (!(s.gain > pen.per_changepoint)) break;
    found.push_back({s.t, s.gain});
    offer(s.b, s.t);
    offer(s.t, s.e);
  }
  Segmentation out = changepoint_detail::finish_scored(cost, std::move(found), pen);
  out.evaluations = evaluations;
  return out;
}

struct WbsOptions {
  std::size_t intervals = 5000;  // M random intervals (with replacement), besides the whole series
  std::uint64_t seed = 1;        // of the calx stream that draws them
  double threshold = 0;          // > 0: keep the path's change points of gain ≥ threshold; 0: penalised choice
};

// Wild binary segmentation (Fryzlewicz 2014) on any cost, the gain as its contrast.
template <SegmentCostFunction Cost>
inline Segmentation wild_binary_segmentation(const Cost& cost, PenaltyTerms pen, std::size_t min_size, const WbsOptions& o = {},
                                             std::size_t max_changepoints = std::numeric_limits<std::size_t>::max()) {
  const std::size_t n = cost.size();
  if (pen.n == 0) pen.n = n;
  const std::size_t ms = std::max<std::size_t>(min_size, 1);
  if (n == 0) return {};
  struct Interval {
    std::size_t b, e, t;
    double gain;
    bool alive;
  };
  std::vector<Interval> iv;
  std::size_t evaluations = 0;
  auto add = [&](std::size_t b, std::size_t e) {
    const auto [t, g] = changepoint_detail::best_split(cost, pen, b, e, ms, false, evaluations);
    if (g > -std::numeric_limits<double>::infinity()) iv.push_back({b, e, t, g, true});
  };
  add(0, n);
  calx::random::Stream rng(o.seed, 0x57425321u, 0, 0);  // "WBS!"
  for (std::size_t m = 0; m < o.intervals; m++) {
    std::size_t s = rng.below(n + 1), e = rng.below(n + 1);
    if (s > e) std::swap(s, e);
    add(s, e);
  }
  // The solution path (WBS.NONREC).
  std::vector<std::pair<std::size_t, double>> path;
  for (;;) {
    std::size_t best = iv.size();
    for (std::size_t m = 0; m < iv.size(); m++)
      if (iv[m].alive && (best == iv.size() || iv[m].gain > iv[best].gain || (iv[m].gain == iv[best].gain && iv[m].t < iv[best].t))) best = m;
    if (best == iv.size() || path.size() >= max_changepoints) break;
    const std::size_t t = iv[best].t;
    path.push_back({t, iv[best].gain});
    for (auto& v : iv)
      if (v.alive && v.b < t && t < v.e) v.alive = false;
  }
  // The choice along the path.
  std::size_t keep = 0;
  if (o.threshold > 0) {
    while (keep < path.size() && path[keep].second >= o.threshold) keep++;
  } else {
    std::set<std::size_t> cut{0, n};
    double objective = changepoint_detail::penalised(cost, pen, 0, n), best_objective = objective;
    for (std::size_t k = 0; k < path.size(); k++) {
      const std::size_t t = path[k].first;
      const auto hi = cut.upper_bound(t);
      const std::size_t e = *hi, b = *std::prev(hi);
      objective += changepoint_detail::penalised(cost, pen, b, t) + changepoint_detail::penalised(cost, pen, t, e) - changepoint_detail::penalised(cost, pen, b, e) + pen.per_changepoint;
      cut.insert(t);
      if (objective < best_objective) {
        best_objective = objective;
        keep = k + 1;
      }
    }
  }
  path.resize(keep);
  Segmentation out = changepoint_detail::finish_scored(cost, std::move(path), pen);
  out.evaluations = evaluations;
  return out;
}

// ---------------------------------------------------------------------------------------------
// The univariate front ends.

struct ChangepointOptions {
  CostOptions cost;
  Penalty penalty = Penalty::bic;
  double beta = 0;               // the penalty per change point for Penalty::manual (scale: −2 log-likelihood)
  std::size_t min_size = 0;      // the shortest segment; 0: the model's own (2 for mean_variance, else 1)
  std::size_t max_changepoints = std::numeric_limits<std::size_t>::max();  // binary and wild binary segmentation
};

namespace changepoint_detail {

inline Segmentation describe(const SegmentCost& cost, Segmentation s) {
  for (auto& g : s.segments) g = cost.segment(g.begin, g.end);
  s.variance = cost.variance();
  return s;
}

inline PenaltyTerms terms(const SegmentCost& cost, const ChangepointOptions& o) {
  return penalty_terms(o.penalty, cost.size(), cost.parameters(), o.beta);
}

}  // namespace changepoint_detail

inline Segmentation optimal_partitioning(std::span<const double> y, const ChangepointOptions& o = {}) {
  const SegmentCost cost(y, o.cost);
  return changepoint_detail::describe(cost, optimal_partitioning(cost, changepoint_detail::terms(cost, o), changepoint_detail::effective_min_size(o.min_size, cost.min_size())));
}

inline Segmentation pelt(std::span<const double> y, const ChangepointOptions& o = {}) {
  const SegmentCost cost(y, o.cost);
  return changepoint_detail::describe(cost, pelt(cost, changepoint_detail::terms(cost, o), changepoint_detail::effective_min_size(o.min_size, cost.min_size())));
}

inline Segmentation binary_segmentation(std::span<const double> y, const ChangepointOptions& o = {}) {
  const SegmentCost cost(y, o.cost);
  return changepoint_detail::describe(cost, binary_segmentation(cost, changepoint_detail::terms(cost, o), changepoint_detail::effective_min_size(o.min_size, cost.min_size()), o.max_changepoints));
}

inline Segmentation wild_binary_segmentation(std::span<const double> y, const ChangepointOptions& o = {}, const WbsOptions& w = {}) {
  const SegmentCost cost(y, o.cost);
  return changepoint_detail::describe(cost, wild_binary_segmentation(cost, changepoint_detail::terms(cost, o), changepoint_detail::effective_min_size(o.min_size, cost.min_size()), w, o.max_changepoints));
}

// Any segmentation of y, given its change points (ascending, in (0, n)), with its segments, cost and
// objective under the options' model and penalty.
inline Segmentation describe_segmentation(std::span<const double> y, std::span<const std::size_t> changepoints, const ChangepointOptions& o = {}) {
  const SegmentCost cost(y, o.cost);
  for (std::size_t k = 0; k < changepoints.size(); k++)
    changepoint_detail::require(changepoints[k] > 0 && changepoints[k] < y.size() && (k == 0 || changepoints[k] > changepoints[k - 1]), "change points ascending in (0, n)");
  return changepoint_detail::describe(cost, changepoint_detail::finish(cost, std::vector<std::size_t>(changepoints.begin(), changepoints.end()), changepoint_detail::terms(cost, o)));
}

// ---------------------------------------------------------------------------------------------
// Circular binary segmentation.

struct CbsOptions {
  double alpha = 0.01;               // the level of each split's permutation test
  std::size_t permutations = 10000;  // B
  double eta = 0.05;                 // the early-rejection boundary's error; 0: no early rejection
  std::size_t min_size = 2;          // no segment shorter (DNAcopy's min.width)
  std::uint64_t seed = 1;            // of the permutation streams
  double prune = 0;                  // γ > 0: prune by the sum of squares (Olshen et al. 2004)
  double undo_sd = 0;                // > 0: undo splits whose neighbouring means differ by < undo_sd σ̂
  double variance = 0;               // σ² for the segments' costs and undo_sd; ≤ 0: robust_variance(y)
};

// The arc of a segment that maximises |Z|.
struct ArcStatistic {
  std::size_t begin = 0, end = 0;  // the arc's observations [begin, end); begin = 0 or end = m: one change point
  double z = 0;                    // Z = (S̃_end − S̃_begin)/√(k(m − k)/m), in the data's units
  double t = 0;                    // the two-sample t statistic of the arc against the rest (the sign of z)
  std::size_t evaluations = 0;     // pairs of index ranges bounded
};

namespace changepoint_detail {

// Branch and bound over pairs of index ranges of the centred partial sums S̃_0..S̃_m.
class ArcSearch {
 public:
  ArcSearch(std::span<const double> S, std::size_t w) : S_(S), m_(S.size() - 1), w_(w) {
    nodes_.reserve(2 * (m_ + 1) / leaf + 8);
    root_ = build(0, m_);
  }

  // The maximum of |Z| over the admissible arcs (or, with threshold > 0, whether some arc reaches it:
  // the search stops at the first that does; the arc returned is then that one).
  ArcStatistic run(double threshold = 0) {
    ArcStatistic best;
    double top = -1;
    auto consider = [&](std::size_t i, std::size_t j) {
      const double z = (S_[j] - S_[i]) / denominator(j - i);
      if (std::fabs(z) > top) {
        top = std::fabs(z);
        best.begin = i;
        best.end = j;
        best.z = z;
      }
    };
    // A first lower bound: the single change points (i = 0 or j = m), O(m).
    for (std::size_t j = w_; j + w_ <= m_; j++) {
      consider(0, j);
      consider(j, m_);
    }
    if (threshold > 0 && top >= threshold) return finish(best, top);
    struct Pair {
      double bound;
      std::size_t a, b;
      bool operator<(const Pair& o) const { return bound < o.bound; }
    };
    // Exact: a pair is worth refining while its bound exceeds the best value found. With a threshold:
    // while its bound reaches the threshold. The heap is best-first, so the first pair that is not
    // worth refining ends the search.
    const bool exact = !(threshold > 0);
    auto worth = [&](double bd) { return exact ? bd > top : bd >= threshold; };
    std::priority_queue<Pair> heap;
    auto push = [&](std::size_t a, std::size_t b) {
      evaluations_++;
      const double bd = bound(a, b);
      if (worth(bd)) heap.push({bd, a, b});
    };
    push(root_, root_);
    while (!heap.empty()) {
      const Pair p = heap.top();
      heap.pop();
      if (!worth(p.bound)) break;
      const Node &A = nodes_[p.a], &B = nodes_[p.b];
      if (A.leaf && B.leaf) {
        for (std::size_t i = A.lo; i <= A.hi; i++) {
          if (!start_ok(i)) continue;
          for (std::size_t j = std::max(B.lo, i + 1); j <= B.hi; j++) {
            if (!end_ok(j)) continue;
            const std::size_t k = j - i;
            if (k < w_ || k + w_ > m_) continue;
            consider(i, j);
          }
        }
        if (threshold > 0 && top >= threshold) return finish(best, top);
        continue;
      }
      if (p.a == p.b) {
        push(A.left, A.left);
        push(A.left, A.right);
        push(A.right, A.right);
      } else if (!A.leaf && (B.leaf || A.hi - A.lo >= B.hi - B.lo)) {
        push(A.left, p.b);
        push(A.right, p.b);
      } else {
        push(p.a, B.left);
        push(p.a, B.right);
      }
    }
    return finish(best, top);
  }

  std::size_t evaluations() const { return evaluations_; }

 private:
  static constexpr std::size_t leaf = 8;
  struct Node {
    std::size_t lo, hi, left = 0, right = 0;
    bool leaf = true;
    double min_i, max_i, min_j, max_j;  // extremes of S̃ over the admissible starts and ends in [lo, hi]
  };

  bool start_ok(std::size_t i) const { return i == 0 || (i >= w_ && i + w_ <= m_); }
  bool end_ok(std::size_t j) const { return j == m_ || (j >= w_ && j + w_ <= m_); }
  double denominator(std::size_t k) const {
    const double kk = static_cast<double>(k), mm = static_cast<double>(m_);
    return std::sqrt(kk * (mm - kk) / mm);
  }

  std::size_t build(std::size_t lo, std::size_t hi) {
    const std::size_t id = nodes_.size();
    nodes_.push_back({});
    Node nd;
    nd.lo = lo;
    nd.hi = hi;
    const double inf = std::numeric_limits<double>::infinity();
    nd.min_i = nd.min_j = inf;
    nd.max_i = nd.max_j = -inf;
    if (hi - lo + 1 <= leaf) {
      nd.leaf = true;
      for (std::size_t x = lo; x <= hi; x++) {
        if (start_ok(x)) {
          nd.min_i = std::min(nd.min_i, S_[x]);
          nd.max_i = std::max(nd.max_i, S_[x]);
        }
        if (end_ok(x)) {
          nd.min_j = std::min(nd.min_j, S_[x]);
          nd.max_j = std::max(nd.max_j, S_[x]);
        }
      }
    } else {
      nd.leaf = false;
      const std::size_t mid = lo + (hi - lo) / 2;
      nd.left = build(lo, mid);
      nd.right = build(mid + 1, hi);
      const Node &L = nodes_[nd.left], &R = nodes_[nd.right];
      nd.min_i = std::min(L.min_i, R.min_i);
      nd.max_i = std::max(L.max_i, R.max_i);
      nd.min_j = std::min(L.min_j, R.min_j);
      nd.max_j = std::max(L.max_j, R.max_j);
    }
    nodes_[id] = nd;
    return id;
  }

  // An upper bound of |Z| over i in A, j in B, i < j, admissible.
  double bound(std::size_t a, std::size_t b) const {
    const Node &A = nodes_[a], &B = nodes_[b];
    if (!(A.max_i >= A.min_i) || !(B.max_j >= B.min_j)) return -1;  // no admissible start or end
    std::size_t klo, khi;
    if (a == b) {
      klo = 1;
      khi = A.hi - A.lo;
    } else {
      klo = B.lo - A.hi;
      khi = B.hi - A.lo;
    }
    klo = std::max(klo, w_);
    khi = std::min(khi, m_ - w_);
    if (klo > khi) return -1;
    const double num = std::max(B.max_j - A.min_i, A.max_i - B.min_j);
    return num / std::min(denominator(klo), denominator(khi));
  }

  ArcStatistic finish(ArcStatistic best, double top) const {
    best.evaluations = evaluations_;
    if (top < 0) best.z = 0;
    return best;
  }

  std::span<const double> S_;
  std::size_t m_, w_;
  std::vector<Node> nodes_;
  std::size_t root_ = 0;
  std::size_t evaluations_ = 0;
};

// The centred partial sums S̃_0 = 0, …, S̃_m of y (compensated), and the total sum of squares.
inline std::vector<double> centred_partial_sums(std::span<const double> y, double& tss) {
  const std::size_t m = y.size();
  const double mean = descriptive::mean(y);
  auto S = prefix(m, [&](std::size_t i) { return y[i] - mean; });
  // Remove the residual drift so that S̃_m = 0 exactly.
  const double drift = S[m];
  for (std::size_t t = 0; t <= m; t++) S[t] -= drift * static_cast<double>(t) / static_cast<double>(m);
  S[m] = 0;
  const auto Q = prefix(m, [&](std::size_t i) { return (y[i] - mean) * (y[i] - mean); });
  tss = Q[m];
  return S;
}

inline double t_from_z(double z, double tss, std::size_t m) {
  const double rest = std::max(tss - z * z, 0.0);
  if (rest <= 0) return std::copysign(std::numeric_limits<double>::infinity(), z);
  return z / std::sqrt(rest / static_cast<double>(m - 2));
}

// The early-rejection boundary: b_i for i = 1, 2, …, the least b with P(H ≤ i − 1) ≤ η 2^{−i}, H
// hypergeometric (population B, r* marked, b drawn).
inline std::vector<std::size_t> cbs_boundary(std::size_t B, std::size_t r_star, double eta) {
  std::vector<std::size_t> b;
  if (!(eta > 0) || r_star == 0 || r_star > B) return b;
  using calx::special_functions::log_choose;
  const double Bd = static_cast<double>(B), rd = static_cast<double>(r_star);
  auto cdf = [&](std::size_t x, std::size_t draws) {  // P(H ≤ x)
    const double dd = static_cast<double>(draws), total = log_choose(Bd, dd);
    double p = 0;
    for (std::size_t h = 0; h <= std::min(x, std::min(draws, r_star)); h++) {
      const double hd = static_cast<double>(h);
      if (dd - hd > Bd - rd) continue;
      p += std::exp(log_choose(rd, hd) + log_choose(Bd - rd, dd - hd) - total);
    }
    return p;
  };
  std::size_t from = 1;
  for (std::size_t i = 1; i <= r_star; i++) {
    const double level = eta * std::ldexp(1.0, -static_cast<int>(i));
    if (level < 1e-15) break;
    std::size_t lo = std::max(from, i), hi = B;
    if (cdf(i - 1, hi) > level) break;
    while (lo < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (cdf(i - 1, mid) <= level) hi = mid;
      else lo = mid + 1;
    }
    if (lo >= B) break;
    b.push_back(lo);
    from = lo;
  }
  return b;
}

// The options of the mean model with a given σ².
inline CostOptions mean_model(double sigma2) {
  CostOptions o;
  o.variance = sigma2;
  return o;
}

}  // namespace changepoint_detail

// The maximal circular statistic of y (a whole segment), exactly, over arcs that leave no segment
// shorter than min_size.
inline ArcStatistic cbs_statistic(std::span<const double> y, std::size_t min_size = 2) {
  const std::size_t m = y.size();
  ArcStatistic a;
  const std::size_t w = std::max<std::size_t>(min_size, 1);
  if (m < 3 || m < 2 * w) return a;
  double tss = 0;
  const auto S = changepoint_detail::centred_partial_sums(y, tss);
  changepoint_detail::ArcSearch search(S, w);
  a = search.run();
  a.t = changepoint_detail::t_from_z(a.z, tss, m);
  return a;
}

// Circular binary segmentation of y with permutation p-values.
inline Segmentation circular_binary_segmentation(std::span<const double> y, const CbsOptions& o = {}) {
  const std::size_t n = y.size();
  changepoint_detail::require(o.alpha > 0 && o.alpha < 1, "CBS: 0 < alpha < 1");
  changepoint_detail::require(o.permutations >= 1, "CBS: at least one permutation");
  const std::size_t w = std::max<std::size_t>(o.min_size, 1);
  for (std::size_t i = 0; i < n; i++) changepoint_detail::require(std::isfinite(y[i]), "finite observations");
  const double sigma2 = o.variance > 0 ? o.variance : robust_variance(y);
  const SegmentCost cost(y, changepoint_detail::mean_model(sigma2));

  const std::size_t B = o.permutations;
  const double limit = o.alpha * static_cast<double>(B + 1);  // reject iff 1 + R ≤ α(B + 1)
  const bool testable = limit >= 1;
  const std::size_t r_star = testable ? static_cast<std::size_t>(std::floor(limit)) : 0;  // the least R not rejected
  const auto boundary = testable ? changepoint_detail::cbs_boundary(B, r_star, o.eta) : std::vector<std::size_t>{};

  struct Found {
    std::size_t at;
    double score, p;
  };
  std::vector<Found> found;
  std::size_t evaluations = 0;
  std::vector<std::pair<std::size_t, std::size_t>> stack{{0, n}};
  std::vector<double> perm;
  while (!stack.empty() && testable) {
    const auto [b, e] = stack.back();
    stack.pop_back();
    const std::size_t m = e - b;
    if (m < 3 || m < 2 * w) continue;
    const std::span<const double> seg = y.subspan(b, m);
    double tss = 0;
    const auto S = changepoint_detail::centred_partial_sums(seg, tss);
    if (!(tss > 0)) continue;
    changepoint_detail::ArcSearch search(S, w);
    ArcStatistic arc = search.run();
    evaluations += search.evaluations();
    const double observed = std::fabs(arc.z);
    if (!(observed > 0)) continue;
    arc.t = changepoint_detail::t_from_z(arc.z, tss, m);
    // The permutation test, curtailed and with early rejection.
    calx::random::Stream rng(o.seed, static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(e), 0x43425321u);  // "CBS!"
    perm.assign(seg.begin(), seg.end());
    std::size_t R = 0, next = 0;
    bool significant = false, decided = false;
    double p = 1;
    const double reach = observed * (1 - 1e-12);
    for (std::size_t k = 1; k <= B; k++) {
      for (std::size_t i = m - 1; i > 0; i--) std::swap(perm[i], perm[rng.below(i + 1)]);
      double tp = 0;
      const auto Sp = changepoint_detail::centred_partial_sums(perm, tp);
      changepoint_detail::ArcSearch ps(Sp, w);
      const ArcStatistic hit = ps.run(reach);
      evaluations += ps.evaluations();
      if (std::fabs(hit.z) >= reach) R++;
      if (R >= r_star) {
        p = static_cast<double>(R) / static_cast<double>(k);  // Besag and Clifford's estimate
        decided = true;
        break;
      }
      while (next < boundary.size() && boundary[next] == k) {
        if (R < next + 1) {
          significant = true;
          decided = true;
          p = static_cast<double>(1 + R) / static_cast<double>(1 + k);
          break;
        }
        next++;
      }
      if (decided) break;
    }
    if (!decided) {
      p = static_cast<double>(1 + R) / static_cast<double>(1 + B);
      significant = p <= o.alpha;
    }
    if (!significant) continue;
    const std::size_t i = b + arc.begin, j = b + arc.end;
    const double score = std::fabs(arc.t);
    if (arc.begin > 0) found.push_back({i, score, p});
    if (arc.end < m) found.push_back({j, score, p});
    // Recurse on the pieces (pushed so that the leftmost is tested first).
    if (arc.end < m) stack.push_back({j, e});
    stack.push_back({arc.begin > 0 ? i : b, j});
    if (arc.begin > 0) stack.push_back({b, i});
  }
  std::sort(found.begin(), found.end(), [](const Found& a, const Found& c) { return a.at < c.at; });

  // Pruning by the sum of squares: the fewest change points whose best subset keeps SS_j/SS_K − 1 < γ.
  if (o.prune > 0 && !found.empty()) {
    const std::size_t K = found.size();
    std::vector<std::size_t> pos{0};
    for (const auto& f : found) pos.push_back(f.at);
    pos.push_back(n);
    const SegmentCost unit(y, changepoint_detail::mean_model(1.0));
    const double inf = std::numeric_limits<double>::infinity();
    // D[j][q]: the least RSS of y[0, pos[q]) cut at j of the found points, the last at pos[q] (q ≥ 1 interior).
    std::vector<std::vector<double>> D(K + 1, std::vector<double>(K + 2, inf));
    std::vector<std::vector<std::size_t>> from(K + 1, std::vector<std::size_t>(K + 2, 0));
    for (std::size_t q = 1; q <= K + 1; q++) D[0][q] = unit(0, pos[q]);
    for (std::size_t j = 1; j <= K; j++)
      for (std::size_t q = j + 1; q <= K + 1; q++)
        for (std::size_t r = j; r < q; r++) {
          if (!(D[j - 1][r] < inf)) continue;
          const double v = D[j - 1][r] + unit(pos[r], pos[q]);
          if (v < D[j][q]) {
            D[j][q] = v;
            from[j][q] = r;
          }
        }
    const double full = D[K][K + 1];
    std::size_t choose = K;
    for (std::size_t j = 0; j < K; j++)
      if (D[j][K + 1] / full - 1 < o.prune) {
        choose = j;
        break;
      }
    std::vector<Found> kept;
    for (std::size_t j = choose, q = K + 1; j > 0; j--) {
      const std::size_t r = from[j][q];
      kept.push_back(found[r - 1]);
      q = r;
    }
    std::reverse(kept.begin(), kept.end());
    found = std::move(kept);
  }
  // Undoing splits whose neighbouring means differ by less than undo_sd σ̂, the closest first.
  if (o.undo_sd > 0) {
    const double limit_sd = o.undo_sd * std::sqrt(sigma2);
    for (;;) {
      if (found.empty()) break;
      std::size_t worst = found.size();
      double gap = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < found.size(); k++) {
        const std::size_t b0 = k == 0 ? 0 : found[k - 1].at, t = found[k].at, e0 = k + 1 < found.size() ? found[k + 1].at : n;
        const double d = std::fabs(cost.segment(b0, t).mean - cost.segment(t, e0).mean);
        if (d < gap) {
          gap = d;
          worst = k;
        }
      }
      if (!(gap < limit_sd)) break;
      found.erase(found.begin() + static_cast<std::ptrdiff_t>(worst));
    }
  }
  std::vector<std::size_t> cps;
  std::vector<double> scores, ps;
  for (const auto& f : found) {
    cps.push_back(f.at);
    scores.push_back(f.score);
    ps.push_back(f.p);
  }
  Segmentation out = changepoint_detail::describe(cost, changepoint_detail::finish(cost, std::move(cps), PenaltyTerms{0, 0, n}));
  out.scores = std::move(scores);
  out.p_values = std::move(ps);
  out.evaluations = evaluations;
  return out;
}

}  // namespace tyche::time_series
