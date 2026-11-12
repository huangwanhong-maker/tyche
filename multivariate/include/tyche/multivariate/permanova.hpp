// Do groups of samples differ in composition, or in spread? Tests on a matrix of dissimilarities.
//
// PERMANOVA (Anderson 2001, Austral Ecol. 26: 32–46; for unequal groups McArdle and Anderson 2001,
// Ecology 82: 290–297). N samples in a groups of sizes n_g, with dissimilarities d_ij:
//
//   SS_T = (1/N) Σ_{i<j} d_ij²,   SS_W = Σ_g (1/n_g) Σ_{i<j in g} d_ij²,   SS_A = SS_T − SS_W,
//   F = [SS_A/(a − 1)] / [SS_W/(N − a)],   R² = SS_A/SS_T.
//
// For Euclidean distances SS_W is the within-group sum of squared deviations from the group
// centroids (Huygens's theorem: Σ_{i<j in g} ‖x_i − x_j‖² = n_g Σ_{i in g} ‖x_i − x̄_g‖²), and F is the
// classical one-way ANOVA F (the trace F in several dimensions); for any other dissimilarity the
// same sums are taken in the space of principal coordinates, imaginary axes included. The null
// hypothesis is that the groups' labels are exchangeable; the p-value is (b + 1)/(B + 1), b of B
// random relabellings with F* ≥ F (Phipson and Smyth 2010, Stat. Appl. Genet. Mol. Biol. 9, art. 39),
// or, by `permanova_exact`, the share of all N!/Π n_g! distinct relabellings with F* ≥ F, the exact
// permutation p-value. Ties are F* ≥ F(1 − 10⁻¹⁰). F is sensitive to differences in spread as well as
// in location, chiefly in unbalanced designs (Anderson and Walsh 2013, Ecol. Monogr. 83: 557–574),
// which is why the test of dispersions comes with it.
//
// PERMDISP, the homogeneity of multivariate dispersions (Anderson 2006, Biometrics 62: 245–253): each
// sample's distance z_i to its group's centroid in the space of principal coordinates, from the
// dissimilarities alone,
//
//   z_i² = (1/n_g) Σ_{j in g} d_ij² − (1/(2 n_g²)) Σ_{j,k in g} d_jk²,
//
// an identity of the Gram form that holds in any pseudo-Euclidean embedding; where the imaginary
// axes make it negative, |z_i²| is used (as vegan's betadisper does). F is the one-way ANOVA F of the
// z_i, and its p-value is by permuting the residuals z_i − z̄_g among the samples (Anderson 2006).
//
// Relabellings are drawn by Fisher–Yates shuffles on calx streams under a seed (PERMANOVA's on the
// identity (0x7065, 0, 0), PERMDISP's on (0x7065, 1, 0)), so a run repeats itself exactly.

#pragma once

#include <tyche/multivariate/dissimilarity.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

namespace tyche::multivariate {

namespace detail {

// Labels made 0, …, a − 1 in the order of the distinct values; the values themselves.
inline std::vector<std::size_t> compact(std::span<const std::size_t> group, std::vector<std::size_t>& values, const char* where) {
  std::map<std::size_t, std::size_t> index;
  for (std::size_t g : group) index.emplace(g, 0);
  values.clear();
  for (auto& [v, i] : index) i = values.size(), values.push_back(v);
  std::vector<std::size_t> out;
  out.reserve(group.size());
  for (std::size_t g : group) out.push_back(index[g]);
  check(values.size() >= 2, where, "at least two groups");
  check(group.size() > values.size(), where, "more samples than groups");
  return out;
}

inline double pseudo_f(const std::vector<double>& d2, std::size_t N, const std::vector<std::size_t>& label, const std::vector<double>& size, double ss_total, double& ss_within) {
  ss_within = 0;
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = i + 1; j < N; j++)
      if (label[i] == label[j]) ss_within += d2[i * N + j] / size[label[i]];
  const double a = static_cast<double>(size.size()), n = static_cast<double>(N);
  return ((ss_total - ss_within) / (a - 1)) / (ss_within / (n - a));
}

// The one-way ANOVA F of values z in groups.
inline double anova_f(std::span<const double> z, const std::vector<std::size_t>& label, std::size_t groups) {
  std::vector<double> sum(groups, 0.0), count(groups, 0.0);
  double all = 0;
  for (std::size_t i = 0; i < z.size(); i++) sum[label[i]] += z[i], count[label[i]]++, all += z[i];
  const double mean = all / static_cast<double>(z.size());
  double among = 0, within = 0;
  for (std::size_t g = 0; g < groups; g++) among += count[g] * (sum[g] / count[g] - mean) * (sum[g] / count[g] - mean);
  for (std::size_t i = 0; i < z.size(); i++) {
    const double r = z[i] - sum[label[i]] / count[label[i]];
    within += r * r;
  }
  return (among / static_cast<double>(groups - 1)) / (within / static_cast<double>(z.size() - groups));
}

template <class T>
void shuffle(std::vector<T>& v, calx::random::Stream& rng) {
  for (std::size_t i = v.size(); i > 1; i--) std::swap(v[i - 1], v[rng.below(i)]);
}

}  // namespace detail

struct Permanova {
  std::size_t n = 0, groups = 0, df_among = 0, df_within = 0;
  double ss_total = 0, ss_within = 0, ss_among = 0;
  double F = 0, r_squared = 0;
  double p_value = 1;
  std::size_t permutations = 0;  // relabellings drawn, or enumerated when exact
  bool exact = false;
};

namespace detail {

struct PermanovaSetup {
  std::size_t N = 0;
  std::vector<double> d2, size;
  std::vector<std::size_t> label, values;
  double ss_total = 0;
  Permanova base;
};

inline PermanovaSetup setup(const calx::linalg::Matrix& d, std::span<const std::size_t> group, const char* where) {
  require_dissimilarities(d, where);
  check(group.size() == d.n, where, "a group for each sample");
  PermanovaSetup s;
  s.N = d.n;
  s.label = compact(group, s.values, where);
  s.size.assign(s.values.size(), 0.0);
  for (std::size_t g : s.label) s.size[g]++;
  s.d2.assign(s.N * s.N, 0.0);
  for (std::size_t i = 0; i < s.N; i++)
    for (std::size_t j = 0; j < s.N; j++) s.d2[i * s.N + j] = d(i, j) * d(i, j);
  for (std::size_t i = 0; i < s.N; i++)
    for (std::size_t j = i + 1; j < s.N; j++) s.ss_total += s.d2[i * s.N + j];
  s.ss_total /= static_cast<double>(s.N);
  Permanova& r = s.base;
  r.n = s.N;
  r.groups = s.values.size();
  r.df_among = r.groups - 1;
  r.df_within = s.N - r.groups;
  r.ss_total = s.ss_total;
  r.F = pseudo_f(s.d2, s.N, s.label, s.size, s.ss_total, r.ss_within);
  r.ss_among = r.ss_total - r.ss_within;
  r.r_squared = r.ss_among / r.ss_total;
  return s;
}

}  // namespace detail

// PERMANOVA with B random relabellings.
inline Permanova permanova(const calx::linalg::Matrix& d, std::span<const std::size_t> group, std::size_t permutations = 999, std::uint64_t seed = 20261112) {
  auto s = detail::setup(d, group, "permanova");
  Permanova r = s.base;
  r.permutations = permutations;
  calx::random::Stream rng(seed, 0x7065, 0, 0);
  auto label = s.label;
  std::size_t b = 0;
  double w;
  for (std::size_t k = 0; k < permutations; k++) {
    detail::shuffle(label, rng);
    if (detail::pseudo_f(s.d2, s.N, label, s.size, s.ss_total, w) >= r.F * (1 - 1e-10)) b++;
  }
  r.p_value = (static_cast<double>(b) + 1) / (static_cast<double>(permutations) + 1);
  return r;
}

// PERMANOVA's exact permutation p-value, over every distinct relabelling (refused beyond `limit`).
inline Permanova permanova_exact(const calx::linalg::Matrix& d, std::span<const std::size_t> group, std::size_t limit = 2000000) {
  auto s = detail::setup(d, group, "permanova_exact");
  Permanova r = s.base;
  // N!/Π n_g!, the number of distinct relabellings.
  double count = 1;
  {
    double placed = 0;
    for (double ng : s.size)
      for (double k = 1; k <= ng; k++) count = count * (++placed) / k;
  }
  detail::check(count <= static_cast<double>(limit), "permanova_exact", "too many relabellings to enumerate");
  auto label = s.label;
  std::sort(label.begin(), label.end());
  std::size_t total = 0, b = 0;
  double w;
  do {
    total++;
    if (detail::pseudo_f(s.d2, s.N, label, s.size, s.ss_total, w) >= r.F * (1 - 1e-10)) b++;
  } while (std::next_permutation(label.begin(), label.end()));
  r.permutations = total;
  r.exact = true;
  r.p_value = static_cast<double>(b) / static_cast<double>(total);
  return r;
}

struct Dispersion {
  std::vector<double> distances;    // z_i, each sample's distance to its group's centroid
  std::vector<std::size_t> labels;  // the distinct group labels, in increasing order
  std::vector<double> group_mean;   // the mean distance in each group, in the order of `labels`
  std::size_t df_among = 0, df_within = 0;
  double F = 0, p_value = 1;
  std::size_t permutations = 0;
};

// Distances to the group centroids in the space of principal coordinates.
inline std::vector<double> centroid_distances(const calx::linalg::Matrix& d, std::span<const std::size_t> group) {
  require_dissimilarities(d, "centroid_distances");
  detail::check(group.size() == d.n, "centroid_distances", "a group for each sample");
  std::map<std::size_t, std::vector<std::size_t>> members;
  for (std::size_t i = 0; i < group.size(); i++) members[group[i]].push_back(i);
  std::vector<double> z(d.n, 0.0);
  for (const auto& [g, idx] : members) {
    const double n = static_cast<double>(idx.size());
    double inner = 0;
    for (std::size_t j : idx)
      for (std::size_t k : idx) inner += d(j, k) * d(j, k);
    for (std::size_t i : idx) {
      double s = 0;
      for (std::size_t j : idx) s += d(i, j) * d(i, j);
      z[i] = std::sqrt(std::fabs(s / n - inner / (2 * n * n)));
    }
  }
  return z;
}

// PERMDISP: the test that the groups' dispersions are equal.
inline Dispersion dispersion(const calx::linalg::Matrix& d, std::span<const std::size_t> group, std::size_t permutations = 999, std::uint64_t seed = 20261112) {
  Dispersion r;
  r.distances = centroid_distances(d, group);
  const auto label = detail::compact(group, r.labels, "dispersion");
  const std::size_t a = r.labels.size(), N = d.n;
  r.df_among = a - 1;
  r.df_within = N - a;
  std::vector<double> sum(a, 0.0), count(a, 0.0);
  for (std::size_t i = 0; i < N; i++) sum[label[i]] += r.distances[i], count[label[i]]++;
  for (std::size_t g = 0; g < a; g++) r.group_mean.push_back(sum[g] / count[g]);
  r.F = detail::anova_f(r.distances, label, a);
  std::vector<double> resid(N);
  for (std::size_t i = 0; i < N; i++) resid[i] = r.distances[i] - r.group_mean[label[i]];
  calx::random::Stream rng(seed, 0x7065, 1, 0);
  std::size_t b = 0;
  for (std::size_t k = 0; k < permutations; k++) {
    detail::shuffle(resid, rng);
    if (detail::anova_f(resid, label, a) >= r.F * (1 - 1e-10)) b++;
  }
  r.permutations = permutations;
  r.p_value = (static_cast<double>(b) + 1) / (static_cast<double>(permutations) + 1);
  return r;
}

}  // namespace tyche::multivariate
