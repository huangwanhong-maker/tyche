// The conditional tables of a network learnt from records with missing values, by expectation–maximisation
// (decision 0023).
//
// A record gives each variable a state or leaves it missing. With θ_ijk = P(X_i = k | Pa_i = j), the log-likelihood
// of records e_1, …, e_N is ℓ(θ) = Σ_r log P(e_r | θ), the missing values summed out. EM (Dempster, Laird and Rubin
// 1977, J. R. Statist. Soc. B 39, 1–38; in networks Lauritzen 1995, Computational Statistics & Data Analysis 19,
// 191–201) alternates
//   E-step   the expected counts N̄_ijk = Σ_r P(X_i = k, Pa_i = j | e_r, θ), each family's posterior computed
//            exactly by variable elimination (for a record, one elimination over its missing variables when their
//            joint table has at most joint_cells cells, 2¹⁶ by default, else one for each family with a missing
//            member);
//   M-step   θ_ijk = (N̄_ijk + a_ijk) / Σ_k (N̄_ijk + a_ijk).
// With pseudo-counts a_ijk ≥ 0 the M-step is the mode of the posterior under independent Dirichlet priors with
// parameters a + 1 on the rows (maximum a posteriori; Heckerman 1995, Microsoft Research MSR-TR-95-06, for the
// conjugacy), and the objective is the log-posterior up to its constant, ℓ(θ) + Σ a_ijk log θ_ijk. Either way the
// objective never decreases from one iteration to the next (Dempster, Laird and Rubin 1977, Theorem 1); EM reaches a
// stationary point, not necessarily the global maximum, and missing values are taken to be missing at random
// (Rubin 1976, Biometrika 63, 581–592). A row with no expected count and no pseudo-count keeps its previous values.
//
// The objective is recorded at the starting tables and after each M-step; EM stops when an iteration raises it by
// less than tolerance × (1 + |objective|), or after max_iterations M-steps. Identical records are counted once
// with their multiplicity.

#pragma once

#include <tyche/graphical/common.hpp>
#include <tyche/graphical/elimination.hpp>
#include <tyche/graphical/factor.hpp>
#include <tyche/graphical/network.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tyche::graphical {

using Record = std::vector<std::optional<std::size_t>>;  // one entry per variable; nullopt when missing

struct EmOptions {
  std::size_t max_iterations = 200;
  double tolerance = 1e-10;                        // the least relative gain that counts as progress
  double pseudo_count = 0;                         // a ≥ 0 added to every cell
  std::vector<std::vector<double>> pseudo_counts;  // when not empty: by variable, one per cell of its table (overrides pseudo_count)
  std::size_t joint_cells = std::size_t{1} << 16;  // the largest joint table of a record's missing values formed at once; above it, one elimination per family
  EliminationOptions elimination;
};

struct EmResult {
  Network network;                          // the network with the tables learnt
  std::vector<double> objective;            // at the start and after each M-step: ℓ(θ), plus Σ a log θ with pseudo-counts
  std::size_t iterations = 0;               // M-steps made (objective.size() − 1)
  bool converged = false;
  std::vector<std::vector<double>> counts;  // the expected counts N̄ at the tables returned, by variable and cell
};

namespace detail {

inline std::map<Record, double> group_records(const Network& net, const std::vector<Record>& records) {
  std::map<Record, double> g;
  for (std::size_t r = 0; r < records.size(); r++) {
    require(records[r].size() == net.size(), "records: record " + std::to_string(r) + " does not give one entry per variable");
    for (std::size_t v = 0; v < net.size(); v++)
      if (records[r][v]) require(*records[r][v] < net.cardinality(v), "records: record " + std::to_string(r) + " gives a state out of range");
    g[records[r]] += 1;
  }
  return g;
}

inline std::size_t cell_of(const Network& net, std::size_t v, const std::vector<std::size_t>& x) {
  std::size_t c = 0;
  for (std::size_t u : net.tables[v].scope) c = c * net.cardinality(u) + x[u];
  return c;
}

// The expected counts at the network's tables, and the log-likelihood.
inline double expected_counts(const Network& net, const std::map<Record, double>& groups, const EliminationOptions& o, std::size_t joint_cells,
                              std::vector<std::vector<double>>& counts) {
  const std::size_t n = net.size();
  counts.assign(n, {});
  for (std::size_t v = 0; v < n; v++) counts[v].assign(net.tables[v].size(), 0.0);
  double ll = 0;
  for (const auto& [rec, mult] : groups) {
    Evidence e;
    std::vector<std::size_t> missing, x(n, 0);
    for (std::size_t v = 0; v < n; v++) {
      if (rec[v]) {
        e.observed[v] = *rec[v];
        x[v] = *rec[v];
      } else {
        missing.push_back(v);
      }
    }
    std::vector<char> complete(n, 1);  // the family of v has no missing member
    for (std::size_t v = 0; v < n; v++)
      for (std::size_t u : net.tables[v].scope)
        if (!rec[u]) complete[v] = 0;
    double logp;
    if (missing.empty()) {
      const double p = net.probability(x);
      require(p > 0, "em: a record has probability zero under the tables");
      logp = std::log(p);
    } else {
      std::size_t cells = 1;
      bool small = true;
      for (std::size_t v : missing) {
        cells *= net.cardinality(v);
        if (cells > joint_cells) {
          small = false;
          break;
        }
      }
      if (small) {
        const auto r = eliminate(network_factors(net, e, missing, true), missing, o);
        const double z = total(r.factor);
        require(z > 0, "em: a record has probability zero under the tables");
        logp = std::log(z) + r.log_scale;
        std::vector<std::size_t> a(missing.size(), 0);
        for (std::size_t i = 0; i < r.factor.size(); i++) {
          const double w = r.factor.values[i] / z;
          if (w > 0) {
            for (std::size_t k = 0; k < missing.size(); k++) x[missing[k]] = a[k];
            for (std::size_t v = 0; v < n; v++)
              if (!complete[v]) counts[v][cell_of(net, v, x)] += mult * w;
          }
          for (std::size_t k = missing.size(); k-- > 0;) {
            if (++a[k] < r.factor.cards[k]) break;
            a[k] = 0;
          }
        }
      } else {
        logp = log_evidence_probability(net, e, o);
        require(std::isfinite(logp), "em: a record has probability zero under the tables");
        for (std::size_t v = 0; v < n; v++) {
          if (complete[v]) continue;
          std::vector<std::size_t> q;
          for (std::size_t u : net.tables[v].scope)
            if (!rec[u]) q.push_back(u);
          const Factor post = joint(net, q, e, o);
          std::vector<std::size_t> a(q.size(), 0);
          for (std::size_t i = 0; i < post.size(); i++) {
            for (std::size_t k = 0; k < q.size(); k++) x[q[k]] = a[k];
            counts[v][cell_of(net, v, x)] += mult * post.values[i];
            for (std::size_t k = q.size(); k-- > 0;) {
              if (++a[k] < post.cards[k]) break;
              a[k] = 0;
            }
          }
        }
      }
    }
    for (std::size_t v = 0; v < n; v++)
      if (complete[v]) counts[v][cell_of(net, v, x)] += mult;
    ll += mult * logp;
  }
  return ll;
}

inline double log_prior(const Network& net, const std::vector<std::vector<double>>& a) {
  double s = 0;
  for (std::size_t v = 0; v < net.size(); v++)
    for (std::size_t c = 0; c < a[v].size(); c++)
      if (a[v][c] > 0) s += net.tables[v].values[c] > 0 ? a[v][c] * std::log(net.tables[v].values[c]) : -std::numeric_limits<double>::infinity();
  return s;
}

}  // namespace detail

// Σ_r log P(e_r): the log-likelihood of the records under the network's tables (−∞ when a record is impossible).
inline double log_likelihood(const Network& net, const std::vector<Record>& records, const EliminationOptions& o = {}) {
  double ll = 0;
  for (const auto& [rec, mult] : detail::group_records(net, records)) {
    Evidence e;
    for (std::size_t v = 0; v < net.size(); v++)
      if (rec[v]) e.observed[v] = *rec[v];
    ll += mult * log_evidence_probability(net, e, o);
  }
  return ll;
}

inline EmResult em(const Network& start, const std::vector<Record>& records, const EmOptions& o = {}) {
  const std::size_t n = start.size();
  detail::require(!records.empty(), "em: no records");
  detail::require(std::isfinite(o.pseudo_count) && o.pseudo_count >= 0, "em: the pseudo-count is finite and non-negative");
  std::vector<std::vector<double>> a(n);
  if (o.pseudo_counts.empty()) {
    for (std::size_t v = 0; v < n; v++) a[v].assign(start.tables[v].size(), o.pseudo_count);
  } else {
    detail::require(o.pseudo_counts.size() == n, "em: pseudo-counts for every variable");
    for (std::size_t v = 0; v < n; v++) {
      detail::require(o.pseudo_counts[v].size() == start.tables[v].size(), "em: one pseudo-count per cell of the table of '" + start.variables[v].name + "'");
      for (double x : o.pseudo_counts[v]) detail::require(std::isfinite(x) && x >= 0, "em: pseudo-counts are finite and non-negative");
      a[v] = o.pseudo_counts[v];
    }
  }
  const auto groups = detail::group_records(start, records);
  EmResult r;
  r.network = start;
  for (std::size_t it = 0;; it++) {
    const double ll = detail::expected_counts(r.network, groups, o.elimination, o.joint_cells, r.counts);
    const double obj = ll + detail::log_prior(r.network, a);
    r.objective.push_back(obj);
    if (it > 0 && obj - r.objective[it - 1] < o.tolerance * (1 + std::fabs(obj))) {
      r.converged = true;
      break;
    }
    if (it == o.max_iterations) break;
    for (std::size_t v = 0; v < n; v++) {
      const std::size_t k = r.network.cardinality(v), rows = r.network.tables[v].size() / k;
      auto& theta = r.network.tables[v].values;
      for (std::size_t j = 0; j < rows; j++) {
        double s = 0;
        for (std::size_t c = 0; c < k; c++) s += r.counts[v][j * k + c] + a[v][j * k + c];
        if (!(s > 0)) continue;
        for (std::size_t c = 0; c < k; c++) theta[j * k + c] = (r.counts[v][j * k + c] + a[v][j * k + c]) / s;
      }
    }
  }
  r.iterations = r.objective.size() - 1;
  return r;
}

}  // namespace tyche::graphical
