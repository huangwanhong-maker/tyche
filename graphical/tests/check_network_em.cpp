// Checks of tyche::graphical's learning of tables by EM, on records drawn from a seeded random network of six
// binary variables: ancestral sampling against the exact marginals; on complete records, the maximum-likelihood
// tables are the counts' proportions and the maximum a posteriori ones the proportions with the pseudo-counts
// added; with a fifth of the values missing at random, the objective never decreases, the tables learnt explain
// the records at least as well as the true ones and lie within a few standard errors of them, the recorded
// objective is the log-likelihood of the records, the expected counts add up to the number of records, and the
// E-step by families equals the E-step by a record's whole missing part; the refusals of malformed and
// impossible records.

#include <tyche/graphical/graphical.hpp>

#include "brute_force.hpp"

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace tyche::graphical;
using calx::verification::relative;

namespace {

template <class F>
std::string refusal(F f) {
  try {
    f();
  } catch (const Refusal& e) {
    return e.what();
  } catch (...) {
    return "(another exception)";
  }
  return "";
}

double worst_drop(const std::vector<double>& obj) {
  double w = 0;
  for (std::size_t i = 1; i < obj.size(); i++) w = std::min(w, (obj[i] - obj[i - 1]) / (1 + std::fabs(obj[i])));
  return w;
}

Network uniform_start(const Network& net) {
  std::vector<std::vector<double>> t;
  for (std::size_t v = 0; v < net.size(); v++) t.push_back(std::vector<double>(net.tables[v].size(), 1.0 / static_cast<double>(net.cardinality(v))));
  return with_tables(net, t);
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(23, 0x67726170, 4, 0);
  const auto nodes = brute::random_network(rng, 6, 2, 2, 0.15, 0.85);
  const Network truth = network(nodes);
  const std::size_t n = truth.size(), N = 20000;

  std::vector<std::vector<std::size_t>> full;
  for (std::size_t r = 0; r < N; r++) full.push_back(sample(truth, rng));
  {
    const auto m = marginals(truth);
    double z = 0;
    for (std::size_t v = 0; v < n; v++) {
      double f = 0;
      for (const auto& x : full) f += x[v] == 0;
      f /= static_cast<double>(N);
      z = std::max(z, std::fabs(f - m[v][0]) / std::sqrt(m[v][0] * (1 - m[v][0]) / static_cast<double>(N)));
    }
    s.check("sampling: each variable's frequency against its exact marginal, in standard errors", z, 4);
  }

  // The counts of each table's cells in the complete records.
  std::vector<std::vector<double>> counts(n);
  for (std::size_t v = 0; v < n; v++) {
    counts[v].assign(truth.tables[v].size(), 0.0);
    for (const auto& x : full) {
      std::size_t c = 0;
      for (std::size_t u : truth.parents[v]) c = c * 2 + x[u];
      counts[v][c * 2 + x[v]] += 1;
    }
  }
  const Network start = uniform_start(truth);

  {
    std::vector<Record> complete;
    for (const auto& x : full) complete.emplace_back(x.begin(), x.end());
    const auto ml = em(start, complete);
    double d = 0;
    for (std::size_t v = 0; v < n; v++)
      for (std::size_t c = 0; c < counts[v].size(); c++) {
        const std::size_t row = c / 2;
        const double rowsum = counts[v][2 * row] + counts[v][2 * row + 1];
        d = std::max(d, std::fabs(ml.network.tables[v].values[c] - counts[v][c] / rowsum));
      }
    s.check("complete records: the tables learnt are the counts' proportions", d, 1e-14);
    s.require("complete records: EM converges at once", ml.converged && ml.iterations == 2, std::to_string(ml.iterations));
    EmOptions o;
    o.pseudo_count = 2;
    const auto map = em(start, complete, o);
    d = 0;
    for (std::size_t v = 0; v < n; v++)
      for (std::size_t c = 0; c < counts[v].size(); c++) {
        const std::size_t row = c / 2;
        const double rowsum = counts[v][2 * row] + counts[v][2 * row + 1];
        d = std::max(d, std::fabs(map.network.tables[v].values[c] - (counts[v][c] + 2) / (rowsum + 4)));
      }
    s.check("complete records: with pseudo-counts a, the tables are (N + a)/(N_row + k a)", d, 1e-14);
  }

  // A fifth of the values missing completely at random.
  std::vector<Record> records;
  calx::random::Stream hide(23, 0x67726170, 5, 0);
  for (const auto& x : full) {
    Record r(x.begin(), x.end());
    for (auto& v : r)
      if (hide.uniform() < 0.2) v.reset();
    records.push_back(r);
  }
  {
    const auto res = em(start, records);
    s.check("missing values: the log-likelihood never decreases (largest relative drop)", worst_drop(res.objective), 1e-12);
    s.require("missing values: EM converges", res.converged, std::to_string(res.iterations) + " iterations");
    s.check("missing values: the recorded objective is the records' log-likelihood", relative(res.objective.back(), log_likelihood(res.network, records)), 1e-12);
    const double lt = log_likelihood(truth, records);
    s.require("missing values: the tables learnt explain the records at least as well as the true ones", res.objective.back() >= lt,
              std::to_string(res.objective.back()) + " against " + std::to_string(lt));
    double z = 0;
    for (std::size_t v = 0; v < n; v++) {
      const double seen = std::pow(0.8, static_cast<double>(truth.tables[v].scope.size()));
      for (std::size_t c = 0; c < counts[v].size(); c++) {
        const std::size_t row = c / 2;
        const double nrow = counts[v][2 * row] + counts[v][2 * row + 1], th = truth.tables[v].values[c];
        z = std::max(z, std::fabs(res.network.tables[v].values[c] - th) / std::sqrt(th * (1 - th) / (nrow * seen)));
      }
    }
    s.check("missing values: every cell learnt within a few standard errors of the truth", z, 4.5);
    double cs = 0;
    for (std::size_t v = 0; v < n; v++) {
      double t = 0;
      for (double c : res.counts[v]) t += c;
      cs = std::max(cs, std::fabs(t - static_cast<double>(N)));
    }
    s.check("missing values: each table's expected counts add up to the number of records", cs, 1e-7);
    EmOptions a, b;
    a.max_iterations = b.max_iterations = 3;
    b.joint_cells = 1;
    const auto ra = em(start, records, a), rb = em(start, records, b);
    double d = std::fabs(ra.objective.back() - rb.objective.back()) / std::fabs(ra.objective.back());
    for (std::size_t v = 0; v < n; v++) d = std::max(d, calx::verification::max_difference(ra.network.tables[v].values, rb.network.tables[v].values));
    s.check("missing values: the E-step by families equals the E-step by a record's missing part", d, 1e-12);
    EmOptions p;
    p.pseudo_count = 1;
    const auto mp = em(start, records, p);
    s.check("missing values: with pseudo-counts the log-posterior never decreases (largest relative drop)", worst_drop(mp.objective), 1e-12);
    s.require("missing values: the log-posterior rises from the start", mp.objective.back() > mp.objective.front());
  }

  // Refusals.
  {
    s.require("records: a record of the wrong length is refused", !refusal([&] { em(start, {Record(n - 1)}); }).empty());
    Record r(n);
    r[0] = 2;
    s.require("records: a state out of range is refused", !refusal([&] { em(start, {r}); }).empty());
    auto t = std::vector<std::vector<double>>{};
    for (std::size_t v = 0; v < n; v++) t.push_back(start.tables[v].values);
    const std::size_t root = truth.order.front();
    t[root] = {1, 0};
    Record q(n);
    q[root] = 1;
    s.require("records: a record impossible under the starting tables is refused",
              refusal([&] { em(with_tables(truth, t), {q}); }).find("probability zero") != std::string::npos);
  }
  return s.done();
}
