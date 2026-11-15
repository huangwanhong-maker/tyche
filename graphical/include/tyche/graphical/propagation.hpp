// Belief propagation (sum-product) on factor graphs (decision 0023).
//
// A factor graph joins each variable to the factors that hold it (Kschischang, Frey and Loeliger 2001, IEEE
// Trans. Inf. Theory 47, 498–519). The sum-product messages are
//   variable to factor   μ_{v→f}(x) ∝ Π_{g ∋ v, g ≠ f} μ_{g→v}(x)
//   factor to variable   μ_{f→v}(x) ∝ Σ_{x_f : x_v = x} f(x_f) Π_{u ∈ f, u ≠ v} μ_{u→f}(x_u)
// and the belief of a variable is b_v(x) ∝ Π_{g ∋ v} μ_{g→v}(x). Messages are normalised to sum to one.
//
// On a factor graph without cycles (the graph of a polytree's tables is one: Pearl 1988, Probabilistic Reasoning
// in Intelligent Systems, ch. 4) the messages reach a fixed point after as many rounds as the graph's diameter,
// and the beliefs are then the exact marginals P(X_v | e). On a graph with cycles the same updates ("loopy"
// propagation; Murphy, Weiss and Jordan 1999, Proc. UAI 15, 467–475) need not converge, and their fixed points
// are stationary points of the Bethe free energy (Yedidia, Freeman and Weiss 2005, IEEE Trans. Inf. Theory 51,
// 2282–2312), not the marginals: the result then says it is approximate.
//
// Schedule: in each round every factor-to-variable message is computed from the previous round's variable-to-
// factor messages, damped as m ← (1 − d) m_new + d m_old (0 ≤ d < 1), and then every variable-to-factor
// message from them. The round's change is the largest absolute change of a factor-to-variable message; the
// propagation has converged when it is at most the tolerance, and stops after max_iterations rounds otherwise.
// A message that vanishes means the evidence is impossible (on a tree, exactly when P(e) = 0) or that loopy
// propagation met a contradiction; it is refused with that reason.
//
// On a network, hard evidence reduces the tables (an observed variable leaves every factor, its marginal the
// indicator of its state); a likelihood is a factor over its variable. Whether the factor graph is a tree is
// decided on the factors so reduced, so evidence that cuts every cycle makes the result exact.

#pragma once

#include <tyche/graphical/common.hpp>
#include <tyche/graphical/factor.hpp>
#include <tyche/graphical/network.hpp>
#include <tyche/graphical/elimination.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

namespace tyche::graphical {

struct PropagationOptions {
  std::size_t max_iterations = 1000;
  double tolerance = 1e-13;  // the largest change of a message at which the propagation has converged
  double damping = 0;        // d in [0, 1)
};

struct Propagation {
  std::vector<std::vector<double>> marginals;  // by variable: exact marginals when `exact`, otherwise approximate beliefs
  bool tree = false;                           // the factor graph has no cycle
  bool converged = false;
  bool exact = false;                          // tree and converged
  std::size_t iterations = 0;                  // rounds made
  double last_change = 0;                      // the largest change of a message in the last round
};

namespace detail {

inline std::size_t find_root(std::vector<std::size_t>& parent, std::size_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

inline void normalise_message(std::vector<double>& m, const char* what) {
  double s = 0;
  for (double v : m) s += v;
  require(s > 0 && std::isfinite(s), std::string("belief propagation: a ") + what +
                                         " vanished: the evidence is impossible, or loopy propagation met a contradiction");
  for (double& v : m) v /= s;
}

}  // namespace detail

// Sum-product on the factor graph of `factors` over the variables 0, …, cards.size() − 1.
inline Propagation propagate(const std::vector<std::size_t>& cards, const std::vector<Factor>& factors, const PropagationOptions& o = {}) {
  detail::require(o.damping >= 0 && o.damping < 1, "propagate: the damping lies in [0, 1)");
  detail::require(o.tolerance >= 0, "propagate: the tolerance is not negative");
  const std::size_t n = cards.size(), F = factors.size();
  for (std::size_t c : cards) detail::require(c >= 1, "propagate: every variable has at least one state");
  for (const Factor& f : factors) {
    detail::require(f.values.size() == detail::table_size(f.cards), "propagate: a malformed factor");
    for (std::size_t k = 0; k < f.scope.size(); k++) {
      detail::require(f.scope[k] < n, "propagate: a factor holds a variable out of range");
      detail::require(f.cards[k] == cards[f.scope[k]], "propagate: a factor disagrees on a variable's number of states");
    }
    if (f.scope.empty())
      detail::require(f.values[0] > 0, "propagate: a constant factor is zero: the evidence is impossible");
  }
  // Edges (factor, position) and the edges of each variable.
  std::vector<std::size_t> edge_factor, edge_var, first(F + 1, 0);
  std::vector<std::vector<std::size_t>> var_edges(n);
  for (std::size_t f = 0; f < F; f++) {
    first[f] = edge_var.size();
    for (std::size_t v : factors[f].scope) {
      var_edges[v].push_back(edge_var.size());
      edge_factor.push_back(f);
      edge_var.push_back(v);
    }
  }
  first[F] = edge_var.size();
  const std::size_t E = edge_var.size();

  Propagation out;
  // A forest exactly when no edge closes a cycle (union–find over variables and factors).
  {
    std::vector<std::size_t> parent(n + F);
    std::iota(parent.begin(), parent.end(), 0);
    out.tree = true;
    for (std::size_t e = 0; e < E; e++) {
      const std::size_t a = detail::find_root(parent, edge_var[e]), b = detail::find_root(parent, n + edge_factor[e]);
      if (a == b) out.tree = false;
      else parent[a] = b;
    }
  }

  std::vector<std::vector<double>> f2v(E), v2f(E);
  for (std::size_t e = 0; e < E; e++) {
    const std::size_t c = cards[edge_var[e]];
    f2v[e].assign(c, 1.0 / static_cast<double>(c));
    v2f[e] = f2v[e];
  }
  std::vector<std::vector<double>> fresh(E);
  for (std::size_t it = 0; it < o.max_iterations; it++) {
    double change = 0;
    for (std::size_t f = 0; f < F; f++) {
      const Factor& phi = factors[f];
      const std::size_t k = phi.scope.size(), e0 = first[f];
      if (k == 0) continue;
      for (std::size_t j = 0; j < k; j++) fresh[e0 + j].assign(phi.cards[j], 0.0);
      std::vector<std::size_t> x(k, 0);
      std::vector<double> pre(k + 1), suf(k + 1);
      for (std::size_t i = 0; i < phi.values.size(); i++) {
        const double val = phi.values[i];
        if (val != 0) {
          pre[0] = 1;
          for (std::size_t j = 0; j < k; j++) pre[j + 1] = pre[j] * v2f[e0 + j][x[j]];
          suf[k] = 1;
          for (std::size_t j = k; j-- > 0;) suf[j] = suf[j + 1] * v2f[e0 + j][x[j]];
          for (std::size_t j = 0; j < k; j++) fresh[e0 + j][x[j]] += val * pre[j] * suf[j + 1];
        }
        for (std::size_t j = k; j-- > 0;) {
          if (++x[j] < phi.cards[j]) break;
          x[j] = 0;
        }
      }
      for (std::size_t j = 0; j < k; j++) {
        auto& m = fresh[e0 + j];
        detail::normalise_message(m, "message from a factor");
        for (std::size_t s = 0; s < m.size(); s++) {
          const double d = (1 - o.damping) * m[s] + o.damping * f2v[e0 + j][s];
          change = std::max(change, std::fabs(d - f2v[e0 + j][s]));
          m[s] = d;
        }
        f2v[e0 + j].swap(m);
      }
    }
    for (std::size_t v = 0; v < n; v++) {
      const auto& ev = var_edges[v];
      for (std::size_t a = 0; a < ev.size(); a++) {
        std::vector<double> m(cards[v], 1.0);
        for (std::size_t b = 0; b < ev.size(); b++)
          if (b != a)
            for (std::size_t s = 0; s < cards[v]; s++) m[s] *= f2v[ev[b]][s];
        detail::normalise_message(m, "message from a variable");
        v2f[ev[a]] = std::move(m);
      }
    }
    out.iterations = it + 1;
    out.last_change = change;
    if (change <= o.tolerance) {
      out.converged = true;
      break;
    }
  }
  out.marginals.resize(n);
  for (std::size_t v = 0; v < n; v++) {
    std::vector<double> b(cards[v], 1.0);
    for (std::size_t e : var_edges[v])
      for (std::size_t s = 0; s < cards[v]; s++) b[s] *= f2v[e][s];
    detail::normalise_message(b, "belief");
    out.marginals[v] = std::move(b);
  }
  out.exact = out.tree && out.converged;
  return out;
}

// Sum-product on a network's tables with the evidence entered.
inline Propagation propagate(const Network& net, const Evidence& e = {}, const PropagationOptions& o = {}) {
  detail::check_evidence(net, e);
  std::vector<Factor> fs;
  for (std::size_t v = 0; v < net.size(); v++) {
    Factor f = net.tables[v];
    for (std::size_t u : net.tables[v].scope) {
      const auto it = e.observed.find(u);
      if (it != e.observed.end()) f = reduce(f, u, it->second);
    }
    fs.push_back(std::move(f));
  }
  for (const auto& [u, lam] : e.likelihood) {
    Factor f{{u}, {net.cardinality(u)}, lam};
    const auto it = e.observed.find(u);
    if (it != e.observed.end()) f = reduce(f, u, it->second);
    fs.push_back(std::move(f));
  }
  Propagation out = propagate(net.cards(), fs, o);
  for (const auto& [u, s] : e.observed) {
    out.marginals[u].assign(net.cardinality(u), 0.0);
    out.marginals[u][s] = 1;
  }
  return out;
}

}  // namespace tyche::graphical
