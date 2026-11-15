// Bayesian networks over discrete variables (decision 0023).
//
// A Bayesian network is a directed acyclic graph on variables X₁, …, X_n, each with finitely many named states,
// and for each variable a conditional table P(X_i | Pa_i); the joint law is P(x) = Π_i P(x_i | x_{Pa_i}), and
// every law that factorises so is Markov to the graph (Pearl 1988, Probabilistic Reasoning in Intelligent
// Systems, ch. 3; Lauritzen and Spiegelhalter 1988, J. R. Statist. Soc. B 50, 157–224).
//
// A node is given by its name, its states, its parents by name and its table: one row per joint state of the
// parents, in row-major order (the last parent varying fastest), each row a law over the node's states. The
// network is refused, with the reason, when
//   - two variables share a name, a variable has no state or two states share a name;
//   - a parent is unknown, repeated, or the variable itself;
//   - the parents make a cycle (found by morphe's topological order, Kahn 1962; the cycle is named);
//   - a table has the wrong size, an entry is negative or not finite, or a row does not sum to one within the
//     stated tolerance (the row is named by its parents' states).
// Each table is kept as a factor over (parents…, X), X varying fastest, so the node's table is its values.
//
// sample(network, stream) draws a full assignment by ancestral sampling (each variable after its parents, by
// inversion of its row).

#pragma once

#include <tyche/graphical/common.hpp>
#include <tyche/graphical/factor.hpp>

#include <morphe/combinatorics/acyclic.hpp>

#include <calx/random/stream.hpp>

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tyche::graphical {

struct Variable {
  std::string name;
  std::vector<std::string> states;
};

struct Node {
  std::string name;
  std::vector<std::string> states;
  std::vector<std::string> parents;  // by name, in the order of the table's rows
  std::vector<double> table;         // P(X | parents): rows by the parents' joint state (last parent fastest), each a law over the states
};

struct Network {
  std::vector<Variable> variables;
  std::vector<std::vector<std::size_t>> parents;  // by variable, in the order of its table
  std::vector<Factor> tables;                     // by variable: the factor P(X | parents) over (parents…, X)
  std::vector<std::size_t> order;                 // a topological order: every variable after its parents

  std::size_t size() const { return variables.size(); }
  std::size_t cardinality(std::size_t v) const { return variables.at(v).states.size(); }
  std::vector<std::size_t> cards() const {
    std::vector<std::size_t> c(size());
    for (std::size_t v = 0; v < size(); v++) c[v] = cardinality(v);
    return c;
  }
  std::size_t index(std::string_view name) const {
    for (std::size_t v = 0; v < size(); v++)
      if (variables[v].name == name) return v;
    throw Refusal("no variable named '" + std::string(name) + "'");
  }
  std::size_t state(std::size_t v, std::string_view s) const {
    detail::require(v < size(), "no such variable");
    const auto& st = variables[v].states;
    for (std::size_t i = 0; i < st.size(); i++)
      if (st[i] == s) return i;
    throw Refusal("variable '" + variables[v].name + "' has no state '" + std::string(s) + "'");
  }
  // The arcs (parent, child).
  std::vector<morphe::combinatorics::Arc> arcs() const {
    std::vector<morphe::combinatorics::Arc> a;
    for (std::size_t v = 0; v < size(); v++)
      for (std::size_t p : parents[v]) a.push_back({p, v});
    return a;
  }
  // P(x) of a full assignment, the product of the tables' entries.
  double probability(std::span<const std::size_t> x) const {
    detail::require(x.size() == size(), "probability: one state for each variable");
    double p = 1;
    for (std::size_t v = 0; v < size(); v++) {
      std::vector<std::size_t> s;
      for (std::size_t u : tables[v].scope) s.push_back(x[u]);
      p *= tables[v].at(s);
    }
    return p;
  }
};

namespace detail {

inline std::string row_name(const std::vector<Variable>& vars, const std::vector<std::size_t>& parents, std::size_t row) {
  if (parents.empty()) return "(no parents)";
  std::vector<std::size_t> s(parents.size());
  for (std::size_t k = parents.size(); k-- > 0;) {
    const std::size_t n = vars[parents[k]].states.size();
    s[k] = row % n;
    row /= n;
  }
  std::string out = "(";
  for (std::size_t k = 0; k < parents.size(); k++) out += (k ? ", " : "") + vars[parents[k]].name + " = " + vars[parents[k]].states[s[k]];
  return out + ")";
}

inline void check_table(const std::vector<Variable>& vars, const std::vector<std::size_t>& parents, std::size_t v,
                        const std::vector<double>& table, double tolerance) {
  std::size_t rows = 1;
  for (std::size_t p : parents) rows *= vars[p].states.size();
  const std::size_t k = vars[v].states.size();
  require(table.size() == rows * k, "the table of '" + vars[v].name + "' has " + std::to_string(table.size()) + " entries, not " +
                                        std::to_string(rows) + " rows of " + std::to_string(k));
  for (std::size_t r = 0; r < rows; r++) {
    for (std::size_t j = 0; j < k; j++) {
      const double x = table[r * k + j];
      require(std::isfinite(x) && x >= 0, "the table of '" + vars[v].name + "' has a negative or non-finite entry in the row " +
                                              row_name(vars, parents, r));
    }
    const double s = neumaier(std::span<const double>(table).subspan(r * k, k));
    require(std::fabs(s - 1) <= tolerance, "the table of '" + vars[v].name + "' has the row " + row_name(vars, parents, r) +
                                               " summing to " + std::to_string(s) + ", not to 1");
  }
}

}  // namespace detail

inline Network network(const std::vector<Node>& nodes, double tolerance = 1e-9) {
  const std::size_t n = nodes.size();
  Network net;
  for (std::size_t v = 0; v < n; v++) {
    const Node& node = nodes[v];
    detail::require(!node.name.empty(), "a variable has no name");
    for (std::size_t u = 0; u < v; u++) detail::require(nodes[u].name != node.name, "two variables are named '" + node.name + "'");
    detail::require(!node.states.empty(), "the variable '" + node.name + "' has no state");
    for (std::size_t i = 0; i < node.states.size(); i++)
      for (std::size_t j = i + 1; j < node.states.size(); j++)
        detail::require(node.states[i] != node.states[j], "the variable '" + node.name + "' has two states named '" + node.states[i] + "'");
    net.variables.push_back({node.name, node.states});
  }
  net.parents.resize(n);
  for (std::size_t v = 0; v < n; v++) {
    for (const auto& pname : nodes[v].parents) {
      std::size_t p = n;
      for (std::size_t u = 0; u < n; u++)
        if (nodes[u].name == pname) p = u;
      detail::require(p < n, "the parent '" + pname + "' of '" + nodes[v].name + "' is not a variable");
      detail::require(p != v, "the variable '" + nodes[v].name + "' is its own parent");
      for (std::size_t q : net.parents[v]) detail::require(q != p, "the parent '" + pname + "' of '" + nodes[v].name + "' is named twice");
      net.parents[v].push_back(p);
    }
  }
  const auto ordering = morphe::combinatorics::topological_order(n, net.arcs());
  if (!ordering.acyclic) {
    std::string c;
    for (std::size_t v : ordering.cycle) c += net.variables[v].name + " → ";
    c += net.variables[ordering.cycle.front()].name;
    throw Refusal("the parents make a cycle: " + c);
  }
  net.order = ordering.order;
  for (std::size_t v = 0; v < n; v++) {
    detail::check_table(net.variables, net.parents[v], v, nodes[v].table, tolerance);
    std::vector<std::size_t> scope = net.parents[v], cards;
    scope.push_back(v);
    for (std::size_t u : scope) cards.push_back(net.cardinality(u));
    net.tables.push_back(Factor{scope, cards, nodes[v].table});
  }
  return net;
}

// The same graph with other tables (one per variable, laid out as a node's table), validated as above.
inline Network with_tables(const Network& net, const std::vector<std::vector<double>>& tables, double tolerance = 1e-9) {
  detail::require(tables.size() == net.size(), "with_tables: one table for each variable");
  Network out = net;
  for (std::size_t v = 0; v < net.size(); v++) {
    detail::check_table(net.variables, net.parents[v], v, tables[v], tolerance);
    out.tables[v].values = tables[v];
  }
  return out;
}

// Hard evidence given by names: (variable, state) pairs.
inline Evidence evidence(const Network& net, const std::vector<std::pair<std::string, std::string>>& observed) {
  Evidence e;
  for (const auto& [var, st] : observed) {
    const std::size_t v = net.index(var);
    detail::require(!e.observed.count(v), "evidence: the variable '" + var + "' is observed twice");
    e.observed[v] = net.state(v, st);
  }
  return e;
}

// One full assignment drawn from the joint law by ancestral sampling.
inline std::vector<std::size_t> sample(const Network& net, calx::random::Stream& rng) {
  std::vector<std::size_t> x(net.size(), 0);
  for (std::size_t v : net.order) {
    const Factor& t = net.tables[v];
    const std::size_t k = net.cardinality(v);
    std::size_t row = 0;
    for (std::size_t p : net.parents[v]) row = row * net.cardinality(p) + x[p];
    double u = rng.uniform();
    std::size_t s = 0;
    for (; s + 1 < k; s++) {
      u -= t.values[row * k + s];
      if (u < 0) break;
    }
    // Skip states of probability zero that rounding would land on.
    while (t.values[row * k + s] == 0 && s > 0) s--;
    x[v] = s;
  }
  return x;
}

}  // namespace tyche::graphical
