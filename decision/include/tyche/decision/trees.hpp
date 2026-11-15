// Decision trees: decisions and chances in sequence, solved by folding back.
//
// A decision tree has decision nodes (the decision maker chooses a branch), chance nodes (nature chooses, with stated
// probabilities) and terminal nodes; every node may carry a payoff gained on reaching it (a cost is a negative payoff). Its
// value is computed from the leaves up: a terminal node is worth its payoff, a chance node its payoff plus the expectation of
// its children, a decision node its payoff plus the best of its children ("averaging out and folding back"; Raiffa 1968,
// Decision Analysis, ch. 2). The branch chosen at each decision node is an optimal strategy.
//
// That folding back finds the best strategy is Bellman's principle of optimality for a finite tree; the checks hold it to the
// enumeration of every strategy (a choice at every decision node), each evaluated by itself. The value of information is a tree
// too: deciding after a chance node is worth at least deciding before it, and the difference is the value of what that chance
// reveals (value.hpp).

#pragma once

#include <tyche/decision/choice.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::decision {

enum class NodeKind { decision, chance, terminal };

struct Node {
  NodeKind kind = NodeKind::terminal;
  std::string label;
  double payoff = 0;                   // gained on reaching the node
  std::vector<std::size_t> children;
  std::vector<double> probabilities;   // for a chance node: one per child, summing to 1
  std::vector<std::string> branches;   // the names of the branches, optionally
};

struct Tree {
  std::vector<Node> nodes;
  std::size_t root = 0;

  std::size_t add(Node n) {
    nodes.push_back(std::move(n));
    return nodes.size() - 1;
  }
};

// Every node reachable once from the root, the children of chance nodes with laws, decision nodes with at least one branch.
inline void validate(const Tree& t) {
  if (t.root >= t.nodes.size()) throw std::invalid_argument("tyche::decision: the tree has no root");
  std::vector<int> seen(t.nodes.size(), 0);
  std::vector<std::size_t> stack = {t.root};
  while (!stack.empty()) {
    const std::size_t v = stack.back();
    stack.pop_back();
    if (seen[v]++) throw std::invalid_argument("tyche::decision: node " + std::to_string(v) + " is reached twice: a tree, not a graph");
    const Node& n = t.nodes[v];
    if (!std::isfinite(n.payoff)) throw std::invalid_argument("tyche::decision: payoffs are finite");
    if (n.kind == NodeKind::terminal && !n.children.empty()) throw std::invalid_argument("tyche::decision: a terminal node has no children");
    if (n.kind != NodeKind::terminal && n.children.empty()) throw std::invalid_argument("tyche::decision: node " + std::to_string(v) + " has no branch");
    if (n.kind == NodeKind::chance) validate_law(n.probabilities, n.children.size(), "tree (a chance node)");
    for (const std::size_t c : n.children) {
      if (c >= t.nodes.size()) throw std::invalid_argument("tyche::decision: a branch leads to no node");
      stack.push_back(c);
    }
  }
}

struct Folded {
  std::vector<double> value;                // each node's value
  std::map<std::size_t, std::size_t> best;  // each decision node → the index of the branch chosen (the first of the best)
};

// Averaging out and folding back.
inline Folded fold_back(const Tree& t) {
  validate(t);
  Folded f;
  f.value.assign(t.nodes.size(), 0);
  // Post-order, iteratively.
  std::vector<std::pair<std::size_t, bool>> stack = {{t.root, false}};
  while (!stack.empty()) {
    auto [v, done] = stack.back();
    stack.pop_back();
    const Node& n = t.nodes[v];
    if (!done) {
      stack.push_back({v, true});
      for (const std::size_t c : n.children) stack.push_back({c, false});
      continue;
    }
    double x = 0;
    if (n.kind == NodeKind::chance) {
      for (std::size_t i = 0; i < n.children.size(); i++) x += n.probabilities[i] * f.value[n.children[i]];
    } else if (n.kind == NodeKind::decision) {
      for (std::size_t i = 0; i < n.children.size(); i++)
        if (i == 0 || f.value[n.children[i]] > x + tie * std::max(1.0, std::abs(x))) x = f.value[n.children[i]], f.best[v] = i;
    }
    f.value[v] = n.payoff + x;
  }
  return f;
}

// A strategy: a branch for every decision node. Its expected value, by itself.
inline double evaluate(const Tree& t, const std::map<std::size_t, std::size_t>& strategy) {
  validate(t);
  std::function<double(std::size_t)> value = [&](std::size_t v) -> double {
    const Node& n = t.nodes[v];
    if (n.kind == NodeKind::terminal) return n.payoff;
    if (n.kind == NodeKind::chance) {
      double x = 0;
      for (std::size_t i = 0; i < n.children.size(); i++) x += n.probabilities[i] * value(n.children[i]);
      return n.payoff + x;
    }
    const auto it = strategy.find(v);
    if (it == strategy.end() || it->second >= n.children.size()) throw std::invalid_argument("tyche::decision::evaluate: the strategy chooses no branch at node " + std::to_string(v));
    return n.payoff + value(n.children[it->second]);
  };
  return value(t.root);
}

// Every strategy of the tree (refused above `limit` of them).
inline std::vector<std::map<std::size_t, std::size_t>> strategies(const Tree& t, std::size_t limit = 1u << 20) {
  validate(t);
  std::vector<std::size_t> decisions;
  for (std::size_t v = 0; v < t.nodes.size(); v++)
    if (t.nodes[v].kind == NodeKind::decision) decisions.push_back(v);
  double count = 1;
  for (const std::size_t d : decisions) count *= static_cast<double>(t.nodes[d].children.size());
  if (count > static_cast<double>(limit)) throw std::invalid_argument("tyche::decision::strategies: too many strategies to list");
  std::vector<std::map<std::size_t, std::size_t>> out;
  std::vector<std::size_t> pick(decisions.size(), 0);
  while (true) {
    std::map<std::size_t, std::size_t> s;
    for (std::size_t i = 0; i < decisions.size(); i++) s[decisions[i]] = pick[i];
    out.push_back(std::move(s));
    std::size_t i = 0;
    while (i < pick.size() && ++pick[i] == t.nodes[decisions[i]].children.size()) pick[i++] = 0;
    if (i == pick.size()) break;
  }
  return out;
}

// The tree of deciding on a table before or after observing a test: a chance node over the signals, each followed by a decision
// over the acts, each followed by a chance node over the states given the signal. Folding it back gives the value with the
// test, so that its root value less the prior value is EVSI (value.hpp), computed another way.
inline Tree informed_tree(const Table& t, std::span<const double> p, const std::vector<std::vector<double>>& likelihood) {
  validate(t);
  validate_law(p, t.n_states(), "informed_tree");
  Tree tree;
  Node root;
  root.kind = NodeKind::chance;
  root.label = "the test";
  tree.root = tree.add(root);
  const std::size_t Y = likelihood.at(0).size();
  for (std::size_t y = 0; y < Y; y++) {
    double py = 0;
    for (std::size_t s = 0; s < t.n_states(); s++) py += p[s] * likelihood[s][y];
    Node d;
    d.kind = NodeKind::decision;
    d.label = "after signal " + std::to_string(y);
    const std::size_t dv = tree.add(d);
    tree.nodes[tree.root].children.push_back(dv);
    tree.nodes[tree.root].probabilities.push_back(py);
    for (std::size_t a = 0; a < t.n_acts(); a++) {
      Node c;
      c.kind = NodeKind::chance;
      const std::size_t cv = tree.add(c);
      tree.nodes[dv].children.push_back(cv);
      for (std::size_t s = 0; s < t.n_states(); s++) {
        Node leaf;
        leaf.payoff = t.u[a][s];
        const std::size_t lv = tree.add(leaf);
        tree.nodes[cv].children.push_back(lv);
        tree.nodes[cv].probabilities.push_back(py > 0 ? p[s] * likelihood[s][y] / py : 1.0 / static_cast<double>(t.n_states()));
      }
    }
  }
  return tree;
}

}  // namespace tyche::decision
