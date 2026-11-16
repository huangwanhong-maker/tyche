// Information bought under a budget: which of several tests to run before deciding (decision 0026).
//
// Several decisions are to be taken, each over its own states with its own prior and its own acts, and before each a test could
// be run at a cost: a site inspected, a sample assayed, a survey fielded. The budget does not run to all of them. value.hpp says
// what one test is worth for one decision, its EVSI; here the tests are chosen together.
//
// When the decisions are separate (the states of one independent of the states and signals of the others, and the utilities
// adding), the worth of running a set of tests is the sum of their EVSIs: each test bears on its own decision alone, and the
// preposterior analysis of the product problem factorises (the theory note, Proposition 1; the checks hold it by forming the
// product problem outright). The best set within the budget is then a 0/1 knapsack, the EVSIs its values and the costs its
// weights, solved by calx's branch and bound with its certificate: exact, or the best found with an upper bound. With equal
// costs it is the tests of greatest EVSI.
//
// A cost may be of another kind than the utilities (inspectors' days against harm avoided): then it only uses the budget. Or it
// is in the utilities' own units and is paid out of what the test gains (`charged`): a test is then worth its EVSI less its
// cost, and one that is not worth its cost is never run, whatever budget is left.
//
// Tests on decisions that share states are not separate, and their worths do not add (two tests of one state are partly
// redundant); that case is the sequential one of the POMDP modules, and is refused here by construction: each problem carries
// its own table.

#pragma once

#include <tyche/decision/value.hpp>

#include <calx/optimisation/knapsack.hpp>

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::decision {

// One decision, and the test that could be run before it.
struct Informable {
  std::string name;
  Table table;
  std::vector<double> prior;
  Test test;
  double cost = 0;  // of running the test
};

struct Purchase {
  std::vector<Preposterior> analysis;  // of each test, whether bought or not
  std::vector<double> worth;           // each test's EVSI, less its cost when costs are charged
  std::vector<std::size_t> bought;     // the tests to run, in increasing order
  double gained = 0;                   // Σ of their worths: what running them adds in expectation
  double spent = 0;                    // Σ of their costs
  double without = 0;                  // Σ of the decisions' values with no test run
  double bound = 0;                    // an upper bound on what any set within the budget could gain; `gained` when `exact`
  bool exact = true;
};

// The tests to run within `budget`. With `charged`, a test's cost is also subtracted from what it gains.
inline Purchase buy(std::span<const Informable> problems, double budget, bool charged = false, std::size_t node_limit = 2000000) {
  if (!(budget >= 0) || !std::isfinite(budget)) throw std::invalid_argument("tyche::decision::buy: the budget is finite and not negative");
  Purchase out;
  std::vector<double> cost;
  for (const auto& q : problems) {
    if (!(q.cost >= 0) || !std::isfinite(q.cost)) throw std::invalid_argument("tyche::decision::buy: the cost of the test of " + q.name + " is finite and not negative");
    out.analysis.push_back(evsi(q.table, q.prior, q.test));
    out.without += out.analysis.back().prior_value;
    out.worth.push_back(out.analysis.back().evsi - (charged ? q.cost : 0.0));
    cost.push_back(q.cost);
  }
  const auto k = calx::optimisation::knapsack(out.worth, cost, budget, node_limit);
  out.bought = k.chosen;
  out.gained = k.value, out.spent = k.weight, out.bound = k.bound, out.exact = k.exact;
  return out;
}

}  // namespace tyche::decision
