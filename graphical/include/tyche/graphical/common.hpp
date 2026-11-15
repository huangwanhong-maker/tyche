// What tyche::graphical's headers share: the refusal and the evidence (decision 0023).
//
// Every computation that cannot be carried out throws a Refusal (an std::invalid_argument) whose message says why:
// a cycle among the parents, a row of a conditional table that does not sum to one, evidence of probability zero,
// a variable or state that does not exist.
//
// Evidence is of two kinds (Pearl 1988, Probabilistic Reasoning in Intelligent Systems, ch. 2):
//   observed     a variable is known to be in one state (hard evidence);
//   likelihood   a variable's states are weighted by λ(x) ≥ 0, the likelihood of an observation not itself in the
//                network ("virtual evidence"); only the ratios of λ matter to a posterior, while P(e) scales with λ.
// A variable may carry both; then both apply.

#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::graphical {

class Refusal : public std::invalid_argument {
 public:
  explicit Refusal(const std::string& reason) : std::invalid_argument("tyche::graphical: " + reason) {}
};

struct Evidence {
  std::map<std::size_t, std::size_t> observed;            // variable → its observed state
  std::map<std::size_t, std::vector<double>> likelihood;  // variable → λ(state) ≥ 0, one per state
  bool empty() const { return observed.empty() && likelihood.empty(); }
};

namespace detail {
inline void require(bool ok, const std::string& what) {
  if (!ok) throw Refusal(what);
}
}  // namespace detail

}  // namespace tyche::graphical
