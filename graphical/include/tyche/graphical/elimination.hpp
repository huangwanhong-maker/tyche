// Exact inference in Bayesian networks by variable elimination (decision 0023).
//
// Variable elimination (Zhang and Poole 1994, Proc. 10th Canadian Conf. on AI, 171–178; Dechter 1996, Proc.
// UAI 12, 211–219; Koller and Friedman 2009, Probabilistic Graphical Models, ch. 9) computes Σ_{x_E} Π_j φ_j by
// taking the variables of E one at a time: the factors that hold the variable are multiplied and the variable
// is summed out of their product. The result is exact; its cost is exponential in the induced width of the
// order (the largest number of other variables a variable meets when it is eliminated), which a greedy rule
// keeps small:
//   min_fill     the variable whose elimination joins the fewest pairs of its neighbours not yet joined
//                (Kjærulff 1990, Triangulation of graphs: algorithms giving small total state space, Aalborg
//                University, report R 90-09), ties to the fewer neighbours, then to the lower index;
//   min_degree   the variable with the fewest neighbours, ties to the lower index.
// A caller may give the order instead. Each factor formed is divided by its largest value and the logarithm of
// that value kept apart, so that the probability of much evidence does not underflow: log_evidence_probability
// is exact to rounding when evidence_probability is below the smallest double.
//
// On a network, evidence e enters as follows: the tables are reduced to the observed states; an observed
// variable that is also queried keeps an indicator instead; a likelihood λ is a factor over its variable. Only
// the variables queried or observed and their ancestors are kept: the others are barren, their tables summing
// to one whatever their parents (Shachter 1986, Operations Research 34, 871–882).
//
//   evidence_probability(net, e)               P(e) = Σ_x Π_i P(x_i | x_{Pa_i}) Π_{observed} 1[x_v = e_v] Π λ_v(x_v)
//   joint_with_evidence(net, Q, e)             P(Q, e) as a factor over Q in the order given
//   joint(net, Q, e), marginal(net, X, e)      P(Q | e), refused when P(e) = 0
//   marginals(net, e)                          P(X_i | e) for every variable, one elimination each
//   likelihood_ratio(net, X, S, e)             P(e | X ∈ S) / P(e | X ∉ S), with the prior and posterior of X ∈ S;
//                                              the posterior odds are the ratio times the prior odds (Good 1950,
//                                              Probability and the Weighing of Evidence)
//   most_probable_explanation(net, e)          argmax_x P(x, e) by max-product elimination and traceback
//                                              (Dawid 1992, Statistics and Computing 2, 25–36; Koller and Friedman
//                                              2009, ch. 13); ties go to the lower state, each variable in turn
//                                              from the last eliminated
//
// Accuracy: exact up to rounding (a relative error of a few units in the last place per factor formed).

#pragma once

#include <tyche/graphical/common.hpp>
#include <tyche/graphical/factor.hpp>
#include <tyche/graphical/network.hpp>

#include <morphe/combinatorics/acyclic.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tyche::graphical {

enum class Heuristic { min_fill, min_degree };

struct EliminationOptions {
  Heuristic heuristic = Heuristic::min_fill;
  std::vector<std::size_t> order;  // when not empty: the order of elimination (every variable to eliminate, once; others ignored)
};

struct Plan {
  std::vector<std::size_t> order;  // the variables eliminated, in turn
  std::size_t width = 0;           // the induced width: the most neighbours a variable has when it is eliminated
};

namespace detail {

// The variables of a list of factors with their numbers of states, checked to agree.
inline std::map<std::size_t, std::size_t> variables_of(const std::vector<Factor>& factors) {
  std::map<std::size_t, std::size_t> card;
  for (const Factor& f : factors) {
    require(f.scope.size() == f.cards.size() && f.values.size() == table_size(f.cards), "elimination: a malformed factor");
    for (std::size_t k = 0; k < f.scope.size(); k++) {
      auto [it, fresh] = card.emplace(f.scope[k], f.cards[k]);
      require(fresh || it->second == f.cards[k], "elimination: a variable has different numbers of states in two factors");
    }
  }
  return card;
}

}  // namespace detail

// The order in which to eliminate `eliminate` from the product of `factors`, and its induced width; or the width
// of a given order.
inline Plan plan(const std::vector<Factor>& factors, const std::vector<std::size_t>& eliminate, Heuristic heuristic = Heuristic::min_fill,
                 const std::vector<std::size_t>& given = {}) {
  const auto card = detail::variables_of(factors);
  std::map<std::size_t, std::set<std::size_t>> adj;
  for (const auto& [v, c] : card) adj[v];
  for (const Factor& f : factors)
    for (std::size_t a : f.scope)
      for (std::size_t b : f.scope)
        if (a != b) adj[a].insert(b);
  std::set<std::size_t> left;
  for (std::size_t v : eliminate) {
    detail::require(card.count(v), "plan: a variable to eliminate is in no factor");
    detail::require(left.insert(v).second, "plan: a variable to eliminate is named twice");
  }
  Plan p;
  auto remove = [&](std::size_t v) {
    const auto nb = adj[v];
    p.width = std::max(p.width, nb.size());
    for (std::size_t a : nb) {
      adj[a].erase(v);
      for (std::size_t b : nb)
        if (a != b) adj[a].insert(b);
    }
    adj.erase(v);
    left.erase(v);
    p.order.push_back(v);
  };
  if (!given.empty()) {
    std::set<std::size_t> seen;
    for (std::size_t v : given) {
      detail::require(seen.insert(v).second, "plan: the order names a variable twice");
      if (left.count(v)) remove(v);
    }
    detail::require(left.empty(), "plan: the order given leaves a variable to eliminate out");
    return p;
  }
  while (!left.empty()) {
    std::size_t best = 0;
    std::tuple<std::size_t, std::size_t, std::size_t> key{std::numeric_limits<std::size_t>::max(), 0, 0};
    for (std::size_t v : left) {
      const auto& nb = adj[v];
      std::size_t fill = 0;
      if (heuristic == Heuristic::min_fill)
        for (auto a = nb.begin(); a != nb.end(); ++a)
          for (auto b = std::next(a); b != nb.end(); ++b)
            if (!adj[*a].count(*b)) fill++;
      const std::tuple<std::size_t, std::size_t, std::size_t> k{fill, nb.size(), v};
      if (k < key) {
        key = k;
        best = v;
      }
    }
    remove(best);
  }
  return p;
}

namespace detail {

struct Scaled {
  Factor factor;
  double log_scale = 0;  // the factor's values are to be multiplied by exp(log_scale)
  Plan plan;
};

inline void rescale(Factor& f, double& log_scale) {
  double m = 0;
  for (double v : f.values) m = std::max(m, v);
  if (m > 0 && std::isfinite(m) && m != 1) {
    for (double& v : f.values) v /= m;
    log_scale += std::log(m);
  }
}

// The product of the factors, rescaled after each multiplication.
inline Factor scaled_product(const std::vector<Factor>& fs, double& log_scale) {
  Factor r = constant(1);
  for (const Factor& f : fs) {
    r = product(r, f);
    rescale(r, log_scale);
  }
  return r;
}

inline Scaled eliminate(std::vector<Factor> factors, const std::vector<std::size_t>& keep, const EliminationOptions& o) {
  const auto card = variables_of(factors);
  std::set<std::size_t> kept;
  for (std::size_t v : keep) {
    require(card.count(v), "elimination: a variable kept is in no factor");
    require(kept.insert(v).second, "elimination: a variable kept is named twice");
  }
  std::vector<std::size_t> out;
  for (const auto& [v, c] : card)
    if (!kept.count(v)) out.push_back(v);
  Scaled r;
  r.plan = plan(factors, out, o.heuristic, o.order);
  for (std::size_t v : r.plan.order) {
    std::vector<Factor> with, without;
    for (auto& f : factors) (f.position(v) < f.scope.size() ? with : without).push_back(std::move(f));
    Factor g = sum_out(scaled_product(with, r.log_scale), v);
    rescale(g, r.log_scale);
    without.push_back(std::move(g));
    factors = std::move(without);
  }
  r.factor = reorder(scaled_product(factors, r.log_scale), keep);
  return r;
}

inline void check_evidence(const Network& net, const Evidence& e) {
  for (const auto& [v, s] : e.observed) {
    require(v < net.size(), "evidence: no such variable");
    require(s < net.cardinality(v), "evidence: the variable '" + net.variables[v].name + "' has no state " + std::to_string(s));
  }
  for (const auto& [v, lam] : e.likelihood) {
    require(v < net.size(), "evidence: no such variable");
    require(lam.size() == net.cardinality(v), "evidence: the likelihood of '" + net.variables[v].name + "' needs one value per state");
    for (double x : lam) require(std::isfinite(x) && x >= 0, "evidence: a likelihood is finite and non-negative");
  }
}

// The factors of P(x) with the evidence entered, over the variables queried or observed and their ancestors
// (all variables when prune is false).
inline std::vector<Factor> network_factors(const Network& net, const Evidence& e, const std::vector<std::size_t>& query, bool prune) {
  check_evidence(net, e);
  const std::size_t n = net.size();
  std::vector<char> keep(n, prune ? 0 : 1), queried(n, 0);
  for (std::size_t q : query) {
    require(q < net.size(), "no such variable");
    queried[q] = 1;
  }
  if (prune) {
    const auto arcs = net.arcs();
    std::vector<std::size_t> seeds = query;
    for (const auto& [v, s] : e.observed) seeds.push_back(v);
    for (const auto& [v, l] : e.likelihood) seeds.push_back(v);
    for (std::size_t s : seeds) {
      if (keep[s]) continue;
      keep[s] = 1;
      for (const auto& r : morphe::combinatorics::down(n, arcs, s)) keep[r.element] = 1;
    }
  }
  std::vector<Factor> fs;
  for (std::size_t v = 0; v < n; v++) {
    if (!keep[v]) continue;
    Factor f = net.tables[v];
    for (std::size_t u : net.tables[v].scope) {
      const auto it = e.observed.find(u);
      if (it != e.observed.end() && !queried[u]) f = reduce(f, u, it->second);
    }
    fs.push_back(std::move(f));
  }
  for (const auto& [u, s] : e.observed)
    if (queried[u]) {
      std::vector<double> ind(net.cardinality(u), 0.0);
      ind[s] = 1;
      fs.push_back(Factor{{u}, {net.cardinality(u)}, ind});
    }
  for (const auto& [u, lam] : e.likelihood) {
    Factor f{{u}, {net.cardinality(u)}, lam};
    const auto it = e.observed.find(u);
    if (it != e.observed.end() && !queried[u]) f = reduce(f, u, it->second);
    fs.push_back(std::move(f));
  }
  return fs;
}

}  // namespace detail

// Σ over the other variables of the product of the factors, as a factor over `keep` in its order.
inline Factor sum_product(std::vector<Factor> factors, const std::vector<std::size_t>& keep, const EliminationOptions& o = {}) {
  auto r = detail::eliminate(std::move(factors), keep, o);
  const double s = std::exp(r.log_scale);
  for (double& v : r.factor.values) v *= s;
  return r.factor;
}

inline double log_evidence_probability(const Network& net, const Evidence& e = {}, const EliminationOptions& o = {}) {
  const auto r = detail::eliminate(detail::network_factors(net, e, {}, true), {}, o);
  return r.factor.values[0] > 0 ? std::log(r.factor.values[0]) + r.log_scale : -std::numeric_limits<double>::infinity();
}

inline double evidence_probability(const Network& net, const Evidence& e = {}, const EliminationOptions& o = {}) {
  return std::exp(log_evidence_probability(net, e, o));
}

inline Factor joint_with_evidence(const Network& net, const std::vector<std::size_t>& query, const Evidence& e = {}, const EliminationOptions& o = {}) {
  return sum_product(detail::network_factors(net, e, query, true), query, o);
}

inline Factor joint(const Network& net, const std::vector<std::size_t>& query, const Evidence& e = {}, const EliminationOptions& o = {}) {
  auto r = detail::eliminate(detail::network_factors(net, e, query, true), query, o);
  const double z = total(r.factor);
  detail::require(z > 0, "the evidence has probability zero, so no posterior is defined");
  for (double& v : r.factor.values) v /= z;
  return r.factor;
}

inline std::vector<double> marginal(const Network& net, std::size_t variable, const Evidence& e = {}, const EliminationOptions& o = {}) {
  return joint(net, {variable}, e, o).values;
}

inline std::vector<std::vector<double>> marginals(const Network& net, const Evidence& e = {}, const EliminationOptions& o = {}) {
  std::vector<std::vector<double>> out;
  for (std::size_t v = 0; v < net.size(); v++) out.push_back(marginal(net, v, e, o));
  return out;
}

struct LikelihoodRatio {
  double ratio = 0;                    // P(e | H) / P(e | ¬H); +∞ when only ¬H is excluded by the evidence
  double log_ratio = 0;                // its natural logarithm, computed without underflow
  double given_h = 0, given_not_h = 0; // P(e | H), P(e | ¬H)
  double prior = 0, posterior = 0;     // P(H), P(H | e)
  double evidence = 0;                 // P(e)
};

// The likelihood ratio of the evidence for H: X ∈ states against ¬H: X ∉ states. Refused when P(H) is 0 or 1, or
// when the evidence is impossible under both.
inline LikelihoodRatio likelihood_ratio(const Network& net, std::size_t variable, const std::vector<std::size_t>& states, const Evidence& e = {},
                                        const EliminationOptions& o = {}) {
  detail::require(variable < net.size(), "likelihood_ratio: no such variable");
  const std::size_t k = net.cardinality(variable);
  std::vector<char> in(k, 0);
  for (std::size_t s : states) {
    detail::require(s < k, "likelihood_ratio: a state out of range");
    detail::require(!in[s], "likelihood_ratio: a state named twice");
    in[s] = 1;
  }
  const auto prior = marginal(net, variable, {}, o);
  double ph = 0, pn = 0;
  for (std::size_t s = 0; s < k; s++) (in[s] ? ph : pn) += prior[s];
  detail::require(ph > 0, "likelihood_ratio: the hypothesis has prior probability zero, so P(e | H) is undefined");
  detail::require(pn > 0, "likelihood_ratio: the hypothesis has prior probability one, so P(e | not H) is undefined");
  const auto r = detail::eliminate(detail::network_factors(net, e, {variable}, true), {variable}, o);
  double fh = 0, fn = 0;  // P(X ∈ S, e) and P(X ∉ S, e) divided by exp(log_scale)
  for (std::size_t s = 0; s < k; s++) (in[s] ? fh : fn) += r.factor.values[s];
  detail::require(fh > 0 || fn > 0, "likelihood_ratio: the evidence has probability zero under both H and not H");
  LikelihoodRatio out;
  const double scale = std::exp(r.log_scale);
  out.prior = ph;
  out.given_h = fh * scale / ph;
  out.given_not_h = fn * scale / pn;
  out.evidence = (fh + fn) * scale;
  out.posterior = fh / (fh + fn);
  constexpr double inf = std::numeric_limits<double>::infinity();
  if (fn == 0) {
    out.ratio = inf;
    out.log_ratio = inf;
  } else if (fh == 0) {
    out.ratio = 0;
    out.log_ratio = -inf;
  } else {
    out.log_ratio = std::log(fh) - std::log(ph) - std::log(fn) + std::log(pn);
    out.ratio = std::exp(out.log_ratio);
  }
  return out;
}

inline LikelihoodRatio likelihood_ratio(const Network& net, std::size_t variable, std::size_t state, const Evidence& e = {}, const EliminationOptions& o = {}) {
  return likelihood_ratio(net, variable, std::vector<std::size_t>{state}, e, o);
}

struct Explanation {
  std::vector<std::size_t> states;  // the most probable full assignment consistent with the evidence
  double probability = 0;           // max_x P(x, e) (with the likelihoods' weights)
  double log_probability = 0;
};

inline Explanation most_probable_explanation(const Network& net, const Evidence& e = {}, const EliminationOptions& o = {}) {
  std::vector<Factor> factors = detail::network_factors(net, e, {}, false);
  std::vector<std::size_t> all;
  for (const auto& [v, c] : detail::variables_of(factors)) all.push_back(v);
  const Plan p = plan(factors, all, o.heuristic, o.order);
  double log_scale = 0;
  std::vector<Factor> recorded;
  for (std::size_t v : p.order) {
    std::vector<Factor> with, without;
    for (auto& f : factors) (f.position(v) < f.scope.size() ? with : without).push_back(std::move(f));
    Factor psi = detail::scaled_product(with, log_scale);
    Factor g = max_out(psi, v);
    detail::rescale(g, log_scale);
    recorded.push_back(std::move(psi));
    without.push_back(std::move(g));
    factors = std::move(without);
  }
  const Factor c = detail::scaled_product(factors, log_scale);
  detail::require(c.values[0] > 0, "most_probable_explanation: the evidence has probability zero");
  Explanation x;
  x.log_probability = std::log(c.values[0]) + log_scale;
  x.probability = std::exp(x.log_probability);
  x.states.assign(net.size(), 0);
  for (const auto& [v, s] : e.observed) x.states[v] = s;
  for (std::size_t i = p.order.size(); i-- > 0;) {
    const std::size_t v = p.order[i];
    const Factor& psi = recorded[i];
    const std::size_t pv = psi.position(v);
    std::vector<std::size_t> a(psi.scope.size());
    for (std::size_t k = 0; k < psi.scope.size(); k++) a[k] = x.states[psi.scope[k]];
    double best = -1;
    for (std::size_t s = 0; s < psi.cards[pv]; s++) {
      a[pv] = s;
      const double val = psi.at(a);
      if (val > best) {
        best = val;
        x.states[v] = s;
      }
    }
  }
  return x;
}

}  // namespace tyche::graphical
