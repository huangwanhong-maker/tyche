// Factors: non-negative tables over ordered sets of discrete variables, and the operations that inference is
// made of (decision 0023).
//
// A factor φ over the variables (v₁, …, v_k), with n₁, …, n_k states, holds n₁⋯n_k values φ(x₁, …, x_k) ≥ 0 in
// row-major order: the last variable of the scope varies fastest, so the index of (x₁, …, x_k) is
// Σ_i x_i s_i with s_k = 1 and s_i = s_{i+1} n_{i+1}. A factor with an empty scope is a constant.
//
//   product(φ, ψ)          (φψ)(x) = φ(x_φ) ψ(x_ψ) over the union of the scopes, φ's variables first, then ψ's
//                          not in φ, in their order (Koller and Friedman 2009, Probabilistic Graphical Models,
//                          ch. 4 and ch. 9, whose incremental index walk is used)
//   sum_out(φ, v)          Σ_{x_v} φ: v leaves the scope (marginalisation)
//   max_out(φ, v)          max_{x_v} φ (max-marginalisation, for the most probable explanation)
//   reduce(φ, v, s)        φ restricted to x_v = s: v leaves the scope (conditioning on hard evidence)
//   marginalise(φ, keep)   every variable but those in keep summed out, the scope in keep's order
//   reorder(φ, scope)      the same function, its scope permuted
//   total(φ), normalise(φ) Σ φ (compensated, Neumaier 1974), and φ/Σφ (refused when Σφ is zero or not finite)
//
// Costs: product O(|φψ| · k), sum_out, max_out and reduce O(|φ|), reorder O(|φ| · k). Exact in floating point up
// to the rounding of one product or one sum per entry.

#pragma once

#include <tyche/graphical/common.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tyche::graphical {

struct Factor {
  std::vector<std::size_t> scope;  // the variables, distinct
  std::vector<std::size_t> cards;  // the number of states of each
  std::vector<double> values;      // Π cards entries, the last variable varying fastest

  std::size_t size() const { return values.size(); }
  // The position of variable v in the scope, or scope.size() when it is absent.
  std::size_t position(std::size_t v) const {
    return static_cast<std::size_t>(std::find(scope.begin(), scope.end(), v) - scope.begin());
  }
  // The index of an assignment given in the order of the scope.
  std::size_t index(std::span<const std::size_t> states) const {
    detail::require(states.size() == scope.size(), "factor: one state for each variable of the scope");
    std::size_t i = 0;
    for (std::size_t k = 0; k < scope.size(); k++) {
      detail::require(states[k] < cards[k], "factor: a state out of range");
      i = i * cards[k] + states[k];
    }
    return i;
  }
  double at(std::span<const std::size_t> states) const { return values[index(states)]; }
};

namespace detail {

inline std::size_t table_size(const std::vector<std::size_t>& cards) {
  std::size_t n = 1;
  for (std::size_t c : cards) {
    require(c >= 1, "factor: every variable has at least one state");
    require(n <= std::numeric_limits<std::size_t>::max() / c, "factor: the table is too large");
    n *= c;
  }
  return n;
}

inline std::vector<std::size_t> strides(const std::vector<std::size_t>& cards) {
  std::vector<std::size_t> s(cards.size(), 1);
  for (std::size_t k = cards.size(); k-- > 1;) s[k - 1] = s[k] * cards[k];
  return s;
}

// The stride of each variable of `scope` within f (0 when f does not hold it).
inline std::vector<std::size_t> strides_in(const Factor& f, const std::vector<std::size_t>& scope) {
  const auto s = strides(f.cards);
  std::vector<std::size_t> out(scope.size(), 0);
  for (std::size_t k = 0; k < scope.size(); k++) {
    const std::size_t p = f.position(scope[k]);
    if (p < f.scope.size()) out[k] = s[p];
  }
  return out;
}

inline double neumaier(std::span<const double> v) {
  double s = 0, c = 0;
  for (double x : v) {
    const double t = s + x;
    c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s;
    s = t;
  }
  return s + c;
}

}  // namespace detail

// A factor, validated: distinct variables, at least one state each, Π cards values, each finite and non-negative.
inline Factor factor(std::vector<std::size_t> scope, std::vector<std::size_t> cards, std::vector<double> values) {
  detail::require(scope.size() == cards.size(), "factor: one number of states for each variable");
  for (std::size_t i = 0; i < scope.size(); i++)
    for (std::size_t j = i + 1; j < scope.size(); j++) detail::require(scope[i] != scope[j], "factor: a variable appears twice in the scope");
  detail::require(values.size() == detail::table_size(cards), "factor: the table holds Π cards values");
  for (double v : values) detail::require(std::isfinite(v) && v >= 0, "factor: every value is finite and non-negative");
  return Factor{std::move(scope), std::move(cards), std::move(values)};
}

inline Factor constant(double c) { return Factor{{}, {}, {c}}; }

inline Factor product(const Factor& a, const Factor& b) {
  Factor r;
  r.scope = a.scope;
  r.cards = a.cards;
  for (std::size_t k = 0; k < b.scope.size(); k++) {
    const std::size_t p = a.position(b.scope[k]);
    if (p < a.scope.size()) {
      detail::require(a.cards[p] == b.cards[k], "product: a variable has different numbers of states in the two factors");
    } else {
      r.scope.push_back(b.scope[k]);
      r.cards.push_back(b.cards[k]);
    }
  }
  r.values.assign(detail::table_size(r.cards), 0.0);
  const auto sa = detail::strides_in(a, r.scope), sb = detail::strides_in(b, r.scope);
  const std::size_t k = r.scope.size();
  std::vector<std::size_t> x(k, 0);
  std::size_t ia = 0, ib = 0;
  for (std::size_t i = 0; i < r.values.size(); i++) {
    r.values[i] = a.values[ia] * b.values[ib];
    for (std::size_t j = k; j-- > 0;) {
      x[j]++;
      ia += sa[j];
      ib += sb[j];
      if (x[j] < r.cards[j]) break;
      ia -= sa[j] * r.cards[j];
      ib -= sb[j] * r.cards[j];
      x[j] = 0;
    }
  }
  return r;
}

inline Factor product(const std::vector<Factor>& fs) {
  if (fs.empty()) return constant(1);
  Factor r = fs[0];
  for (std::size_t i = 1; i < fs.size(); i++) r = product(r, fs[i]);
  return r;
}

namespace detail {
// Sums (or maximises) out the variable at position p.
inline Factor collapse(const Factor& f, std::size_t p, bool maximise) {
  Factor r;
  for (std::size_t k = 0; k < f.scope.size(); k++)
    if (k != p) {
      r.scope.push_back(f.scope[k]);
      r.cards.push_back(f.cards[k]);
    }
  const std::size_t low = strides(f.cards)[p], n = f.cards[p];
  r.values.assign(f.values.size() / n, maximise ? -std::numeric_limits<double>::infinity() : 0.0);
  for (std::size_t i = 0; i < f.values.size(); i++) {
    const std::size_t j = (i / (low * n)) * low + i % low;
    if (maximise) r.values[j] = std::max(r.values[j], f.values[i]);
    else r.values[j] += f.values[i];
  }
  return r;
}
}  // namespace detail

inline Factor sum_out(const Factor& f, std::size_t variable) {
  const std::size_t p = f.position(variable);
  return p < f.scope.size() ? detail::collapse(f, p, false) : f;
}

inline Factor max_out(const Factor& f, std::size_t variable) {
  const std::size_t p = f.position(variable);
  return p < f.scope.size() ? detail::collapse(f, p, true) : f;
}

inline Factor reduce(const Factor& f, std::size_t variable, std::size_t state) {
  const std::size_t p = f.position(variable);
  if (p == f.scope.size()) return f;
  detail::require(state < f.cards[p], "reduce: a state out of range");
  Factor r;
  for (std::size_t k = 0; k < f.scope.size(); k++)
    if (k != p) {
      r.scope.push_back(f.scope[k]);
      r.cards.push_back(f.cards[k]);
    }
  const std::size_t low = detail::strides(f.cards)[p], n = f.cards[p];
  r.values.assign(f.values.size() / n, 0.0);
  for (std::size_t j = 0; j < r.values.size(); j++) r.values[j] = f.values[(j / low) * low * n + state * low + j % low];
  return r;
}

inline Factor reorder(const Factor& f, const std::vector<std::size_t>& scope) {
  detail::require(scope.size() == f.scope.size(), "reorder: the new scope is a permutation of the old");
  Factor r;
  r.scope = scope;
  for (std::size_t v : scope) {
    const std::size_t p = f.position(v);
    detail::require(p < f.scope.size(), "reorder: the new scope is a permutation of the old");
    r.cards.push_back(f.cards[p]);
  }
  for (std::size_t i = 0; i < scope.size(); i++)
    for (std::size_t j = i + 1; j < scope.size(); j++) detail::require(scope[i] != scope[j], "reorder: a variable appears twice");
  r.values.assign(f.values.size(), 0.0);
  const auto sf = detail::strides_in(f, r.scope);
  const std::size_t k = r.scope.size();
  std::vector<std::size_t> x(k, 0);
  std::size_t src = 0;
  for (std::size_t i = 0; i < r.values.size(); i++) {
    r.values[i] = f.values[src];
    for (std::size_t j = k; j-- > 0;) {
      x[j]++;
      src += sf[j];
      if (x[j] < r.cards[j]) break;
      src -= sf[j] * r.cards[j];
      x[j] = 0;
    }
  }
  return r;
}

inline Factor marginalise(const Factor& f, const std::vector<std::size_t>& keep) {
  for (std::size_t v : keep) detail::require(f.position(v) < f.scope.size(), "marginalise: a variable kept is not in the scope");
  Factor r = f;
  for (std::size_t v : f.scope)
    if (std::find(keep.begin(), keep.end(), v) == keep.end()) r = sum_out(r, v);
  return reorder(r, keep);
}

inline double total(const Factor& f) { return detail::neumaier(f.values); }

inline Factor normalise(const Factor& f) {
  const double z = total(f);
  detail::require(std::isfinite(z) && z > 0, "normalise: the factor sums to " + std::to_string(z) + ", not a positive number");
  Factor r = f;
  for (double& v : r.values) v /= z;
  return r;
}

}  // namespace tyche::graphical
