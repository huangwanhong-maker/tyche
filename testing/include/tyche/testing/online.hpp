// Online control of the false discovery rate (decision 0027): hypotheses that arrive one at a time, each decided before the
// next arrives and never revised, as when a search admits what it finds to an archive in the order it finds it.
//
// Every procedure here spends a level α_t on the t-th hypothesis, rejects when its p-value is at most α_t (its e-value at
// least 1/α_t), and lets the levels it spends later depend on the rejections so far. They differ in what they assume and
// what they guarantee:
//
// - LOND (Javanmard and Montanari 2018, Ann. Statist. 46, 526–554, §2): α_t = α γ_t (D_{t−1} + 1), D_{t−1} the rejections
//   before t. FDR ≤ α under independence of the null p-values and, as Javanmard and Montanari prove, under positive regression
//   dependence (PRDS) of all the p-values.
// - LORD++ (Ramdas, Yang, Wainwright and Jordan 2017, NeurIPS, the LORD++ rule; after Javanmard and Montanari's LORD):
//   α_t = γ_t W₀ + (α − W₀) γ_{t−τ₁} + α Σ_{j≥2} γ_{t−τ_j}, τ_j the time of the j-th rejection, the terms present once τ_j < t.
//   FDR ≤ α when the null p-values are independent of each other and of the non-nulls.
// - SAFFRON (Ramdas, Zrnic, Wainwright and Jordan 2018, ICML, PMLR 80, 4286–4294): it counts candidates, p_t ≤ λ, and spends
//   only on what is not a candidate: α_t = min{λ, (1 − λ)[W₀ γ_{t−C₀₊} + (α − W₀) γ_{t−τ₁−C₁₊} + α Σ_{j≥2} γ_{t−τ_j−C_j₊}]},
//   C_j₊ the candidates after τ_j and before t (C₀₊ those before t). FDR ≤ α under the same independence; adaptive to the
//   proportion of nulls, as Storey's procedure is offline.
// - α-investing (Foster and Stine 2008, JRSS B 70, 429–444): a wealth, W₀ at the start (α/2 by default); after each test
//   W ← W − α_t/(1 − α_t) on an acceptance and W ← W + ω on a rejection (ω ≤ α); the level α_t = W/(1 + t − τ_last), a share of
//   the wealth that shrinks with the time since the last rejection, capped at W/(1 + W) so that the wealth stays non-negative.
//   With W₀ ≤ α(1 − α) it controls the marginal FDR, E[V]/(E[R] + 1 − α) ≤ α, under independence (their Theorem 1).
// - e-LOND (Xu and Ramdas 2024, AISTATS, PMLR 238, 3997–4005): the LOND levels, an e-value rejected when e_t ≥ 1/α_t. FDR ≤ α
//   under arbitrary dependence: the safe default for tests of overlapping groups on one sample.
//
// The sequences γ are non-increasing, positive and sum to at most 1. Two are given: Javanmard and Montanari's
// γ_j ∝ log(max(j, 2)) / (j e^{√log j}), and γ_j ∝ j^{−s}, s > 1 (Ramdas et al. 2018 use s = 1.6 for SAFFRON). Their normalising
// sums are computed here, not copied: the first by its sum to 10⁶ and the exact integral of the tail, 2e^{−a}(a³ + 3a² + 6a + 6)
// with a = √log N (the sum is 12.6451078729, so the constant is 0.0790820; the 0.07720838 that Javanmard and Montanari print
// makes the sum 0.976, valid but short of 1); the second by Euler–Maclaurin.
//
// Limits: decisions are final, and a later rejection never revives an earlier acceptance. The guarantees are for the order in
// which the hypotheses are tested, which must not depend on their own p-values.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyche::testing {

// A non-increasing sequence γ₁, γ₂, … summing to 1.
class Spending {
 public:
  enum class Kind { lord, power };

  static Spending lord() { return Spending(Kind::lord, 0); }
  static Spending power(double s) {
    if (!(s > 1)) throw std::invalid_argument("Spending::power: the exponent exceeds 1, or the sequence does not sum");
    return Spending(Kind::power, s);
  }

  double operator()(std::size_t j) const {
    if (j == 0) return 0;
    return raw(static_cast<double>(j)) / total_;
  }
  double normaliser() const { return 1 / total_; }
  Kind kind() const { return kind_; }

 private:
  Spending(Kind k, double s) : kind_(k), s_(s) {
    const std::size_t N = k == Kind::lord ? 1000000 : 2000;
    double sum = 0;
    for (std::size_t j = N - 1; j >= 1; j--) sum += raw(static_cast<double>(j));  // smallest first
    const double n = static_cast<double>(N);
    if (k == Kind::lord) {
      // Σ_{j ≥ N} f(j) ≈ ∫_N^∞ f + f(N)/2, with ∫_N^∞ log x/(x e^{√log x}) dx = 2e^{−a}(a³ + 3a² + 6a + 6), a = √log N.
      const double a = std::sqrt(std::log(n));
      sum += 2 * std::exp(-a) * (a * a * a + 3 * a * a + 6 * a + 6) + raw(n) / 2;
    } else {
      // Euler–Maclaurin: Σ_{j ≥ N} j^{−s} = N^{1−s}/(s − 1) + N^{−s}/2 + s N^{−s−1}/12 − …
      sum += std::pow(n, 1 - s_) / (s_ - 1) + std::pow(n, -s_) / 2 + s_ * std::pow(n, -s_ - 1) / 12;
    }
    total_ = sum;
  }
  double raw(double j) const {
    if (kind_ == Kind::power) return std::pow(j, -s_);
    const double l = std::log(std::max(j, 2.0));
    return l / (j * std::exp(std::sqrt(std::log(j))));
  }
  Kind kind_;
  double s_ = 0;
  double total_ = 1;
};

// One hypothesis decided online.
struct OnlineDecision {
  double level = 0;      // α_t spent (for e-LOND, the e-value must reach 1/level)
  bool rejected = false;
  double wealth = std::numeric_limits<double>::quiet_NaN();  // α-investing's wealth after the test
  bool candidate = false;  // SAFFRON: p_t ≤ λ
};

enum class OnlineRule { lond, lord_plus, saffron, alpha_investing, e_lond };

inline const char* name(OnlineRule r) {
  switch (r) {
    case OnlineRule::lond: return "lond";
    case OnlineRule::lord_plus: return "lord++";
    case OnlineRule::saffron: return "saffron";
    case OnlineRule::alpha_investing: return "alpha-investing";
    default: return "e-lond";
  }
}

inline OnlineRule online_rule(const std::string& s) {
  for (OnlineRule r : {OnlineRule::lond, OnlineRule::lord_plus, OnlineRule::saffron, OnlineRule::alpha_investing, OnlineRule::e_lond})
    if (s == name(r)) return r;
  throw std::invalid_argument("no online rule '" + s + "' (there are lond, lord++, saffron, alpha-investing, e-lond)");
}

// The procedures share one object: the hypotheses are given one at a time by test(), each with its p-value (or, for e-LOND,
// its e-value), and the object keeps what the rule needs of the past. Its state is plain, so a caller can save and resume it.
class Online {
 public:
  struct Options {
    OnlineRule rule = OnlineRule::lond;
    double alpha = 0.05;
    double initial = std::numeric_limits<double>::quiet_NaN();  // W₀ (LORD++, SAFFRON, α-investing): α/2 by default
    double lambda = 0.5;                                        // SAFFRON's candidate level
    double payout = std::numeric_limits<double>::quiet_NaN();   // α-investing's ω: α by default
  };

  explicit Online(Options o) : o_(o), gamma_(o.rule == OnlineRule::saffron ? Spending::power(1.6) : Spending::lord()) { start(); }
  Online(Options o, Spending gamma) : o_(o), gamma_(gamma) { start(); }

  // The level the next hypothesis would be given, before its p-value is seen.
  double next_level() const {
    const std::size_t t = tests_ + 1;
    switch (o_.rule) {
      case OnlineRule::lond:
      case OnlineRule::e_lond: return o_.alpha * gamma_(t) * static_cast<double>(rejections_.size() + 1);
      case OnlineRule::lord_plus: {
        double a = gamma_(t) * w0_;
        for (std::size_t j = 0; j < rejections_.size(); j++) a += (j == 0 ? o_.alpha - w0_ : o_.alpha) * gamma_(t - rejections_[j]);
        return a;
      }
      case OnlineRule::saffron: {
        // Candidates after each rejection and before t.
        double a = w0_ * gamma_(t - candidates_before(0, t));
        for (std::size_t j = 0; j < rejections_.size(); j++) {
          const std::size_t tau = rejections_[j];
          a += (j == 0 ? o_.alpha - w0_ : o_.alpha) * gamma_(t - tau - candidates_before(tau, t));
        }
        return std::min(o_.lambda, (1 - o_.lambda) * a);
      }
      default: {  // α-investing: spend a share of the wealth that shrinks with the time since the last rejection
        const std::size_t last = rejections_.empty() ? 0 : rejections_.back();
        const double a = wealth_ / (1 + static_cast<double>(t - last));
        return std::min(a, wealth_ / (1 + wealth_));  // so that W − α_t/(1 − α_t) ≥ 0
      }
    }
  }

  // Decides the next hypothesis: a p-value, or for e-LOND an e-value.
  OnlineDecision test(double value) {
    if (o_.rule == OnlineRule::e_lond) {
      if (!(value >= 0) || std::isnan(value)) throw std::invalid_argument("Online: an e-value is a non-negative number");
    } else if (!(value >= 0 && value <= 1)) {
      throw std::invalid_argument("Online: a p-value lies in [0, 1]");
    }
    OnlineDecision d;
    d.level = next_level();
    const std::size_t t = ++tests_;
    if (o_.rule == OnlineRule::e_lond) d.rejected = d.level > 0 && value >= 1 / d.level;
    else d.rejected = value <= d.level;
    if (o_.rule == OnlineRule::saffron) {
      d.candidate = value <= o_.lambda;
      candidate_.push_back(d.candidate ? 1 : 0);
    }
    if (o_.rule == OnlineRule::alpha_investing) {
      if (d.rejected) wealth_ += payout_;
      else wealth_ -= d.level / (1 - d.level);
      wealth_ = std::max(0.0, wealth_);
      d.wealth = wealth_;
    }
    if (d.rejected) rejections_.push_back(t);
    return d;
  }

  std::vector<OnlineDecision> run(std::span<const double> values) {
    std::vector<OnlineDecision> out;
    out.reserve(values.size());
    for (double v : values) out.push_back(test(v));
    return out;
  }

  std::size_t tests() const { return tests_; }
  const std::vector<std::size_t>& rejections() const { return rejections_; }  // 1-based times
  double wealth() const { return wealth_; }
  const Options& options() const { return o_; }

 private:
  void start() {
    if (!(o_.alpha > 0 && o_.alpha < 1)) throw std::invalid_argument("Online: α lies strictly between 0 and 1");
    w0_ = std::isnan(o_.initial) ? o_.alpha / 2 : o_.initial;
    if (o_.rule == OnlineRule::lord_plus || o_.rule == OnlineRule::saffron) {
      if (!(w0_ > 0 && w0_ <= o_.alpha)) throw std::invalid_argument("Online: the initial wealth lies in (0, α]");
    }
    if (o_.rule == OnlineRule::saffron && !(o_.lambda > 0 && o_.lambda < 1)) throw std::invalid_argument("Online: SAFFRON's λ lies in (0, 1)");
    if (o_.rule == OnlineRule::alpha_investing) {
      wealth_ = std::isnan(o_.initial) ? o_.alpha / 2 : o_.initial;
      payout_ = std::isnan(o_.payout) ? o_.alpha : o_.payout;
      if (!(wealth_ > 0) || !(payout_ > 0 && payout_ <= o_.alpha)) throw std::invalid_argument("Online: α-investing needs W₀ > 0 and 0 < ω ≤ α");
    }
  }
  // Candidates among the tests s + 1, …, t − 1 (1-based).
  std::size_t candidates_before(std::size_t s, std::size_t t) const {
    std::size_t c = 0;
    for (std::size_t i = s + 1; i < t; i++) c += candidate_[i - 1];
    return c;
  }

  Options o_;
  Spending gamma_;
  double w0_ = 0;
  double wealth_ = std::numeric_limits<double>::quiet_NaN();
  double payout_ = 0;
  std::size_t tests_ = 0;
  std::vector<std::size_t> rejections_;
  std::vector<char> candidate_;
};

}  // namespace tyche::testing
