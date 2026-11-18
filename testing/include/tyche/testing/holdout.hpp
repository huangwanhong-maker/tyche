// The reusable holdout (decision 0027): a part of the data kept aside that a search may consult many times, adaptively,
// without the answers fitting the part as plain reuse would.
//
// Thresholdout (Dwork, Feldman, Hardt, Pitassi, Reingold and Roth 2015, Science 349, 636–638, and its supplement). A query φ
// is answered by its value on the training part unless the training value and the holdout value differ by more than a noisy
// threshold; then the holdout value is answered with Laplace noise, and the budget of such answers falls by one:
//
//     T̂ ← T + Lap(2σ)
//     for each query:  if the budget is spent, answer nothing;
//                      if |φ_holdout − φ_train| > T̂ + Lap(4σ):  budget −= 1;  T̂ ← T + Lap(2σ);  answer φ_holdout + Lap(σ)
//                      else answer φ_train.
//
// Its guarantee comes from differential privacy: for statistical queries (means of functions with values in [0, 1]), the answers
// stay within a tolerance of the population's for a number of adaptive queries that grows exponentially in the holdout's size
// for a fixed budget, where plain reuse allows only about as many as the holdout's size. The answer that says the training value
// is told apart from the holdout's by `overfit`, so the caller knows which it was given.
//
// Limits: the guarantee is for bounded statistical queries; a caller that asks about another statistic (a standardised
// difference) scales it into [0, 1] and states that the guarantee is then approximate. The noise is drawn from the caller's
// stream, so a run is reproducible.
#pragma once

#include <calx/random/stream.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>

namespace tyche::testing {

inline double laplace(calx::random::Stream& s, double scale) {
  const double u = s.uniform() - 0.5;
  return -scale * (u < 0 ? -1.0 : 1.0) * std::log(1 - 2 * std::fabs(u));
}

struct HoldoutAnswer {
  double value = 0;
  bool overfit = false;   // the holdout's value was answered, with noise
  bool spent = false;     // the budget was spent: no answer
};

class Thresholdout {
 public:
  struct Options {
    double threshold = 0.04;  // T
    double sigma = 0.01;      // σ, the Laplace scale of the answers
    std::size_t budget = 32;  // B: how many times the holdout may be told
  };

  // What a Thresholdout must keep to be resumed: the noisy threshold and the counts.
  struct Saved {
    double threshold = 0;
    std::size_t left = 0, queries = 0, overfits = 0;
  };

  Thresholdout(Options o, calx::random::Stream& s) : o_(o), s_(&s) {
    if (!(o_.threshold > 0) || !(o_.sigma >= 0)) throw std::invalid_argument("Thresholdout: T > 0 and σ ≥ 0");
    left_ = o_.budget;
    redraw();
  }
  Thresholdout(Options o, calx::random::Stream& s, const Saved& v) : o_(o), s_(&s), that_(v.threshold), left_(v.left), queries_(v.queries), overfits_(v.overfits) {
    if (!(o_.threshold > 0) || !(o_.sigma >= 0)) throw std::invalid_argument("Thresholdout: T > 0 and σ ≥ 0");
  }
  Saved saved() const { return Saved{that_, left_, queries_, overfits_}; }
  void use(calx::random::Stream& s) { s_ = &s; }

  HoldoutAnswer ask(double train, double holdout) {
    HoldoutAnswer a;
    queries_++;
    if (left_ == 0) {
      a.spent = true;
      return a;
    }
    const double eta = o_.sigma > 0 ? laplace(*s_, 4 * o_.sigma) : 0.0;
    if (std::fabs(holdout - train) > that_ + eta) {
      left_--, overfits_++;
      redraw();
      a.value = holdout + (o_.sigma > 0 ? laplace(*s_, o_.sigma) : 0.0);
      a.overfit = true;
    } else {
      a.value = train;
    }
    return a;
  }

  std::size_t left() const { return left_; }
  std::size_t queries() const { return queries_; }
  std::size_t overfits() const { return overfits_; }
  const Options& options() const { return o_; }

 private:
  void redraw() { that_ = o_.threshold + (o_.sigma > 0 ? laplace(*s_, 2 * o_.sigma) : 0.0); }

  Options o_;
  calx::random::Stream* s_;
  double that_ = 0;
  std::size_t left_ = 0, queries_ = 0, overfits_ = 0;
};

}  // namespace tyche::testing
