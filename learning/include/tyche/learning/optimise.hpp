// First-order stochastic optimisers of a parameter vector from its gradients: Adam, with decoupled
// weight decay (AdamW), and gradient descent with momentum.
//
// - Adam (Kingma and Ba 2015, Proc. ICLR, Algorithm 1): with gradient g_t at step t,
//     m_t = β₁ m_{t−1} + (1 − β₁) g_t,   v_t = β₂ v_{t−1} + (1 − β₂) g_t²,
//     m̂_t = m_t/(1 − β₁ᵗ),   v̂_t = v_t/(1 − β₂ᵗ),   θ_t = θ_{t−1} − α m̂_t/(√v̂_t + ε),
//   α = 10⁻³, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ by default (their §2). Its first step moves every
//   coordinate by α sign(g₁) (|m̂₁/√v̂₁| = 1), a property the checks use.
// - AdamW (Loshchilov and Hutter 2019, Proc. ICLR, Algorithm 2, line 12): weight decay applied to θ
//   directly, θ_t = θ_{t−1} − η_t (α′ m̂_t/(√v̂_t + ε) + λ θ_{t−1}), not through the gradient, so that
//   it is not rescaled by v̂; here with α′ = 1 and the schedule multiplier η_t = α, as PyTorch's
//   AdamW parameterises it: θ_t = θ_{t−1} − α (m̂_t/(√v̂_t + ε) + λ θ_{t−1}); λ = 0 gives Adam.
// - Momentum (Polyak 1964; Sutskever, Martens, Dahl and Hinton 2013, Proc. ICML, eq. 1):
//   v_t = μ v_{t−1} − α g_t, θ_t = θ_{t−1} + v_t.
//
// The moments are kept in double whatever the parameters' type.

#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::learning {

struct AdamOptions {
  double rate = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
  double weight_decay = 0;  // λ, decoupled (AdamW)
};

class Adam {
 public:
  explicit Adam(std::size_t n, AdamOptions o = {}) : o_(o), m_(n, 0.0), v_(n, 0.0) {}
  template <class T>
  void step(std::span<T> theta, std::span<const T> g) {
    if (theta.size() != m_.size() || g.size() != m_.size()) throw std::invalid_argument("tyche::learning::Adam: vectors of the optimiser's size");
    t_++;
    const double c1 = 1 - std::pow(o_.beta1, static_cast<double>(t_)), c2 = 1 - std::pow(o_.beta2, static_cast<double>(t_));
    for (std::size_t i = 0; i < m_.size(); i++) {
      const double gi = static_cast<double>(g[i]);
      m_[i] = o_.beta1 * m_[i] + (1 - o_.beta1) * gi;
      v_[i] = o_.beta2 * v_[i] + (1 - o_.beta2) * gi * gi;
      const double update = (m_[i] / c1) / (std::sqrt(v_[i] / c2) + o_.epsilon) + o_.weight_decay * static_cast<double>(theta[i]);
      theta[i] = static_cast<T>(static_cast<double>(theta[i]) - o_.rate * update);
    }
  }
  std::size_t steps() const { return t_; }
  const AdamOptions& options() const { return o_; }
  void set_rate(double rate) { o_.rate = rate; }

 private:
  AdamOptions o_;
  std::vector<double> m_, v_;
  std::size_t t_ = 0;
};

class Momentum {
 public:
  Momentum(std::size_t n, double rate, double mu = 0.9) : rate_(rate), mu_(mu), v_(n, 0.0) {}
  template <class T>
  void step(std::span<T> theta, std::span<const T> g) {
    if (theta.size() != v_.size() || g.size() != v_.size()) throw std::invalid_argument("tyche::learning::Momentum: vectors of the optimiser's size");
    for (std::size_t i = 0; i < v_.size(); i++) {
      v_[i] = mu_ * v_[i] - rate_ * static_cast<double>(g[i]);
      theta[i] = static_cast<T>(static_cast<double>(theta[i]) + v_[i]);
    }
  }

 private:
  double rate_, mu_;
  std::vector<double> v_;
};

}  // namespace tyche::learning
