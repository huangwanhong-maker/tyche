// A feed-forward network as a sequence of layers, its loss, the gradient of the loss by
// backpropagation, and the gradient by central differences that checks it.
//
// - The network is the composition f = f_K ∘ … ∘ f_1 of its layers (layers.hpp); its parameters
//   are one flat vector θ, each layer owning a slice, so that an optimiser sees one vector.
// - Losses of a prediction ŷ against a target y, summed over outputs: squared error (ŷ − y)²,
//   whose gradient is 2(ŷ − y); Huber's ρ_δ(r) = r²/2 for |r| ≤ δ and δ(|r| − δ/2) beyond, its
//   gradient clip(r, −δ, δ) (Huber 1964, Ann. Math. Statist. 35, 73–101, eq. 8); Poisson deviance
//   with a log link, ŷ the log of the mean: e^ŷ − y ŷ, gradient e^ŷ − y.
// - Backpropagation: the chain rule applied from the loss backwards, each layer's backward map a
//   vector–Jacobian product (Rumelhart, Hinton and Williams 1986), so that the whole gradient costs
//   a small multiple of a forward pass (Griewank and Walther 2008, Evaluating Derivatives, §4.6).
// - The check: ∂L/∂θ_i ≈ (L(θ + h e_i) − L(θ − h e_i))/(2h), whose error is O(h²) from truncation
//   and O(ε L/h) from rounding (Nocedal and Wright 2006, Numerical Optimization, §8.1, eq. 8.6);
//   with T = double and h ~ 10⁻⁵ they agree to about 10⁻⁸ relative where the loss is smooth. ReLU
//   and max pooling are not differentiable where an input is exactly 0 or ties another; random
//   inputs avoid those sets with probability 1.
//
// Thread safety: forward and backward read the network and write only the workspace and the
// gradient they are given; one network serves many threads, each with its own workspace.

#pragma once

#include <tyche/learning/layers.hpp>

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyche::learning {

inline constexpr std::uint32_t INITIALISE = 0x494e4954;  // "INIT": the draws of the initial parameters

enum class LossKind { squared, huber, poisson };

struct Loss {
  LossKind kind = LossKind::squared;
  double delta = 1;  // Huber's δ

  template <class T>
  T value(T yhat, T y) const {
    const T r = yhat - y;
    switch (kind) {
      case LossKind::squared: return r * r;
      case LossKind::huber: return std::fabs(r) <= delta ? T(0.5) * r * r : static_cast<T>(delta) * (std::fabs(r) - static_cast<T>(0.5 * delta));
      case LossKind::poisson: return std::exp(yhat) - y * yhat;
    }
    return 0;
  }
  template <class T>
  T gradient(T yhat, T y) const {
    const T r = yhat - y;
    switch (kind) {
      case LossKind::squared: return 2 * r;
      case LossKind::huber: return std::clamp(r, static_cast<T>(-delta), static_cast<T>(delta));
      case LossKind::poisson: return std::exp(yhat) - y;
    }
    return 0;
  }
};

// The activations and their gradients of one pass of one example.
template <class T>
struct Workspace {
  std::vector<std::vector<T>> a, g;  // a[0] the input, a[k] the output of layer k
};

template <class T>
class Network {
 public:
  explicit Network(Shape input) : input_(input) {}
  Network(const Network&) = delete;
  Network& operator=(const Network&) = delete;
  Network(Network&&) = default;
  Network& operator=(Network&&) = default;

  template <class L, class... Args>
  Network& add(Args&&... args) {
    return add(std::make_unique<L>(std::forward<Args>(args)...));
  }
  Network& add(std::unique_ptr<Layer<T>> layer) {
    const Shape in = shapes_.empty() ? input_ : shapes_.back();
    const Shape out = layer->output(in);
    offsets_.push_back(count_);
    count_ += layer->parameters(in);
    shapes_.push_back(out);
    layers_.push_back(std::move(layer));
    theta.assign(count_, T(0));
    return *this;
  }

  // Draws every layer's initial parameters from one stream per layer.
  void initialise(std::uint64_t seed) {
    theta.assign(count_, T(0));
    for (std::size_t k = 0; k < layers_.size(); k++) {
      calx::random::Stream r(seed, INITIALISE, static_cast<std::uint32_t>(k), 0);
      layers_[k]->initialise(shape_in(k), slice(theta, k), r);
    }
  }

  std::vector<T> theta;  // the parameters

  std::size_t size() const { return count_; }
  std::size_t layers() const { return layers_.size(); }
  const Layer<T>& layer(std::size_t k) const { return *layers_[k]; }
  Shape input() const { return input_; }
  Shape output() const { return shapes_.empty() ? input_ : shapes_.back(); }
  Shape shape_in(std::size_t k) const { return k == 0 ? input_ : shapes_[k - 1]; }
  Shape shape_out(std::size_t k) const { return shapes_[k]; }
  std::size_t offset(std::size_t k) const { return offsets_[k]; }
  std::size_t parameters(std::size_t k) const { return (k + 1 < offsets_.size() ? offsets_[k + 1] : count_) - offsets_[k]; }
  std::span<const T> slice(std::size_t k) const { return std::span<const T>(theta).subspan(offsets_[k], parameters(k)); }
  std::span<T> slice(std::vector<T>& v, std::size_t k) const { return std::span<T>(v).subspan(offsets_[k], parameters(k)); }
  std::string describe() const {
    std::string s;
    for (std::size_t k = 0; k < layers_.size(); k++) s += (k ? " → " : "") + layers_[k]->name();
    return s;
  }

  void prepare(Workspace<T>& w) const {
    w.a.resize(layers_.size() + 1);
    w.g.resize(layers_.size() + 1);
    w.a[0].resize(input_.size());
    w.g[0].resize(input_.size());
    for (std::size_t k = 0; k < layers_.size(); k++) {
      w.a[k + 1].resize(shapes_[k].size());
      w.g[k + 1].resize(shapes_[k].size());
    }
  }

  // The forward pass; the output is w.a.back().
  std::span<const T> forward(std::span<const T> x, Workspace<T>& w, Pass pass = {}) const {
    if (x.size() != input_.size()) throw std::invalid_argument("tyche::learning::Network: an input of the wrong size");
    if (w.a.size() != layers_.size() + 1) prepare(w);
    std::copy(x.begin(), x.end(), w.a[0].begin());
    for (std::size_t k = 0; k < layers_.size(); k++) {
      pass.layer = static_cast<std::uint32_t>(k);
      layers_[k]->forward(shape_in(k), w.a[k], w.a[k + 1], slice(k), pass);
    }
    return w.a.back();
  }

  // The loss of one example, its gradient added into `gradient` (of the size of θ).
  T accumulate(std::span<const T> x, std::span<const T> y, const Loss& loss, std::span<T> gradient, Workspace<T>& w, Pass pass = {}) const {
    const auto yhat = forward(x, w, pass);
    T value = 0;
    auto& gy = w.g.back();
    for (std::size_t i = 0; i < yhat.size(); i++) {
      value += loss.value(yhat[i], y[i]);
      gy[i] = loss.gradient(yhat[i], y[i]);
    }
    for (std::size_t k = layers_.size(); k-- > 0;) {
      pass.layer = static_cast<std::uint32_t>(k);
      std::span<T> gx = k == 0 ? std::span<T>() : std::span<T>(w.g[k]);
      layers_[k]->backward(shape_in(k), w.a[k], w.a[k + 1], w.g[k + 1], gx, slice(k), gradient.subspan(offsets_[k], parameters(k)), pass);
    }
    return value;
  }

  // The loss of one example.
  T loss_of(std::span<const T> x, std::span<const T> y, const Loss& loss, Workspace<T>& w, Pass pass = {}) const {
    const auto yhat = forward(x, w, pass);
    T value = 0;
    for (std::size_t i = 0; i < yhat.size(); i++) value += loss.value(yhat[i], y[i]);
    return value;
  }

 private:
  Shape input_;
  std::vector<std::unique_ptr<Layer<T>>> layers_;
  std::vector<Shape> shapes_;
  std::vector<std::size_t> offsets_;
  std::size_t count_ = 0;
};

// The gradient of the loss of the examples by central differences in θ (for checks).
template <class T>
std::vector<double> numerical_gradient(Network<T>& net, const std::vector<std::vector<T>>& xs, const std::vector<std::vector<T>>& ys, const Loss& loss, double h = 1e-5,
                                       const Pass& pass = {}) {
  Workspace<T> w;
  const auto total = [&] {
    double s = 0;
    for (std::size_t e = 0; e < xs.size(); e++) {
      Pass p = pass;
      p.example = e;
      s += static_cast<double>(net.loss_of(xs[e], ys[e], loss, w, p));
    }
    return s;
  };
  std::vector<double> g(net.size());
  for (std::size_t i = 0; i < net.size(); i++) {
    const T keep = net.theta[i];
    net.theta[i] = static_cast<T>(keep + h);
    const double up = total();
    net.theta[i] = static_cast<T>(keep - h);
    const double down = total();
    net.theta[i] = keep;
    g[i] = (up - down) / (2 * h);
  }
  return g;
}

}  // namespace tyche::learning
