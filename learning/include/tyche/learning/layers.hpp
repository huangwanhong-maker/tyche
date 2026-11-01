// The layers of a feed-forward network: affine maps, one-dimensional convolutions, rectifiers,
// pooling and dropout, each with its forward map and the backward map of reverse-mode
// differentiation (backpropagation; Rumelhart, Hinton and Williams 1986, Nature 323, 533–536).
//
// A layer is a map y = f(x; θ) from an input of shape (channels, length), stored channel by
// channel, to an output of the shape output(in) says, with parameters θ that are a slice of the
// network's one flat vector. Its backward map receives ∂L/∂y and adds ∂L/∂θ into the gradient's
// slice, and writes ∂L/∂x (the vector–Jacobian products of the chain rule). Layers keep no state
// between calls: what the backward map needs it reads from the input and output of the forward one,
// which the network keeps, so that one layer serves many threads at once.
//
// - Dense: y = W x + b, x flattened (channels × length); ∂L/∂W = (∂L/∂y) xᵀ, ∂L/∂b = ∂L/∂y,
//   ∂L/∂x = Wᵀ ∂L/∂y. Initialised by He et al.'s rule for rectifiers, W ~ N(0, 2/fan_in) (He, Zhang,
//   Ren and Sun 2015, Proc. ICCV, eq. 10), or Glorot and Bengio's uniform U(±√(6/(fan_in + fan_out)))
//   (2010, Proc. AISTATS, eq. 16); biases 0.
// - Conv1d: y[o, t] = b[o] + Σ_c Σ_k W[o, c, k] x[c, t + k − p], a cross-correlation of the input
//   with each of the filters (the convolution of deep learning; Goodfellow, Bengio and Courville
//   2016, Deep Learning, §9.1, eq. 9.6), x zero outside [0, L); padding p = 0 (valid: output length
//   L − w + 1) or p = ⌊(w − 1)/2⌋ (same: output length L). Its backward map is again a
//   cross-correlation: ∂L/∂W[o, c, k] = Σ_t ∂L/∂y[o, t] x[c, t + k − p], ∂L/∂x[c, s] = Σ_o Σ_k W[o,
//   c, k] ∂L/∂y[o, s − k + p]. Read on a one-hot sequence, a filter's weights are a position weight
//   matrix and its output the score of each window (Alipanahi, Delong, Weirauch and Frey 2015,
//   Nature Biotechnology 33, 831–838).
// - ReLU, max(0, x) (Nair and Hinton 2010), its derivative 1 where x > 0 and 0 elsewhere (0 at 0, a
//   subgradient); Tanh.
// - MaxPool1d: y[c, j] = max_{0 ≤ k < m} x[c, j s + k], window m, stride s (s = m by default; m = 0
//   pools the whole length); the gradient goes to the first maximiser of each window.
// - Dropout: during training each input is kept with probability 1 − p and scaled by 1/(1 − p)
//   ("inverted" dropout, so that prediction is the identity; Srivastava, Hinton, Krizhevsky,
//   Sutskever and Salakhutdinov 2014, JMLR 15, 1929–1958, §10); the mask is a calx random stream
//   keyed by the seed, the layer, the epoch and the example, so a backward pass draws the forward
//   pass's mask again, and training is the same on any number of threads.
//
// Complexity per example: Dense O(F · out); Conv1d O(filters · C · w · L), forward and backward.

#pragma once

#include <calx/random/stream.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace tyche::learning {

inline constexpr std::uint32_t DROPOUT = 0x44524f50;  // "DROP": the masks of dropout

struct Shape {
  std::size_t channels = 1, length = 1;
  std::size_t size() const { return channels * length; }
  bool operator==(const Shape&) const = default;
};

// What a layer is told of the pass it is in.
struct Pass {
  bool training = false;
  std::uint64_t seed = 0;
  std::uint32_t epoch = 0;
  std::uint64_t example = 0;   // the example's index in its data set
  std::uint32_t layer = 0;     // the layer's index in its network
};

enum class Init { he, glorot };

template <class T>
class Layer {
 public:
  virtual ~Layer() = default;
  virtual std::string name() const = 0;
  virtual Shape output(Shape in) const = 0;
  virtual std::size_t parameters(Shape in) const { (void)in; return 0; }
  virtual void initialise(Shape in, std::span<T> theta, calx::random::Stream& r) const { (void)in; (void)theta; (void)r; }
  virtual void forward(Shape in, std::span<const T> x, std::span<T> y, std::span<const T> theta, const Pass& pass) const = 0;
  // Adds ∂L/∂θ into g_theta and writes ∂L/∂x into gx (an empty gx: not wanted).
  virtual void backward(Shape in, std::span<const T> x, std::span<const T> y, std::span<const T> gy, std::span<T> gx, std::span<const T> theta, std::span<T> g_theta,
                        const Pass& pass) const = 0;
};

// ---- dense ------------------------------------------------------------------------------------------
template <class T>
class Dense final : public Layer<T> {
 public:
  explicit Dense(std::size_t outputs, Init init = Init::he) : out_(outputs), init_(init) {
    if (outputs == 0) throw std::invalid_argument("tyche::learning::Dense: at least one output");
  }
  std::string name() const override { return "dense(" + std::to_string(out_) + ")"; }
  Shape output(Shape) const override { return {out_, 1}; }
  std::size_t parameters(Shape in) const override { return out_ * in.size() + out_; }
  void initialise(Shape in, std::span<T> theta, calx::random::Stream& r) const override {
    const std::size_t F = in.size();
    const double sd = std::sqrt(2.0 / static_cast<double>(F)), a = std::sqrt(6.0 / static_cast<double>(F + out_));
    for (std::size_t i = 0; i < out_ * F; i++) theta[i] = static_cast<T>(init_ == Init::he ? r.normal(0, sd) : r.uniform(-a, a));
    for (std::size_t i = out_ * F; i < theta.size(); i++) theta[i] = 0;
  }
  void forward(Shape in, std::span<const T> x, std::span<T> y, std::span<const T> theta, const Pass&) const override {
    const std::size_t F = in.size();
    const T* W = theta.data();
    const T* b = theta.data() + out_ * F;
    for (std::size_t o = 0; o < out_; o++) {
      T s = b[o];
      const T* w = W + o * F;
      for (std::size_t i = 0; i < F; i++) s += w[i] * x[i];
      y[o] = s;
    }
  }
  void backward(Shape in, std::span<const T> x, std::span<const T>, std::span<const T> gy, std::span<T> gx, std::span<const T> theta, std::span<T> g_theta,
                const Pass&) const override {
    const std::size_t F = in.size();
    const T* W = theta.data();
    T* gW = g_theta.data();
    T* gb = g_theta.data() + out_ * F;
    for (std::size_t o = 0; o < out_; o++) {
      const T d = gy[o];
      gb[o] += d;
      T* gw = gW + o * F;
      for (std::size_t i = 0; i < F; i++) gw[i] += d * x[i];
    }
    if (gx.empty()) return;
    std::fill(gx.begin(), gx.end(), T(0));
    for (std::size_t o = 0; o < out_; o++) {
      const T d = gy[o];
      const T* w = W + o * F;
      for (std::size_t i = 0; i < F; i++) gx[i] += w[i] * d;
    }
  }

 private:
  std::size_t out_;
  Init init_;
};

// ---- one-dimensional convolution ------------------------------------------------------------------------
enum class Padding { valid, same };

template <class T>
class Conv1d final : public Layer<T> {
 public:
  Conv1d(std::size_t filters, std::size_t width, Padding padding = Padding::valid, Init init = Init::he) : filters_(filters), width_(width), padding_(padding), init_(init) {
    if (filters == 0 || width == 0) throw std::invalid_argument("tyche::learning::Conv1d: at least one filter of positive width");
  }
  std::string name() const override { return "conv1d(" + std::to_string(filters_) + " × " + std::to_string(width_) + (padding_ == Padding::same ? ", same)" : ")"); }
  std::size_t filters() const { return filters_; }
  std::size_t width() const { return width_; }
  std::size_t pad() const { return padding_ == Padding::same ? (width_ - 1) / 2 : 0; }
  Shape output(Shape in) const override {
    if (padding_ == Padding::valid && in.length < width_) throw std::invalid_argument("tyche::learning::Conv1d: an input shorter than the filters");
    return {filters_, padding_ == Padding::same ? in.length : in.length - width_ + 1};
  }
  std::size_t parameters(Shape in) const override { return filters_ * in.channels * width_ + filters_; }
  void initialise(Shape in, std::span<T> theta, calx::random::Stream& r) const override {
    const std::size_t fan_in = in.channels * width_, fan_out = filters_ * width_, n = filters_ * fan_in;
    const double sd = std::sqrt(2.0 / static_cast<double>(fan_in)), a = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
    for (std::size_t i = 0; i < n; i++) theta[i] = static_cast<T>(init_ == Init::he ? r.normal(0, sd) : r.uniform(-a, a));
    for (std::size_t i = n; i < theta.size(); i++) theta[i] = 0;
  }
  void forward(Shape in, std::span<const T> x, std::span<T> y, std::span<const T> theta, const Pass&) const override {
    const std::size_t C = in.channels, L = in.length, Lo = output(in).length, p = pad();
    const T* W = theta.data();
    const T* b = theta.data() + filters_ * C * width_;
    for (std::size_t o = 0; o < filters_; o++) {
      T* yo = y.data() + o * Lo;
      std::fill(yo, yo + Lo, b[o]);
      for (std::size_t c = 0; c < C; c++) {
        const T* xc = x.data() + c * L;
        for (std::size_t k = 0; k < width_; k++) {
          const T w = W[(o * C + c) * width_ + k];
          // t + k − p ∈ [0, L)  ⇔  t ∈ [max(0, p − k), L + p − k)
          const std::size_t lo = k < p ? p - k : 0, hi = L + p > k ? std::min(Lo, L + p - k) : 0;
          const T* src = xc + (k + lo) - p;
          for (std::size_t t = lo; t < hi; t++) yo[t] += w * src[t - lo];
        }
      }
    }
  }
  void backward(Shape in, std::span<const T> x, std::span<const T>, std::span<const T> gy, std::span<T> gx, std::span<const T> theta, std::span<T> g_theta,
                const Pass&) const override {
    const std::size_t C = in.channels, L = in.length, Lo = output(in).length, p = pad();
    const T* W = theta.data();
    T* gW = g_theta.data();
    T* gb = g_theta.data() + filters_ * C * width_;
    if (!gx.empty()) std::fill(gx.begin(), gx.end(), T(0));
    for (std::size_t o = 0; o < filters_; o++) {
      const T* go = gy.data() + o * Lo;
      T sb = 0;
      for (std::size_t t = 0; t < Lo; t++) sb += go[t];
      gb[o] += sb;
      for (std::size_t c = 0; c < C; c++) {
        const T* xc = x.data() + c * L;
        T* gxc = gx.empty() ? nullptr : gx.data() + c * L;
        for (std::size_t k = 0; k < width_; k++) {
          const std::size_t lo = k < p ? p - k : 0, hi = L + p > k ? std::min(Lo, L + p - k) : 0;
          const T* src = xc + (k + lo) - p;
          T s = 0;
          for (std::size_t t = lo; t < hi; t++) s += go[t] * src[t - lo];
          gW[(o * C + c) * width_ + k] += s;
          if (gxc) {
            const T w = W[(o * C + c) * width_ + k];
            T* dst = gxc + (k + lo) - p;
            for (std::size_t t = lo; t < hi; t++) dst[t - lo] += w * go[t];
          }
        }
      }
    }
  }

 private:
  std::size_t filters_, width_;
  Padding padding_;
  Init init_;
};

// ---- activations ----------------------------------------------------------------------------------------
template <class T>
class Relu final : public Layer<T> {
 public:
  std::string name() const override { return "relu"; }
  Shape output(Shape in) const override { return in; }
  void forward(Shape, std::span<const T> x, std::span<T> y, std::span<const T>, const Pass&) const override {
    for (std::size_t i = 0; i < x.size(); i++) y[i] = x[i] > 0 ? x[i] : T(0);
  }
  void backward(Shape, std::span<const T> x, std::span<const T>, std::span<const T> gy, std::span<T> gx, std::span<const T>, std::span<T>, const Pass&) const override {
    if (gx.empty()) return;
    for (std::size_t i = 0; i < x.size(); i++) gx[i] = x[i] > 0 ? gy[i] : T(0);
  }
};

template <class T>
class Tanh final : public Layer<T> {
 public:
  std::string name() const override { return "tanh"; }
  Shape output(Shape in) const override { return in; }
  void forward(Shape, std::span<const T> x, std::span<T> y, std::span<const T>, const Pass&) const override {
    for (std::size_t i = 0; i < x.size(); i++) y[i] = std::tanh(x[i]);
  }
  void backward(Shape, std::span<const T>, std::span<const T> y, std::span<const T> gy, std::span<T> gx, std::span<const T>, std::span<T>, const Pass&) const override {
    if (gx.empty()) return;
    for (std::size_t i = 0; i < y.size(); i++) gx[i] = gy[i] * (T(1) - y[i] * y[i]);
  }
};

// ---- pooling ----------------------------------------------------------------------------------------------
template <class T>
class MaxPool1d final : public Layer<T> {
 public:
  // window 0: the whole length (global max pooling); stride 0: the window.
  explicit MaxPool1d(std::size_t window = 0, std::size_t stride = 0) : window_(window), stride_(stride) {}
  std::string name() const override { return window_ ? "maxpool(" + std::to_string(window_) + ")" : "global maxpool"; }
  Shape output(Shape in) const override {
    const std::size_t m = window(in), s = stride(in);
    if (m > in.length) throw std::invalid_argument("tyche::learning::MaxPool1d: a window longer than its input");
    return {in.channels, (in.length - m) / s + 1};
  }
  void forward(Shape in, std::span<const T> x, std::span<T> y, std::span<const T>, const Pass&) const override {
    const std::size_t m = window(in), s = stride(in), Lo = output(in).length;
    for (std::size_t c = 0; c < in.channels; c++)
      for (std::size_t j = 0; j < Lo; j++) {
        const T* w = x.data() + c * in.length + j * s;
        T best = w[0];
        for (std::size_t k = 1; k < m; k++) best = std::max(best, w[k]);
        y[c * Lo + j] = best;
      }
  }
  void backward(Shape in, std::span<const T> x, std::span<const T>, std::span<const T> gy, std::span<T> gx, std::span<const T>, std::span<T>, const Pass&) const override {
    if (gx.empty()) return;
    std::fill(gx.begin(), gx.end(), T(0));
    const std::size_t m = window(in), s = stride(in), Lo = output(in).length;
    for (std::size_t c = 0; c < in.channels; c++)
      for (std::size_t j = 0; j < Lo; j++) {
        const std::size_t base = c * in.length + j * s;
        std::size_t arg = 0;
        for (std::size_t k = 1; k < m; k++)
          if (x[base + k] > x[base + arg]) arg = k;
        gx[base + arg] += gy[c * Lo + j];
      }
  }

 private:
  std::size_t window(Shape in) const { return window_ ? window_ : in.length; }
  std::size_t stride(Shape in) const { return stride_ ? stride_ : window(in); }
  std::size_t window_, stride_;
};

// ---- dropout --------------------------------------------------------------------------------------------
template <class T>
class Dropout final : public Layer<T> {
 public:
  explicit Dropout(double p) : p_(p) {
    if (!(p >= 0 && p < 1)) throw std::invalid_argument("tyche::learning::Dropout: a probability in [0, 1)");
  }
  std::string name() const override { return "dropout(" + std::to_string(p_).substr(0, 4) + ")"; }
  Shape output(Shape in) const override { return in; }
  void forward(Shape, std::span<const T> x, std::span<T> y, std::span<const T>, const Pass& pass) const override {
    if (!pass.training || p_ == 0) { std::copy(x.begin(), x.end(), y.begin()); return; }
    calx::random::Stream r = stream(pass);
    const T scale = static_cast<T>(1 / (1 - p_));
    for (std::size_t i = 0; i < x.size(); i++) y[i] = r.uniform() < p_ ? T(0) : x[i] * scale;
  }
  void backward(Shape, std::span<const T> x, std::span<const T>, std::span<const T> gy, std::span<T> gx, std::span<const T>, std::span<T>, const Pass& pass) const override {
    if (gx.empty()) return;
    if (!pass.training || p_ == 0) { std::copy(gy.begin(), gy.end(), gx.begin()); return; }
    calx::random::Stream r = stream(pass);
    const T scale = static_cast<T>(1 / (1 - p_));
    for (std::size_t i = 0; i < x.size(); i++) gx[i] = r.uniform() < p_ ? T(0) : gy[i] * scale;
  }

 private:
  static calx::random::Stream stream(const Pass& pass) {
    return calx::random::Stream(pass.seed ^ (static_cast<std::uint64_t>(pass.layer) << 48), DROPOUT, pass.epoch, static_cast<std::uint32_t>(pass.example));
  }
  double p_;
};

}  // namespace tyche::learning
