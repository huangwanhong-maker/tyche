// Training a network on a data set: minibatch stochastic optimisation with Adam, the learning
// curve, early stopping on a validation set, prediction, and the splitting of data.
//
// - An epoch visits the training examples in an order drawn afresh (Fisher and Yates's shuffle on a
//   calx random stream keyed by the seed and the epoch), in minibatches of b examples; each batch's
//   gradient is the mean of its examples' gradients, and one Adam step follows (optimise.hpp).
//   This is stochastic gradient descent in Robbins and Monro's sense (1951, Ann. Math. Statist. 22,
//   400–407) with Adam's step.
// - Determinism. A batch is cut into a fixed number of shards, each a contiguous run of its examples
//   whose gradients are summed in order into the shard's own vector; the shards are then added in
//   order. The shards may run on any number of threads (calx's parallel_for); since neither their
//   boundaries nor their order depend on the threads, the parameters after training are the same,
//   bit for bit, on one thread or eight.
// - Early stopping (Prechelt 1998, in Neural Networks: Tricks of the Trade, LNCS 1524, 55–69):
//   after each epoch the mean loss on the validation set; training stops when it has not improved
//   for `patience` epochs, and the parameters of the best epoch are restored.
// - Splits: the indices of a data set shuffled by a stream and cut into training, validation and
//   test parts of the given fractions.
//
// Complexity: an epoch costs one forward and one backward pass per training example, and one
// forward pass per validation example.

#pragma once

#include <tyche/learning/network.hpp>
#include <tyche/learning/optimise.hpp>

#include <calx/random/stream.hpp>
#include <calx/scheduler/parallel_for.hpp>
#include <calx/scheduler/thread_pool.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace tyche::learning {

inline constexpr std::uint32_t SHUFFLE = 0x53485546;  // "SHUF": the order of an epoch
inline constexpr std::uint32_t SPLIT = 0x53504c54;    // "SPLT": the splitting of a data set

// Examples as rows: x is n × input.size(), y is n × outputs.
template <class T>
struct Data {
  Shape input;
  std::size_t outputs = 1;
  std::vector<T> x, y;
  std::size_t size() const { return input.size() ? x.size() / input.size() : 0; }
  std::span<const T> example(std::size_t i) const { return std::span<const T>(x).subspan(i * input.size(), input.size()); }
  std::span<const T> target(std::size_t i) const { return std::span<const T>(y).subspan(i * outputs, outputs); }
  // The examples at the given indices, in that order.
  Data subset(const std::vector<std::size_t>& rows) const {
    Data d;
    d.input = input;
    d.outputs = outputs;
    d.x.reserve(rows.size() * input.size());
    d.y.reserve(rows.size() * outputs);
    for (std::size_t r : rows) {
      const auto e = example(r), t = target(r);
      d.x.insert(d.x.end(), e.begin(), e.end());
      d.y.insert(d.y.end(), t.begin(), t.end());
    }
    return d;
  }
};

struct Split {
  std::vector<std::size_t> train, validation, test;
};

inline Split split(std::size_t n, double validation, double test, std::uint64_t seed) {
  if (validation < 0 || test < 0 || validation + test >= 1) throw std::invalid_argument("tyche::learning::split: fractions in [0, 1) that leave a training part");
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  calx::random::Stream r(seed, SPLIT, 0, 0);
  for (std::size_t i = n; i > 1; i--) std::swap(order[i - 1], order[r.below(i)]);
  const auto nv = static_cast<std::size_t>(std::llround(validation * static_cast<double>(n)));
  const auto nt = static_cast<std::size_t>(std::llround(test * static_cast<double>(n)));
  Split s;
  s.test.assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(nt));
  s.validation.assign(order.begin() + static_cast<std::ptrdiff_t>(nt), order.begin() + static_cast<std::ptrdiff_t>(nt + nv));
  s.train.assign(order.begin() + static_cast<std::ptrdiff_t>(nt + nv), order.end());
  return s;
}

struct TrainOptions {
  std::size_t epochs = 20, batch = 64;
  AdamOptions adam;
  Loss loss;
  std::size_t patience = 0;   // 0: no early stopping
  bool restore_best = true;   // with a validation set: end with the best epoch's parameters
  std::size_t shards = 8;     // per batch; fixed, so the result does not depend on the threads
  std::uint64_t seed = 1;
};

struct EpochRecord {
  std::size_t epoch = 0;
  double train_loss = 0;       // the mean loss of the epoch's batches, as they were met
  double validation_loss = std::numeric_limits<double>::quiet_NaN();
  double seconds = 0;
};

struct Training {
  std::vector<EpochRecord> curve;
  std::size_t best_epoch = 0;   // 1-based; 0 without a validation set
  double best_validation = std::numeric_limits<double>::quiet_NaN();
  std::size_t steps = 0;
};

// The mean loss over a data set (prediction mode).
template <class T>
double mean_loss(const Network<T>& net, const Data<T>& d, const Loss& loss, calx::scheduler::ThreadPool* pool = nullptr) {
  const std::size_t n = d.size();
  if (n == 0) return std::numeric_limits<double>::quiet_NaN();
  constexpr std::size_t CHUNK = 256;
  std::vector<double> part((n + CHUNK - 1) / CHUNK, 0.0);
  const auto run = [&](std::size_t b, std::size_t e) {
    Workspace<T> w;
    net.prepare(w);
    for (std::size_t c = b; c < e; c++) {
      double s = 0;
      for (std::size_t i = c * CHUNK; i < std::min(n, (c + 1) * CHUNK); i++) s += static_cast<double>(net.loss_of(d.example(i), d.target(i), loss, w));
      part[c] = s;
    }
  };
  if (pool) calx::scheduler::parallel_for(*pool, part.size(), 1, [&](std::size_t b, std::size_t e, std::size_t) { run(b, e); });
  else run(0, part.size());
  double s = 0;
  for (double v : part) s += v;
  return s / static_cast<double>(n);
}

// The network's outputs for a data set, n × outputs.
template <class T>
std::vector<T> predict(const Network<T>& net, const Data<T>& d, calx::scheduler::ThreadPool* pool = nullptr) {
  const std::size_t n = d.size(), m = net.output().size();
  std::vector<T> out(n * m);
  const auto run = [&](std::size_t b, std::size_t e) {
    Workspace<T> w;
    net.prepare(w);
    for (std::size_t i = b; i < e; i++) {
      const auto y = net.forward(d.example(i), w);
      std::copy(y.begin(), y.end(), out.begin() + static_cast<std::ptrdiff_t>(i * m));
    }
  };
  if (pool) calx::scheduler::parallel_for(*pool, n, 64, [&](std::size_t b, std::size_t e, std::size_t) { run(b, e); });
  else run(0, n);
  return out;
}

template <class T>
Training train(Network<T>& net, const Data<T>& train, const Data<T>* validation, const TrainOptions& o, calx::scheduler::ThreadPool* pool = nullptr) {
  using clock = std::chrono::steady_clock;
  const std::size_t n = train.size(), P = net.size();
  if (n == 0 || o.batch == 0 || o.shards == 0) throw std::invalid_argument("tyche::learning::train: examples, a positive batch and shards");
  if (train.input != net.input() || train.outputs != net.output().size()) throw std::invalid_argument("tyche::learning::train: data of the network's shapes");
  Adam adam(P, o.adam);
  Training result;
  std::vector<T> best = net.theta;
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const std::size_t S = std::min(o.shards, o.batch);
  std::vector<std::vector<T>> grad(S, std::vector<T>(P));
  std::vector<double> shard_loss(S);
  std::vector<Workspace<T>> work(S);
  for (auto& w : work) net.prepare(w);
  std::vector<T> total(P);
  std::size_t since_best = 0;
  for (std::size_t epoch = 1; epoch <= o.epochs; epoch++) {
    const auto t0 = clock::now();
    calx::random::Stream r(o.seed, SHUFFLE, static_cast<std::uint32_t>(epoch), 0);
    for (std::size_t i = n; i > 1; i--) std::swap(order[i - 1], order[r.below(i)]);
    double epoch_loss = 0;
    for (std::size_t start = 0; start < n; start += o.batch) {
      const std::size_t stop = std::min(n, start + o.batch), m = stop - start;
      const auto shard = [&](std::size_t s) {
        std::fill(grad[s].begin(), grad[s].end(), T(0));
        double l = 0;
        const std::size_t b = start + s * m / S, e = start + (s + 1) * m / S;
        for (std::size_t j = b; j < e; j++) {
          Pass pass;
          pass.training = true;
          pass.seed = o.seed;
          pass.epoch = static_cast<std::uint32_t>(epoch);
          pass.example = order[j];
          l += static_cast<double>(net.accumulate(train.example(order[j]), train.target(order[j]), o.loss, grad[s], work[s], pass));
        }
        shard_loss[s] = l;
      };
      if (pool) calx::scheduler::parallel_for(*pool, S, 1, [&](std::size_t b, std::size_t e, std::size_t) { for (std::size_t s = b; s < e; s++) shard(s); });
      else
        for (std::size_t s = 0; s < S; s++) shard(s);
      std::fill(total.begin(), total.end(), T(0));
      for (std::size_t s = 0; s < S; s++) {
        for (std::size_t i = 0; i < P; i++) total[i] += grad[s][i];
        epoch_loss += shard_loss[s];
      }
      const T scale = T(1) / static_cast<T>(m);
      for (auto& v : total) v *= scale;
      adam.step(std::span<T>(net.theta), std::span<const T>(total));
      result.steps++;
    }
    EpochRecord rec;
    rec.epoch = epoch;
    rec.train_loss = epoch_loss / static_cast<double>(n);
    if (validation && validation->size()) {
      rec.validation_loss = mean_loss(net, *validation, o.loss, pool);
      if (std::isfinite(rec.validation_loss) && (std::isnan(result.best_validation) || rec.validation_loss < result.best_validation)) {
        result.best_validation = rec.validation_loss;
        result.best_epoch = epoch;
        best = net.theta;
        since_best = 0;
      } else {
        since_best++;
      }
    }
    rec.seconds = std::chrono::duration<double>(clock::now() - t0).count();
    result.curve.push_back(rec);
    if (o.patience && validation && since_best >= o.patience) break;
  }
  if (validation && validation->size() && o.restore_best) net.theta = best;
  return result;
}

}  // namespace tyche::learning
