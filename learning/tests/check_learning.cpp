// Checks of the learning module: every layer's backward map against central differences of the
// loss (in double), Adam's first steps and AdamW's decay against their closed forms, the recovery of
// a planted linear model and of a planted convolutional filter by training, early stopping, and the
// sameness of training on one thread and on four.

#include <tyche/learning/train.hpp>

#include <calx/random/stream.hpp>
#include <calx/scheduler/thread_pool.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace tyche::learning;

namespace {

double relative_error(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); i++) {
    num = std::max(num, std::fabs(a[i] - b[i]));
    den = std::max(den, std::max(std::fabs(a[i]), std::fabs(b[i])));
  }
  return den > 0 ? num / den : num;
}

// The analytic gradient of the summed loss of the examples.
std::vector<double> analytic(Network<double>& net, const std::vector<std::vector<double>>& xs, const std::vector<std::vector<double>>& ys, const Loss& loss, const Pass& pass = {}) {
  std::vector<double> g(net.size(), 0);
  Workspace<double> w;
  for (std::size_t e = 0; e < xs.size(); e++) {
    Pass p = pass;
    p.example = e;
    net.accumulate(xs[e], ys[e], loss, g, w, p);
  }
  return g;
}

void examples(calx::random::Stream& r, Shape in, std::size_t outputs, std::size_t n, std::vector<std::vector<double>>& xs, std::vector<std::vector<double>>& ys) {
  xs.assign(n, std::vector<double>(in.size()));
  ys.assign(n, std::vector<double>(outputs));
  for (auto& x : xs)
    for (auto& v : x) v = r.normal();
  for (auto& y : ys)
    for (auto& v : y) v = r.normal();
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream r(17, 0x43484b53, 0, 0);
  std::vector<std::vector<double>> xs, ys;

  // ---- gradients against central differences --------------------------------------------------------
  {
    Network<double> net({3, 1});
    net.add<Dense<double>>(5).add<Tanh<double>>().add<Dense<double>>(2, Init::glorot);
    net.initialise(1);
    examples(r, net.input(), 2, 4, xs, ys);
    const Loss loss;
    s.check("dense → tanh → dense: backpropagation against central differences (squared error), relative", relative_error(analytic(net, xs, ys, loss), numerical_gradient(net, xs, ys, loss)), 1e-8);
    const Loss huber{LossKind::huber, 0.3};
    s.check("... with Huber's loss (δ = 0.3)", relative_error(analytic(net, xs, ys, huber), numerical_gradient(net, xs, ys, huber)), 1e-7);
    const Loss poisson{LossKind::poisson};
    for (auto& y : ys)
      for (auto& v : y) v = std::fabs(v) * 3;
    s.check("... with the Poisson deviance (log link)", relative_error(analytic(net, xs, ys, poisson), numerical_gradient(net, xs, ys, poisson)), 1e-8);
  }
  for (const auto padding : {Padding::valid, Padding::same}) {
    Network<double> net({4, 17});
    net.add<Conv1d<double>>(3, 5, padding).add<Relu<double>>().add<MaxPool1d<double>>(3, 2).add<Conv1d<double>>(2, 3, Padding::same).add<MaxPool1d<double>>().add<Dense<double>>(1);
    net.initialise(2);
    examples(r, net.input(), 1, 3, xs, ys);
    const Loss loss;
    s.check(std::string("conv1d (") + (padding == Padding::same ? "same" : "valid") + ") → relu → maxpool(3, stride 2) → conv1d → global maxpool → dense: " + std::to_string(net.size()) +
                " parameters against central differences, relative",
            relative_error(analytic(net, xs, ys, loss), numerical_gradient(net, xs, ys, loss)), 1e-7);
  }
  {
    // Dropout in training: the mask is drawn again in the backward pass, so the gradient is that of
    // the masked network, which central differences see too (same seed, epoch and example).
    Network<double> net({6, 1});
    net.add<Dense<double>>(8).add<Relu<double>>().add<Dropout<double>>(0.4).add<Dense<double>>(1);
    net.initialise(3);
    examples(r, net.input(), 1, 3, xs, ys);
    Pass pass;
    pass.training = true;
    pass.seed = 9;
    pass.epoch = 2;
    s.check("dense → relu → dropout(0.4) → dense in training against central differences", relative_error(analytic(net, xs, ys, Loss{}, pass), numerical_gradient(net, xs, ys, Loss{}, 1e-5, pass)), 1e-7);
    Workspace<double> w;
    const auto a = net.forward(xs[0], w);
    const double plain = a[0];
    Network<double> twin({6, 1});
    twin.add<Dense<double>>(8).add<Relu<double>>().add<Dense<double>>(1);
    twin.theta = net.theta;
    Workspace<double> w2;
    s.check("in prediction dropout is the identity", plain - twin.forward(xs[0], w2)[0], 1e-15);
  }

  // ---- the convolution by hand ----------------------------------------------------------------------
  {
    Network<double> net({1, 5});
    net.add<Conv1d<double>>(1, 3, Padding::same);
    net.theta = {1, 2, 3, 0.5};  // w = (1, 2, 3), b = 0.5
    Workspace<double> w;
    const std::vector<double> x = {1, 0, 2, 0, 1};
    const auto y = net.forward(x, w);
    // y[t] = b + w₀ x[t − 1] + w₁ x[t] + w₂ x[t + 1], x zero outside: (2.5, 7.5, 4.5, 9.5, 2.5).
    const std::vector<double> want = {0.5 + 2 * 1 + 3 * 0, 0.5 + 1 * 1 + 2 * 0 + 3 * 2, 0.5 + 1 * 0 + 2 * 2 + 3 * 0, 0.5 + 1 * 2 + 2 * 0 + 3 * 1, 0.5 + 1 * 0 + 2 * 1};
    double worst = 0;
    for (std::size_t i = 0; i < 5; i++) worst = std::max(worst, std::fabs(y[i] - want[i]));
    s.check("a same-padded convolution by hand (a cross-correlation, zero outside)", worst, 1e-15);
  }

  // ---- optimisers ---------------------------------------------------------------------------------------
  {
    std::vector<double> theta = {1.0, -2.0, 0.5}, g = {0.3, -4.0, 1e-3};
    Adam adam(3, {0.01});
    adam.step(std::span<double>(theta), std::span<const double>(g));
    // The first step moves each coordinate by α g/(|g| + ε√(1 − β₂)) ≈ α sign(g) (m̂₁ = g, v̂₁ = g²).
    s.check("Adam's first step is α sign(g) (Kingma and Ba, §3), up to ε", std::fabs(theta[0] - 0.99) + std::fabs(theta[1] + 1.99) + std::fabs(theta[2] - 0.49), 1e-6);
    std::vector<double> t2 = {2.0};
    Adam w(1, {0.1, 0.9, 0.999, 1e-8, 0.5});
    w.step(std::span<double>(t2), std::span<const double>(std::vector<double>{0.0}));
    s.check("AdamW's decoupled decay with a zero gradient: θ − α λ θ = 2 − 0.1 · 0.5 · 2", t2[0] - 1.9, 1e-12);
    // Adam on a quadratic ½ ‖θ − c‖² converges to c.
    std::vector<double> q = {5, -3}, c = {1, 2};
    Adam a(2, {0.05});
    for (int k = 0; k < 4000; k++) {
      std::vector<double> gq = {q[0] - c[0], q[1] - c[1]};
      a.step(std::span<double>(q), std::span<const double>(gq));
    }
    s.check("Adam minimises a quadratic: ‖θ − c‖ after 4000 steps", std::hypot(q[0] - c[0], q[1] - c[1]), 1e-3);
  }

  // ---- training recovers planted models --------------------------------------------------------------
  calx::scheduler::ThreadPool pool(4);
  {
    // y = 2 x₀ − x₁ + 0.5 x₂ + 1 + noise (sd 0.1), learnt by a single dense layer.
    Data<double> d;
    d.input = {3, 1};
    d.outputs = 1;
    for (int i = 0; i < 2000; i++) {
      const double a = r.normal(), b = r.normal(), c = r.normal();
      d.x.insert(d.x.end(), {a, b, c});
      d.y.push_back(2 * a - b + 0.5 * c + 1 + r.normal(0, 0.1));
    }
    Network<double> net(d.input);
    net.add<Dense<double>>(1, Init::glorot);
    net.initialise(4);
    TrainOptions o;
    o.epochs = 80;
    o.batch = 32;
    o.adam.rate = 0.005;
    const auto t = train(net, d, static_cast<const Data<double>*>(nullptr), o, &pool);
    const double err = std::fabs(net.theta[0] - 2) + std::fabs(net.theta[1] + 1) + std::fabs(net.theta[2] - 0.5) + std::fabs(net.theta[3] - 1);
    s.check("a planted linear model (2, −1, 0.5; 1) recovered by Adam, Σ|error| of the coefficients", err, 0.03);
    s.check("its training loss falls to the noise's variance, 0.01", t.curve.back().train_loss - 0.01, 0.003);
  }
  {
    // A planted motif: y = 3 · [the word TGACTC occurs] + noise, on random one-hot sequences of 40.
    const std::size_t L = 40;
    const std::vector<int> motif = {3, 2, 0, 1, 3, 1};
    Data<double> d;
    d.input = {4, L};
    d.outputs = 1;
    calx::random::Stream q(5, 0x4d4f5446, 0, 0);
    for (int i = 0; i < 3000; i++) {
      std::vector<int> seq(L);
      for (auto& b : seq) b = static_cast<int>(q.below(4));
      const bool planted = q.bernoulli(0.5);
      if (planted) {
        const std::size_t at = q.below(L - motif.size() + 1);
        for (std::size_t k = 0; k < motif.size(); k++) seq[at + k] = motif[k];
      }
      std::vector<double> x(4 * L, 0.0);
      for (std::size_t t = 0; t < L; t++) x[static_cast<std::size_t>(seq[t]) * L + t] = 1;
      d.x.insert(d.x.end(), x.begin(), x.end());
      d.y.push_back(3.0 * planted + q.normal(0, 0.2));
    }
    const auto parts = split(d.size(), 0.15, 0.15, 6);
    const auto tr = d.subset(parts.train), va = d.subset(parts.validation), te = d.subset(parts.test);
    const auto make = [&] {
      Network<double> net(d.input);
      net.add<Conv1d<double>>(16, 6).add<Relu<double>>().add<MaxPool1d<double>>().add<Dense<double>>(1, Init::glorot);
      net.initialise(7);
      return net;
    };
    auto net = make();
    TrainOptions o;
    o.epochs = 60;
    o.batch = 32;
    o.adam.rate = 0.02;
    o.patience = 15;
    const auto t = train(net, tr, &va, o, &pool);
    const double test = mean_loss(net, te, o.loss, &pool);
    s.require("a CNN learns a planted 6-mer: held-out mean squared error " + std::to_string(test).substr(0, 5) + ", under a tenth of the label's variance (2.25; the noise's is 0.04)", test < 0.225);
    // The filter that fires on the motif: its strongest weights spell ACGTA.
    const std::size_t F = 16, W = 6;
    std::size_t best = 0;
    double top = -1e300;
    for (std::size_t f = 0; f < F; f++) {
      double score = 0;
      for (std::size_t k = 0; k < W; k++) score += net.theta[(f * 4 + static_cast<std::size_t>(motif[k])) * W + k];
      if (score > top) { top = score; best = f; }
    }
    std::string word;
    for (std::size_t k = 0; k < W; k++) {
      int arg = 0;
      for (int b = 1; b < 4; b++)
        if (net.theta[(best * 4 + static_cast<std::size_t>(b)) * W + k] > net.theta[(best * 4 + static_cast<std::size_t>(arg)) * W + k]) arg = b;
      word += "ACGT"[arg];
    }
    int mismatches = 0;
    for (std::size_t k = 0; k < W; k++) mismatches += word[k] != "ACGT"[motif[k]];
    s.require("its best filter, read as a weight matrix (each column's largest weight), spells the planted TGACTC to within one base: " + word, mismatches <= 1);
    s.require("early stopping kept the best of " + std::to_string(t.curve.size()) + " epochs (epoch " + std::to_string(t.best_epoch) + ")", t.best_epoch >= 1 && t.best_epoch <= t.curve.size());
    // The same training on one thread.
    auto alone = make();
    train(alone, tr, &va, o, nullptr);
    s.require("training gives the same parameters, bit for bit, on one thread and on four", alone.theta == net.theta);
  }
  return s.done();
}
