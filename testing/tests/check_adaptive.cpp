// Checks of testing/online.hpp, evalues.hpp and holdout.hpp (decision 0027).
//
// - The spending sequences sum to 1: their normalisers against ζ(1.6) and the sum of Javanmard and Montanari's sequence
//   (fixtures/make_online_fixtures.py).
// - LOND, LORD++, SAFFRON, α-investing and e-LOND on short sequences worked by hand, each level from the rule's formula.
// - FDR by simulation: 200 hypotheses, 40 of them non-null, 2000 replications; each rule's false discovery proportion averages
//   below α within three Monte Carlo errors, independent p-values for the rules that assume independence, equicorrelated ones
//   (ρ = 0.5) for e-LOND, which assumes nothing; α-investing's marginal FDR likewise.
// - E-values: the calibrators against their closed forms and quadrature; e-BH by hand, and as BH on 1/e; combination.
// - Thresholdout: with no noise it answers the training value within the threshold and the holdout's beyond it, each answer
//   beyond costing a unit of budget; the Laplace draws' mean and variance; and Dwork et al.'s demonstration: features with no
//   signal chosen by an analyst who consults the holdout, whose classifier the plain holdout overrates and Thresholdout does not.

#include <tyche/testing/evalues.hpp>
#include <tyche/testing/holdout.hpp>
#include <tyche/testing/multiple.hpp>
#include <tyche/testing/online.hpp>

#include <tyche/probability/distributions.hpp>

#include <calx/random/stream.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tt = tyche::testing;

namespace {

std::map<std::string, std::vector<double>> read_fixture(const std::string& path) {
  std::map<std::string, std::vector<double>> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string name, token;
    ss >> name;
    std::vector<double> v;
    while (ss >> token) v.push_back(std::stod(token));
    out[name] = v;
  }
  return out;
}

double upper_p(double z) { return tyche::probability::Normal{}.survival(z); }

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto f = read_fixture(std::string(TYCHE_FIXTURES) + "/online.txt");
  if (f.empty()) {
    s.require("the fixture online.txt is read", false, TYCHE_FIXTURES);
    return s.done();
  }

  // ---- spending sequences ----
  const tt::Spending lord = tt::Spending::lord(), pw = tt::Spending::power(1.6);
  s.check("spending: j^−1.6 normalised by ζ(1.6) (mpmath)", 1 / pw.normaliser() - f.at("zeta_1.6")[0], 1e-9);
  s.check("spending: Javanmard and Montanari's sequence normalised by its sum (partial sums and the exact tail)",
          1 / lord.normaliser() - f.at("lord_sum")[0], 1e-8);
  s.require("spending: the constant 0.0790820, not the 0.0772 printed (which sums to 0.976)",
            std::fabs(lord.normaliser() - 0.0790820) < 1e-7 && std::fabs(0.07720838 * f.at("lord_sum")[0] - 0.9763) < 1e-3);
  {
    bool down = true;
    for (std::size_t j = 1; j < 2000; j++) down = down && lord(j + 1) <= lord(j) && pw(j + 1) <= pw(j);
    s.require("spending: both sequences are non-increasing", down);
    double sum = 0;
    for (std::size_t j = 1; j <= 100000; j++) sum += pw(j);
    s.require("spending: Σ_{j ≤ 10⁵} j^−1.6/ζ(1.6) is below 1 by about the tail", sum < 1 && 1 - sum < 0.002, std::to_string(sum));
  }

  // ---- by hand ----
  {
    const double a = 0.05;
    tt::Online lond({tt::OnlineRule::lond, a});
    const auto d = lond.run(std::vector<double>{1e-4, 0.5, 2e-3, 0.3});
    s.check("LOND: α₁ = α γ₁", d[0].level - a * lord(1), 1e-16);
    s.require("LOND: p₁ = 10⁻⁴ ≤ α γ₁ = 0.0027 is rejected", d[0].rejected);
    s.check("LOND: α₂ = α γ₂ (D₁ + 1) = 2 α γ₂", d[1].level - 2 * a * lord(2), 1e-16);
    s.check("LOND: α₃ = 2 α γ₃", d[2].level - 2 * a * lord(3), 1e-16);
    s.require("LOND: p₃ = 0.002 ≤ 2αγ₃ is rejected", d[2].rejected == (2e-3 <= 2 * a * lord(3)));
    s.check("LOND: α₄ = α γ₄ (D₃ + 1)", d[3].level - a * lord(4) * (1 + (d[2].rejected ? 2 : 1)), 1e-16);
  }
  {
    const double a = 0.05, w0 = 0.025;
    tt::Online lp({tt::OnlineRule::lord_plus, a, w0});
    const auto d = lp.run(std::vector<double>{1e-4, 0.4, 1e-4, 0.5, 0.2});
    s.check("LORD++: α₁ = γ₁ W₀", d[0].level - lord(1) * w0, 1e-16);
    s.require("LORD++: p₁ rejected (τ₁ = 1)", d[0].rejected);
    s.check("LORD++: α₂ = γ₂ W₀ + (α − W₀) γ₁", d[1].level - (lord(2) * w0 + (a - w0) * lord(1)), 1e-16);
    s.check("LORD++: α₃ = γ₃ W₀ + (α − W₀) γ₂", d[2].level - (lord(3) * w0 + (a - w0) * lord(2)), 1e-16);
    s.require("LORD++: p₃ = 10⁻⁴ ≤ α₃ ≈ 0.00055 is rejected (τ₂ = 3)", d[2].rejected);
    s.check("LORD++: α₄ = γ₄ W₀ + (α − W₀) γ₃ + α γ₁", d[3].level - (lord(4) * w0 + (a - w0) * lord(3) + a * lord(1)), 1e-16);
    s.check("LORD++: α₅ = γ₅ W₀ + (α − W₀) γ₄ + α γ₂", d[4].level - (lord(5) * w0 + (a - w0) * lord(4) + a * lord(2)), 1e-16);
  }
  {
    const double a = 0.05, w0 = 0.025, lam = 0.5;
    tt::Online sf({tt::OnlineRule::saffron, a, w0, lam});
    const auto d = sf.run(std::vector<double>{0.6, 0.2, 1e-3, 0.9});
    s.check("SAFFRON: α₁ = (1 − λ) W₀ γ₁", d[0].level - (1 - lam) * w0 * pw(1), 1e-16);
    s.require("SAFFRON: p₁ = 0.6 is no candidate", !d[0].candidate && !d[0].rejected);
    s.check("SAFFRON: α₂ = (1 − λ) W₀ γ₂ (no candidates before)", d[1].level - (1 - lam) * w0 * pw(2), 1e-16);
    s.require("SAFFRON: p₂ = 0.2 is a candidate, not rejected", d[1].candidate && !d[1].rejected);
    s.check("SAFFRON: α₃ = (1 − λ) W₀ γ_{3−1} (one candidate before)", d[2].level - (1 - lam) * w0 * pw(2), 1e-16);
    s.require("SAFFRON: p₃ = 10⁻³ ≤ 0.0018 is rejected (τ₁ = 3)", d[2].rejected);
    s.check("SAFFRON: α₄ = (1 − λ)[W₀ γ_{4−2} + (α − W₀) γ_{4−3−0}]", d[3].level - (1 - lam) * (w0 * pw(2) + (a - w0) * pw(1)), 1e-16);
  }
  {
    const double a = 0.05;
    tt::Online ai({tt::OnlineRule::alpha_investing, a});
    double W = a / 2;
    const std::vector<double> p{0.3, 1e-4, 0.2, 0.7};
    std::size_t last = 0;
    bool ok = true;
    for (std::size_t t = 1; t <= p.size(); t++) {
      const double level = std::min(W / (1 + static_cast<double>(t - last)), W / (1 + W));
      const auto d = ai.test(p[t - 1]);
      ok = ok && std::fabs(d.level - level) < 1e-16;
      if (d.rejected) W += a, last = t;
      else W -= level / (1 - level);
      ok = ok && std::fabs(d.wealth - W) < 1e-15;
    }
    s.require("α-investing: levels W/(1 + t − τ_last) and wealth −α_t/(1 − α_t) or +ω, by hand over four tests", ok);
  }
  {
    tt::Online el({tt::OnlineRule::e_lond, 0.05});
    const auto d1 = el.test(1 / (0.05 * lord(1)) + 1);
    s.require("e-LOND: an e-value beyond 1/(α γ₁) is rejected", d1.rejected);
    const auto d2 = el.test(1 / (0.05 * lord(2) * 2) - 1e-9);
    s.require("e-LOND: one just short of 1/(2 α γ₂) is not", !d2.rejected);
    s.check("e-LOND: its level is LOND's", d2.level - 0.05 * lord(2) * 2, 1e-16);
  }

  // ---- FDR by simulation ----
  {
    const std::size_t m = 200, reps = 2000;
    const double a = 0.1, mu = 3;
    calx::random::Stream rs(20261118, 1, 0, 0);
    std::map<std::string, double> fdp, power, fdp2;
    std::map<std::string, double> V, R;
    for (std::size_t r = 0; r < reps; r++) {
      std::vector<char> nonnull(m, 0);
      for (std::size_t i = 0; i < m; i++) nonnull[i] = rs.uniform() < 0.2;
      std::vector<double> pind(m), pdep(m);
      const double common = rs.normal();
      for (std::size_t i = 0; i < m; i++) {
        const double shift = nonnull[i] ? mu : 0;
        pind[i] = upper_p(rs.normal() + shift);
        pdep[i] = upper_p(std::sqrt(0.5) * common + std::sqrt(0.5) * rs.normal() + shift);
      }
      for (tt::OnlineRule rule : {tt::OnlineRule::lond, tt::OnlineRule::lord_plus, tt::OnlineRule::saffron, tt::OnlineRule::alpha_investing, tt::OnlineRule::e_lond}) {
        tt::Online o({rule, a});
        const bool dep = rule == tt::OnlineRule::e_lond;
        std::size_t v = 0, rr = 0, tp = 0, nn = 0;
        for (std::size_t i = 0; i < m; i++) {
          const double p = dep ? pdep[i] : pind[i];
          const auto d = o.test(dep ? tt::calibrate_mixture(p) : p);
          nn += nonnull[i];
          if (d.rejected) rr++, v += !nonnull[i], tp += nonnull[i];
        }
        const std::string k = tt::name(rule);
        const double q = rr ? static_cast<double>(v) / rr : 0.0;
        fdp[k] += q / reps, fdp2[k] += q * q / reps, power[k] += (nn ? static_cast<double>(tp) / nn : 0.0) / reps;
        V[k] += static_cast<double>(v) / reps, R[k] += static_cast<double>(rr) / reps;
      }
    }
    for (const char* k : {"lond", "lord++", "saffron", "e-lond"}) {
      const double se = std::sqrt((fdp2[k] - fdp[k] * fdp[k]) / reps);
      s.check(std::string("simulation: ") + k + "'s FDR at α = 0.1, in Monte Carlo errors above α (≤ 3)", std::max(0.0, (fdp[k] - a) / std::max(se, 1e-12)), 3.0,
              "FDR " + std::to_string(fdp[k]) + ", power " + std::to_string(power[k]));
    }
    const double mfdr = V["alpha-investing"] / (R["alpha-investing"] + 1 - a);
    s.require("simulation: α-investing's marginal FDR E[V]/(E[R] + 1 − α) ≤ α", mfdr <= a,
              "mFDR " + std::to_string(mfdr) + ", power " + std::to_string(power["alpha-investing"]));
    s.require("simulation: SAFFRON and LORD++ find more than LOND, which spends without regaining", power["saffron"] > power["lond"] && power["lord++"] > power["lond"],
              "LOND " + std::to_string(power["lond"]) + ", LORD++ " + std::to_string(power["lord++"]) + ", SAFFRON " + std::to_string(power["saffron"]));
  }

  // ---- e-values ----
  for (const char* p : {"0.3", "0.01", "0.999"}) {
    const auto& v = f.at(std::string("mixture_") + p);
    s.check(std::string("mixture calibrator at p = ") + p + " against mpmath's closed form", tt::calibrate_mixture(std::stod(p)) - v[0], 1e-13);
    s.check(std::string("mixture calibrator at p = ") + p + " is ∫ κ p^{κ−1} dκ (mpmath's quadrature)", tt::calibrate_mixture(std::stod(p)) - v[1], 1e-13);
  }
  s.check("mixture calibrator: the series near p = 1 meets the closed form", tt::calibrate_mixture(1 - 1e-5) - tt::calibrate_mixture(1 - 2e-4) - (-(1e-5 - 2e-4) * (-1.0 / 6)), 1e-7);
  s.check("calibrator κ = 1/2 at p = 0.01 is 5", tt::calibrate(0.01, 0.5) - 5, 1e-14);
  {
    const std::vector<double> e{40, 25, 3, 0.5, 12};
    const auto r = tt::e_bh(e, 0.1);
    s.require("e-BH by hand: thresholds 50/k reject the two largest (40 ≥ 50? no; 25 ≥ 25 yes)", r == std::vector<char>{1, 1, 0, 0, 0});
    std::vector<double> p;
    for (double v : e) p.push_back(tt::e_to_p(v));
    const auto adj = tt::benjamini_hochberg(p);
    bool same = true;
    for (std::size_t i = 0; i < e.size(); i++) same = same && ((adj[i] <= 0.1) == (r[i] != 0));
    s.require("e-BH on e is BH on 1/e", same);
    s.check("combination: the mean of e-values", tt::e_mean(e) - 16.1, 1e-13);
    s.check("combination: the product", tt::e_product(std::vector<double>{2, 3, 0.5}) - 3, 1e-15);
  }

  // ---- Thresholdout ----
  {
    calx::random::Stream rs(7, 7, 0, 0);
    tt::Thresholdout th({0.05, 0.0, 2}, rs);
    const auto a1 = th.ask(0.50, 0.53);
    s.require("Thresholdout without noise: within T it answers the training value", !a1.overfit && a1.value == 0.50);
    const auto a2 = th.ask(0.50, 0.60);
    s.require("Thresholdout without noise: beyond T it answers the holdout's, at a unit of budget", a2.overfit && a2.value == 0.60 && th.left() == 1);
    th.ask(0.2, 0.9);
    const auto a4 = th.ask(0.2, 0.9);
    s.require("Thresholdout: the budget spent, nothing is answered", a4.spent && th.left() == 0 && th.overfits() == 2 && th.queries() == 4);
    const auto saved = th.saved();
    tt::Thresholdout again({0.05, 0.0, 2}, rs, saved);
    s.require("Thresholdout: resumed from what it saved, still spent", again.ask(0.1, 0.1).spent);
  }
  {
    calx::random::Stream rs(11, 3, 0, 0);
    double m = 0, v = 0;
    const int n = 200000;
    for (int i = 0; i < n; i++) {
      const double x = tt::laplace(rs, 0.3);
      m += x / n, v += x * x / n;
    }
    s.check("Laplace draws: mean 0, in standard errors", m / std::sqrt(2 * 0.09 / n), 4.0);
    s.check("Laplace draws: variance 2b² (relative)", (v - 2 * 0.09) / (2 * 0.09), 0.02);
  }
  {
    // Dwork et al. (2015): labels independent of d features; an analyst keeps the features whose correlation with the label is
    // large and of one sign on the training part and as the holdout answers, and reports the holdout accuracy of the vote they
    // make. The truth is 1/2. Plain reuse of the holdout overrates it; Thresholdout does not.
    const std::size_t n = 1000, d = 1000, reps = 6;
    double plain = 0, thresh = 0, fresh = 0;
    for (std::size_t r = 0; r < reps; r++) {
      calx::random::Stream rs(99, static_cast<std::uint32_t>(r), 0, 0);
      auto draw = [&](std::vector<std::vector<double>>& X, std::vector<double>& y) {
        X.assign(n, std::vector<double>(d));
        y.assign(n, 0);
        for (std::size_t i = 0; i < n; i++) {
          y[i] = rs.uniform() < 0.5 ? -1 : 1;
          for (std::size_t j = 0; j < d; j++) X[i][j] = rs.normal();
        }
      };
      std::vector<std::vector<double>> Xt, Xh, Xf;
      std::vector<double> yt, yh, yf;
      draw(Xt, yt), draw(Xh, yh), draw(Xf, yf);
      auto corr = [&](const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::size_t j) {
        double c = 0;
        for (std::size_t i = 0; i < n; i++) c += X[i][j] * y[i];
        return c / n;
      };
      const double cut = 1 / std::sqrt(static_cast<double>(n));
      calx::random::Stream noise(5, static_cast<std::uint32_t>(r), 0, 0);
      tt::Thresholdout th({0.15, 0.01, 1000000}, noise);  // T about three standard deviations of a difference between the parts
      std::vector<double> sp(d, 0), st(d, 0);
      for (std::size_t j = 0; j < d; j++) {
        const double t = corr(Xt, yt, j), h = corr(Xh, yh, j);
        if (std::fabs(t) > cut && std::fabs(h) > cut && t * h > 0) sp[j] = t > 0 ? 1 : -1;
        const double ans = th.ask(t, h).value;
        if (std::fabs(t) > cut && std::fabs(ans) > cut && t * ans > 0) st[j] = t > 0 ? 1 : -1;
      }
      auto accuracy = [&](const std::vector<double>& w, const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        double ok = 0;
        for (std::size_t i = 0; i < n; i++) {
          double v = 0;
          for (std::size_t j = 0; j < d; j++) v += w[j] * X[i][j];
          ok += (v >= 0 ? 1 : -1) == y[i];
        }
        return ok / n;
      };
      plain += accuracy(sp, Xh, yh) / reps;
      thresh += th.ask(accuracy(st, Xt, yt), accuracy(st, Xh, yh)).value / reps;
      fresh += accuracy(st, Xf, yf) / reps;
    }
    s.require("Dwork et al.: the plain holdout overrates a classifier of noise (accuracy above 0.55)", plain > 0.55, std::to_string(plain));
    s.check("Dwork et al.: Thresholdout's report of its classifier, against 1/2", thresh - 0.5, 0.03, "fresh data " + std::to_string(fresh));
  }
  return s.done();
}
