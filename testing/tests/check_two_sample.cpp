// Checks of testing/two_sample.hpp: comparing two groups, or a group with a level, against SciPy and statsmodels
// (fixtures/make_two_sample_fixtures.py), and against what follows from the definitions.
//
// - The weighted mean with equal weights is the plain mean, its standard error s/√n; Kish's effective size is (Σw)²/Σw².
// - Welch's t, Satterthwaite's degrees of freedom, the p-value and the interval as SciPy's ttest_ind(equal_var=False), on two
//   pairs of samples; a mean against a level as ttest_1samp.
// - Newcombe's interval for a difference of shares and the pooled z test as statsmodels' confint_proportions_2indep("newcomb")
//   and proportions_ztest, a case with a zero count among them; a share against a level as proportions_ztest with the null
//   variance, and Wilson's interval as proportion_confint("wilson").
// - Scaling every weight leaves the mean, its error and the test unchanged; a group with one effective row has no error, and
//   the test then says nothing (NaN); refusals.

#include <tyche/testing/bootstrap.hpp>
#include <tyche/testing/two_sample.hpp>

#include <calx/random/stream.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tt = tyche::testing;
using calx::verification::relative;

namespace {

std::map<std::string, std::vector<double>> read_fixture(const std::string& path) {
  std::map<std::string, std::vector<double>> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto colon = line.find(" :");
    if (colon == std::string::npos) continue;
    std::istringstream values(line.substr(colon + 2));
    std::vector<double> v;
    std::string token;
    while (values >> token) v.push_back(std::stod(token));
    out[line.substr(0, colon)] = v;
  }
  return out;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  const auto f = read_fixture(std::string(TYCHE_FIXTURES) + "/two_sample.txt");
  if (f.empty()) {
    s.require("the fixture two_sample.txt is read", false, TYCHE_FIXTURES);
    return s.done();
  }

  // ---- weighted means ----
  {
    const std::vector<double> x{1, 2, 3, 4, 10};
    const auto m = tt::weighted_mean(x);
    double mean = 0, ss = 0;
    for (double v : x) mean += v / 5;
    for (double v : x) ss += (v - mean) * (v - mean);
    s.check("with equal weights the weighted mean is the mean", m.mean - mean, 1e-15);
    s.check("and its standard error s/√n", m.se - std::sqrt(ss / 4 / 5), 1e-15);
    const std::vector<double> w{1, 1, 2, 2, 4};
    const auto mw = tt::weighted_mean(x, w);
    s.check("Kish's effective size is (Σw)²/Σw²", mw.effective - 100.0 / 26, 1e-14);
    std::vector<double> w3(w);
    for (double& v : w3) v *= 3;
    const auto m3 = tt::weighted_mean(x, w3);
    s.check("scaling every weight leaves the mean and its error alone", std::abs(m3.mean - mw.mean) + std::abs(m3.se - mw.se), 1e-14);
    const std::vector<double> gaps{1, NAN, 3};
    s.require("a missing value is left out", tt::weighted_mean(gaps).rows == 2);
    const auto one = tt::weighted_mean(std::vector<double>{5});
    s.require("one row has a mean and no error, and a test on it says nothing", one.mean == 5 && std::isnan(one.se) && std::isnan(tt::welch(one, m).p_value));
  }

  // ---- Welch, and a mean against a level, against SciPy ----
  for (int k = 1; k <= 2; k++) {
    const auto a = f.at("a" + std::to_string(k)), b = f.at("b" + std::to_string(k)), ref = f.at("welch" + std::to_string(k));
    const auto c = tt::welch(tt::weighted_mean(a), tt::weighted_mean(b));
    const double worst = std::max({relative(c.statistic, ref[0]), relative(c.df, ref[1]), relative(c.p_value, ref[2]), relative(c.low, ref[3]), relative(c.high, ref[4]), relative(c.difference, ref[5])});
    s.check("Welch's t, Satterthwaite's ν, the p-value and the interval as SciPy's, pair " + std::to_string(k), worst, 1e-9);
    const auto one = f.at("one" + std::to_string(k));
    const double level = k == 1 ? 9.5 : 0.3;
    const auto t = tt::against_level(tt::weighted_mean(a), level);
    const double w1 = std::max({relative(t.statistic, one[0]), relative(t.df, one[1]), relative(t.p_value, one[2]), relative(t.low + level, one[3]), relative(t.high + level, one[4])});
    s.check("a mean against a level as SciPy's ttest_1samp, sample " + std::to_string(k), w1, 1e-9);
  }

  // ---- shares, against statsmodels ----
  for (int k = 1; k <= 3; k++) {
    const auto n = f.at("shares" + std::to_string(k)), ref = f.at("newcombe" + std::to_string(k));
    const auto c = tt::share_difference(n[0], n[1], n[2], n[3]);
    s.check("Newcombe's interval and the pooled z test as statsmodels', case " + std::to_string(k) + (k == 3 ? " (a zero count)" : ""),
            std::max({relative(c.low, ref[0]), relative(c.high, ref[1]), relative(c.statistic, ref[2]), relative(c.p_value, ref[3])}), 1e-9);
  }
  for (int k = 1; k <= 2; k++) {
    const auto r = f.at("level" + std::to_string(k));
    const auto c = tt::share_against_level(r[0], r[1], r[2]);
    s.check("a share against a level: the score test and Wilson's interval as statsmodels', case " + std::to_string(k),
            std::max({relative(c.statistic, r[3]), relative(c.p_value, r[4]), relative(c.low + r[2], r[5]), relative(c.high + r[2], r[6])}), 1e-9);
  }

  // ---- the bootstrap ----
  {
    const auto a = f.at("a1");
    auto mean_of = [&](const std::vector<double>& mult) {
      double sw = 0, sx = 0;
      for (std::size_t i = 0; i < a.size(); i++) sw += mult[i], sx += mult[i] * a[i];
      return sw > 0 ? sx / sw : NAN;
    };
    calx::random::Stream st(27, 1, 0, 0);
    const auto b = tt::bootstrap(a.size(), mean_of, 20000, st);
    const auto m = tt::weighted_mean(a);
    const double n = static_cast<double>(a.size());
    // The bootstrap standard error of a mean tends to s √((n − 1)/n) / √n, the plug-in estimate.
    s.check("the bootstrap error of a mean is the plug-in s√((n−1)/n)/√n, within its Monte Carlo error", b.se / (m.se * std::sqrt((n - 1) / n)) - 1, 0.03);
    s.require("its percentile interval holds the mean", b.low < m.mean && m.mean < b.high);
    calx::random::Stream st2(27, 1, 0, 0);
    const auto again = tt::bootstrap(a.size(), mean_of, 20000, st2);
    s.require("a bootstrap is reproduced from its key", again.values == b.values);
    // Clusters of two identical units: resampling clusters, each cluster's two units move together.
    std::vector<double> twice;
    std::vector<long> label;
    for (std::size_t i = 0; i < 10; i++) twice.push_back(a[i]), twice.push_back(a[i]), label.push_back(static_cast<long>(i)), label.push_back(static_cast<long>(i));
    bool together = true;
    calx::random::Stream st3(5, 1, 0, 0);
    (void)tt::bootstrap(twice.size(), [&](const std::vector<double>& mult) {
      for (std::size_t i = 0; i + 1 < mult.size(); i += 2) together = together && mult[i] == mult[i + 1];
      return 0.0;
    }, 200, st3, 0.95, label);
    s.require("resampling clusters moves a cluster's units together", together);
  }

  // ---- refusals ----
  {
    bool refused = false;
    try { (void)tt::share_difference(5, 4, 1, 10); } catch (const std::invalid_argument&) { refused = true; }
    s.require("more successes than trials are refused", refused);
    refused = false;
    try { (void)tt::share_against_level(1, 10, 1.0); } catch (const std::invalid_argument&) { refused = true; }
    s.require("a level of one is refused", refused);
  }
  return s.done();
}
