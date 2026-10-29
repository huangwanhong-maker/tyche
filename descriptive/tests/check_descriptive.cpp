// Checks of tyche/descriptive.
//
// - NIST's Statistical Reference Datasets for univariate summary statistics (fixtures/README.md):
//   the certified mean and standard deviation of PiDigits, Lottery, Lew, Mavro, Michelso and
//   NumAcc1–4 (NumAcc2–4: 1001 values about 1.2, 10⁶ + 0.2 and 10⁷ + 0.2, exact means and an exact
//   standard deviation of 0.1), as log relative errors (McCullough 1998).
// - Hyndman and Fan's nine sample quantiles of 1, …, 10 at p = 0.25 from their definitions:
//   3, 3, 2, 2.5, 3, 2.75, 3.25, 35/12, 2.9375 (as R's quantile() gives them); quantiles at p = 0
//   and 1, the median of an even sample.
// - Moments in closed form: 1, …, 5 has g₂ = −1.3 and G₂ = −1.2; the sample (0, 0, 0, 1) has the
//   moments of a Bernoulli(¼): g₁ = (1 − 2p)/√(pq) = 2/√3 and g₂ = (1 − 6pq)/(pq).
// - Weighted statistics: integer frequency weights equal replication; reliability weights equal
//   to a constant give the unbiased variance; the weighted median.
// - Covariance and correlation matrices against the pairwise functions, a perfectly correlated
//   pair, and the unit diagonal.

#include <tyche/descriptive/summary.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace tyche::descriptive;
using calx::verification::relative;

namespace {

struct Univariate {
  std::vector<double> x;
  double mean = 0, sd = 0;
};

Univariate read_univariate(const std::string& name) {
  std::ifstream in(std::string(TYCHE_FIXTURES) + "/" + name);
  if (!in) throw std::runtime_error("cannot open fixture " + name);
  std::vector<std::string> lines;
  for (std::string l; std::getline(in, l);) {
    if (!l.empty() && l.back() == '\r') l.pop_back();
    lines.push_back(l);
  }
  Univariate u;
  int d0 = 0, d1 = 0;
  std::smatch m;
  for (const auto& l : lines) {
    if (std::regex_search(l, m, std::regex(R"(Data\s*:\s*lines\s+(\d+)\s+to\s+(\d+))"))) d0 = std::stoi(m[1]), d1 = std::stoi(m[2]);
    if (std::regex_search(l, m, std::regex(R"(ybar:\s+(\S+))"))) u.mean = std::stod(m[1]);
    if (std::regex_search(l, m, std::regex(R"(\s+s:\s+(\S+))"))) u.sd = std::stod(m[1]);
  }
  for (int i = d0; i <= d1; i++) {
    std::istringstream ss(lines[i - 1]);
    for (double v; ss >> v;) u.x.push_back(v);
  }
  return u;
}

double lre(double got, double want) {
  if (got == want) return 15;
  return std::min(15.0, -std::log10(std::fabs(got - want) / std::fabs(want)));
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- NIST univariate ----
  struct Case {
    const char* file;
    std::size_t n;
    double digits_mean, digits_sd;
  };
  // The digits demanded: 15 where the data are exact in binary, otherwise what the decimal
  // rounding of the input allows (NumAcc3 and NumAcc4: values 10⁶ + 0.1 and 10⁷ + 0.1 carry an
  // input error near 10⁻¹⁰ and 10⁻⁹ against deviations of 0.1).
  const Case cases[] = {{"nist-univ-pidigits.dat", 5000, 14, 14}, {"nist-univ-lottery.dat", 218, 14, 14}, {"nist-univ-lew.dat", 200, 14, 14},
                        {"nist-univ-mavro.dat", 50, 14, 12},     {"nist-univ-michelso.dat", 100, 14, 13}, {"nist-univ-numacc1.dat", 3, 15, 15},
                        {"nist-univ-numacc2.dat", 1001, 14, 13}, {"nist-univ-numacc3.dat", 1001, 14, 9},  {"nist-univ-numacc4.dat", 1001, 14, 8}};
  for (const auto& c : cases) {
    const Univariate u = read_univariate(c.file);
    const double lm = lre(mean(u.x), u.mean), ls = lre(standard_deviation(u.x), u.sd);
    char buf[96];
    std::snprintf(buf, sizeof buf, "n = %zu, mean %.1f digits, sd %.1f digits", u.x.size(), lm, ls);
    s.require(std::string("nist ") + c.file + ": the certified mean and standard deviation", u.x.size() == c.n && lm >= c.digits_mean && ls >= c.digits_sd, buf);
  }

  // ---- Hyndman–Fan quantiles ----
  {
    std::vector<double> x(10);
    for (int i = 0; i < 10; i++) x[i] = 10 - i;  // unsorted on purpose
    const double want[10] = {0, 3, 3, 2, 2.5, 3, 2.75, 3.25, 35.0 / 12, 2.9375};
    double worst = 0;
    for (int t = 1; t <= 9; t++) worst = std::max(worst, std::fabs(quantile(x, 0.25, t) - want[t]));
    s.check("quantile: the nine types of Hyndman and Fan at p = ¼ of 1, …, 10", worst, 1e-15);
    double ends = 0;
    for (int t = 1; t <= 9; t++) ends = std::max(ends, std::fabs(quantile(x, 0, t) - 1) + std::fabs(quantile(x, 1, t) - 10));
    s.check("quantile: every type gives the minimum at p = 0 and the maximum at p = 1", ends, 0);
    s.check("quantile: the median of 1, …, 10 is 5.5", median(x) - 5.5, 0);
    s.check("quantile: type 2 averages at a jump: the median of 1, …, 10 is 5.5 too", quantile(x, 0.5, 2) - 5.5, 0);
    s.check("quantile: type 1 at p = 0.5 is x₍₅₎ = 5", quantile(x, 0.5, 1) - 5, 0);
    s.check("quantile: the interquartile range of type 7, 7.75 − 3.25", interquartile_range(x) - 4.5, 1e-15);
  }

  // ---- moments ----
  {
    const std::vector<double> a{1, 2, 3, 4, 5};
    s.check("moments: the variance of 1, …, 5 is 2.5 (ddof 1) and 2 (ddof 0)", std::fabs(variance(a) - 2.5) + std::fabs(variance(a, 0) - 2), 1e-15);
    s.check("moments: its skewness is 0", skewness(a), 1e-15);
    s.check("moments: its excess kurtosis g₂ = 34/5 / 4 − 3 = −1.3", kurtosis(a) + 1.3, 1e-15);
    s.check("moments: and G₂ = ((n + 1)g₂ + 6)(n − 1)/((n − 2)(n − 3)) = −1.2", kurtosis(a, true, true) + 1.2, 1e-14);
    // (0, 0, 0, 1): the empirical distribution is Bernoulli(¼): g₁ = (1 − 2p)/√(pq), g₂ = (1 − 6pq)/(pq).
    const std::vector<double> b{0, 0, 0, 1};
    const double p = 0.25, q = 0.75;
    s.check("moments: (0, 0, 0, 1) has a Bernoulli(¼)'s skewness (1 − 2p)/√(pq) = 2/√3", skewness(b) - (1 - 2 * p) / std::sqrt(p * q), 1e-15);
    s.check("moments: and its excess kurtosis (1 − 6pq)/(pq)", kurtosis(b) - (1 - 6 * p * q) / (p * q), 1e-14);
    s.check("moments: G₁ = g₁ √(n(n − 1))/(n − 2)", skewness(b, true) - skewness(b) * std::sqrt(12.0) / 2, 1e-15);
  }

  // ---- weighted ----
  {
    const std::vector<double> x{2.0, 3.5, 1.0, 4.25, 3.0}, w{3, 1, 2, 4, 1};
    std::vector<double> r;
    for (std::size_t i = 0; i < x.size(); i++)
      for (int c = 0; c < static_cast<int>(w[i]); c++) r.push_back(x[i]);
    s.check("weighted: integer frequency weights, the mean of the replicated data", weighted_mean(x, w) - mean(r), 1e-15);
    s.check("weighted: and its variance", weighted_variance(x, w) - variance(r), 1e-14);
    const std::vector<double> c(5, 2.5);
    s.check("weighted: constant reliability weights give the unbiased variance", weighted_variance(x, c, Weights::reliability) - variance(x), 1e-14);
    s.check("weighted: the weighted median, the least x whose cumulative weight reaches half", weighted_quantile(x, w, 0.5) - 3.0, 0);
    const std::vector<double> e(5, 1.0);
    s.check("weighted: with equal weights the weighted quantile is type 1", weighted_quantile(x, e, 0.3) - quantile(x, 0.3, 1), 0);
  }

  // ---- covariance and correlation ----
  {
    const std::vector<double> u{1, 2, 4, 7, 11}, v{2, 1, 3, 3, 6}, t{3, 5, 9, 15, 23};  // t = 2u + 1
    const Rectangular D = Rectangular::from_columns({u, v, t});
    const Matrix C = covariance_matrix(D), R = correlation_matrix(D);
    s.check("covariance matrix: entries against the pairwise covariance", std::fabs(C(0, 1) - covariance(u, v)) + std::fabs(C(1, 2) - covariance(v, t)) + std::fabs(C(2, 2) - variance(t)), 1e-14);
    s.check("correlation matrix: a variable and 2u + 1 correlate perfectly", R(0, 2) - 1, 1e-15);
    s.check("correlation matrix: unit diagonal, symmetric, and r(u, v) as correlation()", std::fabs(R(1, 1) - 1) + std::fabs(R(0, 1) - R(1, 0)) + std::fabs(R(0, 1) - correlation(u, v)), 1e-15);
  }

  // ---- summary ----
  {
    std::vector<double> x;
    for (int i = 1; i <= 9; i++) x.push_back(i * i);
    const Summary m = summarise(x);
    s.require("summary: n, min, max", m.n == 9 && m.min == 1 && m.max == 81);
    s.check("summary: the median and quartiles of type 7 (9, 25, 49)", std::fabs(m.q1 - 9) + std::fabs(m.median - 25) + std::fabs(m.q3 - 49), 0);
    s.check("summary: the mean 285/9", m.mean - 285.0 / 9, 1e-14);
  }
  return s.done();
}
