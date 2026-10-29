// Checks of tyche/regression's least squares against certified and published values.
//
// - NIST's Statistical Reference Datasets for linear least squares (fixtures/README.md): Norris,
//   Pontius, NoInt1 and NoInt2 (through the origin), Longley (collinear), Wampler1 and Wampler4
//   (degree 5), Filip (degree 10): coefficients, standard errors, residual standard deviation,
//   R² and F against the certified values, as log relative errors (LRE, McCullough 1998, "Assessing
//   the reliability of statistical software", Amer. Statist. 52, 358–366).
// - Petersen's (2009) test panel: OLS, White (HC1) and clustered (by firm, by year) standard
//   errors against the values he publishes from Stata, to their four decimals.
// - Closed forms: HC0 of a simple regression, Σ(x − x̄)²e² / (Σ(x − x̄)²)² (White 1980); HC0–HC3
//   of a mean (h = 1/n); clusters of one observation each reproduce HC1; a single restriction's
//   Wald F is t²; the Wald test of all slopes is the ANOVA F; WLS with integer weights is OLS on
//   the replicated data; a duplicated column is dropped with drop_collinear and refused without.
// - Recovery of known coefficients from seeded simulations, within four standard errors, and
//   the coverage of 95% intervals over 400 replications.

#include <tyche/regression/linear.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace tyche::regression;
using calx::verification::relative;

namespace {

// A NIST StRD linear least-squares file: its data and certified values.
struct Nist {
  std::vector<std::vector<double>> rows;  // y, x1, x2, …
  std::vector<double> b, sd;
  double residual_sd = 0, r_squared = 0, f = 0, ss_regression = 0, ss_residual = 0;
};

Nist read_nist(const std::string& name) {
  std::ifstream in(std::string(TYCHE_FIXTURES) + "/" + name);
  if (!in) throw std::runtime_error("cannot open fixture " + name);
  std::vector<std::string> lines;
  for (std::string l; std::getline(in, l);) {
    if (!l.empty() && l.back() == '\r') l.pop_back();
    lines.push_back(l);
  }
  Nist d;
  int c0 = 0, c1 = 0, d0 = 0, d1 = 0;
  std::smatch m;
  for (const auto& l : lines) {
    if (std::regex_search(l, m, std::regex(R"(Certified Values\s+\(lines (\d+) to (\d+)\))"))) c0 = std::stoi(m[1]), c1 = std::stoi(m[2]);
    if (std::regex_search(l, m, std::regex(R"(Data\s+\(lines (\d+) to (\d+)\))"))) d0 = std::stoi(m[1]), d1 = std::stoi(m[2]);
  }
  std::map<int, std::pair<double, double>> bs;
  for (int i = c0; i <= c1; i++) {
    const std::string& l = lines[i - 1];
    if (std::regex_search(l, m, std::regex(R"(^\s*B(\d+)\s+(\S+)\s+(\S+))"))) bs[std::stoi(m[1])] = {std::stod(m[2]), std::stod(m[3])};
    if (std::regex_search(l, m, std::regex(R"(Standard Deviation\s+(\S+)\s*$)"))) d.residual_sd = std::stod(m[1]);
    if (std::regex_search(l, m, std::regex(R"(R-Squared\s+(\S+))"))) d.r_squared = std::stod(m[1]);
    if (std::regex_search(l, m, std::regex(R"(^Regression\s+\d+\s+(\S+)\s+(\S+)\s+(\S+))"))) {
      d.ss_regression = std::stod(m[1]);
      d.f = m[3] == "Infinity" ? INFINITY : std::stod(m[3]);
    }
    if (std::regex_search(l, m, std::regex(R"(^Residual\s+\d+\s+(\S+))"))) d.ss_residual = std::stod(m[1]);
  }
  for (const auto& [j, v] : bs) {
    d.b.push_back(v.first);
    d.sd.push_back(v.second);
  }
  for (int i = d0; i <= d1; i++) {
    std::istringstream ss(lines[i - 1]);
    std::vector<double> row;
    for (double v; ss >> v;) row.push_back(v);
    if (!row.empty()) d.rows.push_back(row);
  }
  return d;
}

// The log relative error, capped at 15 (−log₁₀ of the relative error; digits agreeing).
double lre(double got, double want) {
  if (got == want) return 15;
  const double e = want == 0 ? std::fabs(got) : std::fabs(got - want) / std::fabs(want);
  return std::min(15.0, -std::log10(e));
}

struct Design {
  Rectangular X;
  std::vector<double> y;
};

// Polynomial in x of degree p (with intercept), or the given columns (with or without one).
Design polynomial(const Nist& d, int degree, bool intercept = true) {
  Design out{Rectangular(d.rows.size(), static_cast<std::size_t>(degree) + (intercept ? 1 : 0)), {}};
  for (std::size_t i = 0; i < d.rows.size(); i++) {
    out.y.push_back(d.rows[i][0]);
    std::size_t c = 0;
    if (intercept) out.X(i, c++) = 1;
    for (int e = 1; e <= degree; e++) out.X(i, c++) = std::pow(d.rows[i][1], e);
  }
  return out;
}
Design columns(const Nist& d) {
  const std::size_t k = d.rows[0].size();
  Design out{Rectangular(d.rows.size(), k), {}};
  for (std::size_t i = 0; i < d.rows.size(); i++) {
    out.y.push_back(d.rows[i][0]);
    out.X(i, 0) = 1;
    for (std::size_t j = 1; j < k; j++) out.X(i, j) = d.rows[i][j];
  }
  return out;
}

// Least LRE over the coefficients, the standard errors, and the summary statistics.
void certify(calx::verification::Suite& s, const std::string& name, const Nist& d, const LinearFit& f, double digits_b, double digits_se, double digits_stats) {
  double lb = 15, ls = 15;
  for (std::size_t j = 0; j < d.b.size(); j++) {
    lb = std::min(lb, lre(f.beta[j], d.b[j]));
    ls = std::min(ls, d.sd[j] == 0 ? (f.se[j] < 1e-9 ? 15.0 : 0.0) : lre(f.se[j], d.sd[j]));
  }
  double lstat = std::min(lre(f.r_squared, d.r_squared), d.residual_sd == 0 ? (f.sigma < 1e-9 ? 15.0 : 0.0) : lre(f.sigma, d.residual_sd));
  if (std::isfinite(d.f)) lstat = std::min(lstat, lre(f.f_statistic, d.f));
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.1f digits", lb);
  s.require("nist " + name + ": the coefficients to " + std::to_string(static_cast<int>(digits_b)) + " certified digits (LRE)", lb >= digits_b, buf);
  std::snprintf(buf, sizeof buf, "%.1f digits", ls);
  s.require("nist " + name + ": their standard errors to " + std::to_string(static_cast<int>(digits_se)) + " digits", ls >= digits_se, buf);
  std::snprintf(buf, sizeof buf, "%.1f digits", lstat);
  s.require("nist " + name + ": the residual standard deviation, R² and F to " + std::to_string(static_cast<int>(digits_stats)) + " digits", lstat >= digits_stats, buf);
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- NIST StRD ----
  {
    const Nist d = read_nist("nist-norris.dat");
    const Design z = polynomial(d, 1);
    certify(s, "Norris", d, ols(z.X, z.y), 12, 12, 12);
  }
  {
    const Nist d = read_nist("nist-pontius.dat");
    const Design z = polynomial(d, 2);
    certify(s, "Pontius", d, ols(z.X, z.y), 11, 11, 11);
  }
  for (const char* name : {"nist-noint1.dat", "nist-noint2.dat"}) {
    const Nist d = read_nist(name);
    const Design z = polynomial(d, 1, false);
    const LinearFit f = ols(z.X, z.y);
    s.require(std::string(name) + ": no constant column, so R² is uncentred", !f.has_intercept);
    certify(s, name, d, f, 12, 12, 12);
  }
  {
    const Nist d = read_nist("nist-longley.dat");
    const Design z = columns(d);
    certify(s, "Longley", d, ols(z.X, z.y), 11, 11, 11);
  }
  {
    const Nist d = read_nist("nist-wampler1.dat");
    const Design z = polynomial(d, 5);
    const LinearFit f = ols(z.X, z.y);
    double lb = 15;
    for (std::size_t j = 0; j < 6; j++) lb = std::min(lb, lre(f.beta[j], 1.0));
    s.require("nist Wampler1: an exact fit of degree 5, every coefficient 1 to 9 digits", lb >= 9, std::to_string(lb));
    s.check("nist Wampler1: its residual standard deviation, certified 0, relative to ȳ", f.sigma / 1e6, 1e-9);
  }
  {
    const Nist d = read_nist("nist-wampler4.dat");
    const Design z = polynomial(d, 5);
    certify(s, "Wampler4", d, ols(z.X, z.y), 7, 7, 7);
  }
  {
    const Nist d = read_nist("nist-filip.dat");
    const Design z = polynomial(d, 10);
    certify(s, "Filip", d, ols(z.X, z.y), 6, 6, 7);
  }

  // ---- Petersen's panel ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/petersen-test-data.txt");
    std::vector<double> x, y;
    std::vector<std::int64_t> firm, year;
    for (double a, b, c, e; in >> a >> b >> c >> e;) {
      firm.push_back(static_cast<std::int64_t>(a));
      year.push_back(static_cast<std::int64_t>(b));
      x.push_back(c);
      y.push_back(e);
    }
    s.require("petersen: 5000 firm-years read", x.size() == 5000);
    const Rectangular X = design_matrix({x});
    auto at4 = [](double v) { return std::round(v * 1e4) / 1e4; };
    const LinearFit f = ols(X, y);
    s.require("petersen: OLS coefficients 0.0297 and 1.0348, R² 0.2078", at4(f.beta[0]) == 0.0297 && at4(f.beta[1]) == 1.0348 && at4(f.r_squared) == 0.2078);
    s.require("petersen: OLS standard errors 0.0284 and 0.0286", at4(f.se[0]) == 0.0284 && at4(f.se[1]) == 0.0286);
    OlsOptions o;
    o.covariance = Covariance::hc1;
    const LinearFit h = ols(X, y, o);
    s.require("petersen: White (Stata's robust, HC1) standard errors 0.0284 and 0.0284", at4(h.se[0]) == 0.0284 && at4(h.se[1]) == 0.0284);
    o.covariance = Covariance::cluster;
    o.clusters = firm;
    const LinearFit cf = ols(X, y, o);
    s.require("petersen: clustered by firm, 0.0670 and 0.0506", at4(cf.se[0]) == 0.0670 && at4(cf.se[1]) == 0.0506 && cf.clusters == 500);
    o.clusters = year;
    const LinearFit cy = ols(X, y, o);
    s.require("petersen: clustered by year, 0.0234 and 0.0334, on G − 1 = 9 degrees of freedom", at4(cy.se[0]) == 0.0234 && at4(cy.se[1]) == 0.0334 && cy.df_inference == 9);
  }

  // ---- closed forms ----
  const std::vector<double> xs{1, 2, 3, 4, 5, 6, 7, 8}, ys{2.1, 3.9, 6.2, 7.8, 10.5, 11.6, 14.4, 15.7};
  const Rectangular Xs = design_matrix({xs});
  {
    OlsOptions o;
    o.covariance = Covariance::hc0;
    const LinearFit f = ols(Xs, ys, o);
    double xbar = 0;
    for (double v : xs) xbar += v / 8;
    double sxx = 0, num = 0;
    for (std::size_t i = 0; i < 8; i++) {
      const double d = xs[i] - xbar;
      sxx += d * d;
      num += d * d * f.residuals[i] * f.residuals[i];
    }
    s.check("hc0: the slope's variance of a simple regression, Σ(x − x̄)²e²/(Σ(x − x̄)²)² (White 1980)", relative(f.covariance(1, 1), num / (sxx * sxx)), 1e-13);
    OlsOptions oc;
    oc.covariance = Covariance::cluster;
    for (std::int64_t i = 0; i < 8; i++) oc.clusters.push_back(i);
    o.covariance = Covariance::hc1;
    const LinearFit c1 = ols(Xs, ys, oc), h1 = ols(Xs, ys, o);
    s.check("cluster: one observation per cluster reproduces HC1", relative(c1.covariance(1, 1), h1.covariance(1, 1)) + relative(c1.covariance(0, 1), h1.covariance(0, 1)), 1e-13);
  }
  {
    // A mean: X = 1, h = 1/n. HC0 = Σe²/n², HC1 = HC2 = classical = s²/n, HC3 = Σe²/(n − 1)².
    const std::vector<double> v{3.1, 4.7, 2.2, 5.9, 4.4, 3.8};
    const double n = 6;
    double mean = 0, ss = 0;
    for (double a : v) mean += a / n;
    for (double a : v) ss += (a - mean) * (a - mean);
    Rectangular X1(6, 1);
    for (std::size_t i = 0; i < 6; i++) X1(i, 0) = 1;
    OlsOptions o;
    double worst = 0;
    const std::pair<Covariance, double> cases[] = {{Covariance::classical, ss / (n - 1) / n}, {Covariance::hc0, ss / (n * n)}, {Covariance::hc1, ss / (n * (n - 1))},
                                                   {Covariance::hc2, ss / (n * (n - 1))}, {Covariance::hc3, ss / ((n - 1) * (n - 1))}};
    for (const auto& [cov, want] : cases) {
      o.covariance = cov;
      worst = std::max(worst, relative(ols(X1, v, o).covariance(0, 0), want));
    }
    s.check("hc0–hc3: the variances of a mean in closed form", worst, 1e-14);
  }
  {
    const LinearFit f = ols(Xs, ys);
    Rectangular R(1, 2);
    R(0, 1) = 1;
    const std::vector<double> r{2.0};
    const auto w = wald(f, R, r);
    s.check("wald: one restriction's F is t², t = (b − 2)/se", relative(w.f_statistic, std::pow((f.beta[1] - 2) / f.se[1], 2)), 1e-13);
    s.check("wald: its p-value is the t test's", w.f_p_value - 2 * tyche::probability::StudentT{6}.survival(std::fabs((f.beta[1] - 2) / f.se[1])), 1e-14);
    const std::vector<double> zero{0.0};
    s.check("wald: the test of all slopes is the ANOVA F", relative(wald(f, R, zero).f_statistic, f.f_statistic), 1e-12);
  }
  {
    // Integer weights = replicated observations: the same β and RSS.
    const std::vector<double> wts{1, 3, 2, 1, 4, 1, 2, 2};
    OlsOptions o;
    o.weights = wts;
    const LinearFit fw = ols(Xs, ys, o);
    std::vector<double> xr, yr;
    for (std::size_t i = 0; i < 8; i++)
      for (int c = 0; c < static_cast<int>(wts[i]); c++) {
        xr.push_back(xs[i]);
        yr.push_back(ys[i]);
      }
    const LinearFit fr = ols(design_matrix({xr}), yr);
    s.check("wls: integer weights give the β of the replicated data", std::fabs(fw.beta[0] - fr.beta[0]) + std::fabs(fw.beta[1] - fr.beta[1]), 1e-13);
    s.check("wls: and its residual sum of squares", relative(fw.rss, fr.rss), 1e-13);
    s.check("wls: and its R²", fw.r_squared - fr.r_squared, 1e-14);
  }
  {
    Rectangular X(8, 3);
    for (std::size_t i = 0; i < 8; i++) {
      X(i, 0) = 1;
      X(i, 1) = xs[i];
      X(i, 2) = 2 * xs[i];  // collinear with column 1
    }
    bool threw = false;
    try { ols(X, ys); } catch (const std::invalid_argument&) { threw = true; }
    s.require("collinear: a column twice another is refused by default", threw);
    OlsOptions o;
    o.drop_collinear = true;
    const LinearFit f = ols(X, ys, o), g = ols(Xs, ys);
    const int dropped = f.aliased[1] ? 1 : 2;
    const int kept = 3 - dropped;
    s.require("collinear: with drop_collinear one of them is aliased, β = NaN", std::isnan(f.beta[dropped]) && f.rank == 2);
    s.check("collinear: and the fit is the regression on the other", relative(f.beta[kept] * (kept == 2 ? 2 : 1), g.beta[1]) + std::fabs(f.rss - g.rss), 1e-12);
  }

  // ---- seeded simulations ----
  {
    calx::random::Stream rng(20261024, 31, 0, 0);
    const std::size_t n = 2000;
    std::vector<double> x1(n), x2(n), y(n);
    const double b0 = 1.5, b1 = -2.0, b2 = 0.75;
    for (std::size_t i = 0; i < n; i++) {
      x1[i] = rng.normal();
      x2[i] = rng.uniform(0, 3);
      y[i] = b0 + b1 * x1[i] + b2 * x2[i] + (0.5 + std::fabs(x1[i])) * rng.normal();  // heteroskedastic
    }
    OlsOptions o;
    o.covariance = Covariance::hc3;
    const LinearFit f = ols(design_matrix({x1, x2}), y, o);
    const double z = std::max({std::fabs(f.beta[0] - b0) / f.se[0], std::fabs(f.beta[1] - b1) / f.se[1], std::fabs(f.beta[2] - b2) / f.se[2]});
    s.check("simulation: heteroskedastic errors, β recovered within 4 HC3 standard errors (largest |β̂ − β|/se)", z, 4);
  }
  {
    // Coverage of the classical 95% interval over 400 replications with normal errors.
    int covered = 0;
    for (std::uint32_t rep = 0; rep < 400; rep++) {
      calx::random::Stream rng(20261024, 32, rep, 0);
      std::vector<double> x(30), y(30);
      for (std::size_t i = 0; i < 30; i++) {
        x[i] = rng.normal();
        y[i] = 0.3 + 1.2 * x[i] + rng.normal();
      }
      const LinearFit f = ols(design_matrix({x}), y);
      covered += (f.ci_lower[1] <= 1.2 && 1.2 <= f.ci_upper[1]) ? 1 : 0;
    }
    // Binomial(400, 0.95): mean 380, sd 4.36.
    s.check("simulation: the 95% interval covers the slope in 380 ± 4.4 of 400 replications (deviation in sd)", (covered - 380) / 4.36, 3);
  }
  {
    // predict(): at the data, the mean is the fitted value; se² = x V xᵀ.
    const LinearFit f = ols(Xs, ys);
    const auto p = predict(f, Xs);
    double worst = 0;
    for (std::size_t i = 0; i < 8; i++) worst = std::max(worst, std::fabs(p.mean[i] - f.fitted[i]));
    s.check("predict: the mean at the data is the fitted value", worst, 1e-13);
    s.check("predict: the variance at x is x V xᵀ = hᵢ σ̂² (classical)", relative(p.se[3] * p.se[3], f.leverage[3] * f.sigma * f.sigma), 1e-12);
  }
  return s.done();
}
