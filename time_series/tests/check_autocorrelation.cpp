// Checks of tyche/time_series's autocorrelation functions and the Ljung–Box test.
//
// - NIST's univariate reference datasets certify the lag-1 autocorrelation r(1) of nine series
//   (descriptive/tests/fixtures): against acf(x, 1)[1], as log relative errors.
// - The Durbin–Levinson recursion on exact autocovariances: an AR(2)'s gives back φ₁, φ₂ and
//   φ_kk = 0 beyond lag 2, and its innovation variance; the partial autocorrelations of an MA(1)
//   x_t = ε_t + θε_{t−1} are −(−θ)^k (1 − θ²)/(1 − θ^{2(k+1)}) (Box, Jenkins and Reinsel 2008, §3.3,
//   whose θ has the opposite sign).
// - Δ log income of Lütkepohl's data (time_series/tests/fixtures): acf, pacf ("ywm") and the
//   Ljung–Box Q(10) against statsmodels 0.15.0 (computed 2026-10-24), a cross-check.
// - The size of the Ljung–Box test on Gaussian white noise over 400 seeded series.

#include <tyche/time_series/autocorrelation.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace tyche::time_series;
using calx::verification::relative;

namespace {

struct Univariate {
  std::vector<double> x;
  double r1 = 0;
};

Univariate read_univariate(const std::string& name) {
  std::ifstream in(std::string(TYCHE_NIST_UNIVARIATE) + "/" + name);
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
    if (std::regex_search(l, m, std::regex(R"(r\(1\):\s+(\S+))"))) u.r1 = std::stod(m[1]);
  }
  for (int i = d0; i <= d1; i++) {
    std::istringstream ss(lines[i - 1]);
    for (double v; ss >> v;) u.x.push_back(v);
  }
  return u;
}

std::vector<double> e1_column(int column) {
  std::ifstream in(std::string(TYCHE_FIXTURES) + "/lutkepohl-e1.dat");
  std::vector<double> out;
  for (std::string l; std::getline(in, l);) {
    std::istringstream ss(l);
    double a, b, c;
    if (ss >> a >> b >> c) out.push_back(column == 0 ? a : column == 1 ? b : c);
  }
  return out;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- NIST r(1) ----
  const std::pair<const char*, double> nist[] = {{"nist-univ-pidigits.dat", 13}, {"nist-univ-lottery.dat", 13}, {"nist-univ-lew.dat", 13},
                                                 {"nist-univ-mavro.dat", 11},    {"nist-univ-michelso.dat", 12}, {"nist-univ-numacc1.dat", 14},
                                                 {"nist-univ-numacc2.dat", 12},  {"nist-univ-numacc3.dat", 8},   {"nist-univ-numacc4.dat", 7}};
  for (const auto& [file, digits] : nist) {
    const Univariate u = read_univariate(file);
    const double r = acf(u.x, 1)[1];
    const double lre = r == u.r1 ? 15 : std::min(15.0, -std::log10(std::fabs(r - u.r1) / std::fabs(u.r1)));
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.1f digits", lre);
    s.require(std::string("nist ") + file + ": the certified lag-1 autocorrelation", lre >= digits, buf);
  }

  // ---- Durbin–Levinson on exact autocovariances ----
  {
    const double p1 = 0.5, p2 = 0.3, sigma2 = 2.0;
    std::vector<double> rho(8);
    rho[0] = 1;
    rho[1] = p1 / (1 - p2);
    for (int k = 2; k < 8; k++) rho[k] = p1 * rho[k - 1] + p2 * rho[k - 2];
    const double gamma0 = sigma2 / (1 - p1 * rho[1] - p2 * rho[2]);
    std::vector<double> g(8);
    for (int k = 0; k < 8; k++) g[k] = gamma0 * rho[k];
    const auto d = durbin_levinson(g, 6);
    double beyond = 0;
    for (int k = 2; k < 6; k++) beyond = std::max(beyond, std::fabs(d.partial[k]));
    s.check("durbin–levinson: an AR(2)'s exact autocovariances give φ₁ = 0.5", d.phi[0] - p1, 1e-15);
    s.check("durbin–levinson: and φ₂ = 0.3", d.phi[1] - p2, 1e-15);
    s.check("durbin–levinson: and partial autocorrelations zero beyond lag 2", beyond, 1e-15);
    s.check("durbin–levinson: and the innovation variance σ² = 2", d.variance.back() - sigma2, 1e-14);
  }
  {
    const double th = 0.6;
    std::vector<double> g(9, 0.0);
    g[0] = 1 + th * th;
    g[1] = th;
    const auto d = durbin_levinson(g, 8);
    double worst = 0;
    for (int k = 1; k <= 8; k++) {
      const double want = -std::pow(-th, k) * (1 - th * th) / (1 - std::pow(th, 2 * (k + 1)));
      worst = std::max(worst, std::fabs(d.partial[k - 1] - want));
    }
    s.check("durbin–levinson: an MA(1)'s partial autocorrelations −(−θ)^k(1 − θ²)/(1 − θ^{2k+2})", worst, 1e-15);
  }

  // ---- Lütkepohl's income, against statsmodels ----
  {
    const auto inc = e1_column(1);
    std::vector<double> dl;
    for (std::size_t t = 1; t < inc.size(); t++) dl.push_back(std::log(inc[t]) - std::log(inc[t - 1]));
    const auto r = acf(dl, 5), pr = pacf(dl, 5);
    const double want_r[6] = {1, 0.110494720629871, 0.135554574692433, 0.259673595489018, 0.022535526321509, 0.116642194772163};
    const double want_p[6] = {1, 0.110494720629871, 0.12487004012531, 0.239377882082326, -0.038237621890374, 0.064387399221819};
    double er = 0, ep = 0;
    for (int k = 0; k < 6; k++) {
      er = std::max(er, std::fabs(r[k] - want_r[k]));
      ep = std::max(ep, std::fabs(pr[k] - want_p[k]));
    }
    s.check("acf: Δ log income (Lütkepohl's data), lags 0–5, against statsmodels", er, 1e-14);
    s.check("pacf: the Yule–Walker partial autocorrelations, against statsmodels' \"ywm\"", ep, 1e-14);
    const auto lb = ljung_box(dl, 10);
    s.check("ljung–box: Q(10) = 14.4514742638 against statsmodels", relative(lb.statistic, 14.451474263837929), 1e-12);
    s.check("ljung–box: its p-value on 10 degrees of freedom", relative(lb.p_value, 0.153376307171838), 1e-11);
    double q = 0;
    const double n = static_cast<double>(dl.size());
    const auto r10 = acf(dl, 10);
    for (int k = 1; k <= 10; k++) q += r10[k] * r10[k] / (n - k);
    s.check("ljung–box: Q = n(n + 2) Σ r_k²/(n − k)", relative(lb.statistic, n * (n + 2) * q), 1e-14);
    const auto lb2 = ljung_box(dl, 10, 2);
    s.require("ljung–box: fitted parameters reduce the degrees of freedom", lb2.df == 8 && lb2.p_value < lb.p_value);
  }

  // ---- size on white noise ----
  {
    int rejected = 0;
    for (std::uint32_t rep = 0; rep < 400; rep++) {
      calx::random::Stream rng(20261024, 51, rep, 0);
      std::vector<double> x(200);
      for (double& v : x) v = rng.normal();
      rejected += ljung_box(x, 10).p_value < 0.05 ? 1 : 0;
    }
    // Binomial(400, 0.05): mean 20, sd 4.36.
    s.check("ljung–box: rejects white noise at 5% in 20 ± 4.4 of 400 series (deviation in sd)", (rejected - 20) / 4.36, 3);
  }
  return s.done();
}
