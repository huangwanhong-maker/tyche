// Checks of tyche/compositions/sample and zeros: the centre, the variation matrix, the total variance, and
// the multiplicative replacement of zeros.
//
// The centre is the closure of the parts' geometric means, invariant to the scale of each row and
// equivariant under perturbation and powering; the centred sample's centre is the neutral element. The
// variation matrix is symmetric with a zero diagonal, vanishes for proportional parts and equals the
// variances of the log-ratios computed directly. The total variance equals (1/(2D)) Σ t_ij, Σ var(clr_i),
// the mean squared Aitchison distance to the centre and (D/2) times the mean of the variation matrix; it is
// invariant under perturbation and scales by a² under powering. On seeded logistic-normal samples the
// centre and the total variance approach ilr⁻¹(μ) and tr Σ. Multiplicative replacement keeps the ratios of
// the non-zero parts exactly and closes to the total; essential zeros and impossible values are refused.
// Cross-checks against scikit-bio (pairwise_vlr, centralize, multi_replace).

#include <tyche/compositions/balances.hpp>
#include <tyche/compositions/sample.hpp>
#include <tyche/compositions/zeros.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace tyche::compositions;

namespace {

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

double gap(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return INFINITY;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

Rectangular random_sample(calx::random::Stream& rng, std::size_t n, std::size_t D) {
  Rectangular X(n, D);
  std::vector<double> loc(D);
  for (double& v : loc) v = rng.normal();
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < D; j++) X(i, j) = std::exp(loc[j] + 0.6 * rng.normal()) * (1 + 99 * rng.uniform());  // rows at random scales
  return X;
}

// Each row perturbed by y and powered by a.
Rectangular transform(const Rectangular& X, const std::vector<double>& y, double a) {
  Rectangular Y(X.rows, X.cols);
  for (std::size_t i = 0; i < X.rows; i++) {
    const auto row = X.row(i);
    const auto z = perturb(power(row, a), y);
    for (std::size_t j = 0; j < X.cols; j++) Y(i, j) = z[j];
  }
  return Y;
}

std::vector<double> parse_list(const std::string& s, char sep = ',') {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) v.push_back(std::stod(item));
  return v;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261113, 0x636f6d70, 2, 0);

  // ---- the centre ----
  {
    const auto X = random_sample(rng, 30, 5);
    std::vector<double> g(5, 0.0);
    for (std::size_t i = 0; i < X.rows; i++)
      for (std::size_t j = 0; j < 5; j++) g[j] += std::log(X(i, j)) / 30.0;
    for (double& v : g) v = std::exp(v);
    s.check("the centre is the closure of the parts' geometric means", gap(centre(X), closure(g)), 1e-15);
    auto S = X;
    for (std::size_t i = 0; i < S.rows; i++)
      for (std::size_t j = 0; j < S.cols; j++) S(i, j) *= 1 + i;  // rows rescaled
    s.check("the centre is invariant to the scale of each row", gap(centre(S), centre(X)), 1e-15);
    std::vector<double> y{0.1, 0.4, 0.2, 0.2, 0.1};
    const double a = -1.7;
    s.check("the centre is equivariant: cen(a ⊙ X ⊕ y) = a ⊙ cen(X) ⊕ y", gap(centre(transform(X, y, a)), perturb(power(centre(X), a), y)), 1e-15);
    s.check("the centre closed to 1440 is the centre closed to 1, times 1440", gap(centre(X, 1440), [&] {
              auto c = centre(X);
              for (double& v : c) v *= 1440;
              return c;
            }()) / 1440, 1e-15);
    s.check("the centred sample's centre is the neutral element", gap(centre(centred(X)), neutral(5)), 1e-15);
  }

  // ---- the variation matrix and the total variance ----
  {
    const auto X = random_sample(rng, 25, 6);
    const std::size_t D = 6, n = 25;
    for (double ddof : {1.0, 0.0}) {
      const auto T = variation_matrix(X, ddof);
      double sym = 0, diag = 0, direct = 0, sum = 0;
      for (std::size_t a = 0; a < D; a++) {
        diag = std::max(diag, std::fabs(T(a, a)));
        for (std::size_t b = 0; b < D; b++) {
          sym = std::max(sym, std::fabs(T(a, b) - T(b, a)));
          sum += T(a, b);
          std::vector<double> r(n);
          double m = 0;
          for (std::size_t i = 0; i < n; i++) m += r[i] = std::log(X(i, a) / X(i, b));
          m /= static_cast<double>(n);
          double v = 0;
          for (double x : r) v += (x - m) * (x - m);
          v /= static_cast<double>(n) - ddof;
          direct = std::max(direct, std::fabs(v - T(a, b)));
        }
      }
      const std::string tag = ddof == 1 ? " (ddof 1)" : " (ddof 0)";
      s.require("the variation matrix is symmetric with a zero diagonal" + tag, sym == 0 && diag == 0);
      s.check("t_ij equals var(ln(x_i/x_j)) computed directly" + tag, direct, 1e-14);
      const double tv = total_variance(X, ddof);
      const auto C = clr_covariance(X, ddof);
      double clrsum = 0;
      for (std::size_t j = 0; j < D; j++) clrsum += C(j, j);
      const auto g = centre(X);
      double d2 = 0;
      for (std::size_t i = 0; i < n; i++) {
        const auto row = X.row(i);
        d2 += distance(row, g) * distance(row, g);
      }
      d2 /= static_cast<double>(n) - ddof;
      s.check("total variance: (1/(2D)) Σ t_ij = Σ var(clr_i)" + tag, tv - clrsum, 1e-14);
      s.check("total variance: = (1/(n − ddof)) Σ_k d_a²(x_k, ĝ)" + tag, tv - d2, 1e-14);
      s.check("total variance: = (D/2) · mean of the D² entries of T" + tag, tv - 0.5 * static_cast<double>(D) * sum / static_cast<double>(D * D), 1e-14);
      s.check("metric standard deviation: √(totvar/(D − 1))" + tag, metric_standard_deviation(X, ddof) - std::sqrt(tv / (D - 1.0)), 1e-15);
      const auto sm = describe(X, ddof, 1440);
      s.check("describe(): the same total variance, metric standard deviation and centre" + tag,
              std::fabs(sm.total_variance - tv) + std::fabs(sm.metric_standard_deviation - metric_standard_deviation(X, ddof)) + gap(sm.centre, centre(X, 1440)) / 1440, 1e-14);
    }
    double rowsum = 0;
    const auto C = clr_covariance(X);
    for (std::size_t a = 0; a < D; a++) {
      double r = 0;
      for (std::size_t b = 0; b < D; b++) r += C(a, b);
      rowsum = std::max(rowsum, std::fabs(r));
    }
    s.check("the clr covariance's rows sum to zero", rowsum, 1e-14);
    std::vector<double> y{3, 1, 4, 1, 5, 9};
    s.check("the total variance is invariant under perturbation of every row", total_variance(transform(X, y, 1)) - total_variance(X), 1e-13);
    s.check("and scales by a² under powering", total_variance(transform(X, y, 2.5)) - 6.25 * total_variance(X), 1e-12);
    // Proportional parts: part 1 is 3 times part 0 in every row.
    auto P = X;
    for (std::size_t i = 0; i < n; i++) P(i, 1) = 3 * P(i, 0);
    s.check("t_01 = 0 when parts 0 and 1 are proportional throughout", variation_matrix(P)(0, 1), 1e-15);
    // A sample of one composition repeated has no variance.
    Rectangular R(4, 3);
    for (std::size_t i = 0; i < 4; i++) R(i, 0) = 2.0 * (i + 1), R(i, 1) = 5.0 * (i + 1), R(i, 2) = 1.0 * (i + 1);
    s.check("one composition at four scales: total variance 0", total_variance(R), 1e-15);
  }

  // ---- seeded logistic-normal samples ----
  {
    // ilr coordinates (pivot basis of 4 parts) ~ N(μ, LLᵀ).
    const std::vector<double> mu{0.8, -0.3, 0.5};
    const double L[3][3] = {{0.5, 0, 0}, {0.2, 0.4, 0}, {-0.1, 0.15, 0.3}};
    double trace = 0;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) trace += L[a][b] * L[a][b];
    const auto bas = pivot_basis(4);
    const std::size_t n = 20000;
    Rectangular X(n, 4);
    for (std::size_t i = 0; i < n; i++) {
      double e[3] = {rng.normal(), rng.normal(), rng.normal()};
      std::vector<double> z(3);
      for (int a = 0; a < 3; a++) {
        z[a] = mu[a];
        for (int b = 0; b < 3; b++) z[a] += L[a][b] * e[b];
      }
      const auto x = ilr_inverse(z, bas, 1440);
      for (std::size_t j = 0; j < 4; j++) X(i, j) = x[j];
    }
    s.check("logistic-normal sample of 20 000: the centre within 0.01 (Aitchison distance) of ilr⁻¹(μ)", distance(centre(X), ilr_inverse(mu, bas)), 0.01);
    s.check("logistic-normal sample: the total variance within 2% of tr Σ", total_variance(X) / trace - 1, 0.02);
  }

  // ---- multiplicative replacement of zeros ----
  {
    const std::vector<double> day{480, 0, 300, 120, 0, 540};  // minutes, two activities not recorded
    ZeroOptions o;
    o.delta = 5;  // five minutes each
    const auto r = replace_zeros(day, o);
    double total = 0, ratios = 0;
    for (double v : r) total += v;
    const std::size_t nz[4] = {0, 2, 3, 5};
    for (std::size_t a : nz)
      for (std::size_t b : nz) ratios = std::max(ratios, std::fabs(r[a] / r[b] - day[a] / day[b]) / (day[a] / day[b]));
    s.check("replacement keeps the total (1440 minutes)", total - 1440, 1e-12);
    s.check("replacement keeps the ratios of the non-zero parts exactly (relative)", ratios, 4e-16);
    s.require("the zero parts take δ", r[1] == 5 && r[4] == 5);
    s.check("the non-zero parts are scaled by 1 − Σδ/c", r[0] - 480 * (1 - 10.0 / 1440), 1e-12);
    ZeroOptions lim;
    lim.detection_limits = {15, 15, 15, 15, 30, 15};  // a diary by the quarter hour, one activity by the half hour
    const auto rl = replace_zeros(day, lim);
    s.check("with detection limits: δ_j = 0.65 × limit_j", std::fabs(rl[1] - 0.65 * 15) + std::fabs(rl[4] - 0.65 * 30), 1e-15);
    ZeroOptions closed;
    closed.delta = 0.01;
    closed.total = 1;
    const auto rc = replace_zeros(day, closed);
    double t1 = 0;
    for (double v : rc) t1 += v;
    s.check("closed to a stated total (1), δ in its units", std::fabs(t1 - 1) + std::fabs(rc[1] - 0.01), 1e-15);
    const std::vector<double> none{1, 2, 3};
    s.check("a composition without zeros comes back closed and otherwise unchanged", gap(replace_zeros(none, o), none), 0);
    // Ratios for many random compositions with zeros.
    double worst = 0;
    for (int t = 0; t < 50; t++) {
      std::vector<double> x(7);
      for (double& v : x) v = rng.uniform() < 0.3 ? 0.0 : std::exp(rng.normal());
      x[t % 7] = 1;  // never all zero
      ZeroOptions oo;
      oo.delta = 1e-3;
      const auto y = replace_zeros(x, oo);
      for (std::size_t a = 0; a < 7; a++)
        for (std::size_t b = 0; b < 7; b++)
          if (x[a] > 0 && x[b] > 0) worst = std::max(worst, std::fabs(y[a] / y[b] / (x[a] / x[b]) - 1));
    }
    s.check("the ratios of non-zero parts kept in 50 random compositions with zeros", worst, 1e-15);
    ZeroOptions ess = o;
    ess.essential = {false, true, false, false, false, false};
    s.require("a zero in a part marked essential is refused", throws([&] { replace_zeros(day, ess); }));
    ess.essential = {false, false, false, false, false, true};
    s.require("an essential flag on a non-zero part is no obstacle", !throws([&] { replace_zeros(day, ess); }));
    ZeroOptions big;
    big.delta = 720;
    s.require("imputed values reaching the total are refused", throws([&] { replace_zeros(day, big); }));
    s.require("a zero without a value to impute is refused", throws([&] { replace_zeros(day, ZeroOptions{}); }));
    Rectangular D2(2, 6);
    for (std::size_t j = 0; j < 6; j++) D2(0, j) = day[j], D2(1, j) = day[5 - j];
    const auto R2 = replace_zeros(D2, o);
    s.require("a sample is replaced row by row", R2(0, 1) == 5 && R2(1, 1) == 5 && std::fabs(R2(1, 0) - 540 * (1 - 10.0 / 1440)) < 1e-12);
  }

  // ---- cross-checks against scikit-bio ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/skbio_composition.txt");
    if (!in) {
      s.skip("scikit-bio composition fixture not found");
    } else {
      std::map<std::string, std::vector<double>> zeros;
      Rectangular sample;
      std::string line;
      double vlr = 0, cen = 0, rep = 0;
      int rows = 0;
      while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string kind, name, list;
        ls >> kind >> name >> list;
        if (kind == "zeros") zeros[name] = parse_list(list);
        else if (kind == "sample") {
          std::vector<std::vector<double>> r;
          std::stringstream rs(list);
          std::string row;
          while (std::getline(rs, row, ';')) r.push_back(parse_list(row));
          sample = Rectangular::from_rows(r);
        } else if (kind == "multi_replace") {
          const auto& x = zeros.at(name);
          ZeroOptions o;
          o.delta = 1.0 / static_cast<double>(x.size() * x.size());  // scikit-bio's default δ = 1/D²
          o.total = 1;
          rep = std::max(rep, gap(replace_zeros(x, o), parse_list(list)));
          rows++;
        } else if (kind == "pairwise_vlr") {
          const auto T = variation_matrix(sample, 1);
          const auto want = parse_list(list);
          for (std::size_t k = 0; k < want.size(); k++) vlr = std::max(vlr, std::fabs(T.a[k] - want[k]));
          rows++;
        } else if (kind == "centralize") {
          const auto C = centred(sample, 1);
          const auto want = parse_list(list);
          for (std::size_t k = 0; k < want.size(); k++) cen = std::max(cen, std::fabs(C.a[k] - want[k]));
          rows++;
        }
      }
      s.check("cross-check (scikit-bio 0.7.4): the variation matrix (pairwise_vlr, ddof 1) of a sample of seven days", vlr, 1e-15);
      s.check("cross-check (scikit-bio 0.7.4): the sample centred on its centre (centralize)", cen, 1e-15);
      s.check("cross-check (scikit-bio 0.7.4): multiplicative replacement with δ = 1/D² (multi_replace), three compositions", rep, 1e-15);
      s.require("cross-check: every fixture row read", rows == 5, std::to_string(rows) + " rows");
    }
  }

  return s.done();
}
