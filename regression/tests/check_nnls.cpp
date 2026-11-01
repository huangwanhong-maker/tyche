// Checks of tyche/regression/nnls.hpp: non-negative least squares by Lawson and Hanson's algorithm.
//
// - The Karush–Kuhn–Tucker conditions (x ≥ 0, w = Aᵀ(b − Ax) ≤ 0, x_j w_j = 0) on random problems of
//   several shapes, overdetermined and underdetermined.
// - Brute force: on 300 small problems (n ≤ 7), the least residual over every support whose
//   unconstrained solution is positive equals NNLS's.
// - A problem whose unconstrained solution is positive: NNLS is least squares.
// - scipy 1.18.1 (cross-check, fixtures/nnls-reference.txt): scipy.optimize.nnls's solutions and
//   residual norms.
// - nnls_columns: one solution per column.

#include <tyche/regression/nnls.hpp>

#include <calx/linalg/qr.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace rg = tyche::regression;
using calx::linalg::Rectangular;

namespace {

double residual_norm(const Rectangular& A, const std::vector<double>& x, const std::vector<double>& b) {
  double s = 0;
  for (std::size_t i = 0; i < A.rows; i++) {
    double r = b[i];
    for (std::size_t j = 0; j < A.cols; j++) r -= A(i, j) * x[j];
    s += r * r;
  }
  return std::sqrt(s);
}

// The largest violation of the KKT conditions, relative to ‖A‖ ‖b‖.
double kkt(const Rectangular& A, const std::vector<double>& b, const rg::NnlsResult& r) {
  double scale = 0;
  for (const double v : A.a) scale = std::max(scale, std::fabs(v));
  double bn = 0;
  for (const double v : b) bn = std::max(bn, std::fabs(v));
  scale = std::max(scale * bn * static_cast<double>(A.rows), 1e-300);
  double worst = 0;
  for (std::size_t j = 0; j < A.cols; j++) {
    worst = std::max(worst, std::max(0.0, -r.x[j]));
    if (r.x[j] > 0) worst = std::max(worst, std::fabs(r.dual[j]) / scale);
    else worst = std::max(worst, std::max(0.0, r.dual[j]) / scale);
  }
  return worst;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- KKT on random problems ----
  {
    calx::random::Stream z(1, 0, 0, 0);
    double worst = 0;
    std::size_t unconverged = 0;
    for (int rep = 0; rep < 200; rep++) {
      const std::size_t m = 3 + z.below(60), n = 1 + z.below(25);
      Rectangular A(m, n);
      for (auto& v : A.a) v = rep % 2 ? std::fabs(z.normal()) : z.normal();
      std::vector<double> b(m);
      for (auto& v : b) v = z.normal();
      const auto r = rg::nnls(A, b);
      if (!r.converged) unconverged++;
      worst = std::max(worst, kkt(A, b, r));
    }
    s.check("KKT: x ≥ 0, w ≤ 0 off the support and w = 0 on it, on 200 random problems (m ≤ 62, n ≤ 25)", worst, 1e-12);
    s.require("every random problem converged", unconverged == 0, std::to_string(unconverged) + " did not");
  }

  // ---- brute force over supports ----
  {
    calx::random::Stream z(2, 0, 0, 0);
    double worst = 0;
    for (int rep = 0; rep < 300; rep++) {
      const std::size_t n = 1 + z.below(7), m = n + z.below(8);
      Rectangular A(m, n);
      for (auto& v : A.a) v = z.normal();
      std::vector<double> b(m);
      for (auto& v : b) v = z.normal();
      const auto r = rg::nnls(A, b);
      double best = residual_norm(A, std::vector<double>(n, 0.0), b);
      for (std::uint32_t mask = 1; mask < (1u << n); mask++) {
        std::vector<std::size_t> cols;
        for (std::size_t j = 0; j < n; j++)
          if (mask & (1u << j)) cols.push_back(j);
        Rectangular As(m, cols.size());
        for (std::size_t i = 0; i < m; i++)
          for (std::size_t k = 0; k < cols.size(); k++) As(i, k) = A(i, cols[k]);
        const auto xs = calx::linalg::least_squares(As, b);
        if (std::any_of(xs.begin(), xs.end(), [](double v) { return v < 0; })) continue;
        std::vector<double> x(n, 0.0);
        for (std::size_t k = 0; k < cols.size(); k++) x[cols[k]] = xs[k];
        best = std::min(best, residual_norm(A, x, b));
      }
      worst = std::max(worst, std::fabs(r.residual_norm - best) / (1 + best));
    }
    s.check("brute force: the least residual over all non-negative supports equals NNLS's (300 problems, n ≤ 7)", worst, 1e-12);
  }

  // ---- a positive unconstrained solution ----
  {
    calx::random::Stream z(3, 0, 0, 0);
    Rectangular A(40, 6);
    for (auto& v : A.a) v = z.normal();
    std::vector<double> x0{1, 2, 0.5, 3, 1.5, 0.25}, b(40);
    for (std::size_t i = 0; i < 40; i++) {
      b[i] = 0.01 * z.normal();
      for (std::size_t j = 0; j < 6; j++) b[i] += A(i, j) * x0[j];
    }
    const auto r = rg::nnls(A, b);
    const auto ls = calx::linalg::least_squares(A, b);
    double d = 0;
    for (std::size_t j = 0; j < 6; j++) d = std::max(d, std::fabs(r.x[j] - ls[j]));
    s.check("a positive unconstrained solution: NNLS is least squares", d, 1e-12);
  }

  // ---- scipy (cross-check) ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/nnls-reference.txt");
    s.require("scipy fixtures: the reference file opens", static_cast<bool>(in));
    std::string line;
    double worst_x = 0, worst_r = 0;
    std::size_t cases = 0;
    auto numbers = [](const std::string& l) {
      std::vector<double> v;
      std::istringstream is(l);
      for (double x; is >> x;) v.push_back(x);
      return v;
    };
    while (std::getline(in, line)) {
      if (line.rfind("case ", 0) != 0) continue;
      std::istringstream h(line.substr(5));
      std::size_t m, n;
      h >> m >> n;
      Rectangular A(m, n);
      std::getline(in, line);
      A.a = numbers(line);
      std::getline(in, line);
      const auto b = numbers(line);
      std::getline(in, line);
      const auto x = numbers(line);
      std::getline(in, line);
      const double rn = std::stod(line);
      const auto r = rg::nnls(A, b);
      // Where A has full column rank the solution is unique; otherwise only the residual is.
      if (m >= n)
        for (std::size_t j = 0; j < n; j++) worst_x = std::max(worst_x, std::fabs(r.x[j] - x[j]) / (1 + std::fabs(x[j])));
      worst_r = std::max(worst_r, std::fabs(r.residual_norm - rn) / (1 + rn));
      cases++;
    }
    s.check("scipy 1.18.1 nnls: the solutions (full column rank; relative)", worst_x, 1e-10);
    s.check("scipy 1.18.1 nnls: the residual norms of all " + std::to_string(cases) + " cases (relative)", worst_r, 1e-10);
  }

  // ---- many right-hand sides ----
  {
    calx::random::Stream z(4, 0, 0, 0);
    Rectangular A(20, 4), B(20, 3);
    for (auto& v : A.a) v = std::fabs(z.normal());
    for (auto& v : B.a) v = std::fabs(z.normal());
    const auto X = rg::nnls_columns(A, B);
    double d = 0;
    for (std::size_t k = 0; k < 3; k++) {
      const auto r = rg::nnls(A, B.column(k));
      for (std::size_t j = 0; j < 4; j++) d = std::max(d, std::fabs(X(j, k) - r.x[j]));
    }
    s.check("nnls_columns: each column the solution for B's column", d, 0);
  }
  return s.done();
}
