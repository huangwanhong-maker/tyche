// Checks of spatial regression (regression.hpp) against PySAL's spreg 1.9.1 (fixtures/make_regression_fixtures.py).
//
// - Columbus (CRIME on INC and HOVAL, queen contiguity row-standardised): the lag model's β, ρ, σ², log-likelihood and standard errors
//   from the inverse of the full information matrix; the error model's; the Lagrange multiplier tests, plain and robust, and SARMA.
// - The lag model's impacts against numpy's traces of (I − ρW)⁻¹; the total impact equals β/(1 − ρ) for row-standardised weights.
// - Sixty points with their 5 nearest neighbours (unsymmetric W, complex eigenvalues in the Jacobian): the lag model again.
// - Data drawn from a lag model with ρ = 0.6 on a lattice: the estimate within four standard errors.

#include <tyche/spatial/regression.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "fixtures/regression_fixtures.hpp"

namespace ts = tyche::spatial;

namespace {
ts::SpatialWeights weights(const std::vector<std::vector<std::pair<std::size_t, double>>>& rows) {
  std::vector<std::vector<ts::WeightEntry>> r(rows.size());
  for (std::size_t i = 0; i < rows.size(); i++)
    for (const auto& [j, w] : rows[i]) r[i].push_back({j, w});
  return ts::weights_from_rows(r, false);
}

calx::linalg::Rectangular design(const std::vector<double>& a, const std::vector<double>& b) {
  calx::linalg::Rectangular X(a.size(), 3);
  for (std::size_t i = 0; i < a.size(); i++) X(i, 0) = 1, X(i, 1) = a[i], X(i, 2) = b[i];
  return X;
}

double relative(const std::vector<double>& got, const std::vector<double>& want) {
  double worst = 0;
  for (std::size_t i = 0; i < want.size(); i++) worst = std::max(worst, std::fabs(got[i] - want[i]) / std::max(1.0, std::fabs(want[i])));
  return worst;
}
}  // namespace

int main() {
  calx::verification::Suite s;
  using namespace fixtures;
  const auto W = weights(columbus_w);
  const auto X = design(columbus_inc, columbus_hoval);
  {
    const auto f = ts::spatial_lag(X, columbus_crime, W);
    s.check("lag model, Columbus: β against spreg (relative)", relative(f.beta, columbus_lag_beta), 1e-6);
    s.check("... ρ, σ² and the log-likelihood", std::fabs(f.parameter - columbus_lag_rho) + std::fabs(f.sigma2 - columbus_lag_sigma2) / columbus_lag_sigma2 +
                                                     std::fabs(f.log_likelihood - columbus_lag_loglik) / std::fabs(columbus_lag_loglik), 1e-6);
    std::vector<double> se = f.se;
    se.push_back(f.parameter_se), se.push_back(std::sqrt(f.covariance(f.k + 1, f.k + 1)));
    s.check("... the standard errors of β, ρ and σ² from the full information matrix (relative)", relative(se, columbus_lag_se), 1e-5);
    const auto im = ts::lag_impacts(f, W);
    const std::vector<double> d{im.direct[1], im.direct[2]}, t{im.total[1], im.total[2]};
    s.check("the lag model's impacts, direct and total, against numpy's traces of (I − ρW)⁻¹ (relative)", std::max(relative(d, columbus_lag_direct), relative(t, columbus_lag_total)), 1e-5);
    s.check("... the total impact is β/(1 − ρ) for row-standardised weights", std::fabs(im.total[1] - f.beta[1] / (1 - f.parameter)), 1e-9);
  }
  {
    const auto f = ts::spatial_error(X, columbus_crime, W);
    s.check("error model, Columbus: β against spreg (relative)", relative(f.beta, columbus_error_beta), 1e-6);
    s.check("... λ, σ² and the log-likelihood", std::fabs(f.parameter - columbus_error_lambda) + std::fabs(f.sigma2 - columbus_error_sigma2) / columbus_error_sigma2 +
                                                     std::fabs(f.log_likelihood - columbus_error_loglik) / std::fabs(columbus_error_loglik), 1e-6);
    std::vector<double> se = f.se;
    se.push_back(f.parameter_se), se.push_back(std::sqrt(f.covariance(f.k + 1, f.k + 1)));
    s.check("... the standard errors of β, λ and σ² (relative)", relative(se, columbus_error_se), 1e-5);
  }
  {
    const auto t = ts::lagrange_multipliers(X, columbus_crime, W);
    s.check("Lagrange multiplier tests, Columbus: LM error and lag, their robust forms and SARMA against spreg",
            std::fabs(t.lm_error - columbus_lm_error) + std::fabs(t.lm_lag - columbus_lm_lag) + std::fabs(t.robust_error - columbus_rlm_error) +
                std::fabs(t.robust_lag - columbus_rlm_lag) + std::fabs(t.sarma - columbus_sarma), 1e-9);
    s.check("... and their p-values", std::fabs(t.p_error - columbus_lm_error_p) + std::fabs(t.p_lag - columbus_lm_lag_p), 1e-12);
  }
  {
    const auto K = weights(knn_w);
    const auto Xk = design(knn_x1, knn_x2);
    const auto f = ts::spatial_lag(Xk, knn_out, K);
    std::vector<double> se = f.se;
    se.push_back(f.parameter_se), se.push_back(std::sqrt(f.covariance(f.k + 1, f.k + 1)));
    s.check("lag model on 5 nearest neighbours (unsymmetric W, complex eigenvalues): β, ρ and the log-likelihood against spreg",
            relative(f.beta, knn_lag_beta) + std::fabs(f.parameter - knn_lag_rho) + std::fabs(f.log_likelihood - knn_lag_loglik) / std::fabs(knn_lag_loglik), 1e-6);
    s.check("... and the standard errors (relative)", relative(se, knn_lag_se), 1e-5);
  }
  {
    // A lattice of 15 × 15 with rook weights; y from a lag model with ρ = 0.6, β = (1, 2).
    const std::size_t m = 15, n = m * m;
    std::vector<std::vector<ts::WeightEntry>> rows(n);
    for (std::size_t i = 0; i < m; i++)
      for (std::size_t j = 0; j < m; j++) {
        const std::size_t p = i * m + j;
        if (i > 0) rows[p].push_back({p - m, 1});
        if (i + 1 < m) rows[p].push_back({p + m, 1});
        if (j > 0) rows[p].push_back({p - 1, 1});
        if (j + 1 < m) rows[p].push_back({p + 1, 1});
      }
    const auto L = ts::weights_from_rows(rows, true);
    calx::random::Stream st(20261109, 0x73706c67u, 0, 0);  // "splg"
    calx::linalg::Rectangular Xl(n, 2);
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; i++) Xl(i, 0) = 1, Xl(i, 1) = st.normal(), v[i] = 1 + 2 * Xl(i, 1) + st.normal();
    calx::linalg::Matrix A = calx::linalg::Matrix::identity(n), B(n);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t q = L.row_start[i]; q < L.row_start[i + 1]; q++) A(i, L.column[q]) -= 0.6 * L.weight[q];
    for (std::size_t i = 0; i < n; i++) B(i, 0) = v[i];
    const auto Y = calx::linalg::solve(A, B);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; i++) y[i] = Y(i, 0);
    const auto f = ts::spatial_lag(Xl, y, L);
    s.check("data drawn from a lag model with ρ = 0.6 on a 15 × 15 lattice: ρ̂ within four standard errors", std::fabs(f.parameter - 0.6) / f.parameter_se, 4);
    const auto t = ts::lagrange_multipliers(Xl, y, L);
    s.require("... and the LM test of a lag rejects its null on those data", t.p_lag < 0.001);
  }
  return s.done();
}
