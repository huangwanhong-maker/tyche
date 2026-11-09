"""Reference values for tyche/spatial/regression.hpp, from PySAL's spreg 1.9.1 and libpysal 4.15.1 (BSD 3-Clause), and numpy.

    python -m venv env && env/bin/pip install spreg libpysal
    env/bin/python make_regression_fixtures.py

Writes regression_fixtures.hpp beside this file. Two data sets:
- Columbus (Anselin 1988, the 49 neighbourhoods bundled with libpysal): CRIME on INC and HOVAL, queen contiguity row-standardised
  (symmetric before standardising): the lag and error models (spreg ML_Lag and ML_Error, method "full"), the Lagrange multiplier
  tests (spreg LMtests on the least-squares fit), and the lag model's impacts computed by numpy from spreg's estimates
  (tr((I − ρW)⁻¹)/n and the mean row sum, times each β);
- sixty points of a seeded generator with their 5 nearest neighbours, row-standardised and unsymmetric (complex eigenvalues), y
  drawn from a lag model with ρ = 0.5: the lag model again.
The standard errors are the square roots of spreg's vm1, the inverse of the full information matrix of (β, ρ, σ²).
"""

from __future__ import annotations

from pathlib import Path

import libpysal
import numpy as np
import spreg

OUT = Path(__file__).with_name("regression_fixtures.hpp")


def lit(a) -> str:
    return "{" + ", ".join(f"{float(x):.17g}" for x in np.ravel(a)) + "}"


def rows(W) -> str:
    out = []
    for ident in W.id_order:
        out.append("{" + ", ".join(f"{{{W.id_order.index(j)}, {float(wt):.17g}}}" for j, wt in zip(W.neighbors[ident], W.weights[ident])) + "}")
    return "{" + ", ".join(out) + "}"


def lag_block(name, y, X, W):
    m = spreg.ML_Lag(y.reshape(-1, 1), X, w=W, method="full")
    se = np.sqrt(np.diag(m.vm1))
    n = len(y)
    Wd = W.full()[0]
    Ai = np.linalg.inv(np.eye(n) - float(np.ravel(m.rho)[0]) * Wd)
    b = np.ravel(m.betas)[1:-1]  # without the constant and ρ
    direct = b * np.trace(Ai) / n
    total = b * Ai.sum() / n
    return (f"  // {name}: spreg ML_Lag (full)\n"
            f"  inline const std::vector<double> {name}_beta = {lit(np.ravel(m.betas)[:-1])};\n"
            f"  inline const double {name}_rho = {float(np.ravel(m.rho)[0]):.17g};\n"
            f"  inline const double {name}_sigma2 = {float(np.ravel(m.sig2)[0]):.17g};\n"
            f"  inline const double {name}_loglik = {float(np.ravel(m.logll)[0]):.17g};\n"
            f"  inline const std::vector<double> {name}_se = {lit(se)};  // β…, ρ, σ²\n"
            f"  inline const std::vector<double> {name}_direct = {lit(direct)};\n"
            f"  inline const std::vector<double> {name}_total = {lit(total)};\n")


def main():
    path = libpysal.examples.get_path("columbus.dbf")
    db = libpysal.io.open(path, "r")
    y = np.array(db.by_col("CRIME"), dtype=float)
    X = np.column_stack([db.by_col("INC"), db.by_col("HOVAL")]).astype(float)
    W = libpysal.weights.Queen.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    W.transform = "r"
    err = spreg.ML_Error(y.reshape(-1, 1), X, w=W, method="full")
    ols = spreg.OLS(y.reshape(-1, 1), X, w=W, spat_diag=True)
    lm = spreg.diagnostics_sp.LMtests(ols, W)
    text = ["// Reference values for tyche/spatial/regression.hpp, written by make_regression_fixtures.py (spreg 1.9.1, libpysal 4.15.1).",
            "#pragma once", "#include <cstddef>", "#include <utility>", "#include <vector>", "", "namespace fixtures {",
            f"  inline const std::vector<double> columbus_crime = {lit(y)};",
            f"  inline const std::vector<double> columbus_inc = {lit(X[:, 0])};",
            f"  inline const std::vector<double> columbus_hoval = {lit(X[:, 1])};",
            f"  inline const std::vector<std::vector<std::pair<std::size_t, double>>> columbus_w = {rows(W)};",
            lag_block("columbus_lag", y, X, W),
            "  // columbus: spreg ML_Error (full)",
            f"  inline const std::vector<double> columbus_error_beta = {lit(np.ravel(err.betas)[:-1])};",
            f"  inline const double columbus_error_lambda = {float(np.ravel(err.lam)[0]):.17g};",
            f"  inline const double columbus_error_sigma2 = {float(np.ravel(err.sig2)[0]):.17g};",
            f"  inline const double columbus_error_loglik = {float(np.ravel(err.logll)[0]):.17g};",
            f"  inline const std::vector<double> columbus_error_se = {lit(np.concatenate([np.sqrt(np.diag(err.vm))[:-1], np.sqrt(np.diag(err.vm1))]))};  // β…, λ, σ²",
            "  // columbus: spreg LMtests on the least-squares fit (statistic, p)",
            f"  inline const double columbus_lm_error = {float(lm.lme[0]):.17g}, columbus_lm_error_p = {float(lm.lme[1]):.17g};",
            f"  inline const double columbus_lm_lag = {float(lm.lml[0]):.17g}, columbus_lm_lag_p = {float(lm.lml[1]):.17g};",
            f"  inline const double columbus_rlm_error = {float(lm.rlme[0]):.17g}, columbus_rlm_lag = {float(lm.rlml[0]):.17g};",
            f"  inline const double columbus_sarma = {float(lm.sarma[0]):.17g};"]
    rng = np.random.default_rng(20261109)
    n = 60
    pts = rng.uniform(0, 10, size=(n, 2))
    K = libpysal.weights.KNN.from_array(pts, k=5)
    K.transform = "r"
    Kd = K.full()[0]
    Xk = np.column_stack([rng.normal(size=n), rng.uniform(0, 3, size=n)])
    yk = np.linalg.solve(np.eye(n) - 0.5 * Kd, 1.0 + Xk @ np.array([2.0, -1.0]) + rng.normal(scale=0.5, size=n))
    text += [f"  inline const std::vector<double> knn_x = {lit(pts[:, 0])};",
             f"  inline const std::vector<double> knn_y = {lit(pts[:, 1])};",
             f"  inline const std::vector<double> knn_x1 = {lit(Xk[:, 0])};",
             f"  inline const std::vector<double> knn_x2 = {lit(Xk[:, 1])};",
             f"  inline const std::vector<double> knn_out = {lit(yk)};",
             f"  inline const std::vector<std::vector<std::pair<std::size_t, double>>> knn_w = {rows(K)};",
             lag_block("knn_lag", yk, Xk, K), "}  // namespace fixtures", ""]
    OUT.write_text("\n".join(text), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
