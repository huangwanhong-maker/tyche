"""Reference fits for check_penalized.cpp and check_nnls.cpp, from scikit-learn and scipy.

Writes tests/fixtures/penalized-reference.txt and tests/fixtures/nnls-reference.txt. The data are drawn
by numpy's PCG64 (seed 20261101).

Penalised regression (scikit-learn): ElasticNet and Lasso minimise
    (1/2n) ||y - Xw - c||^2 + a r ||w||_1 + (a/2)(1 - r) ||w||^2,
which is tyche's gaussian elastic net with lambda = a, alpha = r and standardize = false (sample
weights, which scikit-learn rescales to sum to n, give tyche's normalised weights). For the
standardised cases the columns are centred and divided by their population standard deviations
(ddof = 0) here, the fit made on them, and the coefficients converted back to X's scale, which is
what tyche's standardize = true does inside. LogisticRegression (solver saga, elastic-net penalty
l1_ratio = r, inverse strength C) minimises
    C sum_i log(1 + exp(-s_i (x_i w + c))) + r ||w||_1 + (1 - r)/2 ||w||^2,
which is tyche's binomial elastic net with lambda = 1/(C n), alpha = r.

Non-negative least squares (scipy): scipy.optimize.nnls.

Run with a Python that has numpy, scipy and scikit-learn:  python make_penalized_fixtures.py
"""

import os
import sys
import warnings

import numpy as np
import scipy
import sklearn
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")


def fmt(v):
    return " ".join("%.17g" % x for x in np.ravel(v))


def correlated(rng, n, p, rho=0.5):
    z = rng.standard_normal((n, p))
    x = np.empty((n, p))
    x[:, 0] = z[:, 0]
    for j in range(1, p):
        x[:, j] = rho * x[:, j - 1] + np.sqrt(1 - rho * rho) * z[:, j]
    scales = np.exp(rng.normal(0, 1, size=p))
    return x * scales + rng.normal(0, 2, size=p)


def standardised(x, w=None):
    if w is None:
        w = np.ones(x.shape[0])
    v = w / w.sum()
    mu = v @ x
    sd = np.sqrt(v @ (x - mu) ** 2)
    return (x - mu) / sd, mu, sd


def main():
    rng = np.random.Generator(np.random.PCG64(20261101))
    cases = []

    # Gaussian, unstandardised and standardised, with and without weights.
    datasets = []
    for n, p in [(120, 8), (40, 60)]:
        X = correlated(rng, n, p)
        beta = np.zeros(p)
        beta[: min(p, 5)] = rng.normal(0, 2, size=min(p, 5)) / np.std(X[:, : min(p, 5)], axis=0)
        y = 1.5 + X @ beta + rng.standard_normal(n)
        w = np.exp(rng.normal(0, 0.5, size=n))
        d = len(datasets)
        datasets.append((X, y, w))
        for alpha in [1.0, 0.5, 0.1]:
            for lam in [0.5, 0.05]:
                for std in [False, True]:
                    for weighted in [False, True]:
                        sw = w if weighted else None
                        if std:
                            Z, mu, sd = standardised(X, sw)
                        else:
                            Z, mu, sd = X, np.zeros(p), np.ones(p)
                        if alpha == 1.0:
                            m = Lasso(alpha=lam, fit_intercept=True, tol=1e-15, max_iter=1000000, selection="cyclic")
                        else:
                            m = ElasticNet(alpha=lam, l1_ratio=alpha, fit_intercept=True, tol=1e-15, max_iter=1000000, selection="cyclic")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            m.fit(Z, y, sample_weight=sw)
                        b = m.coef_ / sd
                        c = m.intercept_ - b @ mu
                        cases.append((d, "gaussian", alpha, lam, std, weighted, c, b))

    # Binomial (logistic), standardised data, so that saga converges well.
    for n, p in [(200, 6)]:
        X = correlated(rng, n, p, rho=0.3)
        Z, mu, sd = standardised(X)
        beta = np.array([1.5, -1.0, 0.0, 0.7, 0.0, 0.0])
        eta = -0.3 + Z @ beta
        yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
        d = len(datasets)
        datasets.append((Z, yb, np.ones(n)))
        for alpha in [1.0, 0.5]:
            for lam in [0.05, 0.01]:
                C = 1.0 / (lam * n)
                m = LogisticRegression(l1_ratio=alpha, C=C, solver="saga", tol=1e-13, max_iter=10000000, fit_intercept=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(Z, yb)
                cases.append((d, "binomial", alpha, lam, False, False, m.intercept_[0], m.coef_[0]))

    with open(os.path.join(FIX, "penalized-reference.txt"), "w", encoding="utf-8") as f:
        f.write("# Reference fits from scikit-learn %s (numpy %s, scipy %s, Python %s); data by numpy's PCG64, seed 20261101.\n"
                % (sklearn.__version__, np.__version__, scipy.__version__, sys.version.split()[0]))
        f.write("# Made by ../make_penalized_fixtures.py on 2026-11-01. Gaussian: Lasso/ElasticNet (tol 1e-15); binomial: LogisticRegression, saga (tol 1e-13).\n")
        f.write("# data <id> <n> <p>; X by rows; y; weights. fit <data id> <family> <alpha> <lambda> <standardize> <weighted>; intercept and coefficients on X's scale.\n")
        for d, (X, y, w) in enumerate(datasets):
            f.write("data %d %d %d\n" % (d, X.shape[0], X.shape[1]))
            f.write(fmt(X) + "\n" + fmt(y) + "\n" + fmt(w) + "\n")
        for d, fam, alpha, lam, std, weighted, c, b in cases:
            f.write("fit %d %s %.17g %.17g %d %d\n" % (d, fam, alpha, lam, int(std), int(weighted)))
            f.write(fmt(np.concatenate([[c], b])) + "\n")
    print("penalized:", len(cases), "cases")

    # NNLS.
    ncases = []
    for m_, n_ in [(30, 12), (12, 30), (50, 5), (96, 10)]:
        A = np.abs(rng.normal(0, 1, size=(m_, n_))) if m_ == 96 else rng.normal(0, 1, size=(m_, n_))
        b = A @ np.maximum(rng.normal(0, 1, size=n_), 0) + 0.3 * rng.standard_normal(m_)
        x, rnorm = nnls(A, b)
        ncases.append((A, b, x, rnorm))
    with open(os.path.join(FIX, "nnls-reference.txt"), "w", encoding="utf-8") as f:
        f.write("# Reference solutions from scipy %s (scipy.optimize.nnls; numpy %s, Python %s); data by numpy's PCG64, seed 20261101.\n"
                % (scipy.__version__, np.__version__, sys.version.split()[0]))
        f.write("# Made by ../make_penalized_fixtures.py on 2026-11-01. case <m> <n>; A by rows; b; x; the residual norm.\n")
        for A, b, x, rnorm in ncases:
            f.write("case %d %d\n" % A.shape)
            f.write(fmt(A) + "\n" + fmt(b) + "\n" + fmt(x) + "\n" + "%.17g\n" % rnorm)
    print("nnls:", len(ncases), "cases")


if __name__ == "__main__":
    main()
