"""Reference values for tyche's testing/permutation.hpp, from SciPy.

Writes permutation.txt:
- beta_sample: 2000 seeded draws of Beta(0.9, 37) and SciPy's maximum-likelihood fit with location 0 and
  scale 1 fixed (scipy.stats.beta.fit(x, floc=0, fscale=1));
- r2_sample: 1000 seeded squared correlations, the largest of 40 correlated tests per draw, and the
  effective degrees of freedom of tensorQTL's df_cost (the ν at which the moment estimate of the beta's
  first shape of their p-values is 1), found here by SciPy's brentq on tensorQTL's formula
  (Taylor-Weiner et al. 2019, tensorqtl/core.py: df_cost, get_t_pval), with nominal ν = 96;
- student_tail: scipy.stats.t.sf(t, ν) at four (ν, t) where the tail is small and ν large.

    python make_permutation_fixtures.py

Run 2026-11-01 with numpy 2.5.4 and scipy 1.18.1.
"""

from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent


def t_pval(t, dof):
    return 2 * stats.t.cdf(-np.abs(t), dof)


def df_cost(r2, dof):
    pval = t_pval(np.sqrt(r2 * dof / (1 - r2)), dof)
    mean, var = np.mean(pval), np.var(pval)
    return mean * (mean * (1.0 - mean) / var - 1.0) - 1.0


def main() -> None:
    rng = np.random.default_rng(1101)
    x = rng.beta(0.9, 37.0, size=2000)
    a, b, _, _ = stats.beta.fit(x, floc=0, fscale=1)
    # Squared correlations: 1000 draws of the largest r² among 40 AR(1)-correlated predictors (ρ = 0.8)
    # with an independent y, n = 100 (ν = 98 − 2 covariates = 96 nominally).
    n, m = 100, 40
    r2max = []
    for _ in range(1000):
        z = rng.normal(size=(n, m))
        g = np.empty_like(z)
        g[:, 0] = z[:, 0]
        for j in range(1, m):
            g[:, j] = 0.8 * g[:, j - 1] + np.sqrt(1 - 0.64) * z[:, j]
        y = rng.normal(size=n)
        gc = (g - g.mean(0)) / g.std(0)
        yc = (y - y.mean()) / y.std()
        r = gc.T @ yc / n
        r2max.append(np.max(r * r))
    r2max = np.array(r2max)
    dof = optimize.brentq(lambda d: df_cost(r2max, d), 96 / 50, 96 * 50, xtol=1e-12)
    lines = [
        "# tyche testing: reference values from SciPy (see make_permutation_fixtures.py), retrieved 2026-11-01.",
        "beta_sample : " + " ".join(f"{v:.17g}" for v in x),
        f"beta_fit : {a:.17g} {b:.17g}",
        "r2_sample : " + " ".join(f"{v:.17g}" for v in r2max),
        f"effective_df : {dof:.17g}",
    ]
    # Student's upper tail where it is small and ν large (the regime of association scans).
    for k, (df, t) in enumerate(((96, 9), (3000, 10), (3000, 30), (500, 4))):
        lines.append(f"student_tail{k} : {df} {t} {stats.t.sf(t, df):.17g}")
    (HERE / "permutation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
