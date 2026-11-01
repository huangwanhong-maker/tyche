"""Reference values for tyche's testing/multiple.hpp and regression/smoothing_spline.hpp, from R.

Writes multiple.txt: a seeded mixture of 800 null and 200 non-null p-values, R's p.adjust (Bonferroni,
Holm, BH), qvalue's pi0est and q-values by its three methods (smoother, the default; bootstrap; fixed
lambda = 0.5), the degrees of freedom R's smoother actually reached, and a smoothing spline by
smooth.spline(df = 5) read at its knots and between them, beside two independent B-spline solutions at R's
reached df: the exact one (penalty integrated with 1/3) and one with R's sgram weight 0.3330, which
reproduces R and shows where R departs from the exact spline.

    RSCRIPT=<path to Rscript with qvalue> python make_multiple_fixtures.py

Run 2026-11-01 with R 4.4.3 (conda-forge), qvalue 2.38.0 (Bioconductor 3.20), numpy 2.5.4, scipy 1.18.1.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent


def run_r(script: str) -> str:
    rscript = os.environ.get("RSCRIPT")
    if not rscript:
        sys.exit("set RSCRIPT to an Rscript that has qvalue")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "script.R"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([rscript, str(path)], capture_output=True, text=True, check=True)
        return out.stdout


def bspline_smoother(x, y, df, slope_weight):
    """The penalised cubic B-spline fit with knots at every x, as R's smooth.spline builds it, its
    penalty's Gram matrix integrated with `slope_weight` for the product of the slopes of the second
    derivatives (1/3 exactly; R's sgram uses 0.3330), at the λ whose trace of the hat matrix is df.
    An independent solution of the smoothing problem (the C++ uses Reinsch's natural-spline form)."""
    t = np.concatenate([[x[0]] * 4, x[1:-1], [x[-1]] * 4])
    nb = len(t) - 4

    def basis(xx, d=0):
        out = np.zeros((len(xx), nb))
        for j in range(nb):
            c = np.zeros(nb)
            c[j] = 1
            b = BSpline(t, c, 3, extrapolate=False)
            out[:, j] = np.nan_to_num((b.derivative(d) if d else b)(xx))
        return out

    B = basis(x)
    B[-1, :] = 0
    B[-1, -1] = 1  # the value at the right end (the half-open basis gives 0 there)
    O = np.zeros((nb, nb))
    for i in range(len(x) - 1):
        a, b2 = x[i], x[i + 1]
        w = b2 - a
        left = basis(np.array([a + 1e-12 * w]), 2)[0]
        right = basis(np.array([b2 - 1e-12 * w]), 2)[0]
        y1, y2 = left, right - left
        O += w * (np.outer(y1, y1) + 0.5 * (np.outer(y2, y1) + np.outer(y1, y2)) + slope_weight * np.outer(y2, y2))

    def hat(lam):
        return B @ np.linalg.solve(B.T @ B + lam * O, B.T)

    lam = np.exp(brentq(lambda s_: np.trace(hat(np.exp(s_))) - df, -30, 30, xtol=1e-14))
    return hat(lam) @ y


def main() -> None:
    rng = np.random.default_rng(20261101)
    from math import erfc, sqrt
    null = rng.uniform(size=800)
    z = rng.normal(3.0, 1.0, size=200)
    alt = np.array([0.5 * erfc(v / sqrt(2)) for v in z])  # one-sided p of N(3, 1) statistics
    p = np.concatenate([null, alt])
    rng.shuffle(p)
    # A second spline example: an irregular grid, a smooth curve and noise.
    x2 = np.sort(rng.uniform(0, 10, size=25))
    y2 = np.sin(x2) + 0.3 * rng.normal(size=25)
    ps = " ".join(f"{v:.17g}" for v in p)
    xs = " ".join(f"{v:.17g}" for v in x2)
    ys = " ".join(f"{v:.17g}" for v in y2)
    script = f"""
suppressMessages(library(qvalue))
p <- c({",".join(f"{v:.17g}" for v in p)})
x2 <- c({",".join(f"{v:.17g}" for v in x2)})
y2 <- c({",".join(f"{v:.17g}" for v in y2)})
w <- function(name, v) cat(name, ":", sprintf("%.17g", v), "\\n")
w("bonferroni", p.adjust(p, "bonferroni"))
w("holm", p.adjust(p, "holm"))
w("bh", p.adjust(p, "BH"))
s <- pi0est(p)
w("smoother_lambda", s$lambda); w("smoother_pi0_lambda", s$pi0.lambda); w("smoother_pi0_smooth", s$pi0.smooth); w("smoother_pi0", s$pi0)
w("smoother_q", qvalue(p)$qvalues)
w("smoother_spline_df", smooth.spline(s$lambda, s$pi0.lambda, df = 3)$df)
b <- pi0est(p, pi0.method = "bootstrap")
w("bootstrap_pi0", b$pi0); w("bootstrap_q", qvalue(p, pi0.method = "bootstrap")$qvalues)
f <- pi0est(p, lambda = 0.5)
w("fixed_pi0", f$pi0); w("fixed_q", qvalue(p, lambda = 0.5)$qvalues)
sp <- smooth.spline(x2, y2, df = 5)
w("spline_df5_fit", predict(sp, x = x2)$y); w("spline_df5_df", sp$df)
w("spline_df5_between", predict(sp, x = c(0.5, 2.5, 5.5, 9.5))$y)
cat("versions :", R.version.string, as.character(packageVersion("qvalue")), "\\n")
"""
    out = run_r(script)
    lines = [
        "# tyche testing: reference values from R (see make_multiple_fixtures.py), retrieved 2026-11-01.",
        f"p : {ps}",
        f"spline_x : {xs}",
        f"spline_y : {ys}",
    ]
    lines += [line.strip() for line in out.splitlines() if line.strip()]
    reached = next(float(l.split(" : ")[1]) for l in lines if l.startswith("spline_df5_df :"))
    exact = bspline_smoother(x2, y2, reached, 1 / 3)
    quirk = bspline_smoother(x2, y2, reached, 0.333)
    lines.append("spline_df5_exact : " + " ".join(f"{v:.17g}" for v in exact))
    lines.append("spline_df5_sgram333 : " + " ".join(f"{v:.17g}" for v in quirk))
    (HERE / "multiple.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
