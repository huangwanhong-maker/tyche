"""Reference values for tyche's regression/susie.hpp, from the authors' R package susieR.

Writes susie.txt: a seeded genotype-like matrix X (n = 400 samples, p = 80 columns in linkage, dosages
0/1/2 from pairs of thresholded AR(1) Gaussian haplotypes), a trait y with three non-zero effects, and
susieR's fits:
- `ind`: susie(X, y, L = 10), the defaults (prior variances by "optim", residual variance estimated);
- `em`: susie(X, y, L = 5, estimate_prior_method = "EM");
- `ser`: susie(X, y, L = 1, scaled_prior_variance = 0.1, residual_variance = 0.9 var(y),
  estimate_prior_variance = FALSE, estimate_residual_variance = FALSE): the single-effect regression;
- `ss`: susie_suff_stat(XtX, Xty, yty, n, L = 10) of the centred data;
- `rss`: susie_rss(z, R, n, L = 10) with z = univariate_regression(X, y)'s β̂/ŝ and R = cor(X)
  (susie_rss's default: the residual variance fixed).
For each: alpha, mu, mu2 (L × p, row-major), V, sigma2, elbo, niter, pip, lbf, the credible sets (their
effects, members and purity), and for `ind` the intercept and coefficients.

    RSCRIPT=<path to Rscript with susieR> python make_susie_fixtures.py

Run 2026-11-01 with R 4.4.3 (conda-forge) and susieR 0.14.2; numpy 2.5.4, scipy 1.18.1.
"""

import os
import subprocess
import sys
import tempfile
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.special import erfinv

HERE = Path(__file__).resolve().parent


def genotypes(rng, n, p, rho=0.975):
    f = rng.uniform(0.1, 0.5, size=p)
    thresh = sqrt(2) * erfinv(1 - 2 * f)  # P(z > threshold) = f
    x = np.zeros((n, p), dtype=int)
    for copy in range(2):
        z = np.empty((n, p))
        z[:, 0] = rng.normal(size=n)
        for j in range(1, p):
            z[:, j] = rho * z[:, j - 1] + sqrt(1 - rho * rho) * rng.normal(size=n)
        x += (z > thresh).astype(int)
    return x


def run_r(script: str) -> str:
    rscript = os.environ.get("RSCRIPT")
    if not rscript:
        sys.exit("set RSCRIPT to an Rscript that has susieR")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "script.R"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([rscript, str(path)], capture_output=True, text=True)
        if out.returncode:
            sys.exit(out.stderr)
        return out.stdout


def main() -> None:
    rng = np.random.default_rng(64)
    n, p = 400, 80
    X = genotypes(rng, n, p)
    b = np.zeros(p)
    b[[12, 40, 46]] = [0.3, -0.25, 0.25]
    g = X @ b
    y = g + rng.normal(scale=np.std(g) * 2.0, size=n)
    xs = ",".join(str(v) for v in X.flatten())
    ys = ",".join(f"{v:.17g}" for v in y)
    script = f"""
suppressMessages(library(susieR))
X <- matrix(c({xs}), nrow = {n}, byrow = TRUE)
y <- c({ys})
w <- function(name, v) cat(name, ":", sprintf("%.17g", as.numeric(v)), "\\n")
put <- function(tag, fit) {{
  w(paste0(tag, "_alpha"), t(fit$alpha)); w(paste0(tag, "_mu"), t(fit$mu)); w(paste0(tag, "_mu2"), t(fit$mu2))
  w(paste0(tag, "_V"), fit$V); w(paste0(tag, "_sigma2"), fit$sigma2); w(paste0(tag, "_elbo"), fit$elbo)
  w(paste0(tag, "_niter"), fit$niter); w(paste0(tag, "_pip"), fit$pip); w(paste0(tag, "_lbf"), fit$lbf)
  cs <- fit$sets
  if (is.null(cs$cs)) w(paste0(tag, "_cs_count"), 0) else {{
    w(paste0(tag, "_cs_count"), length(cs$cs)); w(paste0(tag, "_cs_index"), cs$cs_index - 1)
    for (k in seq_along(cs$cs)) w(paste0(tag, "_cs", k - 1), cs$cs[[k]] - 1)
    w(paste0(tag, "_cs_min_abs_corr"), cs$purity[, 1]); w(paste0(tag, "_cs_mean_abs_corr"), cs$purity[, 2])
    w(paste0(tag, "_cs_coverage"), cs$coverage)
  }}
}}
f <- susie(X, y, L = 10); put("ind", f)
w("ind_intercept", f$intercept); w("ind_coef", coef(f)[-1]); w("ind_fitted", f$fitted)
put("em", susie(X, y, L = 5, estimate_prior_method = "EM"))
put("ser", susie(X, y, L = 1, scaled_prior_variance = 0.1, residual_variance = 0.9 * var(y),
                 estimate_prior_variance = FALSE, estimate_residual_variance = FALSE))
Xc <- scale(X, center = TRUE, scale = FALSE); yc <- y - mean(y)
put("ss", susie_suff_stat(XtX = crossprod(Xc), Xty = drop(crossprod(Xc, yc)), yty = sum(yc^2), n = nrow(X), L = 10))
u <- univariate_regression(X, y); z <- u$betahat / u$sebetahat
w("rss_z", z)
put("rss", susie_rss(z = z, R = cor(X), n = nrow(X), L = 10))
cat("versions :", R.version.string, as.character(packageVersion("susieR")), "\\n")
"""
    out = run_r(script)
    lines = [
        "# tyche regression: susieR's fits on seeded data (see make_susie_fixtures.py), retrieved 2026-11-01.",
        f"dims : {n} {p}",
        "X : " + " ".join(str(v) for v in X.flatten()),
        "y : " + " ".join(f"{v:.17g}" for v in y),
    ]
    lines += [line.strip() for line in out.splitlines() if line.strip()]
    (HERE / "susie.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
