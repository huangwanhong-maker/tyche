"""Reference values for tyche's regression/variance_components.hpp, from R's regress package.

Writes variance_components.txt: seeded genotypes G (n = 200 samples, m = 400 biallelic sites, dosages
0/1/2), a covariate x, and two traits:
- y1 = 1 + 0.5 x + g + e with g ~ N(0, 0.5 K) and e ~ N(0, 0.5 I), K = ZZᵀ/m the standardised Gram
  matrix of G (columns centred by 2p̂ and scaled by √(2p̂(1 − p̂)), Yang et al. 2011's GRM);
- y2 = 1 + 0.5 x + g1 + g2 + e with g1 ~ N(0, 0.3 K1), g2 ~ N(0, 0.3 K2), e ~ N(0, 0.4 I), K1 and K2
  the Gram matrices of the first and the last 200 sites;
and regress(y1 ~ x, ~K) and regress(y2 ~ x, ~K1 + K2) (REML by Fisher scoring; Clifford and McCullagh
2006, R News 6/2, 6–10): the variance components (kernels, then the identity), their covariance (the
inverse Fisher information), the fixed effects, and the REML log-likelihood ℓ_R of Harville's form
−½[log|V| + log|XᵀV⁻¹X| − log|XᵀX| + yᵀPy + (n − r) log 2π] evaluated by R's own matrix algebra at
regress's estimates.

    RSCRIPT=<path to Rscript with regress> python make_variance_components_fixtures.py

Run 2026-11-01 with R 4.4.3 (conda-forge), regress 1.3.22 (CRAN), numpy 2.5.4.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def run_r(script: str) -> str:
    rscript = os.environ.get("RSCRIPT")
    if not rscript:
        sys.exit("set RSCRIPT to an Rscript that has regress")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "script.R"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([rscript, str(path)], capture_output=True, text=True)
        if out.returncode:
            sys.exit(out.stderr)
        return out.stdout


def gram(G):
    p = G.mean(0) / 2
    Z = (G - 2 * p) / np.sqrt(2 * p * (1 - p))
    return Z @ Z.T / G.shape[1]


def main() -> None:
    rng = np.random.default_rng(1102)
    n, m = 200, 400
    f = rng.uniform(0.05, 0.5, size=m)
    G = rng.binomial(2, f, size=(n, m))
    keep = (G.sum(0) > 0) & (G.sum(0) < 2 * n)
    assert keep.all(), "every site polymorphic in the sample"
    x = rng.normal(size=n)
    K, K1, K2 = gram(G), gram(G[:, :200]), gram(G[:, 200:])

    def draw(cov):
        w, v = np.linalg.eigh(cov)
        return v @ (np.sqrt(np.clip(w, 0, None)) * rng.normal(size=len(w)))

    y1 = 1 + 0.5 * x + draw(0.5 * K) + rng.normal(scale=np.sqrt(0.5), size=n)
    y2 = 1 + 0.5 * x + draw(0.3 * K1) + draw(0.3 * K2) + rng.normal(scale=np.sqrt(0.4), size=n)
    gs = ",".join(str(v) for v in G.flatten())
    fmt = lambda a: ",".join(f"{v:.17g}" for v in a)
    script = f"""
suppressMessages(library(regress))
G <- matrix(c({gs}), nrow = {n}, byrow = TRUE)
x <- c({fmt(x)}); y1 <- c({fmt(y1)}); y2 <- c({fmt(y2)})
gram <- function(G) {{ p <- colMeans(G) / 2; Z <- sweep(G, 2, 2 * p) / matrix(sqrt(2 * p * (1 - p)), nrow(G), ncol(G), byrow = TRUE); tcrossprod(Z) / ncol(G) }}
K <- gram(G); K1 <- gram(G[, 1:200]); K2 <- gram(G[, 201:400])
X <- cbind(1, x)
reml <- function(y, Ks, s) {{
  V <- diag(s[length(s)], length(y)); for (i in seq_along(Ks)) V <- V + s[i] * Ks[[i]]
  Vi <- solve(V); XVX <- t(X) %*% Vi %*% X; P <- Vi - Vi %*% X %*% solve(XVX) %*% t(X) %*% Vi
  -0.5 * (determinant(V)$modulus + determinant(XVX)$modulus - determinant(crossprod(X))$modulus + drop(t(y) %*% P %*% y) + (length(y) - ncol(X)) * log(2 * pi))
}}
w <- function(name, v) cat(name, ":", sprintf("%.17g", as.numeric(v)), "\\n")
f1 <- regress(y1 ~ x, ~K, tol = 1e-10, maxcyc = 500)
w("one_sigma", f1$sigma); w("one_sigma_cov", f1$sigma.cov); w("one_beta", f1$beta); w("one_reml", reml(y1, list(K), f1$sigma))
f2 <- regress(y2 ~ x, ~K1 + K2, tol = 1e-10, maxcyc = 500)
w("two_sigma", f2$sigma); w("two_sigma_cov", f2$sigma.cov); w("two_beta", f2$beta); w("two_reml", reml(y2, list(K1, K2), f2$sigma))
cat("versions :", R.version.string, as.character(packageVersion("regress")), "\\n")
"""
    out = run_r(script)
    lines = [
        "# tyche regression: R regress's REML fits on seeded data (see make_variance_components_fixtures.py), retrieved 2026-11-01.",
        f"dims : {n} {m}",
        "G : " + " ".join(str(v) for v in G.flatten()),
        "x : " + " ".join(f"{v:.17g}" for v in x),
        "y1 : " + " ".join(f"{v:.17g}" for v in y1),
        "y2 : " + " ".join(f"{v:.17g}" for v in y2),
    ]
    lines += [line.strip() for line in out.splitlines() if line.strip()]
    (HERE / "variance_components.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
