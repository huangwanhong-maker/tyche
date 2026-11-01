"""Reference segmentations for check_changepoint.cpp, from ruptures.

Writes tests/fixtures/changepoint-reference.txt: series drawn by numpy's PCG64 (seed 20261101) with
planted changes in mean, or in mean and variance, and the breakpoints that ruptures finds on them:

- Pelt(model="l2", jump=1, min_size) with penalty pen: the exact minimiser of
  sum over segments of sum (y - mean)^2 + pen * (number of change points), i.e. tyche's mean model
  with sigma^2 = 1 and Penalty::manual;
- Pelt(model="normal", jump=1, min_size=2, add_small_diag=False): segment cost n log(var), which
  differs from tyche's mean_variance cost by n (log 2 pi + 1), a constant;
- Binseg(model="l2", jump=1, min_size) with pen: splits accepted while the gain exceeds pen.

Run with a Python that has numpy and ruptures:  python make_changepoint_fixtures.py
"""

import os
import sys
import warnings

import numpy as np
import ruptures as rpt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "changepoint-reference.txt")


def planted(rng, n, cps, means, sds):
    y = np.empty(n)
    bounds = [0] + list(cps) + [n]
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        y[a:b] = means[k] + sds[k] * rng.standard_normal(b - a)
    return y


def main():
    rng = np.random.Generator(np.random.PCG64(20261101))
    cases = []
    # Changes in mean, unit noise.
    for c in range(6):
        n = int(rng.integers(150, 400))
        m = int(rng.integers(1, 6))
        cps = sorted(rng.choice(np.arange(10, n - 10), size=m, replace=False).tolist())
        means = rng.normal(0, 2.0, size=m + 1)
        y = planted(rng, n, cps, means, np.ones(m + 1))
        for pen, ms in [(2 * np.log(n), 1), (10.0, 3), (25.0, 5)]:
            bk = rpt.Pelt(model="l2", jump=1, min_size=ms).fit(y).predict(pen=pen)
            cases.append(("pelt", "l2", pen, ms, y, bk))
            bk = rpt.Binseg(model="l2", jump=1, min_size=ms).fit(y).predict(pen=pen)
            cases.append(("binseg", "l2", pen, ms, y, bk))
    # Changes in mean and variance.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in range(4):
            n = int(rng.integers(200, 500))
            m = int(rng.integers(1, 5))
            cps = sorted(rng.choice(np.arange(15, n - 15), size=m, replace=False).tolist())
            means = rng.normal(0, 1.5, size=m + 1)
            sds = np.exp(rng.normal(0, 0.7, size=m + 1))
            y = planted(rng, n, cps, means, sds)
            pen = 3 * np.log(n)
            algo = rpt.Pelt(custom_cost=rpt.costs.CostNormal(add_small_diag=False), jump=1, min_size=2)
            bk = algo.fit(y).predict(pen=pen)
            cases.append(("pelt", "normal", pen, 2, y, bk))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Reference segmentations from ruptures %s (numpy %s, Python %s); series by numpy's PCG64,\n"
                % (rpt.__version__, np.__version__, sys.version.split()[0]))
        f.write("# seed 20261101. Made by ../make_changepoint_fixtures.py on 2026-11-01.\n")
        f.write("# case <algorithm> <cost> <penalty> <min_size> <n>, then the series, then ruptures' breakpoints (ending at n).\n")
        for algo, cost, pen, ms, y, bk in cases:
            f.write("case %s %s %.17g %d %d\n" % (algo, cost, pen, ms, len(y)))
            f.write(" ".join("%.17g" % v for v in y) + "\n")
            f.write(" ".join(str(b) for b in bk) + "\n")
    print("wrote", OUT, len(cases), "cases")


if __name__ == "__main__":
    main()
