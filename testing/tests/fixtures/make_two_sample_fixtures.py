"""Reference values for tyche's testing/two_sample.hpp, from SciPy and statsmodels.

Writes two_sample.txt: two pairs of samples (written out, so that the check reads the same numbers) with SciPy's Welch test
(ttest_ind, equal_var=False) and its confidence interval; a one-sample t test against a level; statsmodels' Newcombe interval for a
difference of two shares (confint_proportions_2indep, method="newcomb") and its pooled z test (proportions_ztest); the score test
of one share against a level (proportions_ztest with prop_var at the level) and Wilson's interval (proportion_confint, "wilson").

    python make_two_sample_fixtures.py

Run 2026-11-18 with Anaconda's Python 3.12, numpy 1.26.4, scipy 1.13.1 and statsmodels 0.14.2.
"""

from pathlib import Path

import numpy as np
import scipy
import statsmodels
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint, proportions_ztest

rng = np.random.default_rng(20261118)
out = [f"# tyche testing: two-sample references from SciPy {scipy.__version__} and statsmodels {statsmodels.__version__} (make_two_sample_fixtures.py), retrieved 2026-11-18."]


def put(name, values):
    out.append(f"{name} : " + " ".join(repr(float(v)) for v in np.atleast_1d(values)))


a1 = rng.normal(10.0, 2.0, 23)
b1 = rng.normal(11.3, 3.5, 31)
a2 = rng.normal(0.0, 1.0, 7)
b2 = rng.normal(0.4, 0.3, 12)
for k, (a, b) in enumerate([(a1, b1), (a2, b2)], start=1):
    put(f"a{k}", a)
    put(f"b{k}", b)
    r = stats.ttest_ind(a, b, equal_var=False)
    ci = r.confidence_interval(0.95)
    put(f"welch{k}", [r.statistic, r.df, r.pvalue, ci.low, ci.high, a.mean() - b.mean()])
    r1 = stats.ttest_1samp(a, 9.5 if k == 1 else 0.3)
    c1 = r1.confidence_interval(0.95)
    put(f"one{k}", [r1.statistic, r1.df, r1.pvalue, c1.low, c1.high])

for k, (x1, n1, x2, n2) in enumerate([(42, 120, 25, 130), (3, 17, 9, 15), (0, 30, 4, 28)], start=1):
    low, high = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb", compare="diff", alpha=0.05)
    z, p = proportions_ztest([x1, x2], [n1, n2])
    put(f"shares{k}", [x1, n1, x2, n2])
    put(f"newcombe{k}", [low, high, z, p])

for k, (x, n, level) in enumerate([(31, 200, 0.10), (8, 40, 0.30)], start=1):
    z, p = proportions_ztest(x, n, value=level, prop_var=level)
    lo, hi = proportion_confint(x, n, alpha=0.05, method="wilson")
    put(f"level{k}", [x, n, level, z, p, lo, hi])

Path(__file__).with_name("two_sample.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote two_sample.txt")
