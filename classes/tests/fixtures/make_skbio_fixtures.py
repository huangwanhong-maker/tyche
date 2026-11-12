"""Cross-check values for tyche::classes from scikit-bio's alpha diversity.

Run with scikit-bio installed (any scratch environment):

    python make_skbio_fixtures.py > skbio_alpha.txt

Each sample's counts are written, then scikit-bio's Hill numbers and Rényi entropies at several orders,
its ACE (rare threshold 10), its Chao1 in both forms (without the factor (n - 1)/n that tyche's carries,
which the check accounts for), and Good's coverage 1 - f1/n. These are cross-checks, not references:
the references of the checks are the closed forms and hand-worked examples in check_richness.cpp.
"""

import datetime

import numpy as np
import skbio
from skbio.diversity import alpha

SAMPLES = {
    "small": [1, 1, 1, 2, 2, 3, 5, 8],
    "zeros": [0, 4, 1, 0, 7, 1, 2, 2, 12, 1, 3, 0, 1],
    "rare": [1, 1, 1, 2, 4, 6, 12, 15],
    "even": [5, 5, 5, 5, 5, 5],
    "long": [int(x) for x in np.random.default_rng(20261112).negative_binomial(0.6, 0.08, size=120)],
}
ORDERS = [0, 0.5, 1, 2, 3, float("inf")]

print(f"# scikit-bio {skbio.__version__}, numpy {np.__version__}; generated {datetime.date.today().isoformat()} by make_skbio_fixtures.py")
for name, counts in SAMPLES.items():
    c = np.array(counts, dtype=int)
    print("sample", name, ",".join(str(x) for x in counts))
    for q in ORDERS:
        print("hill", name, repr(q), repr(float(alpha.hill(c, order=q))))
        print("renyi", name, repr(q), repr(float(alpha.renyi(c, order=q))))
    try:
        print("ace", name, repr(float(alpha.ace(c))))
    except ValueError:
        print("ace", name, "undefined")
    print("chao1_bias_corrected", name, repr(float(alpha.chao1(c, bias_corrected=True))))
    print("chao1_uncorrected", name, repr(float(alpha.chao1(c, bias_corrected=False))))
    print("goods_coverage", name, repr(float(alpha.goods_coverage(c))))
