"""Cross-check values for tyche::compositions from scikit-bio's composition module.

Run with scikit-bio installed (any scratch environment):

    python make_skbio_composition_fixtures.py > skbio_composition.txt

Each line is a kind, a name, and values separated by commas: compositions, their closure, clr and
inverse clr, perturbation, its inverse and powering, the Aitchison inner product, ilr coordinates in
the bases of sequential binary partitions (sbp_basis), multiplicative replacement of zeros
(multi_replace, closed to 1), the variation matrix (pairwise_vlr, ddof 1) and a sample centred on its
geometric mean (centralize). Partitions are written row by row, rows separated by ';'. These are
cross-checks, not references: the references of the checks are the identities and closed forms in
check_simplex.cpp and check_composition_sample.cpp.
"""

import datetime

import numpy as np
import skbio
from skbio.stats import composition as cs


def fmt(values):
    return ",".join(repr(float(v)) for v in np.ravel(values))


COMPOSITIONS = {
    "day": [480.0, 420.0, 300.0, 120.0, 90.0, 30.0],  # minutes of a day, six activities
    "three": [0.1, 0.3, 0.6],
    "five": [2.0, 7.0, 1.0, 13.0, 4.0],
}
SBPS = {
    "day": [[1, 1, -1, -1, -1, -1], [1, -1, 0, 0, 0, 0], [0, 0, 1, -1, -1, -1], [0, 0, 0, 1, -1, -1], [0, 0, 0, 0, 1, -1]],
    "five": [[1, -1, 1, -1, -1], [1, 0, -1, 0, 0], [0, 1, 0, 1, -1], [0, 1, 0, -1, 0]],
    "three": [[1, -1, -1], [0, 1, -1]],
}
ZEROS = {"zeros": [0.2, 0.4, 0.4, 0.0], "zeros2": [0.0, 0.5, 0.5, 0.0], "zeros3": [3.0, 0.0, 5.0, 2.0, 0.0]}
SAMPLE = np.array(
    [
        [480, 420, 300, 120, 90, 30],
        [510, 360, 270, 180, 60, 60],
        [450, 480, 330, 90, 45, 45],
        [420, 300, 420, 150, 120, 30],
        [540, 390, 240, 135, 75, 60],
        [465, 435, 285, 105, 105, 45],
        [495, 330, 375, 120, 80, 40],
    ],
    dtype=float,
)

print(f"# scikit-bio {skbio.__version__}, numpy {np.__version__}; generated {datetime.date.today().isoformat()} by make_skbio_composition_fixtures.py")
for name, x in COMPOSITIONS.items():
    x = np.array(x)
    print("composition", name, fmt(x))
    print("closure", name, fmt(cs.closure(x)))
    print("clr", name, fmt(cs.clr(x)))
    print("clr_inv", name, fmt(cs.clr_inv(cs.clr(x) + 0.25 * np.arange(len(x)))))
    print("power", name, fmt(cs.power(x, 0.7)))
    print("perturb_inv", name, fmt(cs.perturb_inv(x, x[::-1])))
    print("perturb", name, fmt(cs.perturb(x, x[::-1])))
    print("inner", name, fmt([cs.inner(x, x[::-1])]))
    sbp = np.array(SBPS[name])
    print("sbp", name, ";".join(",".join(str(int(s)) for s in row) for row in sbp))
    print("ilr", name, fmt(cs.ilr(x, cs.sbp_basis(sbp))))
for name, x in ZEROS.items():
    x = np.array(x)
    print("zeros", name, fmt(x))
    print("multi_replace", name, fmt(cs.multi_replace(x[np.newaxis, :])))
print("sample", "day", ";".join(",".join(repr(float(v)) for v in row) for row in SAMPLE))
print("pairwise_vlr", "day", fmt(cs.pairwise_vlr(SAMPLE, ddof=1).data))
print("centralize", "day", fmt(cs.centralize(SAMPLE)))
