# testing/tests/fixtures: Reference Data

| File | Content | Source | Retrieved |
|---|---|---|---|
| `multiple.txt` | 1000 seeded p-values (800 null); R's p.adjust, qvalue's pi0est and q-values (three methods), smooth.spline(df = 5) on 25 points; two B-spline solutions of the same smoothing problem (exact, and with R's 0.3330) | `make_multiple_fixtures.py`, running R 4.4.3 with qvalue 2.38.0 (Bioconductor), numpy 2.5.4 and scipy 1.18.1 | 2026-11-01 |
| `permutation.txt` | 2000 draws of Beta(0.9, 37) and SciPy's ML fit; 1000 maxima of 40 correlated r² and tensorQTL's effective df by SciPy; scipy.stats.t.sf at four (ν, t) | `make_permutation_fixtures.py`, numpy 2.5.4, scipy 1.18.1 | 2026-11-01 |
