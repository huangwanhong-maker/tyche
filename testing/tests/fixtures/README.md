# testing/tests/fixtures: Reference Data

| File | Content | Source | Retrieved |
|---|---|---|---|
| `multiple.txt` | 1000 seeded p-values (800 null); R's p.adjust, qvalue's pi0est and q-values (three methods), smooth.spline(df = 5) on 25 points; two B-spline solutions of the same smoothing problem (exact, and with R's 0.3330) | `make_multiple_fixtures.py`, running R 4.4.3 with qvalue 2.38.0 (Bioconductor), numpy 2.5.4 and scipy 1.18.1 | 2026-11-01 |
| `two_sample.txt` | Two pairs of samples with SciPy's Welch test and one-sample t tests; three pairs of counts with statsmodels' Newcombe interval and pooled z test; two shares against a level with the score test and Wilson's interval | `make_two_sample_fixtures.py`, Anaconda's Python 3.12 with numpy 1.26.4, scipy 1.13.1, statsmodels 0.14.2 | 2026-11-18 |
| `online.txt` | ζ(1.6); the normalising sum of Javanmard and Montanari's spending sequence (12.6451078729, by partial sums to 10⁶ and 10⁷ and the exact tail; the constant they print, 0.07720838, makes the sum 0.976); the mixture calibrator at three p, closed form and quadrature | `make_online_fixtures.py`, Anaconda's Python 3.12 with mpmath 1.3.0 and numpy 1.26.4 | 2026-11-18 |
| `permutation.txt` | 2000 draws of Beta(0.9, 37) and SciPy's ML fit; 1000 maxima of 40 correlated r² and tensorQTL's effective df by SciPy; scipy.stats.t.sf at four (ν, t) | `make_permutation_fixtures.py`, numpy 2.5.4, scipy 1.18.1 | 2026-11-01 |
