# regression/tests/fixtures: Reference Data

Each file is kept exactly as published; the checks read it from here (`TYCHE_FIXTURES`).

| File | Content | Source | Retrieved |
|---|---|---|---|
| `nist-norris.dat` | Norris (calibration of ozone monitors): y on x, 36 observations; certified β, standard errors, residual standard deviation, R², ANOVA | NIST/ITL Statistical Reference Datasets, Linear Least Squares Regression, lower difficulty. https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Norris.dat | 2026-10-24 |
| `nist-pontius.dat` | Pontius (load cell calibration): quadratic in x, 40 observations | https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Pontius.dat | 2026-10-24 |
| `nist-noint1.dat`, `nist-noint2.dat` | Regression through the origin (no intercept; R² uncentred), 11 and 3 observations | https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/NoInt1.dat, …/NoInt2.dat | 2026-10-24 |
| `nist-longley.dat` | Longley (1967): US employment on six macroeconomic series, 16 observations, higher difficulty (highly collinear) | https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat | 2026-10-24 |
| `nist-wampler1.dat`, `nist-wampler4.dat` | Wampler (1970): polynomials of degree 5, 21 observations; Wampler1 an exact fit, Wampler4 with large residuals; higher difficulty | https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Wampler1.dat, …/Wampler4.dat | 2026-10-24 |
| `nist-filip.dat` | Filip: a polynomial of degree 10, 82 observations, higher difficulty (condition number about 10¹⁵ unscaled) | https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Filip.dat | 2026-10-24 |
| `petersen-test-data.txt` | Petersen's test panel: 500 firms × 10 years, columns firm, year, x, y (simulated with a firm effect in x and in the residual) | M. A. Petersen (2009), "Estimating standard errors in finance panel data sets", Rev. Financial Studies 22, 435–480; test data and the standard errors it gives (OLS, White, clustered by firm and by year, computed in Stata) at https://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt and …/test_data.htm | 2026-10-24 |

The NIST certified values were computed in 500-digit arithmetic and are given to 15 significant digits (NIST, *StRD: Background on the linear least squares datasets*). NIST data are works of the US Government and in the public domain. Petersen's published standard errors are printed to four decimals; Stata's "robust" is HC1 and its clustered errors use the CR1 factor G/(G − 1) · (n − 1)/(n − k).
