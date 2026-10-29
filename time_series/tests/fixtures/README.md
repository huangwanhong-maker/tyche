# time_series/tests/fixtures: Reference Data

| File | Content | Source | Retrieved |
|---|---|---|---|
| `lutkepohl-e1.dat` | West German fixed investment, disposable income and consumption expenditures, quarterly, seasonally adjusted, billions of DM, 1960Q1–1982Q4 (source: Deutsche Bundesbank) | The data of H. Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*, Springer, Table E.1, as distributed with JMulTi: http://www.jmulti.de/download/datasets/e1.dat (the book's data page, http://www.jmulti.de/data_imtsa.html) | 2026-10-24 |

The checks of `var` compare with the estimates Lütkepohl prints for a VAR(2) of the first differences of the logarithms, 1960Q1–1978Q4 (§3.2.3), and with his choice of order (§4.3). The univariate checks of `autocorrelation` read NIST's univariate datasets from `descriptive/tests/fixtures`.
