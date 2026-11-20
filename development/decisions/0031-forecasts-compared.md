# 0031. Forecasts Compared: Naive Forecasts, the Long-Run Variance and the Test of Equal Accuracy

- **Status:** Accepted, provisionally (2026-11-21).
- **Date:** 2026-11-21
- **Decided by:** Claude, for kinesis's laws learned from a record and held to baselines (its decision 0009), at the infrastructure's Stage 7 of evolutionary allocation (its ADR 0139; the survey's Part VIII §6, recommendation A.3).

## 1. Context

A law learned from a record must be held to the forecasts that need no law before it is used to evaluate anything: the random walk, the mean, a fitted autoregression. Comparing two forecasts needs a test of equal accuracy, and the test needs the long-run variance of the loss differential. tyche used Newey and West's estimator inside KPSS and had neither a standalone long-run variance nor a test of equal forecast accuracy.

## 2. Decision

`time_series/forecasts.hpp`:

- `Forecaster` (a history to a forecast), `random_walk`, `historical_mean`, `ar1` (least squares on the history; the last value when too short or constant);
- `rolling(y, start, f)`: one-step forecasts from rolling origins, each from the history before the value it forecasts, with their mean squared error;
- `long_run_variance(d, lags)`: Newey and West's estimator with Bartlett's weights (1987), ⌊4 (T/100)^{2/9}⌋ lags by default;
- `diebold_mariano(truth, a, b, h, lags)`: the statistic on squared losses with h − 1 autocovariances by default (Diebold and Mariano 1995), Harvey, Leybourne and Newbold's correction referred to Student's t with T − 1 degrees of freedom (1997).

## 3. Alternatives Considered

- **Clark and West's test for nested models, and Giacomini and White's conditional test.** Planned; the test of equal accuracy of forecasts is the first a learned law needs.
- **Placing it in kinesis.** It names no law, and is used wherever forecasts are compared.

## 4. Consequences

`check_tyche_forecasts` (13 checks): the long-run variance of white noise with no lags equal to its variance; that of an AR(1) series against its expectation under Bartlett's weights; identical forecasts giving 0 and p = 1; the corrected statistic and its p-value exactly; the size, 5.0% on 4,000 runs of equally good forecasts; the better forecast found; rolling forecasts' mean squared errors σ², 2σ²/(1 + φ) and σ²/(1 − φ²); the autoregression on a short or constant history.
