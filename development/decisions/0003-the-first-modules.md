# 0003. The First Modules: Probability, Descriptive Statistics, Regression and Time Series

- **Status:** Accepted, provisionally
- **Date:** 2026-10-24
- **Decided by:** the user, who approved tyche with these four modules first, for the economics domain and for nomos's econometrics, which estimates on tyche's regressions; Claude designed their interfaces, their numerics and their checks, and wrote this record.

## 1. Context

The map of the mathematics (§6) expected the first module to be the one a domain needs first. The economics domain and nomos need, before anything else, distributions with accurate tails, the description of samples, linear and generalised linear regression with the covariances applied work uses (heteroskedasticity- and cluster-robust), and the time-series toolkit of macroeconomics: autocorrelation, ARIMA, unit roots and cointegration, filters and vector autoregressions. `estimation` and `testing`, which the map places between `probability` and `regression`, are not yet separate modules: the likelihoods and tests these four need are written where they are used, and will move when a second user appears (an abstraction after its pattern has appeared twice).

## 2. Decision

### 2.1 Modules and order

`probability` stands alone (on calx's special functions and roots); `descriptive` on calx's linear algebra; `regression` on `probability` and calx's QR; `time_series` on all three and on calx's optimisation and eigenvalues. Each is a header-only target `tyche::<module>` in namespace `tyche::<module>`.

### 2.2 Content

- **probability:** `Normal`, `StudentT`, `ChiSquared`, `FisherF`, `Gamma`, `Beta`, `Poisson`, `Binomial` as structs of parameters with `pdf`/`pmf`, `log_pdf`/`log_pmf`, `cdf`, `survival` (computed directly), `quantile`, `upper_quantile`, `mean`, `variance`; the normal quantile by Wichura's AS 241, the others by calx's inverse incomplete gamma or by safeguarded root finding in a transformed variable; `upper_tail`, `lower_tail`, `two_sided`, `two_sided_normal`, `two_sided_t`.
- **descriptive:** compensated and corrected two-pass moments; skewness and kurtosis (g and G of Joanes and Gill); the nine sample quantiles of Hyndman and Fan; weighted mean, variance and quantile (frequency and reliability weights); covariance and correlation of two samples and of the columns of a matrix; `Summary`.
- **regression:** `ols(X, y, OlsOptions)` → `LinearFit`, by Householder QR with column pivoting on unit-norm columns, with weights, the classical, HC0–HC3 and one-way cluster-robust (CR1) covariances, rank detection (an error, or R's aliasing), R², F, the Gaussian log-likelihood and information criteria; `wald(fit, R, r)`; `predict`; `design_matrix`. `glm(X, y, GlmOptions)` → `GlmFit`, by IRLS, for the gaussian (identity), binomial (logit, probit) and Poisson (log) families with prior weights and offsets, model-based, HC0, HC1 and cluster covariances (Poisson with a robust covariance is PPML), deviances, likelihoods and McFadden's pseudo-R².
- **time_series:** `autocovariance`, `acf`, `pacf`, `durbin_levinson`, `ljung_box`; `ar_ols`, `ar_yule_walker`, `arima` by conditional sum of squares (calx's BFGS, Nelder–Mead as a polish), `psi_weights`, `forecast`, `stationary`, `invertible`; `adf` with fixed or AIC/BIC lags, `mackinnon_p` (MacKinnon 1994), `mackinnon_critical` (MacKinnon 2010), `kpss`, `engle_granger`; `hodrick_prescott` by calx's band Cholesky; `var`, `select_order`, `impulse_responses`, `forecast` for VARs.

### 2.3 Interfaces

Inputs are `std::span<const double>` (series, responses, weights), `calx::linalg::Rectangular` (design matrices, n × k, one row per observation; multivariate series, T × K) and option structs with defaults; outputs are plain structs whose fields are documented in the headers, with covariances as square `calx::linalg::Matrix`. Errors of input are `std::invalid_argument` with a message saying what was expected. Conventions (MA signs, divisors, the counting of parameters in AIC, quantile types) are fixed in `docs/glossary.md`.

### 2.4 References and checks

Each module is checked against certified or published values first: NIST's Statistical Reference Datasets for linear regression (Norris, Pontius, NoInt1–2, Longley, Wampler1 and 4, Filip) and univariate statistics (nine datasets, with their lag-1 autocorrelations); Petersen's (2009) test panel for robust and clustered standard errors; Dobson's Poisson example and Bliss's beetles (Dobson and Barnett 2008) for GLMs; MacKinnon's (2010) tables and worked example; Kwiatkowski et al.'s (1992) critical values; Lütkepohl's (2005) VAR example and order selection; tables of the distributions, and 40-digit values computed independently with mpmath. Then closed forms (ψ-weights, forecasts, the independence log-linear model, Bernoulli moments, the Durbin–Levinson recursion on exact autocovariances) and the recovery of known parameters from seeded simulations (calx's random streams), with tolerances from standard errors. statsmodels 0.15.0 is a cross-check of the digits beyond the printed references, labelled as such.

## 3. Alternatives Considered

- **Normal equations for regression.** Rejected: they square the condition number (NIST's Longley and Filip).
- **The exact ARMA likelihood.** Better for short series and near the boundary of invertibility; it needs a state-space form and a Kalman filter, planned with `assimilation`. Conditional sum of squares is what Box and Jenkins start from and what R offers as "CSS".
- **Borrowing a statistics library in C++** (e.g. Boost.Math's distributions, Boost licence). Its distributions would serve, but tyche's tails, quantiles and checks would then be another library's; the regression and time-series methods have no mature licence-compatible C++ implementation to borrow.
- **`estimation` and `testing` as modules now.** Postponed until a second user of their abstractions appears.

## 4. Consequences

- Check programs: `check_tyche_probability`, `check_tyche_descriptive`, `check_tyche_linear`, `check_tyche_glm`, `check_tyche_autocorrelation`, `check_tyche_arima`, `check_tyche_unit_root`, `check_tyche_var`.
- calx gained what these modules needed and does not name statistics (calx decision 0008).
- nomos's econometrics estimates on `tyche::regression`; the economics domain and the Economy Atlas can call all four modules from their engines.
