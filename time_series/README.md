# time_series: Autocorrelation, ARIMA, Unit Roots, Filters and Vector Autoregressions

*Status: draft (2026-10-24) ([decision 0003](../development/decisions/0003-the-first-modules.md)); spectral inference, harmonic analysis and Lomb–Scargle (2026-10-31, [decision 0005](../development/decisions/0005-spectral-inference-harmonics-and-point-patterns.md)).* The exact ARMA likelihood and state-space models (with the Kalman filter), seasonal ARIMA, multivariate spectra, GARCH, Johansen's cointegration test, structural VARs, local projections and change points are planned.

## 1. Concepts

A time series is one realisation of a process; its statistics rest on stationarity, the invariance of its joint distributions under shifts in time. The autocorrelation function describes the linear memory of a stationary series, and the partial autocorrelation function (the last coefficient of the best linear predictor of each order, by the Durbin–Levinson recursion) separates autoregressive memory from moving-average memory: an AR(p) has partial autocorrelations zero beyond p, an MA(q) autocorrelations zero beyond q. ARIMA models combine both after d differences; their moving-average representation, the ψ-weights, gives the variance of forecasts.

Many economic series are not stationary but integrated: a random walk's shocks never die out. The Dickey–Fuller test asks whether ρ = 0 in Δy_t = ρy_{t−1} + …; under that null, ρ̂'s t statistic does not have a t distribution but a functional of Brownian motion, tabulated by simulation (MacKinnon's response surfaces). KPSS reverses the hypotheses (stationarity under the null). Two integrated series are cointegrated when a combination of them is stationary; the Engle–Granger test applies a Dickey–Fuller test to the residuals of their regression, with critical values that account for the estimated combination.

A vector autoregression models several series together, each on the past of all; its companion matrix decides stability; its impulse responses trace a shock through the system, orthogonalised by the Cholesky factor of the innovations' covariance (so the ordering of the variables matters). The Hodrick–Prescott filter splits a series into a smooth trend and a cycle by penalised least squares.

A stationary series' variance is spread over frequency by its spectrum, which calx's `signal` estimates (periodograms, Welch's and Thomson's estimates, cross spectra, bispectra); what an estimate means is statistics. An estimate with ν equivalent degrees of freedom is S χ²_ν/ν, which gives its confidence interval. A peak is a line only if it stands above what the background would show by chance, and the background of most geophysical and social series is red: an AR(1) of the series' own lag-one autocorrelation, with its level at each frequency, and surrogates (phase-randomised copies, AR(1) paths) for nulls that keep more of the series. Coherence and bicoherence have levels that independent series exceed at a known rate. Rhythms of known frequency (a day, a week, a lunar month, a year) are measured as tides are, by harmonic analysis: least squares at those frequencies, with amplitude and phase, Rayleigh's criterion for what a record of length T can resolve, and errors from the residual's spectrum near each frequency rather than its average, since a red residual makes low-frequency constituents less certain than white noise would. Series sampled unevenly have the Lomb–Scargle periodogram, a least-squares fit of one sinusoid at each frequency, with Baluev's probability that its highest peak is chance.

## 2. Content

| Header | Content |
|---|---|
| `autocorrelation.hpp` | `autocovariance(x, max_lag, demean)` (divisor n), `acf`, `pacf` (Yule–Walker), `durbin_levinson(gamma, order)` → `{phi, partial, variance}`, `ljung_box(x, lags, fitted_parameters)` → `{statistic, p_value, lags, df}` |
| `arima.hpp` | `ar_ols(x, p, constant)` and `ar_yule_walker(x, p)` → `ArFit{p, constant, mean, phi, sigma2, se, is_stationary}`; `arima(x, ArimaOrder{p, d, q}, ArimaOptions{constant, confidence})` → `ArimaFit{mu, phi, theta, se, covariance, sigma2, css, n_residuals, log_likelihood, aic, bic, residuals, converged, is_stationary, is_invertible}`; `forecast(fit, h)` → `Forecast{mean, se, lower, upper}`; `psi_weights(phi, theta, n)`; `stationary(phi)`, `invertible(theta)`, `companion_radius` |
| `unit_root.hpp` | `adf(y, AdfOptions{deterministic, selection, lags})` → `AdfResult{statistic, p_value, critical, lags, n_obs, rho, information_criterion, regression}`; `mackinnon_p(tau, deterministic, N)` (1994); `mackinnon_critical(deterministic, N, T)` (2010); `kpss(y, trend, lags)` → `KpssResult{statistic, p_value, lags, critical}`; `engle_granger(y, X, deterministic, adf_options)` → `EngleGrangerResult{statistic, p_value, critical, lags, cointegrating_regression, adf}`; `Deterministic{none, constant, trend, quadratic_trend}`, `LagSelection{fixed, aic, bic}` |
| `filters.hpp` | `hodrick_prescott(y, lambda = 1600)` → `HpFilter{trend, cycle}`, by calx's band Cholesky in O(n) |
| `spectral.hpp` | `spectral_interval(estimate, dof, confidence)` → `Interval{lower, upper}`; `welch_dof(segments, L, overlap)`; `coherence_level(K, α)`; `bicoherence_level(S, α)`; `RedNoise{rho, variance, density(f, fs)}`, `red_noise(x)`, `red_noise_level(r, f, fs, dof, confidence)`; `phase_surrogate(x, rng)`, `red_noise_surrogate(r, n, rng)`; `alias(f, fs)` |
| `harmonic.hpp` | `harmonic_fit(t, y, constituents, HarmonicOptions{t0, trend, coloured, rayleigh, band})` → `HarmonicFit{mean, trend, terms, fitted, residual, record_length, variance_explained, observations, coloured}` with `HarmonicTerm{name, frequency, amplitude, phase, a, b, amplitude_se, phase_se, snr, resolved}`; `resolvable(f1, f2, T, R)`; `harmonic_predict(fit, t, t0)` |
| `lomb_scargle.hpp` | `lomb_scargle(t, y, frequency, floating_mean)` → `LombScargle{frequency, power}` (astropy's standard normalisation); `false_alarm_baluev(power, t, y, fmax)` |
| `var.hpp` | `var(Y, p, constant)` → `VarFit{K, p, T, T_effective, nu, A, sigma_u, sigma_ml, residuals, equations, log_likelihood, aic, bic, hq, fpe, spectral_radius, is_stable}` (Y is a `Rectangular`, T × K); `select_order(Y, p_max)` → `VarOrder{aic, bic, hq, fpe, …_values}`; `impulse_responses(fit, h, orthogonalised)` (a K × K `Matrix` per horizon); `forecast(fit, h)` → `VarForecast{mean, se}` |

Conventions (signs of the MA polynomial, divisors, information criteria, one-sided densities, the phases of cross spectra and of harmonic constituents) are in the [glossary](../docs/glossary.md), §4.

## 3. Checks

- `check_tyche_autocorrelation`: the certified lag-1 autocorrelations of NIST's nine univariate datasets; Durbin–Levinson on an AR(2)'s and an MA(1)'s exact autocovariances; acf, pacf and Ljung–Box on Lütkepohl's income data against statsmodels (cross-check); the size of the Ljung–Box test on white noise.
- `check_tyche_arima`: ψ-weights of an ARMA(1, 1) in closed form; roots on either side of the unit circle; AR(2) by least squares and Yule–Walker within four standard errors; CSS for a pure AR equal to least squares; an ARMA(1, 1) of 2000 recovered, with Hessian standard errors within 10% of the asymptotic ones; ARIMA(0, 1, 1) with drift; forecasts of an AR(1) and of a random walk with drift in closed form.
- `check_tyche_unit_root`: MacKinnon's (2010) asymptotic critical values and his worked example (−4.89111); his 1994 p-values at his 2010 critical values (within 0.003); the ADF statistic as a t statistic; ADF (constant, trend, none; AIC, BIC, fixed lags), KPSS and Engle–Granger on Lütkepohl's data against statsmodels, with the Engle–Granger 1% value from MacKinnon's table (statsmodels' coefficient for N = 2 is mistyped); KPSS's critical values (Kwiatkowski et al. 1992, Table 1) and its statistic rebuilt; the size of ADF and KPSS and the power of ADF, and Engle–Granger on cointegrated and independent pairs, in seeded simulations.
- `check_tyche_spectral` (33): spectral intervals and red-noise levels against scipy's χ² quantiles, and the intervals' coverage; Welch's degrees of freedom in closed form (ρ(1) = 1/6) and by simulation; the rates at which independent Gaussian series exceed the coherence and bicoherence levels; the red-noise density's integral, the mean periodogram of AR(1) paths, ρ and σ² recovered; phase surrogates keeping the periodogram to rounding; M2 aliased to 14.765 days (daily sampling) and 62.11 days (TOPEX/Poseidon); Lomb–Scargle (floating and fixed mean) and Baluev's false-alarm probability against astropy 8.0.2, and the classical periodogram at a Fourier frequency.
- `check_tyche_harmonic` (25): noise-free amplitudes, phases, mean and trend recovered exactly from hourly and from uneven, gapped times; the phase convention; Rayleigh's flags and the refusals; 95 % coverage of white, coloured and gapped errors in simulation; under AR(1) residuals the coloured errors exceed the white ones as √(S(f)/(2σ²/fs)) and cover 95 %, where the white ones cover 74 %.
- `check_tyche_var`: Lütkepohl's (2005, §3.2.3) VAR(2) estimates and residual covariance as printed, and further digits, standard errors, log-likelihood, impulse responses and forecasts against statsmodels; the orders chosen by AIC, FPE, HQ and SC (§4.3); Θ₀ = P, Φ₁ = A₁, Φ₂ = A₁² + A₂; a simulated VAR(2) recovered and its order chosen; a VAR(1)'s forecasts and MSE in closed form; an explosive VAR flagged; the HP filter's exact properties, λ = 0, and log income's trend against statsmodels.

## 4. Dependencies

`regression`, `descriptive`, `probability`; calx `linalg` (`eigenvalues`, `cholesky`, band Cholesky, QR), `optimisation` (BFGS, Nelder–Mead, Hessians by differences), `signal` and `transforms` (spectral estimates, the FFT), `random` (surrogates) and `special_functions`.

## 5. References

- Baluev, R. V. (2008), Assessing the statistical significance of periodogram peaks, *MNRAS* 385, 1279–1285.
- Bendat, J. S. and Piersol, A. G. (2010), *Random Data: Analysis and Measurement Procedures*, 4th ed., Wiley.
- Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (2008), *Time Series Analysis: Forecasting and Control*, 4th ed., Wiley.
- Brockwell, P. J. and Davis, R. A. (1991), *Time Series: Theory and Methods*, 2nd ed., Springer.
- Dickey, D. A. and Fuller, W. A. (1979), Distribution of the estimators for autoregressive time series with a unit root, *JASA* 74, 427–431.
- Durbin, J. (1960), The fitting of time-series models, *Rev. Int. Statist. Inst.* 28, 233–244.
- Elgar, S. and Guza, R. T. (1988), Statistics of bicoherence, *IEEE Trans. Acoust. Speech Signal Process.* 36, 1667–1668.
- Engle, R. F. and Granger, C. W. J. (1987), Co-integration and error correction: representation, estimation and testing, *Econometrica* 55, 251–276.
- Foreman, M. G. G. (1977), *Manual for Tidal Heights Analysis and Prediction*, Pacific Marine Science Report 77-10, Institute of Ocean Sciences, Patricia Bay.
- Gilman, D. L., Fuglister, F. J. and Mitchell, J. M. (1963), On the power spectrum of "red noise", *J. Atmos. Sci.* 20, 182–184.
- Hasselmann, K. (1976), Stochastic climate models, part I: theory, *Tellus* 28, 473–485.
- Hodrick, R. J. and Prescott, E. C. (1997), Postwar U.S. business cycles: an empirical investigation, *J. Money, Credit and Banking* 29, 1–16.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. and Shin, Y. (1992), Testing the null hypothesis of stationarity against the alternative of a unit root, *J. Econometrics* 54, 159–178.
- Lomb, N. R. (1976), Least-squares frequency analysis of unequally spaced data, *Astrophys. Space Sci.* 39, 447–462.
- Ljung, G. M. and Box, G. E. P. (1978), On a measure of lack of fit in time series models, *Biometrika* 65, 297–303.
- Lütkepohl, H. (2005), *New Introduction to Multiple Time Series Analysis*, Springer.
- MacKinnon, J. G. (1994), Approximate asymptotic distribution functions for unit-root and cointegration tests, *J. Bus. Econ. Stat.* 12, 167–176.
- MacKinnon, J. G. (2010), Critical values for cointegration tests, Queen's Economics Department Working Paper 1227.
- Mann, M. E. and Lees, J. M. (1996), Robust estimation of background noise and signal detection in climatic time series, *Climatic Change* 33, 409–445.
- Pawlowicz, R., Beardsley, B. and Lentz, S. (2002), Classical tidal harmonic analysis including error estimates in MATLAB using T_TIDE, *Comput. Geosci.* 28, 929–937.
- Percival, D. B. and Walden, A. T. (1993), *Spectral Analysis for Physical Applications*, Cambridge University Press.
- Ravn, M. O. and Uhlig, H. (2002), On adjusting the Hodrick–Prescott filter for the frequency of observations, *Rev. Econ. Stat.* 84, 371–376.
- Said, S. E. and Dickey, D. A. (1984), Testing for unit roots in autoregressive-moving average models of unknown order, *Biometrika* 71, 599–607.
- Scargle, J. D. (1982), Studies in astronomical time series analysis II: statistical aspects of spectral analysis of unevenly spaced data, *ApJ* 263, 835–853.
- Schwert, G. W. (1989), Tests for unit roots: a Monte Carlo investigation, *J. Bus. Econ. Stat.* 7, 147–159.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. and Farmer, J. D. (1992), Testing for nonlinearity in time series: the method of surrogate data, *Physica D* 58, 77–94.
- Thompson, R. O. R. Y. (1979), Coherence significance levels, *J. Atmos. Sci.* 36, 2020–2021.
- Torrence, C. and Compo, G. P. (1998), A practical guide to wavelet analysis, *Bull. Amer. Meteor. Soc.* 79, 61–78.
- VanderPlas, J. T. (2018), Understanding the Lomb–Scargle periodogram, *ApJS* 236, 16.
- Zechmeister, M. and Kürster, M. (2009), The generalised Lomb–Scargle periodogram, *A&A* 496, 577–584.

## 6. Open Questions

1. The exact Gaussian likelihood of ARMA models by the Kalman filter, with physis's linear dynamics (CLAUDE.md §8, open decision 2).
2. MacKinnon's (1996) finite-sample p-values (response surfaces in T) in place of the asymptotic 1994 ones.
3. The data-dependent bandwidth of Hobijn, Franses and Ooms (1998) for KPSS.
4. Coloured errors of harmonic fits for uneven or gapped sampling (from the residual's Lomb–Scargle periodogram), and astronomical arguments with nodal corrections in a tides domain above.
5. Levels over all frequencies at once (the multiple testing of a spectrum's peaks), beyond Baluev's for Lomb–Scargle.
