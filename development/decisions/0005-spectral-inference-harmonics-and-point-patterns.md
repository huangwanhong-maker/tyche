# 0005. Spectral Inference, Harmonic Analysis, Lomb–Scargle, and Point Patterns of Marks

- **Status:** Accepted, provisionally (univariate series; errors of harmonic fits coloured only for even spacing; planar point patterns in a rectangle for Clark and Evans)
- **Date:** 2026-10-31
- **Decided by:** the infrastructure's street analysis ("street oceanography": activity on streets as a forced system with daily, weekly, lunar and annual rhythms, nonlinear intermodulation and red-noise nulls; and the gathering of kinds of holdings along a street) needs inference on spectra, the harmonic analysis of tides for any rhythm, and the statistics of marked points. Claude wrote the methods and this record at the user's request; the estimators they stand on are calx's (its decision 0013).

## 1. Context

calx's `signal` gives spectral estimates; what they mean needs statistics. Is a peak a line or the red background's chance excursion? How wide is its interval? Do two series cohere beyond chance? Is a bicoherence real? Rhythms of known frequency (a day, a week, a lunar month, a year) are measured best as tides are, by least squares at those frequencies, with errors from the residual's spectrum near each one. Counts at irregular times need a periodogram that does not ask for even sampling. Kinds of holdings along a street gather or keep apart, and the question is whether they do so beyond what their common terrain explains.

`time_series` listed spectral analysis as planned; `spatial` listed point patterns as planned.

## 2. Decision

### 2.1 `time_series/spectral.hpp`

- **Intervals and levels.**
  - `spectral_interval(estimate, dof, confidence)`: Percival and Walden eq. 258.
  - `welch_dof(segments, L, overlap)`: their eq. 292b, for calx's Hann segments.
  - `coherence_level(K, α)`: 1 − α^{1/(K−1)}, exact for independent Gaussian segments.
  - `bicoherence_level(S, α)`: −ln α / S, asymptotic.
- **The red-noise null.**
  - `RedNoise{rho, variance}` with its one-sided density, which integrates to σ²; `red_noise(x)` from the lag-one autocorrelation and the variance (divisor n).
  - `red_noise_level(r, f, fs, dof, confidence)`: the background times χ²_ν(confidence)/ν.
- **Surrogates.** `phase_surrogate(x, rng)` keeps the periodogram exactly; `red_noise_surrogate(r, n, rng)` draws an AR(1) path from its stationary law.
- **Aliasing.** `alias(f, fs)`.

### 2.2 `time_series/harmonic.hpp`

- **The fit.** `harmonic_fit(t, y, constituents, options)` → `HarmonicFit`: least squares on calx's pivoted QR at given frequencies, for any spacing and gaps, with an optional trend.
- **The phase.** φ = atan2(b, a) ∈ [0, 2π) for A cos(2πf(t − t₀) − φ): the phase lag of tidal analysis, greatest φ/(2πf) after t₀.
- **Errors.** White errors are σ²(XᵀX)⁻¹. With even spacing they are coloured, T_TIDE's: the residual's Hann-tapered periodogram averaged over a band about each frequency, the bins within 2/T of the fitted lines left out, and σ²_eff = S_r fs/2 scaling that constituent's block of (XᵀX)⁻¹. Amplitude and phase errors are linearised; snr = (A/A_se)².
- **Resolution.** Rayleigh's criterion flags unresolved pairs. `resolvable(f1, f2, T, R)`; `harmonic_predict(fit, t, t0)`.

### 2.3 `time_series/lomb_scargle.hpp`

- `lomb_scargle(t, y, frequency, floating_mean)`: astropy's standard normalisation, with a fixed or floating mean (Zechmeister and Kürster).
- `false_alarm_baluev(power, t, y, fmax)`: Baluev's (2008) false-alarm probability, as astropy's.

### 2.4 `spatial/point_patterns.hpp`

- `colocation(x, y, mark, marks, stratum, radius, permutations, seed)` → `Colocation`: Leslie and Kronenfeld's co-location quotients, by nearest neighbours (ties shared) or within a radius. A permutation null of the marks, within strata when given, with one-sided p-values.
- `clark_evans(x, y, x0, x1, y0, y1)` → `ClarkEvans`: R = r̄ / E r̄ with Donnelly's correction of the expectation and variance for a rectangle; z and a two-sided p.

### 2.5 Departures from the design handed in

The design of these methods (the lead's contract of 2026-10-30) was followed in names and signatures, with these changes:

- **Two members added**, both defaulted and additive:
  - `HarmonicOptions::band`, the half-width of the residual band; 0 means max(f_k/10, 10/T).
  - `HarmonicFit::coloured`, saying whether the errors were coloured, so that uneven spacing's fallback to white is visible.
- **The coloured errors** scale (XᵀX)⁻¹ rather than use 2σ²_eff/n. They equal it for well-separated constituents on a long record, and they stay right for close ones, whose coefficients are correlated. The notch excluded about each fitted line is ±2/T, not ±1/T, since the Hann taper widens it.
- **Clark and Evans's ratio** divides by Donnelly's corrected expectation. A square lattice has R = 2 (uncorrected), not 2.15; 2.1491 is the hexagonal lattice's, the greatest.
- **Elgar and Guza's (1988) bicoherence levels** are cited from IEEE Trans. Acoust. Speech Signal Process. 36, 1667, and Thompson's (1979) coherence levels from J. Atmos. Sci. 36, 2020.
- **The degrees of freedom of a multitaper estimate** (calx) are K, not 2K, at zero and at the Nyquist frequency.

## 3. Alternatives Considered

- **Spectral inference in calx.** Rejected: intervals, levels and nulls are statistics (tyche's scope); calx keeps the estimators, which name no statistical object.
- **Harmonic analysis with astronomical arguments and nodal corrections** (T_TIDE's full machinery). Rejected here: they name the Moon and the Sun, and belong to a domain (tides); this module fits any given frequencies.
- **Coloured errors for uneven spacing from the Lomb–Scargle periodogram of the residual.** Deferred: the fallback to white errors is stated and flagged.
- **Monte Carlo errors of amplitude and phase** (T_TIDE's alternative to linearisation). Deferred: the linearised errors cover 95 % in the checks at the signal-to-noise of interest.
- **Ripley's K and the pair-correlation function.** Planned for point patterns; the co-location quotient answers the first question, which kinds gather about which.

## 4. Consequences

- `check_tyche_spectral` (33):
  - intervals and red-noise levels against scipy's χ² quantiles, and the coverage of intervals;
  - Welch's degrees of freedom in closed form and by simulation;
  - the coherence and bicoherence levels' rates by simulation;
  - the red-noise density's integral, the mean AR(1) periodogram, and red_noise and its surrogate;
  - phase surrogates to rounding; aliases of M2 (14.765 and 62.11 days);
  - Lomb–Scargle and Baluev's false-alarm probability against astropy 8.0.2.
- `check_tyche_harmonic` (25):
  - exact recovery, even and gapped; the phase convention; Rayleigh flags and refusals;
  - 95 % coverage of white, coloured and gapped errors;
  - coloured errors exceeding white ones as √(S(f)/(2σ²/fs)) under AR(1) residuals, and covering where white ones do not.
- `check_tyche_point_patterns` (24):
  - the co-location quotient by hand, with a tie and a radius; its invariants; the null's mean and the p-value's size under random labelling;
  - attraction found; strata keeping a terrain's composition;
  - Clark and Evans on square and hexagonal lattices, Poisson points (size α) and a Thomas process.
- The fixtures are made by `time_series/tests/make_spectral_fixtures.py` from scipy 1.18.1 and astropy 8.0.2.
- `time_series` now depends on calx `signal`, `random` and `special_functions`; `spatial` on calx `random`.
- The conventions (one-sided densities, the cross spectrum's phase, the harmonic phase, the red-noise background) are in the glossary, §4.
