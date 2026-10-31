"""Reference values for tyche's spectral checks (check_spectral.cpp), from scipy and astropy.

Writes tests/fixtures/spectral-reference.txt: one line per record, a key and then its numbers,

    key v0 v1 v2 …

(keys may repeat; '#' starts a comment).

- chi2: ν, confidence, χ²_ν(α/2), χ²_ν(1 − α/2), χ²_ν(confidence), α = 1 − confidence
  (scipy.stats.chi2.ppf), for the spectral intervals [ν S/χ²_ν(1 − α/2), ν S/χ²_ν(α/2)] and the
  red-noise levels S χ²_ν(confidence)/ν.
- ls/t, ls/y, ls/frequency: an unevenly sampled series with a gap (numpy's PCG64, seeded), and a grid of
  frequencies.
- ls/power/floating, ls/power/fixed: astropy.timeseries.LombScargle(t, y, fit_mean=…, center_data=True,
  normalization='standard').power(frequency, method='cython') — the exact sums, not the 'fast' method's
  approximation that astropy's 'auto' would choose for a regular grid.
- ls/fap: fmax, then pairs (power, false-alarm probability) of LombScargle.false_alarm_probability(power,
  method='baluev', maximum_frequency=3.0); fmax is the highest frequency the class then uses, 3.0 rounded
  to its automatic grid (autofrequency(…, return_freq_limits=True)), which tyche's function takes as given.

    .venv/Scripts/python.exe -I foundations/tyche/time_series/tests/make_spectral_fixtures.py
"""

import datetime
from pathlib import Path

import astropy
import numpy as np
import scipy
from astropy.timeseries import LombScargle
from scipy.stats import chi2

OUT = Path(__file__).resolve().parent / "fixtures" / "spectral-reference.txt"


def row(key, values):
    return key + " " + " ".join(f"{float(v):.17g}" for v in values)


def main() -> None:
    lines = [
        f"# tyche time_series: reference values from scipy {scipy.__version__} and astropy {astropy.__version__} (numpy {np.__version__}), {datetime.date.today().isoformat()}.",
        "# Written by make_spectral_fixtures.py; each line is a key and its numbers.",
    ]
    # ---- χ² quantiles ----
    for nu, conf in [(2, 0.95), (14, 0.95), (14, 0.90), (3.7, 0.99), (50.5, 0.95), (200, 0.68), (1.3, 0.8)]:
        a = 1 - conf
        lines.append(row("chi2", [nu, conf, chi2.ppf(a / 2, nu), chi2.ppf(1 - a / 2, nu), chi2.ppf(conf, nu)]))

    # ---- Lomb–Scargle ----
    rng = np.random.default_rng(20261030)
    t = np.sort(rng.uniform(0, 60, 220))
    t = t[(t < 21) | (t > 33)]  # a gap
    y = 4 + 2.5 * np.sin(2 * np.pi * 0.37 * t + 0.3) + 0.8 * np.cos(2 * np.pi * 1.3 * t) + 1.1 * rng.standard_normal(t.size)
    frequency = np.linspace(0.005, 3.0, 600)
    lines.append(row("ls/t", t))
    lines.append(row("ls/y", y))
    lines.append(row("ls/frequency", frequency))
    for fit_mean, name in ((True, "floating"), (False, "fixed")):
        ls = LombScargle(t, y, fit_mean=fit_mean, center_data=True, normalization="standard")
        power = ls.power(frequency, method="cython")
        lines.append(row(f"ls/power/{name}", power))
    ls = LombScargle(t, y, fit_mean=True, center_data=True, normalization="standard")
    # The class rounds the highest frequency to its automatic grid (fmin + df round((fmax − fmin)/df));
    # record the fmax it uses, which tyche takes as given.
    fmax = float(ls.autofrequency(maximum_frequency=3.0, return_freq_limits=True)[1])
    pairs = []
    for z in (0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6):
        pairs += [z, float(ls.false_alarm_probability(z, method="baluev", maximum_frequency=3.0))]
    lines.append(row("ls/fap", [fmax] + pairs))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines, {OUT.stat().st_size} bytes); t has {t.size} points")


if __name__ == "__main__":
    main()
