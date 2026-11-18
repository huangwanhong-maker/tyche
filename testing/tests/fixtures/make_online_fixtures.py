"""Reference values for check_online.cpp (decision 0027).

- zeta(1.6), the normalising sum of gamma_j = j^-1.6 (Ramdas et al. 2018's SAFFRON sequence), by mpmath;
- the normalising sum of Javanmard and Montanari's gamma_j = log(max(j, 2)) / (j exp(sqrt(log j))). Its series converges so
  slowly (the tail beyond 10^6 is still 5.9) that mpmath's nsum misjudges it (12.484). It is computed here as the partial sum
  to N in float64, smallest terms first, plus the exact integral of the tail, 2 e^-a (a^3 + 3a^2 + 6a + 6) with a = sqrt(log N),
  plus f(N)/2 (Euler-Maclaurin), at N = 10^6 and 10^7, which agree to 12 digits. The constant Javanmard and Montanari print,
  0.07720838 (Ann. Statist. 46 (2018)), makes the sum 0.976: short of 1, and so still a valid spending sequence;
- the mixture calibrator (1 - p + p log p)/(p log^2 p) at a few p, and the integral of kappa p^(kappa - 1) over kappa there.

Run:  python make_online_fixtures.py > online.txt
"""

import mpmath as mp
import numpy as np

mp.mp.dps = 30


def lord_sum(N):
    j = np.arange(1, N, dtype=np.float64)
    f = np.log(np.maximum(j, 2)) / (j * np.exp(np.sqrt(np.log(j))))
    s = np.sort(f).sum()
    a = np.sqrt(np.log(N))
    tail = 2 * np.exp(-a) * (a**3 + 3 * a**2 + 6 * a + 6)
    fN = np.log(N) / (N * np.exp(np.sqrt(np.log(N))))
    return s + tail + fN / 2


print("# name value  (mpmath %s, numpy %s)" % (mp.__version__, np.__version__))
print("zeta_1.6", mp.nstr(mp.zeta(1.6), 20))
s6, s7 = lord_sum(10**6), lord_sum(10**7)
assert abs(s6 - s7) < 1e-9, (s6, s7)
print("lord_sum %.15g" % s7)
print("lord_constant %.15g" % (1 / s7))
for p in ["0.3", "0.01", "0.999"]:
    pp = mp.mpf(p)
    closed = (1 - pp + pp * mp.log(pp)) / (pp * mp.log(pp) ** 2)
    integral = mp.quad(lambda k: k * pp ** (k - 1), [0, 1])
    print("mixture_%s" % p, mp.nstr(closed, 20), mp.nstr(integral, 20))
