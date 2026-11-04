"""The reference values of the Hodrick–Prescott check (check_var.cpp): the trend of ln income in
Lütkepohl's E1 (fixtures/lutkepohl-e1.dat, column 2) at λ = 1600, solved exactly.

The doubles the check reads (the natural logarithms of the incomes) are taken as exact rationals,
and (I + λKᵀK)τ = y is solved by Gaussian elimination in rational arithmetic, so that the values
printed carry no error but the final rounding to double. statsmodels' hpfilter, which solves the
same system in floating point, differs from them by up to 1.5e-12 (its condition number is about
1 + 16λ ≈ 2.6e4).

    python make_hp_fixtures.py fixtures/lutkepohl-e1.dat
"""
import math
import sys
from fractions import Fraction


def main(path: str) -> None:
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            v = list(map(float, line.split()))
        except ValueError:
            continue
        if len(v) == 3:
            rows.append(v)
    y = [Fraction(math.log(r[1])) for r in rows]
    n, lam = len(y), Fraction(1600)
    A = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        A[i][i] += 1
    d = (1, -2, 1)
    for t in range(n - 2):
        for a in range(3):
            for b in range(3):
                A[t + a][t + b] += lam * d[a] * d[b]
    rhs = y[:]
    for k in range(n):  # the matrix is pentadiagonal: two rows below each pivot
        for i in range(k + 1, min(n, k + 3)):
            if A[i][k] == 0:
                continue
            f = A[i][k] / A[k][k]
            for j in range(k, min(n, k + 3)):
                A[i][j] -= f * A[k][j]
            rhs[i] -= f * rhs[k]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (rhs[i] - sum(A[i][j] * x[j] for j in range(i + 1, min(n, i + 3)))) / A[i][i]
    print(f"n = {n}; the exact trend at t = 0, 10, 45, 91:")
    print(", ".join(f"{float(x[t]):.17g}" for t in (0, 10, 45, 91)))


if __name__ == "__main__":
    main(sys.argv[1])
