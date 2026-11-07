"""Reference values for tyche/spatial/autocorrelation.hpp, from PySAL's esda and libpysal (BSD 3-Clause).

    python -m venv env && env/bin/pip install esda libpysal
    env/bin/python make_autocorrelation_fixtures.py

Writes autocorrelation_fixtures.hpp beside this file. Two sets of values on two kinds of weights, so that both a
symmetric and an unsymmetric W are held:
- a rook lattice of 5 × 6 cells, its weights binary or row-standardised, with values that are a fixed sequence;
- forty points of a seeded generator with their 4 nearest neighbours, row-standardised, unsymmetric.

Moran's statistics and each place's local I and quadrant are deterministic. esda's permutation p-values are not compared:
its stream is not ours, and the checks hold our test's size instead.
"""

from __future__ import annotations

from pathlib import Path

import esda
import libpysal
import numpy as np

OUT = Path(__file__).with_name("autocorrelation_fixtures.hpp")


def lit(a) -> str:
    return "{" + ", ".join(f"{float(x):.17g}" for x in a) + "}"


def ints(a) -> str:
    return "{" + ", ".join(str(int(x)) for x in a) + "}"


def block(name: str, xs, W, transformation: str, coords=None) -> str:
    W.transform = transformation
    m = esda.Moran(xs, W, transformation=transformation, permutations=0)
    lo = esda.Moran_Local(xs, W, transformation=transformation, permutations=0) if transformation == "r" else None
    rows = []
    for i, ident in enumerate(W.id_order):
        rows.append("{" + ", ".join(f"{{{int(j)}, {float(wt):.17g}}}" for j, wt in zip(W.neighbors[ident], W.weights[ident])) + "}")
    text = f"inline const Case {name}{{\n  {lit(xs)},\n  {{{', '.join(rows)}}},\n"
    text += f"  {m.I:.17g}, {m.EI:.17g}, {m.VI_norm:.17g}, {m.VI_rand:.17g}, {m.z_norm:.17g}, {m.z_rand:.17g}, {m.p_norm:.17g}, {m.p_rand:.17g},\n"
    text += f"  {lit(lo.Is) if lo is not None else '{}'},\n  {ints(lo.q) if lo is not None else '{}'},\n"
    text += f"  {lit(coords[:, 0]) if coords is not None else '{}'},\n  {lit(coords[:, 1]) if coords is not None else '{}'},\n}};\n"
    return text


def main() -> None:
    lattice = libpysal.weights.lat2W(5, 6, rook=True)
    values = np.array([3.1, 2.7, 4.4, 1.0, 0.2, 5.5, 2.2, 3.9, 4.8, 1.7, 0.9, 6.1, 1.4, 3.3, 5.2, 2.5, 1.1, 7.0, 0.7, 2.9, 4.1, 3.0, 2.0, 6.6, 1.9, 2.3, 3.6, 4.0, 2.8, 5.0])
    rng = np.random.default_rng(20261107)
    points = rng.uniform(0, 100, size=(40, 2))
    knn = libpysal.weights.KNN.from_array(points, k=4)
    scattered = points[:, 0] * 0.04 + rng.normal(0, 1, 40)

    parts = [
        "// Made by make_autocorrelation_fixtures.py from esda " + esda.__version__ + " and libpysal " + libpysal.__version__ + "; do not edit.\n#pragma once\n\n#include <cstddef>\n#include <utility>\n#include <vector>\n\nnamespace fixtures {\n\n",
        "struct Case {\n  std::vector<double> values;\n  std::vector<std::vector<std::pair<std::size_t, double>>> rows;\n  double I, expected, variance_normal, variance_randomisation, z_normal, z_randomisation, p_normal, p_randomisation;\n  std::vector<double> local;\n  std::vector<int> quadrant;\n  std::vector<double> x, y;\n};\n\n",
        block("lattice_binary", values, lattice, "b"),
        block("lattice_standardised", values, lattice, "r"),
        block("knn_standardised", scattered, knn, "r", points),
        "\n}  // namespace fixtures\n",
    ]
    OUT.write_text("".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
