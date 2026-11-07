# 0013. Spatial Autocorrelation

- **Status:** Accepted, provisionally (2026-11-07).
- **Decided by:** Claude, for the toolbox of quantitative economics on the Urban Science Workbench's map (the infrastructure's ADR 0080). A regression on places reports standard errors that assume the places are independent, and neighbouring places seldom are. The test belongs here, since it names no object in nature.

## 1. Decision

`spatial/autocorrelation.hpp` says whether values at neighbouring places resemble each other:

- **Weights.** `SpatialWeights` holds sparse rows. `weights_from_rows` builds them from each place's neighbours, row-standardised or not. `k_nearest_weights` gives each point its k nearest in the plane, or in space, where unit vectors find neighbours on the sphere exactly (chord length is monotone in the angle), across the antimeridian too.
- **Moran's I** (Moran 1950; Cliff and Ord 1981):
  - its expectation;
  - its variances for normal values and under random assignment;
  - their z-scores and two-sided p-values;
  - a permutation test from a seeded calx stream, one-sided in the direction observed.
- **Local Moran's I** (Anselin 1995): each place's I and its quadrant, with a p-value by conditional permutation (the place's own value held), uncorrected for testing every place.

## 2. Verification

`check_tyche_spatial_autocorrelation` (44 checks):

- **Against PySAL's esda 2.10 and libpysal 4.15** (BSD 3-Clause), on a rook lattice with binary and row-standardised weights and on forty points with their four nearest neighbours (an unsymmetric W): I, its expectation, both variances, z and p, and each place's local I and quadrant. The fixtures are made by `tests/fixtures/make_autocorrelation_fixtures.py`.
- **Nearest neighbours** as libpysal finds them; on the sphere, the nearest across the antimeridian.
- **Closed forms:** a checkerboard gives I = −1 exactly, binary or row-standardised. Two halves give I = 1 − 2·6/82, by counting the edges across. The local I sum to (n − 1) I.
- **Invariance:** under shifting and scaling.
- **The permutation test's size** under random assignment (300 draws), and its least p on a smooth surface.
- **A hot spot:** its centre is high among high, with a small p-value.
- **Refusals:** values that do not vary are refused, and so is a place given as its own neighbour.

## 3. Consequences

- The toolbox's `moran` tool, and its test of a regression's residuals wherever the rows are distinct places, stand on this.
- Not yet: Geary's c, Getis–Ord G*, the correction of local tests for their number (`testing` holds the methods), and spatial regression models (lag and error) estimated as such.
