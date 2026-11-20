# 0033. Latent Positions of a Network, with the Group Under Which They Are Identified

- **Status:** Accepted, provisionally (2026-11-21).
- **Date:** 2026-11-21
- **Decided by:** Claude, at the infrastructure's Stage 8 of evolutionary allocation (the user: "please continue all remain stage if unfinish"), after its survey's Part VII (*Evolutionary heterogeneous resource allocation*, §6.1, item 2): latent space network models with their identifiability group declared as data, and every reported quantity checked for invariance under it; the economic complexity index as a corollary of the spectral embedding, not a separate index.

## 1. Context

Embedding a network's nodes in a space is common in economics (relatedness, the product space), sociology and neuroscience, and its coordinates are often read as if they meant something. They are identified only up to a group: rotations and reflections for a random dot product graph, the indefinite orthogonal group for a network that joins unlike nodes more than like ones, under which even distances are not identified. tyche's `networks` described networks; it held no latent positions.

## 2. Decision

`networks/latent.hpp`:

- `Embedding{p, q, x, eigenvalues}` with `group()` and `probability(i, j)`; `spectral_embedding(A, d, generalised)`, the adjacency spectral embedding (Sussman et al. 2012), the signature read from the signs of the kept eigenvalues (Rubin-Delanchy et al. 2022);
- `scree`, `profile_dimension` (Zhu and Ghodsi 2006);
- `act`, `random_orthogonal`, `random_element(p, q)` (orthogonal blocks and a hyperbolic rotation), `identified(embedding, statistic)`: a statistic tested against random elements of the declared group before it is reported;
- `procrustes(X, Y)`: the orthogonal alignment, the polar factor of XᵀY (Schönemann 1966);
- `bipartite_ordering(M)`: rows ordered by the second eigenvector of the row-to-row walk, the economic complexity index as Mealy, Farmer and Teytelboym (2019) read it.

## 3. Alternatives Considered

- **Hoff's latent distance model, fitted by likelihood.** Planned: identified up to rigid motions, it needs an optimiser over all positions; the spectral embedding is consistent, closed in form, and shows the discipline first.
- **Reporting coordinates.** Refused by construction: what is reported is what the group leaves unchanged, tested.

## 4. Consequences

`check_tyche_latent_positions` (13 checks): positions of a random dot product graph recovered after alignment, the error falling as 1/√n; probabilities and distances identified under O(2); the profile likelihood choosing the true dimension where the leading eigenvalues are comparable and stopping at one where one dominates (its caveat, said in the header); a disassortative two-block graph of signature (1, 1), its probabilities identified and its distances not, its estimated probabilities the blocks'; a known rotation recovered exactly; a nested incidence ordered by its diversity.
