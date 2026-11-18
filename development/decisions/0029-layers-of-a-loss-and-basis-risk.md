# 0029. Layers of a Loss and Basis Risk

- **Status:** Accepted, provisionally (2026-11-18).
- **Date:** 2026-11-18
- **Decided by:** Claude, at the user's request for disaster risk financing in the infrastructure's new domain of risk management (its ADR 0129).

## 1. Context

A government or a city finances a disaster's loss in layers: what it retains, what a reserve or a contingent credit pays, what insurance or a catastrophe bond takes above, each attaching at a loss and paying up to a limit. How much of a loss each layer carries, a year on average and in its spread, is a statement about the law of the loss, whatever instrument the layer is: a statistical object, and so tyche's. So is how far a payout on an index (a depth, a wind speed, a modelled loss) is from the loss it is meant to meet, its basis risk. Which instruments take which layers, and at what price, is the risk management domain's. Decision 0012 gave the annual loss and exceedance curves on which these stand.

## 2. Decision

`descriptive/layers.hpp`:

- `layer_loss` and `Layer`; `layer_statistics` over a record read per year (each loss a year's total) or per event (a Poisson catalogue, each event under the layer on its own, the annual layer loss compound Poisson with variance Σ x² / years);
- `split`, a tower from 0 whose parts sum to every loss;
- `layer_from_curve`, ∫ P(L > x) dx over the layer on a published curve, exactly under `curve_area`'s conventions, so that the layer from 0 without limit is the curve's area;
- `basis_risk` over paired, weighted samples of payout and covered loss (absolute and squared error, shortfall and excess, losses met by no payout and payouts met by no loss, correlation), and `fit_payout`, the linear payout's trigger and exhaustion among quantiles of the index, by a search on a grid said to be one.

## 3. Consequences

`check_tyche_layers` (16 checks) holds the layers to an exponential law's closed forms, the compound Poisson variance to simulated years, the curve's layers to its area exactly and to the closed form on a refined grid, and the fit to a payout recovered exactly. Prices, loadings and the costs of instruments are not here. A record of events without their years (as a catastrophe model's event-loss table often is) supports occurrence covers only; aggregate covers need the years.
