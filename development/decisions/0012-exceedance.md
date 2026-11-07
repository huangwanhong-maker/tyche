# 0012. Annual Losses and Exceedance Curves

- **Status:** Accepted, provisionally (2026-11-07).
- **Decided by:** Claude, for the Urban Science Workbench's flood risk (the infrastructure's ADR 0078); the method as stated by Woo (1999), Grossi and Kunreuther (2005) and CCDR-tools, none of whose code (GPL-3.0) is used.

## 1. Decision

`descriptive/exceedance.hpp` says what a record of losses says of the year:
- `average_annual_loss`: the sum of a catalogue's event losses over its years;
- `occurrence_curve`: the loss at each return period, the k-th largest of the years' maxima at T = years / k, linear in log T between them;
- `curve_area`: the area under loss against annual exceedance probability, by trapezoids, with what is assumed below the first point and beyond the last reported apart, since that is where the assumption is.

It stands in `descriptive` because it describes a sample and assumes no law. Fitting a law to the tail belongs to a later `extremes` module, when one is needed.

## 2. Verification

`check_tyche_exceedance`: the rank points exactly; L(T) = ln T has a mean of 1, which the trapezoids reach at second order; a catalogue from an exponential law within 2 %; refusals. On real data, in the data library's checks: the Freetown Flood Risk Assessment's published occurrence curve is its events in order, to 1 part in 10⁷, at every rank a return period falls on.
