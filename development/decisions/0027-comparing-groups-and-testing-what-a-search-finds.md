# 0027. Comparing Groups, and Testing What a Search Finds

- **Status:** Accepted, provisionally (2026-11-18)
- **Date:** 2026-11-18
- **Decided by:** Claude, for the infrastructure's search for needs (its ADR 0125), where statements of need found by evolution are tested on data kept aside; and at the survey made for it (`research/literature/reports/Evolutionary exploration and validation of needs.md`, §3), which found that the first design's correction for multiplicity was offline where the search is online, and that its held-out part leaked into the search.

## 1. Context

A search that tries thousands of hypotheses on one sample finds large effects by chance. The infrastructure's needs domain keeps three parts of its data apart: one the search sees, one it may consult, one it never sees, used once for each hypothesis admitted. That asks of tyche what it did not hold:
- comparisons of two groups, or of a group with a level, with weights and fractional counts;
- a bootstrap of clusters for stability;
- decisions about hypotheses that arrive one at a time and are never revised;
- statistics that combine under any dependence;
- a held-out part that can be consulted many times without being fitted.

## 2. Decision

`testing` gains four headers.

1. **`two_sample.hpp`.** The weighted mean with Kish's effective size (Kish 1965, §11.7). Welch's test with Satterthwaite's degrees of freedom. A mean against a level. A share against a level by the score test with Wilson's interval. Two shares by Newcombe's hybrid score interval (1998, method 10) with the pooled z test. Counts may be fractional. Held to SciPy and statsmodels (`fixtures/two_sample.txt`).
2. **`bootstrap.hpp`.** Resamples of units or of clusters, by multiplicities given to the caller's statistic, with the percentile interval.
3. **`online.hpp`.** Online control of the false discovery rate:
   - LOND (Javanmard and Montanari 2018);
   - LORD++ (Ramdas, Yang, Wainwright and Jordan 2017);
   - SAFFRON (Ramdas, Zrnic, Wainwright and Jordan 2018);
   - α-investing (Foster and Stine 2008), which controls the marginal FDR;
   - e-LOND (Xu and Ramdas 2024), which controls the FDR under arbitrary dependence.

   One object holds a rule's state; a sequence of values replays it exactly. The spending sequences are normalised here, not copied. The sum of Javanmard and Montanari's sequence is 12.6451078729, by partial sums to 10⁶ and 10⁷ and the exact integral of the tail. Their printed constant, 0.07720838, makes the sum 0.976: valid, but short of 1. mpmath's `nsum` misjudges this series, at 12.484, so the fixture does not rely on it.
4. **`evalues.hpp`** and **`holdout.hpp`.**
   - E-values: the calibrators κp^{κ−1} and their mixture (Vovk and Wang 2021; Shafer et al. 2011); the mean and the product; e-BH (Wang and Ramdas 2022); e-values read back as p-values.
   - The reusable holdout: Thresholdout (Dwork et al. 2015), with its noisy threshold, its budget of answers from the holdout, and a state that can be saved and resumed. Laplace draws come from the caller's stream.

## 3. Alternatives Considered

- **Benjamini–Hochberg re-run as hypotheses arrive.** It can withdraw a label already given, and needs independence or positive dependence; rejected for a search that admits as it goes.
- **Online procedures in peira.** They are inference about many hypotheses in a given model, not a mechanism of observation; tyche's `testing` already holds multiple comparisons.
- **The Ladder** (Blum and Hardt 2015) beside Thresholdout. It answers a leaderboard's question (is this better than the best so far?), which no caller asks yet.

## 4. Consequences

- `check_two_sample` (SciPy, statsmodels, refusals) and `check_adaptive` (52 checks):
  - every rule's levels worked by hand on short sequences;
  - the FDR at α = 0.1 by 2000 replications of 200 hypotheses: LOND 0.005, LORD++ 0.014, SAFFRON 0.084, and e-LOND 0 on equicorrelated tests. α-investing's marginal FDR is 0.097. SAFFRON and LORD++ find more than LOND;
  - the calibrators against mpmath;
  - e-BH by hand and as BH on 1/e;
  - Thresholdout without noise, its budget, and Dwork et al.'s demonstration: features with no signal, chosen by an analyst who consults the holdout. The plain holdout rates the classifier at 0.629, Thresholdout at 0.516, and the truth is 1/2.
- tyche 1673 checks.
- The needs domain admits statements through Thresholdout and confirms them online, e-LOND by default.
- Open: betting e-processes and confidence sequences, for revoking on new waves of data however often they are looked at; asynchronous online testing (Zrnic, Ramdas and Jordan 2021).
