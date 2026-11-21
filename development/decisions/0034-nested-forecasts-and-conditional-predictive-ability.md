# 0034. Nested Forecasts and Conditional Predictive Ability

- **Status:** Accepted, provisionally (2026-11-21).
- **Date:** 2026-11-21
- **Decided by:** Claude, at the infrastructure's word to continue what evolutionary allocation left for later (its ADR 0141, the design's *What Is Not Built*: "Clark and West's and Giacomini and White's tests of forecasts"), planned in [decision 0031](0031-forecasts-compared.md).

## 1. Context

Decision 0031 gave the test of equal accuracy, and said what it does not answer. A learned law is held to a simpler law it nests (an autoregression to the mean), and between nested models the test of equal accuracy is undersized: under the null the larger model's forecasts carry the noise of estimating terms that are zero. And whether one forecast is better on average is not the question a person choosing between them asks when the advantage moves with circumstances: that is whether the loss differential can be predicted from what is known when the forecasts are made.

## 2. Decision

`time_series/forecasts.hpp`:

- `clark_west(truth, small, large, h, lags)` → `NestedAccuracy{n, lags, mse_small, mse_large, mean_adjusted, statistic, p}`: the differential adjusted by the squared gap of the two forecasts, f_t = 2 e_{1,t} (ŷ_{2,t} − ŷ_{1,t}), its mean over Newey and West's long-run standard error with h − 1 lags, referred to the standard normal one-sided (Clark and West 2007);
- `giacomini_white(truth, a, b, h, τ)` → `ConditionalAbility{n, q, lags, mean_difference, mean_moment, statistic, p}`: the moments h_t d_{t+τ}, the statistic n Z̄ᵀ Ω̂⁻¹ Z̄ with Ω̂ uncentred one step ahead and Bartlett-weighted with τ − 1 lags beyond, referred to χ² with q degrees of freedom; test functions (1, d_{t−τ}) by default (Giacomini and White 2006). A singular Ω̂ is refused with its likely reason.

## 3. Alternatives Considered

- **Clark and McCracken's bootstrap for nested models.** Their critical values are not standard normal; Clark and West's adjustment gives an approximately normal test, which is what their paper offers for practice, and its size is checked.
- **A centred Ω̂ for Giacomini and White.** Under the null E[Z] = 0, and the uncentred form is theirs.

## 4. Consequences

`check_tyche_forecasts` gains 12 checks (25):

- Clark and West's adjusted differential equal to 2 e₁ (ŷ₂ − ŷ₁), its statistic and p-value exactly;
- the historical mean nested in a fitted autoregression, 600 runs. Under the null, Clark and West's test rejects 6.7% at 10%, where the one-sided test of equal accuracy rejects 1.7%. Under an autoregression of φ = 0.25 the first finds the larger model in 86% of runs and the second in 43%;
- Giacomini and White's statistic with h_t = 1 equal to n d̄² / ((1/n) Σ d²), and with (1, d_{t−1}) to the 2 × 2 form inverted by hand;
- its size 3.1% at 5% on 1,000 runs of equal, unpredictable losses;
- a differential of mean zero that persists (the better forecast changing rarely) found in 97.9% of runs. The unconditional test rejects 44%, more than it should: one step ahead it takes no autocovariances of a differential that is serially correlated.

tyche 1789.
