# descriptive: The Description of Samples

*Status: draft (2026-10-24): moments, sample quantiles, weighted statistics, covariance and correlation, summaries ([decision 0003](../development/decisions/0003-the-first-modules.md)).* Robust statistics (trimmed means, the median absolute deviation, M-estimates of location), empirical distribution functions, kernel density estimation and the numbers behind visual diagnostics are planned.

## 1. Concepts

A sample is described by its location (mean, median), spread (variance, standard deviation, interquartile range), shape (skewness, kurtosis) and quantiles, and two samples by their covariance and correlation. Each has competing definitions, and statistical packages disagree: the divisor of a variance (n or n − 1), the correction of skewness and kurtosis for bias, and nine ways of interpolating a quantile between order statistics (Hyndman and Fan 1996). tyche offers them by parameter, with the common default named (glossary §1).

Numerically, the danger is cancellation. The textbook variance Σx² − n x̄² subtracts two numbers that agree in all their digits when the mean is large against the spread, as in NIST's NumAcc4 (values 10⁷ + 0.1 … 0.3), where it returns garbage. Compensated sums, a corrected mean and the corrected two-pass formula of Chan, Golub and LeVeque keep every digit the data carry.

## 2. Content

`summary.hpp`:

| Function | Content |
|---|---|
| `sum`, `mean` | Neumaier's compensated sum; the mean corrected by a second pass |
| `variance(x, ddof)`, `standard_deviation` | the corrected two-pass formula (Chan, Golub and LeVeque 1983, eq. 1.7) |
| `central_moment(x, r)`, `skewness(x, bias_corrected)`, `kurtosis(x, excess, bias_corrected)` | g₁, G₁, g₂, G₂ (Joanes and Gill 1998) |
| `quantile(x, p, type)`, `quantiles`, `quantile_sorted`, `median`, `interquartile_range` | the nine types of Hyndman and Fan (1996), computed as R's quantile() |
| `weighted_mean`, `weighted_variance(x, w, Weights::frequency / reliability, ddof)`, `weighted_standard_deviation`, `weighted_quantile` | frequency and reliability weights; the lower inverse of the weighted distribution function |
| `covariance(x, y)`, `correlation(x, y)`, `covariance_matrix(data)`, `correlation_matrix(data)` | data with one row per observation (`calx::linalg::Rectangular`); a `calx::linalg::Matrix` out |
| `Summary`, `summarise(x)` | n, mean, variance, standard deviation, minimum, quartiles (type 7), median, maximum, skewness, excess kurtosis |

`rank.hpp` ([decision 0008](../development/decisions/0008-the-learning-module.md)):

| Function | Content |
|---|---|
| `ranks(x)` | mid-ranks (ties share the mean of their ranks; Kendall and Gibbons 1990, §3.1) |
| `spearman(x, y)` | Spearman's ρ, Pearson's correlation of the mid-ranks |
| `kendall_tau_b(x, y)` | Kendall's τ_b with ties, by Knight's O(n log n) count of discordant pairs (Knight 1966) |

`exceedance.hpp` ([decision 0012](../development/decisions/0012-exceedance.md)): the average annual loss of a catalogue, the occurrence curve from the years' maxima, the area under a published curve.

`layers.hpp` ([decision 0029](../development/decisions/0029-layers-of-a-loss-and-basis-risk.md)):

| Function | Content |
|---|---|
| `layer_loss(L, a, l)`, `Layer{attachment, limit}` | min(max(L − a, 0), l), the part of a loss between a and a + l |
| `layer_statistics(losses, years, layer, Record::per_year / per_event)` → `LayerStatistics{expected, sd, attachment_rate, exhaustion_rate, share}` | a year's expected layer loss and its spread, per year from the years' own, per event as compound Poisson (Klugman, Panjer and Willmot 2012, §9.3) |
| `split(losses, attachments)` | each loss over a tower from 0, the parts summing to it |
| `layer_from_curve(T, L, layer, first_loss, hold_last)` | ∫ P(L > x) dx over the layer, exactly on a published curve under `curve_area`'s conventions |
| `Payout{trigger, exhaustion, limit}`, `basis_risk(payout, covered, weight)` → `BasisRisk{covered, paid, absolute, rmse, shortfall, excess, missed, unearned, correlation}` | how far a payout on an index is from what an indemnity would pay (Clarke et al. 2016) |
| `fit_payout(index, covered, limit, weight, grid)` → `FittedPayout{payout, risk, tried}` | the trigger and exhaustion among quantiles of the index of least squared error, by a search on a grid |

## 3. Checks

`check_tyche_descriptive` (33): the certified means and standard deviations of NIST's nine univariate reference datasets (PiDigits, Lottery, Lew, Mavro, Michelso, NumAcc1–4), to 15 digits where the data are exact in binary and to 8.3 on NumAcc4, whose decimal inputs carry an error of 10⁻⁹ against deviations of 0.1; Hyndman and Fan's nine quantiles of 1, …, 10 at ¼ (3, 3, 2, 2.5, 3, 2.75, 3.25, 35/12, 2.9375) and at 0 and 1; the moments of 1, …, 5 (g₂ = −1.3, G₂ = −1.2) and of a Bernoulli sample; frequency weights against replication, reliability weights, the weighted median; covariance and correlation matrices; a summary.

`check_tyche_rank` (7): mid-ranks by hand; Spearman's ρ against 1 − 6Σd²/(n(n² − 1)) without ties and by hand with them; ρ and τ_b unchanged by increasing maps; τ_b by hand with ties and against the O(n²) count of pairs on 200 samples with ties.

`check_tyche_layers` (16): a tower's parts sum to every loss and its expected losses to the average annual loss; an exponential law's layer, attachment and exhaustion from 400,000 years against their closed forms; per event, the compound Poisson variance, met by simulated Poisson years; from a curve, the layer without limit equal to the curve's area exactly and convergent to the closed form on L(T) = μ ln T; basis risk by hand; a linear payout recovered by `fit_payout`; refusals.

## 4. Dependencies

calx `linalg` (`Rectangular`, `Matrix`).

## 5. References

- Chan, T. F., Golub, G. H. and LeVeque, R. J. (1983), Algorithms for computing the sample variance: analysis and recommendations, *Amer. Statist.* 37, 242–247.
- Hyndman, R. J. and Fan, Y. (1996), Sample quantiles in statistical packages, *Amer. Statist.* 50, 361–365.
- Joanes, D. N. and Gill, C. A. (1998), Comparing measures of sample skewness and kurtosis, *J. Roy. Statist. Soc. D (The Statistician)* 47, 183–189.
- McCullough, B. D. (1998), Assessing the reliability of statistical software: part I, *Amer. Statist.* 52, 358–366.
- Neumaier, A. (1974), Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen, *ZAMM* 54, 39–51.
- NIST/ITL, *Statistical Reference Datasets: Univariate Summary Statistics*, https://www.itl.nist.gov/div898/strd/univ/homepage.html (retrieved 2026-10-24).

## 6. Open Questions

1. A one-pass, mergeable form of the moments (Welford; Chan, Golub and LeVeque's pairwise update) for streams and for parallel reduction on calx's scheduler.
