# compositions: Parts of a Whole

*Status: draft (2026-11-13, [decision 0018](../development/decisions/0018-compositions.md)).* Compositions in Aitchison's geometry of the simplex: shares of a whole whose information lies in the ratios of their parts. Written for the infrastructure's sociology domain (how the 1,440 minutes of a day are divided among activities) and its bridge of social provision (how a household's time and spending are divided), and general: the parts may be the minerals of a rock, the species of a sample read as proportions, the shares of a budget or of an electorate.

## 1. Concepts

**A composition is a set of ratios.** A vector of positive parts carries information only in the ratios x_i/x_j: (480, 420, 540) minutes and (1/3, 7/24, 3/8) of a day are the same composition. Its representative is the closure C(x) to a stated total (1, 100, 1440). Ordinary statistics of closed parts mislead, because the closure forces negative covariances among them and a part's change is read against a total that moves with it (Aitchison 1986).

**Aitchison's geometry.** Perturbation x ⊕ y = C(x_i y_i) and powering a ⊙ x = C(x_i^a) make the simplex S^D a real vector space of dimension D − 1, and the inner product ⟨x, y⟩_a = Σ clr_i(x) clr_i(y) a Euclidean one. The centred log-ratio clr(x)_i = ln(x_i/g(x)) carries it isometrically onto the hyperplane of zero sum; an orthonormal basis gives the isometric log-ratio coordinates ilr, D − 1 real numbers in which any multivariate method may be used and whose results are carried back (Egozcue et al. 2003).

**Balances.** A sequential binary partition splits the parts into two groups, then each group again, until every part stands alone; each split gives a balance √(rs/(r + s)) ln(g₊/g₋), the log-ratio of the groups' geometric means scaled to unit norm, and the D − 1 balances are orthonormal coordinates that can be read (Egozcue and Pawlowsky-Glahn 2005). Pivot coordinates set one part against all the others first (Hron, Filzmoser and Thompson 2012), the default here.

**A sample's centre and spread.** The centre is the closure of the parts' geometric means, the mean in Aitchison's geometry; the variation matrix holds the variances of every pairwise log-ratio; the total variance is half their sum over D, equal to the sum of the clr variances and to the mean squared Aitchison distance from the centre.

**Zeros.** A zero has no log-ratio. A rounded zero (present but below what was recorded) is imputed by multiplicative replacement, which keeps every ratio among the parts that were present (Martín-Fernández et al. 2003); an essential zero (truly absent) is not imputed and is refused with the reason.

**Compositions in models.** As a predictor, a composition enters a linear model through its D − 1 ilr coordinates; the fitted values do not depend on the basis, and a part's role is read from the pivot coordinate with it first. Moving Δ minutes from one part to another, the predicted change of the outcome is the difference of the fitted model at the two compositions: compositional isotemporal substitution, the method of time-use epidemiology (Dumuid et al. 2018). As a response, its coordinates are regressed on covariates and the predictions carried back to the simplex.

## 2. Content

All in namespace `tyche::compositions`; `compositions.hpp` includes every header. Compositions are `std::span<const double>` (positive, any units) and samples `calx::linalg::Rectangular` (a row for each composition).

`simplex.hpp`:

| Name | Content |
|---|---|
| `closure(x, total = 1)` | C(x); zeros allowed, a vanishing sum refused |
| `neutral(D, total)` | C(1, …, 1) |
| `perturb(x, y, total)`, `perturb_inverse(x, total)`, `difference(x, y, total)` | x ⊕ y, ⊖x, x ⊖ y |
| `power(x, a, total)` | a ⊙ x |
| `geometric_mean(x)` | g(x), in x's units |
| `clr(x)`, `clr_inverse(z, total)` | centred log-ratio and its inverse |
| `alr(x, k = D − 1)`, `alr_inverse(y, k, total)` | additive log-ratio (not an isometry) and its inverse |
| `inner(x, y)`, `norm(x)`, `distance(x, y)` | ⟨x, y⟩_a, ‖x‖_a, d_a(x, y) |

`balances.hpp`:

| Name | Content |
|---|---|
| `SignTable` | a sequential binary partition: D − 1 rows of +1, −1, 0 |
| `basis(partition)` → `Basis` | `parts`, `partition`, `contrast` (Ψ, (D − 1) × D, orthonormal rows summing to 0); the partition validated |
| `pivot_partition(D, order = {})`, `pivot_basis(D, part = 0)` | pivot coordinates, a given order or one part first |
| `ilr(x, basis)`, `ilr(x)`, `ilr_inverse(z, basis, total)`, `ilr_inverse(z, total)` | coordinates in a basis (default: pivot coordinates of the parts in order) |
| `ilr(X, basis)`, `ilr_inverse(Z, basis, total)` | the same, row by row |
| `change_of_basis(from, to)` | Ψ_to Ψ_fromᵀ, orthogonal |
| `balance(x, plus, minus)` | the balance between two groups of parts |

`sample.hpp`:

| Name | Content |
|---|---|
| `centre(X, total = 1)` | the closed geometric mean |
| `centred(X, total)` | each row ⊖ the centre |
| `variation_matrix(X, ddof = 1)` | t_ij = var(ln(x_i/x_j)), D × D `Matrix` |
| `clr_covariance(X, ddof)` | covariance of the clr coordinates |
| `total_variance(X, ddof)`, `metric_standard_deviation(X, ddof)` | (1/(2D)) Σ t_ij; √(totvar/(D − 1)) |
| `describe(X, ddof, total)` → `Summary` | `n`, `parts`, `ddof`, `centre`, `variation`, `clr_variances`, `total_variance`, `metric_standard_deviation` |

`zeros.hpp`:

| Name | Content |
|---|---|
| `replace_zeros(x, ZeroOptions)`, `replace_zeros(X, ZeroOptions)` | multiplicative replacement; `ZeroOptions{detection_limits, fraction = 0.65, delta, total, essential}`: δ_j = fraction × limit_j, or one δ; closed to `total` (default the composition's sum); a zero in a part marked essential refused |
| `has_zeros(x)` | whether any part is 0 |

`regression.hpp`:

| Name | Content |
|---|---|
| `fit_predictor(X, y, covariates = {}, PredictorOptions)` → `CompositionPredictor` | y on [1,] ilr(x), covariates by `regression::ols`; `PredictorOptions{partition, intercept = true, ols}`; the model holds `n`, `parts`, `covariates`, `intercept`, `basis`, `fit` (a `regression::LinearFit`), `clr_coefficients` (b = Ψᵀγ, Σ b = 0), `centre` and `total` (the sample's mean total) |
| `design(model, X, covariates)`, `predict(model, X, covariates)` | the design rows, and `regression::Prediction` (mean, se, interval) |
| `pivot_effect(model, part)` → `Estimate` | the coefficient of the first pivot coordinate with `part` first: `estimate`, `se`, `t`, `p`, `lower`, `upper` |
| `reallocation(model, before, after)` → `Reallocation` | ŷ(after) − ŷ(before) as an `Estimate` in `change`, with `before`, `after` |
| `one_for_one(model, reference, to, from, delta)` | Δ units moved from part `from` to part `to` |
| `one_for_remaining(model, reference, part, delta)` | Δ added to `part`, taken from the others in proportion |
| `fit_response(Y, covariates, ResponseOptions)` → `CompositionResponse` | each ilr coordinate on [1,] covariates; `fits`, `coefficients` (columns × (D − 1)), `effects` (columns × D, compositions), `residual_covariance`, `total` |
| `predict(response, covariates, total)` | fitted compositions |

A minimal use, the minutes of a day and an outcome:

```cpp
#include <tyche/compositions/compositions.hpp>
namespace co = tyche::compositions;
const auto days = co::replace_zeros(minutes, {.detection_limits = limits});   // n × 4, rows summing to 1440
const auto m = co::fit_predictor(days, outcome, covariates);                   // pivot coordinates, OLS
const auto r = co::one_for_one(m, m.centre, /*to*/ 3, /*from*/ 1, 30.0);       // 30 minutes from sitting to activity
// r.change.estimate, r.change.lower, r.change.upper
```

## 3. Checks

`check_tyche_simplex` (55): closure to 1, 100 and 1440 and the invariance of every operation to the scale of its arguments; the clr summing to zero and inverted; perturbation a commutative group (neutral, inverse, associativity) with powering distributing over it; the two forms of the inner product, the distance as the norm of the difference, invariant under perturbation and scaling under powering; the contrast matrices of 35 pivot bases and 60 random partitions orthonormal with rows summing to zero; ilr an isometry and a linear isomorphism, inverted; balances equal to their coordinates; the first pivot coordinate as Hron et al.'s formula; changes of basis orthogonal; invalid partitions refused; cross-checks against scikit-bio 0.7.4 (closure, clr, clr⁻¹, powering, perturbation, the inner product, ilr in the bases of `sbp_basis`; `tests/fixtures/make_skbio_composition_fixtures.py`).

`check_tyche_composition_sample` (43): the centre as the closure of geometric means, invariant to the rows' scales and equivariant under perturbation and powering; the variation matrix symmetric with a zero diagonal, equal to the log-ratio variances computed directly and vanishing for proportional parts; the total variance as (1/(2D)) Σ t_ij, Σ var(clr_i), the mean squared distance to the centre and (D/2) times the mean of T, for divisors n − 1 and n, invariant under perturbation and scaling by a² under powering; on 20,000 seeded logistic-normal compositions the centre within 0.005 of ilr⁻¹(μ) and the total variance within 0.4% of tr Σ; multiplicative replacement keeping the total and the ratios of non-zero parts exactly, with δ or detection limits, and refusing essential zeros and impossible values; cross-checks against scikit-bio's `pairwise_vlr`, `centralize` and `multi_replace`.

`check_tyche_compositional_regression` (40): exact recovery of known coefficients without noise and within four standard errors with it (n = 2000); fitted values, clr coefficients, R² and predictions independent of the basis; the two-part closed form b ln(x₁/x₂); each part's pivot effect equal to the refitted model's first coefficient, standard error and p-value; reallocations equal to the fitted model's difference and to Σ b_i ln(x¹_i/x⁰_i), with standard error √(dᵀVd); the reverse move exactly opposite; one for remaining keeping the total and the others' ratios; a robust covariance changing the standard error only; the response model recovering its coefficients, its effects and predictions independent of the basis, its intercept-only prediction the sample's centre.

## 4. Dependencies

`regression` (least squares, predictions), `probability` (Student's t); calx `linalg` (`Rectangular`, `Matrix`).

## 5. References

- Aitchison, J. (1986), *The Statistical Analysis of Compositional Data*, Chapman and Hall.
- Dumuid, D. et al. (2018), Compositional data analysis for physical activity, sedentary time and sleep research, *Stat. Methods Med. Res.* 27, 3726–3738.
- Dumuid, D. et al. (2019), The compositional isotemporal substitution model: a method for estimating changes in a health outcome for reallocation of time between sleep, physical activity and sedentary behaviour, *Stat. Methods Med. Res.* 28, 846–857.
- Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G. and Barceló-Vidal, C. (2003), Isometric logratio transformations for compositional data analysis, *Math. Geol.* 35, 279–300.
- Egozcue, J. J. and Pawlowsky-Glahn, V. (2005), Groups of parts and their balances in compositional data analysis, *Math. Geol.* 37, 795–828.
- Hron, K., Filzmoser, P. and Thompson, K. (2012), Linear regression with compositional explanatory variables, *J. Appl. Stat.* 39, 1115–1128.
- Martín-Fernández, J. A., Barceló-Vidal, C. and Pawlowsky-Glahn, V. (2003), Dealing with zeros and missing values in compositional data sets using nonparametric imputation, *Math. Geol.* 35, 253–278.
- Pawlowsky-Glahn, V. and Egozcue, J. J. (2001), Geometric approach to statistical analysis on the simplex, *Stoch. Environ. Res. Risk Assess.* 15, 384–398.
- Pawlowsky-Glahn, V., Egozcue, J. J. and Tolosana-Delgado, R. (2015), *Modelling and Analysis of Compositional Data*, Wiley.
- van den Boogaart, K. G. and Tolosana-Delgado, R. (2013), *Analyzing Compositional Data with R*, Springer.

## 6. Open Questions

1. Model-based replacement of zeros (log-ratio EM, Bayesian-multiplicative replacement of count zeros) and the treatment of essential zeros as a separate part of the model.
2. Multivariate tests of a covariate's effect on a composition as response (a Wald test on its row of coefficients, invariant to the basis), and the regression of a composition on a composition.
3. Robust estimation (the MCD in ilr coordinates), compositional biplots and principal components, and the Dirichlet and logistic-normal laws as `probability` objects with their likelihoods.
4. Compositional data with an informative total (the time recorded is not always 1440 minutes): a total kept beside its composition.
