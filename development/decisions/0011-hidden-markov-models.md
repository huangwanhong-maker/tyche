# 0011. Hidden Markov Models with Gaussian Emissions

- **Status:** Accepted, provisionally (Gaussian emissions only; other emissions, covariates of the transitions and duration models planned)
- **Date:** 2026-11-05
- **Decided by:**
  - **The need.** The infrastructure's economics domain reads the regime of an economy (contraction, normal growth, a boom) from its growth of real GDP, and a policymaker's POMDP acts on that belief ([ADR 0074](https://github.com/huangwanhong-maker/The-Science-Infrastructures/blob/main/development/decisions/0074-perception-and-decision-in-the-economy-atlas.md)). A chain whose state is seen only through noisy observations names no economy.
  - **What Claude did.** It wrote the header, its checks and this record, under the user's request to visualise "such algorithms in economics state perception and decision making".

## 1. Context

`processes` held Markov chains in continuous time seen at discrete times (decision 0010), whose misclassification is a hidden Markov model of one observation per sighting. `decision` holds POMDPs, whose belief update is Bayes's rule on a known model. Neither fits a model with continuous observations to many sequences, nor gives the smoothed law or the most probable path.

## 2. Decision

**`processes/hidden_markov.hpp`:**
- `GaussianHmm` (π, A, μ, σ) and `decode(M, y)` → `Decoded`:
  - the filtered laws, by the forward recursion scaled at each step, so that each is the law itself and the log-likelihood is the sum of the logs of the scales (Rabiner 1989, §V.A);
  - the smoothed laws, by the backward pass;
  - the most probable path, by Viterbi's recursion in logs.

  A missing observation (NaN) emits with likelihood 1, so the filter only predicts across it.
- `fit_hmm(sequences, K, starts, max_iterations, tolerance, seed, shared_sd)` → `HmmFit`:
  - Baum–Welch over many sequences at once;
  - starts from the pooled quantiles, keeping the best of several;
  - σ held above 10⁻³ of the pooled deviation, where the likelihood is unbounded;
  - the states numbered by increasing mean;
  - the log-likelihood traced along the iterations.
- `shared_sd`: one σ for every state, Hamilton's (1989) model of the business cycle, whose states differ by their means alone.

## 3. Alternatives Considered

- **In `decision`, beside the POMDP's belief.** Rejected: the belief update there is a step of a known model; fitting the model to sequences is the statistics of a process, which is `processes`'.
- **One sequence at a time.** Rejected: a regime that occurs a few times in one economy's sixty years is estimable only pooled across economies, which share the model and keep their own paths.
- **A deviation for each state, always.** Kept as the default, and made optional. On the World Bank's growth of 212 economies standardised each by its own median and deviation, three free deviations make a regime of turbulence rather than of contraction (its mean near zero, its σ large). One shared σ makes the regimes differ by their means: −3.86σ, 0.01σ and 3.39σ.
- **Unscaled forward variables in logs.** Rejected: the scaled recursion is as stable for these lengths, and gives the filtered laws, which a POMDP's belief needs, without renormalising.

## 4. Consequences

- **Checks.** `check_tyche_hidden_markov` (7):
  - the log-likelihood and every smoothed marginal against the sum over all 729 paths of a three-state model and six observations (one missing), to 10⁻¹²;
  - the most probable path against brute force;
  - the last filtered law equal to the last smoothed one, and every law summing to 1;
  - a two-state model's means, deviations and staying probabilities recovered from 300 sequences of 60 steps (5 % missing), the log-likelihood never falling along EM;
  - with a shared deviation, three means and their one σ recovered.
- **Its first use.** economics' `regimes` reads every economy's regime from its growth, and `policy` gives the belief to a POMDP.
- **`decision`, alongside.** HSVI records its bounds before each trial (`HsviResult::trace`), so that a page can draw the interval closing.
- **Planned.** Emissions of other families (Poisson, multivariate Gaussian); transitions that depend on covariates; durations other than geometric (hidden semi-Markov models).
