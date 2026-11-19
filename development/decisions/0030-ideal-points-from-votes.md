# 0030. Ideal Points from Votes

- **Status:** Accepted, provisionally (2026-11-19).
- **Date:** 2026-11-19
- **Decided by:** the infrastructure's ADR 0132, which founds its diplomacy domain and asks for members of a body placed on a line from their recorded votes (its survey's §A.11.2: "dynamic ideal points from ordinal votes" in tyche).

## 1. Context

A body's recorded votes (a parliament's, the UN General Assembly's) are a matrix of yes, no and missing. The standard summary of how members divide is one-dimensional ideal points: the probit model of roll calls, in which a member votes yes when α_j + β_j x_i plus a normal error is positive (Clinton, Jackman and Rivers 2004). It is an item response model whose method names no body, so it is tyche's, beside factor analysis and latent classes in `latent`. tyche had neither the model nor a fitting method for it.

## 2. Decision

`latent` gains `ideal_points.hpp`:

- **The posterior mode by EM** under normal priors on the positions and on the items' cuts and discriminations (Imai, Lo and Olmsted 2016). The E-step takes the latent utilities' truncated-normal means; the M-steps are closed-form ridge regressions for each member and each item.
- **SQUAREM's acceleration** (Varadhan and Roland 2008), with a safeguard that keeps the log posterior from falling. EM alone creeps along the direction the prior barely pins, the line's scale: a bootstrap replicate ran to its cap of iterations, and the check took more than half an hour. With SQUAREM it takes seconds.
- **Stopping** when no parameter moves by more than a tolerance in an iteration, or when the log posterior stops changing.
- **Reflection** fixed by an anchor member taken to lie on the positive side.
- **`standardised`**, which re-expresses a fit with the members' mean 0 and standard deviation 1. Every α + β x is unchanged.
- **φ/Φ and log Φ stable in the far tail** by Laplace's continued fraction for Mills's ratio, so that unanimous items and members who never break ranks give finite answers.
- **Starting values** from the leading principal component of the centred votes.
- **Fit:** the share of votes classified, and the aggregate proportional reduction in error (APRE, Poole and Rosenthal 1997).
- **Intervals by a parametric bootstrap** (Lewis and Poole 2004). Votes are drawn from the fit where they were cast, and each replicate is refitted from the fit to a looser rule. Each replicate's line is put on the fit's location and scale, since the model fixes these only through its prior: an interval is of a member's place relative to the others.

## 3. Verification

`check_tyche_ideal_points` (15 checks):

- **Recovery.** On 120 simulated members and 400 items, a fifth of the votes missing, the positions recovered at r = 0.996. The fitted line classifies the votes as well as the true line does, 0.819 against 0.815.
- **The posterior mode by another route.** EM never lowers the log posterior. BFGS on the log posterior with its analytic gradient reaches the same parameters to 5 × 10⁻⁶ and the same value to 4 × 10⁻¹².
- **Invariances and the far tail.** Reflection leaves the posterior unchanged. `standardised` leaves every α + β x unchanged. The far tail is continuous where the continued fraction takes over, and a unanimous item stays finite.
- **The bootstrap.** In one chamber, four times the votes narrow the intervals. A member who casts a fifth of its votes has an interval 2.5 times as wide as when it casts them all, about √5. The 90% intervals cover the truth, carried onto the fit's scale, for 30 of 40 members.

*Found while checking.* The bootstrap's first intervals hardly narrowed with more votes. The line's location and scale wandered from replicate to replicate, which was the prior's looseness, not the members' uncertainty. Each replicate is now put on the fit's location and scale.

*Caveat.* Covering 30 of 40 at a nominal 90% says the parametric bootstrap at the mode understates the uncertainty somewhat, as Lewis and Poole observed. The intervals are a guide, not a guarantee.

tyche's checks rise from 1722 to 1737.

## 4. Alternatives Considered

- **Markov chain Monte Carlo** (Clinton, Jackman and Rivers 2004). `bayes` could sample the posterior; EM gives its mode in a fraction of the time, and the bootstrap its spread, as Imai, Lo and Olmsted recommend for large records.
- **W-NOMINATE** (Poole and Rosenthal). It uses a Gaussian utility and its own scaling, and its implementation is GPL. The probit model is the one with a likelihood tyche can check.
- **Dynamic ideal points across sessions** (Bailey, Strezhnev and Voeten 2017). Left until a caller needs positions that move, with bridging votes.

## 5. Consequences

- Several dimensions, ordinal votes (yes, abstain, no as ordered), and positions that move over time are left until a module asks for them.
- The infrastructure's API (`science/multilateral.hpp`, `ideal_points`) and Multilateral Phronimos's negotiation view use it.
