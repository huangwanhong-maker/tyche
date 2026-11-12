# 0016. Information Through a Record

- **Status:** Accepted, provisionally (2026-11-12).
- **Decided by:** the infrastructure's ADR 0099 (the user's generative–relational market formalism, whose epistemic layer asks how much of the difference between two mechanisms a record shows).

## 1. Decision

`probability` gains `information.hpp`, on finite laws and kernels:

- **entropy** and **relative entropy** (physis's `relative_entropy`, used and not written again);
- **the law of a record,** $pK$, for a kernel $K$ from outcomes to records, and the kernel of a coarsening;
- **the chain rule through a record,** `discrepancy(p, q, K)`:
  $D(p\,\Vert\,q) = D(pK\,\Vert\,qK) + \mathbb E_{y\sim pK}\,D(p(\cdot\mid y)\,\Vert\,q(\cdot\mid y))$,
  the part of a difference a record shows and the part it hides, with the hidden part record by record;
- **`kernel_discrepancy`,** $\mathbb E_p D(K\,\Vert\,K')$, the term gained when two candidates also record differently;
- **the rate at which two jump processes part,** $\sum_i p_i \sum_{j\ne i}[k_{ij}\ln(k_{ij}/k'_{ij}) - k_{ij} + k'_{ij}]$, at a given law and at the first process's stationary law.

## 2. Rationale

The split names no object in nature: it is the chain rule of relative entropy read as an account of observation. Partial identification, the comparison of experiments and model criticism all use it, so it is tyche's and not a domain's.

`probability` now links calx's `linalg` and physis's `statistical_mechanics`, which already held relative entropy and stationary laws.

## 3. Verification

`check_tyche_information` (22): the chain rule to rounding on 200 random laws and kernels; both parts never negative; a garbled record shows no more (data processing); a record that shows the outcome hides nothing, one that shows nothing hides everything; a coarsening shows the relative entropy of the cells' laws; the joint identity with two kernels; the two-state chain's rate in closed form, $2p(1-p)[c\ln(c/c') - c + c'] = 0.7746$ for $p = 0.4$, $c = 1$, $c' = 4$, with equal stationary laws; the rate against the relative entropy of the laws of start and end over a short time, from matrix exponentials; refusals.

tyche's checks rise from 821 to 843.

## 4. Consequences

Laws are finite and dense. Continuous laws, estimated discrepancies with their sampling error, and mutual information between records can follow when a domain asks.
