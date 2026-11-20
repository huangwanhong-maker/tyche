# 0032. The Order of Two Questions: Classical and Quantum-Like Models, Nested, and the QQ Equality

- **Status:** Accepted, provisionally (2026-11-21).
- **Date:** 2026-11-21
- **Decided by:** Claude, at the infrastructure's Stage 8 of evolutionary allocation (the user: "please continue all remain stage if unfinish"), after its survey's Part IX (*Evolutionary heterogeneous resource allocation*): quantum-like models of judgement fit an equality and claim no advantage; they are to be nested over their classical rivals, with the parameter-free test, so that whatever they explain is weighed against what a classical model explains.

## 1. Context

When two questions are asked in two orders, answers can depend on the order. Quantum-like models of judgement, in which an answer is a projection of a state, predict order effects and, in any dimension, the QQ equality: the probability of answering the two differently is the same in both orders (Wang and Busemeyer 2013). The equality is parameter-free and so testable without fitting anything; but the order-free classical model satisfies it too, as does a symmetric anchoring, so it does not single out projective models. tyche held tests of proportions but nothing that set these models side by side.

## 2. Decision

`testing/question_order.hpp`:

- `OrderCounts`, `Laws`, `Fit`, `QQ`, `Equivalence`, `Nested`, `OrderAnalysis`;
- `projective_laws(θ, φ)`: a real plane, the state at θ, A's "yes" along 0 and B's along φ;
- `qq_test(counts)`: q̂ = d̂_AB − d̂_BA with its unpooled standard error and z-test;
- `qq_equivalence(counts, Δ, α)`: the equality shown within a margin by two one-sided tests (Schuirmann 1987);
- `analyse_order(counts, level, margin)`: the saturated, order-free, q = 0 (in closed form) and projective (grid and Nelder and Mead) fits by maximum likelihood, the nested likelihood-ratio tests (an order effect, 3 degrees of freedom; the equality, 1; the plane within q = 0, 3), and a reading that says "not rejected" for an equality not rejected and "shown within a margin" only when the equivalence test shows it.

## 3. Alternatives Considered

- **Quantum models in higher dimensions, or with complex amplitudes.** The equality holds for all of them; the plane is the smallest projective model and is fitted to show what a projective fit is, not to claim the plane.
- **Reading a non-rejection of the equality as support.** Refused: equivalence within a declared margin is the test that supports it.

## 4. Consequences

`check_tyche_question_order` (14 checks): counts made by a projective model give q = 0 exactly, the plane fitting them as well as the saturated model, an order effect and the reading that says so; one joint law in both orders gives no order effect; a classical Markov update of the first answer, asymmetric between the questions, breaks the equality (q = −0.18) and is read so; a symmetric anchoring keeps it exactly, though classical; the constrained fit keeps q = 0; over 1,000 surveys from a projective model the z-test rejects at about its level; with 2,000 per order the equality is shown within 0.05 and with 100 it is not. peira's `design/order.hpp` (its decision 0014) sizes surveys for these tests.
