# 0035. Census Records and the Price Decomposition

- **Status:** Accepted, provisionally (2026-11-23).
- **Date:** 2026-11-23
- **Decided by:** the infrastructure's survey *Evolutionary economics models and methods* (its §7.5, first step), at the user's choice to begin the economics of evolution with the Price decomposition in tyche, since both heuresis's searches and the economics domain's firm panels stand on tyche.

## 1. Context

Evolution as an object of economics (firms selected by markets, routines spreading by imitation) and evolution as a method (candidate solutions selected by a search) are records of one form: a population at two times, a weight and a character for each member, and who descends from whom. Price's equation accounts exactly for any such record; the productivity decompositions of firms (Foster, Haltiwanger and Krizan; Griliches and Regev; Melitz and Polanec) are instances of it with entry and exit added against a reference. No published system computes it over arbitrary genealogies with outside proposals (the survey's Part V). It names no object in nature, so it belongs in a foundation, and tyche is the one both evolutions stand on.

## 2. Decision

A new module `census` (`census/include/tyche/census/price.hpp`), header-only, templated on the number type:

- a record of two censuses joined by transmission shares, with arrivals by source, operators and exit reasons, validated;
- the decomposition in four forms (`survivors`, `incumbent`, `average`, `declared`), each an identity; Price's fitness form and selection by nested groups in the survivors' form; transmission credited to operators, entry to sources, exit to reasons;
- the second moment, the composition of steps into spans, and records built from panels keyed by identity.

## 3. Alternatives Considered

- **The decomposition in heuresis.** It would serve searches and not firms; tyche serves both, and heuresis's lineage can be read into a record.
- **The decomposition in nomos or the economics domain.** It names no economic object; firm panels are one use among several.
- **One reference only.** The reference carries meaning (in a population whose every member grows by γ, the incumbent form credits entrants with s_N γ and the survivors' form with nothing), so each is offered and named, and the economics domain chooses with reasons.
- **Doubles only.** An identity is best checked exactly; the template takes morphe's rationals, and doubles agree with them to 10⁻¹².

## 4. Consequences

- `check_tyche_price`, 30 checks (see the module's README); tyche 1819.
- The economics domain's productivity decompositions will compose on it, and the infrastructure's API can read heuresis's runs into records, so that what outside proposals contributed to a search's improvement is accounted exactly, and never claimed as a cause.
- Not built: vector characters, causal credit for entrants, Baldwin and Gu's reference.
