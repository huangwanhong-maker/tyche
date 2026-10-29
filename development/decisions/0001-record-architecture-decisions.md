# 0001. Record Architecture Decisions

- **Status:** Accepted
- **Date:** 2026-10-24
- **Decided by:** the user and Claude, following the convention shared with the sibling libraries and the infrastructure

## 1. Context

tyche is long-lived, and the infrastructure's domains and works will depend on it. Future sessions need to know why its interfaces are as they are, including sessions that do not remember the original discussion.

## 2. Decision

- Record each decision that shapes the structure, interfaces or tooling of tyche as a numbered record in `development/decisions/`.
- Never rewrite a record. A changed decision gets a new record that supersedes the old one.

## 3. Alternatives Considered

- **Commit messages only.** Rejected, because the reasons become hard to find.
- **A single design document edited in place.** Rejected, because it loses the history of why things changed.

## 4. Consequences

- Each decision costs a small amount of writing.
- The READMEs and `docs/` describe the current state. The records explain how it came about.
