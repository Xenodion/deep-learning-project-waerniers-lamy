# 0001. Record decisions as short ADRs

- **Status:** Accepted
- **Date:** 2026-10-19

**Context.** The brief grades justification ("every claim backed by evidence") and the Q&A will ask "why did you
do it this way?". Git history shows what changed, not why.

**Decision.** We will keep one short file per significant decision in `adr/`, from [0000-template.md](0000-template.md),
written when we decide and committed with the code. A decision is significant if changing it means re-running
experiments. We never rewrite an accepted ADR's meaning: we add a new one that supersedes it.

**Rejected.** No record (reasons forgotten by the defence); one big report at the end (written from memory).

**Consequences.** Easier defence. A few minutes of writing per decision. If we write one per hyperparameter, the bar
is too low: those go in notebook tables instead.
