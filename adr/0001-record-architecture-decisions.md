# 0001. Record architecture decisions as ADRs in the repository

- **Status:** Accepted
- **Date:** 2026-10-19
- **Deciders:** Waerniers, Lamy
- **Milestone:** m1

## Context

The project brief asks us to "build, diagnose and defend" the neural-network components of the museum guide,
with "every number measured, every claim backed by evidence". We will be assessed on a Git deliverable with a
clear history, a 10-minute presentation and a 5-minute Q&A, and the Q&A will mostly be "why did you do it
this way?". Two people share the work, and we will make many small choices over several weeks (classes, input
size, architecture, regularisation, pretrained backbone, how to compare with a classical model). Without a
written trace, the reasons for these choices will be forgotten by the defence, and the git history alone only
shows what changed, not why.

## Options considered

1. **No written record.** Zero effort, but the reasoning lives in our heads and in chat messages.
2. **Long-form report at the end.** One document, but written from memory after the fact, so it rewrites
   history instead of recording it.
3. **One short ADR per significant decision, in `adr/`, versioned with the code.** A few minutes per decision,
   written when the decision is made, and committed together with the code it affects.

## Decision

We will keep one Markdown file per significant decision in `adr/`, numbered `NNNN-short-title.md`, using the
template in [0000-template.md](0000-template.md). A decision is "significant" if changing it later would force
us to re-run experiments or redo part of the deliverable. An accepted ADR is never edited to change its
meaning; if we change our mind, we write a new ADR that supersedes it and update the old one's status line.

## Consequences

- Easier: preparing the oral defence, since each "why" has a dated answer and the evidence it relied on.
- Easier: onboarding the second team member, and reviewing each other's work.
- Harder: a small writing cost at each decision, which we accept because the brief rewards justification.
- To check: the ADR count should stay small (roughly 8 to 12 for the whole project). If we are writing one per
  hyperparameter, the bar for "significant" is too low, and those belong in notebook tables instead.
