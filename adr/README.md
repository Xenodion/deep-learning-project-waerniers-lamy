# Architecture Decision Records

Short, dated notes explaining **why** we made each significant choice in the museum-guide project. The code
shows what we did; these notes show the reasoning and the evidence, so we can defend it at the oral.
Format and rules: [0001](0001-record-architecture-decisions.md). Copy [0000-template.md](0000-template.md)
to start a new one.

Statuses: **Proposed** (drafted, not yet agreed by both of us) → **Accepted** → **Superseded by NNNN**.

## Index

| # | Decision | Status | Milestone |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions as ADRs | Accepted | m1 |
| [0002](0002-classification-target-and-dataset-subset.md) | Classify 5 genres, 300 images each, downloaded at 224 px | Proposed | m1 |

## Decisions we know are coming

Not yet written. Each becomes an ADR when we actually decide it, with the numbers that justified it.

| Milestone | Decision to take | What the evidence will be |
|---|---|---|
| m1 | Train / validation / test split, and how we avoid artist leakage | Number of artists shared between splits |
| m1 | Input representation (image size, flattening, standardisation statistics) | Baseline accuracy at 2 sizes |
| m1 | Dense baseline architecture and training loop | Learning curves, validation accuracy |
| m2 | Regularisation and monitoring protocol (what we log, what counts as a failure) | Train vs validation curves, diagnosis table |
| later | Pretrained ResNet: variant, frozen vs fine-tuned, input size | Accuracy, training time, CPU inference time |
| later | Classical baseline and the production recommendation (neural network or not) | Accuracy, model size and CPU latency, side by side |
| to confirm | Scope of the natural-language search (feature 2 of the brief) | Answer from the instructor |
