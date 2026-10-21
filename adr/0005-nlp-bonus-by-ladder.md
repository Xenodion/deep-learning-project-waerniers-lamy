# 0005. Language feature as a bonus, built one rung at a time

- **Status:** Accepted
- **Date:** 2026-10-20

**Context.** [0004](0004-focus-on-genre-classifier.md) left the language feature aside. The teacher now calls it a
bonus and suggests a pretrained sentence encoder (BERT-like), fine-tuned on the labels, with a smaller model plus LoRA
as a variant, or simply embedding the paintings' metadata and ranking by cosine similarity. The dataset has no captions,
only genre, style and artist (54% of the images have no named artist).

**Decision.** The graded genre classifier keeps priority. The language feature is built as a ladder, simplest first,
and stops when the gain no longer pays for the effort: BM25 on metadata, a pretrained encoder with cosine, then generated
descriptions, then a fine-tuned encoder (full, then smaller + LoRA). Predictions, cost estimates, go / no-go thresholds
and the evaluation protocol (hand-written queries, blind graded relevance, museum A to develop, museum B to test once)
are in [nlp/EXPERIMENT_PLAN.md](../src/deep_learning_project_waerniers_lamy/nlp/EXPERIMENT_PLAN.md).

**Rejected.** Building the full language pipeline first (it is a bonus and takes time from graded work); judging with
an LLM or a trained judge (it adds a second model to validate; kept as an idea for later).

**Consequences.** Stopping after the first rungs is a valid outcome, as long as we report what we would do next. The
retrieval is evaluated on the same museum A / museum B split as the vision models ([0003](0003-simulated-museum-b-test-set.md)).
