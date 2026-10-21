# NLP bonus: experiment plan

Working document, not an ADR. It is written **before any result** so the numbers below are predictions to be
confirmed or corrected. Values marked `(guess)` come from intuition and must be challenged; `(measure)` means a
cost to measure on a small sample before committing.

## 1. The ladder (one pipeline, one text per painting changes)

Task: query text -> top-k paintings of the collection. Every rung reuses the same retrieval and evaluation code;
only the way a painting is turned into a vector (or a score) changes.

| Rung | Method | Text per painting | Training | Human hours | Compute |
|---|---|---|---|---|---|
| R0 | BM25 / TF-IDF (classical floor) | genre, style, artist | none | ~1 h | ~0 |
| R1 | Pretrained sentence encoder (MiniLM-class), cosine | genre, style, artist | none | ~2 h | CPU, seconds |
| R2 | R1 + short generated description (small vision-language model, <= X words) | metadata + description | none | ~3 h | `(measure)` ~2 s/img GPU, ~10 s/img CPU `(guess)` |
| R3 | Fine-tuned encoder, BERT-base 110M, contrastive on pseudo-queries | metadata (+ description if R2 exists) | train(A) only | ~6 h | `(measure)` 1-3 h CPU `(guess)` |
| R4 | Smaller encoder (6 layers) + LoRA (own ~15-line `LoRALinear`) | same as R3 | train(A) only | ~3 h | `(measure)` |

Notes: R3 pseudo-queries are random subsets of a painting's fields ("portrait", "baroque, Rembrandt"), so no
sentence is ever labelled by hand. Only the paintings in the two retrieval corpora need descriptions for R2
(validation(A) 480 + B 600 = 1,080 images, not 3,000); R3 on descriptions also needs train(A).

## 2. Predictions and go / no-go (nDCG@10, by query family)

| Rung | Attribute queries | Content queries | Hypothesis |
|---|---|---|---|
| R0 | 0.85 | 0.10 | Wins on attributes (circular), blind to content ("stormy sea") |
| R1 | 0.85 | 0.25 | The encoder bridges "sea" to "landscape", nothing finer |
| R2 | 0.85 | 0.50 | Descriptions carry the content: expected biggest jump |
| R3 | 0.95 | 0.55 | Fine-tuning on labels sharpens attributes, adds little on content (+0.05, maybe inside the noise) |
| R4 | 0.93 | 0.53 | Within ~0.02 of R3 with ~10x fewer trainable parameters |

All values `(guess)`. The honest bet: **R2 is the best gain per hour, and R3 may not be worth its cost.**

Decision rule (thresholds to confirm):

- **Climb to the next rung** only if it is expected to gain >= +0.05 nDCG@10 on *content* queries for <= 6 human-hours.
- **At the small tier (24 queries) only a gap >= +0.10 is visible** (see section 4); smaller gaps need the large tier.
- **Production verdict for the director:** a neural rung is recommended only if it beats the best non-neural rung by
  >= +0.10 on content queries **and** answers in <= 200 ms/query on a laptop-class CPU and fits in ~500 MB `(assumed kiosk limits)`.
  Otherwise we recommend the classical or no-training option, and say so.

## 3. Corpus and split (reuses [ADR 0003](../../../adr/0003-simulated-museum-b-test-set.md))

| Use | Corpus searched | Trained on |
|---|---|---|
| Development, tuning, 24-query tier | validation(A), 480 paintings | train(A) |
| Final test, large tier, **once per finished rung** | museum B, 600 paintings | train(A) |

- B never trains or tunes anything. If B reveals a problem we report it, as in ADR 0003.
- **The artist field is empty for 54% of the images ("Unknown Artist")**: artist queries only apply to the named
  ones, and the artist signal is weak for half of the collection. Artist-disjoint B also means "by Monet" cannot be
  learned in A and tested in B (R3/R4 must not memorise artist names).
- Each genre has ~100-120 paintings per corpus: a query like "portraits" has >100 relevant answers, so nDCG@10 saturates.
  Attribute queries are a sanity check, not evidence.

## 4. Evaluation: query decks in tiers

| Tier | Queries | Role | Written by | Judging effort `(guess)` |
|---|---|---|---|---|
| S (smoke) | 24 | first opinion, go/no-go, tuning on validation(A) | hand | ~300 judgments, ~2.5 h split in two |
| V (validation) | 72 (S + 48 new, locked) | real comparison, run on B once | hand + LLM-assisted | ~900 judgments, ~7.5 h split in two |

Query families in every tier (rough mix): attributes ~30%, open content ~40%, compound ("baroque portrait of a man")
~15%, paraphrases of an earlier query ~10%, no-answer ("a motorcycle") ~5%. A few French queries as a robustness probe.

- **Relevance by pooling:** for each query take the union of every rung's top-5, a human grades each painting 0 / 1 / 2
  **blind** (does not know which rung returned it). 20% of queries graded by both, report the agreement (kappa).
  A new rung only needs the new paintings graded.
- **Attribute queries** get their ground truth from the labels: no grading.
- **LLM help:** may write queries and paraphrases (a human reviews them). It may **not** be the judge unless, on ~100
  paintings, its agreement with the humans reaches kappa >= 0.7; even then it is flagged, and never the same model that wrote the descriptions.
- **Locked test queries** are written before any result and not used to tune anything.
- **Statistics:** report per-query scores, paired differences between rungs, bootstrap 95% intervals. Assuming a
  per-query difference of SD ~0.25 `(guess)`, the detectable gap is ~0.10 at 24 queries, ~0.07 at 48, ~0.06 at 72,
  ~0.04 at 150. We do not claim differences below that.
- **Metrics:** nDCG@10 (main), precision@5, MRR; recall only relative to the pool. A score threshold for "no answer"
  is tuned on S and its false-answer rate reported.

## 5. Cost and verdict table (to fill)

| Rung | nDCG@10 content | CPU ms/query | Model + index size | Train time | Human h | Verdict |
|---|---|---|---|---|---|---|
| R0 |  |  |  |  |  |  |
| R1 |  |  |  |  |  |  |
| R2 |  |  |  |  |  |  |
| R3 |  |  |  |  |  |  |
| R4 |  |  |  |  |  |  |

## 6. If we had more time

Larger query deck and a second annotator pass; CLIP-style joint image-text encoder (the only option that reads pixels
for retrieval); combine with the ResNet genre prediction as a filter; multi-seed repeat of the museum B split.

## 7. Log (hypothesis vs observed)

| Date | Rung | Predicted | Observed | What we learned / decision |
|---|---|---|---|---|
