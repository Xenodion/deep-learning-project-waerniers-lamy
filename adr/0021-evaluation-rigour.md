# 0021. Museum B is not easier, the artist shortcut is small, and only the network-against-forest gap is certain

- **Status:** Accepted (answers open points of [0003](0003-simulated-museum-b-test-set.md), [0011](0011-resnet-improvement-round.md) and [0012](0012-final-model.md))
- **Date:** 2026-10-26

**Context.** Three points were left open. Every model scored 1.5 to 5 points higher on museum B than on validation, read as
"museum B looks easier" (ADR 0011, 0012) but never checked. ADR 0003 feared an artist shortcut and proposed to repeat the split
"with a few seeds and report the spread". And each museum B score had only its own interval, so no gap between two models was
tested. `vision/P18_evaluation_rigour.ipynb` answers the three, with rules written before the runs; nothing is chosen and no
model changes (`results/grouped_cv.csv`, `results/museum_b_tests.csv`).

**Decision.** We report the project's results with these measures:

1. **Validation split by kind of artist.** For every network, validation's named-artist paintings (161) score within 2 points of
   museum B (final model: 85.7% against 86.0%); "Unknown Artist" paintings (319) score lower (final model: 82.1%). The gap
   between validation and museum B comes from validation's two thirds of "Unknown Artist", not from an easier museum B. The
   forest is the exception: 66.5% on validation's named artists, all seen in training, but 58.2% on museum B.
2. **Artist-grouped folds in museum A** (5 repeats of 5 folds, against random folds, fixed epochs): seeing the painter in training
   is worth **1.5 points** for the frozen ResNet-18, **2.0** for the feed-forward network on handcrafted features and **3.4** for
   the forest (positive in every repeat). The accuracy on unseen artists varies by about **5 points** (std) from fold to fold.
3. **Paired tests on museum B** (exact McNemar on the paintings where one model only is right, 95% bootstrap intervals, Holm
   correction for six tests fixed in advance):

| comparison | A - B on museum B | 95% interval | p (Holm) |
|---|---|---|---|
| final model against random forest | **+27.8 points** | 23.7 to 32.0 | < 0.001 |
| feed-forward on handcrafted features against forest | +4.5 | 0.7 to 8.3 | 0.14 |
| final model against frozen ResNet-18 (fine-tuning) | +1.5 | -0.7 to 3.7 | 0.89 |
| final model against MobileNetV3-Large | +1.2 | -1.2 to 3.5 | 1.0 |
| dense with mirrored copies against dense baseline | +0.2 | -3.7 to 4.2 | 1.0 |
| final model against its int8 version | -0.2 | -1.0 to 0.7 | 1.0 |

**Rejected.** Recomputing the old comparisons with unpaired intervals only (they cannot tell 1.5 points apart); folds mixing
museum B into training (it would stop being a test set); repeating the fine-tuned model in the folds (10 minutes per training,
the frozen ResNet-18 shares its features).

**Consequences.**
- "Museum B looks easier" (ADR 0011, 0012) is replaced by: **validation is harder**, because of its "Unknown Artist" paintings.
- The fear of ADR 0003 is answered: the networks learn the genre much more than the painter. The choice of museum B's 44 artists
  moves its score by about +/- 5 points, more than the shortcut: 86.0% is one draw.
- **Only the gap between a pretrained network and the classical model is certain on museum B.** Fine-tuning (+1.5), MobileNet,
  int8, the feed-forward network against the forest, and mirrored copies are within what 600 paintings can tell apart. The
  choices of ADR 0012, 0016, 0017 and 0019 were made on validation by rules written in advance and stand; museum B neither
  confirms nor refutes their small steps.
- Two references were scored on museum B here for the paired tests only (the frozen ResNet-18 and the dense baseline re-run on
  this computer: 84.5% and 40.8%); no decision depends on them.
