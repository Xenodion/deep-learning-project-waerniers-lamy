# 0010. The pretrained ResNet-18 is our best model; we propose the frozen version as the final one

- **Status:** Proposed (to confirm: which ResNet stage is the final model)
- **Date:** 2026-10-22

**Context.** ADR 0009 moved to a pretrained ResNet-18. P06 (`vision/P06_resnet_transfer.ipynb`) trained it in two stages, at 224 px,
standardised with the ImageNet mean and std, with no augmentation. Measured (`results/experiments.csv`); the test scores are those of
the seed-0 model of each stage, scored once on museum B (ADR 0003):

| model | trainable weights | validation (museum A) | test (museum B), loss |
|---|---|---|---|
| best small CNN (`cnn_longer`) | 23,909 | about 59% | not scored |
| ResNet-18, frozen, new final layer | 2,565 | **80.3%** mean of 5 seeds (std 0.5; 79.8% to 81.0%) | **84.5%**, 0.425 |
| ResNet-18, last block (`layer4`) also trained | 8,396,293 | 83.3% best epoch, 81.9% mean of its 8 epochs, one seed | **86.0%**, 0.423 |

- **Transfer learning gives about 20 points over our best CNN.** The genres the CNNs struggled with improved most (frozen stage,
  validation): religious painting 34% to 72%, still life 42% to 93%.
- **The test score is higher than the validation score, not lower** (about +4 points for both stages), so there is no sign of a
  loss on unseen artists. This is within the combined noise (about +/- 3.6 points on 480 validation images and +/- 2.9 on 600 test
  images). Museum B only holds named artists, which may simply be easier.
- **Fine-tuning the last block adds little:** 2 to 3 points on the smoother validation measures and 1.5 on test, with the same test
  loss. It reaches 100% train accuracy, and it needs about 10 minutes of training against 1 second for the frozen version (once
  the frozen features are computed, about a minute). At prediction time both are the same ResNet-18.

**Decision.** The pretrained ResNet-18 is our best model by far. We propose the **frozen version** as the final model, and keep the
fine-tuned one as the documented alternative. We choose on validation and cost, **not on the test scores**: both stages were scored
on museum B, but museum B must not select a model (ADR 0003), and the 1.5-point difference is within its noise anyway.

**Rejected.** Choosing the fine-tuned version because it scored 1.5 points higher on test (that would use museum B for model
selection); tuning the small CNN further (ADR 0009).

**Consequences.** Caveats for the report: the ResNet works at 224 px against 64 px for the CNNs, so part of the gain may be
resolution and not pretraining; no data augmentation was tried; only the seed-0 model of each stage was scored on museum B;
ImageNet contains photographs, not paintings. If the team prefers the fine-tuned version for its accuracy, this ADR is superseded by a
new one. Still to do: the cost per image on a CPU, the final script `run_final.py` that reproduces the test score
for the grader, and the presentation (deadline: 30 October 2026).
