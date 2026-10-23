# 0010. ResNet results, and the improvements we will try

- **Status:** Accepted
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
- **The frozen ResNet has 93% train accuracy against about 80% on validation:** generalisation, not capacity, is the limit.

**Decision.** Here are the possible improvements of the ResNet. We will try them, starting with the cheapest. Which ones we run, and
how we judge them, is in ADR 0011.

| idea | why it might help | cost (estimated) |
|---|---|---|
| features from several layers (pool layers 2, 3 and 4 and join them) | genres depend on texture, which the middle layers hold | about a minute of features, seconds to train |
| data augmentation (crop, mirror, colour) as precomputed copies, and averaging the prediction over an image and its mirror | the usual remedy for 1,920 images | about 5 minutes of features |
| a frozen ResNet-50 | better features | about twice the compute of ResNet-18 to run |
| dropout or weight decay on the final layer | it overfits a little | seconds, small gain at best |
| training more layers | more capacity | the last block gave 2 to 3 points for 8.4 million weights: not pursued |
| resolution above 224 px, or the whole painting instead of the centre crop | resolution gave about +5 points per doubling up to 224 (P07); the crop drops a median 25% of a painting | a new download and about 4 times the compute at 448 px |
| more training images | usually the most reliable gain | a large new download, and it changes the splits |

**Rejected.** Choosing the final model before trying the improvements; tuning the small CNN further (ADR 0009).

**Consequences.** The final model is chosen later, once the improvements have been tried and the cost per image on a CPU is measured.
Both ResNet stages were already scored on museum B, so that choice must not use those test numbers (museum B must not select a model,
ADR 0003). Caveats for the report: the ResNet works at 224 px against 64 px for the CNNs, so part of the gain may be resolution and not
pretraining (P07 measures both effects); only the seed-0 model of each stage was scored on museum B; ImageNet contains photographs, not
paintings. Still to do: the cost per image on a CPU, the final script `run_final.py` that reproduces the test score for the grader, and
the presentation (deadline: 30 October 2026).
