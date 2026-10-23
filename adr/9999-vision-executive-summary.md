# Vision: executive summary

**What we built.** A classifier that tells the genre of a painting (portrait, landscape, cityscape, still life, religious painting) from its
photo. The final model, a ResNet-18 pretrained on ImageNet with its last block fine-tuned, is right on **86.0% of 600 test paintings**,
by artists it never saw, and takes about 70 ms per photo on one CPU thread. This summary covers the vision part only: the natural-language part of the guide is a separate experimental ladder (ADR 0005).

Every step below is justified in an ADR (the numbered files of this folder) and measured in a notebook
(`../src/deep_learning_project_waerniers_lamy/vision/`). Scores
are validation accuracy (480 images, about +/- 3.6 points) unless stated; "test" is museum B (600 images, about +/- 2.9 points).

## What we did, in order

| # | Step | What we found | Decision |
|---|---|---|---|
| 1 | **Scope** (ADR 0002, 0004) | The brief asks for two features; the deliverables and the course are about the vision side. | Genre classifier only, 5 genres x 600 images from WikiArt (3,000 images, 224 px). The language part is handled separately (ADR 0004, 0005). |
| 2 | **Test design** (ADR 0003) | A painter's style gives away the genre: the artist alone predicts the genre of 70% of the paintings that have a named artist (chance 20%). | Two fictional museums: A (train 1,920 and validation 480) and B (test 600). No named artist in both; same number of images per genre. B is used once per model. |
| 3 | **Dense network** (P02, ADR 0006) | 39.0% (5 seeds); it memorises the images (98% train). | Kept unchanged as the reference. Test 41.8%. |
| 4 | **Small CNNs** (P03 to P05, ADR 0007, 0008) | 53.6%, then 58 to 61% with longer training. Batch norm learned faster but made validation erratic. A deeper CNN memorised the training set (99.7%) without gaining. | Stop at about 59%. Test 58.7 to 63.7%. |
| 5 | **Pretrained ResNet-18** (P06, ADR 0009) | Frozen, with only a new final layer trained: **80.3%** (5 seeds), test 84.5%. | The jump (+20 points) comes from pretraining on ImageNet. |
| 6 | **Pretraining or resolution?** (P07) | The ResNet saw 224 px, the CNNs 64 px. On the same 64 px images the ResNet still scores 70.2% (112 px: 75.4%). | About 12 points come from pretraining and 10 from resolution. |
| 7 | **Improvement round** (P08, ADR 0011) | Features from several layers 80.6%, augmentation 80.8%, flip averaging +0.1, frozen ResNet-50 81.9% (twice the time). | Nothing beat our 2-point bar. No stronger model. |
| 8 | **Cost per photo** (P09) | One CPU thread: dense 0.1 ms, small CNN 1.7 ms, ResNet-18 70 ms (45 MB), ResNet-50 139 ms (94 MB). | A neural network is worth it: fast enough for one photo at a time. |
| 9 | **Final model** (P06 with 3 seeds, ADR 0012) | Training the last block: 83.3% in all 3 seeds against 80.3% frozen (+3.0), at the same prediction cost. | **Fine-tuned ResNet-18, 224 px. Test 86.0%.** `run_final.py` reproduces it. |
| 10 | **Error analysis** (P10) | See below. | We stop here. |

## Why 86% is close to the ceiling, and why we stopped

On the test set, the errors are concentrated in pairs of genres that overlap:

- **Cityscape and landscape**: the two genres most alike in the network's own features, and the largest confusion (19 cityscapes called
  landscapes on test, 18 on validation).
- **Religious painting and portrait**: the second most alike pair (16 religious paintings called portraits on test, 13 on validation).
- Across the 10 pairs of genres, the more alike two genres are, the more they are mixed up (rank correlation 0.86).
- The true genre is the model's **first or second choice for 95%** of the validation paintings (top-2 accuracy 95.4%, against 83.3% for the first choice alone, the usual "accuracy"; two genres picked at
  random out of five would give 40%; this top-2 figure is a validation diagnostic only: on the test set we only look at the first choice),
  so the errors are between two plausible
  genres, not blindness. And 29 of the 80 validation mistakes were made with a confidence above 0.8: the model is often sure about a
  painting whose label is debatable (a city view is also a landscape, a saint's portrait is also a portrait).

So the remaining errors look like a property of the labels, not of the model. Three different ways of giving the network more
information all stopped between 80% and 82% (step 7), and the last real gain came from fine-tuning (+3 points). A stronger model would
cost about twice as much to run for one or two points, so **we do not expect to go much above 86% and we did not try stronger models.**

## What to keep in mind

- This is our reading of the evidence, not a proven ceiling. We did not look at the paintings one by one to check the labels, we only
  tested ResNet-50 frozen (never fine-tuned), and we did not add training images.
- Every model scored higher on museum B than on validation, so museum B looks easier (named artists only): do not expect 86% in a real
  second museum.
- No classical model was compared (this is a deep-learning course): "worth it" means worth it among neural networks.
- Timings come from one laptop; the ratios between models matter more than the milliseconds.
