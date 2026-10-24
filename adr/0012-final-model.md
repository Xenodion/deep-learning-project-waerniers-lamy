# 0012. The final model: ResNet-18 pretrained on ImageNet, last block fine-tuned, 224 px

- **Status:** Accepted
- **Date:** 2026-10-23

**Context.** ADR 0010 left the final model open, and ADR 0011 tried cheap improvements (none helped). The last open question was whether
training the last block of the ResNet beats the frozen ResNet, which had one seed only. P06 repeated it with 3 seeds under a rule written
before the run: it wins if its mean is more than about 2 points above the frozen 80.3%. Measured (`results/experiments.csv`; costs from
P09, one image at a time on one CPU thread; the error analysis is P10):

| model | validation | test (museum B) | time per photo | size |
|---|---|---|---|---|
| dense baseline, 32 px | 39.0% | 41.8% | 0.1 ms | 1.6 MB |
| best small CNN, 64 px | 58.8% | 63.7% | 1.7 ms | 0.1 MB |
| ResNet-18 frozen, 224 px | 80.3% (5 seeds, std 0.5) | 84.5% | 70.3 ms | 45 MB |
| **ResNet-18, last block trained, 224 px** | **83.3%** (3 seeds, all 83.3%) | **86.0%** (loss 0.423) | 70.5 ms | 45 MB |
| ResNet-50 frozen, 224 px | 81.9% (5 seeds) | not scored | 139.1 ms | 94 MB |

**Decision.** The final model is the **ResNet-18 pretrained on ImageNet, with layers 1 to 3 frozen, layer 4 trained and a new final layer
(512 numbers to 5 genres)**, on 224 px images standardised with the ImageNet mean and std. Trained for 8 epochs (Adam, learning rate
1e-4, batches of 64, seed 0, best epoch kept on validation), without augmentation. It is the run `resnet18_layer4` of
`results/experiments.csv`. The rule is met (+3.0 points), and the two ResNet-18 models cost the same at prediction time, so fine-tuning
only costs training time (about 10 minutes, once). `vision/run_final.py` reloads it from `results/final_model.pt` (32 MB) and reproduces
the test score; `--train` trains it again.

**Rejected.** The frozen ResNet-18 (3 points lower on validation, 1.5 on test; it stays the documented alternative, simpler and trained in a
second); ResNet-50 (+1.6 points over ResNet-18 for twice the time and size, below our 2-point bar; set aside on validation and cost, so it is not
scored on museum B, and it was only tested frozen, never fine-tuned); smaller images, which are the options for a slower kiosk
(112 px: 75.4% for 21.6 ms; 64 px: 70.2% for 10.5 ms); the CNNs and the dense network (20 or more points lower). A classical model (colour and edge features, random forest) was compared afterwards, once the final model was fixed: 58.5% on
museum B (ADR 0014). The language feature was not built at the time of this decision (ADR 0004; it came later, ADR 0005 and 0013).

**Consequences.**
- The choice was made on validation (480 images, about +/- 3.6 points) and the model was scored once on museum B (600 images, about
  +/- 2.9 points). All three seeds reach exactly 83.3%; the seeds are independent (checked), but the best epoch is a maximum over noisy
  epochs and hides the real spread, so no "std 0.0".
- Every model scored higher on museum B than on validation, so museum B looks easier (named artists only). Do not expect 86% in a real
  second museum.
- Weak points (P10, validation; on test the same pairs dominate): cityscape predicted as landscape (19% of the cityscapes), religious
  painting and portrait mixed up, portrait called still life. The true genre is the first or second choice for 95% of the validation images (a diagnostic: the test set is scored on the first choice only), but 29 of
  the 80 mistakes had a confidence above 0.8: many errors are probably ambiguous labels, not fixable by a better model. Cubism and
  Symbolism are the hardest styles.
- The ResNet works at 224 px and the CNNs at 64 px: pretraining and resolution each explain about half of the gap (P07).
- The first run downloads the ImageNet weights (45 MB) and needs the images in `data/` (not in Git, licence).
