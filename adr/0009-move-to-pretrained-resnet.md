# 0009. Stop tuning the small CNN and move to a pretrained ResNet-18 (P06)

- **Status:** Accepted for the move to the ResNet; the P06 plan below is a first plan, to adjust with the results.
- **Date:** 2026-10-22

**Context.** ADR 0008 said to try a fourth convolution block (P05, `vision/P05_cnn_deeper.ipynb`) and, if it did not clearly beat
`cnn_longer`, to move to the pretrained ResNet. Measured (`results/experiments.csv`), 100 epochs each:

| run | weights | validation accuracy | train accuracy, last | validation loss, lowest to last |
|---|---|---|---|---|
| `cnn_longer` (3 blocks), seed 0 | 23,909 | 60.8% best epoch, 58.8% mean of the last 10 | 68.2% | 1.05 to 1.10 |
| `cnn_deeper` (4 blocks), seed 0 | 98,085 | 59.0% best epoch, 56.8% mean of the last 10 | 99.7% | 1.11 to 2.47 |
| `cnn_deeper`, 5 seeds | 98,085 | 58.4% mean (std 1.1; 56.5% to 59.2%) | | |

The deeper CNN fits its training images almost perfectly but is no better on validation: the gap between train and validation is
44 points. More depth cannot fix that; the limit is the amount of data (1,920 training images), not the size of the network. The
best from-scratch scores are about 58 to 59% (chance is 20%), with about +/- 4.4 points of sampling uncertainty on 480 validation
images. A pretrained ResNet is also a deliverable of the brief.

**Decision.** We will move to a pretrained ResNet-18 (torchvision, ImageNet weights), implemented in `vision/P06_resnet_transfer.ipynb`.
The small CNN work stops at `cnn_longer` and `cnn_deeper`; dropout, weight decay and augmentation stay possible later.

First plan for P06:
1. Same data and split (ADRs 0002 and 0003), at 224 px, standardised with the **ImageNet mean and std** (what the pretrained
   network expects), not with our own statistics.
2. **Stage 1, feature extraction:** the whole ResNet stays frozen. We compute the 512 features of each image once and train only a
   new final layer (512 to 5) with the existing `fit`. After the features are computed, a run takes seconds, so 5 seeds are cheap.
3. **Stage 2, fine-tuning:** we also train the last block (`layer4`) with the new final layer, at a small learning rate. The earlier
   layers stay frozen, so their outputs are also computed once.
4. No data augmentation for now (it needs a change to `fit`, and the cached features rely on its absence).
5. Validation scores go into `experiments.csv` next to the other models. The test set is scored once, at the end, on the model we
   choose (ADR 0003).

**Rejected.** Tuning the small CNN further (the bottleneck is the data, not the depth); ResNet-50 (about twice the compute of
ResNet-18, little expected benefit with 1,920 images); AlexNet and ViT (heavier or data-hungry, no advantage here). A MobileNet
stays an optional extra for the CPU-speed question.

**Consequences.** `torchvision` becomes a dependency, and the ImageNet weights (about 45 MB) are downloaded the first time. Two
caveats for the report. The ResNet works at 224 px while our CNNs used 64 px, so a gain would mix the effect of pretraining and
the effect of resolution. And ImageNet contains photographs, not paintings: whether its features transfer to painting genres is
what P06 measures. We expect a clear gain over 58 to 59%, but that is an expectation, not a result. Still to do after P06: the
comparison for the director (neural networks only), the final script and the presentation (deadline: 30 October 2026).
