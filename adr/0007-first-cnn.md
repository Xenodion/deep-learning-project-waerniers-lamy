# 0007. Keep the first CNN as the reference for convolutional models: 3 blocks, 64 px

- **Status:** Accepted
- **Date:** 2026-10-22

**Context.** The dense baseline (ADR 0006) scores 39.0% on validation and overfits. The next question is what
convolutions bring. Setup, in `vision/P03_first_cnn.ipynb` (run `cnn_first_2026-10-21_19-54-39` in
`results/experiments.csv`): 64 x 64 images, three blocks of convolution (3 x 3 filters), ReLU and max pooling with 16, 32
and 64 filters, then an average over each feature map and one final layer. No batch norm, no dropout, Adam (learning rate
1e-3), batches of 64, 30 epochs, best epoch kept on validation. Same split and same data preparation as the baseline.
Measured, chance being 20%:

- 23,909 weights (the baseline has 393,989); one run takes about 2 minutes on a CPU;
- best validation accuracy 53.8% for seed 0; over 5 seeds the mean is 53.6% (std 0.8, from 52.7% to 54.8%), against 39.0%
  for the baseline;
- it **underfits**, unlike the baseline: train accuracy is 60% at epoch 30 against 54% on validation, the best epoch is the
  last one (30), the validation loss is still falling (1.14 at epoch 29), and validation accuracy was still rising
  (50.4% at epoch 20, 53.8% at epoch 30);
- per genre: portrait 75%, landscape 62%, cityscape 55%, still life 42%, religious painting 34%. The largest confusions
  are religious paintings predicted as portraits (36 of 96) and still lifes predicted as portraits (27 of 96).

**Decision.** We will keep this CNN, with this config, unchanged as the reference for the convolutional models, as ADR 0006
does for the dense network. Anything that tries to improve it is a separate experiment with its own name in `results/`,
compared with this one. Its test score is computed once, at the end (ADR 0003).

**Rejected.** Tuning it in place (we could no longer say what each change brings); stopping here (the curves show it has not
finished learning).

**Consequences.** Convolutions help: about 15 points more than the dense network with 16 times fewer weights, a gap far larger
than the seed spread (about 1 point). The result is still poor in absolute terms. The curves point to a model that is too
small or trained too briefly, but part of the ceiling may also come from the labels: genres overlap (ADR 0002), so some
errors may be label ambiguity. We have not checked that. Every score has about +/- 4.5 points of sampling uncertainty (480
validation images). The CNN has not been scored on museum B yet.

## What we might try next

One change at a time, each saved as its own run. Explore with `seeds=1` (about 2 minutes), then repeat the promising
ones with 5 seeds. Keep a change only if its 5-seed mean beats 53.6% by more than the seed spread (about 1 point). This list
is a set of ideas ordered by cost, not a result.

1. **Train longer** (`epochs`: 100). The best epoch was the last and the validation loss was still falling. About 7 minutes
   per run, estimated from 129 s for 30 epochs.
2. **Batch norm** (`batch_norm`: True). A short 3-epoch trial during development hinted that it learns faster, but it was not
   recorded, so it has to be measured properly.
3. **Wider or deeper blocks**, for example `conv_channels` [32, 64, 128] or four blocks. The network would still have far
   fewer weights than the dense baseline.
4. **Larger images** (`image_size`: 96 or 128). More detail for the model, at a higher cost per run (about 2.3 and 4 times
   the pixels of 64 px, not measured).
5. **Data augmentation** (random flips and crops), the usual cure for small datasets. It needs a change to `fit` (an
   optional transform step), the first change to existing code, and matters more once the model fits the training set.
6. **Dropout and weight decay.** Not needed while the model underfits; they become relevant once train accuracy is far above
   validation accuracy.
7. **Look at the errors**: which religious paintings are predicted as portraits, and whether the style explains it (ADR 0003
   notes that genre and style overlap). Done in a notebook, with no painting kept in the saved outputs (licence).
8. **The pretrained ResNet-18** (P06), the deliverable. Compared with this CNN, it separates what convolutions bring from what
   pretraining on ImageNet brings.
