# 0016. The final feed-forward network: the P02 network with mirrored training copies

- **Status:** Accepted
- **Date:** 2026-10-26

**Context.** The first deliverable of the brief is a feed-forward network. P02 kept it unchanged as the baseline (ADR 0006:
39.0% on validation, 98% on train), and ADR 0015 (P12) showed that its training is healthy and that it memorises: it
learns shuffled labels by heart. ADR 0006 left its improvement to separate experiments. `vision/P13_dense_regularisation.ipynb`
runs them with a rule written before the runs: each change alone, 5 seeds, compared with the baseline re-run on the same
computer; a change counts if its mean beats that re-run by more than 2 points (the bar of ADR 0011); only changes that count
are combined; museum B once, for the final network only. Measured (`results/experiments.csv`, names `dense_*` of 2026-10-26):

| variant (one change) | validation, mean of 5 seeds (std) | gain | seed 0: train - validation, last epoch |
|---|---|---|---|
| baseline re-run (P02) | 39.0% (0.8) | | 60 points |
| weight decay 1e-4 / 1e-3 / 1e-2 | 39.2% / 39.3% / 39.3% | +0.3 to +0.4 | 57 to 58 points |
| weight decay 1e-1 | 38.4% (0.9) | -0.5 | 23 points |
| dropout 0.2 / 0.5 | 39.3% / 39.2% | +0.4 / +0.2 | 61 / 54 points |
| 16 px / 64 px images | 39.1% / 39.0% | +0.2 / 0.0 | 58 / 61 points |
| two hidden layers of 128 | 40.2% (0.3) | +1.3 | 61 points |
| learning rate 1e-4 | 39.0% (0.9) | +0.1 | 50 points |
| **mirrored copies of the training images** | **42.5% (0.6)** | **+3.6** | 54 points |

**Decision.** The final feed-forward network is **`dense_flip`**: the P02 network (32 px, one hidden layer of 128 ReLUs, Adam
at 1e-3, batches of 64, 30 epochs, best epoch kept on validation) trained on the 1,920 training images plus their mirror
images. It is the only change that counts, so there is nothing to combine. Scored once on museum B: **41.0%** (loss 2.598),
against 41.8% (loss 1.577) for the P02 baseline. The P02 baseline stays the reference of ADR 0006.

**Rejected.** Weight decay, dropout, other image sizes, a second layer, a lower learning rate (none beats the bar; the second
layer comes closest, +1.3); combining changes below the bar (rule fixed before the runs); scoring the re-run baseline or the
other variants on museum B (they were not chosen; museum B would become a selection set).

**Consequences.**
- **Overfitting was the symptom, not the limit.** Weight decay at 1e-1 removes most of it (gap from 60 to 23 points) and
  validation does not rise. A dense network ties each weight to one pixel position; regularisation cannot change that.
  Convolutions (P03, weight sharing) and pretraining (P06) do, which is where the project's gains came from.
- **The gain of the mirrored copies does not show on museum B:** 41.0% against 41.8%, a difference of 5 paintings out of 600,
  inside the noise (about +/- 4 points). The comparison is imperfect: the P02 score comes from the seed-0 model of another
  computer, and the re-run baseline was not scored on museum B. `dense_flip` is the first model of the project with a test score
  below its validation score (seed 0: 43.5%), where every earlier model scored 1.5 to 5 points higher on museum B.
- **Its test loss is much higher** (2.60 against 1.58): its best epoch on validation accuracy is 22, where the network is far
  more confident than at the baseline's epoch 8. Choosing the epoch on validation accuracy favours late, over-confident epochs.
- **For the director (ADR 0014):** the best feed-forward network stays near 40%, 17 points below the classical model (58.5%)
  and 45 below the fine-tuned ResNet-18 (86.0%). A plain feed-forward network is not worth deploying.
- **Limits:** one family (dense, ReLU, Adam, 30 epochs), each change alone with a few values, mirrors as the only augmentation
  (no crops or colour changes), best epochs on 480 validation images. P12 found a 1e-4 learning rate promising on one seed;
  with 5 seeds it is exactly the baseline, which is the reason for the 5-seed rule.
