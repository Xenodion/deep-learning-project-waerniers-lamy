# 0008. Train the CNN for 100 epochs without batch norm, and try a deeper CNN next (P05)

- **Status:** Proposed (to confirm: `cnn_longer` becomes the reference, batch norm is set aside)
- **Date:** 2026-10-22

**Context.** ADR 0007 found that the first CNN underfits and listed what to try. P04
(`vision/P04_cnn_longer_batchnorm.ipynb`) tried the first two ideas, one at a time and together, with the same data, split and
seed (0), one seed per run. Measured, in `results/experiments.csv` (each run is logged twice because the notebook ran twice;
the numbers are identical). "Swing" is how much the validation accuracy moves from one epoch to the next over the last 10
epochs.

| run | epochs | batch norm | best val acc (epoch) | last epoch | mean of last 10 | at lowest val loss | train acc, last | swing |
|---|---|---|---|---|---|---|---|---|
| `cnn_first` | 30 | no | 53.8% (30) | 53.8% | 51.1%* | 52.1% | 59.7% | 1.9 pts |
| `cnn_longer` | 100 | no | 60.8% (95) | 57.3% | 58.8% | 58.3% | 68.2% | 2.2 pts |
| `cnn_batchnorm` | 30 | yes | 59.4% (21) | 51.7% | 52.2% | 57.9% | 60.9% | 6.3 pts |
| `cnn_longer_batchnorm` | 100 | yes | 59.8% (54) | 57.5% | 52.0% | 58.1% | 74.6% | 8.6 pts |

*still rising during those epochs, so it understates the model.

- **Training longer helps.** The curve is steady and `cnn_longer` is 5 to 7 points above the first CNN on every smoother
  measure. Its validation loss is lowest at epoch 86 (1.05) and then rises slightly: the gap between train and validation is
  11 points at the end.
- **Batch norm learns faster at first** (epoch 5: 50.0% on validation against 37.9%), **but makes the validation score
  erratic:** it jumps 6 to 9 points from one epoch to the next, so its best epochs (59 to 60%) are lucky peaks and a typical
  epoch (about 52%) is no better than the first CNN. Combined with 100 epochs it overfits the most: train 74.6% against 57.5% on
  validation, and a validation loss that goes from 1.01 (epoch 42) up to 1.34.
- All scores come from one seed and have about +/- 4.4 points of sampling uncertainty (480 validation images); "best epoch"
  is optimistic because it is the best of many noisy epochs.
- **The CNN still cannot fit its own training images** (68% train accuracy after 100 epochs). With three blocks, each unit
  before the final averaging sees only 22 x 22 pixels of the 64 x 64 image, and the averaging then discards where it was seen.
  This is a hypothesis, not tested.

**Decision.**
1. `cnn_longer` (the same CNN, 100 epochs, no batch norm) is the reference for the next CNN experiments. The first CNN stays
   the 30-epoch reference of ADR 0007.
2. Set batch norm aside. It is not ruled out for good, but here it gives no better model and a very unstable validation score.
3. The next experiment is a **deeper CNN**, in `vision/P05_cnn_deeper.ipynb`: a fourth convolution block, `conv_channels`
   [16, 32, 64, 128], everything else as `cnn_longer` (100 epochs, no batch norm, seed 0), one seed first. A fourth block raises
   what each unit sees from 22 x 22 to 46 x 46 pixels. It has 98,085 weights (a quarter of the dense baseline) and takes about
   7 minutes per run (estimated from 2 measured epochs).

**Rejected.** Combining longer training and batch norm (no better than either alone, and the most overfit); picking a winner
among the P04 variants from their best epochs alone (too noisy).

**Consequences.** P05 is judged on the smoother measures (last epoch, mean of the last 10 epochs, score at the lowest
validation loss), not on the best epoch alone. A deeper network overfits more and the gap is already 11 points: if the gap
grows without validation gaining, the next ideas are dropout, weight decay and data augmentation (ADR 0007). If the deeper CNN
beats `cnn_longer` by clearly more than the epoch-to-epoch swing (about 2 points), we confirm it with 5 seeds. If not, we stop
tuning the CNN and move to the pretrained ResNet, a deliverable that is not started yet (deadline: 30 October 2026). None of
these CNN variants has been repeated over several seeds or scored on museum B.
