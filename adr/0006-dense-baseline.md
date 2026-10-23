# 0006. Keep the dense baseline simple and unchanged: 32 px, one hidden layer of 128

- **Status:** Accepted (confirmed by both of us; the baseline stays as it is, and improving it is discussed separately)
- **Date:** 2026-10-21

**Context.** Every other model (regularised dense networks, the CNNs, the pretrained ResNet) needs a
reference to be compared with. Setup, in `vision/P02_dense_baseline.ipynb`: images shrunk to 32 x 32 and flattened
(3,072 numbers), standardised with the train statistics, one hidden layer of 128 ReLU units (393,989 weights), Adam
(learning rate 1e-3), batches of 64, 30 epochs, best epoch kept on validation. Data: 1,920 train and 480 validation
images from museum A. Measured (`results/experiments.csv`, run `dense_baseline_2026-10-21_17-29-52`), chance being 20%:

- best validation accuracy 38.5% for seed 0 (epoch 8); over 5 seeds the mean is 39.0%, from 37.9% to 40.2%;
- it overfits: train accuracy 98% at epoch 30 while validation stays around 37%, and the validation loss rises from
  1.57 (epoch 1) to 3.12 (epoch 30);
- per genre (seed 0): landscape 52%, portrait 47%, cityscape 39%, religious painting 35%, still life 20%.

**Decision.** We will keep this model, with these settings, unchanged as the baseline, even though it is poor.
Anything that tries to improve it (weight decay, dropout, another image size or width) is a separate experiment with its
own name in `results/`, compared with this one. Its test score is computed once, at the end (ADR 0003).

**Rejected.** Tuning the baseline until it looks better (we could no longer say what each later change brings); a bigger
network (394k weights are already a lot for 1,920 images).

**Consequences.** A low bar is honest: it shows what a network that ignores the structure of an image can do, and its
overfitting is the problem the next experiments address. 32 px, 128 units and 30 epochs were not compared with
alternatives, so a better dense network may exist: say so in the report. Scores are the best of 30 epochs on 480 images,
so slightly optimistic, with about +/- 4 points of sampling uncertainty (the seed adds only about 1 point): do not claim
smaller differences. The baseline has not been scored on museum B yet.
