# 0014. A classical baseline: handcrafted features and a random forest

- **Status:** Accepted (amends [0004](0004-focus-on-genre-classifier.md))
- **Date:** 2026-10-25

**Context.** The director asks whether a neural network is worth it. ADR 0004 planned a classical baseline, then dropped it
("this is a deep-learning course"), so until now the question was only answered among neural networks, which is a weaker
answer than the brief asks for. The baseline is cheap (about 5 minutes, no GPU), so we added it back, with the same rules as the
networks ([0003](0003-simulated-museum-b-test-set.md)): chosen on the validation images of museum A, scored once on museum B.

**Decision.** The baseline is the notebook `vision/P11_classical_baseline.ipynb` (museum B was first scored on 2026-10-25 by a script
that became this notebook: same steps, seeded, and the notebook checks that museum B gives the recorded 58.5%, choosing nothing from
it). Each painting becomes 1,802 handcrafted numbers, fixed before any
result: colour histograms (hue 16 bins, saturation 8, value 8) with the mean and spread of each, and HOG (histograms of
edge orientations, 16 px cells) on a 128 px grayscale copy. Five models were trained on the 1,920 training paintings, and the best
on validation was kept; museum B was scored once with it.

| model | validation (480) |
|---|---|
| logistic regression, C = 0.001 / 0.01 / 0.1 / 1 | 57.7% / 54.2% / 52.3% / 51.7% |
| **random forest, 300 trees** | **60.0%** |

The ladder, all scored on museum B (`results/experiments.csv`, `results/classical_baseline.csv`; costs from one image at a time
on one CPU thread, features included; the networks' costs are P09's):

| model | museum B | time per photo | size |
|---|---|---|---|
| dense network, 32 px | 41.8% | 0.1 ms | 1.6 MB |
| **classical: colour + edges, random forest** | **58.5%** | **43 ms** | **22 MB** |
| small CNN, 64 px | 63.7% | 1.7 ms | 0.1 MB |
| ResNet-18 frozen, 224 px | 84.5% | 70 ms | 45 MB |
| ResNet-18, last block trained, 224 px | 86.0% | 70 ms | 45 MB |

**Rejected.** Tuning the features until the forest improves (we would be spending museum B's honesty on the weaker side; the
point is a fair, unfussed reference, not the best possible classical model); features taken from a pretrained network with a
logistic regression on top (a different question: that is the frozen ResNet of ADR 0009).

**Consequences.**
- **The answer to the director.** Yes, a neural network is worth it, but not every one: the fine-tuned ResNet-18 is +27.5 points
  over the classical model for +28 ms and twice the size, whereas the dense network is 17 points *below* it. What pays is
  pretraining, not "being a network". The small CNN and the forest are close: on validation they tie (60.0% against 58.8 to 60.8%),
  on museum B the CNN is 5 points ahead, a gap near the noise (about +/- 2.9 points on 600 images for each).
- **The small CNN dominates the forest on cost**: 25 times faster and 200 times smaller, with a slightly higher accuracy. The forest is slow
  because 300 trees on 1,802 numbers is not cheap, and its features cost 12 ms.
- **The classical model confuses the same genres as the networks, more so**: religious paintings (40% right) are mixed with portraits
  and still lifes, the genre the networks also find hardest.
- **Limits.** One feature set and one split, no seeds to average, a forest we did not tune beyond its size. A better-engineered
  classical model could gain a few points; we did not look for them. P09 was not re-run: the cost comparison with the networks is
  in P11 (section 9, with P09's times and sizes) and in this ADR.
