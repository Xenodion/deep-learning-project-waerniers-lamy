# 0017. At equal inputs a feed-forward network ties with the forest: the features decide

- **Status:** Accepted
- **Date:** 2026-10-26

**Context.** On raw pixels the feed-forward network stays near 40% whatever the cure (ADR 0016). The director's question "is a
neural network worth it?" (ADR 0014) compared models that also differ by their inputs: a dense network on pixels, a random
forest on handcrafted colour and edge features, a ResNet on its own learnt features. `vision/P14_feedforward_heads.ipynb`
gives a feed-forward network the inputs of the two other models. Rules written before the runs: 5 seeds per network; on the
handcrafted features, eight networks fixed in advance, the best on validation scored once on museum B; on the frozen ResNet-18
features, a hidden-layer head counts only if it beats the re-run linear head by more than 2 points (ADR 0011). Measured
(`results/experiments.csv`, names `features_*`, `resnet18_head_rerun`, `resnet18_mlp_head*`):

| inputs | model | validation | museum B |
|---|---|---|---|
| handcrafted, 1,802 numbers (P11) | random forest, 300 trees (P11), 5 random states | 60.7% (57.7% to 63.7%) | 58.5% (state 0) |
| handcrafted | feed-forward, no hidden layer | 57.5% | |
| handcrafted | **feed-forward, 256 ReLUs, dropout 0.5, weight decay 1e-2** (best of 8) | **60.5%** | **62.7%** |
| frozen ResNet-18, 512 numbers | linear head (P06), re-run | 80.3% | 84.5% (P06) |
| frozen ResNet-18 | 256 ReLUs, dropout 0.5 | 80.8% (+0.5) | not scored |

**Decision.** The feed-forward network on the handcrafted features is recorded as the classical baseline's neural counterpart
(`features_mlp256_drop_wd`), next to the forest. The ResNet keeps its linear head; the final model (ADR 0012) does not change.

**Rejected.** A hidden-layer head on the ResNet (+0.5 points for 52 times the trained weights); tuning the forest or the
feature set (ADR 0014: a fair reference, not a competitor); scoring the other seven feature networks on museum B.

**Consequences.**
- **The answer to the director is sharper:** at equal inputs the feed-forward network and the forest tie on validation, and their
  museum B gap (62.7% against 58.5%) is inside the noise (about +/- 4 points each). The same feed-forward network goes from 41%
  on pixels to 63% on handcrafted features to 80% on pretrained features. What pays is the features, and the best ones were
  learnt by a network on ImageNet.
- **P11's forest is noisier than one run showed:** std 2.4 points across random states on validation. Random state 0 gives
  60.4% on this computer against 60.0% recorded (2 images).
- **The ResNet head is not the limit:** a hidden layer adds nothing, in line with ADR 0011 (several layers, augmentation and
  ResNet-50 all within 2 points).
- The feature networks memorise within 3 to 6 epochs (99.7% train accuracy for the chosen one); 50 epochs were more than enough.
