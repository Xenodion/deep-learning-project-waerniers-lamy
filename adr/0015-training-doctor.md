# 0015. Diagnose training from inside the network, and keep the P02 baseline as is

- **Status:** Accepted
- **Date:** 2026-10-26

**Context.** The brief asks for "training monitoring and failure diagnosis". Until now the project read only learning curves
(loss and accuracy, P02 to P08): they show *that* a model trains badly, rarely *why*. P02 diagnosed overfitting (98% train,
39% validation) without checking whether the training itself was healthy. The course (TP05, TP06) diagnoses with the
gradient per layer, dead ReLUs, the initialisation and the scale of the inputs.

**Decision.** We will monitor the inside of the network, with `fit(..., monitor=True)` in `vision/utils.py`. Per epoch it
records the loss of every batch and, per layer, the gradient norm, the update / weight ratio, the spread of the activations
and the share of dead or saturated units. It does not change the training: the curves and weights of the dense baseline are
bit-identical with it on or off. `fit` also gets plain SGD (`optimizer="sgd"`) and gradient clipping (`clip_grad_norm`); Adam
stays the default, so no earlier run changes. `vision/P12_training_doctor.ipynb` makes the dense network fail on purpose,
one change at a time, and checks each cure. Measured (museum A only, one seed per patient, nothing saved, museum B never
loaded):

| patient | what the monitoring shows | cure, measured |
|---|---|---|
| the P02 baseline | healthy inside (update ratio 4e-3, 0% dead units), yet it reaches 100% train accuracy on **shuffled labels** too (validation 19%) | not a training failure: memorisation, to be treated by regularisation |
| learning rate too high (Adam, 1) | batch loss up to 77,562; update ratio 3.4 at epoch 1 | 1e-3 or 1e-4: ratio near 1e-3 |
| learning rate too low (1e-5) | update ratio 5e-4 down to 2e-4 | a larger learning rate |
| raw 0-255 pixels (SGD) | first-layer gradient 100 to 60,000 times larger, 95 to 100% dead ReLUs, loss stuck at ln 5 | standardised inputs (34.4% at the same lr); leaky ReLU and clipping only mask it |
| 6 sigmoid layers (SGD) | first-layer gradient 1e-4 of the last (bound 0.25^6 = 2.4e-4), loss stuck at ln 5 | ReLU or tanh |
| SGD at lr 1, 6 ReLU layers | dead ReLUs rise first (30% to 79% of the last layer), then NaN at step 758 | smaller steps; clipping at norm 1 (no NaN, 41.0%) |
| 10 ReLU layers, bad initialisation | spread of the outputs /9 per layer (std 0.01), x11 (std 1, NaN), x0.41 (PyTorch default): none trains | He initialisation: spread stays near 0.8, trains (40.0%) |

**Rejected.** Diagnosing on the CNNs or the ResNet (minutes per run instead of a second, and the failures are the same
mechanisms); saving the patients in `results/experiments.csv` (they are diagnoses, not candidate models); tuning each patient
until its cure "wins" (one seed each: the scores only tell "trains" from "does not train").

**Consequences.**
- **The P02 baseline stays as it is (ADR 0006).** Every internal measure is healthy, so its gap between train and validation
  is a generalisation problem, not a bug. The next step treats it with regularisation (weight decay, dropout, smaller
  inputs, augmentation), as separate runs compared with this baseline.
- **Comparisons must be made on one machine.** On a second computer the dense baseline matched the recorded run to 5
  decimals for 20 epochs, then drifted through float rounding (seed 0: best 38.96% at epoch 21 against the recorded 38.54%
  at epoch 8, 2 images out of 480). New variants are compared with the baseline re-run on the same machine.
- **Adam hides failures.** It kept every ReLU alive even at a learning rate of 1, where plain SGD kills them. This supports the
  choice of Adam in every model, and means a run that trains with Adam can still have a step size far from 1e-3.
- **A lead, not a result:** with Adam at lr 1e-4 the baseline reached 40.0% (one seed, still rising at epoch 29), against
  39.0% at 1e-3. It goes into the next round with 5 seeds.
- The settings of each patient were first tried in a scratch run to find a clear example of each failure; the healthy
  references (baseline, He initialisation, standardised inputs) were not tuned.
