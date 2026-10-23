# 0011. Try a few cheap ResNet improvements before choosing the final model (P08)

- **Status:** Accepted for the improvement round; the experiments below are a first plan.
- **Date:** 2026-10-22

**Context.** Two sets of results, after ADR 0010:

- **P07** (`vision/P07_resnet_resolution_control.ipynb`): the frozen ResNet-18 scores 70.2%, 75.4% and 80.3% on validation at 64, 112
  and 224 px (5 seeds each), against 58.4% for our best CNN at 64 px. At equal resolution, pretraining is worth about 12 points; going
  from 64 to 224 px adds about 10 more. This is approximate, because we did not test our CNN at 224 px.
- **The test scores** of the earlier models (museum B, seed-0 models, scored once): dense 41.8%, first CNN 58.7%, `cnn_longer` 63.7%,
  `cnn_deeper` 60.5%, ResNet frozen 84.5%, ResNet with `layer4` trained 86.0%. The ladder is the same as on validation. The test score
  is higher than the validation score for all six models (by 1.5 to 5 points), so museum B looks easier than the validation set. This
  does not show that the models generalise better to new artists.

The frozen ResNet has 93% train accuracy against about 80% on validation: generalisation, not capacity, is the limit (training the last
block reaches 100% train accuracy for only 2 to 3 more points). Part of the remaining error may be label ambiguity (religious
paintings against portraits, cityscapes against landscapes); we have not checked.

**Decision.** We will do one time-boxed round of cheap improvements, in `vision/P08_resnet_improvements.ipynb`. All of them keep the
frozen network, so the features are computed once and each experiment takes minutes:
1. **Features from several layers:** pool the outputs of layer2, layer3 and layer4 and combine them (genres depend on texture, which
   the middle layers hold).
2. **Data augmentation,** as precomputed augmented copies of each training image (flip, crop, colour), plus averaging the prediction
   over an image and its mirror at evaluation time.
3. **A frozen ResNet-50** in place of ResNet-18. ADR 0009 rejected it on expectation only; we never tested it.

Rules: each experiment uses 5 seeds and is judged on the smoother measures (ADR 0008); a change counts only if its 5-seed mean beats
the frozen ResNet-18 (80.3%) by clearly more than the noise (about 2 points); the test set is not used to choose (ADR 0003). If
nothing helps clearly, the frozen ResNet-18 remains our reference ResNet.

**Rejected.** Resolution above 224 px, or keeping the whole painting instead of the centre crop (a new download and about 4 times the
compute at 448 px, against the kiosk-cost question); more training images (changes the splits and breaks comparisons); training more
layers (`layer4` gave 2 to 3 points for 8.4 million trainable weights).

**Consequences.** Each experiment should cost minutes of CPU (estimated, not measured). The deadline is 30 October 2026, and the cost
comparison, `run_final.py` and the presentation must not wait: if an experiment does not help, we stop there. ResNet-50 needs about
2.3 times the compute of ResNet-18 (published figures: 4.1 against 1.8 GFLOPs), so any gain from it has to be weighed against that
cost on a kiosk CPU.

**Outcome (P08, 5 seeds each, validation).** None of the changes left the +/- 2 point band around the frozen ResNet-18 (80.3%, std 0.5):
several layers 80.6% (std 0.4), augmentation 80.8% (std 0.5), flip averaging +0.1 point on top of it, frozen ResNet-50 81.9% (std 0.3).
The ResNet-50 is the only consistent one (all 5 seeds between 81.7% and 82.3%, against 79.8% to 81.0% for the reference), but a gain of
1.6 points is below our threshold, is about 8 images out of 480, and costs about twice the compute (159 s against 77 s for the frozen
features, in line with the 2.3 times expected from the published figures). The
combination was not run. The frozen ResNet-18 remains our reference ResNet; the final model is chosen later, once the cost per image is measured. Three different ways of adding information
end between 80% and 82%, which suggests the limit is now the data and the labels (genres overlap), not the network; we did not look at
the errors.
