# 0020. Show two genres when the confidence is below 0.86

- **Status:** Accepted
- **Date:** 2026-10-26

**Context.** P10 found that 29 of the final model's 80 validation mistakes were made with a confidence above 0.8, and that the true
genre is its first or second choice for 95% of the paintings. `vision/P17_confidence.ipynb` measures calibration (expected
calibration error, ECE, 15 bins) and tries a rule that shows two genres when the model is unsure. Rules written before the
runs: temperature T fitted on validation, its validation ECE measured by 2-fold cross-fitting; the threshold is the lowest
calibrated confidence at which lone answers are right at least 95% of the time on validation; museum B once, for measures
only (the predictions were scored in ADR 0012). Measured (`results/confidence.json`):

| | validation | museum B |
|---|---|---|
| ECE before / after temperature scaling (T = 1.23) | 0.061 / 0.068 (cross-fitted) | 0.059 / **0.044** |
| loss before / after | 0.475 / 0.463 | 0.423 / 0.409 |
| answered with one genre (confidence at least 0.86) | 60.0%, right 95.1% | 68.7%, right **94.2%** |
| the others: true genre among the two shown | 89.6% | 94.1% |
| **right genre on the screen** | 92.9% (one answer: 83.3%) | **94.2%** (one answer: 86.0%) |

**Decision.** The guide divides the scores by T = 1.23 before the softmax and shows one genre when the top probability is at least
0.86, otherwise its two best genres ("a portrait or a religious painting"). T and the threshold are in `results/confidence.json`.

**Rejected.** Claiming temperature scaling as a repair (it does not lower the cross-fitted validation ECE; it is kept because it
is free and helps a little on museum B); a threshold chosen on museum B; abstaining without an answer (the top-2 is right for 90
to 94% of the uncertain paintings, so two genres is more useful than none).

**Consequences.**
- On museum B the right genre is shown for 94.2% of the paintings instead of 86.0%, and one genre alone is right 94.2% of the
  time (just under the 95% target, inside the noise of about 400 paintings).
- On bad photos (ADR 0018) the guide answers alone far less often (7 to 23% of the paintings at the strong level of blur, JPEG
  and distance, against 60%), but about one lone answer in five is still wrong on blur and JPEG: the check of the photo itself is
  still needed.
- Not built: the demo still shows one genre. No out-of-distribution test: a photo of a sculpture or a wall still gets one or two
  genres. The 95% target is our choice, not the museum's.
