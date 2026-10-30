# 0018. The final model resists light and angle, not loss of detail: the app must check the photo

- **Status:** Accepted
- **Date:** 2026-10-26

**Context.** Every score of the project comes from clean, cropped scans; ADR 0003 warned that "the real-world score will likely be
lower". `vision/P15_phone_robustness.ipynb` applies to every painting nine things a visitor's phone does, three levels each,
defined before any result (`utils.PERTURBATIONS`), and runs the final model (ADR 0012, unchanged). Validation first, then
museum B once; nothing is chosen. Measured (`results/robustness.csv`), accuracy at the strong level and drop from clean
(validation 83.3%, museum B 86.0%):

| perturbation (strong level) | validation | museum B | drop (validation / museum B) |
|---|---|---|---|
| warm light (red +35%, blue -35%) | 82.3% | 84.2% | 1.0 / 1.8 points |
| glare (white spot at 70%) | 80.4% | 84.3% | 2.9 / 1.7 |
| tilt (perspective 0.3) | 80.2% | 83.0% | 3.1 / 3.0 |
| dark (brightness x0.3) | 78.8% | 81.8% | 4.6 / 4.2 |
| rotation (20 degrees) | 65.6% | 68.7% | 17.7 / 17.3 |
| sensor noise (std 40) | 64.2% | 67.8% | 19.2 / 18.2 |
| far (painting fills 40% of the photo) | 63.1% | 63.2% | 20.2 / 22.8 |
| blur (radius 4) | 54.8% | 53.0% | 28.5 / 33.0 |
| JPEG (quality 5) | 49.0% | 50.3% | 34.4 / 35.7 |

**Decision.** The final model stays as it is. The guide must protect it from bad photos rather than hope it copes:
frame the painting on screen so it fills the photo, send photos at a normal JPEG quality, retake blurred photos, and show two
genres when the model is unsure (ADR 0020). These are recommendations for the app; none of them is built here.

**Rejected.** Retraining the last block with these perturbations as augmentation (the cure inside the model, a new final model
and a new museum B score; left as the next step); tuning the levels after seeing the results.

**Consequences.**
- The 86.0% of museum B is a clean-scan score. Mild levels cost about 4 points or less; the medium levels of the five harmful
  perturbations cost 5 to 12.5 points; real photos combine several perturbations, so the real loss may be larger.
- Each harmful perturbation pushes predictions towards one genre (strong JPEG: 54% cityscape; strong blur: 42% landscape; a
  distant painting: 38% still life): a bad photo gives a consistent wrong story, not random noise.
- Limits: synthetic perturbations, one at a time, levels not matched to real phones, no real phone photo tested.
