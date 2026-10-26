# 0022. The final model looks at the subject, not at shortcuts (Grad-CAM)

- **Status:** Accepted
- **Date:** 2026-10-26

**Context.** Accuracy says how often the final model is right, not from what. It could use shortcuts: the frame, a signature in a
corner, the dark background of old paintings. `vision/P19_gradcam.ipynb` computes Grad-CAM from PyTorch hooks on `layer4` (7 x 7
cells, each map normalised to sum to 1) for the 480 validation paintings. No painting is shown (licence): every result is a
statistic over many paintings; a flag, off by default, draws a few.

**Decision.** We report these maps as the evidence of what the final model uses:

| | share of the decision from the outer ring (49% of the surface) | from the 3 x 3 centre (18%) |
|---|---|---|
| correct predictions (400) | 30.5% | 29.7% |
| confident mistakes, above 0.8 (29) | 30.2% | 30.2% |
| other mistakes (51) | 34.9% | 25.9% |

- Each genre is decided where expected: a horizontal band in the lower half (ground, buildings, horizon) for landscapes and
  cityscapes; the upper centre, where faces are, for portraits and religious paintings; the centre for still lifes.
- **The maps are faithful:** painting grey the 10 cells a map ranks first drops the accuracy from 83.3% to 64.2% (the genre
  changes for 35% of the paintings); 10 random cells, to 72.5%; the 10 cells it ranks last, to 81.7%.

**Rejected.** A Grad-CAM library (built from hooks, in the spirit of the course); showing paintings with their maps in the committed
notebook (licence); museum B (nothing to choose, validation is enough).

**Consequences.**
- No frame or corner shortcut: the edges weigh less than their surface.
- The confident mistakes are looked at like the right answers, and two thirds of them are the cityscape / landscape and religious
  painting / portrait pairs: the model looks at the right part of the painting and reads an ambiguous subject. This supports the
  reading of ADR 0012 that most remaining errors are label ambiguity.
- Limits: 7 x 7 cells of 32 pixels are coarse and show where, not what; validation only; no map was checked by eye in the
  committed notebook.
