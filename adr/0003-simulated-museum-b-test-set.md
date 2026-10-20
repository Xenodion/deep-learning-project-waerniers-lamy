# 0003. Train in museum A, test once in a simulated museum B

- **Status:** Accepted
- **Date:** 2026-10-20

**Context.** The real use is a guide dropped into a museum it has never seen. A random test split would be
statistically identical to the training data and would not show that.

There is also an artist shortcut. The model never sees the artist's name, but pixels carry a painter's signature
(palette, brushstrokes, favourite subjects), and most painters specialise. In our 3,000 images the most frequent
named artist is Braque in still_life (44 images), Cortès in cityscape (36), Levitan in landscape (27), Perugino in
religious_painting (26) and Modigliani in portrait (18). If the same painter is in train and test, the model can score
well by thinking "this is Braque, so still life" without learning what a still life is. The score would be inflated,
and in a new museum, full of unseen painters, accuracy would drop. Measured: knowing only the artist predicts
the genre of 70% of the named-artist images (leave-one-out majority vote, chance is 20%); knowing only the style
predicts 49% of all images.

WikiArt has no museum field, but `metadata.csv` has the `artist`. Of the 3,000 images, 1,393 have a named artist
(102 artists) and 1,607 are "Unknown Artist", a catch-all for many different painters that cannot be told apart. All figures about the images come from
`python src/deep_learning_project_waerniers_lamy/vision/explore_data.py data/wikiart_genre5`. The subset has no duplicated
image.

**Decision.** We will split the images into two fictional museums:

- **Museum B (test only)** takes 44 of the 102 named artists, chosen by a seeded search so that every genre has
  exactly 120 images. No "Unknown Artist" image goes in B.
- **Museum A (train and validation)** takes the other 58 named artists and all "Unknown Artist" images. Inside A,
  train and validation are an 80/20 random cut by image, with the same class proportions.

| Set | cityscape | landscape | portrait | religious_painting | still_life | Total | Named-artist images | "Unknown Artist" images |
|---|---|---|---|---|---|---|---|---|
| Train (A) | 384 | 384 | 384 | 384 | 384 | 1,920 | 632 (58 artists) | 1,288 |
| Validation (A) | 96 | 96 | 96 | 96 | 96 | 480 | 161 (47 artists) | 319 |
| **Test (B)** | **120** | **120** | **120** | **120** | **120** | **600** | **600 (44 artists)** | **0** |

The rules, in plain words:

- A named artist is in A or in B, never both: 0 named artists are shared between test and train or validation.
- Inside A, train and validation can share artists (all 47 named validation artists are also in train).
- "Unknown Artist" stays in A because we cannot tell which painters are the same person: in B it could leak.
- Every set has 20% of each genre.
- B is used only at the end, to score each finished model once (dense network, ResNet, classical baseline). Early
  stopping, tuning, diagnosis and model choice use validation(A), never B. If B reveals a problem, we report it as a
  finding; fixing the model and re-testing on B would make B a tuning set, and we would say so.
- The split is saved as a file (image id to train, validation or test) made by a committed script. The figures above
  are its output (seed 0, identical on every run):
  `python src/deep_learning_project_waerniers_lamy/vision/make_split.py data/wikiart_genre5`.

**Rejected.** Random test split (no shift to measure); using B for model selection (it would stop being a test).

**Consequences.**

- The gap between validation(A) and test(B) measures the "new museum" effect. If the model learned the genre, the test
  score stays close to validation. If it learned painters, it drops, and that is a diagnosis we can report.
- Validation(A) is optimistic: it shares artists with train. Only B tests unseen painters.
- B is not a real museum. It has only named, mostly famous artists, while two thirds of A is "Unknown Artist", and
  every image is a cropped scan, with none of the glare of a phone photo. The real-world score will likely be lower.
- B also shifts the style mix a little (for example B's religious paintings lean Northern and High Renaissance, A's
  lean Early Renaissance and Baroque), and genre and style overlap, so the model may use style as a shortcut too.
- With 600 images, a 95% interval on an accuracy near 90% is about +/- 2.4 points: do not claim smaller differences.
- Which artists land in B affects the score. If time allows, repeat with a few seeds and report the spread.
