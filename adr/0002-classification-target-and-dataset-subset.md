# 0002. Classify 5 genres, 600 images each, saved at 224 px

- **Status:** Accepted
- **Date:** 2026-10-20

**Context.** We pick 5 classes from WikiArt (81,444 paintings). Measured with the script's `--dry-run`: 5 genres
(portrait, landscape, cityscape, still_life, religious_painting) at 300 images per class need 5 shards (2.3 GB),
and 48% of the images have "Unknown Artist". 5 contrasted styles (Impressionism, Baroque, Cubism, Ukiyo_e, Pop_Art)
need 9 shards (4.2 GB) and 93% are "Unknown Artist". That label is the one thing the per-artist cap cannot limit, so
for styles we could not tell whether the model learns the style or the painter. At 600 per class the genres still
cost 5 shards (2.3 GB), with 54% unknown.

**Decision.** The classifier predicts the **genre** (the folder name, `label` in `metadata.csv`) from the image alone. The
artist, style and genre columns are metadata: the artist is used only to split museums A and B, and none of them is
ever an input to the model. We will use 5 genres (the 5 above), 600 images per class (3,000 total), `--max-per-artist 50`, seed 0, saved at
224 px (what pretrained ResNets expect; shrinking later is free, enlarging means a new download). Images stay out
of Git (`data/` is ignored). The exact command is in `data/wikiart_genre5/subset_info.json`.

**Rejected.** Styles (artist shortcut cannot be audited, twice the download); both labels (doubles the work).

**Consequences.** Genres overlap (a religious painting can be a portrait), so read the confusion matrix before
blaming the network. 600 per class is enough for a pretrained ResNet but modest for the dense baseline. The centre
crop to a square keeps a median 75% of each painting (worst case 28%), so wide or tall paintings lose their edges.
Download figures come from `make_wikiart_subset.py --dry-run`; figures about the downloaded images come from
the notebook `notebooks/vision/P01_explore_data.ipynb`.
