# 0002. Classify 5 artwork genres, 300 images each, downloaded once at 224 px

- **Status:** Proposed (to confirm together before downloading)
- **Date:** 2026-10-19
- **Deciders:** Waerniers, Lamy
- **Milestone:** m1

## Context

The brief says the guide recognises a painting's "style (Impressionism, Baroque, Cubism…) or its genre
(portrait, landscape, still life…)". Session 1 asks each team to pick **5 to 8 classes** from WikiArt
(`huggan/wikiart`, 81,444 paintings) and build a subset with `make_wikiart_subset.py`. The script's licence
note forbids committing the images, so only the script and the exact command are versioned.

Two facts from the script's documentation drive the choice, and both were measured on 2026-10-19 with
`--list-classes` and `--dry-run` on the pinned dataset revision `d559852d`:

- **Artist shortcut.** The dataset is ordered in blocks of artists. A classifier can learn the painter instead
  of the label. The script limits this with `--max-per-artist 50`, but only for *named* artists; the catch-all
  "Unknown Artist" is never capped, so we cannot measure how concentrated it is.
- **Download cost.** Each shard is ~470 MB and is fetched whole.

| Candidate (5 classes x 300 images, seed 0) | Shards / GB (default `fewest-shards`) | "Unknown Artist" share of the subset | Shards / GB (`spread`) | "Unknown Artist" share (`spread`) |
|---|---|---|---|---|
| Genres: portrait, landscape, cityscape, still_life, religious_painting | 5 / ~2.3 GB | 717 / 1500 = 48% | 6 / ~2.8 GB | 520 / 1500 = 35% |
| Styles: Impressionism, Baroque, Cubism, Ukiyo_e, Pop_Art | 9 / ~4.2 GB | 1402 / 1500 = 93% | 19 / ~8.9 GB | 1265 / 1500 = 84% |

Named artists available per class in the full dataset: the five genres have 69 to 116 each; the five styles
have 52 (Impressionism), 12 (Cubism), 9 (Baroque), 2 (Ukiyo_e) and 1 (Pop_Art). Smallest class among the five
genres: still_life with 2,788 images, so 300 per class leaves room to grow.

Reproduce with:

```
python make_wikiart_subset.py --label genre \
    --classes portrait landscape cityscape still_life religious_painting \
    --per-class 300 --size 224 --out data/wikiart_genre5 --dry-run
```

## Options considered

1. **Five genres (portrait, landscape, cityscape, still_life, religious_painting).** Half the download of the
   style set, and the artist cap actually applies to about half of the images (48% unknown vs 93%). The classes
   are content categories, the same vocabulary as the visitor questions of feature 2 ("stormy sea", "portraits
   of children"). Weakness: genres can overlap (a religious painting can also be a portrait), so some errors
   will be label ambiguity, not model failure.
2. **Five contrasted styles (Impressionism, Baroque, Cubism, Ukiyo_e, Pop_Art).** The brief's own examples,
   and a natural fit for "tell the story of that movement". Weakness: 93% of the images have no artist label,
   so the artist cap is nearly inert and we could not tell whether the model learned the style or the painter;
   Pop_Art and Ukiyo_e have only 1 and 2 named artists; twice the download (4.2 GB), or 8.9 GB with `spread`.
3. **Both labels, two classifiers.** Covers the whole brief, but doubles the experiments and the report for a
   mission whose weight is on diagnosis and justification, not coverage.

## Decision

We will use option 1: the five genres above, **300 images per class (1,500 images)**, `--max-per-artist 50`,
seed 0, `fewest-shards` strategy, with the dataset revision pinned by the script. We will download at
**224 px** (the size pretrained ResNets expect) and downscale in code for the dense baseline, because going
bigger later would mean streaming the shards again, while going smaller is free. The images stay out of Git
(`data/` is in `.gitignore`); the exact command is kept in `data/wikiart_genre5/subset_info.json` and copied into
the README.

Style classification is left as an optional extension, not a deliverable, if time remains after the ResNet.

## Consequences

- Easier: a 2.3 GB download, and an artist-leakage check we can actually run on about half of the images using
  the `artist` column of `metadata.csv`.
- Easier: the same genre labels can seed the vocabulary of the natural-language search if that feature turns
  out to be in scope.
- Harder: with 1,500 images and a typical 70/15/15 split the test set has ~225 images, so a 95% interval on an
  accuracy near 90% is about +/- 4 points. Differences between models smaller than that are not evidence. This
  feeds the split ADR and the choice of N.
- Harder: single-label genres are ambiguous. We must read the confusion matrix before blaming the network.
- To check: the 35 to 48% of images with "Unknown Artist" cannot be audited. Revisit this decision if the
  confusion matrix shows most errors between overlapping genres (switch to more separable classes), or if a
  group-by-artist split drops accuracy sharply compared with a random split (the model was learning painters).
