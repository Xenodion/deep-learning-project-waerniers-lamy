"""Split the images into museum A (train + validation) and museum B (test), as decided in adr/0003.

Run from the repository root:
    python src/deep_learning_project_waerniers_lamy/vision/make_split.py

Reads data/wikiart_genre5/metadata.csv and writes data/wikiart_genre5/split.csv.
"""
import random
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/wikiart_genre5")
B_PER_GENRE = 120  # test images per genre
VAL_SHARE = 0.2  # share of museum A kept for validation
SEED = 0

meta = pd.read_csv(DATA_DIR / "metadata.csv")
genres = sorted(meta["label"].unique())

# 1. Museum B: pick whole named artists so that every genre gets exactly 120 images.
# "Unknown Artist" cannot be grouped, so it never goes to museum B.
named = meta[meta["artist"] != "Unknown Artist"]
per_artist = pd.crosstab(named["artist"], named["label"])  # images of each artist in each genre
artists = sorted(per_artist.index)


def gap(chosen):
    """How far the chosen artists are from 120 images in every genre (0 = perfect)."""
    images_per_genre = per_artist.loc[list(chosen)].sum()
    return (images_per_genre - B_PER_GENRE).abs().sum()


# Hill climbing: add or remove a random artist, keep the change if the gap does not get worse.
rng = random.Random(SEED)
museum_b = set(rng.sample(artists, 12))
best = gap(museum_b)
for step in range(20000):
    if best == 0:
        break
    trial = museum_b ^ {rng.choice(artists)}  # ^ adds the artist if absent, removes it if present
    if rng.random() < 0.3:
        trial = trial ^ {rng.choice(artists)}
    if gap(trial) <= best:
        museum_b, best = trial, gap(trial)
if best != 0:
    sys.exit("Could not find artists giving the same number of images in every genre.")

# 2. Everything else is museum A. Cut it at random into train and validation, genre by genre.
meta["split"] = "train"
meta.loc[meta["artist"].isin(museum_b), "split"] = "test"
rng = random.Random(SEED)  # restart the generator so the cut does not depend on the search above
for genre in genres:
    ids = sorted(meta.loc[(meta["label"] == genre) & (meta["split"] == "train"), "id"])
    rng.shuffle(ids)
    meta.loc[meta["id"].isin(ids[: round(len(ids) * VAL_SHARE)]), "split"] = "val"

# 3. Save and check.
meta[["id", "path", "label", "label_idx", "artist", "split"]].to_csv(DATA_DIR / "split.csv", index=False)

print(pd.crosstab(meta["split"], meta["label"], margins=True), "\n")
for name in ["train", "val", "test"]:
    part = meta[meta["split"] == name]
    known = part[part["artist"] != "Unknown Artist"]
    print(f"{name:5s}: {len(part)} images, {len(known)} with a named artist ({known['artist'].nunique()} artists), "
          f"{len(part) - len(known)} Unknown Artist")

named_artists = meta[meta["artist"] != "Unknown Artist"]
test_artists = set(named_artists.loc[named_artists["split"] == "test", "artist"])
other_artists = set(named_artists.loc[named_artists["split"] != "test", "artist"])
assert not test_artists & other_artists, "a named artist is in the test set and in train/validation"
print("\nnamed artists shared between test and train/validation: 0")
print("Wrote", DATA_DIR / "split.csv")
