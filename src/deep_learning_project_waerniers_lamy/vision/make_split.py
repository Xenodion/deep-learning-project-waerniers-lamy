#!/usr/bin/env python
"""Split the subset into museum A (train + validation) and museum B (test), as decided in adr/0003.

    python src/deep_learning_project_waerniers_lamy/vision/make_split.py data/wikiart_genre5

Museum B takes whole named artists, picked so that every class has exactly --b-per-class images; no "Unknown
Artist" image goes to B. Museum A gets everything else, cut at random, class by class, into train and validation.
Writes <data_dir>/split.csv (id, path, label, label_idx, artist, split) and prints the sets with their checks.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

UNKNOWN_ARTIST = "Unknown Artist"
RESTARTS = 8  # independent searches for museum B, stopped at the first exact one
STEPS = 20_000  # moves per search


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the museum A / museum B split (adr/0003).")
    p.add_argument("data_dir", nargs="?", type=Path, default=Path("data/wikiart_genre5"))
    p.add_argument("--b-per-class", type=int, default=120, help="images per class in museum B (the test set)")
    p.add_argument("--val-fraction", type=float, default=0.2, help="share of each class of museum A used for validation")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def pick_museum_b(rows, classes, per_class: int, seed: int) -> set[str]:
    """Named artists whose images make exactly `per_class` images in every class (hill climbing on whole artists)."""
    by_artist = defaultdict(Counter)
    for r in rows:
        if r["artist"] != UNKNOWN_ARTIST:
            by_artist[r["artist"]][r["label"]] += 1
    artists = sorted(by_artist)

    def gap(chosen) -> int:
        total = Counter()
        for a in chosen:
            total.update(by_artist[a])
        return sum(abs(total[c] - per_class) for c in classes)

    for restart in range(RESTARTS):
        rng = random.Random(seed + restart)
        chosen = set(rng.sample(artists, 12))
        best = gap(chosen)
        for _ in range(STEPS):
            if best == 0:
                return chosen
            trial = chosen ^ {rng.choice(artists)}
            if rng.random() < 0.3:
                trial ^= {rng.choice(artists)}
            trial_gap = gap(trial)
            if trial_gap <= best:
                chosen, best = trial, trial_gap
        if best == 0:
            return chosen
    sys.exit(f"No set of artists gives exactly {per_class} images per class. Try a smaller --b-per-class.")


def assign_splits(rows, classes, b_artists: set[str], val_fraction: float, seed: int) -> dict[str, str]:
    """{image id: 'train' | 'val' | 'test'}."""
    split = {r["id"]: "test" if r["artist"] in b_artists else "A" for r in rows}
    rng = random.Random(seed)
    for c in classes:
        ids = sorted(r["id"] for r in rows if r["label"] == c and split[r["id"]] == "A")
        rng.shuffle(ids)
        n_val = round(len(ids) * val_fraction)
        for i in ids[:n_val]:
            split[i] = "val"
        for i in ids[n_val:]:
            split[i] = "train"
    return split


def report(rows, split, classes) -> None:
    print("| set | " + " | ".join(classes) + " | total | named-artist images (artists) | Unknown Artist images |")
    print("|---|" + "---|" * (len(classes) + 3))
    artists = {}
    for s in ("train", "val", "test"):
        rs = [r for r in rows if split[r["id"]] == s]
        n = Counter(r["label"] for r in rs)
        artists[s] = {r["artist"] for r in rs if r["artist"] != UNKNOWN_ARTIST}
        unknown = sum(r["artist"] == UNKNOWN_ARTIST for r in rs)
        print(f"| {s} | " + " | ".join(str(n[c]) for c in classes)
              + f" | {len(rs)} | {len(rs) - unknown} ({len(artists[s])}) | {unknown} |")
    shared_test = artists["test"] & (artists["train"] | artists["val"])
    if shared_test:
        sys.exit(f"BUG: {len(shared_test)} named artists are in the test set and in train or validation.")
    print(f"\nnamed artists shared between test and train/validation: 0 | between train and validation: "
          f"{len(artists['train'] & artists['val'])}")


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.data_dir / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    classes = sorted({r["label"] for r in rows})
    b_artists = pick_museum_b(rows, classes, args.b_per_class, args.seed)
    split = assign_splits(rows, classes, b_artists, args.val_fraction, args.seed)
    out = args.data_dir / "split.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "path", "label", "label_idx", "artist", "split"])
        for r in rows:
            w.writerow([r["id"], r["path"], r["label"], r["label_idx"], r["artist"], split[r["id"]]])
    report(rows, split, classes)
    print(f"\nWrote {out} ({len(rows)} rows, seed {args.seed}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
