#!/usr/bin/env python
"""Explore the WikiArt subset built by make_wikiart_subset.py and print the facts quoted in the ADRs.

Every number in adr/0002 and adr/0003 that describes the downloaded images comes from this report, so that it can
be re-checked with one command. Nothing is written to disk.

USAGE
-----
    python src/deep_learning_project_waerniers_lamy/vision/explore_data.py data/wikiart_genre5

Reads ``metadata.csv`` and ``subset_info.json`` (written by make_wikiart_subset.py) and the JPEG files next to
them. Standard library only; if Pillow is installed it also checks that every image is a 224 x 224 RGB file.

WHAT IT REPORTS
---------------
1. Integrity    files on disk versus metadata.csv, labels versus folders, how the subset was built.
2. Classes      images per class.
3. Artists      per class: images with a named artist versus "Unknown Artist", number of artists, top artist;
                artists that paint in several classes; the per-artist cap.
4. Shortcuts    how well the artist alone, or the style alone, predicts the genre (leave-one-out accuracy).
                This is the evidence behind the "artist shortcut" argument of adr/0003.
5. Styles       the three most frequent styles of each genre.
6. Sizes        size and aspect ratio of the paintings as stored in the dataset, and how much the centre crop to
                a square discards.
7. Files        image format and exact duplicates (same bytes) inside the subset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

UNKNOWN_ARTIST = "Unknown Artist"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def table(header, rows) -> None:
    log("| " + " | ".join(header) + " |")
    log("|" + "---|" * len(header))
    for row in rows:
        log("| " + " | ".join(str(x) for x in row) + " |")
    log()


def pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "n/a"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print the facts about a WikiArt subset that the ADRs rely on.")
    p.add_argument("data_dir", nargs="?", type=Path, default=Path("data/wikiart_genre5"),
                   help="folder written by make_wikiart_subset.py (metadata.csv, subset_info.json, one folder per class)")
    return p.parse_args(argv)


def load(data_dir: Path):
    meta, info = data_dir / "metadata.csv", data_dir / "subset_info.json"
    for f in (meta, info):
        if not f.exists():
            sys.exit(f"{f} not found. Build the subset first with make_wikiart_subset.py.")
    with open(meta, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, json.loads(info.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------------
# 1. Integrity
# --------------------------------------------------------------------------------------------------
def report_integrity(data_dir: Path, rows, info) -> None:
    log("## 1. Integrity\n")
    log(f"- built with: `{info['command']}`")
    log(f"- dataset revision {info['revision'][:8]}, label = {info['label']}, seed {info['seed']}, strategy "
        f"{info['strategy']}, max {info['max_per_artist']} images per named artist, {info['size']} px, "
        f"centre crop = {info['crop']}, created {info['created_utc']}")
    on_disk = {p.relative_to(data_dir).as_posix() for p in data_dir.glob("*/*") if p.is_file()}
    in_meta = {r["path"] for r in rows}
    log(f"- {len(rows)} rows in metadata.csv, {len(on_disk)} files in the class folders")
    log(f"- listed in metadata.csv but missing on disk: {len(in_meta - on_disk)}")
    log(f"- on disk but not in metadata.csv (stale files, `ImageFolder` would read them): {len(on_disk - in_meta)}")
    folder_order = sorted(info["classes"])
    bad_idx = sum(folder_order.index(r["label"]) != int(r["label_idx"]) for r in rows)
    bad_label = sum(r[info["label"]] != r["label"] for r in rows)
    bad_folder = sum(r["path"].split("/")[0] != r["label"] for r in rows)
    log(f"- label_idx different from the sorted folder order: {bad_idx}")
    log(f"- `{info['label']}` column different from `label`: {bad_label}; file in the wrong class folder: {bad_folder}")
    log(f"- duplicated ids in metadata.csv: {len(rows) - len({r['id'] for r in rows})}\n")


# --------------------------------------------------------------------------------------------------
# 2-3. Classes and artists
# --------------------------------------------------------------------------------------------------
def report_classes(rows, classes) -> None:
    log("## 2. Classes\n")
    count = Counter(r["label"] for r in rows)
    table(["class", "images"], [(c, count[c]) for c in classes] + [("total", len(rows))])


def report_artists(rows, classes, cap: int) -> None:
    log("## 3. Artists\n")
    named_by_class = defaultdict(Counter)
    for r in rows:
        if r["artist"] != UNKNOWN_ARTIST:
            named_by_class[r["label"]][r["artist"]] += 1
    out = []
    for c in classes:
        n = sum(r["label"] == c for r in rows)
        arts = named_by_class[c]
        named = sum(arts.values())
        top = arts.most_common(1)[0] if arts else ("-", 0)
        out.append((c, n, f"{n - named} ({pct(n - named, n)})", f"{named} ({pct(named, n)})", len(arts),
                    f"{top[0]} ({top[1]})"))
    table(["class", "images", "Unknown Artist", "named artist", "named artists", "top artist (images)"], out)

    named_rows = [r for r in rows if r["artist"] != UNKNOWN_ARTIST]
    per_artist = Counter(r["artist"] for r in named_rows)
    classes_of = defaultdict(set)
    for r in named_rows:
        classes_of[r["artist"]].add(r["label"])
    multi = sum(len(cs) > 1 for cs in classes_of.values())
    unknown = len(rows) - len(named_rows)
    log(f"- images with a named artist: {len(named_rows)} ({pct(len(named_rows), len(rows))}), "
        f"from {len(per_artist)} artists")
    log(f"- images with \"Unknown Artist\": {unknown} ({pct(unknown, len(rows))})")
    log(f"- named artists with images in more than one class: {multi} of {len(per_artist)}")
    log(f"- most images from one named artist: {per_artist.most_common(1)[0][1]} (cap used when building: {cap})")
    log(f"- top 8 named artists: {per_artist.most_common(8)}\n")


# --------------------------------------------------------------------------------------------------
# 4. Shortcuts: can the artist or the style alone predict the genre?
# --------------------------------------------------------------------------------------------------
def leave_one_out_accuracy(rows, key):
    """Predict each image's class as the most frequent class among the OTHER images with the same key.

    Images whose key is None, or that are alone in their group, are skipped. Ties go to the first class in
    alphabetical order. Returns (accuracy, number of images evaluated).
    """
    groups = defaultdict(Counter)
    for r in rows:
        if key(r) is not None:
            groups[key(r)][r["label"]] += 1
    hit = total = 0
    for r in rows:
        k = key(r)
        if k is None:
            continue
        others = groups[k].copy()
        others[r["label"]] -= 1
        if sum(others.values()) == 0:
            continue
        hit += max(sorted(others), key=lambda c: others[c]) == r["label"]
        total += 1
    return (hit / total if total else float("nan")), total


def report_shortcuts(rows, classes) -> None:
    log("## 4. Shortcuts: predicting the genre without looking at the painting\n")
    log("Leave-one-out majority vote: each image is predicted from the other images with the same artist (or style).\n")
    chance = max(Counter(r["label"] for r in rows).values()) / len(rows)
    out = []
    for name, key in (("artist (named artists only)", lambda r: None if r["artist"] == UNKNOWN_ARTIST else r["artist"]),
                      ("style", lambda r: r["style"])):
        acc, n = leave_one_out_accuracy(rows, key)
        out.append((name, n, f"{acc:.1%}", f"{chance:.1%}"))
    table(["knowing only the...", "images evaluated", "accuracy", "chance (majority class)"], out)


def report_styles(rows, classes) -> None:
    log("## 5. Styles inside each genre (top 3)\n")
    out = []
    for c in classes:
        styles = Counter(r["style"] for r in rows if r["label"] == c)
        out.append((c, ", ".join(f"{s} ({n})" for s, n in styles.most_common(3))))
    table(["genre", "most frequent styles (images)"], out)


# --------------------------------------------------------------------------------------------------
# 6-7. Original sizes and image files
# --------------------------------------------------------------------------------------------------
def report_originals(rows, info) -> None:
    log("## 6. Paintings as stored in the dataset (before our resizing)\n")
    dims = [(int(r["orig_width"]), int(r["orig_height"])) for r in rows if r["orig_width"] and r["orig_height"]]
    if not dims:
        log("- no original sizes recorded\n")
        return
    widths, heights = [d[0] for d in dims], [d[1] for d in dims]
    log(f"- width in px: min {min(widths)}, median {statistics.median(widths):.0f}, max {max(widths)}")
    log(f"- height in px: min {min(heights)}, median {statistics.median(heights):.0f}, max {max(heights)}")
    wide = sum(w / h > 1.5 for w, h in dims)
    tall = sum(h / w > 1.5 for w, h in dims)
    log(f"- wider than 1.5:1: {wide} ({pct(wide, len(dims))}); taller than 1.5:1: {tall} ({pct(tall, len(dims))})")
    if info["crop"]:
        kept = sorted(min(w, h) / max(w, h) for w, h in dims)
        log(f"- the centre crop to a square keeps a median {statistics.median(kept):.0%} of the original area "
            f"(worst case {kept[0]:.0%}); the rest is discarded\n")
    else:
        log("")


def report_files(data_dir: Path, rows, info) -> None:
    log("## 7. Image files\n")
    by_hash = defaultdict(list)
    for r in rows:
        f = data_dir / r["path"]
        if f.exists():
            by_hash[hashlib.sha1(f.read_bytes()).hexdigest()].append(r)
    dup = [rs for rs in by_hash.values() if len(rs) > 1]
    across_classes = sum(len({r["label"] for r in rs}) > 1 for rs in dup)
    across_artists = sum(len({r["artist"] for r in rs}) > 1 for rs in dup)
    log(f"- exact duplicates (identical bytes): {len(dup)} groups, {sum(len(rs) for rs in dup)} images; "
        f"{across_classes} groups span several classes, {across_artists} several artists")
    try:
        from PIL import Image
    except ImportError:
        log("- Pillow is not installed: format check skipped (`pip install pillow`)\n")
        return
    size = info["size"]
    bad = Counter()
    for r in rows:
        f = data_dir / r["path"]
        if not f.exists():
            continue
        with Image.open(f) as im:
            bad["size"] += im.size != (size, size)
            bad["mode"] += im.mode != "RGB"
            bad["format"] += im.format != "JPEG"
    log(f"- not {size} x {size}: {bad['size']}; not RGB: {bad['mode']}; not JPEG: {bad['format']}\n")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args(argv)
    rows, info = load(args.data_dir)
    classes = sorted(info["classes"])
    log(f"# Data report: {args.data_dir}\n")
    report_integrity(args.data_dir, rows, info)
    report_classes(rows, classes)
    report_artists(rows, classes, info["max_per_artist"])
    report_shortcuts(rows, classes)
    report_styles(rows, classes)
    report_originals(rows, info)
    report_files(args.data_dir, rows, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
