"""Small helpers for the retrieval notebooks: ranking metrics and an image grid.

The model logic (tokenizing, encoding, scoring) stays in the notebooks on purpose.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_paintings(root: Path, split: str | None = None) -> pd.DataFrame:
    """One row per painting: id, path, genre, style, artist, split.

    Joins `metadata.csv` with the shared `split.csv`, so every notebook sees the same train / val / test cut.
    """
    meta = pd.read_csv(
        root / "metadata.csv", usecols=["id", "path", "genre", "style", "artist"]
    )
    paintings = meta.merge(
        pd.read_csv(root / "split.csv", usecols=["id", "split"]), on="id"
    )
    if split is not None:
        paintings = paintings.query("split == @split")
    return paintings.reset_index(drop=True)


def precision_at_k(gains: np.ndarray, k: int) -> float:
    """Share of relevant paintings in the top k. `gains` is in ranked order (0 = not relevant)."""
    return float((gains[:k] > 0).mean())


def ndcg_at_k(gains: np.ndarray, ideal_gains: np.ndarray, k: int) -> float:
    """nDCG@k. `gains` is the ranking we got, `ideal_gains` the best possible ranking.

    Gains are graded relevance (0 / 1 / 2). Returns 0 when nothing is relevant.
    """
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    def dcg(g: np.ndarray) -> float:
        g = g[:k]
        return float((g * discount[: len(g)]).sum())

    best = dcg(np.sort(ideal_gains)[::-1])
    return dcg(gains) / best if best > 0 else 0.0


def show_ranking(root: Path, paths: list[str], titles: list[str], title: str) -> None:
    """Plot the top-k paintings of one ranking side by side."""
    fig, axes = plt.subplots(1, len(paths), figsize=(3 * len(paths), 3.4))
    for ax, path, caption in zip(np.atleast_1d(axes), paths, titles):
        ax.imshow(Image.open(root / path))
        ax.set_title(caption, fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
