"""Utilities for the Natural-Language Painting Recommendation pipeline.

Adheres strictly to the Museum A (train/val) / Museum B (test) split hygiene (ADR 0003 & 0005).
Provides data loading, ranking metrics (Recall@k, MRR), and experiment logging in results/nlp_experiments.csv.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "wikiart_genre5"
RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENTS_CSV = RESULTS_DIR / "nlp_experiments.csv"


# --------------------------------------------------------------------------------------------------
# 1. Data loading with strict split hygiene
# --------------------------------------------------------------------------------------------------
def load_paintings(data_dir: Path | None = None, split: str | Sequence[str] | None = "val") -> pd.DataFrame:
    """Load metadata joined with split.csv.

    Parameters
    ----------
    data_dir : Path, optional
        Path to data folder, default DATA_DIR.
    split : str or list of str, optional
        Which split(s) to load: 'train', 'val', or 'test'.
        Default is 'val' (Museum A validation) to prevent any lookahead or leakage.
    """
    path = data_dir or DATA_DIR
    meta = pd.read_csv(path / "metadata.csv")
    split_df = pd.read_csv(path / "split.csv")[["id", "split"]]
    df = meta.merge(split_df, on="id")

    if split is not None:
        if isinstance(split, str):
            split = [split]
        df = df[df["split"].isin(split)].copy()

    return df.reset_index(drop=True)


from deep_learning_project_waerniers_lamy.nlp.judge import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k_graded,
    load_judgments,
    save_judgment,
)
def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int = 5) -> float:
    """Return 1.0 if at least one relevant painting is in the top-k, else 0.0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(item_id in relevant_ids for item_id in top_k) else 0.0


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int = 5) -> float:
    """Proportion of top-k retrieved paintings that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1.0 for item_id in top_k if item_id in relevant_ids) / len(top_k)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank (1 / rank) of the first relevant painting found (0 if none found)."""
    for rank_idx, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank_idx
    return 0.0


def evaluate_retrieval_benchmark(
    retrieval_fn: Callable[[str, int], list[str]],
    benchmark: list[dict],
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Evaluate a retrieval function across the benchmark query deck.

    Parameters
    ----------
    retrieval_fn : Callable[[str, int], list[str]]
        Function taking (query_text, top_k) and returning list of painting ids.
    benchmark : list of dict
        List of dicts with keys: 'query', 'relevant_ids', 'category'
    k_values : tuple of int
        Cutoffs for Recall (e.g. 5, 10).
    """
    recalls = {k: [] for k in k_values}
    mrrs = []
    latencies = []

    for item in benchmark:
        query = item["query"]
        relevant_ids = set(item.get("relevant_ids", []))
        if not relevant_ids:
            continue

        start = time.perf_counter()
        retrieved = retrieval_fn(query, max(k_values))
        latencies.append((time.perf_counter() - start) * 1000.0)

        for k in k_values:
            recalls[k].append(recall_at_k(retrieved, relevant_ids, k=k))
        mrrs.append(reciprocal_rank(retrieved, relevant_ids))

    results = {
        f"recall_at_{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0
        for k in k_values
    }
    results["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0
    results["latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
    results["num_queries"] = len(mrrs)

    return results


# --------------------------------------------------------------------------------------------------
# 3. Experiment logging
# --------------------------------------------------------------------------------------------------
def save_nlp_result(
    name: str,
    method: str,
    corpus: str,
    metrics: dict[str, float],
    settings: dict | None = None,
) -> pd.DataFrame:
    """Append a row to results/nlp_experiments.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "date": now,
        "name": name,
        "method": method,
        "corpus": corpus,
        "recall_at_5": round(metrics.get("recall_at_5", 0.0), 4),
        "recall_at_10": round(metrics.get("recall_at_10", 0.0), 4),
        "mrr": round(metrics.get("mrr", 0.0), 4),
        "latency_ms": round(metrics.get("latency_ms", 0.0), 2),
        "num_queries": int(metrics.get("num_queries", 0)),
    }
    if settings:
        for k, v in settings.items():
            row[k] = v

    new_row_df = pd.DataFrame([row])
    if EXPERIMENTS_CSV.exists():
        existing_df = pd.read_csv(EXPERIMENTS_CSV)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df

    updated_df.to_csv(EXPERIMENTS_CSV, index=False)
    print(f"Logged NLP experiment to {EXPERIMENTS_CSV}")
    return updated_df


def load_nlp_summary() -> pd.DataFrame:
    """Load results/nlp_experiments.csv if it exists."""
    if EXPERIMENTS_CSV.exists():
        return pd.read_csv(EXPERIMENTS_CSV)
    return pd.DataFrame()


# --------------------------------------------------------------------------------------------------
# 4. Visualisation helpers
# --------------------------------------------------------------------------------------------------
def plot_retrieved_paintings(
    query: str,
    retrieved_df: pd.DataFrame,
    data_dir: Path | None = None,
    top_k: int = 5,
) -> None:
    """Plot the top-k retrieved paintings."""
    path_root = data_dir or DATA_DIR
    subset = retrieved_df.head(top_k)
    fig, axes = plt.subplots(1, len(subset), figsize=(3.2 * len(subset), 3.5))
    if len(subset) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, subset.iterrows()):
        img_path = path_root / row["path"]
        if img_path.exists():
            ax.imshow(Image.open(img_path))
        ax.set_title(
            f"{row.get('label', row.get('genre', ''))}\n{row.get('artist', 'Unknown')[:18]}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(f"Query: \"{query}\"", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()
