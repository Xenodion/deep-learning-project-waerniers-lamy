"""Experiment 0: Metadata + Sentence-BERT Baseline (all-MiniLM-L6-v2).

Demonstrates the fundamental limitation of metadata-only semantic search:
metadata describes categories/artists/styles, but completely lacks visual content.

Evaluated strictly on Museum A (validation split, 480 paintings), ADR 0003 & 0005.
Results are logged into results/nlp_experiments.csv.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from deep_learning_project_waerniers_lamy.nlp.benchmark import get_benchmark_queries
from deep_learning_project_waerniers_lamy.nlp.utils import (
    evaluate_retrieval_benchmark,
    load_paintings,
    recall_at_k,
    reciprocal_rank,
    save_nlp_result,
)


def run_experiment_0(model_name: str = "all-MiniLM-L6-v2") -> dict[str, float]:
    print("=" * 70)
    print("EXPERIMENT 0: Metadata + Sentence-BERT Baseline")
    print(f"Model: {model_name} | Corpus: Museum A (val, 480 paintings)")
    print("=" * 70)

    # 1. Load Museum A validation paintings
    paintings = load_paintings(split="val")
    print(f"Loaded {len(paintings)} paintings from Museum A validation split.")

    # 2. Format metadata string for each painting
    # Clean artist name: replace hyphens with spaces
    def format_metadata(row: pd.Series) -> str:
        artist = str(row["artist"]).replace("-", " ")
        style = str(row["style"]).replace("_", " ")
        genre = str(row["genre"]).replace("_", " ")
        if "unknown" in artist.lower():
            return f"{genre}, {style} style"
        return f"{genre} by {artist}, {style} style"

    texts = [format_metadata(row) for _, row in paintings.iterrows()]

    # 3. Compute text embeddings for the metadata
    print("Loading SentenceTransformer model and encoding metadata...")
    embedder = SentenceTransformer(model_name)
    doc_embeddings = embedder.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"Computed metadata embeddings: shape {tuple(doc_embeddings.shape)}")

    # 4. Define retrieval function
    painting_ids = paintings["id"].tolist()

    def retrieve(query: str, top_k: int = 10) -> list[str]:
        query_emb = embedder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine similarity is dot product because embeddings are L2 normalized
        scores = torch.matmul(doc_embeddings, query_emb).cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [painting_ids[i] for i in top_indices]

    # 5. Evaluate on benchmark queries
    benchmark = get_benchmark_queries()
    # Filter out negative queries for accuracy calculation
    eval_benchmark = [q for q in benchmark if q.get("relevant_ids")]

    overall_metrics = evaluate_retrieval_benchmark(
        retrieve, eval_benchmark, k_values=(5, 10)
    )

    # Breakdown by category
    categories = sorted(set(q["category"] for q in eval_benchmark))
    print("\n--- RESULTS BY QUERY CATEGORY ---")
    print(f"{'Category':<18} | {'Queries':<7} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8}")
    print("-" * 65)

    category_results = {}
    for cat in categories:
        cat_queries = [q for q in eval_benchmark if q["category"] == cat]
        res = evaluate_retrieval_benchmark(retrieve, cat_queries, k_values=(5, 10))
        category_results[cat] = res
        print(
            f"{cat:<18} | {res['num_queries']:<7} | "
            f"{res['recall_at_5'] * 100:>8.1f}% | "
            f"{res['recall_at_10'] * 100:>8.1f}% | "
            f"{res['mrr']:>8.3f}"
        )

    print("-" * 65)
    print(
        f"{'OVERALL':<18} | {overall_metrics['num_queries']:<7} | "
        f"{overall_metrics['recall_at_5'] * 100:>8.1f}% | "
        f"{overall_metrics['recall_at_10'] * 100:>8.1f}% | "
        f"{overall_metrics['mrr']:>8.3f}"
    )
    print(f"Mean latency per query: {overall_metrics['latency_ms']:.2f} ms")

    # 6. Log result in results/nlp_experiments.csv
    save_nlp_result(
        name="E0_metadata_baseline",
        method=f"Metadata + {model_name}",
        corpus="val_A",
        metrics=overall_metrics,
        settings={
            "encoder": model_name,
            "text_source": "artist_genre_style",
            "dim": int(doc_embeddings.shape[1]),
        },
    )

    return overall_metrics


if __name__ == "__main__":
    run_experiment_0()
