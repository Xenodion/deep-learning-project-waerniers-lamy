"""Experiment 2: Retrieval using VLM Descriptions + Sentence-BERT (all-MiniLM-L6-v2).

Tests the hypothesis:
Injecting explicit visual descriptions from a VLM unlocks semantic retrieval
for visual concepts that were previously impossible to retrieve via metadata alone.

Isolates the effect of visual data: same text encoder as E0 (all-MiniLM-L6-v2).
Evaluated strictly on Museum A (validation split), ADR 0003 & 0005.
Results logged into results/nlp_experiments.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from deep_learning_project_waerniers_lamy.nlp.benchmark import get_benchmark_queries
from deep_learning_project_waerniers_lamy.nlp.E01_generate_descriptions import (
    DESCRIPTIONS_FILE,
    load_cached_descriptions,
)
from deep_learning_project_waerniers_lamy.nlp.judge import ndcg_at_k
from deep_learning_project_waerniers_lamy.nlp.utils import (
    evaluate_retrieval_benchmark,
    load_paintings,
    save_nlp_result,
)


def run_experiment_2(model_name: str = "all-MiniLM-L6-v2") -> dict[str, float]:
    print("=" * 70)
    print("EXPERIMENT 2: VLM Descriptions + Sentence-BERT Retrieval")
    print(f"Model: {model_name} | Corpus: Museum A (val)")
    print("=" * 70)

    # 1. Load Museum A validation paintings and VLM descriptions
    paintings = load_paintings(split="val")
    descriptions = load_cached_descriptions()

    if not descriptions:
        raise RuntimeError("No cached descriptions found in descriptions_val.json! Run E01_generate_descriptions.py first.")

    # Match descriptions (or fallback to metadata if uncaptioned)
    paintings["description"] = paintings["id"].map(descriptions).fillna("")
    captioned_count = (paintings["description"] != "").sum()
    print(f"Loaded {len(paintings)} paintings ({captioned_count}/{len(paintings)} have VLM descriptions).")

    # 2. Format searchable text: Description + Metadata
    def format_searchable_text(row: pd.Series) -> str:
        desc = row["description"]
        artist = str(row["artist"]).replace("-", " ")
        style = str(row["style"]).replace("_", " ")
        genre = str(row["genre"]).replace("_", " ")

        meta_part = f"{genre} style {style}" if "unknown" in artist.lower() else f"{genre} by {artist}, {style} style"
        if desc:
            return f"{desc}. {meta_part}"
        return meta_part

    texts = [format_searchable_text(row) for _, row in paintings.iterrows()]

    # 3. Compute text embeddings
    print("Encoding descriptions + metadata with SentenceTransformer...")
    embedder = SentenceTransformer(model_name)
    doc_embeddings = embedder.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"Computed embeddings shape: {tuple(doc_embeddings.shape)}")

    # 4. Retrieval function
    painting_ids = paintings["id"].tolist()

    def retrieve(query: str, top_k: int = 10) -> list[str]:
        query_emb = embedder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = torch.matmul(doc_embeddings, query_emb).cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [painting_ids[i] for i in top_indices]

    # 5. Evaluate on benchmark
    benchmark = get_benchmark_queries()
    eval_benchmark = [q for q in benchmark if q.get("relevant_ids")]

    overall_metrics = evaluate_retrieval_benchmark(
        retrieve, eval_benchmark, k_values=(5, 10)
    )

    categories = sorted(set(q["category"] for q in eval_benchmark))
    print("\n--- RESULTS BY QUERY CATEGORY (E2 vs E0) ---")
    print(f"{'Category':<18} | {'Queries':<7} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8}")
    print("-" * 65)

    for cat in categories:
        cat_queries = [q for q in eval_benchmark if q["category"] == cat]
        res = evaluate_retrieval_benchmark(retrieve, cat_queries, k_values=(5, 10))
        print(
            f"{cat:<18} | {res['num_queries']:<7} | "
            f"{res['recall_at_5'] * 100:>8.1f}% | "
            f"{res['recall_at_10'] * 100:>8.1f}% | "
            f"{res['mrr']:>8.3f}"
        )

    print("-" * 65)
    print(
        f"{'OVERALL (E2)':<18} | {overall_metrics['num_queries']:<7} | "
        f"{overall_metrics['recall_at_5'] * 100:>8.1f}% | "
        f"{overall_metrics['recall_at_10'] * 100:>8.1f}% | "
        f"{overall_metrics['mrr']:>8.3f}"
    )
    print(f"Mean latency per query: {overall_metrics['latency_ms']:.2f} ms")

    # 6. Log result in results/nlp_experiments.csv
    save_nlp_result(
        name="E2_vlm_descriptions",
        method=f"VLM (BLIP) + {model_name}",
        corpus="val_A",
        metrics=overall_metrics,
        settings={
            "encoder": model_name,
            "text_source": "vlm_description_and_metadata",
            "captioned_share": round(captioned_count / len(paintings), 2),
            "dim": int(doc_embeddings.shape[1]),
        },
    )

    return overall_metrics


if __name__ == "__main__":
    run_experiment_2()
