"""Experiment 4: Direct Text-to-Image Retrieval using CLIP (openai/clip-vit-base-patch32).

Hypothesis:
Directly mapping natural-language text queries into the visual embedding space
bypasses language description bottlenecks and enables zero-shot retrieval of visual concepts.

Evaluated strictly on Museum A (validation split, 480 paintings), ADR 0003 & 0005.
Results logged into results/nlp_experiments.csv.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from deep_learning_project_waerniers_lamy.nlp.benchmark import get_benchmark_queries
from deep_learning_project_waerniers_lamy.nlp.E04_generate_clip_embeddings import (
    generate_clip_image_embeddings,
)
from deep_learning_project_waerniers_lamy.nlp.utils import (
    evaluate_retrieval_benchmark,
    load_paintings,
    save_nlp_result,
)


def run_experiment_4(model_name: str = "openai/clip-vit-base-patch32") -> dict[str, float]:
    print("=" * 70)
    print("EXPERIMENT 4: Direct Text-to-Image Retrieval (CLIP Zero-Shot)")
    print(f"Model: {model_name} | Corpus: Museum A (val)")
    print("=" * 70)

    # 1. Load pre-computed CLIP image embeddings
    image_embeddings, painting_ids = generate_clip_image_embeddings(model_name=model_name)
    image_embeddings_tensor = torch.tensor(image_embeddings)
    print(f"Loaded {len(painting_ids)} image embeddings of dimension {image_embeddings.shape[1]}.")

    # 2. Load CLIP text encoder
    print(f"Loading CLIP text encoder for {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 3. Define retrieval function
    def retrieve(query: str, top_k: int = 10) -> list[str]:
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_vec = text_features.cpu()

        scores = torch.matmul(image_embeddings_tensor, text_vec.T).squeeze(-1).numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [painting_ids[i] for i in top_indices]

    # 4. Evaluate on benchmark
    benchmark = get_benchmark_queries()
    eval_benchmark = [q for q in benchmark if q.get("relevant_ids")]

    overall_metrics = evaluate_retrieval_benchmark(
        retrieve, eval_benchmark, k_values=(5, 10)
    )

    categories = sorted(set(q["category"] for q in eval_benchmark))
    print("\n--- RESULTS BY QUERY CATEGORY (E4: CLIP Image vs E0/E2) ---")
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
        f"{'OVERALL (E4)':<18} | {overall_metrics['num_queries']:<7} | "
        f"{overall_metrics['recall_at_5'] * 100:>8.1f}% | "
        f"{overall_metrics['recall_at_10'] * 100:>8.1f}% | "
        f"{overall_metrics['mrr']:>8.3f}"
    )
    print(f"Mean latency per query: {overall_metrics['latency_ms']:.2f} ms")

    # 5. Log results in results/nlp_experiments.csv
    save_nlp_result(
        name="E4_clip_direct_image",
        method=f"CLIP Direct Image ({model_name})",
        corpus="val_A",
        metrics=overall_metrics,
        settings={
            "encoder": model_name,
            "text_source": "raw_query_to_image_embedding",
            "dim": int(image_embeddings.shape[1]),
        },
    )

    return overall_metrics


if __name__ == "__main__":
    run_experiment_4()
