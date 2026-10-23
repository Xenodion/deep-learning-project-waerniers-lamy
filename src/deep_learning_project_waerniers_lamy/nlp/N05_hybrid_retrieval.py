"""Experiment 5: Hybrid Multimodal Retrieval (Image + Description + Metadata).

Combines direct CLIP image similarity with VLM description semantic similarity:
    score = alpha * sim_image + (1 - alpha) * sim_text

Evaluates multiple alpha weights (0.0 to 1.0) strictly on Museum A (val), ADR 0003 & 0005.
Selects the optimal trade-off and logs results into results/nlp_experiments.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from deep_learning_project_waerniers_lamy.nlp.benchmark import get_benchmark_queries
from deep_learning_project_waerniers_lamy.nlp.E01_generate_descriptions import (
    load_cached_descriptions,
)
from deep_learning_project_waerniers_lamy.nlp.E04_generate_clip_embeddings import (
    generate_clip_image_embeddings,
)
from deep_learning_project_waerniers_lamy.nlp.utils import (
    evaluate_retrieval_benchmark,
    load_paintings,
    save_nlp_result,
)


def run_experiment_5(
    clip_model_name: str = "openai/clip-vit-base-patch32",
    text_model_name: str = "all-MiniLM-L6-v2",
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, float]:
    print("=" * 70)
    print("EXPERIMENT 5: Hybrid Multimodal Retrieval")
    print(f"CLIP: {clip_model_name} | Text: {text_model_name} | Corpus: Museum A (val)")
    print("=" * 70)

    # 1. Load paintings, descriptions, and CLIP image embeddings
    paintings = load_paintings(split="val")
    descriptions = load_cached_descriptions()
    paintings["description"] = paintings["id"].map(descriptions).fillna("")

    clip_embeddings, clip_ids = generate_clip_image_embeddings(model_name=clip_model_name)
    # Align clip embeddings with paintings DataFrame
    clip_dict = {pid: emb for pid, emb in zip(clip_ids, clip_embeddings)}
    aligned_clip_matrix = np.stack([clip_dict[pid] for pid in paintings["id"]])
    clip_tensor = torch.tensor(aligned_clip_matrix)

    # 2. Compute text embeddings for paintings
    def format_searchable_text(row: pd.Series) -> str:
        desc = row["description"]
        artist = str(row["artist"]).replace("-", " ")
        style = str(row["style"]).replace("_", " ")
        genre = str(row["genre"]).replace("_", " ")
        meta_part = f"{genre} style {style}" if "unknown" in artist.lower() else f"{genre} by {artist}, {style} style"
        return f"{desc}. {meta_part}" if desc else meta_part

    texts = [format_searchable_text(r) for _, r in paintings.iterrows()]
    text_embedder = SentenceTransformer(text_model_name)
    text_doc_embeddings = text_embedder.encode(
        texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )

    # 3. Load CLIP text encoder
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    # 4. Hybrid scoring function
    painting_ids = paintings["id"].tolist()

    def get_hybrid_retriever(alpha: float):
        def retrieve(query: str, top_k: int = 10) -> list[str]:
            # Text similarity score
            text_q_emb = text_embedder.encode(
                query, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
            )
            text_scores = torch.matmul(text_doc_embeddings, text_q_emb).cpu().numpy()

            # Image similarity score (CLIP zero-shot)
            inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = clip_model.get_text_features(**inputs)
                clip_text_f = out.pooler_output if hasattr(out, "pooler_output") else out
                clip_text_f = clip_text_f / clip_text_f.norm(dim=-1, keepdim=True)
            image_scores = torch.matmul(clip_tensor, clip_text_f.cpu().T).squeeze(-1).numpy()

            # Min-Max normalize both score vectors to [0, 1] for fair combination
            def min_max(s):
                rng = s.max() - s.min()
                return (s - s.min()) / rng if rng > 0 else s

            combined = alpha * min_max(image_scores) + (1.0 - alpha) * min_max(text_scores)
            top_indices = np.argsort(combined)[::-1][:top_k]
            return [painting_ids[i] for i in top_indices]

        return retrieve

    # 5. Sweep alpha weights on benchmark
    benchmark = get_benchmark_queries()
    eval_benchmark = [q for q in benchmark if q.get("relevant_ids")]

    print("\n--- HYBRID GRID SEARCH ON MUSEUM A (val) ---")
    print(f"{'Alpha (Image)':<14} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8}")
    print("-" * 50)

    best_alpha = 0.5
    best_recall_10 = -1.0
    best_metrics = {}

    for alpha in alphas:
        retriever = get_hybrid_retriever(alpha)
        metrics = evaluate_retrieval_benchmark(retriever, eval_benchmark, k_values=(5, 10))
        print(
            f"{alpha:<14.2f} | {metrics['recall_at_5'] * 100:>8.1f}% | "
            f"{metrics['recall_at_10'] * 100:>8.1f}% | {metrics['mrr']:>8.3f}"
        )
        if metrics["recall_at_10"] > best_recall_10:
            best_recall_10 = metrics["recall_at_10"]
            best_alpha = alpha
            best_metrics = metrics

    print("-" * 50)
    print(f"Optimal Alpha on Museum A: {best_alpha:.2f} (Recall@10: {best_recall_10 * 100:.1f}%)")

    # 6. Log optimal hybrid run
    save_nlp_result(
        name="E5_hybrid_retrieval",
        method=f"Hybrid CLIP ({clip_model_name}) + VLM ({text_model_name})",
        corpus="val_A",
        metrics=best_metrics,
        settings={
            "alpha_image": best_alpha,
            "alpha_text": round(1.0 - best_alpha, 2),
            "clip_model": clip_model_name,
            "text_model": text_model_name,
        },
    )

    return best_metrics


if __name__ == "__main__":
    run_experiment_5()
