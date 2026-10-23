"""Judgment and Evaluation module supporting Human and LLM-as-a-Judge protocols.

Implements relevance pooling, nDCG@k, and persistent judgment storage (ADR 0005 & LLMOps Session 3).
Judgments are scored on a 3-point Likert scale:
  0 = Irrelevant (no visual or semantic relation)
  1 = Somewhat relevant (thematic overlap, related mood or setting)
  2 = Clearly relevant (direct match to the query concept)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "wikiart_genre5"
JUDGMENTS_FILE = DATA_DIR / "judgments.json"


# --------------------------------------------------------------------------------------------------
# 1. Graded Information Retrieval Metrics (nDCG@k, Precision@k)
# --------------------------------------------------------------------------------------------------
def dcg_at_k(scores: Sequence[int | float], k: int = 5) -> float:
    """Discounted Cumulative Gain at rank k."""
    scores_k = np.asarray(scores[:k], dtype=float)
    if scores_k.size == 0:
        return 0.0
    gains = 2.0 ** scores_k - 1.0
    discounts = np.log2(np.arange(2, scores_k.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(scores: Sequence[int | float], ideal_scores: Sequence[int | float] | None = None, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at rank k (0.0 to 1.0).

    Parameters
    ----------
    scores : Sequence[int | float]
        Relevance scores of retrieved paintings in ranked order.
    ideal_scores : Sequence[int | float], optional
        Ideal ranking scores. If None, sorts `scores` descending.
    k : int
        Cutoff rank.
    """
    actual_dcg = dcg_at_k(scores, k=k)
    if ideal_scores is None:
        ideal_scores = sorted(scores, reverse=True)
    best_dcg = dcg_at_k(ideal_scores, k=k)
    return float(actual_dcg / best_dcg) if best_dcg > 0.0 else 0.0


def precision_at_k_graded(scores: Sequence[int | float], threshold: int = 1, k: int = 5) -> float:
    """Share of top-k paintings with relevance score >= threshold."""
    scores_k = scores[:k]
    if not scores_k:
        return 0.0
    return float(sum(1.0 for s in scores_k if s >= threshold) / len(scores_k))


# --------------------------------------------------------------------------------------------------
# 2. Persistent Judgment Storage (Pooling Cache)
# --------------------------------------------------------------------------------------------------
def load_judgments() -> dict[str, dict[str, int]]:
    """Load persistent judgments from judgments.json: {query: {painting_id: score}}."""
    if JUDGMENTS_FILE.exists():
        try:
            return json.loads(JUDGMENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_judgment(query: str, painting_id: str, score: int, judge: str = "human") -> None:
    """Save or update a single judgment."""
    judgments = load_judgments()
    if query not in judgments:
        judgments[query] = {}
    judgments[query][painting_id] = int(score)
    JUDGMENTS_FILE.write_text(json.dumps(judgments, indent=2), encoding="utf-8")


def batch_save_judgments(new_judgments: dict[str, dict[str, int]]) -> None:
    """Merge and save a batch of judgments."""
    judgments = load_judgments()
    for q, p_map in new_judgments.items():
        if q not in judgments:
            judgments[q] = {}
        judgments[q].update(p_map)
    JUDGMENTS_FILE.write_text(json.dumps(judgments, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# 3. Judge Prompt Template (LLM / VLM as a Judge)
# --------------------------------------------------------------------------------------------------
JUDGE_RUBRIC = """You are an expert art curator evaluating a natural-language search query for paintings.
Your task is to judge the relevance of a retrieved painting for the user's query.

User query: "{query}"

Painting Metadata:
- Genre: {genre}
- Style: {style}
- Artist: {artist}
- Visual Description: {description}

Scoring criteria:
- 2 (Clearly Relevant): The painting directly depicts the subject, objects, scene or concepts requested in the query.
- 1 (Somewhat Relevant): The painting relates thematically, shares a similar mood, setting, or partial elements, but lacks key visual elements requested.
- 0 (Irrelevant): The painting does not match the requested subject or setting.

Respond ONLY with a JSON object:
{
  "score": <0, 1, or 2>,
  "reasoning": "<brief explanation in 1 sentence>"
}
"""


def format_judge_prompt(query: str, row: pd.Series, description: str = "N/A") -> str:
    """Format the evaluation prompt for a single query-painting pair."""
    return JUDGE_RUBRIC.format(
        query=query,
        genre=str(row.get("genre", row.get("label", "N/A"))),
        style=str(row.get("style", "N/A")),
        artist=str(row.get("artist", "Unknown Artist")).replace("-", " "),
        description=description,
    )
