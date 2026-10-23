"""Fixed evaluation query benchmark for Museum A (validation split, 480 paintings).

Adheres to ADR 0003 and ADR 0005: Museum B is strictly preserved for final testing.
This deck covers:
- Visual content queries (concepts absent from metadata)
- Compound queries (visual + mood / style)
- Explicit metadata queries (artist, style, genre)
- Out-of-domain / negative queries (sanity checks)
"""

from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_QUERIES = [
    # -------------------------------------------------------------------------
    # 1. Visual Concepts (Absent from metadata columns)
    # -------------------------------------------------------------------------
    {
        "id": "q01",
        "query": "turbulent stormy sea with large waves",
        "category": "visual_concept",
        "expected_genre": "landscape",
        "relevant_ids": ["s51_r1044", "s51_r1046", "s51_r1120", "s54_r0165"],
    },
    {
        "id": "q02",
        "query": "quiet river flowing through a green forest",
        "category": "visual_concept",
        "expected_genre": "landscape",
        "relevant_ids": ["s51_r1023", "s51_r1089", "s54_r0073", "s54_r0076"],
    },
    {
        "id": "q03",
        "query": "snowy winter landscape with snow covered trees",
        "category": "visual_concept",
        "expected_genre": "landscape",
        "relevant_ids": ["s42_r0607", "s51_r1113", "s54_r0120"],
    },
    {
        "id": "q04",
        "query": "vase with vibrant colourful flowers on a table",
        "category": "visual_concept",
        "expected_genre": "still_life",
        "relevant_ids": ["s54_r0278", "s54_r0459", "s54_r0567", "s54_r0618"],
    },
    {
        "id": "q05",
        "query": "bowl of fresh fruit with apples, pears and grapes",
        "category": "visual_concept",
        "expected_genre": "still_life",
        "relevant_ids": ["s54_r0346", "s54_r0638", "s54_r0791"],
    },
    {
        "id": "q06",
        "query": "portrait of a woman wearing a red dress or red scarf",
        "category": "visual_concept",
        "expected_genre": "portrait",
        "relevant_ids": ["s49_r0212", "s49_r0243", "s49_r0315"],
    },
    {
        "id": "q07",
        "query": "elderly man with a grey beard and wrinkled face",
        "category": "visual_concept",
        "expected_genre": "portrait",
        "relevant_ids": ["s49_r0256", "s49_r0280", "s49_r0302"],
    },
    {
        "id": "q08",
        "query": "busy boulevard with horse carriages and pedestrians in evening light",
        "category": "visual_concept",
        "expected_genre": "cityscape",
        "relevant_ids": ["s51_r0032", "s51_r0044", "s51_r0070", "s51_r0274"],
    },
    {
        "id": "q09",
        "query": "stone bridge across a canal in an old European city",
        "category": "visual_concept",
        "expected_genre": "cityscape",
        "relevant_ids": ["s51_r0382", "s51_r0572", "s51_r0573", "s51_r0798"],
    },
    {
        "id": "q10",
        "query": "Virgin Mary holding baby Jesus in her arms",
        "category": "visual_concept",
        "expected_genre": "religious_painting",
        "relevant_ids": ["s42_r0012", "s42_r0045", "s42_r0078", "s42_r0110"],
    },
    {
        "id": "q11",
        "query": "crucifixion of Christ on Mount Calvary",
        "category": "visual_concept",
        "expected_genre": "religious_painting",
        "relevant_ids": ["s42_r0155", "s42_r0189", "s42_r0210"],
    },
    {
        "id": "q12",
        "query": "golden angels surrounded by divine heavenly light",
        "category": "visual_concept",
        "expected_genre": "religious_painting",
        "relevant_ids": ["s42_r0245", "s42_r0288", "s42_r0312"],
    },

    # -------------------------------------------------------------------------
    # 2. Compound Queries (Visual subject + style / atmosphere)
    # -------------------------------------------------------------------------
    {
        "id": "q13",
        "query": "dramatic nocturnal cityscape with glowing street lanterns",
        "category": "compound",
        "expected_genre": "cityscape",
        "relevant_ids": ["s51_r0032", "s51_r0044", "s51_r0572"],
    },
    {
        "id": "q14",
        "query": "sunlit impressionist country landscape with golden harvest fields",
        "category": "compound",
        "expected_genre": "landscape",
        "relevant_ids": ["s51_r1023", "s54_r0073"],
    },
    {
        "id": "q15",
        "query": "melancholic expressionist portrait with distorted emotional features",
        "category": "compound",
        "expected_genre": "portrait",
        "relevant_ids": ["s49_r0212", "s49_r0243"],
    },
    {
        "id": "q16",
        "query": "geometric cubist still life with wine bottle and musical instrument",
        "category": "compound",
        "expected_genre": "still_life",
        "relevant_ids": ["s54_r0234", "s54_r0235", "s54_r0251"],
    },

    # -------------------------------------------------------------------------
    # 3. Explicit Metadata Queries (Tests baseline retrieval on metadata)
    # -------------------------------------------------------------------------
    {
        "id": "q17",
        "query": "Paris street scene painted by Edouard Cortes",
        "category": "metadata",
        "expected_genre": "cityscape",
        "relevant_ids": ["s51_r0032", "s51_r0044", "s51_r0070", "s51_r0274", "s51_r0382"],
    },
    {
        "id": "q18",
        "query": "landscape painting by Isaac Levitan",
        "category": "metadata",
        "expected_genre": "landscape",
        "relevant_ids": ["s42_r0607", "s51_r1023", "s51_r1089", "s51_r1113", "s54_r0073"],
    },
    {
        "id": "q19",
        "query": "still life artwork by Paul Gauguin",
        "category": "metadata",
        "expected_genre": "still_life",
        "relevant_ids": ["s54_r0278", "s54_r0346", "s54_r0459", "s54_r0567", "s54_r0618"],
    },
    {
        "id": "q20",
        "query": "baroque religious painting",
        "category": "metadata",
        "expected_genre": "religious_painting",
        "relevant_ids": ["s42_r0012", "s42_r0045", "s42_r0155"],
    },
    {
        "id": "q21",
        "query": "renaissance portrait",
        "category": "metadata",
        "expected_genre": "portrait",
        "relevant_ids": ["s49_r0256", "s49_r0280"],
    },
    {
        "id": "q22",
        "query": "pop art still life",
        "category": "metadata",
        "expected_genre": "still_life",
        "relevant_ids": ["s54_r0234", "s54_r0235"],
    },

    # -------------------------------------------------------------------------
    # 4. Out-of-Domain / Negative Queries (Should yield low similarity scores)
    # -------------------------------------------------------------------------
    {
        "id": "q23",
        "query": "red supersonic jet flying over a skyscraper",
        "category": "negative",
        "expected_genre": None,
        "relevant_ids": [],
    },
    {
        "id": "q24",
        "query": "modern electric car charging at a station",
        "category": "negative",
        "expected_genre": None,
        "relevant_ids": [],
    },
]


def get_benchmark_queries(category: str | None = None) -> list[dict]:
    """Return the benchmark queries, optionally filtered by category."""
    if category is None:
        return BENCHMARK_QUERIES
    return [q for q in BENCHMARK_QUERIES if q["category"] == category]


def save_benchmark_json(output_path: Path | None = None) -> Path:
    """Save the benchmark queries to a JSON file."""
    path = output_path or (Path(__file__).parent / "benchmark_queries_val.json")
    path.write_text(json.dumps(BENCHMARK_QUERIES, indent=2), encoding="utf-8")
    return path
