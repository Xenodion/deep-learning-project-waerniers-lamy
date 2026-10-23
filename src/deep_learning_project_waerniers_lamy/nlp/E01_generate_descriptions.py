"""Experiment 1: Generate Visual Descriptions with a VLM (BLIP Image Captioning).

Transforms visual image content into text to overcome the metadata bottleneck.
Implements the LLMOps caching and resilience pattern (Session 2 Slide 25):
- Resumable: skips already processed images.
- Incremental checkpointing: saves periodically to descriptions_val.json.
- Model-agnostic storage: decouples heavy VLM captioning from downstream retrieval.

Evaluated strictly on Museum A (validation split, 480 paintings), ADR 0003 & 0005.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from deep_learning_project_waerniers_lamy.nlp.utils import DATA_DIR, load_paintings

DESCRIPTIONS_FILE = DATA_DIR / "descriptions_val.json"


def load_cached_descriptions() -> dict[str, str]:
    """Load existing descriptions cache."""
    if DESCRIPTIONS_FILE.exists():
        try:
            return json.loads(DESCRIPTIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cached_descriptions(descriptions: dict[str, str]) -> None:
    """Save descriptions cache to JSON."""
    DESCRIPTIONS_FILE.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_descriptions(
    limit: int | None = 50,
    model_name: str = "Salesforce/blip-image-captioning-base",
    save_every: int = 10,
) -> dict[str, str]:
    """Generate visual captions for Museum A validation split images.

    Parameters
    ----------
    limit : int, optional
        Maximum number of images to process in this run. Default is 50 for fast pilot inspection.
        Pass None to process all remaining uncaptioned validation images.
    model_name : str
        HuggingFace model ID.
    save_every : int
        Checkpoint interval.
    """
    paintings = load_paintings(split="val")
    cache = load_cached_descriptions()
    print(f"Loaded {len(paintings)} paintings from Museum A (val). Already cached: {len(cache)}")

    # Filter to remaining images
    to_process = paintings[~paintings["id"].isin(cache.keys())].copy()
    if limit is not None:
        to_process = to_process.head(limit)

    if len(to_process) == 0:
        print("All requested images are already cached! Nothing to generate.")
        return cache

    print(f"Generating descriptions for {len(to_process)} images using {model_name}...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    start_time = time.time()
    count = 0

    for idx, (_, row) in enumerate(to_process.iterrows(), start=1):
        p_id = row["id"]
        rel_path = row["path"]
        img_path = DATA_DIR / rel_path

        if not img_path.exists():
            print(f"Warning: file not found {img_path}, skipping.")
            continue

        try:
            raw_image = Image.open(img_path).convert("RGB")
            inputs = processor(raw_image, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=45)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()

            cache[p_id] = caption
            count += 1

            if count % save_every == 0 or idx == len(to_process):
                save_cached_descriptions(cache)
                elapsed = time.time() - start_time
                avg_speed = elapsed / count
                print(f"  [{idx}/{len(to_process)}] Cached {count} descriptions ({avg_speed:.2f}s/img) - Last: '{caption}'")

        except Exception as e:
            print(f"Error processing {p_id} ({rel_path}): {e}")

    save_cached_descriptions(cache)
    total_time = time.time() - start_time
    print(f"Finished generating {count} descriptions in {total_time:.1f}s ({total_time / max(1, count):.2f}s/img).")
    print(f"Total cached descriptions in {DESCRIPTIONS_FILE}: {len(cache)}")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VLM descriptions for paintings (E1).")
    parser.add_argument("--limit", type=int, default=50, help="Number of images to process (default: 50, None=all)")
    parser.add_argument("--all", action="store_true", help="Process all validation images")
    args = parser.parse_args()

    limit = None if args.all else args.limit
    generate_descriptions(limit=limit)


if __name__ == "__main__":
    main()
