"""Generate and cache direct CLIP image embeddings for Museum A validation split (E4).

Encodes visual pixels directly into the shared vision-language space (openai/clip-vit-base-patch32).
Saves persistent artifacts:
  - data/wikiart_genre5/embeddings/clip_image_embeddings_val.npy (shape N x 512, float32)
  - data/wikiart_genre5/embeddings/clip_image_ids_val.json (list of painting ids)

Decoupled and cached (LLMOps Session 2 Slide 25).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from deep_learning_project_waerniers_lamy.nlp.utils import DATA_DIR, load_paintings

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CLIP_EMB_FILE = EMBEDDINGS_DIR / "clip_image_embeddings_val.npy"
CLIP_IDS_FILE = EMBEDDINGS_DIR / "clip_image_ids_val.json"


def generate_clip_image_embeddings(
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    force: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Compute and save CLIP image embeddings for Museum A validation split."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not force and CLIP_EMB_FILE.exists() and CLIP_IDS_FILE.exists():
        print(f"Loading existing CLIP embeddings from {CLIP_EMB_FILE}...")
        embeddings = np.load(CLIP_EMB_FILE)
        ids = json.loads(CLIP_IDS_FILE.read_text(encoding="utf-8"))
        print(f"Loaded {len(ids)} embeddings with dimension {embeddings.shape[1]}.")
        return embeddings, ids

    paintings = load_paintings(split="val")
    print(f"Loading CLIP model {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_embeddings = []
    all_ids = []
    start_time = time.time()

    print(f"Encoding {len(paintings)} validation images into CLIP space (batch_size={batch_size})...")
    for start_idx in range(0, len(paintings), batch_size):
        batch_rows = paintings.iloc[start_idx : start_idx + batch_size]
        images = []
        valid_ids = []

        for _, row in batch_rows.iterrows():
            img_path = DATA_DIR / row["path"]
            if img_path.exists():
                images.append(Image.open(img_path).convert("RGB"))
                valid_ids.append(row["id"])
            else:
                print(f"Warning: missing image {img_path}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            image_features = out.pooler_output if hasattr(out, "pooler_output") else out
            # L2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy().astype(np.float32))
            all_ids.extend(valid_ids)

        print(f"  Processed [{min(start_idx + batch_size, len(paintings))}/{len(paintings)}] images...")

    embeddings_matrix = np.vstack(all_embeddings)
    np.save(CLIP_EMB_FILE, embeddings_matrix)
    CLIP_IDS_FILE.write_text(json.dumps(all_ids), encoding="utf-8")

    total_time = time.time() - start_time
    print(f"Saved {len(all_ids)} image embeddings ({embeddings_matrix.shape}) in {total_time:.1f}s.")
    return embeddings_matrix, all_ids


if __name__ == "__main__":
    generate_clip_image_embeddings()
