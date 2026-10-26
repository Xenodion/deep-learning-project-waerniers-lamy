"""Build the self-contained demo bundle in app/ (Museum B only: the paintings the model never saw).

    uv run --with gradio python scripts/build_app.py

Needs data/wikiart_genre5 (images + split.csv) and results/final_model.pt. Writes images, thumbnails, the precomputed
predictions of the final model, the SigLIP embeddings, and the app itself (a copy of demo.py). The Docker image of
the online demo is built from this folder (deploy/Dockerfile).
"""
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import demo  # loads the final vision model
from deep_learning_project_waerniers_lamy_mahieddine.nlp import utils as nlp_utils

DATA_DIR = REPO_ROOT / "data" / "wikiart_genre5"
OUT = REPO_ROOT / "app"
THUMB_SIZE = 160
RECORDED_RUN = "resnet18_layer4"  # the final run of results/experiments.csv (adr/0012)
GRADIO_VERSION = "6.30.0"  # the Gradio version the app was tested with

WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_USER_AGENT = "museum-guide-demo/1.0 (student project)"  # Wikipedia asks API clients to identify themselves
WIKIPEDIA_PAGES = {"cityscape": "Cityscape", "landscape": "Landscape painting", "portrait": "Portrait",
                   "religious_painting": "Religious art", "still_life": "Still life"}
NOTE_SENTENCES = 2
CLASSICAL_LABEL = "Classical: colours and edges, forest"  # no network at all
# Vision ladder: name of the run in results/experiments.csv -> label shown in the app (test scores, adr/0012)
VISION_LADDER = {"dense_baseline": "Dense network", "cnn_longer": "Small CNN", "resnet18_head": "ResNet-18, frozen",
                 "resnet18_layer4": "ResNet-18, last block trained"}
DEPLOYED_RUNG = "R4 siglip image"
JUDGE_KAPPA, PROJECT_COST_USD, HYBRID_P_VALUE = 0.65, 3.38, 0.78  # adr/9998 (judge on 150 pairs; OpenRouter bill; hybrid vs SigLIP, Museum B)

REQUIREMENTS = f"""gradio=={GRADIO_VERSION}
torch
torchvision
transformers
sentencepiece
protobuf
pandas
pillow
filelock>=4.2.1
numpy
"""


def genre_notes() -> dict[str, dict[str, str]]:
    """First sentences of the Wikipedia article of each genre, with the link to credit (CC BY-SA)."""
    notes = {}
    for genre, page in WIKIPEDIA_PAGES.items():
        request = urllib.request.Request(WIKIPEDIA_SUMMARY + urllib.parse.quote(page.replace(" ", "_")),
                                         headers={"User-Agent": WIKIPEDIA_USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            summary = json.load(response)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", summary["extract"].strip())
        notes[genre] = {"title": summary["title"], "text": " ".join(sentences[:NOTE_SENTENCES]),
                        "url": summary["content_urls"]["desktop"]["page"]}
    return notes


def scoring() -> dict:
    """The numbers of the Scoring tab: test scores of the vision ladder and nDCG@5 of the language ladder."""
    runs = pd.read_csv(REPO_ROOT / "results" / "experiments.csv")
    classical = pd.read_csv(REPO_ROOT / "results" / "classical_baseline.csv").query("chosen")  # P11, adr/0014
    vision = [{"name": label, "test_acc": float(runs.loc[runs["name"] == run, "test_acc"].dropna().iloc[0])}
              for run, label in VISION_LADDER.items()]
    vision.insert(1, {"name": CLASSICAL_LABEL, "test_acc": float(classical["test_acc"].iloc[0])})  # by score: after the dense network
    nlp = REPO_ROOT / "results" / "nlp"
    dev = pd.read_csv(nlp / "dev_summary.csv").set_index("rung")["nDCG@5"]
    test = pd.read_csv(nlp / "test_summary.csv").set_index("rung")["nDCG@5"]
    language = [{"rung": rung, "dev": float(dev[rung]), "test": float(test[rung]), "deployed": rung == DEPLOYED_RUNG}
                for rung in test.index]
    return {"vision": vision, "language": language, "judge_kappa": JUDGE_KAPPA, "project_cost_usd": PROJECT_COST_USD,
            "hybrid_p_value": HYBRID_P_VALUE}


def main() -> None:
    meta = pd.read_csv(DATA_DIR / "metadata.csv").set_index("id")
    split = pd.read_csv(DATA_DIR / "split.csv")
    ids = split.loc[split["split"] == "test", "id"].tolist()
    museum_b = meta.loc[ids].reset_index()
    print(f"Museum B: {len(museum_b)} paintings")

    for sub in ("images", "thumbs"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    rows = []
    for row in museum_b.itertuples():
        name = Path(row.path).name
        for sub in ("images", "thumbs"):
            (OUT / sub / row.label).mkdir(parents=True, exist_ok=True)
        image = Image.open(DATA_DIR / row.path).convert("RGB")
        image.save(OUT / "images" / row.label / name, quality=92)
        thumb = image.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        thumb.save(OUT / "thumbs" / row.label / name, quality=80)
        probs = demo.classify_probs(image)
        rows.append({"id": row.id, "genre": row.label, "style": row.style, "artist": row.artist,
                     "path": f"images/{row.label}/{name}", "thumb": f"thumbs/{row.label}/{name}",
                     "pred": demo.GENRES[int(probs.argmax())], "conf": float(probs.max())})
    collection = pd.DataFrame(rows)
    collection.to_csv(OUT / "collection.csv", index=False)

    accuracy = (collection["pred"] == collection["genre"]).mean()
    runs = pd.read_csv(REPO_ROOT / "results" / "experiments.csv")
    recorded = runs.loc[runs["name"] == RECORDED_RUN, "test_acc"].dropna().iloc[0]
    print(f"Accuracy of the bundled model on Museum B: {accuracy:.1%} (recorded: {recorded:.1%})")
    if abs(accuracy - recorded) > 0.005:
        raise SystemExit("The bundled model does not reproduce the recorded test accuracy: not shipping it.")

    print("Embedding the paintings with SigLIP (a couple of minutes on CPU)...")
    embeddings = nlp_utils.encode_siglip_images([DATA_DIR / p for p in museum_b["path"]])
    np.save(OUT / "siglip_embeddings.npy", embeddings)

    (OUT / "genre_notes.json").write_text(json.dumps(genre_notes(), indent=1), encoding="utf-8")
    (OUT / "scoring.json").write_text(json.dumps(scoring(), indent=1), encoding="utf-8")

    shutil.copy2(REPO_ROOT / "results" / "final_model.pt", OUT / "final_model.pt")
    shutil.copy2(REPO_ROOT / "demo.py", OUT / "app.py")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    print(f"Bundle ready in {OUT}")


if __name__ == "__main__":
    main()
