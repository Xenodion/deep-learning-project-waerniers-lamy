"""Museum Guide: the web demo of both models (genre classifier + collection search).

The same file is the app of the Cloud Run bundle (app/app.py). It reads a bundle folder made by
scripts/build_app.py (collection.csv, thumbs/, images/, siglip_embeddings.npy, final_model.pt, genre_notes.json,
scoring.json).

    uv run --with gradio python scripts/build_app.py     # once: builds app/ from data/ and results/
    uv run --with gradio python demo.py                  # then: http://localhost:7860
"""
import copy
import html
import json
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

APP_DIR = Path(__file__).resolve().parent
BUNDLE = APP_DIR if (APP_DIR / "collection.csv").exists() else APP_DIR / "app"
MAX_CONCURRENT_JOBS, MAX_QUEUED_JOBS = 2, 16
MAX_QUERY_CHARS = 200  # SigLIP reads 64 tokens anyway
UPLOAD_LIMIT_BYTES = 1  # the app takes no upload: this also shuts gradio's own upload route
UPLOAD_SIDE = 1024  # an uploaded picture is shrunk to this before SigLIP
Image.MAX_IMAGE_PIXELS = 25_000_000  # Pillow refuses above twice this (50 Mpx, ~150 MB in RGB): no decompression bomb

GENRES = ["cityscape", "landscape", "portrait", "religious_painting", "still_life"]
GENRE_NAME = {"cityscape": "Cityscape", "landscape": "Landscape", "portrait": "Portrait",
              "religious_painting": "Religious painting", "still_life": "Still life"}
SIGLIP_MODEL = "google/siglip-base-patch16-224"
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEARCH_RESULTS = 6
EXAMPLE_QUERIES = ["A stormy sea with big waves.", "A portrait of a woman wearing a hat.", "A snowy winter landscape.",
                   "A busy city street with carriages.", "The Virgin Mary with the infant Jesus.", "A still life with fruit and wine.",
                   "A quiet village at sunset.", "A child playing outdoors.", "Flowers in a vase.", "A dark and gloomy forest.",
                   "A sailing boat on calm water."]


# --------------------------------------------------------------------------------------------------
# 1. Vision: ResNet-18 pretrained on ImageNet, last block fine-tuned (adr/0012)
# --------------------------------------------------------------------------------------------------
_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
_stem = nn.Sequential(_resnet.conv1, _resnet.bn1, _resnet.relu, _resnet.maxpool,
                      _resnet.layer1, _resnet.layer2, _resnet.layer3).eval()  # frozen part
_head = nn.Sequential(copy.deepcopy(_resnet.layer4), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                      nn.Dropout(0.0), nn.Linear(512, len(GENRES)))
_weights = next(p for p in (BUNDLE / "final_model.pt", APP_DIR / "results" / "final_model.pt") if p.exists())
_head.load_state_dict(torch.load(_weights, map_location="cpu", weights_only=True))
_head.eval()


def classify_probs(image: Image.Image) -> np.ndarray:
    """Probabilities of the 5 genres (same order as GENRES) for one painting."""
    with torch.no_grad():
        x = _transform(image.convert("RGB")).unsqueeze(0)
        return torch.softmax(_head(_stem(x)), dim=1)[0].numpy()


# --------------------------------------------------------------------------------------------------
# 2. Collection search: SigLIP text or image tower against precomputed image embeddings
# --------------------------------------------------------------------------------------------------
_siglip = None


def _load_siglip():
    global _siglip
    if _siglip is None:
        from transformers import AutoModel, AutoProcessor
        _siglip = (AutoProcessor.from_pretrained(SIGLIP_MODEL), AutoModel.from_pretrained(SIGLIP_MODEL).eval())
    return _siglip


def _normalised(out) -> np.ndarray:
    # transformers returns a tensor or, in recent versions, a model output holding it; same as nlp/utils._pooled
    vec = out if isinstance(out, torch.Tensor) else out.pooler_output
    return torch.nn.functional.normalize(vec, dim=-1).numpy()


def rank(vec: np.ndarray, embeddings: np.ndarray, k: int) -> list[int]:
    """Rows of the k paintings closest to vec (cosine: both sides are L2-normalised)."""
    return list(np.argsort(-(vec @ embeddings.T)[0])[:k])


def search_order(query: str, embeddings: np.ndarray, k: int) -> list[int]:
    processor, model = _load_siglip()
    # SigLIP was trained with max_length padding; other paddings silently degrade its text embeddings.
    inputs = processor(text=[query], padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        return rank(_normalised(model.get_text_features(**inputs)), embeddings, k)


def embed_image(image: Image.Image) -> np.ndarray:
    """SigLIP vector of one picture, made exactly like siglip_embeddings.npy (nlp/utils.encode_siglip_images)."""
    processor, model = _load_siglip()
    inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        return _normalised(model.get_image_features(**inputs))


def image_search_order(image: Image.Image, embeddings: np.ndarray, k: int) -> list[int]:
    image = image.copy()
    image.thumbnail((UPLOAD_SIDE, UPLOAD_SIDE))  # SigLIP shrinks it to 224 anyway; this bounds the work before
    return rank(embed_image(image), embeddings, k)


def timed(fn, *args):
    """Run fn(*args) and return (result, milliseconds it took)."""
    start = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - start) * 1000


# --------------------------------------------------------------------------------------------------
# 3. Views (plain HTML, styled by CSS below)
# --------------------------------------------------------------------------------------------------
def _artist(row) -> str:
    if row["artist"] == "Unknown Artist":
        return "Unknown artist"
    return row["artist"].replace("ã­", "í").replace("-", " ").title()  # one slug in the dataset reads "joaquãn"


def meter_html(ms: float) -> str:
    """The one cost figure of the demo: how long the model took here, just now."""
    return f'<p class="meter">Answered in {ms:.0f} ms on {torch.get_num_threads()} CPU threads</p>'


def bars_html(items: list[tuple[str, float, str]], top: int | None = 0, wide: bool = False) -> str:
    """Horizontal bars: items are (label, share between 0 and 1, text at the end); `top` is the highlighted row."""
    rows = "".join(
        f'<div class="bar{" top" if k == top else ""}"><span class="bar-name">{html.escape(name)}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{share * 100:.1f}%"></span></span>'
        f'<span class="bar-pct">{text}</span></div>' for k, (name, share, text) in enumerate(items))
    return f'<div class="bars{" wide" if wide else ""}">{rows}</div>'


def probs_bars(probs: np.ndarray) -> str:
    order = np.argsort(-probs)
    return bars_html([(GENRE_NAME[GENRES[i]], float(probs[i]), f"{probs[i] * 100:.0f}%") for i in order])


def about_html(note: dict[str, str]) -> str:
    return (f'<div class="about"><p>{html.escape(note["text"])}</p>'
            f'<p class="credit"><a href="{note["url"]}" target="_blank" rel="noopener">Read more on Wikipedia</a>'
            f' (text under CC BY-SA)</p></div>')


def verdict_html(probs: np.ndarray, row: pd.Series | None, ms: float, notes: dict) -> str:
    guess = GENRES[int(np.argmax(probs))]
    if row is None:
        badge = f'<p class="said">The model sees a <strong>{GENRE_NAME[guess].lower()}</strong>.</p>'
        label = ""
    else:
        ok = guess == row["genre"]
        label = (f'<p class="wall-label"><span>{html.escape(str(row["style"]).replace("_", " "))}</span> by '
                 f'{html.escape(_artist(row))}<br>Museum label: <strong>{GENRE_NAME[row["genre"]]}</strong></p>')
        badge = (f'<p class="said {"ok" if ok else "miss"}">The model says <strong>{GENRE_NAME[guess].lower()}</strong>: '
                 f'{"it matches the museum label." if ok else "it differs from the museum label."}</p>')
    return f'<div class="card">{label}{badge}{probs_bars(probs)}{about_html(notes[guess])}{meter_html(ms)}</div>'


def _url(relative: str) -> str:
    return f"/gradio_api/file={(BUNDLE / relative).resolve().as_posix()}"


def wall_html(collection: pd.DataFrame, rows: list[int]) -> str:
    tiles = "".join(
        f'<button class="wall-item" data-row="{i}" aria-label="{GENRE_NAME[collection.loc[i, "genre"]]}">'
        f'<img loading="lazy" decoding="async" src="{_url(collection.loc[i, "thumb"])}" alt=""></button>' for i in rows)
    return f'<div class="wall">{tiles}</div>'


def found_html(collection: pd.DataFrame, rows: list[int], ms: float | None = None) -> str:
    if not rows:
        return '<p class="hint centered">Type a description above, or pick one of the ideas.</p>'
    cards = "".join(
        f'<figure class="found-item"><img src="{_url(collection.loc[i, "path"])}" alt="">'
        f'<figcaption><strong>{rank}.</strong> {GENRE_NAME[collection.loc[i, "genre"]]}<br>'
        f'<span>{html.escape(_artist(collection.loc[i]))}</span></figcaption></figure>' for rank, i in enumerate(rows, 1))
    return f'<div class="found">{cards}</div>{meter_html(ms) if ms is not None else ""}'


PICK_JS = """<script>
document.addEventListener('click', (e) => {
  const tile = e.target.closest('.wall-item');
  const go = document.querySelector('#pick-go');
  if (!tile || !go) return;
  document.querySelectorAll('.wall-item.selected').forEach((t) => t.classList.remove('selected'));
  tile.classList.add('selected');
  window.pickedRow = tile.dataset.row;  // read back by the js= hook of the hidden button
  go.click();
  document.querySelector('#stage')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
</script>"""


EMPTY_HTML = ('<div class="card empty"><p>Choose a painting below.</p>'
              '<p class="hint">The model reads the picture and names its genre.</p></div>')

BRIEF_CONTEXT = ("A fictional French fine-arts museum wants a guide for visitors' phones and gallery kiosks. The kiosks have modest "
                 "CPUs, the team is small and there is no budget for recurring services.")
BRIEF_ASKS = [
    ("Recognise a painting", "A visitor points a camera at a painting. The guide names its genre and tells the story of that tradition."),
    ("Answer in plain words", "A visitor types what they want to see. The guide proposes the artworks of the collection that match."),
    ("Is a neural network worth it?", "The director wants to know if it beats a classical approach, once speed and cost are counted."),
]
METHOD_POINTS = [
    ("Museum A (train and validation) and Museum B (test) share no named artist: a painter's style alone gives away the genre "
     "of 70% of the paintings that have one."),
    "Every choice is made on validation. Museum B is scored once per model.",
    "Image statistics used to normalise the pictures come from the training images only.",
]


def brief_html() -> str:
    rows = "".join(f'<div class="ask"><h3>{html.escape(asked)}</h3><p>{html.escape(done)}</p></div>' for asked, done in BRIEF_ASKS)
    return (f'<div class="card"><h3>The brief</h3><p class="said">{html.escape(BRIEF_CONTEXT)}</p>'
            f'<div class="asks">{rows}</div></div>')


def method_html() -> str:
    points = "".join(f"<li>{html.escape(point)}</li>" for point in METHOD_POINTS)
    return f'<div class="card"><h3>How the test was kept honest</h3><ul class="method">{points}</ul></div>'


LANGUAGE_NAMES = {"R0 random": "Random", "R1 metadata": "Metadata", "R2 description": "Generated description",
                  "R3 description+metadata": "Description + metadata", "R4 siglip image": "SigLIP on the image",
                  "R5 hybrid": "Hybrid"}


def vision_scoring_html(df: pd.DataFrame, scoring: dict) -> str:
    hits = df["pred"] == df["genre"]
    ladder = bars_html([(m["name"], m["test_acc"], f'{m["test_acc"] * 100:.1f}%') for m in scoring["vision"]],
                       top=len(scoring["vision"]) - 1, wide=True)
    matrix = pd.crosstab(df["genre"], df["pred"]).reindex(index=GENRES, columns=GENRES, fill_value=0)
    cells = "".join(
        "<tr><th>" + GENRE_NAME[t] + "</th>" + "".join(
            f'<td class="{"diag" if t == p else ""}" style="--v:{matrix.loc[t, p] / matrix.loc[t].sum():.2f}">'
            f'{matrix.loc[t, p]}</td>' for p in GENRES) + "</tr>" for t in GENRES)
    head = "".join(f"<th>{GENRE_NAME[g]}</th>" for g in GENRES)
    return (f'<div class="card"><h3>Genre recognition</h3><p class="big">{hits.mean() * 100:.1f}%</p>'
            f'<p class="said">of the {len(df)} paintings of Museum B are named correctly.</p>'
            f'<p class="hint">Each model, from the simplest, scored once on Museum B:</p>{ladder}'
            f'<p class="hint spaced">What the final model answered (rows: true genre, columns: its answer):</p>'
            f'<div class="matrix-wrap"><table class="matrix"><tr><th></th>{head}</tr>{cells}</table></div></div>')


def language_scoring_html(scoring: dict) -> str:
    rungs = scoring["language"]
    hybrid = next(r for r in rungs if "hybrid" in r["rung"])
    deployed = next(r for r in rungs if r["deployed"])
    names = {r["rung"]: LANGUAGE_NAMES[r["rung"]] + (" (in this app)" if r["deployed"] else "") for r in rungs}
    ladder = bars_html([(names[r["rung"]], r["test"], f'{r["test"] * 100:.1f}%') for r in rungs],
                       top=next(k for k, r in enumerate(rungs) if r["deployed"]), wide=True)
    return (f'<div class="card"><h3>Language search</h3><p class="big">{deployed["test"] * 100:.1f}%</p>'
            f'<p class="said">Match with the question across the top 5 results (nDCG@5).</p>'
            f'<p class="hint">Each way of searching, from the simplest, on the same 40 questions (nDCG@5 as a percentage):</p>{ladder}'
            f'<p class="spaced">Adding generated descriptions (hybrid) scores {hybrid["test"] * 100:.1f}% against {deployed["test"] * 100:.1f}%. '
            f'That gap cannot be told from chance (p = {scoring["hybrid_p_value"]}), so we kept the simpler search: one model, '
            f'no paid service.</p>'
            f'<p class="hint">Answers were graded by an AI that sees only the question and the painting, checked against a human on '
            f'150 pairs (kappa {scoring["judge_kappa"]}). The whole project cost {scoring["project_cost_usd"]} $, almost all of it to '
            f'measure; running the search costs nothing.</p></div>')


# --------------------------------------------------------------------------------------------------
# 4. Look: a gallery wall at night, brass for the one thing that matters
# --------------------------------------------------------------------------------------------------
CSS = """
:root { --wall:#15262a; --panel:#1d3237; --line:#35504f; --ink:#efece4; --muted:#9db4ad; --brass:#d6ab52; --ok:#86c9a4; --miss:#e39484; }
body, .gradio-container { background: var(--wall) !important; color: var(--ink); }
.gradio-container { width: 100% !important; max-width: 1320px !important; margin: 0 auto; padding: 0 clamp(12px, 3vw, 40px) 48px !important;
  font-family: 'Instrument Sans', system-ui, sans-serif; font-size: clamp(15px, 0.35vw + 13.5px, 18px); }
footer { display: none !important; }
* { scrollbar-width: thin; scrollbar-color: var(--line) transparent; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: var(--line); border-radius: 6px; border: 2px solid transparent; background-clip: content-box; }
*::-webkit-scrollbar-thumb:hover { background: var(--muted); background-clip: content-box; }
#app { display: grid; grid-template-columns: auto 1fr; column-gap: 24px; align-items: end; }
#app .tabs { display: contents; }
#hero { grid-column: 1; grid-row: 1; padding: clamp(10px, 1.6vw, 22px) 0 10px; }
#hero h1 { font-family: 'Bricolage Grotesque', system-ui, sans-serif; font-weight: 600; font-size: clamp(1.5rem, 1rem + 1.5vw, 2.2rem); line-height: 1.1; letter-spacing: -0.02em; margin: 0; padding-left: 12px; border-left: 4px solid var(--brass); color: var(--ink); }
#hero p { color: var(--muted); font-size: .88rem; margin: 5px 0 0 16px; }
#app .tab-wrapper { grid-column: 2; grid-row: 1; justify-self: end; align-self: end; margin-bottom: 4px; }
#app .tabitem { grid-column: 1 / -1; grid-row: 2; border-top: 1px solid var(--line); padding-top: 12px; }
button[role=tab] { color: var(--muted) !important; font-size: 1rem; font-weight: 500; }
button[role=tab][aria-selected=true] { color: var(--ink) !important; border-color: var(--brass) !important; }
.block, .form { background: transparent !important; border: none !important; box-shadow: none !important; }
.tab-intro { color: var(--muted); margin: 0 0 14px; font-size: .88rem; line-height: 1.4; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: clamp(16px, 2vw, 28px); height: 100%; box-sizing: border-box; }
.card.empty { display: flex; flex-direction: column; justify-content: center; text-align: center; min-height: 260px; }
.card h3 { margin: 0 0 10px; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.25rem; font-weight: 600; color: var(--ink); }
.card p { margin: 0 0 12px; line-height: 1.5; }
.hint { color: var(--muted); font-size: .92rem; }
.spaced { margin-top: 20px !important; }
.meter { color: var(--muted); font-size: .85rem; margin: 14px 0 0 !important; }
.about { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: .95rem; }
.about p { margin: 0 0 6px; } .about .credit { font-size: .85rem; }
.about a { color: var(--brass); text-underline-offset: 3px; }
.wall-label { border-left: 3px solid var(--brass); padding-left: 12px; color: var(--muted); font-size: .95rem; }
.wall-label span { color: var(--ink); font-weight: 600; }
.wall-label strong { color: var(--ink); }
.said { font-size: 1.1rem; }
.said.ok strong { color: var(--ok); } .said.miss strong { color: var(--miss); }
.big { font-family: 'Bricolage Grotesque', sans-serif; font-size: clamp(2.6rem, 2rem + 2vw, 3.8rem); font-weight: 600; color: var(--brass); line-height: 1; }
.bars { display: grid; gap: 9px; margin-top: 14px; }
.bar { display: grid; grid-template-columns: clamp(6.5rem, 14vw, 9rem) 1fr 2.8rem; align-items: center; gap: 10px; font-size: .95rem; color: var(--muted); }
.bars.wide .bar { grid-template-columns: minmax(7.5rem, 13rem) 1fr 3rem; }
.bar.top { color: var(--ink); font-weight: 600; }
.bar-track { height: 9px; background: #10201f; border-radius: 6px; overflow: hidden; }
.bar-fill { display: block; height: 100%; background: #5d7f78; border-radius: 6px; }
.bar.top .bar-fill { background: var(--brass); }
.bar-pct { text-align: right; font-variant-numeric: tabular-nums; }
.matrix-wrap { overflow-x: auto; }
.matrix, .matrix th, .matrix td, .matrix tr { border: none !important; }
.matrix { border-collapse: separate; border-spacing: 4px; width: 100%; font-size: .85rem; color: var(--muted); }
.matrix th { font-weight: 500; padding: 6px 8px; text-align: center; }
.matrix th:first-child, .matrix tr th:first-child { text-align: left; white-space: nowrap; }
.matrix td { text-align: center; padding: 8px; border-radius: 6px; color: var(--ink);
  background: color-mix(in srgb, var(--brass) calc(var(--v) * 70%), transparent); }
.matrix td.diag { font-weight: 700; }
.scoring { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 20px; align-items: start; }
.asks { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr)); gap: 14px 24px; margin-top: 16px; }
.info { display: grid; gap: 20px; }
.ask { border-left: 3px solid var(--brass); padding-left: 14px; }
.ask h3 { font-size: 1.05rem; margin-bottom: 6px; } .ask p { margin-bottom: 6px; }
.method { margin: 0; padding-left: 1.2rem; color: var(--muted); line-height: 1.6; }
#stage { gap: clamp(14px, 2vw, 28px); align-items: stretch; flex-wrap: wrap !important; }
#stage > div { display: flex; flex-direction: column; flex: 1 1 320px !important; min-width: min(100%, 320px) !important; }
#stage .block, #stage .html-container, #stage .prose { flex: 1; height: 100%; padding: 0; margin: 0; }
#preview { height: auto !important; min-height: clamp(260px, 34vh, 420px); }
#preview > div, #preview .image-container, #preview .upload-container, #preview .image-frame { height: 100% !important; }
#preview, #preview .image-container, #preview .upload-container, #preview .image-frame { background: var(--panel) !important; border: 1px solid var(--line); border-radius: 14px; }
#preview img { object-fit: contain; }
#pick-note { text-align: center; color: var(--muted); font-size: .9rem; margin: 6px 0 0; }
#genre-filter { display: flex; justify-content: center; margin-top: 12px; }
#genre-filter .wrap { justify-content: center; gap: 8px; }
#genre-filter label { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 999px !important; padding: 6px 16px !important; color: var(--muted) !important; cursor: pointer; }
#genre-filter label.selected { background: var(--brass) !important; border-color: var(--brass) !important; color: #1a1a14 !important; font-weight: 600; }
#genre-filter input { display: none; }
.wall { height: clamp(300px, 52vh, 660px); overflow-y: auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(clamp(88px, 10vw, 150px), 1fr));
  grid-auto-rows: max-content; gap: 8px; padding: 10px; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; align-content: start; }
.wall-item { position: relative; padding: 0; border: 2px solid transparent; border-radius: 8px; overflow: hidden; background: #10201f; cursor: pointer; aspect-ratio: 1; }
.wall-item img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; transition: transform .2s; }
.wall-item:hover { border-color: var(--muted); } .wall-item:hover img { transform: scale(1.05); }
.wall-item.selected { border-color: var(--brass); }
.sr-only { position: absolute !important; width: 1px; height: 1px; overflow: hidden; opacity: 0; pointer-events: none; }
.found { display: grid; grid-template-columns: repeat(2, 1fr); gap: clamp(10px, 1.5vw, 20px); }
.found-item { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.found-item img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }
.found-item figcaption { padding: 10px 12px; font-size: .95rem; line-height: 1.4; } .found-item figcaption span { color: var(--muted); }
.centered { text-align: center; }
#found .meter { text-align: center; }
#query textarea, #query input { background: var(--panel) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; font-size: 1.05rem; }
#search-row { align-items: end; gap: 10px; }
.primary-btn, button.primary { background: var(--brass) !important; color: #1a1a14 !important; border: none !important; font-weight: 600; border-radius: 12px !important; }
#chips { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 10px 0 18px; }
.chip { background: var(--panel) !important; color: var(--muted) !important; border: 1px solid var(--line) !important; border-radius: 999px !important; min-width: 0 !important; width: auto !important; flex: 0 0 auto !important; font-size: .9rem !important; }
.chip:hover { color: var(--ink) !important; border-color: var(--brass) !important; }
button:focus-visible, [role=tab]:focus-visible, label:focus-within { outline: 2px solid var(--brass) !important; outline-offset: 2px; }
@media (min-width: 700px) { .found { grid-template-columns: repeat(3, 1fr); } }
@media (min-width: 1100px) { .found { grid-template-columns: repeat(6, 1fr); } }
@media (max-width: 760px) { #app { display: block; } #app .tabs { display: block; } #app .tab-wrapper { margin: 0; width: 100% !important; } #app .tabitem { padding-top: 10px; } }
@media (max-width: 640px) { button[role=tab] { font-size: .8rem; padding: 6px 3px !important; margin: 0 !important; }
  .bar, .bars.wide .bar { grid-template-columns: 1fr 3rem; row-gap: 4px; } .bar-name { grid-column: 1 / -1; } .bar-track { grid-column: 1; } .bar-pct { grid-column: 2; } }
"""
THEME = gr.themes.Base(
    primary_hue="amber", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Instrument Sans"), "system-ui", "sans-serif"],
).set(body_background_fill="#15262a", body_background_fill_dark="#15262a", body_text_color="#efece4",
      body_text_color_dark="#efece4", block_background_fill="transparent", block_background_fill_dark="transparent",
      input_background_fill="#1d3237", input_background_fill_dark="#1d3237", border_color_primary="#35504f",
      border_color_primary_dark="#35504f", button_primary_background_fill="#d6ab52",
      button_primary_background_fill_dark="#d6ab52", button_primary_text_color="#1a1a14",
      button_primary_text_color_dark="#1a1a14")
HEAD = (PICK_JS + '<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?'
        'family=Bricolage+Grotesque:wght@600&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">')


# --------------------------------------------------------------------------------------------------
# 5. App
# --------------------------------------------------------------------------------------------------
def tab_intro(text: str) -> str:
    return f'<p class="tab-intro">{html.escape(text)}</p>'


def build_app() -> gr.Blocks:
    collection = pd.read_csv(BUNDLE / "collection.csv")
    embeddings = np.load(BUNDLE / "siglip_embeddings.npy")
    notes = json.loads((BUNDLE / "genre_notes.json").read_text(encoding="utf-8"))
    scoring = json.loads((BUNDLE / "scoring.json").read_text(encoding="utf-8"))
    shuffled = collection.sample(frac=1, random_state=0).index.to_list()  # genres mixed on "All", same order every run
    filters = ["All", *GENRE_NAME.values()]

    def pick_rows(choice: str) -> list[int]:
        if choice == "All":
            return shuffled
        return [i for i in shuffled if GENRE_NAME[collection.loc[i, "genre"]] == choice]

    def filter_wall(choice: str) -> str:
        return wall_html(collection, pick_rows(choice))

    def select_painting(picked: str):
        row = collection.loc[int(picked)]
        image = Image.open(BUNDLE / row["path"])
        probs, ms = timed(classify_probs, image)
        return image, verdict_html(probs, row, ms, notes)

    def search(query: str) -> str:
        query = query[:MAX_QUERY_CHARS]
        if not query.strip():
            return found_html(collection, [])
        rows, ms = timed(search_order, query, embeddings, SEARCH_RESULTS)
        return found_html(collection, rows, ms)

    with gr.Blocks(title="Museum Guide") as demo, gr.Column(elem_id="app"):
        gr.HTML('<div id="hero"><h1>Museum Guide</h1>'
                '<p>Name the genre of a painting, or find paintings by describing them.</p></div>')
        with gr.Tabs():
            with gr.Tab("1 · Recognise"):
                gr.HTML(tab_intro("Brief, point 1. The visitor points the camera at a painting; the guide recognises its genre "
                                  "and tells the story of that tradition."))
                with gr.Row(elem_id="stage"):
                    # display only: visitors cannot upload anything, the server never decodes a file it did not ship
                    preview = gr.Image(type="pil", interactive=False, show_label=False, elem_id="preview",
                                       height=320, buttons=[])
                    verdict = gr.HTML(EMPTY_HTML)
                wall = gr.HTML(filter_wall("All"))
                genre_filter = gr.Radio(filters, value="All", show_label=False, container=False, elem_id="genre-filter")
                gr.HTML(f'<p id="pick-note">Museum B: {len(collection)} works by artists the model never saw in training. '
                        'Scroll the wall and click one.</p>')
                picked = gr.Textbox(visible=False)  # only there to give the hidden button an input slot
                pick_go = gr.Button("pick", elem_id="pick-go", elem_classes="sr-only")
                genre_filter.change(filter_wall, genre_filter, wall)
                pick_go.click(select_painting, picked, [preview, verdict], js="() => [window.pickedRow]")

            with gr.Tab("2 · Ask"):
                gr.HTML(tab_intro("Brief, point 2. The visitor asks in plain words; the guide proposes the artworks "
                                  "of the collection that best match."))
                with gr.Row(elem_id="search-row"):
                    query = gr.Textbox(placeholder="a stormy sea with big waves", show_label=False,
                                       container=False, scale=5, elem_id="query", max_length=MAX_QUERY_CHARS)
                    go = gr.Button("Search", variant="primary", scale=1)
                with gr.Row(elem_id="chips"):
                    chips = [gr.Button(q, elem_classes="chip", size="sm") for q in EXAMPLE_QUERIES]
                found = gr.HTML(found_html(collection, []), elem_id="found")
                go.click(search, query, found)
                query.submit(search, query, found)
                for chip, text in zip(chips, EXAMPLE_QUERIES):
                    chip.click(lambda t=text: (t, search(t)), None, [query, found])

            with gr.Tab("3 · Info"):
                gr.HTML(f'<div class="info">{brief_html()}'
                        f'<div class="scoring">{vision_scoring_html(collection, scoring)}{language_scoring_html(scoring)}</div>'
                        f'{method_html()}</div>')
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Generate a temporary public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    _load_siglip()  # load once at start-up, so the first search is not slow
    # queue limits keep a public deployment from being swamped: 2 jobs at a time, at most 16 waiting
    build_app().queue(default_concurrency_limit=MAX_CONCURRENT_JOBS, max_size=MAX_QUEUED_JOBS).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, theme=THEME, css=CSS, head=HEAD,
        max_file_size=UPLOAD_LIMIT_BYTES,
        allowed_paths=[str(BUNDLE / "images"), str(BUNDLE / "thumbs")])  # only the paintings, not the code or weights
