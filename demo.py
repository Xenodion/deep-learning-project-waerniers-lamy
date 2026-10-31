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
import logging
import time
from pathlib import Path
from typing import NamedTuple

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

log = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
BUNDLE = APP_DIR if (APP_DIR / "collection.csv").exists() else APP_DIR / "app"
MAX_CONCURRENT_JOBS, MAX_QUEUED_JOBS = 2, 16
MAX_QUERY_CHARS = 200  # SigLIP reads 64 tokens anyway
UPLOAD_LIMIT_BYTES = 5 * 1024 * 1024  # a phone photo, up to 5 MB; gradio refuses bigger files before decoding them
UPLOAD_SIDE = 512  # an uploaded picture is shrunk to this before SigLIP (which reads 224 px anyway)
Image.MAX_IMAGE_PIXELS = 25_000_000  # Pillow refuses above twice this (50 Mpx, ~150 MB in RGB): no decompression bomb

GENRES = ["cityscape", "landscape", "portrait", "religious_painting", "still_life"]
GENRE_NAME = {"cityscape": "Cityscape", "landscape": "Landscape", "portrait": "Portrait",
              "religious_painting": "Religious painting", "still_life": "Still life"}
SIGLIP_MODEL = "google/siglip-base-patch16-224"
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MAX_RESULTS = 20  # every search ranks this many once; the slider only shows the first k of them
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


MUSEUM_TAG = '<em class="bar-tag">museum</em>'


def bars_html(items: list[tuple[str, float, str]], top: int | None = 0, wide: bool = False, tagged: int | None = None) -> str:
    """Horizontal bars: items are (label, share between 0 and 1, text at the end); `top` is the highlighted row,
    `tagged` the row that gets the "museum" tag (the label the museum gave)."""
    rows = "".join(
        f'<div class="bar{" top" if k == top else ""}"><span class="bar-name">{html.escape(name)}'
        f'{MUSEUM_TAG if k == tagged else ""}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{share * 100:.1f}%"></span></span>'
        f'<span class="bar-pct">{text}</span></div>' for k, (name, share, text) in enumerate(items))
    return f'<div class="bars{" wide" if wide else ""}">{rows}</div>'


def probs_bars(probs: np.ndarray, truth: str | None = None) -> str:
    order = np.argsort(-probs)
    tagged = next((k for k, i in enumerate(order) if GENRES[i] == truth), None)
    return bars_html([(GENRE_NAME[GENRES[i]], float(probs[i]), f"{probs[i] * 100:.0f}%") for i in order], tagged=tagged)


def about_html(note: dict[str, str]) -> str:
    return (f'<div class="about"><p>{html.escape(note["text"])}</p>'
            f'<p class="credit"><a href="{note["url"]}" target="_blank" rel="noopener">Read more on Wikipedia</a>'
            f' (text under CC BY-SA)</p></div>')


def _role_html(kind: str, title: str, genre: str, detail: str) -> str:
    return (f'<div class="role {kind}"><span class="role-title">{title}</span><strong>{GENRE_NAME[genre]}</strong>'
            f'<small>{detail}</small></div>')


def verdict_html(probs: np.ndarray, row: pd.Series | None, ms: float, notes: dict) -> str:
    """Two named blocks (what the museum says, what the model says), a banner right or wrong, then the model's shares."""
    guess = GENRES[int(np.argmax(probs))]
    pred = _role_html("pred", "Model prediction", guess, f"{probs.max() * 100:.0f}% sure")
    if row is None:
        return f'<div class="card"><div class="roles">{pred}</div>{probs_bars(probs)}{about_html(notes[guess])}{meter_html(ms)}</div>'
    truth = row["genre"]
    detail = f'{html.escape(str(row["style"]).replace("_", " "))} by {html.escape(_artist(row))}'
    if guess == truth:
        banner = '<p class="result ok"><span>✓</span> Correct: the model agrees with the museum.</p>'
    else:
        banner = (f'<p class="result miss"><span>✗</span> Wrong: the model said {GENRE_NAME[guess].lower()}, '
                  f'the museum says {GENRE_NAME[truth].lower()}.</p>')
    return (f'<div class="card"><div class="roles">{_role_html("truth", "Museum label", truth, detail)}{pred}</div>{banner}'
            f'<p class="hint">What the model thinks, genre by genre:</p>{probs_bars(probs, truth)}'
            f'{about_html(notes[guess])}{meter_html(ms)}</div>')


def _url(relative: str) -> str:
    return f"/gradio_api/file={(BUNDLE / relative).resolve().as_posix()}"


def wall_html(collection: pd.DataFrame, rows: list[int]) -> str:
    tiles = "".join(
        f'<button class="wall-item" data-row="{i}" aria-label="{GENRE_NAME[collection.loc[i, "genre"]]}">'
        f'<img loading="lazy" decoding="async" src="{_url(collection.loc[i, "thumb"])}" alt=""></button>' for i in rows)
    return f'<div class="wall">{tiles}</div>'


class Found(NamedTuple):
    """The last search of a visitor: kept so the slider can re-cut it without running the model again."""
    rows: list[int]
    subject: str  # what the results are close to, e.g. '"a stormy sea"' or 'your picture'
    ms: float | None


def found_html(collection: pd.DataFrame, found: Found, k: int) -> str:
    rows = found.rows[:k]
    if not rows:
        return '<p class="hint centered">Type a description above, pick one of the ideas, or show a picture.</p>'
    cards = "".join(
        f'<figure class="found-item" data-row="{i}" tabindex="0"><img src="{_url(collection.loc[i, "path"])}" '
        f'alt="{GENRE_NAME[collection.loc[i, "genre"]]} by {html.escape(_artist(collection.loc[i]))}">'
        f'<figcaption><strong>{n}.</strong> {GENRE_NAME[collection.loc[i, "genre"]]}<br>'
        f'<span>{html.escape(_artist(collection.loc[i]))}</span></figcaption></figure>'
        for n, i in enumerate(rows, 1))
    head = f'<p class="found-head">{len(rows)} closest to {html.escape(found.subject)}</p>'
    return f'{head}<div class="found">{cards}</div>{meter_html(found.ms) if found.ms is not None else ""}'


def example_tiles_html(collection: pd.DataFrame, n: int = 30) -> str:
    """Random sample of n paintings from Museum B for the interactive sidebar."""
    sampled = collection.sample(min(n, len(collection)), random_state=None)
    items = []
    for idx, row in sampled.iterrows():
        artist = _artist(row)
        genre = GENRE_NAME[row["genre"]]
        path = _url(row["path"])
        items.append(
            f'<button class="sidebar-item" data-row="{idx}" title="{genre} - {html.escape(artist)}">'
            f'<img src="{path}" alt="{html.escape(artist)}" loading="lazy">'
            f'<div class="sidebar-item-info">'
            f'<span class="sidebar-item-genre">{genre}</span>'
            f'<small class="sidebar-item-artist">{html.escape(artist)}</small>'
            f'</div>'
            f'</button>'
        )
    return f'<div class="sidebar-scroll">{"".join(items)}</div>'


PICK_JS = """<script>
document.addEventListener('click', (e) => {
  const tile = e.target.closest('.wall-item');
  const go = document.querySelector('#pick-go');
  if (tile && go) {
    document.querySelectorAll('.wall-item.selected').forEach((t) => t.classList.remove('selected'));
    tile.classList.add('selected');
    window.pickedRow = tile.dataset.row;  // read back by the js= hook of the hidden button
    go.click();
    document.querySelector('#stage')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return;
  }
  if (e.target.closest('#lightbox')) { document.querySelector('#lightbox').remove(); return; }
  const result = e.target.closest('.found-item');
  if (result) {  // enlarge a search result
    const box = document.createElement('div');
    box.id = 'lightbox';
    box.innerHTML = '<figure><img alt=""><figcaption></figcaption></figure>';
    box.querySelector('img').src = result.querySelector('img').src;
    box.querySelector('figcaption').innerHTML = result.querySelector('figcaption').innerHTML;
    document.body.appendChild(box);
    return;
  }
  const toggleBtn = e.target.closest('#image-toggle-btn');
if (toggleBtn) {
  setPopover(!document.querySelector('#image-popover')?.classList.contains('open'));
  return;
}
const closeBtn = e.target.closest('#close-popover-btn');
if (closeBtn) {
  document.querySelector('#image-popover')?.classList.remove('open');
  document.querySelector('#image-toggle-btn')?.classList.remove('active');
  return;
}
const ex = e.target.closest('.sidebar-item');
const exGo = document.querySelector('#sidebar-go');
if (ex && exGo) {
  document.querySelectorAll('.sidebar-item.selected').forEach((t) => t.classList.remove('selected'));
  ex.classList.add('selected');
  window.pickedExample = ex.dataset.row;
  exGo.click();
  document.querySelector('#image-popover')?.classList.remove('open');
  document.querySelector('#image-toggle-btn')?.classList.remove('active');
  return;
}
const ref = e.target.closest('#refresh-examples-btn');
const refGo = document.querySelector('#refresh-go');
if (ref && refGo) {
  refGo.click();
  return;
}
if (!e.target.closest('#image-popover') && !e.target.closest('#image-toggle-btn')) {
  const popover = document.querySelector('#image-popover');
  if (popover && popover.classList.contains('open')) {
    popover.classList.remove('open');
    document.querySelector('#image-toggle-btn')?.classList.remove('active');
  }
}
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { document.querySelector('#lightbox')?.remove(); setPopover(false); }
  if (e.key === 'Enter' && e.target.matches?.('.found-item')) e.target.click();  // results are focusable
});
// drag a search result onto the picture search: opening the box while dragging gives a target to drop on
const setPopover = (open) => {
  const pop = document.querySelector('#image-popover');
  pop?.classList.toggle('open', open);
  document.querySelector('#image-toggle-btn')?.classList.toggle('active', open);
  // gradio clips what goes below the screen and the page cannot scroll there: the box fits the room left and scrolls inside
  if (open && pop) pop.style.maxHeight = Math.max(240, innerHeight - pop.getBoundingClientRect().top - 12) + 'px';
};
const onDropTarget = (e) => e.target.closest?.('#image-popover, #image-toggle-btn');
document.addEventListener('dragstart', (e) => {
  const result = e.target.closest?.('.found-item');
  if (!result) return;
  e.dataTransfer.setData('text/x-row', result.dataset.row);
  e.dataTransfer.effectAllowed = 'copy';
  setPopover(true);
});
document.addEventListener('dragend', () => setPopover(false));
document.addEventListener('dragover', (e) => {
  if (e.dataTransfer.types.includes('text/x-row') && onDropTarget(e)) e.preventDefault();
}, true);
document.addEventListener('drop', (e) => {
  const row = e.dataTransfer.getData('text/x-row');
  if (!row || !onDropTarget(e)) return;
  e.preventDefault();
  e.stopPropagation();  // gradio's own upload zone would try to read the dropped page image as a file
  window.pickedExample = row;  // same path as clicking an example of the box
  document.querySelector('#sidebar-go')?.click();
  setPopover(false);
}, true);
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
    picture = scoring["picture_search"]
    names ={r["rung"]: LANGUAGE_NAMES[r["rung"]] + (" (in this app)" if r["deployed"] else "") for r in rungs}
    ladder = bars_html([(names[r["rung"]], r["test"], f'{r["test"] * 100:.1f}%') for r in rungs],
                       top=next(k for k, r in enumerate(rungs) if r["deployed"]), wide=True)
    return (f'<div class="card"><h3>Language search</h3><p class="big">{deployed["test"] * 100:.1f}%</p>'
            f'<p class="said">Match with the question across the top 5 results (nDCG@5).</p>'
            f'<p class="hint">Each way of searching, from the simplest, on the same 40 questions (nDCG@5 as a percentage):</p>{ladder}'
            f'<p class="spaced">Adding generated descriptions (hybrid) scores {hybrid["test"] * 100:.1f}% against {deployed["test"] * 100:.1f}%. '
            f'That gap cannot be told from chance (p = {scoring["hybrid_p_value"]}), so we kept the simpler search: one model, '
            f'no paid service.</p>'
            f'<p>Searching with a picture uses the same model: {picture["same_genre_at_5"] * 100:.1f}% of the 5 nearest paintings '
            f'share its genre (chance: {picture["chance"] * 100:.1f}%). No visitor question is behind this number, only the '
            f'labels of the museum.</p>'
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
.roles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 12px; }
.role { display: flex; flex-direction: column; gap: 3px; background: #10201f; border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; min-width: 0; }
.role.truth { border-left: 4px solid var(--ok); } .role.pred { border-left: 4px solid var(--brass); }
.role-title { color: var(--muted); font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.role strong { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 600; color: var(--ink); line-height: 1.2; }
.role small { color: var(--muted); font-size: .85rem; }
.result { display: flex; gap: 10px; align-items: center; margin: 0 0 14px !important; padding: 10px 14px; border-radius: 10px; font-weight: 600; border: 1px solid; }
.result span { font-size: 1.2rem; }
.result.ok { color: var(--ok); background: rgba(134, 201, 164, .12); border-color: var(--ok); }
.result.miss { color: var(--miss); background: rgba(227, 148, 132, .12); border-color: var(--miss); }
.bar-tag { margin-left: 6px; padding: 1px 7px; border: 1px solid var(--ok); border-radius: 999px; color: var(--ok); font-size: .68rem; font-style: normal; font-weight: 500; }
.said { font-size: 1.1rem; }
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
#preview { --sq: min(620px, calc(100vw - 32px), max(420px, 72vh)); height: var(--sq) !important; width: var(--sq) !important; }  /* square display area */
#stage > #preview { flex: 0 0 auto !important; min-width: 0 !important; }
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
.found-item { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; cursor: zoom-in; }
.found-item:hover, .found-item:focus-visible { border-color: var(--brass); }
.found-head { color: var(--muted); margin: 14px 0 10px !important; font-size: .95rem; }
#found { min-height: 140px; }
#lightbox { position: fixed; inset: 0; z-index: 2000; background: rgba(8, 16, 18, .92); display: flex; align-items: center; justify-content: center; cursor: zoom-out; padding: 24px; }
#lightbox figure { margin: 0; text-align: center; color: var(--ink); }
#lightbox img { max-width: 96vw; max-height: 86vh; min-height: 60vh; object-fit: contain; border-radius: 8px; }
#lightbox figcaption { margin-top: 12px; line-height: 1.4; } #lightbox figcaption span { color: var(--muted); }
.found-item img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }
.found-item figcaption { padding: 10px 12px; font-size: .95rem; line-height: 1.4; } .found-item figcaption span { color: var(--muted); }
.centered { text-align: center; }
#found .meter { text-align: center; }
#query textarea, #query input { background: var(--panel) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; font-size: 1.05rem; }
#search-row { align-items: end; gap: 10px; }
#upload { max-width: 420px; margin: 0 auto 18px; }
#upload label, #upload .label-wrap span { color: var(--muted) !important; }
#upload .image-container, #upload .upload-container { background: var(--panel) !important; border: 1px dashed var(--line); border-radius: 14px; }
#search-wrapper { position: relative; width: 100%; margin-bottom: 8px; }
#image-toggle-btn { background: var(--panel) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; font-weight: 500 !important; white-space: nowrap !important; transition: all .15s ease !important; cursor: pointer !important; }
#image-toggle-btn:hover, #image-toggle-btn.active { border-color: var(--brass) !important; color: var(--brass) !important; background: rgba(214, 171, 82, 0.12) !important; }
#image-popover { position: absolute !important; top: calc(100% + 8px) !important; right: 0 !important; width: 420px !important; max-width: calc(100vw - 32px) !important; max-height: calc(100vh - 140px); overflow-y: auto; scrollbar-gutter: stable; background: #172b2f !important; border: 1px solid var(--brass) !important; border-radius: 16px !important; padding: 14px !important; box-shadow: 0 16px 40px rgba(0, 0, 0, 0.7) !important; z-index: 1000 !important; opacity: 0; visibility: hidden; pointer-events: none; transform: translateY(-8px); transition: opacity 0.2s ease, transform 0.2s ease, visibility 0.2s; }
#image-popover.open { opacity: 1 !important; visibility: visible !important; pointer-events: auto !important; transform: translateY(0) !important; }
#image-popover #upload { width: 100% !important; max-width: 100% !important; margin: 0 0 10px 0 !important; height: min(380px, calc(100vw - 74px)) !important; flex-shrink: 0; }  /* square: 420px popover minus padding and scrollbar */
#image-popover #upload > div, #image-popover #upload .image-container, #image-popover #upload .upload-container, #image-popover #upload .image-frame { height: 100% !important; }
#image-popover #upload img { width: 100% !important; height: 100% !important; object-fit: cover !important; }
.header-actions { display: flex; gap: 6px; align-items: center; }
.sidebar-header { display: flex; justify-content: space-between; align-items: center; margin: 4px 0 10px; font-size: 0.85rem; font-weight: 600; color: var(--muted); }
.refresh-btn { background: transparent; border: 1px solid var(--line); color: var(--muted); border-radius: 6px; padding: 2px 7px; cursor: pointer; font-size: 0.8rem; transition: all .15s; }
.refresh-btn:hover { color: var(--ink); border-color: var(--brass); background: rgba(214, 171, 82, 0.1); }
.sidebar-scroll { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding-right: 4px; }
.sidebar-item { position: relative; aspect-ratio: 1; padding: 0; overflow: hidden; background: #10201f; border: 2px solid transparent; border-radius: 8px; cursor: pointer; transition: border-color .15s; }
.sidebar-item:hover { border-color: var(--brass); }
.sidebar-item.selected { border-color: var(--brass); }
.sidebar-item img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.sidebar-item-info { display: none; }
#k-slider { max-width: 300px; margin: 14px auto 0; padding: 0 !important; background: transparent !important; border: none !important; }
#k-slider input[type=range] { accent-color: var(--brass) !important; }
#k-slider label span, #k-slider span { color: var(--muted) !important; font-size: 0.9rem !important; }
.primary-btn, button.primary { background: var(--brass) !important; color: #1a1a14 !important; border: none !important; font-weight: 600; border-radius: 12px !important; }
#chips { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 10px 0 18px; }
.chip { background: var(--panel) !important; color: var(--muted) !important; border: 1px solid var(--line) !important; border-radius: 999px !important; min-width: 0 !important; width: auto !important; flex: 0 0 auto !important; font-size: .9rem !important; }
.chip:hover { color: var(--ink) !important; border-color: var(--brass) !important; }
button:focus-visible, [role=tab]:focus-visible, label:focus-within { outline: 2px solid var(--brass) !important; outline-offset: 2px; }
@media (min-width: 700px) { .found { grid-template-columns: repeat(3, 1fr); } }
@media (min-width: 1100px) { .found { grid-template-columns: repeat(5, 1fr); } }
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

    no_results = Found([], "", None)

    def text_found(query: str) -> Found:
        query = query[:MAX_QUERY_CHARS]
        if not query.strip():
            return no_results
        rows, ms = timed(search_order, query, embeddings, MAX_RESULTS)
        return Found([int(i) for i in rows], f'"{query.strip()}"', ms)

    def picture_found(img: Image.Image, subject: str) -> Found:
        rows, ms = timed(image_search_order, img, embeddings, MAX_RESULTS)
        return Found([int(i) for i in rows], subject, ms)

    def cleanup_old_upload(old_filepath: str | None, current_filepath: str | None = None) -> None:
        """Single-slot policy: remove previous uploaded file to avoid disk/RAM accumulation (adr/0023)."""
        if old_filepath and old_filepath != current_filepath:
            try:
                p = Path(old_filepath)
                if p.is_file():
                    p.unlink(missing_ok=True)
            except OSError:
                log.warning("Could not delete the previous upload %s", old_filepath, exc_info=True)

    # Every handler below returns the page and the Found it shows; the slider only re-cuts that Found.
    def search_by_image(filepath: str | None, current_filepath: str | None, k: int):
        cleanup_old_upload(current_filepath, filepath)
        if not filepath:
            return found_html(collection, no_results, k), "", None, no_results
        try:
            with Image.open(filepath) as img:
                found = picture_found(img, "your picture")
        except (OSError, ValueError):  # unreadable file, or a picture Pillow refuses (too large)
            log.warning("Could not read the uploaded picture", exc_info=True)
            gr.Warning("This picture could not be read. Try a JPEG or PNG under 5 MB.")
            cleanup_old_upload(filepath)
            return found_html(collection, no_results, k), "", None, no_results
        return found_html(collection, found, k), "", filepath, found

    def on_text_search(q: str, current_filepath: str | None, k: int):
        cleanup_old_upload(current_filepath)
        found = text_found(q)
        return found_html(collection, found, k), None, None, found

    def on_chip_search(text: str, current_filepath: str | None, k: int):
        cleanup_old_upload(current_filepath)
        found = text_found(text)
        return text, found_html(collection, found, k), None, None, found

    def on_clear_upload(current_filepath: str | None):
        cleanup_old_upload(current_filepath)
        return found_html(collection, no_results, 1), None, no_results

    def on_slider_change(k: int, found: Found) -> str:
        return found_html(collection, found, int(k))

    def select_sidebar_example(picked_row: str, current_filepath: str | None, k: int):
        cleanup_old_upload(current_filepath)
        row = collection.loc[int(picked_row)]
        image_path = str(BUNDLE / row["path"])
        with Image.open(image_path) as img:
            found = picture_found(img, "the painting you picked")
        return image_path, found_html(collection, found, k), "", None, found

    # uploads are written to gradio's cache, which is memory on Cloud Run: files older than 1 min go every 1 min
    with gr.Blocks(title="Museum Guide", delete_cache=(60, 60)) as demo, gr.Column(elem_id="app"):
        gr.HTML('<div id="hero"><h1>Museum Guide</h1>'
                '<p>Name the genre of a painting, or find paintings by describing or showing them.</p></div>')
        with gr.Tabs():
            with gr.Tab("1 · Recognise"):
                gr.HTML(tab_intro("Brief, point 1. The visitor points the camera at a painting; the guide recognises its genre "
                                  "and tells the story of that tradition."))
                with gr.Row(elem_id="stage"):
                    # display only: the one upload of the app is the picture search of tab 2
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
                gr.HTML(tab_intro("Brief, point 2. The visitor asks in plain words, or shows a picture; the guide proposes "
                                  "the artworks of the collection that best match."))
                with gr.Column(elem_id="search-wrapper"):
                    with gr.Row(elem_id="search-row"):
                        query = gr.Textbox(placeholder="a stormy sea with big waves", show_label=False,
                                           container=False, scale=5, elem_id="query", max_length=MAX_QUERY_CHARS)
                        go = gr.Button("Search", variant="primary", scale=1)
                        gr.Button("📷 By picture ▾", elem_id="image-toggle-btn", scale=1)

                    with gr.Column(elem_id="image-popover"):
                        uploaded_file = gr.State(None)
                        last_found = gr.State(no_results)
                        upload = gr.Image(type="filepath", sources=["upload"],
                                          label="Drop a picture (up to 5 MB)",
                                          height=180, elem_id="upload", buttons=[])
                        gr.HTML('<div class="sidebar-header"><span>30 museum examples (click to search)</span>'
                                '<div class="header-actions">'
                                '<button id="refresh-examples-btn" class="refresh-btn" title="Sample 30 new paintings">↻ New sample</button>'
                                '<button id="close-popover-btn" class="refresh-btn" title="Close">✕</button>'
                                '</div></div>')
                        example_list = gr.HTML(example_tiles_html(collection), elem_id="example-list")
                        sidebar_picked = gr.Textbox(visible=False)
                        sidebar_go = gr.Button("side-pick", elem_id="sidebar-go", elem_classes="sr-only")
                        refresh_go = gr.Button("refresh-ex", elem_id="refresh-go", elem_classes="sr-only")

                with gr.Row(elem_id="chips"):
                    chips = [gr.Button(q, elem_classes="chip", size="sm") for q in EXAMPLE_QUERIES]
                found = gr.HTML(found_html(collection, no_results, 5), elem_id="found")
                k_slider = gr.Slider(minimum=1, maximum=MAX_RESULTS, step=1, value=5,
                                     label="Number of artworks to display", elem_id="k-slider")

                # Single-slot buffer: a text search clears and deletes the picture; an upload clears the text
                text_io = ([query, uploaded_file, k_slider], [found, upload, uploaded_file, last_found])
                go.click(on_text_search, *text_io)
                query.submit(on_text_search, *text_io)
                for chip, text in zip(chips, EXAMPLE_QUERIES):
                    chip.click(lambda f, k, t=text: on_chip_search(t, f, k), [uploaded_file, k_slider],
                               [query, found, upload, uploaded_file, last_found])
                upload.upload(search_by_image, [upload, uploaded_file, k_slider], [found, query, uploaded_file, last_found]).then(
                    None, js="() => { document.querySelector('#image-popover')?.classList.remove('open');"
                             " document.querySelector('#image-toggle-btn')?.classList.remove('active'); }")
                upload.clear(on_clear_upload, [uploaded_file], [found, uploaded_file, last_found])
                # cheap (no model call): re-cuts the last search, so it also works after an example, a drag or the cache cleanup
                k_slider.change(on_slider_change, [k_slider, last_found], found, show_progress="hidden", trigger_mode="always_last")
                sidebar_go.click(select_sidebar_example, [sidebar_picked, uploaded_file, k_slider],
                                 [upload, found, query, uploaded_file, last_found],
                                 # js= must return one value per input: the row, the state (kept by the server), the slider
                                 js="(row, state, k) => [window.pickedExample, null, k]")
                refresh_go.click(lambda: example_tiles_html(collection), None, example_list)

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
    import os
    server_host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    build_app().queue(default_concurrency_limit=MAX_CONCURRENT_JOBS, max_size=MAX_QUEUED_JOBS).launch(
        server_name=server_host, server_port=args.port, share=args.share, theme=THEME, css=CSS, head=HEAD,
        max_file_size=UPLOAD_LIMIT_BYTES,
        allowed_paths=[str(BUNDLE / "images"), str(BUNDLE / "thumbs")])  # only the paintings, not the code or weights
