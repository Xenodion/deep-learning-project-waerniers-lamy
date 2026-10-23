"""Shared code for the NLP notebooks: corpus, queries, VLM descriptions, retrieval rungs, judging, metrics.

Protocol in ADR 0007. Two corpora, never mixed: `dev` = Museum A validation (480 paintings), `test` = Museum B
(600 paintings, unseen artists), scored once after `freeze()`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "wikiart_genre5"
CACHE_DIR = DATA_DIR / "nlp_cache"
RESULTS_DIR = REPO_ROOT / "results" / "nlp"
DESCRIPTIONS_FILE = DATA_DIR / "descriptions_qwen3-vl-8b-instruct.json"
FROZEN_CONFIG = RESULTS_DIR / "frozen_config.json"
CORPUS_SPLITS = {"dev": "val", "test": "test"}

VLM_MODEL = "qwen/qwen3-vl-8b-instruct"
VLM_PROMPT = (
    "Analyze this fine-art painting concisely for museum visitors. "
    "Describe the visible subjects, setting, landscape, weather, colors and atmosphere in 2 sentences."
)
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
HYBRID_ALPHA = 0.5  # ASSUMPTION: equal weight, not tuned (ADR 0007)
K = 5
SEED = 0


# --------------------------------------------------------------------------------------------------
# Corpus and queries
# --------------------------------------------------------------------------------------------------
def load_corpus(corpus: str) -> pd.DataFrame:
    """Paintings of one evaluation corpus ('dev' or 'test'), metadata joined with split.csv."""
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    split_df = pd.read_csv(DATA_DIR / "split.csv")[["id", "split"]]
    df = meta.merge(split_df, on="id")
    return df[df["split"] == CORPUS_SPLITS[corpus]].reset_index(drop=True)


def metadata_text(row: pd.Series) -> str:
    """Searchable text from the only metadata the dataset has (genre, style, artist)."""
    artist = "" if row["artist"] == "Unknown Artist" else f" by {row['artist'].replace('-', ' ')}"
    return f"{row['genre'].replace('_', ' ')} painting, {row['style'].replace('_', ' ')} style{artist}"


# Visual queries only, written before any v1 output existed; the same set is used on both corpora, nothing is tuned
# on them. No relevant ids: relevance comes from blind judgments of the pooled results.
QUERIES_BY_CATEGORY = {
    "object": [
        "a vase of flowers on a table", "a bowl of fruit", "an old man with a white beard", "a portrait of a child",
        "a woman wearing a hat", "a sailing boat", "horses", "a church or a cathedral", "a bridge over a river",
        "a skull", "angels with wings", "a dog or a cat",
    ],
    "scene": [
        "a stormy sea with big waves", "a snowy winter landscape", "a river flowing through a forest",
        "mountains under a cloudy sky", "a sunset over the water", "a rainy city street",
        "a busy street with people and carriages", "a harbour with ships", "a night scene lit by lamps or moonlight",
        "a field of crops in summer",
    ],
    "mood_colour": [
        "a painting dominated by blue tones", "a bright and colourful painting", "a dark and gloomy painting",
        "a calm and peaceful scene", "a sad or melancholic face", "warm golden light",
        "strong contrast between light and shadow", "a painting with mostly red colours", "a misty, hazy atmosphere",
    ],
    "compound": [
        "a garden full of flowers in sunlight", "a crucifixion scene with mourners", "a woman reading a book indoors",
        "a man in a dark suit against a plain background", "fishermen on a beach",
        "a still life with a wine bottle and a glass", "a village with houses along a river",
        "a portrait made of geometric shapes", "the Virgin Mary with the infant Jesus",
    ],
}
QUERIES = pd.DataFrame(
    [(c, t) for c, texts in QUERIES_BY_CATEGORY.items() for t in texts], columns=["category", "text"]
)
QUERIES.insert(0, "query_id", [f"q{i:02d}" for i in range(1, len(QUERIES) + 1)])


# --------------------------------------------------------------------------------------------------
# VLM descriptions (OpenRouter, run once)
# --------------------------------------------------------------------------------------------------
def _describe_image(path: Path, api_key: str) -> str:
    body = json.dumps({
        "model": VLM_MODEL, "temperature": 0, "max_tokens": 60,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()}},
            {"type": "text", "text": VLM_PROMPT},
        ]}],
    }).encode()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def generate_descriptions(limit: int | None = None, workers: int = 8) -> dict[str, str]:
    """Describe every dev + test painting not yet in DESCRIPTIONS_FILE (resumable). Needs OPENROUTER_API_KEY."""
    api_key = os.environ["OPENROUTER_API_KEY"]
    paintings = pd.concat([load_corpus("dev"), load_corpus("test")])
    done = load_descriptions() if DESCRIPTIONS_FILE.exists() else {}
    todo = paintings[~paintings["id"].isin(done)].head(limit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_describe_image, DATA_DIR / r.path, api_key): r.id for r in todo.itertuples()}
        for n, future in enumerate(as_completed(futures), start=1):
            try:
                done[futures[future]] = future.result()
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                print(f"failed {futures[future]}: {e} (re-run to retry)")
            if n % 50 == 0 or n == len(futures):
                DESCRIPTIONS_FILE.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"{n}/{len(futures)} done, {len(done)} descriptions saved")
    return done


def load_descriptions() -> dict[str, str]:
    return json.loads(DESCRIPTIONS_FILE.read_text(encoding="utf-8"))


def artist_leak_rate(paintings: pd.DataFrame, descriptions: dict[str, str]) -> float:
    """Share of known-artist paintings whose description names the artist (the VLM recognising famous works)."""
    known = paintings[paintings["artist"] != "Unknown Artist"]
    surnames = known["artist"].str.split("-").str[-1]
    hits = [s in descriptions[i].lower() for i, s in zip(known["id"], surnames) if len(s) > 3]
    return float(np.mean(hits)) if hits else 0.0


# --------------------------------------------------------------------------------------------------
# Encoders and retrieval rungs: each returns a (n_queries, n_paintings) similarity matrix
# --------------------------------------------------------------------------------------------------
@cache
def _text_model() -> SentenceTransformer:
    return SentenceTransformer(TEXT_MODEL, device="cpu")


@cache
def _siglip() -> tuple[AutoProcessor, AutoModel]:
    return AutoProcessor.from_pretrained(SIGLIP_MODEL), AutoModel.from_pretrained(SIGLIP_MODEL).eval()


def _pooled(output: torch.Tensor | object) -> np.ndarray:
    # transformers returns a tensor or, in recent versions, a model output holding it.
    features = output if isinstance(output, torch.Tensor) else output.pooler_output
    return torch.nn.functional.normalize(features, dim=-1).numpy()


def encode_text(texts: list[str]) -> np.ndarray:
    return _text_model().encode(texts, normalize_embeddings=True)


def encode_siglip_text(texts: list[str]) -> np.ndarray:
    processor, model = _siglip()
    # SigLIP was trained with max_length padding; other paddings silently degrade its text embeddings.
    inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        return _pooled(model.get_text_features(**inputs))


def encode_siglip_images(paths: list[Path], batch_size: int = 32) -> np.ndarray:
    processor, model = _siglip()
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        with torch.no_grad():
            chunks.append(_pooled(model.get_image_features(**processor(images=images, return_tensors="pt"))))
    return np.vstack(chunks)


def _cached(name: str, key: list[str], compute) -> np.ndarray:
    """Corpus embeddings are computed once and cached on disk, keyed by a hash of the model and inputs."""
    path = CACHE_DIR / f"{name}_{hashlib.sha256(chr(10).join(key).encode()).hexdigest()[:16]}.npy"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, compute())
    return np.load(path)


def _text_rung(name: str, docs: list[str], queries: list[str]) -> np.ndarray:
    return encode_text(queries) @ _cached(name, [TEXT_MODEL, *docs], lambda: encode_text(docs)).T


def _descriptions_for(paintings: pd.DataFrame, descriptions: dict[str, str]) -> list[str]:
    missing = set(paintings["id"]) - set(descriptions)
    if missing:
        raise ValueError(f"{len(missing)} paintings have no description: run generate_descriptions().")
    return [descriptions[i] for i in paintings["id"]]


def rung_random(paintings: pd.DataFrame, queries: list[str]) -> np.ndarray:
    """R0: seeded uniform random scores, the floor."""
    return np.random.default_rng(SEED).random((len(queries), len(paintings)))


def rung_metadata(paintings: pd.DataFrame, queries: list[str]) -> np.ndarray:
    """R1 baseline: MiniLM over genre + style + artist."""
    return _text_rung("minilm_metadata", [metadata_text(r) for _, r in paintings.iterrows()], queries)


def rung_description(paintings: pd.DataFrame, queries: list[str], descriptions: dict[str, str]) -> np.ndarray:
    """R2: MiniLM over the VLM description only (metadata -> visual text)."""
    return _text_rung("minilm_description", _descriptions_for(paintings, descriptions), queries)


def rung_description_metadata(paintings: pd.DataFrame, queries: list[str], descriptions: dict[str, str]) -> np.ndarray:
    """R3: MiniLM over description + metadata."""
    docs = [f"{d} {metadata_text(r)}." for d, (_, r) in zip(_descriptions_for(paintings, descriptions), paintings.iterrows())]
    return _text_rung("minilm_description_metadata", docs, queries)


def rung_siglip(paintings: pd.DataFrame, queries: list[str]) -> np.ndarray:
    """R4: SigLIP text tower (query) vs image tower (painting); no text about the painting at all."""
    images = _cached("siglip_image", [SIGLIP_MODEL, *paintings["path"]],
                     lambda: encode_siglip_images([DATA_DIR / p for p in paintings["path"]]))
    return encode_siglip_text(queries) @ images.T


def _zscore(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean(axis=1, keepdims=True)) / scores.std(axis=1, keepdims=True)


def rung_hybrid(paintings: pd.DataFrame, queries: list[str], descriptions: dict[str, str]) -> np.ndarray:
    """R5: R3 + R4 fused on per-query z-scores (raw cosines of the two encoders live on different scales)."""
    text = _zscore(rung_description_metadata(paintings, queries, descriptions))
    return HYBRID_ALPHA * _zscore(rung_siglip(paintings, queries)) + (1 - HYBRID_ALPHA) * text


def latency_ms(rung, *args) -> float:
    """Median time to answer one query (query encoding + similarity), corpus embeddings already cached."""
    rung(*args[:1], QUERIES["text"].tolist()[:1], *args[1:])  # warm-up
    times = []
    for q in QUERIES["text"]:
        start = time.perf_counter()
        rung(*args[:1], [q], *args[1:])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


# --------------------------------------------------------------------------------------------------
# Judging: blind, graded (0/1/2), pooled on the top-K, incremental
# --------------------------------------------------------------------------------------------------
RUBRIC = [
    "2 - clear match: the painting shows what the query asks for.",
    "1 - partial: the main subject is there but a qualifier is missing or weak (e.g. a sea, but calm).",
    "0 - no match.",
]


def qrels_path(corpus: str) -> Path:
    return RESULTS_DIR / f"qrels_{corpus}.json"


def load_qrels(corpus: str) -> dict[str, dict[str, int]]:
    path = qrels_path(corpus)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def top_k_ids(scores: np.ndarray, paintings: pd.DataFrame, k: int = K) -> list[list[str]]:
    ids = paintings["id"].to_numpy()
    return [ids[np.argsort(-row)[:k]].tolist() for row in scores]


def write_judging_page(corpus: str, scores: dict[str, np.ndarray], paintings: pd.DataFrame) -> int:
    """Page with every top-K painting of the given rungs not judged yet. Shows only query + image. Returns #items."""
    qrels = load_qrels(corpus)
    paths = dict(zip(paintings["id"], paintings["path"]))
    rng = random.Random(SEED)
    items = []
    for qi, (qid, text) in enumerate(zip(QUERIES["query_id"], QUERIES["text"])):
        new = sorted({pid for s in scores.values() for pid in top_k_ids(s, paintings)[qi]} - set(qrels.get(qid, {})))
        rng.shuffle(new)
        items += [{"qid": qid, "query": text, "pid": pid, "src": paths[pid]} for pid in new]
    page = (_PAGE.replace("__ITEMS__", json.dumps(items)).replace("__KEY__", f"judging_{corpus}_{len(items)}")
            .replace("__RUBRIC__", "".join(f"<li>{r}</li>" for r in RUBRIC))
            .replace("__EXPORT__", f"judgments_{corpus}.json"))
    (DATA_DIR / f"judging_{corpus}.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} new items to judge: open {DATA_DIR / f'judging_{corpus}.html'} in a browser, then export.")
    return len(items)


def import_judgments(corpus: str, export_file: Path) -> dict[str, dict[str, int]]:
    """Merge a judging-page export into results/nlp/qrels_<corpus>.json."""
    qrels = load_qrels(corpus)
    for qid, grades in json.loads(Path(export_file).read_text(encoding="utf-8")).items():
        if any(g not in (0, 1, 2) for g in grades.values()):
            raise ValueError(f"{qid}: grades must be 0, 1 or 2.")
        qrels.setdefault(qid, {}).update(grades)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qrels_path(corpus).write_text(json.dumps(qrels, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{sum(map(len, qrels.values()))} judgments in {qrels_path(corpus)}")
    return qrels


# --------------------------------------------------------------------------------------------------
# Metrics and statistics (per query, paired)
# --------------------------------------------------------------------------------------------------
def dcg_at_k(grades: list[int], k: int = K) -> float:
    g = np.asarray(grades[:k], dtype=float)
    return float(np.sum((2**g - 1) / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(grades: list[int], judged: list[int], k: int = K) -> float:
    """DCG normalised by the best ordering of every grade judged for the query."""
    ideal = dcg_at_k(sorted(judged, reverse=True), k)
    return dcg_at_k(grades, k) / ideal if ideal > 0 else 0.0


def bootstrap_ci(values, n: int = 10_000, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over queries."""
    means = np.random.default_rng(SEED).choice(np.asarray(values), size=(n, len(values))).mean(axis=1)
    return float(np.quantile(means, (1 - level) / 2)), float(np.quantile(means, (1 + level) / 2))


def paired_test(a, b, n: int = 10_000) -> float:
    """Two-sided paired randomization test (random sign flips of the per-query differences)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    null = np.abs((np.random.default_rng(SEED).choice([-1.0, 1.0], size=(n, d.size)) * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.minimum(np.maximum.accumulate((p.size - np.arange(p.size)) * p[order]), 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()


def per_query_scores(corpus: str, scores: dict[str, np.ndarray], paintings: pd.DataFrame) -> pd.DataFrame:
    """nDCG@K and Success@K per rung and query. Unjudged = 0. Queries with nothing relevant judged are dropped."""
    qrels = load_qrels(corpus)
    rows = []
    for name, s in scores.items():
        for qi, (qid, category) in enumerate(zip(QUERIES["query_id"], QUERIES["category"])):
            judged = qrels.get(qid, {})
            if max(judged.values(), default=0) < 1:
                continue
            grades = [judged.get(pid, 0) for pid in top_k_ids(s, paintings)[qi]]
            rows.append({"rung": name, "query_id": qid, "category": category,
                         "ndcg@5": ndcg_at_k(grades, list(judged.values())), "success@5": float(max(grades) == 2),
                         "judged@5": float(np.mean([pid in judged for pid in top_k_ids(s, paintings)[qi]]))})
    return pd.DataFrame(rows)


def summary_table(per_query: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% bootstrap CI per rung, in ladder order."""
    rows = []
    for name, r in per_query.groupby("rung", sort=False):
        nd, su = bootstrap_ci(r["ndcg@5"]), bootstrap_ci(r["success@5"])
        rows.append({"rung": name, "queries": len(r), "nDCG@5": r["ndcg@5"].mean(), "nDCG low": nd[0], "nDCG high": nd[1],
                     "Success@5": r["success@5"].mean(), "Success low": su[0], "Success high": su[1],
                     "judged@5": r["judged@5"].mean()})
    return pd.DataFrame(rows).round(3)


def plot_ladder(table: pd.DataFrame, title: str) -> None:
    """Horizontal bars of mean nDCG@5 per rung with 95% bootstrap CIs."""
    _, ax = plt.subplots(figsize=(8, 3.5))
    err = [table["nDCG@5"] - table["nDCG low"], table["nDCG high"] - table["nDCG@5"]]
    ax.barh(table["rung"], table["nDCG@5"], xerr=err, color="#2f5d8a", capsize=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("nDCG@5 (mean over queries, 95% bootstrap CI)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def compare(per_query: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired nDCG@5 differences with randomization p-values, Holm-adjusted over the given (pre-registered) pairs."""
    wide = per_query.pivot(index="query_id", columns="rung", values="ndcg@5")
    out = pd.DataFrame([{"system": a, "reference": b, "delta nDCG@5": wide[a].mean() - wide[b].mean(),
                         "p": paired_test(wide[a], wide[b])} for a, b in pairs])
    out["p (Holm)"] = holm(out["p"].tolist())
    return out.round(4)


# --------------------------------------------------------------------------------------------------
# Freeze: Museum B is scored only with the exact config chosen on dev
# --------------------------------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def current_config() -> dict[str, str | float]:
    return {"vlm_model": VLM_MODEL, "vlm_prompt": _sha(VLM_PROMPT), "text_model": TEXT_MODEL,
            "siglip_model": SIGLIP_MODEL, "hybrid_alpha": HYBRID_ALPHA,
            "descriptions": _sha(DESCRIPTIONS_FILE.read_text(encoding="utf-8")), "queries": _sha(QUERIES.to_json())}


def freeze(best_single: str) -> None:
    """Write the config used on dev; `best_single` (R3 or R4, chosen on dev) is the reference for the hybrid on test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_CONFIG.write_text(json.dumps({**current_config(), "best_single": best_single}, indent=1), encoding="utf-8")


def check_frozen() -> dict[str, str | float]:
    """Refuse to touch Museum B unless the frozen config still matches the code, models, queries and descriptions."""
    if not FROZEN_CONFIG.exists():
        raise RuntimeError("Museum B is locked: finish the dev notebooks and call freeze() first.")
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    drift = {k: v for k, v in current_config().items() if frozen[k] != v}
    if drift:
        raise RuntimeError(f"Config changed since freeze: {drift}. Museum B results would not be valid.")
    return frozen


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relevance Judging</title>
<style>
  :root { --bg: #fafaf7; --fg: #1d1d1b; --muted: #6b6b66; --accent: #2f5d8a; --card: #fff; --line: #deded8; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ececea; --muted: #9a9a94; --accent: #8db4dc;
    --card: #20201e; --line: #3a3a37; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 16px; }
  .query { font-size: 1.5rem; font-weight: 600; margin: 8px 0 16px; }
  img { display: block; max-width: 100%; max-height: 60vh; margin: 0 auto; border: 1px solid var(--line); }
  .row { display: flex; gap: 8px; margin: 16px 0; justify-content: space-between; color: var(--muted); }
  button { flex: 1; padding: 12px; font-size: 1rem; border: 1px solid var(--line); background: var(--card);
    color: var(--fg); border-radius: 6px; cursor: pointer; }
  button.on { outline: 2px solid var(--accent); }
  ul { color: var(--muted); font-size: .9rem; padding-left: 20px; }
</style></head><body><main>
<div class="row"><span id="progress"></span><span>keys 0 1 2, backspace = previous</span></div>
<div class="query" id="query"></div>
<img id="img" alt="painting to judge">
<div class="row"><button data-g="0">0 no</button><button data-g="1">1 partial</button><button data-g="2">2 clear</button></div>
<div class="row"><button id="prev">previous</button><button id="export">export JSON</button></div>
<ul>__RUBRIC__</ul>
</main><script>
const ITEMS = __ITEMS__, KEY = "__KEY__";
let grades = {};
try { grades = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { grades = {}; }
const id = it => it.qid + "|" + it.pid;
let pos = Math.max(0, ITEMS.findIndex(it => !(id(it) in grades)));
function show() {
  if (!ITEMS.length) { document.getElementById("query").textContent = "Nothing new to judge."; return; }
  const it = ITEMS[pos];
  document.getElementById("query").textContent = it.query;
  document.getElementById("img").src = it.src;
  document.getElementById("progress").textContent = (pos + 1) + " / " + ITEMS.length + ", judged " + Object.keys(grades).length;
  document.querySelectorAll("[data-g]").forEach(b => b.classList.toggle("on", grades[id(it)] == b.dataset.g));
}
function grade(g) {
  grades[id(ITEMS[pos])] = g;
  try { localStorage.setItem(KEY, JSON.stringify(grades)); } catch (e) { /* export still works */ }
  if (pos < ITEMS.length - 1) pos++;
  show();
}
document.querySelectorAll("[data-g]").forEach(b => b.onclick = () => grade(+b.dataset.g));
document.getElementById("prev").onclick = () => { if (pos > 0) pos--; show(); };
document.addEventListener("keydown", e => {
  if (["0", "1", "2"].includes(e.key)) grade(+e.key);
  if (e.key === "Backspace" && pos > 0) { pos--; show(); }
});
document.getElementById("export").onclick = () => {
  const out = {};
  for (const [k, g] of Object.entries(grades)) { const [q, p] = k.split("|"); (out[q] ||= {})[p] = g; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "__EXPORT__"; a.click();
};
show();
</script></body></html>
"""
