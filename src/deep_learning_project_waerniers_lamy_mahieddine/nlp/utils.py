"""Shared code for the NLP notebooks: corpus, queries, VLM descriptions, retrieval rungs, judging, metrics.

Protocol in ADR 0013. Two corpora, never mixed: `dev` = Museum A validation (480 paintings), `test` = Museum B
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
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "wikiart_genre5"  # shared with vision: read-only here (metadata.csv, split.csv)
NLP_DATA_DIR = DATA_DIR / "nlp"  # NLP-only scratch (embedding cache, judging pages), never read by vision
CACHE_DIR = NLP_DATA_DIR / "cache"
RESULTS_DIR = REPO_ROOT / "results" / "nlp"
# Committed: id -> description for dev + test only. Joined on id at load time; metadata and split stay in their
# original files so there is one source of truth for the split.
DESCRIPTIONS_FILE = RESULTS_DIR / "paintings_described.csv"
FROZEN_CONFIG = RESULTS_DIR / "frozen_config.json"
CORPUS_SPLITS = {"dev": "val", "test": "test"}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 6  # per request; backoff up to 60 s, or the server's Retry-After

# ponytail: minimal .env reader (KEY=VALUE lines) instead of adding python-dotenv to the shared pyproject.
_ENV_FILE = REPO_ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _key, _value = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))

VLM_MODEL = "qwen/qwen3-vl-8b-instruct"
# The length limit is in the prompt, not in max_tokens: a 60-token cap cut 99% of descriptions mid-sentence.
VLM_PROMPT = (
    "Describe this painting for a museum visitor in exactly 2 short sentences (at most 45 words in total): "
    "the visible subjects and setting, then the colours, light and atmosphere."
)
VLM_MAX_TOKENS = 120  # safety net only, well above 45 words
# Judge from another model family than the description generator, so it does not favour Qwen-written text.
JUDGE_MODEL = "google/gemini-3.8-flash"
MIN_KAPPA = 0.6  # judge accepted only if its agreement with the human sample is at least "substantial"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
HYBRID_ALPHA = 0.5  # ASSUMPTION: equal weight, not tuned (ADR 0013)
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
def ask_vlm(model: str, image: Path, text: str, max_tokens: int, **options: object) -> str:
    """One image + text request to OpenRouter (temperature 0), retried with backoff. Needs OPENROUTER_API_KEY."""
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens, **options,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()}},
            {"type": "text", "text": text},
        ]}],
    }).encode()
    request = urllib.request.Request(OPENROUTER_URL, data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            # Only rate limits (429) and server errors are worth retrying; 4xx like a bad key fail at once.
            if attempt == MAX_ATTEMPTS - 1 or (e.code != 429 and e.code < 500):
                raise
            time.sleep(float(e.headers.get("Retry-After") or min(60, 2 ** (attempt + 1))))
        except (urllib.error.URLError, TimeoutError, KeyError):
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise AssertionError("unreachable")


def _run_parallel(jobs: dict[object, tuple], fn, on_result, save, workers: int = 4) -> int:
    """Run fn(*args) per job in parallel; on_result(key, value) per success, save() every 100 and at the end.

    Returns the number of failures (re-running skips what is already saved)."""
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): key for key, args in jobs.items()}
        for n, future in enumerate(as_completed(futures), start=1):
            try:
                on_result(futures[future], future.result())
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
                failures += 1
                print(f"failed {futures[future]}: {e} (re-run to retry)")
            if n % 100 == 0:
                save()
                print(f"{n}/{len(futures)}")
    save()
    return failures


def generate_descriptions(limit: int | None = None) -> dict[str, str]:
    """Describe every dev + test painting not yet in DESCRIPTIONS_FILE (resumable, saved at the end)."""
    paintings = pd.concat([load_corpus("dev"), load_corpus("test")])
    done = load_descriptions() if DESCRIPTIONS_FILE.exists() else {}
    todo = paintings[~paintings["id"].isin(done)]
    todo = todo if limit is None else todo.head(limit)
    jobs = {r.id: (VLM_MODEL, DATA_DIR / r.path, VLM_PROMPT, VLM_MAX_TOKENS) for r in todo.itertuples()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        pd.Series(done, name="description").rename_axis("id").sort_index().to_csv(DESCRIPTIONS_FILE)

    failures = _run_parallel(jobs, ask_vlm, done.__setitem__, save)
    print(f"{len(done)} descriptions in {DESCRIPTIONS_FILE}, {failures} failed")
    return done


def load_descriptions() -> dict[str, str]:
    return pd.read_csv(DESCRIPTIONS_FILE, index_col="id")["description"].to_dict()


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
# Judging: blind, graded (0/1/2), pooled on the top-K, incremental; LLM judge validated against a human sample
# --------------------------------------------------------------------------------------------------
# Same scale for the human and the LLM judge: Q1 "is the main subject of the query visible?",
# Q2 "does the painting match the whole query?". Grade = 0 if not Q1, 2 if Q1 and Q2, else 1.
RUBRIC = [
    "0 - the main subject of the query is not in the painting.",
    "1 - the main subject is there, but part of the query is not satisfied (e.g. a sea, but calm).",
    "2 - the painting matches the whole query.",
]


def qrels_path(corpus: str) -> Path:
    return RESULTS_DIR / f"qrels_{corpus}.json"


def load_qrels(corpus: str) -> dict[str, dict[str, int]]:
    path = qrels_path(corpus)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def top_k_ids(scores: np.ndarray, paintings: pd.DataFrame, k: int = K) -> list[list[str]]:
    ids = paintings["id"].to_numpy()
    return [ids[np.argsort(-row)[:k]].tolist() for row in scores]


# Two yes/no questions, grade derived by rule: asked for a 0/1/2 digit directly, the judge answered 1 even when it
# wrote that the subject was absent ("no skull visible. Grade: 1").
JUDGE_PROMPT = (
    'A museum visitor searched for: "{query}"\n\n'
    "Look at the attached painting and answer two yes/no questions.\n"
    "Q1: Is the main subject of the search visible in the painting?\n"
    "Q2: Does the painting match the whole search, including every detail it asks for?\n\n"
    "Reply in exactly this format:\nSeen: <one short sentence>\nQ1: yes or no\nQ2: yes or no"
)


def _save_qrels(qrels: dict[str, dict[str, int]], path: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qrels, indent=1, sort_keys=True), encoding="utf-8")


def _llm_grade(query: str, image: Path) -> int:
    # Generous budget: the judge's hidden reasoning tokens count against max_tokens and cut short answers.
    answer = ask_vlm(JUDGE_MODEL, image, JUDGE_PROMPT.format(query=query), 2000)
    fields = dict(line.split(":", 1) for line in answer.splitlines() if ":" in line)
    q1, q2 = (fields.get(k, "").strip().lower().rstrip(".") for k in ("Q1", "Q2"))
    if q1 not in ("yes", "no") or q2 not in ("yes", "no"):
        raise ValueError(f"judge answered {answer!r}")
    return 0 if q1 == "no" else 2 if q2 == "yes" else 1


def judge_with_llm(corpus: str, scores: dict[str, np.ndarray], paintings: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Grade every top-K painting of the given rungs not graded yet. The judge sees only the query and the image."""
    qrels = load_qrels(corpus)
    paths = dict(zip(paintings["id"], paintings["path"]))
    jobs = {
        (qid, pid): (text, DATA_DIR / paths[pid])
        for qi, (qid, text) in enumerate(zip(QUERIES["query_id"], QUERIES["text"]))
        for pid in {p for s in scores.values() for p in top_k_ids(s, paintings)[qi]}
        if pid not in qrels.get(qid, {})
    }
    print(f"{len(jobs)} new (query, painting) pairs to grade with {JUDGE_MODEL}")
    failures = _run_parallel(jobs, _llm_grade, lambda key, g: qrels.setdefault(key[0], {}).__setitem__(key[1], g),
                             lambda: _save_qrels(qrels, qrels_path(corpus)))
    print(f"{sum(map(len, qrels.values()))} judgments in {qrels_path(corpus)}, {failures} failed")
    return qrels


def write_calibration_page(corpus: str, paintings: pd.DataFrame, n: int = 150) -> Path:
    """Seeded random sample of the judged pairs, for a human to grade blind (query + image only) on a static page."""
    pairs = sorted((qid, pid) for qid, grades in load_qrels(corpus).items() for pid in grades)
    sample = random.Random(SEED).sample(pairs, min(n, len(pairs)))
    text = dict(zip(QUERIES["query_id"], QUERIES["text"]))
    paths = dict(zip(paintings["id"], paintings["path"]))
    items = [{"qid": q, "query": text[q], "pid": p, "src": "../" + paths[p]} for q, p in sample]
    page = NLP_DATA_DIR / f"calibration_{corpus}.html"
    NLP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    page.write_text(_PAGE.replace("__ITEMS__", json.dumps(items)).replace("__KEY__", f"calibration_{corpus}")
                    .replace("__RUBRIC__", "".join(f"<li>{r}</li>" for r in RUBRIC))
                    .replace("__EXPORT__", f"calibration_{corpus}.json"), encoding="utf-8")
    print(f"{len(items)} pairs: open {page} in a browser, grade with keys 0/1/2, then click 'export JSON'.")
    return page


def import_calibration(corpus: str, export_file: Path) -> None:
    """Save the human grades of the calibration page as results/nlp/qrels_<corpus>_human.json."""
    if not Path(export_file).exists():
        raise FileNotFoundError(
            f"{export_file} not found. Grade the pairs first: open {NLP_DATA_DIR / f'calibration_{corpus}.html'} in a "
            "browser, grade with keys 0/1/2, click 'export JSON' (it downloads this file), then re-run this cell."
        )
    human = json.loads(Path(export_file).read_text(encoding="utf-8"))
    if any(g not in (0, 1, 2) for grades in human.values() for g in grades.values()):
        raise ValueError("grades must be 0, 1 or 2.")
    _save_qrels(human, RESULTS_DIR / f"qrels_{corpus}_human.json")


def judge_agreement(corpus: str) -> dict[str, float]:
    """Linear-weighted Cohen's kappa between the LLM judge and the human calibration grades, plus the confusion."""
    human = json.loads((RESULTS_DIR / f"qrels_{corpus}_human.json").read_text(encoding="utf-8"))
    llm = load_qrels(corpus)
    pairs = [(g, llm[q][p]) for q, grades in human.items() for p, g in grades.items()]
    h, j = zip(*pairs)
    print(pd.crosstab(pd.Series(h, name="human"), pd.Series(j, name="judge")))
    return {"kappa": float(cohen_kappa_score(h, j, weights="linear")), "n": len(pairs),
            "exact_agreement": float(np.mean(np.asarray(h) == np.asarray(j)))}


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
            "siglip_model": SIGLIP_MODEL, "hybrid_alpha": HYBRID_ALPHA, "judge_model": JUDGE_MODEL,
            "judge_prompt": _sha(JUDGE_PROMPT), "descriptions": _sha(DESCRIPTIONS_FILE.read_text(encoding="utf-8")),
            "queries": _sha(QUERIES.to_json())}


def freeze(best_single: str) -> None:
    """Write the config used on dev; refuses if the LLM judge was not validated on dev.

    `best_single` (R3 or R4, chosen on dev) is the reference for the hybrid on Museum B.
    """
    agreement = judge_agreement("dev")
    if agreement["kappa"] < MIN_KAPPA:
        raise RuntimeError(f"Judge kappa {agreement['kappa']:.2f} < {MIN_KAPPA}: grade dev by hand instead (ADR 0013).")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {**current_config(), "best_single": best_single, "judge_kappa_dev": round(agreement["kappa"], 3)}
    FROZEN_CONFIG.write_text(json.dumps(config, indent=1), encoding="utf-8")


def check_frozen() -> dict[str, str | float]:
    """Refuse to touch Museum B unless the frozen config still matches the code, models, queries and descriptions."""
    if not FROZEN_CONFIG.exists():
        raise RuntimeError("Museum B is locked: finish the dev notebooks and call freeze() first.")
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    drift = {k: v for k, v in current_config().items() if frozen.get(k) != v}
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
