# Natural-Language Painting Recommendation — Experimental Plan

## 1. Objective

Build a system that takes a natural-language query such as:

> "Find me a painting of a turbulent sea under a dark sky"

and retrieves the most relevant paintings from the collection.

Available data:
- Painting image (224 x 224 px, RGB)
- Artist (available for 46% of images; 54% are "Unknown Artist")
- Genre (5 balanced classes: cityscape, landscape, portrait, religious_painting, still_life)
- Style (27 fine art styles)
- No existing captions or visual descriptions

Production & Academic Constraints:
- Modest kiosk CPU hardware (low latency < 200 ms, low RAM < 500 MB)
- Strict split hygiene (Museum A for dev/validation, Museum B isolated for test, ADR 0003)
- No unnecessary model training (prefer frozen open-weights, zero-shot and contrastive embeddings)
- Step-by-step experimental proof (Recall@5, Recall@10, MRR, latency, complexity) for the oral defense

---

## 2. Core Hypothesis & Data Bottleneck

> The primary bottleneck of metadata-only search is not the retrieval algorithm, but the complete absence of visual information in the metadata.

A query like "turbulent sea" or "portraits of children" cannot match paintings via metadata because words like "sea", "turbulent", or "children" do not exist in the genre/style/artist columns.

The experimental ladder systematically tests this hypothesis:

```text
E0: Metadata + BERT/MiniLM (Baseline)
       ↓
E1/E2: VLM-generated visual descriptions + BERT/MiniLM
       ↓
E3:    VLM-generated visual descriptions + CLIP text embeddings
       ↓
E4:    Direct text → image retrieval (CLIP)
       ↓
E5:    Hybrid multimodal retrieval (Image + Description + Metadata)
       ↓
E6:    Structured visual attributes & re-ranking (Optional)
```

---

## 3. Strict Dataset Split Hygiene (ADR 0003 Alignment)

In strict accordance with the project's data science rules:

| Role | Split used | Dataset size | Usage rules |
|---|---|---|---|
| **Development & Tuning** | **Museum A (`validation`)** | 480 paintings | Benchmark definition, prompt tuning for VLM, similarity thresholds, hybrid weight $\alpha$ tuning. |
| **VLM Training / Prompts** | **Museum A (`train`)** | 1,920 paintings | Used if any supervised fine-tuning or few-shot exemplars are needed. |
| **Final Test** | **Museum B (`test`)** | 600 paintings | **Locked during development**. Evaluated **exactly once** on the finalized system to measure out-of-distribution generalization across 44 unseen artists. |

---

## 4. Evaluation Benchmark & Metrics

### 4.1 Fixed Benchmark (Museum A `val`)
A fixed deck of 30–50 representative visitor queries covering:
1. **Visual concepts:** Direct scenes/objects ("boats on water", "stormy sky", "fruit basket", "snowy mountain").
2. **Compound concepts:** Combinations of lighting, colors, and mood ("dark gloomy portrait", "bright sunny landscape").
3. **Metadata queries:** Checks that genre/artist/style queries still work ("baroque portrait", "cubist still life").
4. **Out-of-domain / Negative queries:** Queries for non-existent objects to verify ranking scores remain low ("modern sports car").

### 4.2 Metrics & Definitions

- **Queries**: The set of test questions evaluated (e.g. 22 benchmark queries across visual concepts, compound queries, and metadata).
- **Recall@k (Recall@5 & Recall@10)**:
  Measures the proportion of queries for which **at least one relevant painting is successfully retrieved within the top-$k$ recommendations** shown to the visitor on the kiosk screen.
  - Formally:
    $$\text{Recall}@k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{Top-}k(q) \cap \text{Relevant}(q) \neq \emptyset)$$
  - *Intuition:* Recall@5 measures if the visitor finds what they want immediately in the first 5 paintings without scrolling. Recall@10 checks within 10 paintings.
- **Mean Reciprocal Rank (MRR)**:
  Measures the **quality of the ranking order**. It evaluates how close to the top position the first relevant painting appears:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$
  - If the first relevant painting is ranked 1st $\to$ score = 1.0 (perfect).
  - If 2nd $\to$ score = 0.5; if 5th $\to$ score = 0.2; if not found $\to$ score = 0.0.
  - A higher MRR proves that relevant paintings are pushed to the top of the screen.
- **Latency (ms/query)**: Average retrieval response time on CPU (target < 200 ms).
- **Memory footprint (MB)**: In-memory RAM consumption on the kiosk device (target < 500 MB).

---

## 5. Experimental Sequence

### E0 — Metadata Baseline
- Searchable text: `f"{artist}, {style}, {genre}"`
- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Metric: Expected near 0% on visual content, acceptable on explicit metadata queries.

### E1 — VLM Description Generation (Pilot on 50–100 paintings)
- Model: Small open VLM (e.g. `Qwen2.5-VL-3B-Instruct` or similar)
- Generates rich concise descriptions + structured visual attributes (subject, mood, palette, lighting).
- Quality, hallucination rate, and inference speed checked before full Museum A validation run.

### E2 — VLM Descriptions + Text Encoder
- Searchable text: VLM description + metadata.
- Encoder: Same `all-MiniLM-L6-v2`.
- Isolates the exact contribution of textual visual information.

### E3 — VLM Descriptions + CLIP Text Embedding
- Encoder: CLIP text encoder.
- Tests whether CLIP's text representation provides a stronger semantic space for visual terms.

### E4 — Direct Text-to-Image CLIP
- Search: Text query embedding vs. pre-computed image embeddings.
- No text intermediate; measures direct visual matching.

### E5 — Hybrid Multimodal Retrieval
- Combined score:
  $$\text{score} = \alpha \cdot \text{sim}(\text{query}, \text{image}) + (1 - \alpha) \cdot \text{sim}(\text{query}, \text{description})$$
- Parameter $\alpha$ tuned strictly on Museum A `val`.

---

## 6. Architecture & Results Tracking

All runs record their metrics in `results/nlp_experiments.csv`:
`date, name, method, corpus, recall_at_5, recall_at_10, mrr, ms_per_query, file`
Embeddings are saved in `data/wikiart_genre5/embeddings/` (`.parquet` and `.npy`), keeping disk footprint minimal and search instantaneous.
