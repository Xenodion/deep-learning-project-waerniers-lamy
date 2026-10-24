# 🏛️ Museum Intelligent Guide: Dual-Engine Deep Learning System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Final Project — Deep Learning Course**  
> **Authors:** Alexandre Waerniers & Vincent Lamy  
> **Context:** A fictional mid-sized French fine-arts museum wants an intelligent guide for gallery kiosks and visitors' phones. Gallery kiosks run on modest CPUs, the team is small, and recurring API budgets are zero. Every design choice is backed by evidence and recorded in Architecture Decision Records ([ADR](adr/README.md)).

---

## 🎯 Executive Summary & Deliverables

The guide delivers two core capabilities built with PyTorch primitives:

1. **Artwork Recognition (Vision)**: Point the camera at a painting to identify its genre (*portrait, landscape, cityscape, still life, religious painting*).
2. **Collection Search (NLP / Multimodal)**: Natural language queries (*"a stormy sea with big waves"*, *"portraits of children"*) retrieving the top matching artworks from the collection.

### Key Measured Results (Unseen Artists — Museum B)

| Engine | Final Architecture | Performance | CPU Latency | Recurring Cost | Decision & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision** | **Fine-tuned ResNet-18** (224 px, `layer4` trained) | **86.0% Test Accuracy** (83.3% Val) | **70 ms** / photo | $0 | Beats dense baseline (41.8%) and custom CNNs (63.7%). Pretraining and resolution explain the jump. ResNet-50 rejected (+1.6 pt for 2x latency). ([ADR 0012](adr/0012-final-model.md)) |
| **NLP** | **SigLIP Dual-Tower** (`google/siglip-base-patch16-224`) | **0.79 nDCG@5** (0.81 Dev) | **82 ms** / query | $0 | 100% local CPU dot-product. VLM hybrid fusion (+0.010, p=0.78) was rejected: statistically insignificant gain, eliminating paid API dependencies. ([ADR 0005](adr/0005-nlp-recommendation-ladder.md), [ADR 0013](adr/0013-nlp-evaluation-protocol.md)) |

---

## 🚀 Live Interactive Demo

Experience both production models in an interactive museum interface:

- **Online demo:** *(link added once deployed)*
- **Local demo** (needs `data/`, see below; the first command builds `hf_space/` from Museum B and checks that the bundled model reproduces the 86.0%):
  ```powershell
  uv run --with gradio python scripts/build_hf_space.py
  uv run --with gradio python demo.py
  ```
  *(Opens on `http://localhost:7860`. Add `--share` for a temporary public URL).*

---

## 🛠️ Quickstart & Reproducibility

Dependencies and Python 3.11 are managed with [`uv`](https://github.com/astral-sh/uv).

### 1. Installation

```bash
uv sync
```

### 2. Dataset Setup (WikiArt 5-Genre Subset)

The 3,000-image subset (~2.3 GB) and deterministic split (Museum A train/val vs Museum B test with unseen artists) are built via:

```bash
uv run --with "datasets>=3.0" python src/deep_learning_project_waerniers_lamy/vision/make_wikiart_subset.py --label genre --classes portrait landscape cityscape still_life religious_painting --per-class 600 --size 224 --out data/wikiart_genre5
uv run python -c "from deep_learning_project_waerniers_lamy.vision.utils import make_split; make_split()"
```

### 3. Reproduce Vision Final Score (Museum B Test)

Evaluates the saved production checkpoint (`results/final_model.pt`) on the 600 unseen test paintings of Museum B:

```bash
uv run python src/deep_learning_project_waerniers_lamy/vision/run_final.py
```
*(Add `--train` to retrain the model from scratch in ~15 minutes).*

---

## 📂 Simplified Repository Structure

```
.
├── demo.py                  # Standalone interactive museum web application (Gradio)
├── pyproject.toml, uv.lock  # Project dependencies and environment configuration
├── LICENSE                  # MIT License
├── adr/                     # Architecture Decision Records (why each choice was made)
│   ├── 9999-vision-executive-summary.md  # 1-page summary of the Computer Vision research
│   └── 9998-nlp-executive-summary.md     # 1-page summary of the Multimodal NLP research
├── results/                 # Metrics, benchmark logs, and production weights (final_model.pt)
├── src/deep_learning_project_waerniers_lamy/
│   ├── vision/              # Vision pipeline: from dense baselines (P02) to ResNet-18 (P06-P10)
│   └── nlp/                 # NLP pipeline: recommendation ladder from metadata (N01) to SigLIP (N03-N05)
└── scripts/                 # Utility scripts (e.g. Hugging Face Space packager)
```

---

## 📖 Architecture Decision Records (ADR)

Every experimental milestone is documented with hypothesis, results, and architectural verdict:
- **Vision Executive Summary:** [`adr/9999-vision-executive-summary.md`](adr/9999-vision-executive-summary.md)
- **NLP Executive Summary:** [`adr/9998-nlp-executive-summary.md`](adr/9998-nlp-executive-summary.md)
- **Test Protocol & Museum B Split:** [`adr/0003-simulated-museum-b-test-set.md`](adr/0003-simulated-museum-b-test-set.md)
- **Final Vision Model Selection:** [`adr/0012-final-model.md`](adr/0012-final-model.md)
- **NLP Evaluation & Human Calibration:** [`adr/0013-nlp-evaluation-protocol.md`](adr/0013-nlp-evaluation-protocol.md)

---

## ⚖️ License

Code released under the [MIT License](LICENSE). Dataset images belong to their respective copyright holders on WikiArt and are used strictly under educational and non-commercial research fair-use provisions.
