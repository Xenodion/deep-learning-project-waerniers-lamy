# Synthesis of Models, Methodology and Evaluation Metrics

> **Project:** Intelligent Multimodal Museum Guide  
> **Authors:** Alexandre Waerniers, Vincent Lamy & Yoni Mahieddine  
> **Course:** Deep Learning — Final Project Deliverable  
> **Reference:** [Live Demo](https://museum-guide-643636450457.europe-west1.run.app) | [ADR Index](adr/README.md)

---

## 1. Project Context & Objectives

A fictional mid-sized French fine-arts museum wants an intelligent guide accessible on **visitors' smartphones** and on **gallery kiosks**. The guide performs two distinct tasks:

1. **Artwork Recognition (Vision):** The visitor points their camera at a painting; the guide identifies its genre (*portrait, landscape, cityscape, still life, religious painting*) and provides historical and artistic context.
2. **Collection Search (Natural Language / Multimodal):** The visitor types a free-form question (*"a stormy sea with big waves"*, *"portraits of children"*); the guide retrieves and ranks the top 5 artworks from the collection that best match the query.

### The Director's Strategic Question
> *"The kiosks run on modest CPUs, the team is small, and the recurring budget for external APIs is zero. Is a deep neural network worth its production cost, or would a simpler classical model do the job?"*

Our approach builds, diagnoses, and benchmarks every component from PyTorch primitives, measuring every trade-off with empirical evidence.

---

## 2. Scientific Methodology & Evaluation Protocol

### A. Strict Split: Museum A vs Museum B (Zero Data Leakage)
To assess generalization to completely unseen art collections ([ADR 0003](adr/0003-simulated-museum-b-test-set.md)), the 3,000 paintings from WikiArt (balanced across 5 genres, 600 paintings each) are divided into two strictly isolated sets:

* **Museum A (2,400 paintings — 80%):** Used for development, training, and architectural selection.
  - **Train (1,920 paintings):** Weight optimization. Normalization statistics (ImageNet RGB channel mean and standard deviation) are applied consistently.
  - **Validation (480 paintings):** Hyperparameter tuning, model comparison, and early stopping.
* **Museum B (600 paintings — 20% — 120 per genre):** Simulates an independent partner museum containing **44 artists who never appear in Museum A**.
  - **Strict protocol rule:** Museum B was scored **only once per model candidate**, after freezing all decisions on validation. It was never used for model tuning or selection.

### B. Multi-Seed Replication
To ensure differences are not artifacts of lucky random initialization, core experiments were replicated across **5 independent seeds** (`seed` 0 to 4). Reported metrics include mean and standard deviation (`mean ± std`).

### C. Artist-Grouped Cross-Validation (GroupKFold)
In Museum A, multiple paintings often share the same painter. To verify whether networks learn genuine genre features rather than the idiosyncratic "brushstroke signature" of known artists ([ADR 0021](adr/0021-evaluation-rigour.md)), we benchmarked random 5-fold cross-validation against artist-grouped cross-validation (`GroupKFold`).

### D. Statistical Hypothesis Testing
On Museum B test scores, differences between architectures were validated using **exact paired McNemar tests** with **Holm-Bonferroni family-wise error correction** at significance level $\alpha = 0.05$.

---

## 3. Detailed Explanation of Evaluation Metrics

### A. Computer Vision Metrics (Artwork Genre Recognition)

| Metric | What It Measures Conceptually & Mathematically | Why We Chose It for This Project |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | Proportion of paintings where the class with the highest softmax probability matches the ground-truth genre:<br>$$\text{Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$ | The 5 classes are strictly balanced (120 test paintings per genre in Museum B). Top-1 Accuracy is free of majority-class bias and directly represents the visitor's raw success rate. |
| **Cross-Entropy Loss (Log-Loss)** | Negative log-likelihood of the true class given the predicted softmax probabilities ($p_{i,c}$):<br>$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \ln(p_{i,c})$$ | Measures probabilistic confidence and penalizes over-confident errors. In a museum setting, a model that makes a mistake with 99% certainty misleads the visitor far more than one that expresses honest doubt. |
| **Top-2 Accuracy & Selective Abstention** | Proportion of paintings where the true genre is within the **top two** most probable predictions:<br>$$\text{Acc}_{\text{top-2}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in \{\hat{y}_i^{(1)}, \hat{y}_i^{(2)}\})$$ | Artworks can be stylistically ambiguous (e.g., an allegorical portrait vs. religious art). Showing the top two genres when top-1 confidence falls below **0.86** increases coverage from **86.0% to 94.2%** ([ADR 0020](adr/0020-confidence-and-abstention.md)). |
| **Expected Calibration Error (ECE)** | Weighted difference between predicted softmax confidence and empirical accuracy across confidence bins ($B_m$):<br>$$\text{ECE} = \sum_{m=1}^M \frac{\|B_m\|}{N} \|\text{acc}(B_m) - \text{conf}(B_m)\|$$ | Visitors trust models that state calibrated probabilities. We apply post-training temperature scaling ($T = 1.23$) to bring softmax probabilities into honest alignment with real accuracy. |
| **CPU Latency (ms / photo)** | Median inference time per image on 1 thread of a standard CPU (replicating a modest gallery kiosk processor). | Kiosks lack dedicated GPUs. Sub-100 ms inference provides instantaneous user feedback without UI lag. |
| **Memory Footprint (MB & Parameters)** | Checkpoint file size on disk and RAM consumption during inference. | Ensures fast kiosk startup and prevents memory saturation alongside the operating system and UI services. |

### B. Natural Language Processing Metrics (Collection Search)

| Metric | What It Measures Conceptually & Mathematically | Why We Chose It for This Project |
| :--- | :--- | :--- |
| **nDCG@5 (Normalized Discounted Cumulative Gain @ 5)** | Evaluates the ranked list of 5 retrieved paintings with graded relevance scores ($rel \in \{0, 1, 2\}$), discounted logarithmically by rank:<br>$$\text{DCG}@5 = \sum_{r=1}^5 \frac{2^{rel_r} - 1}{\log_2(r + 1)}, \quad \text{nDCG}@5 = \frac{\text{DCG}@5}{\text{IDCG}@5}$$ | The kiosk interface displays exactly 5 candidate paintings. Artistic relevance to free-text queries is not binary: artworks can be an exact match ($rel=2$), partial match ($rel=1$), or off-topic ($rel=0$). nDCG rewards placing the best matches at ranks 1 and 2. |
| **Linearly Weighted Cohen's Kappa ($\kappa$)** | Inter-annotator agreement between the automated LLM judge (Gemini 3.8 Flash) and a blinded human curator on 150 query-painting pairs:<br>$$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}$$ | Scientifically validates the evaluation judge, proving automated grades do not inflate scores ($\kappa = 0.65$, 72% exact agreement). |

---

## 4. Chronological Summary of All Tested Models

### Part 1: Computer Vision — Artwork Genre Classification

```
[P02 Dense Baseline] ➔ [P03-P05 Custom CNNs] ➔ [P06 Pretrained ResNet-18] ➔ [P11 Classical Baseline]
         │                                                    │
         ▼                                                    ▼
[P12-P14 Diagnosis & MLP]                       [P15-P17 Kiosk Readiness & Robustness] ➔ [P18-P19 Rigour & Grad-CAM]
```

| Step | Model / Experiment | Input Res. | Validation Acc. | Museum B Test Acc. | CPU Latency | Decision & Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **P02** | **Dense Baseline**<br>MLP (1 hidden layer, 128 ReLUs) | 32 px | 39.0% (±0.9) | 41.8% | **0.1 ms** | **Accepted as minimal sanity anchor** ([ADR 0006](adr/0006-dense-baseline.md)). Verifies PyTorch training pipeline. Massive memorization (100% train, 39% val). |
| **P03** | **First Custom CNN**<br>3 Conv-ReLU-MaxPool blocks [16, 32, 64], 30 epochs | 64 px | 53.6% (±0.8) | 58.7% | 1.2 ms | **Accepted** ([ADR 0007](adr/0007-first-cnn.md)). +14.6 pt jump over dense baseline: convolutional filters capture localized spatial structures. |
| **P04** | **CNN 100 Epochs & BatchNorm** | 64 px | 60.8% (no BN)<br>59.8% (with BN) | 63.7% | 1.7 ms | **Accepted without BatchNorm** ([ADR 0008](adr/0008-longer-training-and-deeper-cnn.md)). Longer training yields +7.2 pts. BatchNorm provided no gain on this shallow custom network. |
| **P05** | **Deeper Custom CNN**<br>4 Conv blocks [16, 32, 64, 128], 100 epochs | 64 px | 58.4% (±1.1) | 60.5% | 1.8 ms | **Rejected** ([ADR 0009](adr/0009-move-to-pretrained-resnet.md)). Drops by -2.4 pts. Training deeper CNNs from scratch on 1,920 images accelerates overfitting. Reached ceiling of custom CNNs. |
| **P06** | **ResNet-18 (Frozen)**<br>ImageNet backbone frozen + linear head | 224 px | 80.3% (±0.5) | 84.5% | 70.3 ms | **Major breakthrough (+19.5 pts)** ([ADR 0010](adr/0010-resnet-results-and-improvements.md)). Pretrained ImageNet representations transfer immediately to fine-arts imagery. |
| **P06** | **ResNet-18 (Fine-tuned `layer4`)**<br>Layers 1-3 frozen, block 4 retrained | 224 px | **83.3%** (±0.0) | **86.0%** | **70.5 ms** | **CHOSEN PRODUCTION MODEL** ([ADR 0012](adr/0012-final-model.md)). +3.0 pt gain on validation at identical inference cost to the frozen backbone. |
| **P07** | **Resolution Impact Control**<br>Frozen ResNet-18 at 64 px and 112 px | 64 px<br>112 px | 70.2%<br>75.4% | Not scored | 10.5 ms<br>21.6 ms | Diagnostic milestone ([ADR 0010](adr/0010-resnet-results-and-improvements.md)): Image resolution accounts for ~half the CNN/ResNet gap, pretraining accounts for the other half. |
| **P08** | **ResNet Improvement Round**<br>- Data Augmentation (crop/flip)<br>- ResNet-50 frozen<br>- Test-Time Flip | 224 px | 82.3%<br>81.9%<br>83.5% | Not scored | -<br>139.1 ms<br>141.0 ms | **All rejected** ([ADR 0011](adr/0011-resnet-improvement-round.md)). Augmentation degrades validation, ResNet-50 doubles latency for negligible gain (+1.6 pt, below our 2-pt threshold), flip is statistical noise. |
| **P11** | **Classical Baseline**<br>1,802 colour/texture/edge features + Random Forest | 224 px | 60.7% (±2.2) | 58.5% | 4.2 ms | **Comparative reference** ([ADR 0014](adr/0014-classical-baseline.md)). Random Forest outperforms dense baseline (41.8%) and ties custom CNN (63.7%), but trails ResNet-18 by **-27.5 points**. |
| **P12** | **Training Doctor (Diagnostic)**<br>Gradient monitoring, dead ReLUs, sigmoids | 32 px | Diagnostic run | - | - | **Inside-network audit** ([ADR 0015](adr/0015-training-doctor.md)): Dense network overfitting is not an implementation bug (gradients remain healthy), but an architectural inability of MLPs on raw pixels. |
| **P13** | **Dense Regularization**<br>Weight decay, Dropout, Mirrored copies | 32 px | 42.5% (±0.6) | 41.0% | 0.1 ms | **Confirmed architectural limit** ([ADR 0016](adr/0016-dense-regularisation.md)). Regularization shrinks train/val gap from 60 to 23 pts, but validation accuracy stalls at 41%. |
| **P14** | **Feed-Forward on Good Features**<br>MLP on 1,802 handcrafted features | - | 60.5% (±0.5) | 62.7% | 0.2 ms | **Fundamental finding** ([ADR 0017](adr/0017-feedforward-on-features.md)): Given equal engineered features, the MLP ties with Random Forest (62.7% vs 58.5%, p=0.14). **Features dictate performance, not the algorithm.** |
| **P15** | **Phone Photo Robustness**<br>9 physical camera perturbations | 224 px | Degradation profile | 50% to 84% | - | **Operational audit** ([ADR 0018](adr/0018-phone-photo-robustness.md)): Model resists lighting and tilt (drop ≤ 4 pts), but collapses on blur and heavy JPEG compression. Recommendation: app must reject blurred photos. |
| **P16** | **INT8 Kiosk Quantization**<br>Post-training static quantization (PyTorch FX) | 224 px | 83.3% | **86.2%** | **52.1 ms** | **RECOMMENDED FOR KIOSK PRODUCTION** ([ADR 0019](adr/0019-kiosk-int8.md)). Preserves float32 accuracy, **divides model footprint by 4** (11 MB vs 45 MB), and **accelerates CPU speed by 26%**. |
| **P16** | **MobileNetV3-Large (Frozen)** | 224 px | 82.9% | 84.8% | 88.4 ms | Slower than quantized ResNet-18 on standard desktop x86 CPU (depthwise convolutions require specialized mobile hardware accelerators). |
| **P17** | **Confidence & Abstention** | 224 px | Top-2: 94.2% | - | - | Calibrated via temperature scaling ($T=1.23$). Displaying two genres when confidence is below 0.86 lifts coverage to 94.2%. |
| **P18** | **Evaluation Rigour** | - | Bias quantified | - | - | Seeing the artist during training adds 1.5 to 3.4 points in artificial shortcut. Paired McNemar tests prove only ResNet vs Random Forest is statistically significant (+27.5 pts, $p < 0.001$). |
| **P19** | **Grad-CAM Explainability** | 224 px | Faithful maps | - | - | Proves the model looks at salient subjects (faces in portraits, horizon lines in landscapes) rather than border/corner artifacts (outer border accounts for only 30.5% activation over 49% area). |

---

### Part 2: Multimodal NLP — Natural Language Collection Search

The 6-rung multimodal recommendation ladder evaluated across 40 diverse queries:

| Rung | Search Input Representation | Dev nDCG@5 | Museum B nDCG@5 | CPU Latency | Recurring Cost | Decision & Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **R0: Random** | None (random selection) | 0.07 | 0.10 | < 1 ms | $0 | Theoretical lower bound. |
| **R1: Metadata** | Title, Artist, Genre, Style (MiniLM) | 0.14 | 0.17 | 34 ms | $0 | **Fails completely.** Metadata does not capture what is depicted inside the artwork (e.g., "stormy waves", "red umbrella"). |
| **R2: VLM Description** | Offline text generated by Qwen3-VL-8B | 0.69 | 0.68 | 13 ms | $0.06 / 1k img | **Massive leap (+0.51 pt).** Visual descriptions bridge the semantic gap for depicted objects. |
| **R3: Description + Metadata** | Concatenated textual representations (MiniLM) | 0.71 | 0.68 | 13 ms | $0.06 / 1k img | Marginally helps object queries, but adds negligible value on Museum B over R2. |
| **R4: SigLIP Image** | **Direct visual embeddings via SigLIP Dual-Tower** | **0.81** | **0.79** | **82 ms** | **$0** | **CHOSEN PRODUCTION MODEL** ([ADR 0005](adr/0005-nlp-recommendation-ladder.md)). Direct text-to-image dot-product matching. 100% local, no API dependencies, zero recurring cost. |
| **R5: Hybrid** | Weighted score fusion of R3 + R4 (z-scores) | 0.83 | 0.80 | 115 ms | $0.06 / 1k img | **Rejected.** The gain over SigLIP alone (+0.01 on Museum B) is statistically indistinguishable from noise ($p = 0.78$). Does not justify external API dependencies. |

---

## 5. Final Strategic Recommendation for the Museum Director

1. **Is a neural network worth it for artwork recognition?**  
   **YES, unambiguously.**  
   The classical baseline (Random Forest on 1,802 colour and texture features) achieves only **58.5%**. The fine-tuned **ResNet-18 achieves 86.0%** (+27.5 percentage points, $p < 0.001$). A 58% system fails on almost every second visitor interaction, while 86% (and 94.2% with top-2 suggestion on ambiguous pieces) delivers a reliable cultural tool.
2. **How to satisfy the modest kiosk CPU constraint?**  
   By deploying the **INT8 post-training static quantized ResNet-18** ([ADR 0019](adr/0019-kiosk-int8.md)): memory footprint shrinks to **11 MB** (down from 45 MB), and inference takes only **52 ms per photo** on a single CPU thread with zero loss in accuracy (86.2% on Museum B).
3. **How to maintain zero recurring API costs for natural language search?**  
   By adopting **SigLIP direct image search (R4)**: visual embeddings of the collection are precomputed once offline (1.8 MB total). At runtime, only the lightweight text encoder runs on the kiosk CPU (82 ms), and dot-product ranking takes less than 1 ms. Recurring cloud API cost is strictly **$0.00**.
