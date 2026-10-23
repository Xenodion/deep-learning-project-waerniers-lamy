# 0005. Natural-Language Painting Recommendation as an Experimental Multimodal Ladder

- **Status:** Accepted, amended by [0007](0007-nlp-evaluation-protocol.md) (ladder and evaluation)
- **Date:** 2026-10-22

**Context.** [0004](0004-focus-on-genre-classifier.md) deferred the language feature ("Which painting shows a stormy sea?") to protect the core vision deliverable. With Alex focused on the vision pipeline, we pursue the recommendation feature in parallel. The raw dataset has no captions, only genre, style, and artist (with 54% "Unknown Artist").

A naive metadata-only search cannot answer content queries ("turbulent sea", "portraits of children") because these concepts are absent from the metadata. Conversely, deploying a full generative RAG architecture with heavy vector databases and an LLM chatbot on a museum kiosk CPU would violate the project's latency, compute, and hardware constraints.

**Decision.** We implement a progressive, experimental recommendation ladder without custom deep-learning training:
1. **Strict adherence to Museum A / Museum B ([0003](0003-simulated-museum-b-test-set.md)):**
   - **Museum A (`val`, 480 paintings):** used exclusively for prompt engineering, benchmark tuning, hyperparameter exploration (e.g., hybrid weighting $\alpha$), and intermediate evaluations (E0 to E5).
   - **Museum B (`test`, 600 paintings, 44 unseen artists):** locked during development, evaluated **exactly once** on the final chosen system for the oral defense.
2. **Experimental ladder (E0 $\to$ E5):**
   - **E0 (Baseline):** Metadata + Sentence-BERT (`all-MiniLM-L6-v2`) $\to$ establishes the quantifiable failure floor on visual content.
   - **E1 / E2:** VLM generated descriptions (tested on a sample of Museum A) + text embedding $\to$ demonstrates that the bottleneck was missing visual data.
   - **E3 / E4:** Multimodal CLIP text-to-description and direct text-to-image embeddings.
   - **E5 (Hybrid):** Weighted combination of direct visual similarity and semantic description similarity.
3. **Lightweight production deployment:**
   - In-memory similarity search (`paintings.parquet` + `.npy` embeddings), avoiding unnecessary vector database infrastructure (< 1 ms latency, < 10 MB RAM for 3,000 images).
4. **VLM Captioning Calibration (Precision vs. Semantic Dilution):**
   - Prompt is strictly constrained to 2 sentences (~35–45 words). This strikes the calibrated balance between providing rich visual coverage (subjects, palette, lighting) and preventing **semantic dilution** (where mean-pooling in dense bi-encoders averages out distinct features when matching short visitor queries).

**Consequences.** Clear division of labor between Vision (Alex) and Recommendation (Vincent). The rungs, metrics and test protocol actually used are in [0007](0007-nlp-evaluation-protocol.md): the first benchmark and its Recall/MRR results turned out to be invalid.
