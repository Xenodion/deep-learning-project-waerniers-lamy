# nlp

The language part of the guide: a visitor describes a painting in words ("a stormy sea with big waves") and the kiosk
shows the 5 best matches. Decisions in [ADR 0005](../../../adr/0005-nlp-recommendation-ladder.md) (why) and
[ADR 0007](../../../adr/0007-nlp-evaluation-protocol.md) (how it is measured).

**Results first:** [`SUMMARY.md`](SUMMARY.md), with its figures from `N00_executive_summary` (reads `results/nlp/`
only, no API call).

Run the notebooks in order; each adds one rung and compares it with the previous ones on the same 40 queries.

| Notebook | Adds | Question |
|---|---|---|
| `N01_baseline_metadata` | R0 random, R1 metadata | how far does genre + style + artist go? |
| `N02_vlm_descriptions` | R2 description, R3 description + metadata | does visual text beat metadata? |
| `N03_siglip_image` | R4 SigLIP image | does searching the image itself beat metadata? |
| `N04_hybrid` | R5 hybrid; then `freeze()` | does fusion beat the better branch? |
| `N05_test_museum_b` | all rungs on Museum B, once | do the gaps hold on unseen artists? |

Each notebook sends the new top-5 paintings of its rung to an LLM judge (query + image only); grades go to
`results/nlp/qrels_<corpus>.json`. N04 asks you to grade 150 of them by hand on a local page
(`data/wikiart_genre5/nlp/calibration_dev.html`) to measure the judge's agreement; `freeze()` refuses below kappa 0.6.

All shared code is in `utils.py`. The VLM descriptions (`results/nlp/paintings_described.csv`, committed) and the
judge go through OpenRouter: put `OPENROUTER_API_KEY=...` in the repo's `.env` (gitignored). A few dollars in total;
retrieval itself runs on CPU.

`results/nlp/v0_*` are the results of the first benchmark, kept as a record only: they are invalid (ADR 0007).
