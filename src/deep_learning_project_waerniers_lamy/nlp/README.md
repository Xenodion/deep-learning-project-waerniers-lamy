# nlp

The language part of the guide: a visitor describes a painting in words ("a stormy sea with big waves") and the kiosk
shows the 5 best matches. Decisions in [ADR 0005](../../../adr/0005-nlp-recommendation-ladder.md) (why) and
[ADR 0007](../../../adr/0007-nlp-evaluation-protocol.md) (how it is measured).

Run the notebooks in order; each adds one rung and compares it with the previous ones on the same 40 queries.

| Notebook | Adds | Question |
|---|---|---|
| `N01_baseline_metadata` | R0 random, R1 metadata | how far does genre + style + artist go? |
| `N02_vlm_descriptions` | R2 description, R3 description + metadata | does visual text beat metadata? |
| `N03_siglip_image` | R4 SigLIP image | does searching the image itself beat metadata? |
| `N04_hybrid` | R5 hybrid; then `freeze()` | does fusion beat the better branch? |
| `N05_test_museum_b` | all rungs on Museum B, once | do the gaps hold on unseen artists? |

Each notebook writes a judging page (`data/wikiart_genre5/judging_<corpus>.html`, query + image only) for the paintings
not graded yet; grade them, export, and the next cell merges them into `results/nlp/qrels_<corpus>.json`.

All shared code is in `utils.py`. N02 needs an OpenRouter key (`OPENROUTER_API_KEY`) to write the descriptions once
(about $0.30 for dev + test); everything else runs on CPU.

`results/nlp/v0_*` are the results of the first benchmark, kept as a record only: they are invalid (ADR 0007).
