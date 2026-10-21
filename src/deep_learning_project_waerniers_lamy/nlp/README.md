# nlp

Natural-language bonus of the guide (answering visitor questions), built one rung at a time:
see [ADR 0005](../../../adr/0005-nlp-bonus-by-ladder.md) and [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

- `utils.py`: `load_paintings` (shared `split.csv`), ranking metrics, image grid.
- Notebooks: [`notebooks/nlp/`](../../../notebooks/nlp/).

Constraint to remember: the dataset has no captions, only genre, style and artist.
