# 0007. Evaluate the NLP ladder with blind, pooled, graded judgments

- **Status:** Accepted (amends [0005](0005-nlp-recommendation-ladder.md))
- **Date:** 2026-10-23

**Context.** The first benchmark (`benchmark.py`, commit `e7414f2`) cannot be trusted. Its relevant paintings were not
chosen by looking at the images: 13 of the 74 ids do not exist in the dataset, 11 are outside the validation split, 7
have the wrong genre ("Virgin Mary holding baby Jesus" pointed to a landscape), and the same ids are reused for
unrelated queries (the Gauguin still-life query has the same targets as "vase with flowers" and "bowl of fruit"). It
listed 2 to 5 relevant paintings per query when each genre has 96, so a system finding an unlisted stormy sea was
penalised. With 22 queries, one query is worth 4.5 Recall points, there were no confidence intervals, and the hybrid
weight was tuned on the queries it was reported on. Every v0 number is void; they are kept in
`results/nlp/v0_biased_experiments.csv` only as a record.

**Decision.** We will measure the ladder like this:
- **Use case first.** The kiosk shows about 5 paintings. Main metric **nDCG@5** with grades 0 (no), 1 (partial: the
  subject is there, a qualifier is missing) and 2 (clear match); also **Success@5** (at least one grade 2) and the
  median CPU time per query. No Recall: it needs every relevant painting to be known.
- **Queries.** 40 visual queries in 4 categories (object, scene, mood and colour, compound), written before any v1 result
  existed, the same set on both museums. No metadata queries: their labels would come from the fields the baseline
  searches, so the baseline would win them by construction.
- **Judgments.** For each query, the top 5 of every rung are pooled and graded by hand on a page that shows only the
  query and the image (no description, artist, style or rung). Unjudged paintings count as 0. A query with nothing
  relevant judged is dropped, and the count is reported. Dev is judged rung by rung (one notebook per rung, only the new
  paintings each time); Museum B is judged in one shuffled pass, fully blind.
- **Ladder.** Two branches then a fusion, one change at a time inside the text branch: R0 random, R1 metadata, R2 VLM
  description, R3 description + metadata (MiniLM); R4 SigLIP image; R5 = R3 + R4 on per-query z-scores (raw cosines of
  the two encoders are on different scales) with equal weight, **not tuned**. Descriptions: one model
  (`qwen/qwen3-vl-8b-instruct`, temperature 0) and one prompt for both museums.
- **Statistics.** Per-query scores, 95% bootstrap confidence intervals, and only three comparisons fixed in advance
  (paired randomization test, Holm correction): R2 vs R1, R4 vs R1, R5 vs the better of R3 and R4 on dev. Results per
  category are shown without p-values (about 10 queries each).
- **Museum B once.** After dev, `freeze()` records the models, prompt, weight and the hashes of the descriptions and
  queries. The test notebook refuses to run if any of them changed.

**Rejected.** Fixing the old ids (the qrels would still be incomplete and the queries already seen with results);
tuning the hybrid weight (judgments pooled from a few weights favour those weights); an LLM or VLM judge (cheap, but
needs validation against human grades, and a Qwen judge would favour Qwen descriptions); web scraping of painting texts
(`results/nlp/v0_scraping_feasibility_report.json`: 25% of paintings identified, 0% with an unknown artist).

**Consequences.** About 800 judgments per museum, roughly 45 minutes each. The numbers become defensible but uncertain:
with 40 queries, gaps of a few nDCG points are not detectable, so we read the intervals before the p-values. Known
limits to state in the report: a single judge who also built the systems; dev judged rung by rung; relevant paintings
that no rung retrieved stay invisible (estimated with R0's random paintings: the pool miss rate); the VLM may recognise
famous paintings (checked with the share of descriptions naming the artist). Dev and Museum B absolute scores are not
comparable (480 vs 600 paintings); the gaps between rungs are. A new rung is scored only after its new top-5 paintings
are judged.
