# Museum Guide: oral presentation guide

English rehearsal guide for the [latest ten-slide presentation](Museum_Guide_10min_EN_Six_Models.pptx).

**Total: 10 minutes, including 8 minutes of presentation and 2 minutes of live demo.** Each speaker has 2 minutes 30 seconds: Nahel, Vincent, Yoni, then Alexandre.

The schedule follows the timings displayed in the slides. Use the spoken text below for rehearsal. Stage directions and the reference notes are not part of the speech. Aim for about 110–130 words per minute, with short pauses on the main figures.

## Timing and handovers

| Slide | Title | Speaker | Time | Duration |
|---|---|---|---|---|
| 1 | Museum Guide | Nahel | 0:00–0:20 | 20 s |
| 2 | A guide to explore the collection | Nahel | 0:20–0:45 | 25 s |
| 3 | Data exploration shaped our strategy | Alexandre | 0:45–1:40 | 55 s |
| 4 | Split strategy: two simulated museums | Alexandre | 1:40–2:30 | 50 s |
| 5 | Vision: accuracy and CPU cost | Vincent | 2:30–3:50 | 80 s |
| 6 | The selected model and its limits | Vincent | 3:50–5:00 | 70 s |
| 7 | Image features improve search | Yoni | 5:00–6:20 | 80 s |
| 8 | SigLIP connects a query to paintings | Yoni | 6:20–7:30 | 70 s |
| 9 | Two models in one application | Nahel | 7:30–8:00 | 30 s |
| 10 | Live demonstration | Alexandre | 8:00–10:00 | 120 s |

Nahel hands over after slide 4 at **2:30**, Vincent after slide 6 at **5:00**, and Yoni after slide 8 at **7:30**. Alexandre opens the live application from slide 10 at **8:00**.

## Before presenting

- Rehearse with one shared timer and the final PowerPoint. Do not read every bar label.
- Open [the demo](https://museum-guide-643636450457.europe-west1.run.app) before the talk and allow time for a cold start. Run one classification and one search, then return to **1 · Recognise**.
- Choose a painting in advance and remember its location in the gallery. The demo selects collection images; it does not require a live camera or image upload.
- Keep these two requests ready: **A stormy sea with big waves.** and **A portrait of a woman wearing a hat.**
- Keep the recognition screenshot on slide 9 and the search screenshot on slide 10 available as a fallback.
- For a local demo, start `uv run --with gradio python demo.py` from the repository before the presentation, then open `http://localhost:7860`. The prepared `app/` bundle must be present.

## Spoken script, slide by slide

### Slide 1: Museum Guide

**Nahel, 0:00–0:20 (20 seconds)**

*Stage cue: Introduce the two functions and the two-minute demo.*

Hello. We are presenting Museum Guide, a deep learning application that recognises painting genres and retrieves artworks from natural language. We will explain how we built and evaluated it, then finish with a two-minute live demonstration.

### Slide 2: A guide to explore the collection

**Nahel, 0:20–0:45 (25 seconds)**

*Stage cue: Point to recognition, then search. Keep the constraints brief.*

A fictional museum needs two functions: identify a selected painting’s genre, or search its collection using a description. The app runs on a standard CPU without a paid API call for each request. Before choosing models, we explored the data and designed the test.

### Slide 3: Data exploration shaped our strategy

**Nahel, 0:45–1:40 (55 seconds)**

*Stage cue: Use the stacked bars to explain missing artist identities. Emphasise the shortcut baselines rather than reading every count.*

Our exploration shaped the strategy. WikiArt has over eighty-one thousand paintings but no captions. We selected three thousand images across five balanced genres. Fifty-four percent have an unknown artist. The rest come from one hundred and two named painters. Why genre rather than style? In our style pilot, ninety-three percent had unknown artists, making painter shortcuts difficult to audit, with roughly twice the download size. Those shortcuts are measurable: knowing only the artist predicts the genre in about seventy percent of eligible named-artist images, compared with forty-nine percent from style and twenty percent by chance. We therefore needed a test with unseen painters.

### Slide 4: Split strategy: two simulated museums

**Nahel, 1:40–2:30 (50 seconds)**

*Stage cue: Follow the diagram from the shared subset to A and B. Finish with the validation-only selection rule.*

Museum A contains fifty-eight named artists and all unknown-artist images. Its eighty-twenty split gives one thousand nine hundred and twenty training images and four hundred and eighty validation images. Museum B contains six hundred paintings from forty-four named artists absent from A. Unknown artists stay in A because we cannot verify their identities. Every genre represents twenty percent of each set. All tuning uses validation A. We score each completed model once on B. This tests transfer to unseen painters, rather than a random image split. Vincent will now explain the results.

### Slide 5: Vision: accuracy and CPU cost

**Vincent, 2:30–3:50 (80 seconds)**

*Stage cue: Identify the two series, highlight the final ResNet-18, then explain why ResNet-50 has no test bar.*

Here we compare validation and test accuracy, together with CPU cost. The classical random forest uses colour and edge features. It reaches fifty-eight point five percent on B, almost seventeen points above the dense network. Simply being a neural network is therefore not enough. Fine-tuned ResNet-18 reaches eighty-six percent, twenty-seven point five points above the classical baseline. Pretraining and image resolution drive the gain. ResNet-50 has only a validation bar. It gained one point six over frozen ResNet-18, below our predefined two-point threshold, while roughly doubling latency and model size. We rejected it before test evaluation, so its Museum B score is missing, not zero. The dotted line marks the twenty-percent chance level. We avoid strong claims about small differences, especially the one-point-five test gap between the two ResNet-18 models.

### Slide 6: The selected model and its limits

**Vincent, 3:50–5:00 (70 seconds)**

*Stage cue: Connect the result to unseen artists, then point to the two real classification errors.*

Our final model is ResNet-18 at two hundred and twenty-four pixels, with its last block and output layer fine-tuned. Validation accuracy rises from eighty point three to eighty-three point three percent at the same inference cost. The test score is eighty-six percent, with roughly a three-point uncertainty margin. Four of the five evaluated models improve on B. The classical baseline is the exception, falling from sixty to fifty-eight point five percent. There is no systematic collapse on unseen painters, which supports generalisation, but this is not proof that all shortcuts have disappeared. Museum B may simply be easier. Remaining errors involve overlapping genres, and clean scans do not reproduce phone photos. Yoni will now explain collection search.

### Slide 7: Image features improve search

**Yoni, 5:00–6:20 (80 seconds)**

*Stage cue: Explain nDCG@5 before comparing metadata, descriptions and SigLIP. State why the hybrid was not retained.*

For search, we followed the same approach: begin with a simple baseline and measure each addition. Metadata, such as artist and genre, poorly describes what visitors are looking for in an image. On Museum B, it reaches about zero point seventeen in nDCG at five. This metric measures the relevance of the top five results on a scale from zero to one. Descriptions generated by a vision-language model raise the score to about zero point sixty-eight. SigLIP compares the query directly with images and reaches about zero point seventy-nine. The hybrid combines both approaches and reaches about zero point eighty. However, this gap is too small to establish an improvement across our forty test queries, so we keep SigLIP alone. An AI judge provides the relevance labels, checked against a human on one hundred and fifty pairs, with a weighted kappa of zero point sixty-five.

### Slide 8: SigLIP connects a query to paintings

**Yoni, 6:20–7:30 (70 seconds)**

*Stage cue: Explain image indexing first, then query encoding and the dot product.*

Here is how we turned that choice into a usable search system. Before any visit, SigLIP’s image encoder converts each painting into a numerical vector. We compute these vectors once and store them alongside the artwork identifiers. During a visit, the text encoder turns the user’s sentence into a compatible vector. A dot product then compares the request with every painting, and the app displays the best matches. For a collection of six hundred paintings, this stays simple and runs on CPU, without a dedicated vector database or paid API call for each search. Reference latency is eighty-two milliseconds per query once the collection has been indexed. We can now move from the models to their integration into the application. I will hand over to Alexandre.

### Slide 9: Two models in one application

**Alexandre, 7:30–8:00 (30 seconds)**

*Stage cue: Use the screenshot to connect the models to the interface. Switch to the demo at 8:00.*

We integrated both models into a Gradio interface. A preparation script packages Museum B’s paintings, thumbnails, model weights and search vectors. It also checks that the shipped model reproduces the eighty-six percent score. Let us switch to the live demo: we will recognise a painting’s genre, then search for artworks using a description.

### Slide 10: Live demonstration

**Alexandre, 8:00–10:00 (120 seconds)**

*Stage cue: Keep the app loaded and follow the three timed actions below.*

[8:00–8:40: switch to the already loaded app, open “1 · Recognise”, and click a painting selected during rehearsal.]

Here we choose an artwork from Museum B, whose artist was absent from training. The model shows its predicted genre, the probabilities for the five categories and its response time. We can compare the prediction with the museum label. The probability distribution also shows how the model divides its confidence between related genres.

[8:40–9:30: open “2 · Ask”. Click “A stormy sea with big waves.”, then “A portrait of a woman wearing a hat.”.]

We now describe what we are looking for. Search retrieves paintings that match the subject of the sentence. When we change the request, the results change too. The interface displays six artworks, while our evaluation measures the ranking of the top five. The displayed response time comes from this live search.

[9:30–10:00: open “3 · Info”.]

This page brings together our results and evaluation protocol. We achieved eighty-six percent classification accuracy on six hundred test images, and about zero point seventy-nine in ranking quality for search. Both functions run on CPU without a paid API call for each use. Thank you.

## Reference figures for rehearsal and questions

These figures support the speech. Do not read this section aloud.

### Data exploration and split, slides 3–4

- WikiArt contains 81,444 paintings, 27 styles and 11 genres, with no captions. The genre classifier uses pixels, not artist or style metadata.
- The subset contains 3,000 images at 224 × 224 pixels, with 600 per genre: cityscape, landscape, portrait, religious painting and still life.
- There are 1,607 Unknown Artist images, about 54%, and 1,393 named-artist images from 102 painters.
- The artist-only shortcut baseline is 70.2% on 1,391 eligible named-artist images. Two singleton images cannot receive a leave-one-out prediction. The style-only baseline is 48.6% on all 3,000 images. Random guessing gives 20%.
- The 93% unknown-artist figure concerns the selected style download pilot, not all styles in WikiArt. That pilot required 4.2 GB, compared with 2.3 GB for the genre subset.
- Museum A contains 2,400 images from 58 named artists plus all unknown-artist images. Its stratified random image split gives 1,920 training images and 480 validation images. Named artists may appear in both training and validation within A.
- Museum B contains 600 images from 44 named artists absent from A. It contains no Unknown Artist images. There are 384, 96 and 120 images per genre in training, validation and test respectively.
- Hyperparameters, early stopping and model selection use validation A. Each completed model selected for final evaluation is scored once on B. ResNet-50 was rejected before that evaluation and has no test score.

### Six-model performance comparison, slide 5

| Model | Validation A | Test B | CPU time per photo | Model size |
|---|---:|---:|---:|---:|
| Dense network, 32 px | 39.0% | 41.8% | 0.1 ms | 1.6 MB |
| Classical random forest, colour + edges | 60.0% | 58.5% | 43 ms | 22 MB |
| Small CNN, 64 px | 60.8% | 63.7% | 1.7 ms | 0.1 MB |
| Frozen ResNet-18, 224 px | 80.3% | 84.5% | 70 ms | 45 MB |
| **Fine-tuned ResNet-18, final model** | **83.3%** | **86.0%** | **70 ms** | **45 MB** |
| Frozen ResNet-50, 224 px | 81.9% | **Not tested** | 139 ms | 94 MB |

The small CNN uses **60.8% validation**, as in the latest slide. ADR 0012 lists 58.8%, but the `cnn_longer` CSV row pairs 60.83% validation with 63.67% test. The chart keeps the validation and test values from that same run. Dense and frozen ResNet validation values use the reported multi-seed summaries.

The forest is **16.7 percentage points** above the dense network on B, or almost 17 points. The final ResNet-18 is **27.5 points** above the forest. Pretraining and resolution explain the main gain; the label “neural network” alone does not.

ResNet-50 gained **1.6 points over frozen ResNet-18 on validation**, below the predefined +2-point threshold. It was roughly twice as slow and twice as large. It has **no Museum B measurement**. Never describe its missing bar as 0%, infer a score, or present a test result for it.

### Generalisation and uncertainty, slide 6

Four of the five tested models score higher on B than on validation. The **classical baseline is the exception: 60.0% validation, 58.5% test**. Do not say that all models improve on B.

The split establishes zero overlap of identified artists between A and B. Stable or higher test performance supports transfer to unseen painters, but it does not prove that every shortcut has disappeared. Museum B may be easier, has only named artists, and uses clean cropped scans. Real phone photos can introduce glare and perspective that this test does not measure.

The final model’s 86% accuracy on 600 test images has an approximate **95% margin of ±3 percentage points**. Lower-accuracy baselines have margins closer to ±4 points. Do not claim a demonstrated statistical improvement from the CNN–forest gap or the 1.5-point test gap between frozen and fine-tuned ResNet-18. Fine-tuning was chosen using its **+3.0-point validation gain** at the same inference cost.

### Search results and live latency, slides 7–10

- SigLIP scores **0.788 nDCG@5**, approximately 0.79, on Museum B. This measures ranking quality, not “79% correct answers”.
- The search test contains 40 queries. Relevance labels come from an AI judge, checked against a human on 150 pairs, with weighted kappa 0.65.
- The hybrid scores about 0.797. Its gain over SigLIP is about +0.010 before rounding, with p = 0.78. The experiment does not establish a useful improvement, so SigLIP alone was retained.
- The app displays **six artworks**, while the evaluation measures the **top five**.
- Images are encoded once in advance. A visitor’s query is encoded at search time, then compared with the stored image vectors using a dot product.
- **70 ms per classification** and **82 ms per search query** are reference CPU measurements. The live app displays its own response time, which may differ.
- “No paid API call per request” does not mean hosting is always free. Both inference functions run on the server’s CPU.

## If the demo runs late

Use only the first request in **2 · Ask**, then move to **3 · Info** and finish at 10:00. If the application does not respond, return to slides 9 and 10 and explain the same actions using the screenshots, explicitly saying that they are screenshots. Avoid spending the demonstration waiting for the server to start.

## Project references

- [Project overview and demo instructions](../../README.md)
- [Data exploration notebook P01](../../src/deep_learning_project_waerniers_lamy_mahieddine/vision/P01_explore_data.ipynb)
- [ADR 0002: dataset and classification target](../../adr/0002-classification-target-and-dataset-subset.md)
- [ADR 0003: Museum A / Museum B protocol](../../adr/0003-simulated-museum-b-test-set.md)
- [ADR 0012: final vision model](../../adr/0012-final-model.md)
- [ADR 0014: classical baseline](../../adr/0014-classical-baseline.md)
- [Vision experiment log](../../results/experiments.csv) and [classical baseline results](../../results/classical_baseline.csv)
- [Search evaluation protocol](../../adr/0013-nlp-evaluation-protocol.md) and [NLP summary](../../adr/9998-nlp-executive-summary.md)
- [Demo application](../../demo.py) and [bundle preparation script](../../scripts/build_app.py)
