# FINAL PROJECT GUIDELINES

You are a member of the AI team of a (fictional) mid-sized French fine-arts museum (the Museum below). The
Museum wants an intelligent guide, available on visitors' phones and on kiosks in the galleries, that can do two
things

1.  Recognise what the visitor is looking at. The visitor points the camera at a painting; the guide recognises its
style (Impressionism, Baroque, Cubism…) or its genre (portrait, landscape, still life…) and tells the story of
that movement or tradition.

2.  Answer questions in natural language. A visitor types "Which painting shows a stormy sea?" or "I would like
to see portraits of children"; the guide proposes the artworks of the collection that best match the question.

The director also expects a clear recommendation: the kiosks run on modest CPUs, the team is small and the
budget is limited. Is a neural network worth it in production, or would a simpler classical model do the job?
Your mission is to build, diagnose and defend the neural-network components of this guide, in the spirit of the
course: networks built from PyTorch primitives rather than library magic, every number measured, every claim
backed by evidence. The project is mutualised: the scenario and the assessment format (Git deliverable, group oral
defence, individual deliberation, peer review) are shared at programme level

## DATA
Can be loaded from the attached Python File

## DELIVERABLE
- A feed-forward network in PyTorch for artwork classification
- Training monitoring and failure diagnosis
- A pretrained CNN (ResNet) for image classification
- Deadline is for the 30th of October 2026

The Source Code must be delivered with Clear Git History on Github or Gitlab.

You will also need to prepare a 10 minutes PowerPoint Presentation followed by a 5 Minutes Q&A.

## REPOSITORY LAYOUT
Files marked *(planned)* are not written yet. Decisions and their evidence are in [adr/](adr/README.md).

```
.
├── README.md
├── pyproject.toml, uv.lock               # dependencies, managed with uv
├── adr/                                  # architecture decision records: why we made each choice
├── lectures/                             # course slides and notes
├── notebooks/                            # course practicals TP01 to TP08, and our project notebooks (P01, ...)
│   ├── P01_explore_data.ipynb            # what the downloaded data is (figures quoted in the ADRs)
│   └── P02_monitor_and_diagnose.ipynb    # (planned) learning curves, confusion matrix, failure diagnosis
├── data/                                 # downloaded images and split.csv (ignored by Git, never committed)
└── src/deep_learning_project_waerniers_lamy/
    ├── vision/                           # computer vision: the genre classifier
    │   ├── make_wikiart_subset.py        # downloads the 5-genre subset (course script, unchanged)
    │   ├── make_split.py                 # museum A (train, validation) and museum B (test) -> split.csv
    │   ├── dataset.py                    # (planned) reads split.csv; the transform is a parameter
    │   ├── dense_model.py                # (planned) the feed-forward network
    │   ├── train_loop.py                 # (planned) one epoch of training, evaluation, best-model checkpoint
    │   ├── train_dense.py                # (planned) entry point for the dense baseline
    │   ├── resnet_model.py               # (planned) pretrained ResNet with a new 5-class head
    │   ├── train_resnet.py               # (planned) entry point for transfer learning
    │   ├── classical_baseline.py         # (planned) the non-neural comparison for the director
    │   └── evaluate_test.py              # (planned) scores a finished model on museum B
    └── nlp/                              # natural-language part, deferred (ADR 0004)
```

Conventions: exploration, plots and diagnosis are in notebooks; the `.py` files are things we run or import, one job
per file, kept short. Files that only define things (`dataset.py`, `*_model.py`, `train_loop.py`) are imported by the
short entry points (`train_*.py`, `evaluate_test.py`) that we run.

Reproduce the data (from the repository root):

```
python src/deep_learning_project_waerniers_lamy/vision/make_wikiart_subset.py --label genre --classes portrait landscape cityscape still_life religious_painting --per-class 600 --size 224 --out data/wikiart_genre5
python src/deep_learning_project_waerniers_lamy/vision/make_split.py
```

Then open `notebooks/P01_explore_data.ipynb` to look at the data.
