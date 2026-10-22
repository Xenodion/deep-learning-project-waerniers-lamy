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
├── data/                                 # downloaded images and split.csv (ignored by Git, never committed)
├── results/                              # one dated pickle per run + experiments.csv (settings and scores of every run)
└── src/deep_learning_project_waerniers_lamy/
    ├── vision/                           # computer vision: the genre classifier
    │   ├── make_wikiart_subset.py        # downloads the 5-genre subset (course script, unchanged)
    │   ├── utils.py                      # functions shared by all notebooks (make_split, load_data, fit, run_experiment, ...)
    │   ├── P01_explore_data.ipynb        # what the downloaded data is (figures quoted in the ADRs)
    │   ├── P02_dense_baseline.ipynb      # the dense network that every other model is compared with
    │   ├── P03_first_cnn.ipynb           # our first CNN, described by a config and run with run_experiment
    │   ├── P04_cnn_longer_batchnorm.ipynb  # the first CNN trained longer and with batch norm (ADR 0007)
    │   ├── P05_cnn_deeper.ipynb          # the CNN with a fourth convolution block (ADR 0008)
    │   ├── P06_resnet_transfer.ipynb     # pretrained ResNet-18, transfer learning (ADR 0009)
    │   ├── P07_resnet_resolution_control.ipynb  # the ResNet on small images: pretraining or resolution?
    │   ├── P08_resnet_improvements.ipynb  # cheap ResNet improvements: several layers, augmentation, ResNet-50 (ADR 0011)
    │   ├── P09_comparison.ipynb          # all models side by side: accuracy, and cost per image on a CPU
    │   └── run_final.py                  # (planned) loads the chosen model and scores it once on museum B
    └── nlp/                              # natural-language part, deferred (ADR 0004)
```

Conventions: all the tryouts are notebooks (`vision/P0x_*.ipynb`). They share one pipeline so that results can be
compared: `utils.py` prepares the data (`load_data`), trains (`fit`) and records each experiment in `results/`
(`save_result`). Every run is a row of `results/experiments.csv` (settings and validation scores). Its test columns
stay empty until the model is final: `add_test_score` then fills them, once per run (ADR 0003), and refuses a second
test. When a model is chosen, `run_final.py` reproduces its test score for the grader. Run a notebook with the project's
`.venv` as kernel.

Reproduce the data (from the repository root):

```
python src/deep_learning_project_waerniers_lamy/vision/make_wikiart_subset.py --label genre --classes portrait landscape cityscape still_life religious_painting --per-class 600 --size 224 --out data/wikiart_genre5
python -c "from deep_learning_project_waerniers_lamy.vision.utils import make_split; make_split()"
```

Then open `P01_explore_data.ipynb` to look at the data, and `P02_dense_baseline.ipynb` to train the baseline.
