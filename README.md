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
Decisions and their evidence are in [adr/](adr/README.md): start there to see why each choice was made.

```
.
├── README.md
├── pyproject.toml, uv.lock, .python-version  # dependencies and Python 3.11, managed with uv
├── .gitignore                            # keeps the images (licence) and most model weights out of Git
├── adr/                                  # architecture decision records: why we made each choice;
│                                         # 9999-vision-executive-summary.md is the one-page story of the vision work
├── data/                                 # downloaded images, metadata.csv and split.csv (ignored by Git, never committed)
├── results/                              # one dated pickle per run + experiments.csv (settings and scores of every run);
│                                         # final_model.pt is the trained final model (the other weights are not in Git)
└── src/deep_learning_project_waerniers_lamy/
    ├── vision/                           # computer vision: the genre classifier
    │   ├── make_wikiart_subset.py        # downloads the 5-genre subset (course script, unchanged)
    │   ├── utils.py                      # the functions every notebook shares: split, data, training, results, models, test scoring
    │   ├── P01_explore_data.ipynb        # what the downloaded data is (ADR 0002, 0003)
    │   ├── P02_dense_baseline.ipynb      # the dense network that every other model is compared with (ADR 0006)
    │   ├── P03_first_cnn.ipynb           # our first CNN, described by a config and run with run_experiment (ADR 0007)
    │   ├── P04_cnn_longer_batchnorm.ipynb  # the first CNN trained longer, and with batch norm (ADR 0007, 0008)
    │   ├── P05_cnn_deeper.ipynb          # the CNN with a fourth convolution block (ADR 0008)
    │   ├── P06_resnet_transfer.ipynb     # pretrained ResNet-18, frozen and with its last block trained, 3 seeds (ADR 0009, 0012)
    │   ├── P07_resnet_resolution_control.ipynb  # the ResNet on small images: pretraining or resolution? (ADR 0010, 0011)
    │   ├── P08_resnet_improvements.ipynb  # cheap ResNet improvements, none clearly helped (ADR 0011)
    │   ├── P09_comparison.ipynb          # all models side by side: accuracy, and cost per image on a CPU
    │   ├── P10_error_analysis.ipynb      # which genres the final model mixes up, and why (validation only)
    │   └── run_final.py                  # the final model: reproduces its test score on museum B (ADR 0012)
    └── nlp/                              # natural-language part, not built (ADR 0004)
```

Conventions: all the tryouts are notebooks, to be read in order (`vision/P01` to `P10`). They share one pipeline so that results can
be compared: `utils.py` prepares the data (`load_data`), trains (`fit`, or `run_experiment` for a model described by a config) and
records each run in `results/` (`save_result`): a dated pickle, and one row of `results/experiments.csv` with the settings and the
validation scores. The test columns of that row stay empty on purpose: only the models we would really deliver are scored, once, on
museum B (`score_on_test`, which refuses a second test, ADR 0003). Run a notebook with the project's `.venv` as kernel.

Set up the environment and build the data (from the repository root; the download is about 2.3 GB):

```
uv sync
uv run --with "datasets>=3.0" python src/deep_learning_project_waerniers_lamy/vision/make_wikiart_subset.py --label genre --classes portrait landscape cityscape still_life religious_painting --per-class 600 --size 224 --out data/wikiart_genre5
uv run python -c "from deep_learning_project_waerniers_lamy.vision.utils import make_split; make_split()"
```

The download script needs the `datasets` package, which is not a dependency of the project (hence `--with`). The images are never
committed (WikiArt licence). Then open the notebooks of `vision/` in order, starting with `P01_explore_data.ipynb`, or go straight to the
final model below.

## FINAL MODEL
The final model is a ResNet-18 pretrained on ImageNet, with its last block fine-tuned on our paintings, working on 224 px images
([ADR 0012](adr/0012-final-model.md)). It scores **86.0% on the test set** (museum B: 600 paintings by artists never seen in
training) and 83.3% on validation, and takes about 70 ms per photo on one CPU thread.

To see its test score, once the data is built (above), from the repository root:

```
uv sync
uv run python src/deep_learning_project_waerniers_lamy/vision/run_final.py
```

It takes about a minute. It loads the trained weights from `results/final_model.pt` and prints the accuracy, the loss, the accuracy per
genre and the confusion matrix, next to the score recorded when the model was first scored. Add `--train` to train the model again first
(about 15 minutes). The first run needs an internet connection to download the ImageNet weights of ResNet-18 (45 MB).
