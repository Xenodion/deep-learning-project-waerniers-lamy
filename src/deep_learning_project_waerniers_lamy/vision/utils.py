import copy
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "wikiart_genre5"
RESULTS_DIR = REPO_ROOT / "results"


# --------------------------------------------------------------------------------------------------
# 1. Split: museum A (train + validation) and museum B (test)
# --------------------------------------------------------------------------------------------------
def make_split(data_dir=DATA_DIR, b_per_genre=120, val_share=0.2, seed=0):
    """Split the images into museum A (train + validation) and museum B (test), as decided in adr/0003.

    Reads metadata.csv and writes split.csv in data_dir. Run it once, after downloading the images:
        python -c "from deep_learning_project_waerniers_lamy.vision.utils import make_split; make_split()"
    """
    meta = pd.read_csv(data_dir / "metadata.csv")
    genres = sorted(meta["label"].unique())

    # 1. Museum B: pick whole named artists so that every genre gets exactly b_per_genre images.
    # "Unknown Artist" cannot be grouped, so it never goes to museum B.
    named = meta[meta["artist"] != "Unknown Artist"]
    per_artist = pd.crosstab(named["artist"], named["label"])  # images of each artist in each genre
    artists = sorted(per_artist.index)

    def gap(chosen):
        """How far the chosen artists are from b_per_genre images in every genre (0 = perfect)."""
        images_per_genre = per_artist.loc[list(chosen)].sum()
        return (images_per_genre - b_per_genre).abs().sum()

    # Hill climbing: add or remove a random artist, keep the change if the gap does not get worse.
    rng = random.Random(seed)
    museum_b = set(rng.sample(artists, 12))
    best = gap(museum_b)
    for _ in range(20000):
        if best == 0:
            break
        trial = museum_b ^ {rng.choice(artists)}  # ^ adds the artist if absent, removes it if present
        if rng.random() < 0.3:
            trial = trial ^ {rng.choice(artists)}
        if gap(trial) <= best:
            museum_b, best = trial, gap(trial)
    if best != 0:
        raise RuntimeError("Could not find artists giving the same number of images in every genre.")

    # 2. Everything else is museum A. Cut it at random into train and validation, genre by genre.
    meta["split"] = "train"
    meta.loc[meta["artist"].isin(museum_b), "split"] = "test"
    rng = random.Random(seed)  # restart the generator so the cut does not depend on the search above
    for genre in genres:
        ids = sorted(meta.loc[(meta["label"] == genre) & (meta["split"] == "train"), "id"])
        rng.shuffle(ids)
        meta.loc[meta["id"].isin(ids[: round(len(ids) * val_share)]), "split"] = "val"

    # 3. Save and check.
    meta[["id", "path", "label", "label_idx", "artist", "split"]].to_csv(data_dir / "split.csv", index=False)

    print(pd.crosstab(meta["split"], meta["label"], margins=True), "\n")
    for name in ["train", "val", "test"]:
        part = meta[meta["split"] == name]
        known = part[part["artist"] != "Unknown Artist"]
        print(f"{name:5s}: {len(part)} images, {len(known)} with a named artist ({known['artist'].nunique()} artists), "
              f"{len(part) - len(known)} Unknown Artist")

    named_artists = meta[meta["artist"] != "Unknown Artist"]
    test_artists = set(named_artists.loc[named_artists["split"] == "test", "artist"])
    other_artists = set(named_artists.loc[named_artists["split"] != "test", "artist"])
    assert not test_artists & other_artists, "a named artist is in the test set and in train/validation"
    print("\nnamed artists shared between test and train/validation: 0")
    print("Wrote", data_dir / "split.csv")


# --------------------------------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------------------------------
def load_data(size=32, splits=("train", "val")):
    """Return a dict with X_<split> and y_<split> for each asked split, and the list of classes.

    Preparation, step by step:
      1. read split.csv (made by make_split) to know which image is in train, val or test;
      2. shrink every image to size x size and scale the pixels to 0-1;
      3. standardise each colour channel with the mean and std of the TRAIN images only;
      4. flatten each image into one row of 3 * size * size numbers (for a dense network).

    The test split (museum B) is only loaded if you ask for it. Use it once, at the end (adr/0003).
    """
    for name in splits:
        if name not in ("train", "val", "test"):
            raise ValueError(f"unknown split {name!r}")

    table = pd.read_csv(DATA_DIR / "split.csv")
    classes = sorted(table["label"].unique())

    pixels = {}
    for name in set(splits) | {"train"}:  # train is always needed for the mean and std
        paths = table.loc[table["split"] == name, "path"]
        images = [Image.open(DATA_DIR / p).convert("RGB").resize((size, size), Image.Resampling.BILINEAR) for p in paths]
        pixels[name] = np.stack([np.asarray(im) for im in images]).astype("float32") / 255  # (n, size, size, 3)

    mean = pixels["train"].mean(axis=(0, 1, 2))  # one number per colour channel
    std = pixels["train"].std(axis=(0, 1, 2))

    data = {"classes": classes, "mean": mean, "std": std}
    for name in splits:
        x = (pixels[name] - mean) / std
        data["X_" + name] = torch.tensor(x.reshape(len(x), -1))
        data["y_" + name] = torch.tensor(table.loc[table["split"] == name, "label_idx"].to_numpy())
    return data


# --------------------------------------------------------------------------------------------------
# 3. Training
# --------------------------------------------------------------------------------------------------
def evaluate(model, X, y, batch_size=256):
    """Return (loss, accuracy) of the model on X, y."""
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():  # no gradients needed to just measure
        for i in range(0, len(X), batch_size):
            out = model(X[i:i + batch_size])
            total_loss += loss_fn(out, y[i:i + batch_size]).item() * len(out)
            correct += (out.argmax(dim=1) == y[i:i + batch_size]).sum().item()
    return total_loss / len(X), correct / len(X)


def fit(model, data, epochs=30, lr=1e-3, batch_size=64, weight_decay=0.0, seed=0, verbose=True):
    """Train the model on data["X_train"] and keep the weights of the epoch with the best validation accuracy.

    Only train and validation are used. The model is left with its best weights.
    Returns a dict: history (the curves), best_epoch (counted from 1), best_val_acc, best_weights, seconds.
    """
    torch.manual_seed(seed)  # fixes the shuffling; seed the model creation yourself before building it
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_train, y_train = data["X_train"], data["y_train"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_weights = -1.0, 0, None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_train))  # shuffle the data at every epoch
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # end of the epoch: measure on the whole train set and on the validation set
        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, data["X_val"], data["y_val"])
        for key, value in zip(history, [train_loss, train_acc, val_loss, val_acc]):
            history[key].append(value)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_weights = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}")

    model.load_state_dict(best_weights)
    return {"history": history, "best_epoch": best_epoch, "best_val_acc": best_val_acc,
            "best_weights": best_weights, "seconds": time.time() - start}


# --------------------------------------------------------------------------------------------------
# 4. Results
# --------------------------------------------------------------------------------------------------
# Each run is saved in two places:
#   - results/<name>_<date>_<time>.pkl: the settings, the learning curves and the validation scores (small, committed
#     to Git), with the weights next to it in a .pt file (bigger, ignored by Git). Never test scores. The date and
#     time in the name mean that a run never overwrites an earlier one.
#   - results/experiments.csv: one row per run (date, settings, scores), to see at a glance which model did well.
#     The test columns (test_acc, test_loss, test_date) stay empty until add_test_score() fills them for a run:
#     the test set is used once per model (adr/0003), so most rows have no test score.
# Only load pickles that we made ourselves: loading a pickle from someone else can run code.
def save_result(name, result, weights=None):
    """Save one run (pickle, weights and one row in experiments.csv). Returns the file name without extension."""
    RESULTS_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    stem = f"{name}_{now:%Y-%m-%d_%H-%M-%S}"
    if (RESULTS_DIR / f"{stem}.pkl").exists():  # only if two runs of the same name end within the same second
        raise FileExistsError(f"results/{stem}.pkl already exists: wait a second and save again")
    with open(RESULTS_DIR / f"{stem}.pkl", "wb") as f:
        pickle.dump(result, f)
    if weights is not None:
        torch.save(weights, RESULTS_DIR / f"{stem}.pt")

    # One row for the table of all runs: the settings (CONFIG) in the middle, the scores at the end.
    seeds = result.get("val_acc_by_seed")
    row = {
        "date": f"{now:%Y-%m-%d %H:%M:%S}",
        "name": name,
        **result["config"],
        "n_params": result.get("n_params"),
        "best_epoch": result["best_epoch"],
        "best_val_acc": round(result["best_val_acc"], 4),
        "val_acc_mean": round(float(np.mean(seeds)), 4) if seeds else None,  # over several seeds, if we ran them
        "val_acc_std": round(float(np.std(seeds, ddof=1)), 4) if seeds else None,
        "seconds": round(result["seconds"]),
        "file": stem,
    }
    table = pd.DataFrame([row])
    if (RESULTS_DIR / "experiments.csv").exists():
        table = pd.concat([_read_table(), table], ignore_index=True)  # models with other settings add columns
    _write_table(table)

    print(f"saved results/{stem}.pkl and added a row to results/experiments.csv")
    return stem


def load_result(stem):
    """Read results/<stem>.pkl. The stem is the 'file' column of experiments.csv."""
    with open(RESULTS_DIR / f"{stem}.pkl", "rb") as f:
        return pickle.load(f)


def add_test_score(stem, model, data):
    """Score a saved run on the test set (museum B) and write test_acc, test_loss and test_date in its row.

    Use it when the model is final: the test set is used once per model (adr/0003), so this refuses to score a run
    that already has a test score. It loads the weights saved with the run, so the model tested is exactly the one
    in the table. `model` must have the same architecture; `data` must contain the test split:
        data = load_data(size, splits=("train", "val", "test"))
    Returns (loss, accuracy).
    """
    if "X_test" not in data:
        raise ValueError('the test split is not loaded: use load_data(size, splits=("train", "val", "test"))')
    table = _read_table()
    rows = table.index[table["file"] == stem]
    if len(rows) != 1:
        raise ValueError(f"no run called {stem!r} in experiments.csv")
    row = rows[0]
    if pd.notna(table.loc[row, "test_acc"]):
        raise RuntimeError(f"{stem} already has a test score: the test set is used once per model (adr/0003)")

    model.load_state_dict(torch.load(RESULTS_DIR / f"{stem}.pt"))
    loss, acc = evaluate(model, data["X_test"], data["y_test"])
    table.loc[row, "test_acc"] = round(acc, 4)
    table.loc[row, "test_loss"] = round(loss, 4)
    table.loc[row, "test_date"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    _write_table(table)
    print(f"{stem}: test accuracy {acc:.1%}, test loss {loss:.3f} (written in results/experiments.csv)")
    return loss, acc


def summary(sort_by="best_val_acc"):
    """The table of all runs (results/experiments.csv), best first."""
    return _read_table().sort_values(sort_by, ascending=False)


# Helpers for experiments.csv: date and name first, then the settings, then the scores (empty when not measured).
FIRST_COLUMNS = ["date", "name"]
SCORE_COLUMNS = ["n_params", "best_epoch", "best_val_acc", "val_acc_mean", "val_acc_std",
                 "test_acc", "test_loss", "test_date", "seconds", "file"]


def _read_table():
    return pd.read_csv(RESULTS_DIR / "experiments.csv", dtype={"test_date": str})


def _write_table(table):
    for column in SCORE_COLUMNS:
        if column not in table.columns:
            table[column] = None  # an empty column, filled later if we ever measure it
    settings = [c for c in table.columns if c not in FIRST_COLUMNS + SCORE_COLUMNS]
    table[FIRST_COLUMNS + settings + SCORE_COLUMNS].to_csv(RESULTS_DIR / "experiments.csv", index=False)
