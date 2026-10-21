"""Everything the notebooks share, in one file: split, data, training loop and results.

    from deep_learning_project_waerniers_lamy.vision.utils import load_data, fit, save_result, summary

Using the same functions in every notebook keeps the experiments comparable.
"""
import copy
import pickle
import random
import time
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


def fit(model, data, epochs=30, lr=1e-3, batch_size=64, weight_decay=0.0, seed=0):
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

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}")

    model.load_state_dict(best_weights)
    return {"history": history, "best_epoch": best_epoch, "best_val_acc": best_val_acc,
            "best_weights": best_weights, "seconds": time.time() - start}


# --------------------------------------------------------------------------------------------------
# 4. Results
# --------------------------------------------------------------------------------------------------
# A pickle holds the settings, the learning curves and the validation scores (small, committed to Git).
# The weights go next to it in a .pt file (bigger, ignored by Git). Never test scores: the test set is used once.
# Only load pickles that we made ourselves: loading a pickle from someone else can run code.
def save_result(name, result, weights=None):
    """Write results/<name>.pkl (and results/<name>.pt if weights are given)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(result, f)
    if weights is not None:
        torch.save(weights, RESULTS_DIR / f"{name}.pt")


def load_result(name):
    """Read results/<name>.pkl."""
    with open(RESULTS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def summary():
    """One row per saved experiment: its settings and its best validation score."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.pkl")):
        result = load_result(path.stem)
        rows.append({"name": path.stem, **result["config"], "best_epoch": result["best_epoch"],
                     "best_val_acc": result["best_val_acc"], "seconds": round(result["seconds"])})
    return pd.DataFrame(rows)
