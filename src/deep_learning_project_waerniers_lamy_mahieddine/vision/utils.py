import copy
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
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
        python -c "from deep_learning_project_waerniers_lamy_mahieddine.vision.utils import make_split; make_split()"
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


def fit(model, data, epochs=30, lr=1e-3, batch_size=64, weight_decay=0.0, seed=0, verbose=True,
        clip_grad_norm=None, monitor=False, optimizer="adam", momentum=0.0):
    """Train the model on data["X_train"] and keep the weights of the epoch with the best validation accuracy.

    Only train and validation are used. The model is left with its best weights.
    Returns a dict: history (the curves), best_epoch (counted from 1), best_val_acc, best_weights, seconds.

    optimizer: "adam" (the default of every run) or "sgd" (plain gradient descent, with momentum if momentum > 0).
    clip_grad_norm: if set, the gradient of all the weights together is scaled down to this norm before each step.
    monitor: also record, at every epoch, what the training doctor looks at (P12). It does not change the training:
        history gets batch_loss (the loss of every batch), and per layer (see monitored_layers) grad_norm, update_ratio,
        act_std, dead and saturated. See record_batch and activation_stats for what each one measures.
    """
    torch.manual_seed(seed)  # fixes the shuffling; seed the model creation yourself before building it
    loss_fn = nn.CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}: use 'adam' or 'sgd'")
    X_train, y_train = data["X_train"], data["y_train"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    if monitor:
        weights, activations = monitored_layers(model)
        history["batch_loss"] = []
        for key in ["grad_norm", "update_ratio"]:
            history[key] = {name: [] for name in weights}
        for key in ["act_std", "dead", "saturated"]:
            history[key] = {name: [] for name in activations}
    best_val_acc, best_epoch, best_weights = -1.0, 0, None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        if monitor:
            sums = {name: [0.0, 0.0] for name in weights}  # grad norm and update ratio, summed over the batches
        order = torch.randperm(len(X_train))  # shuffle the data at every epoch
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            if monitor:
                before = record_batch(weights, sums)  # measured before clipping, so an explosion stays visible
                history["batch_loss"].append(loss.item())
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            if monitor:
                for name, layer in weights.items():
                    sums[name][1] += ((layer.weight.detach() - before[name]).norm() / (before[name].norm() + 1e-12)).item()

        # end of the epoch: measure on the whole train set and on the validation set
        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, data["X_val"], data["y_val"])
        for key, value in zip(["train_loss", "train_acc", "val_loss", "val_acc"], [train_loss, train_acc, val_loss, val_acc]):
            history[key].append(value)
        if monitor:
            n_batches = -(-len(order) // batch_size)
            for name in weights:
                history["grad_norm"][name].append(sums[name][0] / n_batches)
                history["update_ratio"][name].append(sums[name][1] / n_batches)
            for name, stats in activation_stats(activations, model, X_train).items():
                for key, value in stats.items():
                    history[key][name].append(value)

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
# 3b. Monitoring: what the training doctor looks at (fit(..., monitor=True), P12)
# --------------------------------------------------------------------------------------------------
ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


def monitored_layers(model):
    """The layers fit(monitor=True) watches, by name: those with a weight matrix (Linear, Conv2d), and the activations."""
    weights = {name: m for name, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}
    activations = {name: m for name, m in model.named_modules() if isinstance(m, ACTIVATIONS)}
    return weights, activations


def record_batch(weights, sums):
    """After backward: add the norm of each layer's weight gradient to sums[name][0], and return a copy of the weights.

    The copy lets fit measure the update ratio after the step: norm of (new weights - old weights) / norm of old weights.
    A healthy ratio is about 1e-3 per step: far below, the layer barely learns; near 1, each step rewrites it.
    """
    before = {}
    for name, layer in weights.items():
        grad = layer.weight.grad
        sums[name][0] += grad.norm().item() if grad is not None else 0.0
        before[name] = layer.weight.detach().clone()
    return before


def activation_stats(activations, model, X, batch_size=256):
    """Pass X through the model (evaluation mode) and describe the output of each activation layer.

    For each layer, by name: act_std, the spread of its outputs (it collapses towards 0 when the signal dies out with depth);
    dead, the share of units (neurons, or channels for a convolution) that never give a positive output on any image of X:
    a dead ReLU gives 0 and its own weights get no gradient, so it comes back only if the layers below change its inputs
    (a leaky ReLU is never fully dead: it still passes 1% of the gradient); saturated, the share of outputs stuck at the flat ends of
    a sigmoid (below 0.02 or above 0.98) or a tanh (beyond +/- 0.96), where the gradient almost vanishes (NaN for ReLUs).
    """
    totals = {name: {"sum": 0.0, "sum_sq": 0.0, "count": 0, "saturated": 0, "alive": None} for name in activations}

    def hook_for(name, layer):
        def hook(module, inputs, out):
            t = totals[name]
            t["sum"] += out.sum().item()
            t["sum_sq"] += (out ** 2).sum().item()
            t["count"] += out.numel()
            if isinstance(layer, nn.Sigmoid):
                t["saturated"] += ((out < 0.02) | (out > 0.98)).sum().item()
            elif isinstance(layer, nn.Tanh):
                t["saturated"] += (out.abs() > 0.96).sum().item()
            per_unit = out.transpose(0, 1).reshape(out.shape[1], -1)  # one row per neuron or channel
            alive = (per_unit > 0).any(dim=1)
            t["alive"] = alive if t["alive"] is None else t["alive"] | alive
        return hook

    handles = [layer.register_forward_hook(hook_for(name, layer)) for name, layer in activations.items()]
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                model(X[i:i + batch_size])
    finally:
        for handle in handles:
            handle.remove()

    stats = {}
    for name, layer in activations.items():
        t = totals[name]
        mean = t["sum"] / t["count"]
        stats[name] = {
            "act_std": max(t["sum_sq"] / t["count"] - mean ** 2, 0.0) ** 0.5,
            "dead": 1 - t["alive"].float().mean().item(),
            "saturated": t["saturated"] / t["count"] if isinstance(layer, (nn.Sigmoid, nn.Tanh)) else float("nan"),
        }
    return stats


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


# --------------------------------------------------------------------------------------------------
# 5. Experiments: a model described by a config, trained and recorded in one call
# --------------------------------------------------------------------------------------------------
def build_cnn(config, n_classes=5):
    """A small CNN described by the config.

    Needs: conv_channels, one entry per convolution block (for example [16, 32, 64]).
    Optional: kernel_size (3), batch_norm (False), dropout (0.0).
    Each block is a convolution, (batch norm), ReLU, then a max pooling that halves the image. After the last block,
    each feature map is averaged into one number, so the final layer only has conv_channels[-1] inputs.
    """
    kernel = config.get("kernel_size", 3)
    layers = []
    in_channels = 3
    for out_channels in config["conv_channels"]:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2))
        if config.get("batch_norm", False):
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [nn.ReLU(), nn.MaxPool2d(2)]
        in_channels = out_channels
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(config.get("dropout", 0.0)),
               nn.Linear(in_channels, n_classes)]
    return nn.Sequential(*layers)


def build_mlp(config, n_classes=5):
    """A dense network of any depth, for the training doctor (P12).

    Needs: image_size, hidden (one width per hidden layer, for example [128] * 6).
    Optional: activation ("relu", "leaky_relu", "sigmoid" or "tanh"; "relu"), init ("default", PyTorch's own; "he",
    Kaiming normal, made for ReLUs; or a number, the std of a normal distribution), dropout (0.0, after each activation).
    The biases start at 0 unless init is "default".
    """
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}[config.get("activation", "relu")]
    sizes = [3 * config["image_size"] ** 2] + list(config["hidden"])
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), activation()]
        if config.get("dropout", 0.0) > 0:
            layers.append(nn.Dropout(config["dropout"]))
    layers.append(nn.Linear(sizes[-1], n_classes))

    init = config.get("init", "default")
    if init != "default":
        for layer in layers:
            if isinstance(layer, nn.Linear):
                if init == "he":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                else:
                    nn.init.normal_(layer.weight, std=float(init))
                nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


def to_images(data, size):
    """load_data gives flat rows (height, width, channels); a CNN wants (channels, height, width). Returns a copy."""
    converted = dict(data)
    for key in data:
        if key.startswith("X_"):
            converted[key] = data[key].view(len(data[key]), size, size, 3).permute(0, 3, 1, 2).contiguous()
    return converted


def run_experiment(name, config, seeds=3, build_model=build_cnn, load_fn=None):
    """Build the model described by config, train it, measure it on validation and record the run.

    Steps: load the data, build the model, train (best epoch kept on validation), repeat with other seeds,
    confusion matrix on validation, save_result. The test set is not used.
    Needs in config: image_size, epochs, lr, batch_size, weight_decay, seed (and what build_model needs).
    Returns a dict: model (with its best weights), result (history, best epoch...), confusion, val_acc_by_seed, stem.
    """
    if load_fn is None:  # the default: small images for our own CNNs
        size = config["image_size"]
        data = to_images(load_data(size=size), size)
    else:  # another way to prepare the data, for example load_resnet_data
        data = load_fn(config)
    classes = data["classes"]

    def train(seed, verbose):
        torch.manual_seed(seed)  # the same seed gives the same starting weights
        model = build_model(config, len(classes))
        if verbose:
            loss0, acc0 = evaluate(model, data["X_train"], data["y_train"])
            print(f"{sum(p.numel() for p in model.parameters())} parameters | before training: loss {loss0:.3f} "
                  f"(ln {len(classes)} = {np.log(len(classes)):.3f}), accuracy {acc0:.1%}")
        result = fit(model, data, epochs=config["epochs"], lr=config["lr"], batch_size=config["batch_size"],
                     weight_decay=config["weight_decay"], seed=seed, verbose=verbose)
        return model, result

    model, result = train(config["seed"], verbose=True)
    print(f"\nbest validation accuracy {result['best_val_acc']:.1%} at epoch {result['best_epoch']} "
          f"({result['seconds']:.0f} s)")

    val_acc_by_seed = [result["best_val_acc"]]
    for k, seed in enumerate(range(config["seed"] + 1, config["seed"] + seeds), start=1):  # the other seeds
        # announce it first: each seed takes as long as the first run and prints nothing while it trains
        print(f"seed {seed} ({k} of {seeds - 1} more): training, about {result['seconds']:.0f} s...", flush=True)
        _, other = train(seed, verbose=False)
        val_acc_by_seed.append(other["best_val_acc"])
        print(f"   best validation accuracy {other['best_val_acc']:.1%} at epoch {other['best_epoch']}")

    model.eval()
    with torch.no_grad():
        predicted = model(data["X_val"]).argmax(dim=1)
    confusion = pd.crosstab(pd.Series([classes[i] for i in data["y_val"]], name="true genre"),
                            pd.Series([classes[i] for i in predicted], name="predicted"))
    confusion = confusion.reindex(index=classes, columns=classes, fill_value=0)

    record = {"config": config, "n_params": sum(p.numel() for p in model.parameters()), "history": result["history"],
              "best_epoch": result["best_epoch"], "best_val_acc": result["best_val_acc"],
              "seconds": result["seconds"], "classes": classes, "val_confusion": confusion}
    if len(val_acc_by_seed) > 1:
        record["val_acc_by_seed"] = val_acc_by_seed
    stem = save_result(name, record, weights=result["best_weights"])
    return {"model": model, "result": result, "confusion": confusion, "val_acc_by_seed": val_acc_by_seed, "stem": stem,
            "data": data}


# --------------------------------------------------------------------------------------------------
# 6. Pretrained ResNet-18 (transfer learning)
# --------------------------------------------------------------------------------------------------
# The network was trained on ImageNet. Most of it stays frozen, so we pass the images through the frozen part once and keep
# the result: training then only touches the small trainable part (a new final layer, and optionally the last block).
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # what the pretrained network expects, not our own statistics
IMAGENET_STD = [0.229, 0.224, 0.225]


# Size of the vector that the frozen part gives for one image: the last block alone, or the three last blocks together ("multi").
FEATURE_SIZE = {("resnet18", "last"): 512, ("resnet18", "multi"): 128 + 256 + 512,
                ("resnet50", "last"): 2048, ("resnet50", "multi"): 512 + 1024 + 2048}


def pretrained_resnet(arch="resnet18"):
    """A ResNet with the ImageNet weights (downloaded the first time), in evaluation mode: "resnet18" or "resnet50"."""
    weights = {"resnet18": torchvision.models.ResNet18_Weights, "resnet50": torchvision.models.ResNet50_Weights}[arch]
    return getattr(torchvision.models, arch)(weights=weights.IMAGENET1K_V1).eval()


def load_resnet_data(config, splits=("train", "val"), batch_size=64):
    """Return a dict like load_data, with X_<split> = the images passed once through the frozen part of the ResNet.

    config["finetune"] decides where the frozen part stops:
      "head":   the whole network up to the average pooling: 512 numbers per image, only a new final layer is trained;
      "layer4": up to the end of layer3: 256 x 14 x 14 numbers per image, so the last block (layer4) can be trained too.
    The images are the files of the subset (224 x 224, or config["image_size"] if smaller), standardised with the ImageNet mean
    and std; the sizes quoted above are for 224 px. The test split is only loaded if you ask for it, as in load_data (adr/0003).

    Optional keys of config (the defaults give the P06 behaviour):
      model:     "resnet18" or "resnet50", the pretrained network;
      features:  "last" (the last block) or "multi" (average-pooled outputs of the three last blocks, joined);
      augment:   number of augmented copies (random crop, mirror, colour) added for each TRAIN image, drawn with config["seed"];
      flip_eval: also give X_<split>_flip, the features of the mirrored validation / test images, for flip averaging.
    """
    table = pd.read_csv(DATA_DIR / "split.csv")
    size = config.get("image_size", 224)  # the files are 224 x 224; a smaller size shrinks them exactly as load_data does
    resnet = pretrained_resnet(config.get("model", "resnet18"))
    stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
    if config.get("finetune", "head") == "layer4":
        extract = nn.Sequential(stem, resnet.layer2, resnet.layer3).eval()  # layer4 is trained afterwards
    elif config.get("features", "last") == "multi":
        def extract(x):
            x2 = resnet.layer2(stem(x))
            x3 = resnet.layer3(x2)
            x4 = resnet.layer4(x3)
            return torch.cat([x2.mean(dim=(2, 3)), x3.mean(dim=(2, 3)), x4.mean(dim=(2, 3))], dim=1)
    else:
        extract = nn.Sequential(stem, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool, nn.Flatten()).eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    augmentation = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    def features_of(paths, transform=None, flip=False):
        """The frozen part applied to the images in `paths`, batch by batch (optionally augmented, or mirrored)."""
        out = []
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(DATA_DIR / path) as im:
                    im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
                    if transform is not None:
                        im = transform(im)
                    images.append(np.asarray(im))
            batch = torch.tensor(np.stack(images)).permute(0, 3, 1, 2).float() / 255
            if flip:
                batch = batch.flip(3)  # mirror image: flip the width
            with torch.no_grad():  # the frozen part is never trained
                out.append(extract((batch - mean) / std))
        return torch.cat(out)

    data = {"classes": sorted(table["label"].unique())}
    for name in splits:
        rows = table[table["split"] == name]
        paths = list(rows["path"])
        labels = torch.tensor(rows["label_idx"].to_numpy())
        X, y = features_of(paths), labels
        if name == "train" and config.get("augment", 0) > 0:
            torch.manual_seed(config.get("seed", 0))  # the same augmented copies every time
            for _ in range(config["augment"]):
                X = torch.cat([X, features_of(paths, transform=augmentation)])
                y = torch.cat([y, labels])
        data["X_" + name], data["y_" + name] = X, y
        if config.get("flip_eval") and name != "train":
            data["X_" + name + "_flip"] = features_of(paths, flip=True)
    return data


def build_resnet_head(config, n_classes=5):
    """The trainable part: a new final layer, preceded by the pretrained last block if finetune is "layer4".

    Optional in config: finetune ("head" or "layer4"), dropout (0.0), and model / features as in load_resnet_data.
    Its "weights" are only the trainable ones.
    """
    arch = config.get("model", "resnet18")
    head = [nn.Dropout(config.get("dropout", 0.0)), nn.Linear(FEATURE_SIZE[(arch, config.get("features", "last"))], n_classes)]
    if config.get("finetune", "head") == "layer4":
        layer4 = copy.deepcopy(pretrained_resnet(arch).layer4)
        return nn.Sequential(layer4, nn.AdaptiveAvgPool2d(1), nn.Flatten(), *head)
    return nn.Sequential(*head)


def evaluate_with_flip(model, X, X_flip, y, batch_size=256):
    """Accuracy when each prediction averages the probabilities of an image and of its mirror image."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            probs = model(X[i:i + batch_size]).softmax(dim=1) + model(X_flip[i:i + batch_size]).softmax(dim=1)
            correct += (probs.argmax(dim=1) == y[i:i + batch_size]).sum().item()
    return correct / len(X)


# --------------------------------------------------------------------------------------------------
# 7. Scoring any saved run on the test set
# --------------------------------------------------------------------------------------------------
def build_dense(config, n_classes=5):
    """The dense baseline of P02: one hidden layer, described by image_size and hidden."""
    n_inputs = 3 * config["image_size"] ** 2
    return nn.Sequential(nn.Linear(n_inputs, config["hidden"]), nn.ReLU(), nn.Linear(config["hidden"], n_classes))


def score_on_test(stem):
    """Score a saved run on the test set (museum B) and write the result in experiments.csv.

    It rebuilds the model and the data preparation from the run's own config (dense, cnn or resnet18), so it works for
    every model of the project. Like add_test_score, it refuses a run that already has a test score (adr/0003).
    Returns (loss, accuracy).
    """
    config = load_result(stem)["config"]
    size = config["image_size"]
    if config["model"] == "dense":
        model = build_dense(config)
        data = load_data(size=size, splits=("test",))
    elif config["model"] == "cnn":
        model = build_cnn(config)
        data = to_images(load_data(size=size, splits=("test",)), size)
    elif config["model"] in ("resnet18", "resnet50"):
        model = build_resnet_head(config)
        data = load_resnet_data(config, splits=("test",))
    else:
        raise ValueError(f"do not know how to rebuild a model of type {config['model']!r}")
    return add_test_score(stem, model, data)
