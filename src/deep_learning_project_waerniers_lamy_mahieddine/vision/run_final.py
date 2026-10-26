"""Reproduce the test score of our final model (adr/0012) on museum B.

The final model is a ResNet-18 pretrained on ImageNet, with its last block fine-tuned on our paintings (run resnet18_layer4 in
results/experiments.csv). This script loads it, runs it on the 600 test images of museum B, and prints the accuracy, the loss, the
accuracy per genre and the confusion matrix, next to the score recorded when the model was first scored. It writes nothing in
results/, and museum B was not used to choose this model (adr/0003).

From the repository root:
    python src/deep_learning_project_waerniers_lamy_mahieddine/vision/run_final.py            # about 1 minute
    python src/deep_learning_project_waerniers_lamy_mahieddine/vision/run_final.py --train    # retrain the model first (about 15 minutes)

What it needs:
  - the images and split.csv in data/wikiart_genre5 (see "Reproduce the data" in the README). If split.csv is missing it is rebuilt;
  - the ImageNet weights of ResNet-18, downloaded the first time (about 45 MB, needs an internet connection);
  - the trained weights results/final_model.pt, unless you use --train.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/: lets this file run without installing the package

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from deep_learning_project_waerniers_lamy_mahieddine.vision import utils  # noqa: E402

FINAL_RUN = "resnet18_layer4"  # name of the final run in results/experiments.csv (adr/0012)
FINAL_WEIGHTS = utils.RESULTS_DIR / "final_model.pt"


def train(config):
    """Train the final model again from the settings of the saved run (seed 0, best epoch kept on validation)."""
    print(f"Training the final model again ({config['epochs']} epochs, seed {config['seed']}): about 15 minutes.")
    data = utils.load_resnet_data(config)  # train and validation images through the frozen part of the ResNet
    torch.manual_seed(config["seed"])
    model = utils.build_resnet_head(config)
    utils.fit(model, data, epochs=config["epochs"], lr=config["lr"], batch_size=config["batch_size"],
              weight_decay=config["weight_decay"], seed=config["seed"])
    return model  # fit leaves the model with its best epoch


def main():
    runs = utils.summary()
    run = runs[runs["name"] == FINAL_RUN].sort_values("date").iloc[0]
    config = utils.load_result(run["file"])["config"]
    print(f"Final model: ResNet-18 pretrained on ImageNet, last block fine-tuned, {config['image_size']} px (run {run['file']})")

    # 1. the data
    if not (utils.DATA_DIR / "metadata.csv").exists():
        sys.exit(f"The images are not in {utils.DATA_DIR}. Build them first: see 'Reproduce the data' in the README.")
    if not (utils.DATA_DIR / "split.csv").exists():
        print("split.csv not found: building it (museum A / museum B split, deterministic).")
        utils.make_split()

    # 2. the model
    if "--train" in sys.argv:
        model = train(config)
    else:
        weights = FINAL_WEIGHTS if FINAL_WEIGHTS.exists() else utils.RESULTS_DIR / f"{run['file']}.pt"
        if not weights.exists():
            sys.exit(f"No trained weights found ({FINAL_WEIGHTS}). Run again with --train to train the model (about 15 minutes).")
        model = utils.build_resnet_head(config)
        model.load_state_dict(torch.load(weights))
        print(f"Loaded the trained weights from {weights.name}")

    # 3. the test images of museum B
    print("Running the model on the test images (about 1 minute)...")
    data = utils.load_resnet_data(config, splits=("test",))
    loss, accuracy = utils.evaluate(model, data["X_test"], data["y_test"])
    with torch.no_grad():
        predicted = model.eval()(data["X_test"]).argmax(dim=1)
    classes = data["classes"]
    confusion = pd.crosstab(pd.Series([classes[i] for i in data["y_test"]], name="true genre"),
                            pd.Series([classes[i] for i in predicted], name="predicted"))
    confusion = confusion.reindex(index=classes, columns=classes, fill_value=0)

    # 4. the report
    print("\n" + "=" * 70)
    print(f"TEST SET (museum B, {len(data['y_test'])} paintings by artists never seen in training)")
    print(f"  accuracy: {accuracy:.1%}    loss: {loss:.3f}")
    if run["test_acc"] == run["test_acc"]:  # not NaN: the score recorded when the model was first scored
        gap = abs(accuracy - run["test_acc"])
        verdict = "matches" if gap < 0.005 else "DIFFERS from"
        print(f"  recorded on {run['test_date']}: {run['test_acc']:.1%} (loss {run['test_loss']:.3f}) -> this run {verdict} it")
    print("\naccuracy per genre:")
    for i, genre in enumerate(classes):
        print(f"  {genre:20s} {confusion.iloc[i, i] / confusion.iloc[i].sum():6.1%}   ({confusion.iloc[i, i]} of {confusion.iloc[i].sum()})")
    print("\nconfusion matrix (rows: true genre, columns: predicted genre):")
    print(confusion.to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
