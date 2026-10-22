# Architecture Decision Records

Short notes on **why** we made each significant choice, so we can defend it at the oral. Rules in
[0001](0001-record-architecture-decisions.md); copy [0000-template.md](0000-template.md) for a new one.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record decisions as short ADRs | Accepted |
| [0002](0002-classification-target-and-dataset-subset.md) | 5 genres, 600 images each, 224 px | Accepted |
| [0003](0003-simulated-museum-b-test-set.md) | Train in museum A, test once in simulated museum B | Accepted |
| [0004](0004-focus-on-genre-classifier.md) | Focus on the genre classifier; leave the language feature for later | Accepted |
| [0006](0006-dense-baseline.md) | Keep the dense baseline simple and unchanged: 32 px, one hidden layer of 128 | Accepted |
| [0007](0007-first-cnn.md) | Keep the first CNN as the reference for convolutional models: 3 blocks, 64 px | Proposed |
| [0008](0008-longer-training-and-deeper-cnn.md) | Train the CNN for 100 epochs without batch norm, and try a deeper CNN next (P05) | Proposed |
| [0009](0009-move-to-pretrained-resnet.md) | Stop tuning the small CNN and move to a pretrained ResNet-18 (P06) | Accepted |
| [0010](0010-resnet-results-and-final-model.md) | The pretrained ResNet-18 is our best model; we propose the frozen version as the final one | Proposed |

## Still to decide

- Which ResNet stage is the final model, frozen or fine-tuned (ADR 0010)
- Cost per image on a CPU for the dense network, a CNN and the ResNet, and the production recommendation (P07)
- The final script for the teacher (`run_final.py`)
