# Architecture Decision Records

Short notes on **why** we made each significant choice, so we can defend it at the oral. Rules in
[0001](0001-record-architecture-decisions.md); copy [0000-template.md](0000-template.md) for a new one.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record decisions as short ADRs | Accepted |
| [0002](0002-classification-target-and-dataset-subset.md) | 5 genres, 600 images each, 224 px | Accepted |
| [0003](0003-simulated-museum-b-test-set.md) | Train in museum A, test once in simulated museum B | Accepted |
| [0004](0004-focus-on-genre-classifier.md) | Focus on the genre classifier; leave the language feature for later | Accepted |
| [0005](0005-nlp-bonus-by-ladder.md) | Language feature as a bonus, built one rung at a time | Accepted |

## Still to decide

- Input representation (image size for the dense net, standardisation)
- Dense baseline architecture and training loop (tag m1)
- Regularisation and monitoring protocol (tag m2)
- Pretrained ResNet: variant, frozen or fine-tuned
- Classical baseline and the production recommendation (neural network or not)
