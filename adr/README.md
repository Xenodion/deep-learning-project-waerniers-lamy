# Architecture Decision Records

Short notes on **why** we made each significant choice, so we can defend it at the oral. Rules in
[0001](0001-record-architecture-decisions.md); copy [0000-template.md](0000-template.md) for a new one.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record decisions as short ADRs | Accepted |
| [0002](0002-classification-target-and-dataset-subset.md) | 5 genres, 600 images each, 224 px | Accepted |
| [0003](0003-simulated-museum-b-test-set.md) | Train in museum A, test once in simulated museum B | Accepted |
| [0004](0004-focus-on-genre-classifier.md) | Focus on the genre classifier; leave the language feature for later | Accepted, amended by 0014 |
| [0005](0005-nlp-recommendation-ladder.md) | Natural-language painting recommendation as an experimental multimodal ladder | Accepted, amended by 0013 |
| [0006](0006-dense-baseline.md) | Keep the dense baseline simple and unchanged: 32 px, one hidden layer of 128 | Accepted |
| [0007](0007-first-cnn.md) | Keep the first CNN as the reference for convolutional models: 3 blocks, 64 px | Accepted |
| [0008](0008-longer-training-and-deeper-cnn.md) | Train the CNN for 100 epochs without batch norm, and try a deeper CNN next (P05) | Accepted |
| [0009](0009-move-to-pretrained-resnet.md) | Stop tuning the small CNN and move to a pretrained ResNet-18 (P06) | Accepted |
| [0010](0010-resnet-results-and-improvements.md) | ResNet results, and the improvements we will try | Accepted |
| [0011](0011-resnet-improvement-round.md) | Try a few cheap ResNet improvements before choosing the final model (P08) | Accepted |
| [0012](0012-final-model.md) | The final model: ResNet-18 pretrained on ImageNet, last block fine-tuned, 224 px | Accepted |
| [0013](0013-nlp-evaluation-protocol.md) | Evaluate the NLP ladder with blind, pooled, graded judgments | Accepted |
| [0014](0014-classical-baseline.md) | A classical baseline (handcrafted features, random forest) to answer "is a neural network worth it?" | Accepted, amends 0004 |
| [0015](0015-training-doctor.md) | Diagnose training from inside the network (P12), and keep the P02 baseline as is | Accepted |
| [0016](0016-dense-regularisation.md) | The final feed-forward network: the P02 network with mirrored training copies (P13) | Accepted |
| [0017](0017-feedforward-on-features.md) | At equal inputs a feed-forward network ties with the forest: the features decide (P14) | Accepted |
| [0018](0018-phone-photo-robustness.md) | The final model resists light and angle, not loss of detail: the app must check the photo (P15) | Accepted |
| [0019](0019-kiosk-int8.md) | On the kiosk, run the final model quantized to int8 (P16) | Accepted, complements 0012 |
| [0020](0020-confidence-and-abstention.md) | Show two genres when the confidence is below 0.86 (P17) | Accepted |

The whole vision work in one page, in order: [9999-vision-executive-summary.md](9999-vision-executive-summary.md).
The whole NLP work in one page: [9998-nlp-executive-summary.md](9998-nlp-executive-summary.md).

## Still to do

- The presentation (10 minutes and 5 minutes of questions), deadline 30 October 2026
