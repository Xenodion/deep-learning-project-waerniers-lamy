# Model architectures: every network trained in the project

A summary of every model we trained, its architecture and its scores, to answer technical questions about the vision side.
Figures come from `src/deep_learning_project_waerniers_lamy_mahieddine/vision/utils.py`, `results/experiments.csv` and the ADRs
([0012](../../adr/0012-final-model.md), [0014](../../adr/0014-classical-baseline.md),
[0016](../../adr/0016-dense-regularisation.md), [0017](../../adr/0017-feedforward-on-features.md),
[0019](../../adr/0019-kiosk-int8.md)).

## What every model shares

- **Output:** always a final linear layer giving **5 scores (logits)**, one per genre. The softmax that turns them into
  probabilities is inside the loss (`CrossEntropyLoss`), so there is no explicit softmax layer in the network.
- **Training:** Adam, batches of 64, the weights of the **epoch with the best validation accuracy** are kept, fixed seeds.
- **Evaluation:** every choice is made on validation (museum A). The test set (museum B) is used once per model we keep
  ([ADR 0003](../../adr/0003-simulated-museum-b-test-set.md)).

## 1. Dense network, the baseline (P02)

```
32×32×3 image → flattened to 3,072 values → Linear 3072→128 + ReLU → Linear 128→5
```

- **One hidden layer** of 128 neurons (ReLU), then the output layer 128→5.
- **393,989 parameters**, almost all in the first layer (3,072 × 128).
- Result: **39.0% on validation, 41.8% on museum B.**
- Weakness: every pixel is a separate input with no notion of space, so the same pattern moved by a few pixels is new to it.
- P13 (Yoni) tried to regularise it: weight decay, dropout 0.2 and 0.5, 2 hidden layers, 16 or 64 px, mirrored copies. It
  stays around 39 to 43%. **The problem is not overfitting, it is the architecture.**

## 2. Small CNNs (P03 to P05)

```
64×64×3 image → [Conv 3×3 → (BatchNorm) → ReLU → MaxPool 2×2] × 3 blocks → global average pooling → Linear 64→5
```

| variant | blocks (channels) | parameters | validation | museum B |
|---|---|---|---|---|
| CNN, 30 epochs | 3 (16, 32, 64) | 23,909 | 53.6% | 58.7% |
| **CNN, 100 epochs (best)** | 3 (16, 32, 64) | 23,909 | 58.8 to 60.8% | **63.7%** |
| + batch norm | 3 | 24,133 | 59.8% | not scored |
| deeper | 4 (16, 32, 64, 128) | 98,085 | 58.4% (5 seeds) | 60.5% |

- **No hidden dense layer:** convolutions, then each feature map averaged into one number (global average pooling), then
  the output layer 64→5 directly.
- **17 times fewer parameters** than the dense network, for +22 points: the 3×3 filters are shared over the whole image, so
  the model is translation-invariant.
- An extra block or batch norm does not help: with 1,920 training images at 64 px, it plateaus around 60%.

## 3. ResNet-18, frozen: only the final layer is trained (P06)

**Architecture of ResNet-18** (pretrained on ImageNet, 1.2 million images, 1,000 classes):

```
224×224×3 image
→ conv1 7×7 (64 channels) + maxpool              112 → 56 px
→ layer1: 2 residual blocks (64 channels)         56 px
→ layer2: 2 residual blocks (128 channels)        28 px
→ layer3: 2 residual blocks (256 channels)        14 px
→ layer4: 2 residual blocks (512 channels)        7 px
→ global average pooling → 512 values
→ fc: ImageNet's Linear 512→1000 is replaced by Linear 512→5
```

- **The "18"** counts the layers with weights: 1 first convolution + 4 stages × 2 blocks × 2 convolutions 3×3 (16) + the fc
  layer = 18.
- **A residual block** is 2 convolutions 3×3 with batch norm, plus the shortcut `output = F(x) + x`. The gradient flows
  straight through the shortcut, which is what lets deep networks train without vanishing gradients.
- **Frozen version:** the whole network is fixed and only the new 512→5 layer learns.
  - **2,565 trained parameters** out of 11.2 million;
  - 30 epochs, learning rate 1e-3;
  - the 512 features of each image are computed once, so training takes a few seconds.
- Result: **80.3% on validation** (5 seeds, std 0.5), **84.5% on museum B.**

## 4. ResNet-18 fine-tuned: the final model (P06, ADR 0012)

The same network, but the **last stage (layer4) is unfrozen** as well as the new output layer:

```
conv1 → layer1 → layer2 → layer3     FROZEN (ImageNet weights)
→ layer4                             TRAINED (starts from the ImageNet weights)
→ global average pooling → Linear 512→5   TRAINED (new)
```

| | ResNet-18 frozen | **ResNet-18 fine-tuned** |
|---|---|---|
| what learns | the output layer | layer4 + the output layer |
| trained parameters | 2,565 | **8,396,293** (75% of the network) |
| learning rate, epochs | 1e-3, 30 epochs | **1e-4**, 8 epochs (best epoch: 4) |
| training time | a few seconds | about 10 minutes, once |
| validation | 80.3% | **83.3%** (3 seeds) |
| museum B | 84.5% | **86.0%** |
| time per photo, size | 70 ms, 45 MB | **the same**: same network, only the weights differ |

**Arguments to defend these choices:**

- **Why only layer4?**
  - The first layers detect generic patterns (edges, textures, colours), useful for any image.
  - The last ones encode ImageNet-specific concepts (dogs, cars…): those are the ones to adapt to paintings.
  - Retraining everything with 1,920 images would risk overfitting. This is the standard argument; we did not test it, as
    full fine-tuning was never tried.
- **Why a learning rate 10 times smaller (1e-4)?** To adjust the pretrained weights without destroying them. At 1e-3 we
  would erase what ImageNet taught.
- **Why it pays:** +3 points on validation for the same prediction cost as the frozen model. Only training costs more, and
  only once.
- **The rule was fixed before the run:** more than 2 points above the frozen model. It gained +3.0.

## 5. ResNet-18 against ResNet-50 (P08)

ResNet-50 was only tried **frozen**, with a new output layer. It was never fine-tuned.

| | ResNet-18 | ResNet-50 |
|---|---|---|
| block type | "basic": 2 conv 3×3 | "**bottleneck**": conv 1×1 (fewer channels) → conv 3×3 → conv 1×1 (back to more channels) |
| blocks per stage | 2, 2, 2, 2 (8 blocks) | 3, 4, 6, 3 (16 blocks × 3 conv + conv1 + fc = 50) |
| output features | 512 | **2,048** |
| output layer | Linear 512→5 (2,565 parameters) | Linear 2048→5 (10,245 parameters) |
| total parameters | 11.2 M | 23.5 M |
| compute per image (published figures) | 1.8 GFLOPs | 4.1 GFLOPs |
| validation, frozen | 80.3% | 81.9% (5 seeds) |
| time per photo, size | 70 ms, 45 MB | **139 ms, 94 MB** |

**Verdict:** +1.6 points, below our 2-point bar, for twice the time and size. Rejected on validation and **never scored on
museum B**.

## 6. Classical model (Vincent, P11, ADR 0014)

- **No network.** We extract **1,802 handcrafted features**:
  - **colour (38 values):** hue, saturation and value histograms (16 + 8 + 8 bins), plus the mean and std of each channel;
  - **edges (1,764 values):** HOG, histograms of gradient orientations, on a 128 px grayscale copy.
- **Classifier:** a random forest of 300 trees. Logistic regression was also tried (57.7% at best).
- Result: **60.0% on validation, 58.5% on museum B**, 43 ms per photo, 22 MB.
- **What it is for:** it answers the director's question. A neural network is worth it, but not any network: the dense
  network does worse than the classical model, the fine-tuned ResNet does +27.5 points better.

## 7. Other attempts

| who | model | architecture | validation | museum B |
|---|---|---|---|---|
| Alexandre (P07) | ResNet-18 frozen at 64 / 112 px | same network, smaller images | 70.2% / 75.4% | not scored |
| Alexandre (P08) | ResNet-18 frozen, multi-scale features | outputs of layer2 + 3 + 4 joined (896) → 5 | 80.6% | not scored |
| Alexandre (P08) | ResNet-18 frozen + 4× augmentation | crop, mirror, colour | 80.8% | not scored |
| Yoni (P14) | MLP on the handcrafted features | 1,802 → **256 ReLU** → dropout 0.5 → 5 | 60.5% | 62.7% |
| Yoni (P14) | ResNet-18 frozen + MLP head | 512 → **256 ReLU** → dropout → 5 | 80.8% | not scored (no gain) |
| Yoni (P16) | MobileNetV3-Large frozen | 960 → 5 | 82.9% | 84.8% |
| Yoni (P16) | **final model in int8** | same network, weights as 8-bit integers | 83.3% | 86.2% |

## Likely technical questions

- **"Why no hidden layer in the ResNet head?"** P14 tested it: a hidden layer of 256 neurons does no better (80.8% against
  80.3%, within the noise). The 512 ResNet features are already good enough for a single linear layer.
- **"Why 224 px?"** It is the size ResNet was pretrained on; at 64 px it drops to 70%. P07 shows that **pretraining and
  resolution each explain about half** of the gap between the CNN and the ResNet.
- **"Why standardise with the ImageNet mean and std?"** The pretrained weights expect inputs standardised exactly as during
  their training.
- **"Your std of 0 over 3 seeds?"** Do not present it as "std 0.0": the best epoch is a maximum over noisy epochs and hides
  the real spread. The realistic margin is about ±3.6 points on validation (480 images) and ±2.9 on museum B (600 images).
