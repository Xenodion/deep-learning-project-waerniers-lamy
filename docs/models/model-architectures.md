# Model architectures: every network trained in the project

A summary of every model we trained, its architecture and its scores, to answer technical questions about the vision side.
Figures come from `src/deep_learning_project_waerniers_lamy_mahieddine/vision/utils.py`, `results/experiments.csv` and the ADRs
([0012](../../adr/0012-final-model.md), [0014](../../adr/0014-classical-baseline.md),
[0016](../../adr/0016-dense-regularisation.md), [0017](../../adr/0017-feedforward-on-features.md),
[0019](../../adr/0019-kiosk-int8.md)).

## Reminder: the general techniques

### Overfitting, and how we see it

A model **overfits** when it learns the training images by heart instead of the rule behind them: training accuracy keeps
rising while validation accuracy stalls or falls. Our clearest case is the dense baseline (P02): **98% on train, 39% on
validation**. P12 even showed that it reaches 100% training accuracy on **shuffled labels**, where there is nothing to learn
but the images themselves: pure memorisation. The gap between the train and validation curves is the symptom to look for.

### Techniques against overfitting

| technique | what it does | why it helps | in our project |
|---|---|---|---|
| **Early stopping** | keep the weights of the epoch with the best validation accuracy, not the last one | stops before the model starts memorising | used in every run (`fit` keeps the best epoch) |
| **Weight decay (L2)** | adds a penalty on the size of the weights to the loss (`weight_decay` in Adam) | large weights fit noise; small weights give a smoother function | P13: no gain on the dense network (39.3% at best against 39.0%); P14: kept in the feature MLP |
| **Dropout** | during training, sets a random share of the neurons to 0 at each step (here 0.2 or 0.5); off at prediction time | no neuron can rely on one specific other: the network learns redundant, more general features | P13: no gain on the dense network; P14: used in the MLP heads |
| **Data augmentation** | random changes of the training images: crop, mirror, colour | the model sees more varied images, so it cannot memorise the exact pixels | P08: 4 augmented copies, 80.8% against 80.3% on the frozen ResNet (within the noise); P13: mirrored copies, best dense network (42.5%) |
| **Fewer trainable parameters** | a smaller model has less capacity to memorise | the training set (1,920 images) is small | the reason for freezing the ResNet and for the CNN's weight sharing |
| **Transfer learning** | start from a network pretrained on a large dataset (ImageNet, 1.2 million images) and reuse its features | the features were learned on far more images than we have, so they generalise | the biggest gain of the project: from about 60% (CNN) to 80 to 83% (ResNet-18) |
| **Freezing / partial fine-tuning** | keep the early layers fixed and train only the last ones | fewer trainable weights, and the generic early features are not damaged | frozen ResNet: 2,565 trained parameters; final model: only layer4 and the output layer |
| **Smaller learning rate for fine-tuning** | 1e-4 instead of 1e-3 when training pretrained layers | small steps adjust the pretrained weights without erasing them | final model, layer4 at 1e-4 |
| **Global average pooling** | averages each feature map into one number instead of flattening every pixel | far fewer weights in the final layer | CNNs and ResNets: 64 or 512 inputs to the output layer instead of thousands |
| **Weight sharing (convolution)** | the same small 3×3 filter slides over the whole image | far fewer parameters than a dense layer, and a pattern is recognised wherever it is | CNN: 23,909 parameters against 393,989 for the dense network, +22 points |

### Techniques for training well (not against overfitting)

| technique | what it does | why it helps | in our project |
|---|---|---|---|
| **Batch normalisation** | standardises the outputs of a layer over each batch (mean 0, std 1), then rescales them with 2 learned numbers | stabilises training, allows larger learning rates; a mild regularising side effect from the batch noise | P04: no gain on the small CNN (59.8%); built into every ResNet block |
| **Residual connections** | a block outputs `F(x) + x`: it learns a correction added to its input | the gradient flows back through the `+ x` shortcut without shrinking, so very deep networks train | what makes ResNet-18 and ResNet-50 trainable; not a regulariser |
| **Standardised inputs** | subtract the mean and divide by the std of each colour channel | every input on the same scale, gradients of a sane size | P12: raw 0-255 pixels killed 95 to 100% of the ReLUs; the ResNets use the ImageNet mean and std |
| **ReLU activation** | `max(0, x)` instead of sigmoid or tanh | its gradient is 1 for positive inputs, so it does not vanish through many layers | P12: 6 sigmoid layers did not train, ReLU did |
| **He initialisation** | starting weights with a spread scaled to the layer size, made for ReLUs | the activations keep the same spread from layer to layer | P12: a 10-layer network trained only with it |
| **Gradient clipping** | caps the size of the gradient before each step | prevents one huge step from destroying the weights | P12: avoided NaN at a too-high learning rate |
| **Adam optimiser** | gradient descent with a step size adapted per weight | trains reliably with little tuning | every run; P12 showed it can hide a bad learning rate |

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
  layer = 18. `layer1` to `layer4` are the names PyTorch gives to the 4 **stages**, not to single layers (see section 4).
- **A residual block** is 2 convolutions 3×3 with batch norm, plus the shortcut `output = F(x) + x`. The gradient flows
  straight through the shortcut, which is what lets deep networks train without vanishing gradients.
- **Frozen version:** the whole network is fixed and only the new 512→5 layer learns.
  - **2,565 trained parameters** out of 11.2 million;
  - 30 epochs, learning rate 1e-3;
  - the 512 features of each image are computed once, so training takes a few seconds.
- Result: **80.3% on validation** (5 seeds, std 0.5), **84.5% on museum B.**

## 4. ResNet-18 fine-tuned: the final model (P06, ADR 0012)

The same network, but the **last stage (layer4) is unfrozen** as well as the new output layer.

**Careful with the word "layer".** In PyTorch, `layer1` to `layer4` are not single layers: they are the 4 **stages** of the
network, each made of 2 residual blocks of 2 convolutions. The final model keeps **all 18 layers**, and every photo goes
through all of them. "Fine-tuning layer4" means training the last stage, its 4 convolutions, plus the new output layer:

```
conv1 (7×7)                              1 layer     FROZEN (ImageNet weights)
layer1: 2 blocks × 2 conv 3×3            4 layers    FROZEN
layer2: 2 blocks × 2 conv 3×3            4 layers    FROZEN
layer3: 2 blocks × 2 conv 3×3            4 layers    FROZEN
layer4: 2 blocks × 2 conv 3×3            4 layers    TRAINED (starts from the ImageNet weights)
global average pooling                   (no weights)
fc: Linear 512→5                         1 layer     TRAINED (new)
                                        ─────────
                                         18 layers: 13 frozen, 5 trained
```

So 5 of the 18 layers are trained, but they hold **75% of the weights** (8.4 of 11.2 million): the last stage works on 512
channels, so its convolutions are by far the largest. Detail: the first block of layer2, layer3 and layer4 also has a 1×1
convolution in its shortcut, to change the number of channels (256 to 512 in layer4). It is trained along with layer4 but,
by convention, not counted in the 18.

**What is replaced, what is retrained.** Only the output layer is removed and replaced; layer4 is kept and retrained:

| part | what happens when fine-tuning |
|---|---|
| conv1, layer1 to layer3 | **kept and frozen**: ImageNet weights, never changed |
| layer4 | **kept, then retrained**: starts from its ImageNet weights, which training adjusts (learning rate 1e-4) |
| fc | **removed and replaced**: ImageNet's `Linear 512→1000` (1,000 ImageNet classes) becomes a new `Linear 512→5` (our 5 genres), with random starting weights |

Two different operations: `fc` must be **replaced** because its output has the wrong size (1,000 classes, not 5 genres);
layer4 has the right shape and is only **unfrozen**, so that it can learn. In `build_resnet_head` (`vision/utils.py`):
`copy.deepcopy(pretrained_resnet(arch).layer4)` keeps layer4 with its ImageNet weights, and `nn.Linear(n_features, n_classes)`
creates the new fc. The frozen ResNet-18 makes the same replacement of `fc`; only its layer4 stays frozen.

**From feature maps to genres.** The output is 5 scores, not 5 feature maps:

```
layer4      → 512 feature maps of 7×7 pixels        (what the convolutions produce)
avg pooling → 512 features                          (each map averaged into one number)
new fc      → 5 scores (logits), one per genre      (a weighted combination of the 512 features)
softmax     → 5 probabilities                       (inside the loss during training)
```

**layer4 and fc learn together, not one after the other.** At every training step the genre error (cross-entropy loss) is
backpropagated through fc, then through layer4, and **both are updated in the same step**:

- **fc** learns *how to combine* the 512 features into a genre;
- **layer4** learns *which features to compute*: its ImageNet features (made to tell a dog from a car) shift towards
  features that tell a landscape from a cityscape;
- the gradient stops at layer3: everything before it is frozen and does not change.

**In one sentence:** we replace the 1,000-class output layer with a 5-genre one, and retrain the last stage together with
it, so that its features adapt to paintings while the earlier, generic layers keep their ImageNet weights.

### What makes it different from the frozen ResNet-18

The difference is **what is allowed to change**: only the decision made on the features, or the features themselves.

**Frozen ResNet-18 (P06): the features are fixed.**

- layer4 outputs the same 512 features as on ImageNet, made to tell dogs, cars and cups apart, not paintings.
- Only the new fc learns: it finds the best **linear combination** of these fixed 512 features to separate the 5 genres.
- If two genres look alike *in ImageNet's features*, fc cannot fix it: it can only draw boundaries between points it cannot
  move.
- This is **feature extraction**: the ResNet is used as a fixed feature calculator.

**Fine-tuned ResNet-18 (final model): the features adapt.**

- layer4 is retrained with fc, so the 512 features themselves change: they move towards what separates a landscape from a
  cityscape, or a religious painting from a portrait.
- The points move, so fc can separate them better.

| | ResNet-18 frozen | **ResNet-18 fine-tuned** |
|---|---|---|
| what learns | the output layer (fc) | layer4 + the output layer (fc) |
| the 512 features | fixed (ImageNet) | **adapted to paintings** |
| trained parameters | 2,565 | **8,396,293** (75% of the network) |
| learning rate, epochs | 1e-3, 30 epochs | **1e-4**, 8 epochs (best epoch: 4) |
| training time | the 512 features computed **once**, then a few seconds | the output of layer3 computed once, layer4 rerun at every epoch: about 10 minutes, once |
| overfitting risk | very low | higher: 8.4 M weights for 1,920 images, hence the small learning rate, 8 epochs and the best epoch kept |
| validation | 80.3% (5 seeds) | **83.3%** (3 seeds) |
| museum B | 84.5% | **86.0%** |
| time per photo, size | 70 ms, 45 MB | **the same**: same network, only the weights differ |

**What to say:** fine-tuning costs nothing at prediction time, only 10 minutes of training once, and gains +3 points on
validation, because the model no longer has to make do with features built for other images.

**Careful with museum B:** the gain there is +1.5 points, inside the ±2.9-point margin. The choice was made on validation,
as the protocol requires; do not present the museum B gain alone as proof.

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
