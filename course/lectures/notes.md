# Stages of AI

## Timeline

- **1943**: McCulloch and Pitts propose the first mathematical model of a neuron.
    - Binary threshold unit: it sums its inputs and "fires" if the sum exceeds a threshold.
    - Showed that networks of such units can compute logical functions (AND, OR, NOT).

- **1958**: Frank Rosenblatt builds the **Perceptron** (the NYT covered it, with hype: "the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence").
    - First neuron model that *learns* its weights from examples.
    - Single layer: it can only learn linearly separable problems.

- **1969**: Minsky and Papert publish *Perceptrons* and prove its limits (e.g. it cannot learn XOR).
    - Funding and interest collapse: first **AI winter**.
    - The missing piece was a way to train multi-layer networks.

- **1986**: Rumelhart, Hinton and Williams popularize **backpropagation**, which makes it possible to train multi-layer neural networks.
    - Idea: compute the error at the output, then propagate the gradient backward through the layers with the chain rule, and update the weights by gradient descent.
    - Limits at the time: not enough computational power and not enough data.

- **1990s**: Yann LeCun creates **LeNet** (LeNet-5 in 1998), a convolutional neural network trained on **MNIST** (handwritten digits).
    - Used in practice to read bank cheques and postal codes.
    - Introduced the core CNN ideas: convolutions, weight sharing, pooling.

- **1999**: NVIDIA releases the **GeForce 256**, marketed as the first GPU.
    - Built for video games (graphics), but its parallel architecture will later be reused for deep learning.

- **2007**: NVIDIA releases **CUDA**, a platform that lets developers program GPUs for general-purpose computation.
    - GPUs become usable for matrix operations, the core of neural network training.

- **2009**: Fei-Fei Li and her team release **ImageNet**.
    - Over 14 million labeled images, around 20,000 categories.
    - Brought the large-scale data that deep learning was missing.
    - Yearly competition: ILSVRC (1,000 classes, 1.2 million training images).

- **2012**: **AlexNet** (Krizhevsky, Sutskever, Hinton) wins ImageNet (ILSVRC).
    - Before: ~30% error rate (top-5).
    - With AlexNet: ~15% error rate, a huge jump.
    - Trained on 2 GPUs. Key ingredients: deep CNN, ReLU activation, dropout, GPU training, lots of data.
    - Started the deep learning boom.

- **2017**: **Transformer**, from the paper *"Attention Is All You Need"* (Vaswani et al., Google).
    - Replaces recurrence (RNN/LSTM) with **self-attention**.
    - Processes all tokens in parallel, so it scales well on GPUs.
    - Basis of BERT, GPT, and today's LLMs.

**Key milestones in AI:** 1986 Backpropagation, 2012 AlexNet, 2017 Transformer

## Why deep learning took off (the 3 ingredients)

1. **Algorithms**: backpropagation, CNNs, then attention.
2. **Data**: MNIST, then ImageNet, then web-scale text.
3. **Compute**: GPUs and CUDA.

The ideas existed decades before they worked at scale. 1986 to 2012 is mostly the wait for data and compute.

# CPU vs. GPU

| | CPU | GPU |
|---|---|---|
| Tasks | Complex, varied tasks | Simple but very repetitive tasks |
| Execution | Sequential | Parallel |
| Cores | Few, powerful (tens) | Thousands of small, simple cores |
| Optimized for | Low latency (do one thing fast) | High throughput (do many things at once) |
| Typical use | OS, general programs, control logic | Graphics, matrix multiplications, deep learning |

- Neural networks are mostly **matrix multiplications**, the same operation repeated on lots of numbers, which is exactly what a GPU is good at.
- Training a model on GPU can be orders of magnitude faster than on CPU.
- Today's specialized hardware goes further: TPUs (Google), tensor cores (NVIDIA).


# Perceptron

With two inputs $x_1, x_2$:

$$z = w_1 x_1 + w_2 x_2 + b$$

$$\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

With $n$ inputs, in vector form, $\mathbf{x}, \mathbf{w} \in \mathbb{R}^n$:

$$z = \mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^{n} w_j x_j + b$$

- $z$ is the score (pre-activation),
- $\hat{y} \in \{0, 1\}$ is the prediction,
- $y \in \{0, 1\}$ is the true label,
- $\mathbf{w}$ are the weights and $b$ is the bias.

The decision boundary is the hyperplane $\mathbf{w}^\top \mathbf{x} + b = 0$, which is why the perceptron only solves linearly separable problems.

## Matrix sizes

Notation: $m$ = number of examples (samples), $n$ = number of features (inputs). Example: $m = 1000$, $n = 10$.

**Difference between $n$ and $m$:**

| | $n$ | $m$ |
|---|---|---|
| Meaning | Number of **features**: values describing **one** example | Number of **examples** (samples): size of the dataset |
| Also called | Number of inputs of the perceptron ($x_1, \dots, x_n$) | Number of rows (or columns) of the dataset |
| Sets the size of | $\mathbf{w}$ and $\mathbf{x}$ | The number of predictions $\hat y$ and labels $y$ |
| Exercise (A, B, C, D) | $n = 2$ ($x_1$, $x_2$) | $m = 4$ |

- "Inputs" is ambiguous: the perceptron has $n$ inputs (the features of one example), while the dataset has $m$ examples. Prefer "features" for $n$ and "examples" for $m$.
- The parameters $\mathbf{w}$ and $b$ depend on $n$ only; adding more examples does not add parameters.
- Layout of $X$ depends on the convention: $(n, m)$ with examples in columns (course notes), or $(m, n)$ with examples in rows (TP notebook). The meaning of $n$ and $m$ stays the same.

**One example** (a single column of $X$):

| Object | Symbol | Shape | Example |
|---|---|---|---|
| Input vector | $\mathbf{x}$ | $(n, 1)$ | $(10, 1)$ |
| Weights | $\mathbf{w}$ | $(n, 1)$ | $(10, 1)$ |
| Bias | $b$ | scalar | $(1, 1)$ |
| Score | $z = \mathbf{w}^\top \mathbf{x} + b$ | scalar | $(1, 1)$ |
| Prediction / label | $\hat{y},\ y$ | scalar | $(1, 1)$ |

Check: $\mathbf{w}^\top$ is $(1, n)$ and $\mathbf{x}$ is $(n, 1)$, so $(1, n) \times (n, 1) = (1, 1)$, a scalar.

**All examples at once** (vectorized), with one example per column:

$$X = \begin{bmatrix} | & | & & | \\ \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \\ | & | & & | \end{bmatrix} \in \mathbb{R}^{n \times m}$$

$$\mathbf{z} = \mathbf{w}^\top X + b$$

| Object | Symbol | Shape | Example |
|---|---|---|---|
| Data matrix | $X$ | $(n, m)$ | $(10, 1000)$ |
| Weights | $\mathbf{w}$ | $(n, 1)$ | $(10, 1)$ |
| Bias | $b$ | scalar (broadcast) | $(1, 1)$ |
| Scores | $\mathbf{z}$ | $(1, m)$ | $(1, 1000)$ |
| Predictions | $\hat{\mathbf{y}}$ | $(1, m)$ | $(1, 1000)$ |
| True labels | $\mathbf{y}$ | $(1, m)$ | $(1, 1000)$ |

Check: $(1, n) \times (n, m) = (1, m)$. The inner dimensions ($n$) must match and disappear; the outer ones ($1$ and $m$) give the result shape.

- The bias $b$ is a scalar added to every entry of $\mathbf{w}^\top X$ (broadcasting).
- The number of parameters is $n + 1$ (here $11$), and does not depend on $m$.
- Other convention: one example per **row**, $X \in \mathbb{R}^{m \times n}$ and $\mathbf{z} = X\mathbf{w} + b$ of shape $(m, 1)$. Always check which convention is used (here: examples in columns).

# How to find w and b

In ML we usually gradient descent :
1) Data $X \rightarrow y$
2) Model function $\mathbf{w}^\top \mathbf{x} + b = \hat y$
3) Calculate the cost function
4) Optimizer

Gradient: for a function of several variables, $\nabla f(\mathbf{x}) = \left(\dfrac{\partial f}{\partial x_1}, \dots, \dfrac{\partial f}{\partial x_n}\right)$

Derivative: for a function of one variable, $f'(x) = \dfrac{df}{dx}$

In deep learning with perceptrons:

1) Start with $\mathbf{w} = \mathbf{0}$, $b = 0$
2) Go through the examples one by one. For each one, predict $\hat y$ and compare it with $y$.
3) Update the parameters with the perceptron rule:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat y)\,\mathbf{x}$$

$$b \leftarrow b + \eta\,(y - \hat y)$$

4) $\eta > 0$ is the learning rate (today $\eta = 1$). The error $y - \hat y$ is $-1$, $0$ or $1$.
5) One pass over the whole dataset is an **epoch**. Repeat epochs until one epoch has no mistakes.

## Is it mandatory to update the weights after each prediction?

No. Updating after every example is only one of three common strategies, and it is the one the classic perceptron uses.

| Variant | When weights update | Typical name |
|---|---|---|
| One example at a time | After every example | **Online / stochastic** (SGD) |
| A few examples at a time | After each batch of $B$ examples | **Mini-batch** |
| All examples | Once per epoch | **Batch** |

In the perceptron rule the update is automatic but not unconditional: the factor $(y - \hat y)$ is $0$ when the prediction is correct, so nothing changes. Updates only happen on mistakes (e.g. point A in epoch 1 of the exercise).

With batch or mini-batch, the update is computed over several examples and combined, usually by averaging:

$$\mathbf{w} \leftarrow \mathbf{w} + \frac{\eta}{B}\sum_{i=1}^{B}\left(y^{(i)} - \hat y^{(i)}\right)\mathbf{x}^{(i)}$$

All $B$ predictions use the same weights, so the result can differ from the one-at-a-time version.

Trade-offs:

- **One at a time**: cheap per step, updates often, but noisy. It is what the perceptron convergence theorem assumes.
- **Mini-batch**: uses the vectorized $(n, m)$ matrix operations, so it is fast on GPUs. It is the standard in deep learning.
- **Full batch**: stable, exact gradient, but slow per update when $m$ is large.

For the exercise below, follow the one-at-a-time rule, as defined in the course.

# Does it always stop ? The convergence theorem

exercise:

| Point | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| A | 1 | 3 | 1 |
| B | 1 | 0 | 0 |
| C | 3 | 1 | 1 |
| D | 0 | 0 | 0 |

Initial values: $\mathbf{w} = (0,0)$, $b = 0$, $\eta = 1$.

## Solution

Points are processed in the order A, B, C, D. Prediction: $\hat y = 1$ if $z \geq 0$, else $0$ (so $z = 0$ gives $\hat y = 1$).

Update rule: $\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat y)\,\mathbf{x}$ and $b \leftarrow b + \eta (y - \hat y)$.

### Epoch 1

Start: $\mathbf{w} = (0,0)$, $b = 0$.

| Step | Point | $z = \mathbf{w}^\top\mathbf{x}+b$ | $\hat y$ | $y$ | $y-\hat y$ | New $\mathbf{w}$ | New $b$ |
|---|---|---|---|---|---|---|---|
| 1 | A $(1,3)$ | $0$ | 1 | 1 | 0 | $(0,0)$ | $0$ |
| 2 | B $(1,0)$ | $0$ | 1 | 0 | $-1$ | $(-1,0)$ | $-1$ |
| 3 | C $(3,1)$ | $-3+0-1=-4$ | 0 | 1 | $+1$ | $(2,1)$ | $0$ |
| 4 | D $(0,0)$ | $0$ | 1 | 0 | $-1$ | $(2,1)$ | $-1$ |

End of epoch 1: $\mathbf{w} = (2,1)$, $b = -1$, 3 mistakes (B, C, D).

### Epoch 2

| Step | Point | $z$ | $\hat y$ | $y$ | $y-\hat y$ | New $\mathbf{w}$ | New $b$ |
|---|---|---|---|---|---|---|---|
| 1 | A $(1,3)$ | $2+3-1=4$ | 1 | 1 | 0 | $(2,1)$ | $-1$ |
| 2 | B $(1,0)$ | $2-1=1$ | 1 | 0 | $-1$ | $(1,1)$ | $-2$ |
| 3 | C $(3,1)$ | $3+1-2=2$ | 1 | 1 | 0 | $(1,1)$ | $-2$ |
| 4 | D $(0,0)$ | $-2$ | 0 | 0 | 0 | $(1,1)$ | $-2$ |

End of epoch 2: $\mathbf{w} = (1,1)$, $b = -2$, 1 mistake (B).

### Epoch 3

| Point | $z$ | $\hat y$ | $y$ | Correct? |
|---|---|---|---|---|
| A $(1,3)$ | $1+3-2=2$ | 1 | 1 | yes |
| B $(1,0)$ | $1-2=-1$ | 0 | 0 | yes |
| C $(3,1)$ | $3+1-2=2$ | 1 | 1 | yes |
| D $(0,0)$ | $-2$ | 0 | 0 | yes |

No mistakes in a full epoch, so the algorithm stops.

**Result:** $\mathbf{w} = (1,1)$, $b = -2$. The decision boundary is $x_1 + x_2 - 2 = 0$, i.e. $x_1 + x_2 = 2$.

The data is linearly separable, so the algorithm converges, as the convergence theorem guarantees.
