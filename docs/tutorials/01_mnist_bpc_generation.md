# Tutorial 01 — Generating and morphing MNIST digits with bidirectional predictive coding

**Goal.** Train a *bidirectional* predictive coding (bPC) network on MNIST. Then use it
as a generator: clamp a label, let the image settle by inference, and then swap the
clamped label to a different digit. Record every few inference steps, so that you can
watch the image morph from one digit into another and export it as a GIF.

**What you will write:** one notebook, `RELAX/notebooks/mnist_bpc_generation.ipynb`.
Each `python` block below is meant to be one notebook cell. Type them yourself rather
than copy-pasting, and run each cell before moving on. Every section ends with a
**Checkpoint** describing what you should see.

**Time:** about 5 minutes of compute on a CPU (training is about 30 s/epoch; one
morph run of 16,000 steps is under a minute).

Contents:

1. [The idea in equations](#1-the-idea-in-equations)
2. [Setup](#2-setup)
3. [Data](#3-data)
4. [The model](#4-the-model)
5. [Training](#5-training)
6. [Saving and loading the weights](#6-saving-and-loading-the-weights)
7. [Generation as inference](#7-generation-as-inference)
8. [Why the inference learning rate matters](#8-why-the-inference-learning-rate-matters)
9. [Looking at the result](#9-looking-at-the-result)
10. [Exporting the video](#10-exporting-the-video)
11. [Experiments to try](#11-experiments-to-try)
12. [Gotchas](#12-gotchas)

---

## 1. The idea in equations

### The network

Notation:

| Symbol | Meaning | Size |
|---|---|---|
| $u$ | image (pixels) | 784 |
| $x_0$, $x_1$ | the two hidden layers of latent activity (the `PCLayer`s) | 256 each |
| $y$ | label, one-hot | 10 |
| $f$ | ReLU | |

A bPC network has two pathways that **share the same latent variables** $x_0, x_1$:

- **Bottom-up (recognition):** each layer predicts the one above it, ending in a prediction of the label.

$$\mu_0^{\uparrow} = W_1 u, \qquad \mu_1^{\uparrow} = W_2 f(x_0), \qquad \hat y = W_3 f(x_1)$$

- **Top-down (generation):** each layer predicts the one below it, ending in a prediction of the image.

$$\mu_1^{\downarrow} = V_1 y, \qquad \mu_0^{\downarrow} = V_2 f(x_1), \qquad \hat u = V_3 f(x_0)$$

(Biases are omitted for readability.)

### The energy

The energy is the sum of all squared prediction errors, weighted by $\alpha_\uparrow$ and $\alpha_\downarrow$:

$$
E = \alpha_\uparrow \Big[ \tfrac12\|x_0-\mu_0^\uparrow\|^2 + \tfrac12\|x_1-\mu_1^\uparrow\|^2 + \tfrac12\|y-\hat y\|^2 \Big]
  + \alpha_\downarrow \Big[ \tfrac12\|x_1-\mu_1^\downarrow\|^2 + \tfrac12\|x_0-\mu_0^\downarrow\|^2 + \tfrac12\|u-\hat u\|^2 \Big]
$$

Every term is **local**: each one involves only one layer and its neighbour. This is
what makes PC biologically interesting.

### One energy, three uses

The *same* energy function is used for everything. What changes is **which variables
are clamped** (fixed) and which are **free** (updated by gradient descent on $E$):

| Use | Clamped | Free during inference | Weights |
|---|---|---|---|
| Training | $u$ = image, $y$ = label | $x_0, x_1$ | updated after inference |
| Classification | $u$ = image | (we use the fast feed-forward approximation: one bottom-up pass) | frozen |
| **Generation (this tutorial)** | $y$ = label | $x_0, x_1$, **and $u$** | frozen |

In generation the image $u$ stops being data and becomes **one more latent variable**.
Its gradient is:

$$
\frac{\partial E}{\partial u} = -\alpha_\uparrow W_1^\top (x_0 - \mu_0^\uparrow) + \alpha_\downarrow (u - \hat u)
$$

So two forces act on the image:

- The first term pushes $u$ to be an image that **explains $x_0$ bottom-up**.
- The second term pulls $u$ towards **what the top-down path predicts** ($\hat u$).

### The label swap

Let inference run with label $A$ clamped until the energy is low: an "A" has formed.
Then **replace the clamp with label $B$**. The two terms that touch $y$,
$\tfrac12\|y-\hat y\|^2$ and $\tfrac12\|x_1 - V_1 y\|^2$, suddenly become large.
Inference now has to reduce them. Because every term is local, the error can only
travel **one layer per step**: from $y$ to $x_1$, then from $x_1$ to $x_0$, then from
$x_0$ to $u$. That is why the morph is gradual and takes thousands of steps, and why
it makes a good video.

> **Difference from library notebook 5.** Notebook `5_bidirectional_pc.ipynb` generates
> images with a single pass, `model.model_down(label)`, with no inference at all. That
> gives a picture, but not a *process*. Here we run real PC inference over the image,
> which is what lets us see it evolve.

---

## 2. Setup

Create the folder `RELAX/notebooks/` and inside it a new notebook
`mnist_bpc_generation.ipynb`. Select the **relax** kernel. In VS Code or Jupyter, pick
the interpreter `~/miniconda3/envs/relax/bin/python`.

```python
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display
from tqdm import tqdm

import predictive_coding as pc

# Reproducibility: same random numbers every run
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Paths are relative to the notebook, which lives in RELAX/notebooks/.
# data/ and runs/ are both listed in .gitignore, so nothing here gets committed.
DATA_DIR = Path("../data")
RUN_DIR = Path("../runs/mnist_bpc_generation")
RUN_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")
```

**Checkpoint:** the cell runs without errors. If `import predictive_coding` fails, the
notebook is not using the `relax` environment.

---

## 3. Data

```python
transform = transforms.Compose([
    transforms.ToTensor(),                      # PIL image -> tensor in [0, 1], shape (1, 28, 28)
    transforms.Lambda(lambda x: torch.flatten(x)),  # -> shape (784,)
])
train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

batch_size = 500
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(len(train_dataset), len(test_dataset))
```

**Checkpoint:** prints `60000 10000`, and `RELAX/data/MNIST/` now exists.

---

## 4. The model

This is notebook 5's model with three small changes, explained below the code.

```python
class BidirectionalPC(nn.Module):
    def __init__(self, alpha_up=1.0, alpha_down=1.0):
        super().__init__()
        # The two latent layers x0 and x1. Each is used by BOTH pathways.
        self.pcs = nn.ModuleList([pc.PCLayer(), pc.PCLayer()])
        act = nn.ReLU()

        # bottom-up: image -> x0 -> x1 -> label
        self.model_up = nn.Sequential(
            nn.Linear(784, 256), self.pcs[0], act,
            nn.Linear(256, 256), self.pcs[1], act,
            nn.Linear(256, 10),
        )
        # top-down: label -> x1 -> x0 -> image   (note: pcs in reverse order)
        self.model_down = nn.Sequential(
            nn.Linear(10, 256), self.pcs[1], act,
            nn.Linear(256, 256), self.pcs[0], act,
            nn.Linear(256, 784),
        )
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down

    def forward(self, inputs):
        img, y = inputs
        # 1) Bottom-up pass. Each PCLayer stores the energy 0.5*(x - mu_up)^2 in ._energy
        out_up = self.model_up(img)
        e_up = [p._energy for p in self.pcs]
        # 2) Top-down pass. It OVERWRITES ._energy with 0.5*(x - mu_down)^2
        out_down = self.model_down(y)
        e_down = [p._energy for p in self.pcs]
        # 3) Store the weighted sum, which is what PCTrainer will read
        for p, eu, ed in zip(self.pcs, e_up, e_down):
            p._energy = self.alpha_up * eu + self.alpha_down * ed
        return out_up, out_down


def loss_fn(outputs, _y, _img, alpha_up, alpha_down):
    """Energy of the two *output* terms: 0.5||y - y_hat||^2 and 0.5||u - u_hat||^2."""
    out_up, out_down = outputs
    return (alpha_up * 0.5 * (out_up - _y).pow(2).sum()
            + alpha_down * 0.5 * (out_down - _img).pow(2).sum())
```

What to understand here:

- **How a `PCLayer` behaves** (read `third_party/PredictiveCoding/predictive_coding/pc_layer.py`, method `forward`):
  - In `train()` mode it **ignores its input $\mu$ and returns its own latent `_x`**. It also stores the energy $\tfrac12(\mu - x)^2$ summed over everything in `_energy`.
  - In `eval()` mode it is transparent: it returns $\mu$ unchanged. That is why an eval-mode `model_up(img)` is a plain feed-forward classifier.
- **Why sharing works:** `self.pcs[0]` is the *same Python object* in both `Sequential`s. So both pathways read and write the same `_x`, and the bottom-up and top-down predictions are compared against the same latent.
- **Change 1 — `nn.ModuleList` instead of a plain list.** It makes `model.pcs` show up properly as a submodule. It changes nothing about the maths.
- **Change 2 — default `alpha_down = 1.0` instead of `1e-4`.** With `1e-4` the top-down pathway is almost silent. Section 11 shows what that does to generation.
- **Change 3 — `0.5 *` in the loss**, so that the output terms have the same form as the `PCLayer` energies, as in the equation for $E$.

```python
alpha_up, alpha_down = 1.0, 1.0
model = BidirectionalPC(alpha_up, alpha_down).to(device)
print(model)
print("weights:", sum(p.numel() for p in model.parameters()))
```

**Checkpoint:** `print(model)` lists `pcs`, `model_up` and `model_down`. The same
`PCLayer()` entries appear in all three: that is the sharing. The weight count is
**539,418**. Work it out by hand from the layer sizes; it's a good check that you
understand the architecture.

---

## 5. Training

For every batch, `PCTrainer.train_on_batch` does the following:

1. **Initialise the latents** with a forward pass ("sample $x$"). For each `PCLayer`, `_x` is set to its input $\mu$.
2. Run **T inference steps**: compute $E$, then take an SGD step on all `_x` only. The weights are untouched.
3. Take **one weight step** (Adam) on the energy at the last inference step (`update_p_at="last"`).

```python
T = 20  # inference steps per batch

trainer = pc.PCTrainer(
    model,
    T=T,
    optimizer_x_fn=optim.SGD, optimizer_x_kwargs={"lr": 0.01},   # inference
    update_p_at="last",
    optimizer_p_fn=optim.Adam, optimizer_p_kwargs={"lr": 1e-3},  # learning
    plot_progress_at=[],
)


def test_accuracy(model):
    """Classify with one bottom-up feed-forward pass (PCLayers transparent in eval mode)."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, label in test_loader:
            pred = model.model_up(img.to(device)).argmax(dim=1)
            correct += (pred == label.to(device)).sum().item()
    model.train()
    return correct / len(test_dataset)
```

```python
epochs = 5
model.train()  # PCLayers must be in train mode while training (the trainer asserts this)

accs = [test_accuracy(model)]
for epoch in range(epochs):
    for img, label in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
        img = img.to(device)
        y = F.one_hot(label, num_classes=10).float().to(device)
        trainer.train_on_batch(
            inputs=(img, y),
            loss_fn=loss_fn,
            loss_fn_kwargs={"_y": y, "_img": img, "alpha_up": alpha_up, "alpha_down": alpha_down},
            is_log_progress=False,
            is_return_results_every_t=False,
            is_checking_after_callback_after_t=False,
        )
    accs.append(test_accuracy(model))
    print(f"epoch {epoch + 1}: test accuracy {accs[-1]:.4f}")

plt.plot(accs, marker="o")
plt.xlabel("epoch"); plt.ylabel("test accuracy"); plt.grid(alpha=0.3)
plt.show()
```

**Checkpoint:** accuracy reaches about **0.85–0.88 after the first epoch**. After that
it **fluctuates rather than climbing steadily**. In two runs with different seeds we
saw anywhere between 0.83 and 0.92 over epochs 2–10. With `SEED = 42` the five
epochs give 0.848, 0.871, 0.867, 0.865 and 0.863. If you train the same network
with notebook 5's `alpha_down = 1e-4`, you get about **0.97** after just two epochs.

Five epochs is enough here: the generator is what we care about, and more epochs did
not improve accuracy. More epochs also make the weights larger, which forces a smaller
inference step later (Section 8).

The accuracy gap is a real trade-off. Every unit of energy the network spends on predicting the
image is energy it doesn't spend on the label. Put this plot in your notes; it's a nice
figure for the thesis.

---

## 6. Saving and loading the weights

```python
def weights_only(state_dict):
    """Drop the PCLayer latents (keys ending in '._x') from a state_dict."""
    return {k: v for k, v in state_dict.items() if not k.endswith("._x")}

ckpt_path = RUN_DIR / f"bpc_up{alpha_up}_down{alpha_down}.pt"
torch.save(weights_only(model.state_dict()), ckpt_path)
print("saved", ckpt_path)
```

**Why `weights_only`?** `PCLayer` stores its latent `_x` as an `nn.Parameter`. PyTorch
therefore includes it in `state_dict()`, with the shape of the **last training batch**
(500 × 256). It is not a weight. It is the network's "current thought" about the last
batch. Two things happen if you keep it:

- `load_state_dict` into a fresh model fails with `Unexpected key(s) ... "pcs.0._x"`.
- Print `model.state_dict().keys()` and you'll see **six** `_x` entries, even though there are only two latent layers. Each shared layer is reachable under three names: `pcs.0`, `model_up.1` and `model_down.4`.

Now load the weights into a **fresh** model, which we'll use only for generation. This
is a check that the checkpoint is self-contained. It also means the generation code
never inherits leftover state from training.

```python
gen_model = BidirectionalPC(alpha_up, alpha_down).to(device)
print(gen_model.load_state_dict(torch.load(ckpt_path)))

# Freeze the weights: during generation only the latents and the image change.
for p in gen_model.parameters():
    p.requires_grad_(False)
```

**Checkpoint:** `load_state_dict` prints `<All keys matched successfully>`.

---

## 7. Generation as inference

We write the inference loop **by hand** instead of using `PCTrainer`. The loop itself
is about fifteen lines, and every line corresponds to something in Section 1. Before it
come two choices about **where inference starts**:

- `image_init`: the image starts either as notebook 5's one-pass picture of the first
  label (`"topdown"`) or as uniform noise (`"noise"`).
- `latent_init`: the latents start either as top-down predictions from the label
  (`"topdown"`) or as bottom-up predictions from the starting image (`"bottomup"`).

With `"noise"` and `"bottomup"`, inference starts from a state that knows nothing about
the label. Everything that turns the noise into a digit comes from the clamp, through
inference. That's the setting we use in this section: generation from scratch. Starting
from noise, a digit takes a few thousand steps to form, so phase A gets 6,000 steps
before the swap.

We generate all ten digits at once, as a batch of 10. Each row of the batch is
independent, because the energy is a sum over rows. The **label schedule** is a list of
phases `(labels, n_steps)`, so A → B → C runs are easy later.

```python
def one_hot(digits):
    return F.one_hot(torch.as_tensor(digits), num_classes=10).float().to(device)


def generate(model, schedule, lr_x=0.02, image_init="topdown", latent_init="topdown",
             record_every=40):
    """
    Run PC inference with the label clamped and the image free.

    schedule:    list of (digits, n_steps), e.g. [([0..9], 300), ([1..9,0], 10000)]
    image_init:  "topdown"  -> image starts as the top-down picture of the first label
                 "noise"    -> image starts as uniform noise
    latent_init: "topdown"  -> latents start as top-down predictions from the first label
                 "bottomup" -> latents start as bottom-up predictions from the starting image
    Returns frames (list of (batch, 784) tensors), the clamped label of each frame,
    and the energy at every step.
    """
    model.train()  # PCLayers must use their latents _x, not be transparent
    y0 = one_hot(schedule[0][0])
    batch = y0.shape[0]

    # (a) Where the IMAGE starts. It becomes a free variable: an nn.Parameter
    #     that the optimiser can update.
    if image_init == "topdown":
        model.eval()                          # transparent PCLayers: a plain top-down pass
        with torch.no_grad():
            img0 = model.model_down(y0)       # notebook 5's picture of the first label
        model.train()
    elif image_init == "noise":
        img0 = torch.rand(batch, 784, device=device)
    else:
        raise ValueError(f"unknown image_init: {image_init}")
    img = nn.Parameter(img0.clone())

    # (b) Where the LATENTS start: ONE forward pass with sampling switched on.
    #     set_is_sample_x(True) makes each PCLayer set _x := mu on the next forward call.
    #     This comes after (a) because bottom-up latents are computed FROM the image.
    for p in model.pcs:
        p.set_is_sample_x(True)
    if latent_init == "topdown":
        model.model_down(y0)                  # x1, x0 := predictions from the label
    elif latent_init == "bottomup":
        model.model_up(img)                   # x0, x1 := predictions from the starting image
    else:
        raise ValueError(f"unknown latent_init: {latent_init}")

    # (c) Inference optimiser: plain gradient descent on x0, x1 and u. The weights are frozen.
    opt = optim.SGD([p._x for p in model.pcs] + [img], lr=lr_x)

    frames, frame_labels, energies = [img.detach().clone()], [schedule[0][0]], []
    step = 0
    for digits, n_steps in schedule:
        y = one_hot(digits)  # <- the clamp. Changing this is the "swap".
        for _ in range(n_steps):
            opt.zero_grad()
            out_up, out_down = model((img, y))
            E = sum(p._energy for p in model.pcs) + loss_fn(
                (out_up, out_down), y, img, model.alpha_up, model.alpha_down)
            E.backward()   # dE/dx0, dE/dx1, dE/du  (the gradient from Section 1)
            opt.step()     # x <- x - lr * dE/dx
            energies.append(E.item())
            step += 1
            if step % record_every == 0:
                frames.append(img.detach().clone())
                frame_labels.append(digits)
    return frames, frame_labels, energies
```

Read it against the equations:

- `model((img, y))` computes all six prediction errors.
- `E` is exactly the $E$ from Section 1.
- `E.backward()` computes $\partial E / \partial u$ and the corresponding gradients for $x_0$ and $x_1$.
- `opt.step()` moves *only* the variables we gave to the optimiser.
- Changing `y` between phases is the unclamp/clamp.
- The image is set up **before** the latents, because `"bottomup"` latents are computed
  from the image.

Now run the morph: each digit $d$ turns into $d+1$ (and 9 turns into 0).

```python
record_every = 40
A = list(range(10))
B = [(d + 1) % 10 for d in A]
schedule = [(A, 6_000), (B, 10_000)]   # from noise, A needs ~6000 steps to form before the swap

frames, frame_labels, energies = generate(gen_model, schedule, lr_x=0.02, image_init="noise",
                                          latent_init="bottomup", record_every=record_every)
print(len(frames), "frames")

plt.semilogy(energies)
plt.axvline(schedule[0][1], color="r", ls="--", label="label swap")
plt.xlabel("inference step"); plt.ylabel("energy E"); plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

**Checkpoint:** 401 frames: the initial image plus one every 40 of the 16,000 steps. In the energy plot (log scale):

- The energy starts **high**, around 950: the noise image agrees with nothing. It falls
  fast at first, then slowly, to about 0.6 by step 6,000.
- At step 6,000 there is a **jump**, to about 15: the new clamp contradicts everything below it.
- After the jump the energy **decreases smoothly** again, to about 0.2 at step 16,000.

If the curve goes *up* before the swap, or oscillates, read the next section.

---

## 8. Why the inference learning rate matters

Inference is gradient descent on $E$. Near a minimum, $E$ is approximately quadratic,
with Hessian $H$. Gradient descent with step $\eta$ on a quadratic is stable only if

$$\eta < \frac{2}{\lambda_{\max}(H)}$$

which is the same condition as for an explicit Euler integrator, if you've seen ODEs.

You can estimate $\lambda_{\max}$ from the weights. Take the latent $x_1$. Its terms in
$E$ give a Hessian of roughly $\alpha_\uparrow I + \alpha_\downarrow I + \alpha_\uparrow W_3^\top W_3 + \alpha_\downarrow V_2^\top V_2$
(ReLU derivatives are at most 1). Each $W^\top W$ contributes at most $\sigma_{\max}(W)^2$,
its largest singular value squared:

```python
W = dict(gen_model.named_parameters())
for name in ["model_up.0.weight", "model_up.3.weight", "model_up.6.weight",
             "model_down.0.weight", "model_down.3.weight", "model_down.6.weight"]:
    s = torch.linalg.matrix_norm(W[name], ord=2).item()
    print(f"{name:22s} sigma_max^2 = {s * s:6.1f}")
```

In the 5-epoch run with `SEED = 42`, `model_down.3.weight` ($V_2$) had $\sigma^2 \approx 38.7$ and
`model_up.6.weight` ($W_3$) had $\sigma^2 \approx 11.8$. So for $x_1$:

$$\lambda_{\max} \lesssim 1 + 1 + 11.8 + 38.7 = 52.5 \quad\Rightarrow\quad \eta < 2/52.5 \approx 0.038$$

The other layers give looser limits; check this yourself with the same reasoning for
$x_0$ and $u$.

In a 10-epoch run the same weights had grown to $\sigma^2 \approx 80$ and $19$, giving
$\eta \lesssim 0.02$. The bound is **pessimistic**: it adds up norms and assumes every
ReLU is active, so `lr_x=0.02` still worked in both runs. But the larger the weights,
the closer you are to the edge. Always print these numbers for *your* model before
choosing `lr_x`.

**Experiment:** rerun `generate` with `lr_x=0.05`. The energy *increases* even before
the swap, and the digits get messy. With `lr_x=0.01` it's stable but about twice as
slow, so the morph is less complete after the same 10,000 steps.

> This is a general PC lesson: inference is a dynamical system, and its step size is
> limited by the **stiffest layer**, the one with the largest weights.

---

## 9. Looking at the result

A grid of snapshots: one column per digit, one row per moment in time. Frame times are
spaced geometrically: densely at the start, where the noise turns into digits, and
sparsely later. With the swap at step 6,000, only the last row comes after it; the GIF
in Section 10 shows the morph itself.

```python
def show_snapshots(frames, frame_labels, record_every, n_rows=8):
    idx = np.unique(np.geomspace(1, len(frames) - 1, n_rows - 1).astype(int))
    idx = [0] + list(idx)
    fig, axes = plt.subplots(len(idx), 10, figsize=(10, 1.1 * len(idx)))
    for r, k in enumerate(idx):
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(frames[k][c].view(28, 28).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{frame_labels[0][c]}→{frame_labels[-1][c]}", fontsize=8)
        axes[r, 0].set_ylabel(f"t={k * record_every}", fontsize=7)
    plt.tight_layout(); plt.show()

show_snapshots(frames, frame_labels, record_every)
```

Use the network's own classifier to check what it "sees" at the end of each phase:

```python
def classify(model, images):
    """Feed-forward classification of a (batch, 784) tensor; returns a list of labels."""
    model.eval()   # transparent PCLayers -> feed-forward classifier
    with torch.no_grad():
        pred = model.model_up(images).argmax(1).tolist()
    model.train()  # back to latent mode
    return pred

before = classify(gen_model, frames[schedule[0][1] // record_every])  # last frame before the swap
after = classify(gen_model, frames[-1])
print("before swap:", before)
print("after  swap:", after)
```

To see *when* each image changes class, classify every 10th frame and flag the changes:

```python
# Print the predicted classes every 10 frames and flag the moments where any of them changes
prev = None
for i in range(0, len(frames), 10):
    pred = classify(gen_model, frames[i])
    flag = "  <- changed" if prev is not None and pred != prev else ""
    print(f"step {i * record_every:6d}: {pred}{flag}")
    prev = pred
```

**Checkpoint:**

- In the grid, the first row is noise. By $t \approx 300$–$800$ it has become blurry blobs on a dark
  background, and by $t \approx 6000$ there are recognisable digits 0–9. The last row shows
  mostly clean 1, 2, …, 9, 0.
- Some digits keep a faint "ghost" of the old shape; in our run 0→1 was the most visible.
  Add steps to the second phase if you want them to finish.
- The classifier prints `[0, 1, ..., 9]` before the swap and `[1, 2, ..., 9, 0]` after.
- The change tracker shows the A digits settling by step 1,200, long before they look
  clean. After the swap, the B labels take over between steps 6,800 and 7,600.

---

## 10. Exporting the video

Matplotlib can write a GIF using Pillow, which is already installed. MP4 would need
`ffmpeg`, which this environment doesn't have.

```python
def save_gif(frames, frame_labels, path, record_every, fps=20):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1.7))
    ims, titles = [], []
    for c, ax in enumerate(axes):
        ims.append(ax.imshow(frames[0][c].view(28, 28).clamp(0, 1), cmap="gray", vmin=0, vmax=1))
        titles.append(ax.set_title("", fontsize=9))
        ax.axis("off")
    suptitle = fig.suptitle("")

    def update(k):
        for c in range(10):
            ims[c].set_data(frames[k][c].view(28, 28).clamp(0, 1))
            titles[c].set_text(f"clamped: {frame_labels[k][c]}")
        suptitle.set_text(f"step {k * record_every}")
        return ims

    FuncAnimation(fig, update, frames=len(frames)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)

gif_path = RUN_DIR / "morph_d_to_d+1.gif"
save_gif(frames, frame_labels, gif_path, record_every)
print("saved", gif_path)
```

**Checkpoint:** `RELAX/runs/mnist_bpc_generation/morph_d_to_d+1.gif` is about 16 MB,
and about 20 s long at 20 fps. Each panel is titled with the label currently clamped on
it. Open it in any image viewer or browser.

`record_every` sets the trade-off: 80 halves the frames and the file size, and 20 doubles
both, making the video smoother. The early noise frames compress badly, which is why
this GIF is large.

---

## 11. Experiments to try

Each experiment is only a change to the arguments of `generate` or of training. Write
down what you expect *before* you run it.

1. **Where inference starts.** Rerun Section 7 with the other combinations of
   `image_init` and `latent_init`. With `"topdown"` for both, the image starts as
   notebook 5's picture of A and the latents already agree with it, so the energy
   starts low. With `image_init="noise"` and `latent_init="topdown"`, the latents know
   the label but the image doesn't. Why does the energy start in the thousands? Which
   combination forms a clean A fastest?

2. **The `alpha_down = 1e-4` network (notebook 5's setting).** Retrain with
   `alpha_down=1e-4` (2 epochs is enough to see the effect), save it under a different
   name, and run the same morph. The classifier *will* report the new label after the
   swap, but the images **never turn into the new digit**. They dissolve into
   structured noise that the bottom-up path reads as B, which is an adversarial example
   produced by inference. The top-down pathway is too weak to hold the image on the
   "manifold" of digits. This is a strong argument for why the generative weight
   matters.

3. **Chains.** `schedule = [(A, 6000), (B, 5000), ([(d + 2) % 10 for d in A], 5000)]`.
   Does the second morph look like the first? Is the order of intermediate shapes
   different? The notebook's last cells (`make_single_digit_gif`) run a chain like
   `[1, 8, 4]` on a single image and save it as a GIF next to its energy curve.

4. **Swap back.** A → B → A. Does the image return to the original A, or to a different
   one? Inference finds *a* local minimum, not *the* minimum.

5. **Hard pairs.** Put the same start digit in every row, e.g. all 4s, and swap to 0–9.
   Which targets are easy (4→9) and which are hard (4→0)? Compare with the energy curves.

6. **Soft clamps.** Clamp a *mixture*, e.g. `0.5*one_hot(3) + 0.5*one_hot(8)`. What
   does the network generate for "half a 3, half an 8"?

---

## 12. Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: PCLayer behaves differently in train and eval modes` | `train_on_batch` called with the model in eval mode | `model.train()` before training (and after any `test_accuracy`) |
| `Unexpected key(s) in state_dict: "pcs.0._x", ...` | latents saved with the weights | save with `weights_only(...)` (Section 6) |
| Generated images don't change at all | model in `eval()` mode → `PCLayer`s are transparent and there are no latents | `model.train()` inside `generate` (already done) |
| Energy rises before the swap / images get noisy | `lr_x` above the stability limit | lower `lr_x` (Section 8) |
| Morph is very slow / incomplete | too few steps: error must travel `y → x1 → x0 → u` | more steps in the second phase, or a slightly larger `lr_x` (staying stable) |
| `RuntimeWarning: The <self._x> has not been initialized yet` | forward pass before `set_is_sample_x(True)` | harmless if it appears once; the layer initialises itself |
| Notebook memory grows a lot | storing a frame at every step (10,000 × 10 × 784 floats ≈ 300 MB) | keep `record_every` ≥ 10 |
