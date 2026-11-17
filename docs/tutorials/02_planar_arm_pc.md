# Tutorial 02 — Teaching a predictive coding network to move a planar arm

**Goal.** Train a predictive coding (PC) network that predicts where the tip of a
two-joint arm ends up for given joint angles. Then **run the network backwards**:
clamp a target position at its output and let the joint angles settle by inference.
The arm reaches the target without ever using an inverse-kinematics formula. We record
the inference and watch the arm move.

**What you will write:** one notebook, `RELAX/notebooks/planar_arm_pc.ipynb`. Each
`python` block below is meant to be one notebook cell. Type them yourself rather than
copy-pasting, and run each cell before moving on. Every section ends with a
**Checkpoint** describing what you should see.

**Time:** about 3 minutes of compute on a CPU (training is about 70 s;
each reaching run is about 10 s).

**Relation to the experiments in `planar_arm/`.** This tutorial is a short, readable
version of experiments PC-01 (learning the arm) and PC-02 (inferring the angles), whose
reports are `docs/pc01.md` and `docs/pc02.md`. Those scripts add a validation split,
checkpoint selection, provenance records and many more diagnostics. Here we keep only
what is needed to understand the idea and to see it. The numbers differ slightly from
the reports because the data is drawn differently.

Contents:

1. [The idea in equations](#1-the-idea-in-equations)
2. [Setup](#2-setup)
3. [The arm](#3-the-arm)
4. [Data](#4-data)
5. [The model](#5-the-model)
6. [Training](#6-training)
7. [What did the network learn?](#7-what-did-the-network-learn)
8. [Saving and loading the weights](#8-saving-and-loading-the-weights)
9. [Reaching as inference](#9-reaching-as-inference)
10. [Watching the arm move](#10-watching-the-arm-move)
11. [Many starts, many answers](#11-many-starts-many-answers)
12. [Experiments to try](#12-experiments-to-try)
13. [Gotchas](#13-gotchas)

---

## 1. The idea in equations

### The arm

The arm has two segments of lengths $L_1 = L_2 = 1$. The shoulder is fixed at the origin.
The joint angles are $q = (\theta_1, \theta_2)$, in radians: $\theta_1$ is measured
from the $x$ axis and $\theta_2$ is measured **relative to the first segment**. The tip
position follows from the **forward kinematics**:

$$
x = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2), \qquad
y = L_1\sin\theta_1 + L_2\sin(\theta_1+\theta_2)
$$

Going forwards (angles → tip) is easy and has exactly one answer. Going backwards
(tip → angles) is the **inverse kinematics** problem. It usually has **two** answers
(two postures reach the same point), none for points outside the disk of radius
$L_1 + L_2$, and infinitely many at the origin. For two segments there is a closed-form
formula, but for a real body there isn't. The project requirement is to reach targets
**without** using such a formula (requirement D.B.2).

### The network

Notation:

| Symbol | Meaning | Size |
|---|---|---|
| $q$ | joint angles $(\theta_1, \theta_2)$ | 2 |
| $y$ | tip position $(x, y)$ | 2 |
| $z_1$, $z_2$ | the two hidden layers of latent activity (the `PCLayer`s) | 32 each |
| $f$ | tanh | |

The network predicts the tip from the angles, layer by layer:

$$
\mu_1 = W_1 q + b_1, \qquad \mu_2 = W_2 f(z_1) + b_2, \qquad \hat y = W_3 f(z_2) + b_3
$$

Read it as a **generative model**: the angles are the hidden *cause*, the tip position
is the *sensation* that cause produces. This is the same direction as in the first
chapter's example $x_0 \to x_1 \to x_2$.

### The energy

$$
E = \tfrac12\|z_1-\mu_1\|^2 + \tfrac12\|z_2-\mu_2\|^2 + \tfrac12\|\hat y - y\|^2
$$

This is the free energy of the chapter with every precision $\pi_l$ set to 1. Each term
is **local**: it involves only one layer and the layer just below it.

### One energy, two uses

The same energy is used for learning and for reaching. What changes is **which variables
are clamped** (held fixed) and which are **free** (moved by gradient descent on $E$):

| Use | Clamped | Free during inference | Weights |
|---|---|---|---|
| Training | $q$ = angles, $y$ = true tip | $z_1, z_2$ | updated after inference |
| Testing | $q$ | none: one feed-forward pass (`eval()` mode) | frozen |
| **Reaching** | $y = y^\ast$, the **target** | $z_1, z_2$ **and $q$** | frozen |

In reaching, the angles stop being data and become **one more latent variable**. The
network answers the question: *"which posture would explain seeing the tip at $y^\ast$?"*
The gradients that drive inference are:

$$
\begin{aligned}
\frac{\partial E}{\partial z_2} &= (z_2-\mu_2) + f'(z_2)\odot W_3^\top(\hat y - y^\ast) \\
\frac{\partial E}{\partial z_1} &= (z_1-\mu_1) - f'(z_1)\odot W_2^\top(z_2-\mu_2) \\
\frac{\partial E}{\partial q} &= -W_1^\top(z_1-\mu_1)
\end{aligned}
$$

At the start, every latent equals its prediction, so the only nonzero error is at the
output. Each step moves the error **one layer down**: from $y^\ast$ to $z_2$, then to
$z_1$, and only then to $q$. That is why the angles barely move in the first steps, and
why reaching takes thousands of steps.

### What this is (and isn't) yet

This is **perception-style inference about the body**: the network infers the posture
that would put the tip at the target. In **active inference**, the next step of the
project, $y^\ast$ becomes a desired observation and the inferred $q$ becomes a
proprioceptive prediction that the motors carry out. In this tutorial the arm in the
pictures "moves" because we draw the network's current belief $q$ at each step.

---

## 2. Setup

Create a new notebook `RELAX/notebooks/planar_arm_pc.ipynb` and select the **relax**
kernel (interpreter `~/.conda/envs/relax/bin/python`).

```python
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display

import predictive_coding as pc

SEED = 42
rng = np.random.default_rng(SEED)   # NumPy: data and starting angles
torch.manual_seed(SEED)             # PyTorch: initial weights
torch.set_num_threads(1)            # a tiny network: extra threads only add overhead

# runs/ is listed in .gitignore, so nothing written here gets committed.
RUN_DIR = Path("../runs/planar_arm_tutorial")
RUN_DIR.mkdir(parents=True, exist_ok=True)

L1, L2 = 1.0, 1.0                   # segment lengths
```

**Checkpoint:** the cell runs without errors. If `import predictive_coding` fails, the
notebook is not using the `relax` environment, or the submodule
`third_party/PredictiveCoding` is empty (`git submodule update --init`).

---

## 3. The arm

Three small functions: the forward kinematics from Section 1, and two drawing helpers
that we reuse in every figure.

```python
def forward_kinematics(theta1, theta2):
    """Joint angles (radians) -> (elbow, tip) positions. Works on numbers and on arrays."""
    elbow = np.stack([L1 * np.cos(theta1), L1 * np.sin(theta1)], axis=-1)
    tip = elbow + np.stack([L2 * np.cos(theta1 + theta2), L2 * np.sin(theta1 + theta2)], axis=-1)
    return elbow, tip


def draw_arm(ax, theta1, theta2, color="C0", alpha=1.0, label=None):
    """Draw the two segments: shoulder (origin) -> elbow -> tip."""
    elbow, tip = forward_kinematics(theta1, theta2)
    ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], "-o",
            color=color, lw=3, ms=5, alpha=alpha, label=label)


def arm_axes(ax, title=""):
    """Square axes with the reachable disk (radius L1 + L2) drawn dashed."""
    ax.add_patch(plt.Circle((0, 0), L1 + L2, fill=False, ls="--", color="gray"))
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

arm_axes(axes[0], "a few postures")
for t1, t2, color in [(0.3, 0.8, "C0"), (2.0, -1.2, "C1"), (-1.0, 2.5, "C2")]:
    draw_arm(axes[0], t1, t2, color=color, label=f"θ1={t1}, θ2={t2}")
axes[0].legend(fontsize=8)

# With L1 = L2, the postures (θ1, θ2) and (θ1 + θ2, -θ2) put the tip in the same place.
t1, t2 = 0.2, 1.4
arm_axes(axes[1], "one tip, two postures")
draw_arm(axes[1], t1, t2, color="C0", label=f"({t1}, {t2})")
draw_arm(axes[1], t1 + t2, -t2, color="C3", label=f"({t1 + t2:.1f}, {-t2})")
axes[1].legend(fontsize=8)
plt.show()

print("tips:", forward_kinematics(t1, t2)[1], forward_kinematics(t1 + t2, -t2)[1])
```

**Checkpoint:** the right panel shows two different arms whose tips touch, and the two
printed tips are equal. That is the ambiguity of the inverse problem: when we later ask
the network "which angles reach this point?", there are two right answers.

---

## 4. Data

We draw 10,000 random postures, with both angles uniform in $[-\pi, \pi)$, and compute
their tips. The labels come from the forward kinematics, which is the only place it is
used for learning. The last 1,000 samples are held out for testing.

```python
N = 10_000
angles = rng.uniform(-np.pi, np.pi, size=(N, 2))          # (θ1, θ2)
_, tips = forward_kinematics(angles[:, 0], angles[:, 1])   # (x, y)

n_train = 9_000
q_train = torch.tensor(angles[:n_train], dtype=torch.float32)
xy_train = torch.tensor(tips[:n_train], dtype=torch.float32)
q_test = torch.tensor(angles[n_train:], dtype=torch.float32)
xy_test = torch.tensor(tips[n_train:], dtype=torch.float32)

loader = DataLoader(TensorDataset(q_train, xy_train), batch_size=128, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))
print(q_train.shape, xy_train.shape, q_test.shape, xy_test.shape)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
color = angles[:3000, 1]   # colour every sample by its elbow angle θ2

axes[0].scatter(angles[:3000, 0], angles[:3000, 1], s=3, c=color, cmap="twilight")
axes[0].set(title="angle space (uniform)", xlabel="θ1", ylabel="θ2", aspect="equal")

arm_axes(axes[1], "tip positions (not uniform)")
axes[1].scatter(tips[:3000, 0], tips[:3000, 1], s=3, c=color, cmap="twilight")
plt.show()
```

**Checkpoint:** prints `torch.Size([9000, 2])` twice (training angles and tips) and
`torch.Size([1000, 2])` twice (test angles and tips). In the plots:

- The angles fill the square evenly.
- The tips fill the disk **unevenly**: they bunch up near the rim (arm stretched,
  $\theta_2 \approx 0$) and near the centre (arm folded, $\theta_2 \approx \pm\pi$).
  The distance to the shoulder depends only on $\theta_2$:
  $r^2 = L_1^2 + L_2^2 + 2L_1L_2\cos\theta_2$. That is why each ring has one colour.

---

## 5. The model

The network from Section 1: two hidden layers of 32 units, each followed by a `PCLayer`
that holds the latent $z$.

```python
class ArmNet(nn.Module):
    """Predicts the tip position (x, y) from the joint angles (θ1, θ2)."""

    def __init__(self, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden), pc.PCLayer(is_keep_energy_per_datapoint=True), nn.Tanh(),
            nn.Linear(hidden, hidden), pc.PCLayer(is_keep_energy_per_datapoint=True), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.pcs = [self.layers[1], self.layers[4]]   # the two PCLayers: z1 and z2

    def forward(self, q):
        return self.layers(q)


def output_energy(output, target):
    """The output term of the energy, 0.5 * ||y_hat - y||^2, summed over the batch."""
    return 0.5 * (output - target).pow(2).sum()


model = ArmNet()
print(model)
print("weights:", sum(p.numel() for p in model.parameters()))
```

What to understand here:

- **How a `PCLayer` behaves** (see `third_party/PredictiveCoding/predictive_coding/pc_layer.py`, method `forward`):
  - In `train()` mode it **ignores its input $\mu$ and returns its own latent $z$** (stored as `_x`). It also stores the energy $\tfrac12\|z-\mu\|^2$.
  - In `eval()` mode it is transparent: it returns $\mu$ unchanged, so the whole model becomes an ordinary feed-forward network.
- **`is_keep_energy_per_datapoint=True`** makes each layer also keep its energy **per sample**. Training doesn't need it. We use it in Section 9 to plot one energy curve per target.
- **`self.pcs` is a plain list** of two modules that are already inside `self.layers`. It is only a shortcut so we can write `model.pcs` instead of `model.layers[1]` and `model.layers[4]`.

**Checkpoint:** the printout shows `Linear → PCLayer → Tanh` twice and a final `Linear`.
The weight count is **1,218**. Check it by hand: $(2\cdot32+32) + (32\cdot32+32) + (32\cdot2+2)$.

---

## 6. Training

For every batch, `PCTrainer.train_on_batch` does the following:

1. **Initialise the latents** with a feed-forward pass: $z_1 := \mu_1$, $z_2 := \mu_2$. At this moment the hidden errors are zero, and the whole error sits at the output.
2. Run **29 inference steps** with the angles and the true tips clamped. Each step is an SGD step on $z_1, z_2$ only. The output error spreads into the hidden layers.
3. Take **one weight step** (Adam) on the energy of the relaxed network.

This is exactly the schedule of experiment PC-01.

```python
T = 30   # steps per batch: 29 inference steps (update_x_at) + 1 weight step (update_p_at)

trainer = pc.PCTrainer(
    model,
    T=T,
    optimizer_x_fn=optim.SGD, optimizer_x_kwargs={"lr": 0.1},      # inference: z1, z2
    update_x_at=list(range(T - 1)),                                # steps 0..28
    optimizer_p_fn=optim.Adam, optimizer_p_kwargs={"lr": 1e-3},    # learning: W, b
    update_p_at="last",                                            # step 29
    update_p_at_early_stop=False,
    plot_progress_at=[],
)


def predict(model, q):
    """One feed-forward pass (PCLayers transparent in eval mode). Returns a NumPy array."""
    model.eval()
    with torch.no_grad():
        out = model(q).numpy()
    model.train()
    return out


def rmse(model, q, xy):
    """Root mean squared distance between predicted and true tips."""
    dist = np.linalg.norm(predict(model, q) - xy.numpy(), axis=1)
    return float(np.sqrt(np.mean(dist ** 2)))
```

The training loop also keeps three things for the pictures of Section 7:

- the RMSE after every epoch;
- the network's predictions on the test set at a few chosen epochs;
- the energy at **every** inference step of the very first batch.

Epoch 0 means "before any training".

```python
epochs = 60
snapshot_epochs = [0, 1, 3, 10, 60]
snapshots = {}
history = {"train": [], "test": []}
first_batch_trace = None

model.train()   # PCLayers must be in train mode while training (the trainer asserts this)
for epoch in range(epochs + 1):
    if epoch > 0:
        for i, (q_batch, xy_batch) in enumerate(loader):
            record = (epoch == 1 and i == 0)   # keep every inference step, only once
            result = trainer.train_on_batch(
                inputs=q_batch,
                loss_fn=output_energy,
                loss_fn_kwargs={"target": xy_batch},
                is_return_results_every_t=record,
            )
            if record:
                first_batch_trace = result

    history["train"].append(rmse(model, q_train, xy_train))
    history["test"].append(rmse(model, q_test, xy_test))
    if epoch in snapshot_epochs:
        snapshots[epoch] = predict(model, q_test)
    if epoch % 10 == 0:
        print(f"epoch {epoch:2d}: train RMSE {history['train'][-1]:.3f}   test RMSE {history['test'][-1]:.3f}")
```

```python
plt.semilogy(history["train"], label="train")
plt.semilogy(history["test"], label="test")
plt.xlabel("epoch")
plt.ylabel("RMSE of the tip position")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
```

**Checkpoint:** the test RMSE drops from about 1.58 (untrained) to about
**0.102** after 60 epochs, with train and test curves on top of each other. That
is an average miss of about a tenth of a segment length. It is the network's
*approximation error*, and it limits how precisely the arm can reach later (Section 9).
PC-01 reports 0.113 with the same settings on its own data split. The curve is still
going down at epoch 60, so more epochs would help.

---

## 7. What did the network learn?

### The workspace, epoch by epoch

Feed the test angles through the network and scatter the **predicted** tips, using the
same colours as in Section 4. The last panel is the truth.

```python
fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(3.2 * (len(snapshots) + 1), 3.6))
color = angles[n_train:, 1]
for ax, (epoch, pred) in zip(axes, snapshots.items()):
    arm_axes(ax, f"epoch {epoch}")
    ax.scatter(pred[:, 0], pred[:, 1], s=3, c=color, cmap="twilight")
arm_axes(axes[-1], "truth")
axes[-1].scatter(tips[n_train:, 0], tips[n_train:, 1], s=3, c=color, cmap="twilight")
plt.tight_layout()
plt.show()
```

**Checkpoint:** at epoch 0 all predictions sit on a short curve near the origin: an
untrained network gives almost the same answer for every input. At epoch 1 they form a
twisted band, and at epoch 3 a folded ring. By epoch 10 the disk is in place but the
colours are still mixed up. By epoch 60 the picture looks like the truth: dark (stretched
arm) at the rim, light (folded arm) in the centre.

### Where is it wrong?

Evaluate the network on a dense grid of angles and colour each point by the distance
between the predicted and the true tip.

```python
grid = np.linspace(-np.pi, np.pi, 201)
T1, T2 = np.meshgrid(grid, grid)
q_grid = np.stack([T1.ravel(), T2.ravel()], axis=1)

_, true_grid = forward_kinematics(q_grid[:, 0], q_grid[:, 1])
pred_grid = predict(model, torch.tensor(q_grid, dtype=torch.float32))
error_map = np.linalg.norm(pred_grid - true_grid, axis=1).reshape(T1.shape)

plt.imshow(error_map, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], cmap="magma")
plt.colorbar(label="tip error")
plt.xlabel("θ1")
plt.ylabel("θ2")
plt.title("prediction error over angle space")
plt.show()
```

**Checkpoint:** most of the square is dark, with soft blobs of error up to about 0.2:
the network is a smooth approximation, good on average but not uniformly. The worst
errors, around 0.6, are in the **corners**, where both angles are near $\pm\pi$.
For the real arm, $\theta = \pi$ and $\theta = -\pi$ are the same posture, but for the
network they are inputs $2\pi$ apart, at opposite edges of its training range. It has to
learn a function that wraps around, and it does worst where both angles wrap at once.
PC-01 measures the same effect: RMSE 0.167 when an angle is within 0.2 rad of $\pm\pi$,
against 0.103 elsewhere. A $(\cos\theta, \sin\theta)$ input encoding would remove the
seam; that's Experiment 6.

### Inside one training step

The energy at every inference step of the first batch, per sample:

```python
steps = np.arange(T)
batch_size = 128
plt.plot(steps, np.array(first_batch_trace["overall"]) / batch_size, "k", lw=2, label="total E")
plt.plot(steps, np.array(first_batch_trace["loss"]) / batch_size, label="output: ½‖ŷ − y‖²")
plt.plot(steps, np.array(first_batch_trace["energy"]) / batch_size, label="hidden: ½‖z1 − μ1‖² + ½‖z2 − μ2‖²")
plt.xlabel("inference step t (weights fixed until the last one)")
plt.ylabel("energy per sample")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
```

**Checkpoint:** the hidden energy starts at **exactly 0**, because the latents start at
their predictions, and grows. The output energy falls. The total falls too, from about
1.28 to about 0.96. This is the error being *pushed into* the
hidden layers, where each layer can see its own share of it. The weight step at the end
uses exactly those local errors: $\Delta W_l \propto (z_l-\mu_l)\,f(z_{l-1})^\top$. No
error is propagated backwards through the whole network.

---

## 8. Saving and loading the weights

```python
def weights_only(state_dict):
    """Drop the PCLayer latents (keys ending in '._x'): they belong to the last batch, not to the model."""
    return {k: v for k, v in state_dict.items() if not k.endswith("._x")}


ckpt_path = RUN_DIR / "armnet.pt"
torch.save(weights_only(model.state_dict()), ckpt_path)

arm_model = ArmNet()
print(arm_model.load_state_dict(torch.load(ckpt_path)))

# Freeze the weights: while reaching, only the angles and the latents change.
for p in arm_model.parameters():
    p.requires_grad_(False)
```

**Checkpoint:** prints `<All keys matched successfully>`.

As in Tutorial 01, the latents `_x` are `nn.Parameter`s with the shape of the last
training batch. If you saved them, `load_state_dict` into a fresh model would fail with
`Unexpected key(s) ... "layers.1._x"`.

---

## 9. Reaching as inference

We write the inference loop **by hand** instead of using `PCTrainer`. It is short, and
every line corresponds to something in Section 1. We reach several targets at once, as
one batch. Each row of the batch is independent, because the energy is a sum over rows.

```python
def reach(model, targets, q_start, steps=20_000, lr_q=0.05, lr_z=0.1, record_every=50):
    """
    Clamp target tip positions at the output and let the angles q (and the latents) settle.

    targets: list of (x, y), one per arm
    q_start: one (θ1, θ2) shared by all arms, or a list with one per arm
    Returns a dict of NumPy arrays recorded every `record_every` steps.
    """
    model.train()                                   # PCLayers use their latents z
    target = torch.tensor(targets, dtype=torch.float32)
    q_start = np.broadcast_to(np.asarray(q_start, dtype=float), (len(targets), 2))

    # (a) The angles are a free variable: an nn.Parameter that the optimiser can move.
    q = nn.Parameter(torch.tensor(q_start, dtype=torch.float32))

    # (b) Initialise z1, z2 with one feed-forward pass from the starting angles (z := mu).
    for layer in model.pcs:
        layer.set_is_sample_x(True)
    model(q)
    z = [layer.get_x() for layer in model.pcs]

    # (c) Plain gradient descent, with one step size for the angles and one for the latents.
    opt = optim.SGD([{"params": [q], "lr": lr_q}, {"params": z, "lr": lr_z}])

    rec = {"step": [], "q": [], "y_relaxed": [], "E_hidden": [], "E_out": []}
    for step in range(steps + 1):
        opt.zero_grad()
        y_relaxed = model(q)                        # y_hat = W3 f(z2) + b3
        E = sum(layer.energy() for layer in model.pcs) + output_energy(y_relaxed, target)

        if step % record_every == 0:
            rec["step"].append(step)
            rec["q"].append(q.detach().numpy().copy())
            rec["y_relaxed"].append(y_relaxed.detach().numpy().copy())
            rec["E_hidden"].append(sum(layer.energy_per_datapoint() for layer in model.pcs).detach().numpy()[:, 0])
            rec["E_out"].append(0.5 * (y_relaxed - target).pow(2).sum(dim=1).detach().numpy())
        if step == steps:
            break

        E.backward()                                # dE/dq, dE/dz1, dE/dz2 (Section 1)
        opt.step()                                  # variable <- variable - lr * gradient
    return {k: np.array(v) for k, v in rec.items()}
```

Read it against the equations:

- `model(q)` computes the three prediction errors, and `E` is exactly the $E$ from Section 1, with $y = y^\ast$.
- `E.backward()` computes $\partial E/\partial q$, $\partial E/\partial z_1$ and $\partial E/\partial z_2$. Autograd only does the bookkeeping: each gradient involves one layer and its neighbours.
- `opt.step()` moves *only* `q` and the latents, because those are the only variables the optimiser was given. The weights are frozen.
- The analytic inverse kinematics appears **nowhere**. Forward kinematics is used only afterwards, to check where the real arm ended up.

Now reach six targets, all starting from the same posture:

```python
targets = [(1.0, 1.0), (-1.0, 1.0), (0.5, -1.5), (1.6, 0.2), (-1.2, -0.6), (0.0, 1.2)]
q_start = (0.5, 1.0)

rec = reach(arm_model, targets, q_start)

q_final = rec["q"][-1]
_, tip_real = forward_kinematics(q_final[:, 0], q_final[:, 1])            # where the arm really is
tip_net = predict(arm_model, torch.tensor(q_final, dtype=torch.float32))  # network's guess from q alone
tip_relaxed = rec["y_relaxed"][-1]                                        # output of the relaxed latents

print("target          real error   network error   relaxed error")
for i, t in enumerate(targets):
    print(f"{str(t):14s}  {np.linalg.norm(tip_real[i] - t):10.3f}   "
          f"{np.linalg.norm(tip_net[i] - t):13.3f}   {np.linalg.norm(tip_relaxed[i] - t):13.4f}")
```

Three different "tips", and why they differ:

| Name | Computed from | Meaning |
|---|---|---|
| **real** | `forward_kinematics(q)` | where the physical arm would actually be. **This is what counts.** |
| **network** | a feed-forward pass from `q` | where the network *believes* the tip is for those angles |
| **relaxed** | the output of the latents $z_2$ | what the network's internal state currently "sees" |

**Checkpoint:** five of the six arms end within 0.1 of their target; `(0.0, 1.2)` stops at about 0.16. Look at `(-1.2, -0.6)`: its network error is 0.000, so the network is completely satisfied, yet the real arm misses by 0.09. That remainder is purely the network's approximation error from Section 6. The relaxed error is always the smallest, because the
latents can bend to fit the target before the angles have caught up. A small relaxed
error therefore doesn't prove that the arm is in the right place. Experiment PC-02D
studies exactly this gap.

```python
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
_, tips_path = forward_kinematics(rec["q"][..., 0], rec["q"][..., 1])   # (frames, arms, 2)
for i, (ax, t) in enumerate(zip(axes.flat, targets)):
    arm_axes(ax, f"target {t}")
    draw_arm(ax, *rec["q"][0, i], color="gray", alpha=0.4, label="start")
    ax.plot(tips_path[:, i, 0], tips_path[:, i, 1], color="C0", lw=1, alpha=0.6)
    draw_arm(ax, *q_final[i], color="C0", label="final")
    ax.plot(*tip_relaxed[i], "o", color="C1", label="relaxed output")
    ax.plot(*t, "rx", ms=12, mew=2, label="target")
axes[0, 0].legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.show()
```

And how it gets there, step by step:

```python
_, tips_path = forward_kinematics(rec["q"][..., 0], rec["q"][..., 1])
real_error = np.linalg.norm(tips_path - np.array(targets), axis=2)   # (frames, arms)
energy = rec["E_hidden"] + rec["E_out"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, t in enumerate(targets):
    axes[0].semilogy(rec["step"], energy[:, i], label=str(t))
    axes[1].plot(rec["step"], real_error[:, i], label=str(t))
axes[0].set(xlabel="inference step", ylabel="energy E", title="energy (what inference minimises)")
axes[1].set(xlabel="inference step", ylabel="distance to target", title="real error (what we care about)")
axes[1].axhline(0.1, color="gray", ls="--")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.show()
```

**Checkpoint:** the energy falls by several orders of magnitude and never goes up. The
real error falls quickly in the first few thousand steps and then levels off, mostly
below the dashed 0.1 line. It cannot go much lower than the network's own error from
Section 6: inference makes the arm agree with the **learned** model, not with the real
arm.

---

## 10. Watching the arm move

Matplotlib can write a GIF with Pillow, which is already installed. Each frame draws
the current belief $q$ as an arm, the path of its tip, and the relaxed output (orange
dot).

```python
def save_reach_gif(rec, targets, path, fps=25):
    n = len(targets)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).flat
    elbows, tips_path = forward_kinematics(rec["q"][..., 0], rec["q"][..., 1])

    arms, trails, dots = [], [], []
    for ax, t in zip(axes, targets):
        arm_axes(ax)
        ax.plot(*t, "rx", ms=10, mew=2)
        trails.append(ax.plot([], [], color="C0", lw=1, alpha=0.5)[0])
        arms.append(ax.plot([], [], "-o", color="C0", lw=3, ms=5)[0])
        dots.append(ax.plot([], [], "o", color="C1", ms=6)[0])
    title = fig.suptitle("")
    fig.tight_layout()

    def update(k):
        for i in range(n):
            arms[i].set_data([0, elbows[k, i, 0], tips_path[k, i, 0]],
                             [0, elbows[k, i, 1], tips_path[k, i, 1]])
            trails[i].set_data(tips_path[:k + 1, i, 0], tips_path[:k + 1, i, 1])
            dots[i].set_data([rec["y_relaxed"][k, i, 0]], [rec["y_relaxed"][k, i, 1]])
        title.set_text(f"inference step {rec['step'][k]}")
        return arms + trails + dots

    FuncAnimation(fig, update, frames=len(rec["step"])).save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


gif_path = RUN_DIR / "reaching.gif"
save_reach_gif(rec, targets, gif_path)
display(Image(filename=str(gif_path)))
```

**Checkpoint:** an animation of 401 frames (about 16 s) appears
inline and is saved to `RELAX/runs/planar_arm_tutorial/reaching.gif` (about
2.5 MB). Watch two things:

- The orange dot jumps to the target almost immediately, but the arm follows slowly.
  That is the error travelling $y^\ast \to z_2 \to z_1 \to q$.
- The arm moves fast at first and then creeps. Near the end the remaining error is
  small, so the gradient, and therefore the step, is small too.

---

## 11. Many starts, many answers

Inference is gradient descent, so where it ends depends on where it starts. Reach **one**
target from twelve random starting postures. `reach` takes one start per arm, so we pass
the same target twelve times.

```python
target = (-1.2, -0.6)
n_starts = 12
starts = rng.uniform(-np.pi, np.pi, size=(n_starts, 2))
rec_many = reach(arm_model, [target] * n_starts, starts)

q_end = rec_many["q"][-1]
_, tip_end = forward_kinematics(q_end[:, 0], q_end[:, 1])
error_end = np.linalg.norm(tip_end - np.array(target), axis=1)
outside = np.any(np.abs(rec_many["q"]) > np.pi, axis=2).any(axis=0)   # ever left [-π, π]?

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
arm_axes(axes[0], f"12 starts, one target {target}")
for i in range(n_starts):
    ok = error_end[i] < 0.2
    draw_arm(axes[0], *q_end[i], color="C2" if ok else "C3", alpha=0.7)
axes[0].plot(*target, "kx", ms=14, mew=3)

for i in range(n_starts):
    axes[1].plot(rec_many["step"], rec_many["q"][:, i, 1], color="C3" if outside[i] else "C0", alpha=0.7)
axes[1].axhline(np.pi, color="gray", ls="--")
axes[1].axhline(-np.pi, color="gray", ls="--")
axes[1].set(xlabel="inference step", ylabel="θ2", title="elbow angle; red = left the trained range")
axes[1].grid(alpha=0.3)
plt.show()

print("final errors:", np.round(error_end, 3))
print("reached (< 0.2):", int(np.sum(error_end < 0.2)), "of", n_starts,
      "  left the trained range:", int(outside.sum()))
```

**Checkpoint:** 9 of the 12 runs end within 0.2 of the target, most of them at about 0.09–0.11, which is the network's own error near this point (compare the 0.091 in Section 9). The other 3 end between 1.2 and 2.0 away, and they are exactly the 3 runs that left the trained range.

What you see:

- **Both postures show up.** Successful runs (green arms) end with the elbow bent one way
  or the other ($\theta_2 \approx 1.7$ or $\theta_2 \approx -1.5$ on the right),
  the two solutions from Section 3, depending on the start. Nobody told the
  network that two answers exist; each run simply rolls downhill into the nearest one.
- **Some runs fail.** Failures (red arms) drift out of $[-\pi, \pi]$. The
  network never saw those angles during training and **extrapolates** badly there, so it
  can predict the target for a posture that is nowhere near it. Experiment PC-02E
  documents a case where more steps lower the energy *and* make the real error worse.
- **Low energy ≠ right answer.** The energy measures agreement with the *learned* model.
  Only the forward kinematics, which inference never sees, tells you where the arm really is.

---

## 12. Experiments to try

Each experiment is only a change to arguments or to one line. Write down what you expect
*before* you run it.

1. **Fewer steps.** `reach(arm_model, targets, q_start, steps=3000)`. This is the budget
   of experiment PC-02. How far from the targets are the arms? Compare with the
   real-error plot of Section 9.
2. **Step sizes.** Try `lr_q=0.2`, then `lr_q=0.01`, keeping `lr_z=0.1`. Then raise
   `lr_z`. When does the energy start to go *up*? (Tutorial 01, Section 8, explains why
   too large a step is unstable.)
3. **The non-PC baseline.** Optimise $q$ directly through the feed-forward network,
   with no latents: put the model in `eval()`, make `q` an `nn.Parameter`, and minimise
   `output_energy(arm_model(q), target)` with SGD. It is faster. Why? (Hint: there is
   no chain of local errors to wait for; autograd sends the output error straight to
   `q`. That is backprop, the thing PC avoids.)
4. **Unreachable target.** Add `(2.5, 0.0)` to `targets`. What posture does the arm
   choose, and what is the relaxed error?
5. **Moving target.** Reach a target for 10,000 steps, then change `target` and continue
   *from the current state* for 10,000 more. You'll need to split `reach` into an
   initialisation and a `for` loop you can call twice. This is the first step towards
   control: the arm tracks a goal that changes over time.
6. **A seamless input.** Retrain with the input $(\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2)$:
   change `nn.Linear(2, hidden)` to `nn.Linear(4, hidden)` and transform the angles
   before feeding them in. Does the bright seam of Section 7 go away? For reaching,
   keep `q` as the free variable and apply the transform inside `forward`.

---

## 13. Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: PCLayer behaves differently in train and eval modes` | `train_on_batch` called with the model in eval mode | `model.train()` before training (`predict` already restores it) |
| `Unexpected key(s) in state_dict: "layers.1._x", ...` | latents saved with the weights | save with `weights_only(...)` (Section 8) |
| `TypeError: output_energy() got an unexpected keyword argument` | the keys in `loss_fn_kwargs` must match the parameter names of the loss | use `{"target": xy_batch}` with `output_energy(output, target)` |
| The angles don't move at all in `reach` | model in `eval()` mode: no latents, no PC errors | `model.train()` inside `reach` (already done) |
| The angles don't move during the **first** steps | normal: the error needs two steps to travel from the output to `q` | wait; check `rec["q"]` after a few hundred steps |
| `RuntimeWarning: The <self._x> has not been initialized yet` | forward pass before `set_is_sample_x(True)` | harmless if it appears once; the layer initialises itself |
| Arm ends far from the target even though the energy is tiny | angles left $[-\pi, \pi]$ (extrapolation) or a poor local minimum | try another start; check `rec["q"]` against $\pm\pi$ (Section 11) |
| Training is much slower than expected | many CPU threads on a tiny network | keep `torch.set_num_threads(1)` |
