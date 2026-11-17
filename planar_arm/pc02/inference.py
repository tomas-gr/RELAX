"""Relajación mediante PCTrainer y referencia de optimización de entradas."""

import copy
from dataclasses import dataclass

import numpy as np
import predictive_coding as pc
import torch

from planar_arm import PlanarArm, normalize_angles
from planar_arm.pc01.model import output_energy


@dataclass
class InferenceConfig:
    """Presupuesto fijo; convergencia se evalúa al finalizar, no detiene antes.

    energy_tol y grad_tol se refieren al objetivo optimizado, no al éxito físico.
    physical_tol es una distancia cartesiana para evaluación únicamente.
    """

    steps: int = 3000
    lr_angles: float = 0.05
    lr_states: float = 0.1
    seed: int = 42
    energy_tol: float = 1e-6
    grad_tol: float = 1e-4
    physical_tol: float = 0.1

    def validate(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps debe ser entero positivo")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed debe ser entero no negativo")
        for name in ["lr_angles", "lr_states", "energy_tol", "grad_tol", "physical_tol"]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo y finito")


def infer(model, target, initial_angles=None, config=None, method="pc", arm=None):
    """Infiere un par angular en CPU float32 sin modificar el modelo recibido.

    method='pc': PCTrainer optimiza ángulos y los dos estados PCLayer; 'input':
    SGD optimiza solo ángulos sobre la red eval. Devuelve (resumen, historia).
    Historia incluye el estado inicial y cada iteración (steps+1 registros).
    No se usa cinemática exacta ni inversa en la energía. Ángulos libres sin wrap
    ni proyección: extrapolación fuera del dominio aprendido se informa.
    """
    config = InferenceConfig() if config is None else config
    config.validate()
    if method not in ("pc", "input"):
        raise ValueError("method debe ser pc o input")
    arm = PlanarArm() if arm is None else arm
    target_np = np.asarray(target, dtype=float)
    initial = (np.random.default_rng(config.seed).uniform(-np.pi, np.pi, 2)
               if initial_angles is None else np.asarray(initial_angles, dtype=float))
    for value in [target_np, initial]:
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("Objetivo e inicialización deben ser pares finitos")
    # Aislamiento: tampoco se alteran modos, gradientes ni latentes del llamador.
    local = copy.deepcopy(model).cpu()
    before = local.weights()
    for p in local.parameters():
        p.requires_grad_(False)
        p.grad = None
    target_t = torch.tensor(target_np[None], dtype=torch.float32)
    if not torch.isfinite(target_t).all():
        raise ValueError("Objetivo fuera del rango float32")
    pcs = [layer for layer in local.modules() if isinstance(layer, pc.PCLayer)]
    if method == "pc" and len(pcs) != 2:
        raise ValueError("PC-02 espera las dos PCLayer de PC-01")
    for layer in pcs:
        layer.is_holding_error = True
    history = {name: [] for name in ["angles", "states", "internal_residuals", "energy_terms",
                                     "relaxed_output", "gradient_norms"]}

    def record(q, output, variables):
        # Gradientes sin escalar; permiten distinguir conexión de gradiente nulo.
        grads = [v.grad for v in variables]
        if any(g is None for g in grads):
            raise RuntimeError("Una variable libre no recibe gradiente")
        terms = ([float(layer.energy().detach()) for layer in pcs] if method == "pc" else [0.0, 0.0])
        terms.append(float(output_energy(output, target_t).detach()))
        if not np.isfinite(terms).all() or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("Inferencia divergente: energía o gradientes no finitos")
        history["angles"].append(q.detach().numpy()[0].copy())
        history["states"].append(np.stack([p.get_x().detach().numpy()[0].copy() for p in pcs])
                                 if method == "pc" else np.empty((0, local.hidden)))
        history["internal_residuals"].append(np.stack([p.error.numpy()[0].copy() for p in pcs])
                                             if method == "pc" else np.empty((0, local.hidden)))
        history["energy_terms"].append(terms)
        history["relaxed_output"].append(output.detach().numpy()[0].copy())
        history["gradient_norms"].append([float(torch.linalg.vector_norm(g)) for g in grads])

    if method == "pc":
        local.train()
        # PCTrainer añade inputs al único grupo de estados. Escalar su gradiente
        # antes del paso SGD permite otra tasa angular sin tocar el submódulo.
        trainer = pc.PCTrainer(local, T=config.steps, update_x_at="all", update_p_at="never",
                               update_p_at_early_stop=False,
                               optimizer_x_fn=torch.optim.SGD,
                               optimizer_x_kwargs={"lr": config.lr_states}, plot_progress_at=[])
        current_output = []

        def loss_fn(output, target):
            current_output[:] = [output]
            return output_energy(output, target)

        def after_backward(t):
            q = trainer.inputs
            states = list(trainer.get_model_xs())
            record(q, current_output[0], [q, *states])
            q.grad.mul_(config.lr_angles / config.lr_states)

        trainer.train_on_batch(torch.tensor(initial[None], dtype=torch.float32),
                               loss_fn, {"target": target_t}, is_optimize_inputs=True,
                               callback_after_backward=after_backward)
        q = trainer.inputs
        variables = [q, *trainer.get_model_xs()]
    else:
        local.eval()
        q = torch.nn.Parameter(torch.tensor(initial[None], dtype=torch.float32))
        variables = [q]
        optimizer = torch.optim.SGD(variables, lr=config.lr_angles)
        for _ in range(config.steps):
            optimizer.zero_grad()
            output = local(q)
            output_energy(output, target_t).backward()
            record(q, output, variables)
            optimizer.step()

    # Registrar el estado posterior al último paso (sin otro paso de optimización).
    for v in variables:
        v.grad = None
    output = local(q)
    energy = output_energy(output, target_t)
    if method == "pc":
        energy = energy + sum(p.energy() for p in pcs)
    energy.backward()
    record(q, output, variables)
    weights_equal = all(torch.equal(v, before[k]) for k, v in local.weights().items())
    if not weights_equal or any(p.grad is not None for n, p in local.named_parameters() if not n.endswith("._x")):
        raise RuntimeError("Se modificaron pesos o recibieron gradientes")
    history = {k: np.asarray(v) for k, v in history.items()}
    final_q = history["angles"][-1]
    local.eval()
    with torch.no_grad():
        feedforward = local(q).numpy()[0]
    actual = arm.forward(final_q)
    total = history["energy_terms"].sum(axis=1)
    gradient_norm = float(np.linalg.norm(history["gradient_norms"][-1]))
    converged = bool(total[-1] <= config.energy_tol and gradient_norm <= config.grad_tol)
    real_error = float(np.linalg.norm(actual - target_np))
    history["total_energy"] = total
    history["exact_tip"] = np.array([arm.forward(angles) for angles in history["angles"]])
    summary = {"method": method, "target": target_np.tolist(), "initial_angles": initial.tolist(),
               "final_angles": final_q.tolist(), "normalized_final_angles": normalize_angles(final_q).tolist(),
               "exact_tip": actual.tolist(), "feedforward_tip": feedforward.tolist(),
               "relaxed_tip": history["relaxed_output"][-1].tolist(),
               "real_error": real_error, "feedforward_error": float(np.linalg.norm(feedforward - target_np)),
               "relaxed_error": float(np.linalg.norm(history["relaxed_output"][-1] - target_np)),
               "final_energy": float(total[-1]), "final_gradient_norm": gradient_norm,
               "converged": converged, "termination": "budget_completed_converged" if converged else "max_iterations_not_converged",
               "physical_success": real_error <= config.physical_tol, "iterations": config.steps,
               "weights_exactly_constant": weights_equal,
               "ever_outside_training_domain": bool(np.any((history["angles"] < -np.pi) | (history["angles"] >= np.pi))),
               "max_angle_gradient_norm": float(history["gradient_norms"][:, 0].max()),
               "max_state_gradient_norm": float(history["gradient_norms"][:, 1:].max()) if method == "pc" else None,
               "max_state_displacement": float(np.max(np.abs(history["states"] - history["states"][0]))) if method == "pc" else None}
    return summary, history
