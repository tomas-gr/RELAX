"""Batería pareada PC-02, sin selección ni entrenamiento de pesos."""

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from planar_arm import PlanarArm, normalize_angles
from planar_arm.pc01 import load_model
from planar_arm.pc01.experiment import provenance, write_json
from .inference import InferenceConfig, infer

DEFAULT_TARGETS = [(1, 1), (1, 0), (0, 1), (-1, 1)]


def run(checkpoint, output, config=None, targets=None, starts=4, initials=None):
    """Usa las mismas inicializaciones por objetivo y método; guarda todas las pruebas.

    initials, si se proporciona, precede a las inicializaciones aleatorias hasta
    completar starts. Nunca usa IK para inicializar ni selecciona la mejor prueba.
    """
    config = InferenceConfig() if config is None else config
    config.validate()
    if not isinstance(starts, int) or starts < 1:
        raise ValueError("starts debe ser entero positivo")
    targets = DEFAULT_TARGETS if targets is None else targets
    supplied = [] if initials is None else initials
    if len(supplied) > starts:
        raise ValueError("Más inicializaciones explícitas que starts")
    output, checkpoint = Path(output), Path(checkpoint)
    if output.resolve() == checkpoint.parent.resolve():
        raise ValueError("La salida debe estar separada de PC-01")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Elegí un directorio de salida vacío")
    # Conservar hash de todos los artefactos de PC-01 antes y después.
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in checkpoint.parent.iterdir() if p.is_file()}
    model, saved = load_model(checkpoint)
    arm = PlanarArm(saved["config"]["l1"], saved["config"]["l2"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config.seed)
    initial_angles = [np.asarray(q, dtype=float).tolist() for q in supplied]
    initial_angles.extend(rng.uniform(-np.pi, np.pi, size=(starts - len(supplied), 2)).tolist())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"inference": asdict(config), "targets": targets,
               "starts": starts, "initial_angles": initial_angles, "checkpoint": str(checkpoint.resolve()),
               "checkpoint_sha256": original_hashes[str(checkpoint)], "lengths": [arm.l1, arm.l2],
               "threads": 1, "angle_constraints": "none", "selection": "all_trials_reported"})
    write_json(output / "environment.json", provenance())
    from .plots import plot_trial, plot_states
    trials = []
    for target_index, target in enumerate(targets):
        # Solo referencia de evaluación; nunca interviene en el optimizador.
        reference = arm.inverse(target)
        for start_index, initial in enumerate(initial_angles):
            name = f"target{target_index}_start{start_index}"
            histories, summaries = {}, {}
            for method in ["pc", "input"]:
                summary, history = infer(model, target, initial, config, method, arm)
                summary.update(target_index=target_index, start_index=start_index,
                               analytic_solutions=[list(q) for q in reference.solutions],
                               analytic_infinite_solutions=reference.infinite_solutions)
                # Distancia angular periódica a cualquier solución analítica.
                summary["nearest_analytic_angle_distance"] = (min(float(np.linalg.norm(normalize_angles(
                    np.asarray(summary["final_angles"]) - q))) for q in reference.solutions)
                    if reference.reachable and not reference.infinite_solutions else None)
                histories[method], summaries[method] = history, summary
                np.savez_compressed(output / f"{name}_{method}.npz", **history)
                trials.append(summary)
                print(f"{name} {method}: real={summary['real_error']:.5f} "
                      f"red={summary['feedforward_error']:.5f} relajado={summary['relaxed_error']:.5f} "
                      f"converged={summary['converged']}", flush=True)
            plot_trial(arm, target, summaries, histories, output / f"{name}.png")
            plot_states(histories["pc"], output / f"{name}_states.png")
            write_json(output / "results.json", trials)
    aggregates = []
    for i, target in enumerate(targets):
        for method in ["pc", "input"]:
            group = [t for t in trials if t["target_index"] == i and t["method"] == method]
            errors = np.array([t["real_error"] for t in group])
            aggregates.append({"target": list(target), "method": method, "trials": len(group),
                               "mean_real_error": float(errors.mean()), "median_real_error": float(np.median(errors)),
                               "min_real_error": float(errors.min()), "max_real_error": float(errors.max()),
                               "mean_feedforward_error": float(np.mean([t["feedforward_error"] for t in group])),
                               "mean_relaxed_error": float(np.mean([t["relaxed_error"] for t in group])),
                               "converged": sum(t["converged"] for t in group),
                               "physical_successes": sum(t["physical_success"] for t in group),
                               "outside_training_domain": sum(t["ever_outside_training_domain"] for t in group)})
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in original_hashes.items())
    if not unchanged:
        raise RuntimeError("Cambió un artefacto de PC-01")
    write_json(output / "summary.json", {"aggregates": aggregates, "pc01_artifacts_unchanged": unchanged,
                                         "all_weights_constant": all(t["weights_exactly_constant"] for t in trials)})
    return aggregates
