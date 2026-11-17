"""Entrenamiento supervisado con PCTrainer y selección exclusiva por validación."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from planar_arm.data import generate_dataset
from planar_arm import PlanarArm
from .evaluation import evaluate, metrics, predict
from .model import ForwardPC, make_trainer, output_energy


@dataclass
class Config:
    """Configuración serializable; entrenamiento determinista CPU float32."""

    samples: int = 10000
    seed: int = 42
    l1: float = 1.0
    l2: float = 1.0
    hidden: int = 32
    epochs: int = 60
    batch_size: int = 128
    steps: int = 30
    lr_x: float = 0.1
    lr_p: float = 0.001
    threads: int = 1

    def validate(self):
        for name in ["samples", "hidden", "epochs", "batch_size", "steps", "threads"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} debe ser entero positivo")
        if self.samples < 10 or self.steps < 3:
            raise ValueError("samples >= 10 y steps >= 3")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed debe ser entero no negativo")
        for value in [self.lr_x, self.lr_p]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Tasas de aprendizaje positivas y finitas")
        PlanarArm(self.l1, self.l2)


def write_json(path, value):
    """Guarda JSON UTF-8 estricto (no admite NaN)."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def provenance():
    """Versiones y huellas del código local/submódulo, incluidas fuentes no commiteadas."""
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = list((root / "planar_arm").rglob("*.py"))
    files += list((root / "third_party/PredictiveCoding/predictive_coding").glob("*.py"))
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "pc_commit": git("-C", "third_party/PredictiveCoding", "rev-parse", "HEAD"),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}


def run(config, output):
    """Entrena, selecciona menor RMSE de validación y evalúa prueba una sola vez.

    Rechaza directorios no vacíos. No reanuda entrenamiento: el checkpoint contiene
    pesos aprendidos, no estados por lote ni momentos de los optimizadores.
    """
    config.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Elegí un directorio de salida vacío para preservar experimentos")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    data = generate_dataset(config.samples, config.seed, PlanarArm(config.l1, config.l2))
    data.validate()
    write_json(output / "config.json", asdict(config))
    write_json(output / "environment.json", provenance())
    np.savez_compressed(output / "dataset.npz", train=data.train, validation=data.validation,
                        test=data.test, seed=config.seed, lengths=[config.l1, config.l2],
                        columns=["theta1", "theta2", "x", "y"])
    model = ForwardPC(config.hidden)
    trainer = make_trainer(model, config.steps, config.lr_x, config.lr_p)
    loader = DataLoader(data.as_torch("train"), batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    history, trace = [], {}
    best_rmse, best_epoch, best_weights = float("inf"), 0, None
    start = time.perf_counter()
    for epoch in range(config.epochs + 1):
        energy_sum, loss_sum = 0.0, 0.0
        if epoch:
            model.train()
            for batch_index, (angles, positions) in enumerate(loader):
                record = epoch == 1 and batch_index == 0
                result = trainer.train_on_batch(
                    inputs=angles, loss_fn=output_energy, loss_fn_kwargs={"target": positions},
                    is_optimize_inputs=False, is_return_results_every_t=record)
                energy_sum += result["energy"][-1]
                loss_sum += result["loss"][-1]
                if record:
                    trace = {k: [v / len(angles) for v in result[k]] for k in ["energy", "loss", "overall"]}
        train_metrics = metrics(data.train[:, 2:], predict(model, data.train[:, :2]))
        val_metrics = metrics(data.validation[:, 2:], predict(model, data.validation[:, :2]))
        row = {"epoch": epoch, "train_rmse": train_metrics["rmse"],
               "validation_rmse": val_metrics["rmse"],
               "train_mean": train_metrics["mean"], "validation_mean": val_metrics["mean"],
               "pc_energy_per_sample": energy_sum / len(data.train) if epoch else None,
               "relaxed_output_energy_per_sample": loss_sum / len(data.train) if epoch else None}
        if not np.isfinite([train_metrics["rmse"], val_metrics["rmse"], energy_sum, loss_sum]).all():
            raise RuntimeError("Entrenamiento divergente; revisar tasas de aprendizaje")
        history.append(row)
        if val_metrics["rmse"] < best_rmse:
            best_rmse, best_epoch, best_weights = val_metrics["rmse"], epoch, model.weights()
        write_json(output / "history.json", history)
        if epoch % 5 == 0 or epoch == config.epochs:
            print(f"epoch={epoch} train_rmse={row['train_rmse']:.6f} val_rmse={row['validation_rmse']:.6f}", flush=True)
    # Una instancia fresca evita conservar latentes del último lote.
    selected = ForwardPC(config.hidden)
    selected.load_state_dict(best_weights, strict=True)
    selected.eval()
    checkpoint = {"weights": best_weights, "architecture": {"hidden": config.hidden, "activation": "tanh", "input": 2, "output": 2},
                  "config": asdict(config), "best_epoch": best_epoch, "validation_rmse": best_rmse,
                  "format_version": 1}
    torch.save(checkpoint, output / "model.pt")
    test_metrics, predicted = evaluate(selected, data.test, (config.l1, config.l2))
    report = {"best_epoch": best_epoch, "best_validation_rmse": best_rmse,
              "training_seconds": time.perf_counter() - start, "test": test_metrics}
    write_json(output / "metrics.json", report)
    write_json(output / "inference_trace.json", trace)
    np.savez_compressed(output / "test_predictions.npz", angles=data.test[:, :2], target=data.test[:, 2:], predicted=predicted)
    from .plots import save_plots
    save_plots(history, trace, data.test, predicted, output)
    print(json.dumps(report, indent=2), flush=True)
    return report
