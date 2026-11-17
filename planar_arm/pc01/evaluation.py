"""Métricas cartesianas y regiones definidas antes de observar prueba."""

import numpy as np
import torch


def metrics(target, predicted):
    """Media/mediana/máximo de distancia euclídea; RMSE = sqrt(mean(||e||²))."""
    errors = np.linalg.norm(np.asarray(predicted) - np.asarray(target), axis=1)
    if not len(errors):
        return {"count": 0, "mean": None, "median": None, "rmse": None, "max": None}
    if not np.isfinite(errors).all():
        raise ValueError("Predicciones no finitas")
    return {"count": len(errors), "mean": float(errors.mean()),
            "median": float(np.median(errors)), "rmse": float(np.sqrt(np.mean(errors**2))),
            "max": float(errors.max())}


def predict(model, angles):
    """Una pasada eval/no_grad en CPU float32; restaura el modo original."""
    previous = model.training
    model.eval()
    with torch.no_grad():
        predicted = model(torch.as_tensor(angles, dtype=torch.float32)).cpu().numpy()
    model.train(previous)
    return predicted


def evaluate(model, rows, lengths=(1, 1), boundary_width=0.2):
    """Evalúa contra etiquetas, sin ajustar pesos ni relajar estados PC.

    Vecindad angular: distancia a ±pi menor a boundary_width en cualquier ángulo.
    Regiones radiales: tercios del intervalo alcanzable. Cuadrantes según objetivo.
    """
    predicted = predict(model, rows[:, :2])
    target = rows[:, 2:]
    seam = np.any(np.pi - np.abs(rows[:, :2]) < boundary_width, axis=1)
    radius = np.linalg.norm(target, axis=1)
    low, high = abs(lengths[0] - lengths[1]), sum(lengths)
    edges = np.linspace(low, high, 4)
    masks = {"angular_boundary": seam, "angular_interior": ~seam}
    for i in range(3):
        masks[f"radial_{i+1}"] = (radius >= edges[i]) & (radius <= edges[i+1] if i == 2 else radius < edges[i+1])
    for name, sx, sy in [("I", 1, 1), ("II", -1, 1), ("III", -1, -1), ("IV", 1, -1)]:
        masks[f"quadrant_{name}"] = ((target[:, 0] >= 0) if sx == 1 else (target[:, 0] < 0)) & ((target[:, 1] >= 0) if sy == 1 else (target[:, 1] < 0))
    report = metrics(target, predicted)
    report["regions"] = {name: metrics(target[mask], predicted[mask]) for name, mask in masks.items()}
    report["evaluation"] = "single_feedforward_eval"
    report["angular_boundary_width_rad"] = boundary_width
    report["radial_edges"] = edges.tolist()
    return report, predicted
