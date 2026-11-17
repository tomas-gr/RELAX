"""Figuras estáticas del experimento, exportadas con Agg."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_plots(history, trace, rows, predicted, output):
    """Exporta evolución, dinámica del primer lote y errores del test seleccionado."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ["train", "validation"]:
        axes[0].plot([h["epoch"] for h in history], [h[f"{split}_rmse"] for h in history], label=split)
    axes[0].set(xlabel="Época", ylabel="RMSE cartesiano (eval)")
    axes[0].legend()
    trained = history[1:]
    for key in ["pc_energy_per_sample", "relaxed_output_energy_per_sample"]:
        axes[1].plot([h["epoch"] for h in trained], [h[key] for h in trained], label=key)
    axes[1].set(xlabel="Época", ylabel="Energía final del lote / muestras")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, values in trace.items():
        ax.plot(values, label=key)
    ax.set(xlabel="Iteración PC del primer lote", ylabel="Energía / muestras")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "inference_trace.png", dpi=150)
    plt.close(fig)
    errors = np.linalg.norm(predicted - rows[:, 2:], axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    indices = np.linspace(0, len(rows)-1, min(100, len(rows)), dtype=int)
    real, pred = rows[indices, 2:], predicted[indices]
    axes[0].scatter(*real.T, s=12, label="Real")
    axes[0].scatter(*pred.T, s=12, marker="x", label="Predicha")
    for a, b in zip(real, pred):
        axes[0].plot([a[0], b[0]], [a[1], b[1]], color="gray", alpha=0.3, lw=0.5)
    axes[0].set_aspect("equal")
    axes[0].legend()
    axes[0].set(title="100 ejemplos de prueba", xlabel="x", ylabel="y")
    for ax, coords, labels in [(axes[1], rows[:, 2:], ("x", "y")),
                               (axes[2], rows[:, :2], ("θ1 (rad)", "θ2 (rad)"))]:
        points = ax.scatter(*coords.T, c=errors, s=8, cmap="magma")
        fig.colorbar(points, ax=ax, label="Distancia cartesiana")
        ax.set(xlabel=labels[0], ylabel=labels[1])
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=150)
    plt.close(fig)
