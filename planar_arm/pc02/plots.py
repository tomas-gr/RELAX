"""Diagnósticos estáticos comparables para inferencia PC y referencia."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from planar_arm.visualization import plot_arm


def plot_trial(arm, target, summaries, histories, path):
    """Inicial/final, energía, ángulos, latentes y trayectorias físicas exactas."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    initial = histories["pc"]["angles"][0]
    plot_arm(arm, initial, target, axes[0, 0], "Configuración inicial común")
    for ax, method in zip(axes[0, 1:], ["pc", "input"]):
        summary, h = summaries[method], histories[method]
        plot_arm(arm, h["angles"][-1], target, ax,
                 f"{method}: error físico={summary['real_error']:.4f}")
        ax.scatter(*summary["feedforward_tip"], marker="+", color="green", s=80, label="Red eval")
        ax.scatter(*summary["relaxed_tip"], marker="*", color="purple", s=70, label="Salida optimizada")
        ax.legend(fontsize=8)
    for method, h in histories.items():
        steps = np.arange(len(h["angles"]))
        axes[1, 0].semilogy(steps, np.maximum(h["total_energy"], 1e-16), label=f"{method}: total")
        for j in range(2):
            axes[1, 1].plot(steps, h["angles"][:, j], label=f"{method}: θ{j+1}")
        axes[1, 2].plot(*h["exact_tip"].T, label=method)
    h = histories["pc"]
    axes[1, 0].semilogy(np.maximum(h["energy_terms"][:, :2].sum(axis=1), 1e-16), "--", label="PC: errores internos")
    axes[1, 0].semilogy(np.maximum(h["energy_terms"][:, 2], 1e-16), ":", label="PC: salida")
    axes[1, 0].set(xlabel="Iteración", ylabel="Energía", title="Objetivos distintos; escalas originales")
    axes[1, 1].set(xlabel="Iteración", ylabel="Ángulos libres (rad)")
    axes[1, 2].scatter(*target, color="crimson", marker="x", s=100, label="Objetivo")
    axes[1, 2].set(xlabel="x", ylabel="y", title="Trayectoria física exacta")
    axes[1, 2].set_aspect("equal")
    for ax in axes[1]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_states(history, path):
    """Desplazamiento de cada estado y norma de gradientes sin escalar."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j in range(2):
        movement = np.linalg.norm(history["states"][:, j] - history["states"][0, j], axis=1)
        axes[0].plot(movement, label=f"Estado {j+1}")
    for j, label in enumerate(["Ángulos", "Estado 1", "Estado 2"]):
        axes[1].semilogy(np.maximum(history["gradient_norms"][:, j], 1e-16), label=label)
    axes[0].set(xlabel="Iteración", ylabel="||estado - estado inicial||₂")
    axes[1].set(xlabel="Iteración", ylabel="Norma de gradiente")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
