"""Dibujo estático con Matplotlib; no contiene lógica de cinemática inversa."""

import numpy as np
import matplotlib.pyplot as plt


def plot_arm(arm, angles, target=None, ax=None, title=None):
    """Dibuja una configuración (radianes) y objetivo opcional; devuelve el Axes.

    arm es PlanarArm; target, si existe, es (x, y). La escala de ambos ejes
    es idéntica. Se muestran las posiciones y los ángulos normalizados.
    """
    from .kinematics import normalize_angles

    if ax is None:
        _, ax = plt.subplots()
    points = arm.joint_positions(angles)
    ax.plot(points[:, 0], points[:, 1], "o-", lw=3, label="Brazo")
    ax.scatter(*points[-1], s=75, marker="s", label="Extremo", zorder=3)
    if target is not None:
        ax.scatter(*target, marker="x", s=100, color="crimson", label="Objetivo", zorder=4)
    bound = 1.15 * (arm.l1 + arm.l2)
    if target is not None:
        bound = max(bound, 1.15 * float(np.max(np.abs(target))))
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel="x", ylabel="y",
           title=title or "Brazo planar 2R")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    q = normalize_angles(angles)
    ax.text(0.03, 0.03, f"θ1={q[0]:.3f} rad; θ2={q[1]:.3f} rad\n"
            f"Extremo=({points[-1, 0]:.3f}, {points[-1, 1]:.3f})",
            transform=ax.transAxes, fontsize=9)
    ax.legend(loc="upper right")
    return ax
