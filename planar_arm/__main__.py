"""Ejemplo: python -m planar_arm --angles 0.5 1 --target 1 1."""

import argparse

import matplotlib
import matplotlib.pyplot as plt

from .kinematics import PlanarArm
from .visualization import plot_arm


def main():
    parser = argparse.ArgumentParser(description="Simulador estático de brazo planar 2R")
    parser.add_argument("--lengths", nargs=2, type=float, default=[1, 1], metavar=("L1", "L2"))
    parser.add_argument("--angles", nargs=2, type=float, default=[0.5, 1], metavar=("THETA1", "THETA2"), help="Radianes")
    parser.add_argument("--target", nargs=2, type=float, default=[1, 1], metavar=("X", "Y"))
    parser.add_argument("--save", help="Guardar figura en vez de abrir una ventana")
    args = parser.parse_args()
    if args.save:
        matplotlib.use("Agg")
    try:
        arm = PlanarArm(*args.lengths)
        result = arm.inverse(args.target)
        arm.forward(args.angles)
    except ValueError as exc:
        parser.error(str(exc))
    fig, axes = plt.subplots(1, 1 + max(1, len(result.solutions)), figsize=(14, 5), squeeze=False)
    axes = axes[0]
    plot_arm(arm, args.angles, args.target, axes[0], "Cinemática directa")
    if result.reachable:
        for i, q in enumerate(result.solutions):
            title = f"Inversa: solución {i + 1}"
            if result.infinite_solutions:
                title += " (representante de infinitas)"
            elif result.singular:
                title += " (singular)"
            plot_arm(arm, q, args.target, axes[i + 1], title)
            print(f"Solución {i + 1}: {q}; extremo: {arm.forward(q)}")
    else:
        plot_arm(arm, args.angles, args.target, axes[1], "Objetivo inalcanzable")
        print("Objetivo inalcanzable.")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
