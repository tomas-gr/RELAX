"""Ejemplo reproducible: python -m planar_arm.explore_data --output runs/arm_data."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from .data import COLUMNS, SPLITS, generate_dataset
from .kinematics import PlanarArm


def main():
    parser = argparse.ArgumentParser(description="Generar y explorar datos cinemáticos 2R")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=float, nargs=2, default=[1, 1])
    parser.add_argument("--output", type=Path, default=Path("runs/arm_data"))
    args = parser.parse_args()
    try:
        dataset = generate_dataset(args.samples, args.seed, PlanarArm(*args.lengths))
    except ValueError as exc:
        parser.error(str(exc))
    error = dataset.validate()
    repeated = generate_dataset(args.samples, args.seed, dataset.arm)
    reproducible = all(np.array_equal(getattr(dataset, s), getattr(repeated, s)) for s in SPLITS)
    if not reproducible:
        raise RuntimeError("Falló la comprobación de reproducibilidad.")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "dataset.npz",
                        **{s: getattr(dataset, s) for s in SPLITS},
                        columns=np.array(COLUMNS), seed=dataset.seed,
                        lengths=[dataset.arm.l1, dataset.arm.l2],
                        numpy_version=np.__version__, generator="PCG64/SeedSequence.spawn")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .visualization import plot_arm

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for split in SPLITS:
        rows = getattr(dataset, split)
        axes[0, 0].scatter(rows[:, 2], rows[:, 3], s=2, alpha=0.3, label=split)
    axes[0, 0].set(title="Extremos: muestreo uniforme en ángulos", xlabel="x", ylabel="y")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(markerscale=4)
    all_rows = np.concatenate([getattr(dataset, s) for s in SPLITS])
    for j in range(2):
        axes[0, j + 1].hist(all_rows[:, j], bins=40, range=(-np.pi, np.pi))
        axes[0, j + 1].set(title=f"Distribución de θ{j + 1}", xlabel="Radianes", ylabel="Muestras")
    # Selección determinista, sin consumir los generadores usados para los datos.
    indices = np.linspace(0, len(dataset.train) - 1, 3, dtype=int)
    for ax, index in zip(axes[1], indices):
        row = dataset.train[index]
        plot_arm(dataset.arm, row[:2], row[2:], ax, f"Entrenamiento: muestra {index}")
    fig.tight_layout()
    fig.savefig(args.output / "distribution.png", dpi=150)
    plt.close(fig)
    print({s: len(getattr(dataset, s)) for s in SPLITS})
    print(f"Error cartesiano máximo: {error:.3e}; reproducible: {reproducible}; sin duplicados.")
    print(f"Datos y figura: {args.output.resolve()}")


if __name__ == "__main__":
    main()
