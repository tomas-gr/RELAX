"""Pares sintéticos [theta1, theta2, x, y], sin dependencia obligatoria de Torch."""

from dataclasses import dataclass

import numpy as np

from .kinematics import PlanarArm

COLUMNS = ("theta1", "theta2", "x", "y")
SPLITS = ("train", "validation", "test")


@dataclass
class ArmDataset:
    """Particiones float64 (N, 4), semilla y geometría que las generaron."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int
    arm: PlanarArm

    def as_torch(self, split, dtype=None):
        """Devuelve TensorDataset(ángulos, posiciones), cada tensor de forma (N, 2).

        split: train, validation o test. dtype: torch.float32 por defecto.
        Crea copias en CPU; no comparte memoria con los arrays originales.
        Torch se importa únicamente al utilizar esta interfaz.
        """
        import torch
        from torch.utils.data import TensorDataset

        if split not in SPLITS:
            raise ValueError(f"Partición desconocida: {split}")
        if dtype is None:
            dtype = torch.float32
        if not dtype.is_floating_point:
            raise ValueError("Los tensores deben usar un dtype de punto flotante.")
        rows = getattr(self, split)
        return TensorDataset(torch.tensor(rows[:, :2], dtype=dtype),
                             torch.tensor(rows[:, 2:], dtype=dtype))

    def validate(self, atol=1e-12):
        """Comprueba rango angular, finitud, unicidad global y cinemática exacta.

        Devuelve el máximo error absoluto cartesiano por componente. Lanza
        ValueError ante datos inválidos, duplicados o error superior a atol.
        """
        arrays = [getattr(self, name) for name in SPLITS]
        for rows in arrays:
            if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
                raise ValueError("Se requieren arrays finitos de forma (N, 4).")
        rows = np.concatenate(arrays)
        if not np.all((rows[:, :2] >= -np.pi) & (rows[:, :2] < np.pi)):
            raise ValueError("Ángulos fuera de [-pi, pi).")
        if len(np.unique(rows[:, :2], axis=0)) != len(rows):
            raise ValueError("Hay configuraciones repetidas dentro o entre particiones.")
        expected = np.array([self.arm.forward(q) for q in rows[:, :2]])
        error = float(np.max(np.abs(expected - rows[:, 2:])))
        if error > atol:
            raise ValueError(f"Error cinemático {error} superior a {atol}.")
        return error


def generate_dataset(n_samples=10_000, seed=42, arm=None):
    """Genera configuraciones uniformes independientes y posiciones con PlanarArm.

    Ambos ángulos se muestrean en [-pi, pi), sin transformar sus valores. Usa
    generadores locales PCG64 con tres secuencias hijas de SeedSequence(seed).
    No altera el RNG global. n_samples debe ser entero >= 10; seed, entero >= 0.
    Tamaños: floor(0.8*N), floor(0.1*N), resto. Se rechazan eventuales pares
    angulares idénticos para garantizar unicidad incluso entre particiones.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 10:
        raise ValueError("n_samples debe ser un entero >= 10.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed debe ser un entero >= 0.")
    arm = PlanarArm() if arm is None else arm
    counts = [8 * n_samples // 10, n_samples // 10]
    counts.append(n_samples - sum(counts))
    children = np.random.SeedSequence(seed).spawn(3)
    seen = set()
    partitions = []
    for count, child in zip(counts, children):
        rng = np.random.Generator(np.random.PCG64(child))
        rows = np.empty((count, 4), dtype=np.float64)
        for i in range(count):
            while True:
                angles = rng.uniform(-np.pi, np.pi, size=2)
                key = tuple(angles)
                if key not in seen and np.all(angles < np.pi):
                    seen.add(key)
                    break
            rows[i, :2] = angles
            rows[i, 2:] = arm.forward(angles)
        partitions.append(rows)
    return ArmDataset(*partitions, seed=int(seed), arm=arm)
