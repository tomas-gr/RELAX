"""Cinemática 2R: hombro absoluto, codo relativo; todos los ángulos en radianes."""

from dataclasses import dataclass

import numpy as np


def normalize_angles(angles):
    """Normaliza un escalar o array de ángulos finitos al intervalo [-pi, pi)."""
    angles = np.asarray(angles, dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Los ángulos deben ser finitos.")
    return (angles + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class IKResult:
    """Soluciones (pares theta1, theta2), singularidad y multiplicidad continua.

    Sin soluciones significa objetivo inalcanzable. Si infinite_solutions es True,
    solutions contiene un representante: theta1 es arbitrario y theta2 = -pi.
    """

    solutions: tuple[tuple[float, float], ...]
    singular: bool = False
    infinite_solutions: bool = False

    @property
    def reachable(self):
        """Indica si existe al menos una solución (dentro de la tolerancia)."""
        return bool(self.solutions)


@dataclass(frozen=True)
class PlanarArm:
    """Brazo con longitudes positivas finitas y hombro fijo en (0, 0)."""

    l1: float = 1.0
    l2: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.l1, self.l2)):
            raise ValueError("Las longitudes deben ser positivas y finitas.")

    def joint_positions(self, angles):
        """De un par (theta1, theta2) devuelve array (3, 2): hombro, codo, extremo."""
        q = normalize_angles(angles)
        if q.shape != (2,):
            raise ValueError("Se requiere un par (theta1, theta2).")
        t1, t2 = q
        elbow = self.l1 * np.array([np.cos(t1), np.sin(t1)])
        tip = elbow + self.l2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
        return np.array([[0.0, 0.0], elbow, tip])

    def forward(self, angles):
        """De un par angular en radianes devuelve la posición cartesiana (x, y)."""
        return self.joint_positions(angles)[-1]

    def inverse(self, target, tol=1e-10):
        """Calcula ambas ramas analíticas, normalizadas a [-pi, pi).

        target es un par cartesiano finito. tol es una tolerancia relativa al
        alcance máximo: puntos apenas fuera de la frontera se proyectan sobre
        ella. En fronteras singulares las ramas coinciden y se devuelve una.
        Con longitudes iguales y objetivo exactamente en el origen hay infinitas
        soluciones; se devuelve (0, -pi) como representante. No se imponen límites
        articulares. Un objetivo inalcanzable produce IKResult sin soluciones.
        """
        p = np.asarray(target, dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            raise ValueError("El objetivo debe ser un par cartesiano finito.")
        if not np.isfinite(tol) or not 0 <= tol < 1:
            raise ValueError("tol debe estar en [0, 1).")
        scale = self.l1 + self.l2
        a, b = self.l1 / scale, self.l2 / scale
        r = float(np.hypot(*(p / scale)))
        inner = abs(a - b)
        if r > 1 + tol or r < inner - tol:
            return IKResult(())
        if r == 0 and self.l1 == self.l2:
            return IKResult(((0.0, -float(np.pi)),), True, True)
        # Snap only numerical boundary discrepancies; retain distinct nearby branches.
        radius = np.clip(r, inner, 1.0)
        c2 = float(np.clip((radius**2 - a*a - b*b) / (2*a*b), -1, 1))
        if radius == inner:
            c2 = -1.0
        elif radius == 1.0:
            c2 = 1.0
        t2 = float(np.arccos(c2))
        singular = c2 == -1.0 or c2 == 1.0
        branches = [t2] if singular else [t2, -t2]
        solutions = []
        for angle2 in branches:
            angle1 = np.arctan2(p[1], p[0]) - np.arctan2(
                b * np.sin(angle2), a + b * np.cos(angle2))
            solutions.append(tuple(float(v) for v in normalize_angles([angle1, angle2])))
        return IKResult(tuple(solutions), singular)
