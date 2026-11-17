"""Simulador cinemático de brazo planar 2R, independiente de PC."""

from .kinematics import IKResult, PlanarArm, normalize_angles

__all__ = ["IKResult", "PlanarArm", "normalize_angles"]
