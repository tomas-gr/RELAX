"""Pruebas sin dependencias adicionales: python -m unittest discover -s tests."""

import unittest

import numpy as np

from planar_arm import PlanarArm, normalize_angles


class KinematicsTests(unittest.TestCase):
    def test_known_forward_and_joints(self):
        arm = PlanarArm()
        for q, expected in [((0, 0), (2, 0)), ((np.pi/2, 0), (0, 2)),
                            ((0, np.pi/2), (1, 1)), ((0, np.pi), (0, 0))]:
            np.testing.assert_allclose(arm.forward(q), expected, atol=1e-12)
        points = PlanarArm(2, 1).joint_positions([np.pi/2, -np.pi/2])
        np.testing.assert_allclose(points, [[0, 0], [0, 2], [1, 2]], atol=1e-12)

    def test_two_known_branches(self):
        arm = PlanarArm()
        result = arm.inverse([1, 1])
        self.assertEqual(len(result.solutions), 2)
        self.assertFalse(result.singular)
        np.testing.assert_allclose(result.solutions, [[0, np.pi/2], [np.pi/2, -np.pi/2]], atol=1e-12)
        for q in result.solutions:
            np.testing.assert_allclose(arm.forward(q), [1, 1], atol=1e-12)

    def test_random_round_trips_both_branches(self):
        rng = np.random.default_rng(42)
        for arm in [PlanarArm(), PlanarArm(2, 0.7), PlanarArm(0.7, 2)]:
            for q in rng.uniform(-np.pi, np.pi, (200, 2)):
                target = arm.forward(q)
                result = arm.inverse(target)
                self.assertTrue(result.reachable)
                for solution in result.solutions:
                    self.assertTrue(np.all(np.asarray(solution) >= -np.pi))
                    self.assertTrue(np.all(np.asarray(solution) < np.pi))
                    np.testing.assert_allclose(arm.forward(solution), target, atol=1e-10)

    def test_singular_boundaries(self):
        for arm in [PlanarArm(), PlanarArm(2, 1), PlanarArm(1, 2)]:
            radii = [arm.l1 + arm.l2]
            if arm.l1 != arm.l2:
                radii.append(abs(arm.l1 - arm.l2))
            for radius in radii:
                result = arm.inverse([radius, 0])
                self.assertTrue(result.singular)
                self.assertEqual(len(result.solutions), 1)
                np.testing.assert_allclose(arm.forward(result.solutions[0]), [radius, 0], atol=1e-12)

    def test_origin_continuum(self):
        arm = PlanarArm()
        result = arm.inverse([0, 0])
        self.assertTrue(result.infinite_solutions)
        self.assertTrue(result.singular)
        np.testing.assert_allclose(arm.forward(result.solutions[0]), [0, 0], atol=1e-12)
        for t1 in [-2, 0, 1]:
            np.testing.assert_allclose(arm.forward([t1, -np.pi]), [0, 0], atol=1e-12)

    def test_unreachable(self):
        for arm, target in [(PlanarArm(), [2.01, 0]), (PlanarArm(2, 1), [0.5, 0]),
                            (PlanarArm(2, 1), [0, 0])]:
            result = arm.inverse(target)
            self.assertFalse(result.reachable)
            self.assertEqual(result.solutions, ())

    def test_boundary_tolerance(self):
        arm = PlanarArm()
        self.assertTrue(arm.inverse([2 + 1e-11, 0]).reachable)
        self.assertFalse(arm.inverse([2 + 1e-7, 0]).reachable)
        self.assertEqual(len(arm.inverse([2 - 1e-7, 0]).solutions), 2)

    def test_normalization(self):
        np.testing.assert_allclose(normalize_angles([0, 2*np.pi, np.pi, -3*np.pi]), [0, 0, -np.pi, -np.pi])
        np.testing.assert_allclose(PlanarArm().forward([7, -8]), PlanarArm().forward(normalize_angles([7, -8])))

    def test_invalid_inputs(self):
        for lengths in [(0, 1), (-1, 1), (np.inf, 1), (1, np.nan)]:
            with self.assertRaises(ValueError):
                PlanarArm(*lengths)
        for value in [[1], [np.nan, 0], [0, np.inf]]:
            with self.assertRaises(ValueError):
                PlanarArm().forward(value)
            with self.assertRaises(ValueError):
                PlanarArm().inverse(value)
        for tol in [-1, np.nan, 1]:
            with self.assertRaises(ValueError):
                PlanarArm().inverse([1, 1], tol)


if __name__ == "__main__":
    unittest.main()
