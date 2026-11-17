"""Pruebas de mecanismos y diagnósticos; no exigen resolver IK exactamente."""

import unittest

import numpy as np
import torch

from planar_arm import PlanarArm
from planar_arm.pc01.model import ForwardPC
from planar_arm.pc02 import InferenceConfig, infer


class PC02Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = ForwardPC()
        self.model.eval()
        self.config = InferenceConfig(steps=30)

    def test_frozen_weights_angles_and_latents(self):
        before = self.model.weights()
        summary, history = infer(self.model, [1, 1], [0.2, -0.3], self.config)
        self.assertTrue(summary["weights_exactly_constant"])
        for key, value in self.model.weights().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertGreater(summary["max_angle_gradient_norm"], 0)
        self.assertGreater(summary["max_state_gradient_norm"], 0)
        self.assertGreater(summary["max_state_displacement"], 0)
        self.assertFalse(np.array_equal(history["angles"][0], history["angles"][-1]))
        self.assertEqual(history["angles"].shape, (31, 2))
        self.assertEqual(history["states"].shape, (31, 2, 32))
        # Primer gradiente angular nulo por inicialización x=mu, luego no nulo.
        self.assertEqual(history["gradient_norms"][0, 0], 0)
        self.assertGreater(history["gradient_norms"][2:, 0].max(), 0)
        np.testing.assert_allclose(history["energy_terms"][:, :2],
                                   0.5 * np.square(history["internal_residuals"]).sum(axis=2),
                                   atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(history["total_energy"], history["energy_terms"].sum(axis=1))
        np.testing.assert_allclose(history["energy_terms"][:, 2],
                                   0.5 * np.square(history["relaxed_output"] - [1, 1]).sum(axis=1),
                                   atol=1e-7)
        self.assertFalse(np.allclose(summary["feedforward_tip"], summary["relaxed_tip"]))

    def test_reproducibility_and_seed(self):
        for method in ["pc", "input"]:
            a, ha = infer(self.model, [1, 0], config=self.config, method=method)
            b, hb = infer(self.model, [1, 0], config=self.config, method=method)
            self.assertEqual(a, b)
            for key in ha:
                np.testing.assert_array_equal(ha[key], hb[key])
            changed, _ = infer(self.model, [1, 0], config=InferenceConfig(steps=30, seed=43), method=method)
            self.assertNotEqual(a["initial_angles"], changed["initial_angles"])

    def test_nonconvergence_and_physical_metrics(self):
        for method in ["pc", "input"]:
            summary, history = infer(self.model, [10, 10], [0, 0],
                                     InferenceConfig(steps=4, energy_tol=1e-30, grad_tol=1e-30), method)
            self.assertFalse(summary["converged"])
            self.assertFalse(summary["physical_success"])
            self.assertEqual(summary["termination"], "max_iterations_not_converged")
            self.assertEqual(summary["iterations"], 4)
            expected = PlanarArm().forward(summary["final_angles"])
            np.testing.assert_allclose(summary["exact_tip"], expected)
            self.assertAlmostEqual(summary["real_error"], np.linalg.norm(expected - [10, 10]))
            np.testing.assert_allclose(history["exact_tip"][-1], expected)

    def test_baseline_has_no_free_states(self):
        summary, h = infer(self.model, [0, 1], [0.2, 0.1], self.config, "input")
        self.assertTrue(summary["weights_exactly_constant"])
        self.assertEqual(h["states"].shape, (31, 0, 32))
        self.assertTrue(np.all(h["energy_terms"][:, :2] == 0))
        self.assertIsNone(summary["max_state_displacement"])
        np.testing.assert_array_equal(summary["relaxed_tip"], summary["feedforward_tip"])

    def test_outside_domain_is_reported_without_wrapping(self):
        summary, h = infer(self.model, [1, 0], [4, 0], InferenceConfig(steps=4))
        self.assertTrue(summary["ever_outside_training_domain"])
        self.assertEqual(h["angles"][0, 0], 4)
        self.assertTrue(all(-np.pi <= q < np.pi for q in summary["normalized_final_angles"]))

    def test_invalid_inputs(self):
        for config in [InferenceConfig(steps=0), InferenceConfig(lr_angles=-1),
                       InferenceConfig(lr_states=float("nan")), InferenceConfig(seed=-1)]:
            with self.assertRaises(ValueError):
                infer(self.model, [0, 0], config=config)
        for target in [[0], [np.nan, 0]]:
            with self.assertRaises(ValueError):
                infer(self.model, target)
        with self.assertRaises(ValueError):
            infer(self.model, [0, 0], initial_angles=[0])
        with self.assertRaises(ValueError):
            infer(self.model, [0, 0], method="unknown")


if __name__ == "__main__":
    unittest.main()
