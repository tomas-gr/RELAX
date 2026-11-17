import unittest
import numpy as np
from planar_arm.pc02e import check_prefix, needs_extension, snapshot
from planar_arm.pc02 import InferenceConfig


class TemporalTests(unittest.TestCase):
    def test_prefix_comparison_rejects_changes(self):
        original = {'angles': np.arange(6).reshape(3, 2), 'energy': np.arange(3)}
        longer = {'angles': np.arange(10).reshape(5, 2), 'energy': np.arange(5)}
        check_prefix(original, longer)
        longer['angles'][2, 0] = 99
        with self.assertRaises(ValueError):
            check_prefix(original, longer)

    def test_extension_and_convergence(self):
        e = np.ones(10001); g = np.ones(10001)
        self.assertFalse(needs_extension(e, g, 1e-6, 1e-4))
        e[-1] = .9
        self.assertTrue(needs_extension(e, g, 1e-6, 1e-4))
        e[-1] = 1e-7; g[-1] = 1e-5
        self.assertFalse(needs_extension(e, g, 1e-6, 1e-4))

    def test_snapshot_keeps_physical_separate(self):
        h = {'energy_terms': np.array([[1., 2., 3.]]), 'gradient_norms': np.array([[1., 2., 3.]]), 'angles': np.array([[0., 1.]])}
        d = {k: np.array([v]) for k, v in [('relaxed_error', .01), ('feedforward_error', .4), ('physical_error', .5), ('latent_output_gap', .39), ('model_error', .1), ('out_of_domain', False)]}
        d['internal_norms'] = np.array([[2., 3.]])
        s = snapshot(0, h, d, InferenceConfig())
        self.assertEqual(s['energy'], 6.)
        self.assertEqual(s['physical_error'], .5)
        self.assertFalse(s['converged'])


if __name__ == '__main__':
    unittest.main()
