import unittest
import numpy as np
import torch
from planar_arm import PlanarArm
from planar_arm.pc01.model import ForwardPC
from planar_arm.pc02 import infer, InferenceConfig
from planar_arm.pc02d import diagnose


class DiagnosticTests(unittest.TestCase):
    def test_gradients_against_autograd_and_finite_differences(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        model = ForwardPC().eval()
        target = np.array([1., 1.])
        config = InferenceConfig(steps=8)
        _, h = infer(model, target, [0.3, -0.4], config)
        d = diagnose(model, PlanarArm(), target, h, config.lr_angles, config.lr_states)
        self.assertLess(d['max_energy_mismatch'], 1e-5)
        self.assertLess(d['max_gradient_norm_mismatch'], 1e-5)
        self.assertLess(max(d['max_step_residual']), 1e-6)
        model = model.double()
        q = torch.tensor(h['angles'][4], dtype=torch.float64, requires_grad=True)
        z1 = torch.tensor(h['states'][4, 0], dtype=torch.float64, requires_grad=True)
        z2 = torch.tensor(h['states'][4, 1], dtype=torch.float64, requires_grad=True)
        def energy(q, z1, z2):
            a, b, c = model.layers[0], model.layers[3], model.layers[6]
            return .5*((a(q)-z1)**2).sum()+.5*((b(torch.tanh(z1))-z2)**2).sum()+.5*((c(torch.tanh(z2))-torch.tensor(target))**2).sum()
        grads = torch.autograd.grad(energy(q, z1, z2), [q, z1, z2])
        actual = np.concatenate([g.detach().numpy() for g in grads])
        np.testing.assert_allclose(d['gradient_vectors'][4], actual, atol=1e-12)
        eps = 1e-5
        for variable, index, offset in [(q, 0, 0), (z1, 3, 2), (z2, 7, 34)]:
            with torch.no_grad():
                variable[index] += eps
                plus = energy(q,z1,z2).item()
                variable[index] -= 2*eps
                minus = energy(q,z1,z2).item()
                variable[index] += eps
            self.assertAlmostEqual((plus-minus)/(2*eps), actual[offset+index], places=7)


if __name__ == '__main__':
    unittest.main()
