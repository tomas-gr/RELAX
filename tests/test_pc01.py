"""Verifica integración real con PCTrainer, modos, checkpoints y métricas."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import predictive_coding as pc
import torch

from planar_arm.data import generate_dataset
from planar_arm.pc01.evaluation import evaluate, metrics, predict
from planar_arm.pc01.experiment import Config
from planar_arm.pc01.model import ForwardPC, load_model, make_trainer, output_energy


class PC01Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.angles, self.positions = generate_dataset(40).as_torch("train").tensors

    def test_inference_changes_states_only_then_learning_changes_weights(self):
        model = ForwardPC(8)
        model.train()
        trainer = make_trainer(model, steps=12)
        initial = model.weights()
        first_states = []

        def after(t):
            if t == 0:
                first_states.extend(x.detach().clone() for x in trainer.get_model_xs())
            if t < 11:
                for name, value in model.weights().items():
                    torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
            if t == 10:
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first_states, trainer.get_model_xs())))

        angles_before = self.angles.clone()
        targets_before = self.positions.clone()
        result = trainer.train_on_batch(self.angles, output_energy, {"target": self.positions}, callback_after_t=after)
        self.assertTrue(np.isfinite(result["overall"][-1]))
        # El error llega a la primera capa a través de los estados PC.
        self.assertFalse(torch.equal(model.weights()["layers.0.weight"], initial["layers.0.weight"]))
        torch.testing.assert_close(self.angles, angles_before, rtol=0, atol=0)
        torch.testing.assert_close(self.positions, targets_before, rtol=0, atol=0)
        xs = set(trainer.get_model_xs())
        self.assertTrue(xs.isdisjoint(set(trainer.get_model_parameters())))

    def test_eval_is_transparent_and_checkpoint_round_trip(self):
        model = ForwardPC()
        model.train()
        trainer = make_trainer(model, steps=5)
        trainer.train_on_batch(self.angles, output_energy, {"target": self.positions})
        predicted = predict(model, self.angles)
        model.eval()
        with torch.no_grad():
            manual = self.angles
            for layer in model.layers:
                if not isinstance(layer, pc.PCLayer):
                    manual = layer(manual)
        np.testing.assert_array_equal(predicted, manual.numpy())
        self.assertFalse(any(k.endswith("._x") for k in model.weights()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"architecture": {"hidden": 32}, "weights": model.weights()}, path)
            loaded, _ = load_model(path)
            self.assertFalse(loaded.training)
            np.testing.assert_array_equal(predict(loaded, self.angles), predicted)
            # Cambiar el tamaño de GEMM puede alterar el redondeo float32.
            np.testing.assert_allclose(predict(loaded, self.angles[:7]), predicted[:7], atol=1e-7, rtol=1e-6)

    def test_deterministic_pc_updates(self):
        def train():
            torch.manual_seed(17)
            model = ForwardPC(8)
            model.train()
            trainer = make_trainer(model, steps=6)
            for _ in range(3):
                trainer.train_on_batch(self.angles, output_energy, {"target": self.positions})
            return model.weights()
        a, b = train(), train()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_metrics_definition_and_region_counts(self):
        result = metrics(np.zeros((2, 2)), [[3, 4], [0, 0]])
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(12.5))
        self.assertEqual(result["max"], 5)
        self.assertIsNone(metrics(np.empty((0, 2)), np.empty((0, 2)))["mean"])
        data = generate_dataset(100)
        model = ForwardPC()
        before = model.weights()
        report, _ = evaluate(model, data.test)
        regions = report["regions"]
        self.assertEqual(sum(regions[k]["count"] for k in ["angular_boundary", "angular_interior"]), 10)
        self.assertEqual(sum(regions[f"radial_{i}"]["count"] for i in range(1, 4)), 10)
        self.assertEqual(sum(regions[f"quadrant_{i}"]["count"] for i in ["I", "II", "III", "IV"]), 10)
        for key, value in model.weights().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_invalid_configuration(self):
        for kwargs in [{"steps": 2}, {"epochs": 0}, {"seed": -1}, {"lr_x": float("nan")}, {"samples": 9}]:
            with self.assertRaises(ValueError):
                Config(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
