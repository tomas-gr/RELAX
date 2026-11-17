"""Validación del generador y su adaptador PyTorch."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from planar_arm import PlanarArm
from planar_arm.data import SPLITS, generate_dataset


class ArmDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset()

    def test_default_counts_ranges_and_uniqueness(self):
        data = self.data
        self.assertEqual([len(getattr(data, s)) for s in SPLITS], [8000, 1000, 1000])
        self.assertEqual(data.validate(), 0.0)
        sets = []
        for name in SPLITS:
            rows = getattr(data, name)
            self.assertEqual(rows.dtype, np.float64)
            self.assertTrue(np.all(np.linalg.norm(rows[:, 2:], axis=1) <= 2 + 1e-12))
            sets.append(set(map(tuple, rows[:, :2])))
        for i in range(3):
            self.assertEqual(len(sets[i]), len(getattr(data, SPLITS[i])))
            for j in range(i):
                self.assertTrue(sets[i].isdisjoint(sets[j]))

    def test_reproducibility_seed_and_global_rng(self):
        state = np.random.get_state()
        first = generate_dataset(101, seed=17)
        repeated = generate_dataset(101, seed=17)
        changed = generate_dataset(101, seed=18)
        after = np.random.get_state()
        for before_part, after_part in zip(state, after):
            np.testing.assert_equal(before_part, after_part)
        for split in SPLITS:
            np.testing.assert_array_equal(getattr(first, split), getattr(repeated, split))
            self.assertFalse(np.array_equal(getattr(first, split), getattr(changed, split)))
        self.assertEqual([len(getattr(first, s)) for s in SPLITS], [80, 10, 11])

    def test_configurable_lengths_against_formula(self):
        data = generate_dataset(100, 5, PlanarArm(0.7, 2))
        self.assertEqual(data.validate(), 0)
        for split in SPLITS:
            rows = getattr(data, split)
            t1, t2 = rows[:, 0], rows[:, 1]
            expected = np.column_stack((0.7*np.cos(t1) + 2*np.cos(t1+t2),
                                        0.7*np.sin(t1) + 2*np.sin(t1+t2)))
            np.testing.assert_allclose(rows[:, 2:], expected, atol=1e-12)

    def test_torch_interface_and_loader(self):
        for split in SPLITS:
            tensors = self.data.as_torch(split)
            q, xy = tensors.tensors
            self.assertEqual(q.dtype, torch.float32)
            self.assertEqual(q.shape, (len(getattr(self.data, split)), 2))
            np.testing.assert_allclose(q.numpy(), getattr(self.data, split)[:, :2], atol=2e-7)
            np.testing.assert_allclose(xy.numpy(), getattr(self.data, split)[:, 2:], atol=1e-7)
            batch = next(iter(DataLoader(tensors, batch_size=32)))
            self.assertEqual(batch[0].shape, (32, 2))
        double = self.data.as_torch("test", dtype=torch.float64)
        np.testing.assert_array_equal(double.tensors[0].numpy(), self.data.test[:, :2])
        double.tensors[0][0, 0] = 99
        self.assertNotEqual(self.data.test[0, 0], 99)

    def test_validation_detects_corruption_and_leakage(self):
        data = generate_dataset(20)
        data.test[0] = data.train[0]
        with self.assertRaises(ValueError):
            data.validate()
        data = generate_dataset(20)
        data.train[0, 2] += 0.1
        with self.assertRaises(ValueError):
            data.validate()

    def test_invalid_arguments(self):
        for n in [0, 9, 10.5, True]:
            with self.assertRaises(ValueError):
                generate_dataset(n)
        for seed in [-1, 0.5, True, None]:
            with self.assertRaises(ValueError):
                generate_dataset(10, seed)
        with self.assertRaises(ValueError):
            self.data.as_torch("unknown")
        with self.assertRaises(ValueError):
            self.data.as_torch("train", torch.int64)


if __name__ == "__main__":
    unittest.main()
