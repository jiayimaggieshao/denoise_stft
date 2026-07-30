from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.frequency_data import MixingConfig, SyntheticFrequencyDataset
from src.frequency_baselines import complex_transfer_cancellation, hybrid_frequency_fusion
from src.frequency_loss import FrequencyDenoiseLoss
from src.frequency_model import CardioSpecNet, FrequencyModelConfig
from src.window_alignment import nearest_timestamp_alignment, overlap_add_windows


class FrequencyPipelineTests(unittest.TestCase):
    def test_model_shape_and_gradient(self) -> None:
        torch.manual_seed(0)
        model = CardioSpecNet(model_config=FrequencyModelConfig(base_channels=4, grid_blocks=1))
        chest = torch.randn(2, 8_000) * 0.01
        reference = torch.randn(2, 8_000) * 0.01
        target = torch.randn(2, 8_000) * 0.005
        output = model(chest, reference, torch.tensor([1.0, 0.0]))
        self.assertEqual(output.shape, chest.shape)
        loss, terms = FrequencyDenoiseLoss()(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.0)
        self.assertIn("over_attenuation", terms)

    def test_dataset_is_deterministic_and_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_dir = root / "clean"
            noise_dir = root / "noise"
            clean_dir.mkdir()
            noise_dir.mkdir()
            rng = np.random.default_rng(1)
            clean = (rng.normal(0, 500, (20, 8_000))).astype(np.int16)
            noise = (rng.normal(0, 2_000, (20, 8_000))).astype(np.int16)
            np.savez_compressed(clean_dir / "clean.npz", x=clean)
            np.savez_compressed(noise_dir / "noise.npz", x=noise)
            config = MixingConfig(reference_dropout_probability=0.0, identity_probability=0.0)
            dataset = SyntheticFrequencyDataset(
                clean_dir,
                noise_dir,
                samples_per_epoch=4,
                source_stride=2,
                seed=7,
                config=config,
            )
            first = dataset[2]
            second = dataset[2]
            self.assertTrue(torch.equal(first["chest"], second["chest"]))
            self.assertTrue(torch.equal(first["clean"], second["clean"]))
            self.assertTrue(torch.equal(first["clean_raw"], second["clean_raw"]))
            self.assertEqual(first["chest"].shape, (8_000,))
            self.assertEqual(first["clean_raw"].shape, (8_000,))
            self.assertEqual(float(first["ref_available"]), 1.0)
            self.assertFalse(torch.equal(first["chest"], first["clean"]))

    def test_timestamp_alignment(self) -> None:
        target = np.array([1_000, 2_000, 3_000, 8_000])
        reference = np.array([900, 2_100, 2_950])
        indices, available = nearest_timestamp_alignment(target, reference, max_delta_us=200)
        np.testing.assert_array_equal(indices, np.array([0, 1, 2, 2]))
        np.testing.assert_array_equal(available, np.array([True, True, True, False]))

    def test_overlap_add_uses_actual_starts(self) -> None:
        windows = np.stack([np.ones(8), np.ones(8) * 3]).astype(np.float32)
        output = overlap_add_windows(windows, np.array([0, 4]), synthesis_window="ones")
        np.testing.assert_allclose(output[:4], 1.0)
        np.testing.assert_allclose(output[4:8], 2.0)
        np.testing.assert_allclose(output[8:], 3.0)


    def test_hybrid_fusion_is_bounded_convex_combination(self) -> None:
        neural = torch.tensor([[1.0, -1.0]])
        adaptive = torch.tensor([[3.0, 1.0]])
        fused = hybrid_frequency_fusion(neural, adaptive, neural_weight=0.25)
        torch.testing.assert_close(fused, torch.tensor([[2.5, 0.5]]))
        fallback = hybrid_frequency_fusion(
            neural, adaptive, neural_weight=0.25, ref_available=torch.tensor([0.0])
        )
        torch.testing.assert_close(fallback, neural)
        with self.assertRaises(ValueError):
            hybrid_frequency_fusion(neural, adaptive, neural_weight=1.1)

    def test_complex_transfer_baseline_falls_back_with_missing_reference(self) -> None:
        torch.manual_seed(3)
        chest = torch.randn(2, 8_000) * 0.01
        estimate = complex_transfer_cancellation(chest, torch.zeros_like(chest))
        self.assertEqual(estimate.shape, chest.shape)
        self.assertTrue(torch.isfinite(estimate).all())


if __name__ == "__main__":
    unittest.main()
