from __future__ import annotations

import unittest

import numpy as np
import torch

from src.methods.otc_mmot import OnlineMixtureBank


class OnlineMixtureBankTest(unittest.TestCase):
    def test_centroids_expand_and_inference_uses_raw_class_mapping(self) -> None:
        bank = OnlineMixtureBank(max_centroids=2, split_threshold=0.1)
        bank.update(torch.tensor([[0.0, 0.0], [0.1, 0.0]]), torch.tensor([20, 20]))
        bank.update(torch.tensor([[4.0, 4.0], [4.1, 4.0]]), torch.tensor([20, 20]))
        bank.update(torch.tensor([[9.0, 9.0], [9.1, 9.0]]), torch.tensor([10, 10]))
        self.assertEqual(len(bank.means[20]), 2)
        scores = bank.logits(torch.tensor([[4.0, 4.0], [9.0, 9.0]]), {10: 0, 20: 1})
        self.assertEqual(scores.argmax(dim=1).tolist(), [1, 0])
        before = bank.snapshot()
        bank.means[20][:] = 100
        bank.restore(before)
        np.testing.assert_array_less(bank.means[20], 10)


if __name__ == "__main__":
    unittest.main()
