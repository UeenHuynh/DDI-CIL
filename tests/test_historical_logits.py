from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data.fixed_budget_replay import FixedBudgetReplayBuffer
from src.data.historical_logits import HistoricalLogitBank


class HistoricalLogitBankTest(unittest.TestCase):
    def test_write_time_logits_survive_head_expansion_and_budget_shrink(self) -> None:
        memory = FixedBudgetReplayBuffer(total_memory_budget=4)
        bank = HistoricalLogitBank((10, 20, 30))
        first = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
        memory.update(first, np.full(3, 20, dtype=np.int64))
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 0.0]]))
        bank.after_buffer_update(memory, model, {20: 0}, "cpu", batch_size=2)
        before = bank.get_all(memory)
        self.assertTrue(np.isnan(before[:, 0]).all())
        self.assertTrue(np.isnan(before[:, 2]).all())
        np.testing.assert_array_equal(before[:, 1], [4.0, 2.0, 6.0])

        second = np.asarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=np.float32)
        memory.update(second, np.full(3, 10, dtype=np.int64))
        expanded = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            expanded.weight.copy_(torch.tensor([[0.0, 3.0], [4.0, 0.0]]))
        bank.after_buffer_update(memory, expanded, {10: 0, 20: 1}, "cpu", batch_size=2)
        old = bank.logits_by_class[20]
        self.assertEqual(old.shape, (2, 3))
        np.testing.assert_array_equal(old[:, 1], [4.0, 2.0])
        self.assertTrue(np.isnan(old[:, 0]).all())
        self.assertTrue(np.isfinite(bank.logits_by_class[10][:, :2]).all())

        bank.after_buffer_update(
            memory, expanded, {10: 0, 20: 1}, "cpu", batch_size=2,
            refresh_new_class_columns=True,
        )
        self.assertTrue(np.isfinite(bank.logits_by_class[20][:, 0]).all())
        np.testing.assert_array_equal(bank.logits_by_class[20][:, 1], [4.0, 2.0])


if __name__ == "__main__":
    unittest.main()
