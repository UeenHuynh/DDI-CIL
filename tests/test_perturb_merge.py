from __future__ import annotations

import unittest

import torch

from src.methods.perturb_merge import (
    TaskOptimum,
    merge_with_new_rows_intact,
    optimal_merge_alpha,
)


class PerturbMergeTest(unittest.TestCase):
    def test_fisher_alpha_balances_old_and_new_task_curvature(self) -> None:
        previous = {"weight": torch.tensor([0.0])}
        current = {"weight": torch.tensor([2.0])}
        old = TaskOptimum(previous, {"weight": torch.tensor([3.0])}, {10: 0})
        alpha = optimal_merge_alpha(
            previous, current, {"weight": torch.tensor([1.0])}, {10: 0}, [old]
        )
        self.assertAlmostEqual(alpha, 0.25)

    def test_new_class_head_row_is_not_pulled_back(self) -> None:
        previous = {
            "weight": torch.tensor([0.0]),
            "head.weight": torch.tensor([[1.0], [0.0]]),
            "head.bias": torch.tensor([1.0, 0.0]),
        }
        trained = {
            "weight": torch.tensor([2.0]),
            "head.weight": torch.tensor([[3.0], [5.0]]),
            "head.bias": torch.tensor([3.0, 5.0]),
        }
        merged = merge_with_new_rows_intact(
            previous, trained, {10: 0}, {10: 0, 20: 1}, 0.25
        )
        self.assertEqual(float(merged["weight"][0]), 0.5)
        self.assertEqual(float(merged["head.weight"][0, 0]), 1.5)
        self.assertEqual(float(merged["head.weight"][1, 0]), 5.0)
        self.assertEqual(float(merged["head.bias"][1]), 5.0)


if __name__ == "__main__":
    unittest.main()
