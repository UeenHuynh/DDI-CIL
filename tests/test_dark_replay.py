from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.methods.dark_replay import derpp_objective, xder_old_new_margin_loss


class DarkReplayObjectiveTest(unittest.TestCase):
    def test_derpp_keeps_current_replay_ce_and_observed_logits_separate(self) -> None:
        logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
        labels = torch.tensor([1, 0])
        history = torch.tensor([[1.0, float("nan")]])
        loss = derpp_objective(
            logits, labels, torch.tensor([0, 1]), current_count=1,
            historical_logits=history, historical_raw_order=(10, 20),
            current_seen_map={10: 0, 20: 1}, criterion=F.cross_entropy,
            logit_weight=2.0, replay_ce_weight=3.0,
        )
        expected = (
            F.cross_entropy(logits[:1], labels[:1])
            + 3 * F.cross_entropy(logits[1:], labels[1:]) + 2.0
        )
        self.assertAlmostEqual(float(loss.detach()), float(expected.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_xder_margin_uses_old_and_new_rows_after_head_growth(self) -> None:
        logits = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        labels = torch.tensor([1, 0])
        penalty = xder_old_new_margin_loss(
            logits, labels, torch.tensor([0, 1]), current_count=1,
            old_indices=[0], margin=0.3,
        )
        self.assertAlmostEqual(float(penalty), 2.6, places=6)


if __name__ == "__main__":
    unittest.main()
