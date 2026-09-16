import unittest

import torch

from src.training.train_cil import merge_post_task_state


class PostTaskMergingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old = {
            "layers.0.weight": torch.zeros(2, 2),
            "head.weight": torch.tensor([[0.0, 0.0], [2.0, 2.0], [9.0, 9.0]]),
            "head.bias": torch.tensor([0.0, 2.0, 9.0]),
        }
        self.new = {
            "layers.0.weight": torch.full((2, 2), 2.0),
            "head.weight": torch.tensor([[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]]),
            "head.bias": torch.tensor([2.0, 4.0, 8.0]),
        }
        self.previous_map = {10: 0, 20: 1}
        self.current_map = {10: 0, 20: 1, 30: 2}

    def merge(self, strategy: str):
        return merge_post_task_state(
            self.old,
            self.new,
            strategy=strategy,
            alpha=0.5,
            previous_seen_map=self.previous_map,
            current_seen_map=self.current_map,
            variant="tddi",
        )

    def test_ema_backbone_does_not_merge_classifier(self) -> None:
        merged = self.merge("ema_backbone")
        torch.testing.assert_close(merged["layers.0.weight"], torch.ones(2, 2))
        torch.testing.assert_close(merged["head.weight"], self.new["head.weight"])
        torch.testing.assert_close(merged["head.bias"], self.new["head.bias"])

    def test_selective_merges_old_rows_and_preserves_new_rows(self) -> None:
        merged = self.merge("selective")
        torch.testing.assert_close(
            merged["head.weight"],
            torch.tensor([[1.0, 1.0], [3.0, 3.0], [8.0, 8.0]]),
        )
        torch.testing.assert_close(merged["head.bias"], torch.tensor([1.0, 3.0, 8.0]))

    def test_invalid_alpha_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "merge alpha"):
            merge_post_task_state(
                self.old,
                self.new,
                strategy="selective",
                alpha=1.1,
                previous_seen_map=self.previous_map,
                current_seen_map=self.current_map,
                variant="tddi",
            )


if __name__ == "__main__":
    unittest.main()
