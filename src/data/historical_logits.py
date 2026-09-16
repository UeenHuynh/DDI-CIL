"""Write-time replay logits aligned by raw class ID across expanding heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - training requires torch
    torch = None

from src.data.fixed_budget_replay import FixedBudgetReplayBuffer


@dataclass
class HistoricalLogitBank:
    """Logits occupy a fixed raw-class axis; unseen columns remain NaN.

    The exemplar order follows ``FixedBudgetReplayBuffer.get_all`` exactly.
    Old rows are only truncated when the fixed budget is reallocated. DER++
    keeps write-time values; X-DER may fill columns for classes introduced
    after the exemplar was written.
    """

    raw_class_order: tuple[int, ...]
    logits_by_class: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.raw_class_order)) != len(self.raw_class_order):
            raise ValueError("Raw class order must contain each class once.")
        self._column = {raw: index for index, raw in enumerate(self.raw_class_order)}

    @property
    def width(self) -> int:
        return len(self.raw_class_order)

    def get_all(self, buffer: FixedBudgetReplayBuffer) -> np.ndarray:
        blocks = []
        for raw in buffer.classes:
            block = self.logits_by_class.get(raw)
            count = buffer.memory_counts[raw]
            if block is None or block.shape != (count, self.width):
                raise RuntimeError(f"Missing historical logits for class {raw}.")
            blocks.append(block)
        if not blocks:
            return np.empty((0, self.width), dtype=np.float32)
        return np.concatenate(blocks, axis=0)

    @staticmethod
    def _predict(model, features: np.ndarray, device: str, batch_size: int) -> np.ndarray:
        if torch is None:
            raise ImportError("torch is required for historical logit capture.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        was_training = model.training
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = torch.as_tensor(features[start : start + batch_size], device=device)
                chunks.append(model(batch).detach().cpu().numpy().astype(np.float32))
        model.train(was_training)
        return np.concatenate(chunks, axis=0)

    def after_buffer_update(
        self,
        buffer: FixedBudgetReplayBuffer,
        model,
        current_seen_map: dict[int, int],
        device: str,
        *,
        batch_size: int,
        refresh_new_class_columns: bool = False,
        future_margin: float = 0.3,
    ) -> None:
        """Capture new exemplars and optionally revise old future-class logits."""
        seen_columns = np.asarray([self._column[raw] for raw in current_seen_map], dtype=np.int64)
        model_columns = np.asarray([current_seen_map[raw] for raw in current_seen_map], dtype=np.int64)
        if sorted(model_columns.tolist()) != list(range(len(current_seen_map))):
            raise ValueError("Current class map is not a dense classifier head.")

        for raw in buffer.classes:
            target_count = buffer.memory_counts[raw]
            if raw in buffer.last_selected_indices_by_class:
                features = buffer.features_by_class[raw]
                predictions = self._predict(model, features, device, batch_size)
                block = np.full((target_count, self.width), np.nan, dtype=np.float32)
                block[:, seen_columns] = predictions[:, model_columns]
                self.logits_by_class[raw] = block
            else:
                old = self.logits_by_class.get(raw)
                if old is None or len(old) < target_count:
                    raise RuntimeError(f"Cannot align retained logits for class {raw}.")
                block = old[:target_count].copy()
                if refresh_new_class_columns and target_count:
                    predictions = self._predict(
                        model, buffer.features_by_class[raw], device, batch_size
                    )
                    missing = np.isnan(block[:, seen_columns])
                    updated = block[:, seen_columns]
                    proposed = predictions[:, model_columns]
                    true_old = block[:, self._column[raw]]
                    proposed = np.minimum(proposed, true_old[:, None] - future_margin)
                    updated[missing] = proposed[missing]
                    block[:, seen_columns] = updated
                self.logits_by_class[raw] = block
        self.get_all(buffer)
