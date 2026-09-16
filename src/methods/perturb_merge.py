"""Full-network P&M adaptation for the expanding-head DDI CIL model.

The original P&M uses LoRA task vectors. This version uses the full T-DDI
parameter vector and preserves newly introduced classifier rows exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.methods.ewc import grow_head_state


@dataclass
class TaskOptimum:
    state: dict[str, torch.Tensor]
    fisher: dict[str, torch.Tensor]
    class_map: dict[int, int]


def optimal_merge_alpha(
    previous_merged: dict[str, torch.Tensor],
    current_optimum: dict[str, torch.Tensor],
    current_fisher: dict[str, torch.Tensor],
    current_map: dict[int, int],
    history: list[TaskOptimum],
) -> float:
    """Diagonal-Fisher closed-form alpha, clipped to the convex interval."""
    delta = {
        name: (current_optimum[name] - previous_merged[name]).float()
        for name in current_fisher
    }
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for optimum in [*history, TaskOptimum(current_optimum, current_fisher, current_map)]:
        aligned_state = grow_head_state(
            optimum.state, optimum.class_map, current_map,
            previous_merged, zero_new_rows=False,
        )
        aligned_fisher = grow_head_state(
            optimum.fisher, optimum.class_map, current_map,
            current_fisher, zero_new_rows=True,
        )
        for name, direction in delta.items():
            importance = aligned_fisher[name].float().abs()
            numerator -= (
                (previous_merged[name].float() - aligned_state[name].float())
                * importance * direction
            ).sum(dtype=torch.float64)
            denominator += (importance * direction.square()).sum(dtype=torch.float64)
    if denominator <= 0 or not torch.isfinite(denominator):
        return 1.0
    alpha = (numerator / denominator).clamp(0.0, 1.0)
    return float(alpha)


def merge_with_new_rows_intact(
    previous_merged: dict[str, torch.Tensor],
    current_optimum: dict[str, torch.Tensor],
    previous_map: dict[int, int],
    current_map: dict[int, int],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Interpolate shared parameters while keeping new-class head rows trained."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Merge alpha must lie in [0, 1].")
    merged = {}
    new_rows = [current_map[raw] for raw in current_map if raw not in previous_map]
    for name, trained in current_optimum.items():
        previous = previous_merged[name]
        if trained.shape != previous.shape:
            raise ValueError(f"Expanded state mismatch for {name}.")
        if not torch.is_floating_point(trained):
            merged[name] = trained.clone()
            continue
        value = torch.lerp(previous, trained, alpha)
        if name in {"head.weight", "head.bias"} and new_rows:
            value[new_rows] = trained[new_rows]
        merged[name] = value
    return merged
