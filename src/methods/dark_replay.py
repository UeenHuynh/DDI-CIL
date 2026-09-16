"""DER++ objectives for fixed-exposure replay with an expanding classifier."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def derpp_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    example_indices: torch.Tensor,
    *,
    current_count: int,
    historical_logits: torch.Tensor,
    historical_raw_order: tuple[int, ...],
    current_seen_map: dict[int, int],
    criterion,
    logit_weight: float,
    replay_ce_weight: float,
) -> torch.Tensor:
    """Current-label loss + replay-label loss + masked historical-logit MSE.

    Historical columns are indexed by raw class ID. This permits logits
    written before the current task to survive arbitrary head-row insertion.
    """
    if current_count < 0 or logit_weight < 0 or replay_ce_weight < 0:
        raise ValueError("DER++ counts and weights must be non-negative.")
    indices = example_indices.to(logits.device)
    current_mask = indices < current_count
    replay_mask = ~current_mask
    loss = logits.sum() * 0.0
    if bool(current_mask.any()):
        loss = loss + criterion(logits[current_mask], labels[current_mask])
    if bool(replay_mask.any()):
        loss = loss + replay_ce_weight * criterion(
            logits[replay_mask], labels[replay_mask]
        )
        global_columns = [
            historical_raw_order.index(raw)
            for raw, _ in sorted(current_seen_map.items(), key=lambda item: item[1])
        ]
        replay_rows = indices[replay_mask] - current_count
        targets = historical_logits[replay_rows][:, global_columns]
        observed = torch.isfinite(targets)
        if not bool(observed.any()):
            raise RuntimeError("Replay exemplars have no historical logits.")
        residual = torch.where(observed, logits[replay_mask] - targets, 0.0)
        loss = loss + logit_weight * residual.square().sum() / observed.sum()
    return loss


def xder_old_new_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    example_indices: torch.Tensor,
    *,
    current_count: int,
    old_indices: list[int],
    margin: float,
) -> torch.Tensor:
    """Project adaptation of X-DER's past/future head constraints."""
    if not old_indices or len(old_indices) == logits.shape[1]:
        return logits.sum() * 0.0
    old_columns = torch.as_tensor(old_indices, device=logits.device)
    new_columns = torch.as_tensor(
        [index for index in range(logits.shape[1]) if index not in old_indices],
        device=logits.device,
    )
    current_mask = example_indices.to(logits.device) < current_count
    replay_mask = ~current_mask
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    terms = []
    if bool(current_mask.any()):
        max_old = logits[current_mask][:, old_columns].max(dim=1).values
        terms.append(F.relu(max_old + margin - true_logits[current_mask]).mean())
    if bool(replay_mask.any()):
        max_new = logits[replay_mask][:, new_columns].max(dim=1).values
        terms.append(F.relu(max_new + margin - true_logits[replay_mask]).mean())
    return sum(terms) if terms else logits.sum() * 0.0
