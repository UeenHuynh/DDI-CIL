"""Online multi-centroid OT adaptation for the DDI descriptor stream.

This adapts OTC/MMOT's dynamic Gaussian centroids, balanced transport,
representation preservation, centroid-guided memory and Mahalanobis inference
to a task-labelled P9 stream. Task boundaries are used for reporting only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class OnlineMixtureBank:
    max_centroids: int = 3
    split_threshold: float = 0.25
    transport_temperature: float = 0.1
    variance_floor: float = 0.01
    means: dict[int, np.ndarray] = field(default_factory=dict)
    variances: dict[int, np.ndarray] = field(default_factory=dict)
    masses: dict[int, np.ndarray] = field(default_factory=dict)

    def snapshot(self) -> dict[str, dict[int, np.ndarray]]:
        return {
            "means": {k: v.copy() for k, v in self.means.items()},
            "variances": {k: v.copy() for k, v in self.variances.items()},
            "masses": {k: v.copy() for k, v in self.masses.items()},
        }

    def restore(self, state: dict[str, dict[int, np.ndarray]]) -> None:
        self.means = {k: v.copy() for k, v in state["means"].items()}
        self.variances = {k: v.copy() for k, v in state["variances"].items()}
        self.masses = {k: v.copy() for k, v in state["masses"].items()}

    def update(self, embeddings: torch.Tensor, raw_labels: torch.Tensor) -> None:
        values = embeddings.detach().cpu().numpy().astype(np.float32)
        labels = raw_labels.detach().cpu().numpy().astype(np.int64)
        for raw in np.unique(labels):
            raw_id = int(raw)
            batch = values[labels == raw]
            if raw_id not in self.means:
                self.means[raw_id] = batch[:1].copy()
                self.variances[raw_id] = np.full_like(batch[:1], self.variance_floor)
                self.masses[raw_id] = np.ones(1, dtype=np.float32)
            means = self.means[raw_id]
            var = self.variances[raw_id]
            distance = ((batch[:, None] - means[None]) ** 2 / var[None]).mean(axis=2)
            farthest = int(np.argmax(distance.min(axis=1)))
            if (
                len(means) < self.max_centroids
                and distance[farthest].min() > self.split_threshold
                and self.masses[raw_id].sum() >= 2
            ):
                self.means[raw_id] = np.concatenate([means, batch[farthest : farthest + 1]])
                self.variances[raw_id] = np.concatenate([
                    var, np.full_like(var[:1], self.variance_floor)
                ])
                self.masses[raw_id] = np.concatenate([
                    self.masses[raw_id], np.ones(1, dtype=np.float32)
                ])
                means = self.means[raw_id]
                var = self.variances[raw_id]
                distance = ((batch[:, None] - means[None]) ** 2 / var[None]).mean(axis=2)
            costs = distance - distance.min(axis=1, keepdims=True)
            weights = np.exp(-costs / max(self.transport_temperature, 1e-6))
            weights = np.maximum(weights, 1e-12)
            # Entropic transport with uniform centroid capacity and row mass.
            for _ in range(4):
                weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
                weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
                weights *= len(batch) / len(means)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            count = weights.sum(axis=0).astype(np.float32)
            old_mass = self.masses[raw_id]
            new_mass = old_mass + count
            new_means = (
                old_mass[:, None] * means + weights.T @ batch
            ) / new_mass[:, None]
            # Each centroid gets its own weighted diagonal variance.
            batch_var = np.stack([
                (weights[:, k, None] * (batch - new_means[k]) ** 2).sum(axis=0)
                for k in range(len(means))
            ])
            new_var = (
                old_mass[:, None] * var + batch_var
            ) / new_mass[:, None]
            self.means[raw_id] = new_means.astype(np.float32)
            self.variances[raw_id] = np.maximum(new_var, self.variance_floor).astype(np.float32)
            self.masses[raw_id] = new_mass

    def logits(self, embeddings: torch.Tensor, class_map: dict[int, int]) -> torch.Tensor:
        if not self.means:
            raise RuntimeError("OTC inference requires initialized centroids.")
        raw_order = [raw for raw, _ in sorted(class_map.items(), key=lambda item: item[1])]
        means = []
        variances = []
        owners = []
        for local, raw in enumerate(raw_order):
            if raw not in self.means:
                continue
            means.extend(self.means[raw])
            variances.extend(self.variances[raw])
            owners.extend([local] * len(self.means[raw]))
        center = torch.as_tensor(np.stack(means), device=embeddings.device)
        var = torch.as_tensor(np.stack(variances), device=embeddings.device)
        precision = var.reciprocal()
        distances = (
            embeddings.square() @ precision.T
            - 2 * embeddings @ (center * precision).T
            + (center.square() * precision).sum(dim=1)[None]
        ) / embeddings.shape[1]
        scores = embeddings.new_full((len(embeddings), len(raw_order)), -1e6)
        for local in range(len(raw_order)):
            selected = [index for index, owner in enumerate(owners) if owner == local]
            if selected:
                scores[:, local] = -distances[:, selected].min(dim=1).values
        return scores

    def memory_ranking(self, embeddings: np.ndarray, raw_labels: np.ndarray) -> dict[int, np.ndarray]:
        orders = {}
        for raw in np.unique(raw_labels):
            raw_id = int(raw)
            block = embeddings[raw_labels == raw]
            centers = self.means[raw_id]
            var = self.variances[raw_id]
            distances = ((block[:, None] - centers[None]) ** 2 / var[None]).mean(axis=2)
            orders[raw_id] = np.lexsort((np.arange(len(block)), distances.min(axis=1)))
        return orders


class OTCInference(nn.Module):
    def __init__(self, backbone: nn.Module, bank: OnlineMixtureBank, class_map: dict[int, int]):
        super().__init__()
        self.backbone = backbone
        self.bank = bank
        self.class_map = class_map

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bank.logits(self.backbone.encode(features), self.class_map)


def centroid_preservation_loss(
    embeddings: torch.Tensor,
    local_labels: torch.Tensor,
    bank: OnlineMixtureBank,
    class_map: dict[int, int],
) -> torch.Tensor:
    return F.cross_entropy(bank.logits(embeddings, class_map), local_labels)
