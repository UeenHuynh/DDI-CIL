#!/usr/bin/env python3
"""D1-H29 seed-0 frozen-classifier and dynamic class-prior diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_c4_grouprior_trajectories import load_task_model
from scripts.run_c4_classifier_diagnostics import (
    DEFAULT_GRID,
    GROUPS,
    frequency_group,
    group_direction,
    infer,
    metric_rows,
)
from src.data.class_mapping import build_seen_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.training.train_cil import load_task_spec, resolve_device


def class_direction(logits: np.ndarray, labels: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Validation-fit log mass correction for each class with additive smoothing."""
    classes = logits.shape[1]
    predicted = logits.argmax(axis=1)
    true_counts = np.bincount(labels, minlength=classes).astype(np.float64) + smoothing
    predicted_counts = np.bincount(predicted, minlength=classes).astype(np.float64) + smoothing
    true_mass = true_counts / true_counts.sum()
    predicted_mass = predicted_counts / predicted_counts.sum()
    return np.log(true_mass / predicted_mass).astype(np.float32)


def macro_f1(labels: np.ndarray, predictions: np.ndarray, groups: dict[str, np.ndarray]) -> float:
    return float(metric_rows(labels, predictions, groups)[0]["f1"])


def choose_strength(
    logits: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
    groups: dict[str, np.ndarray],
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    for strength in DEFAULT_GRID:
        score = macro_f1(labels, (logits + strength * direction).argmax(1), groups)
        rows.append({"strength": float(strength), "validation_macro_f1": score})
    best = max(rows, key=lambda row: (row["validation_macro_f1"], -row["strength"]))
    return float(best["strength"]), rows


def train_balanced_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    groups: dict[str, np.ndarray],
    device: torch.device,
    seed: int,
) -> tuple[nn.Linear, list[dict[str, float]]]:
    torch.manual_seed(seed)
    counts = np.bincount(train_labels, minlength=sum(len(v) for v in groups.values()))
    weights = 1.0 / np.maximum(counts[train_labels], 1)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_labels), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
        batch_size=1024, sampler=sampler, pin_memory=device.type == "cuda",
    )
    head = nn.Linear(train_features.shape[1], len(counts)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    validation_tensor = torch.from_numpy(validation_features)
    best_state = None
    best_score, stale = -1.0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, 21):
        head.train()
        total_loss, total_rows = 0.0, 0
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_rows += len(labels)
        head.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(validation_tensor), 4096):
                chunks.append(head(validation_tensor[start:start + 4096].to(device)).cpu().numpy())
        logits = np.concatenate(chunks)
        score = macro_f1(validation_labels, logits.argmax(1), groups)
        history.append({"epoch": epoch, "train_loss": total_loss / total_rows, "validation_macro_f1": score})
        print(f"[balanced-head] epoch={epoch} val_macro_f1={score:.6f}", flush=True)
        if score > best_score:
            best_score, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("Balanced head produced no checkpoint")
    head.load_state_dict(best_state)
    return head.eval(), history


def head_infer(head: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(features)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), 4096):
            chunks.append(head(tensor[start:start + 4096].to(device)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c4-run", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(resolve_device(args.device))
    config = json.loads((args.c4_run / "run_config.json").read_text())["arguments"]
    feature_columns = load_feature_columns(PROJECT_ROOT / "outputs/audit/feature_columns.json")
    scaler = load_scaler_payload(PROJECT_ROOT / "outputs/preprocess/scaler.pkl")
    tasks = load_task_spec(args.task_file)["tasks"]
    raw_classes = sorted(int(c) for task in tasks for c in task["classes"])
    class_map = build_seen_class_map(raw_classes)
    train = load_split_arrays(PROJECT_ROOT / "train_extracted.parquet", feature_columns, class_ids=raw_classes, scaler_payload=scaler)
    validation = load_split_arrays(PROJECT_ROOT / "validation_extracted.parquet", feature_columns, class_ids=raw_classes, scaler_payload=scaler)
    test = load_split_arrays(PROJECT_ROOT / "test_extracted.parquet", feature_columns, class_ids=raw_classes, scaler_payload=scaler)
    for arrays in (train, validation, test):
        arrays.labels = remap_labels(arrays.labels, class_map)
    counts = pd.read_csv(PROJECT_ROOT / "outputs/class_distribution/class_counts_train.csv")
    count_by_class = dict(zip(counts.class_id.astype(int), counts["count"].astype(int)))
    groups = {
        group: np.asarray([class_map[c] for c in raw_classes if frequency_group(count_by_class[c]) == group], dtype=np.int64)
        for group in GROUPS
    }

    model = load_task_model(args.c4_run, class_map, len(feature_columns), len(tasks) - 1, device)
    validation_logits = infer(model, validation.features, device)
    test_logits = infer(model, test.features, device)
    group_offset = group_direction(validation_logits, validation.labels, groups)
    dynamic_offset = class_direction(validation_logits, validation.labels)
    dynamic_strength, grid = choose_strength(validation_logits, validation.labels, dynamic_offset, groups)

    print("[features] extracting frozen C4 latents", flush=True)
    train_latent = infer(model, train.features, device, latent=True)
    validation_latent = infer(model, validation.features, device, latent=True)
    test_latent = infer(model, test.features, device, latent=True)
    balanced_head, history = train_balanced_head(
        train_latent, train.labels, validation_latent, validation.labels, groups, device, args.seed
    )
    balanced_validation_logits = head_infer(balanced_head, validation_latent, device)
    balanced_test_logits = head_infer(balanced_head, test_latent, device)
    balanced_group_offset = group_direction(balanced_validation_logits, validation.labels, groups)

    schemes = {
        "C4": test_logits.argmax(1),
        "C4+GroupPrior": (test_logits + group_offset).argmax(1),
        "C4+DynamicClassPrior": (test_logits + dynamic_strength * dynamic_offset).argmax(1),
        "C4+FrozenBalancedClassifier": balanced_test_logits.argmax(1),
        "C4+FrozenBalancedClassifier+GroupPrior": (
            balanced_test_logits + balanced_group_offset
        ).argmax(1),
    }
    rows = []
    for scheme, predictions in schemes.items():
        for row in metric_rows(test.labels, predictions, groups):
            rows.append({"setting": "D1-H29", "seed": args.seed, "scheme": scheme, **row})

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "endpoint_metrics.csv", index=False)
    pd.DataFrame(grid).to_csv(args.output / "dynamic_class_prior_validation_grid.csv", index=False)
    pd.DataFrame(history).to_csv(args.output / "balanced_classifier_training.csv", index=False)
    torch.save(balanced_head.state_dict(), args.output / "balanced_classifier.pt")
    (args.output / "contract.json").write_text(json.dumps({
        "setting": "D1-H29", "seed": args.seed, "backbone": "tddi",
        "source_run": str(args.c4_run), "source_method": config["method"],
        "balanced_classifier": "frozen C4 backbone; inverse-frequency sampling; cross-entropy",
        "dynamic_class_prior": "validation p_c/q_c with additive count smoothing=1",
        "dynamic_strength_grid": list(DEFAULT_GRID),
        "selected_dynamic_strength": dynamic_strength,
        "test_used_for_selection": False,
    }, indent=2))
    print(metrics.query("group == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
