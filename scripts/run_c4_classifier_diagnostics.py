#!/usr/bin/env python3
"""Run post-hoc C4 logit correction and frozen-backbone linear probes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_mapping import build_seen_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.training.train_cil import FocalLoss, expand_model_for_seen_classes, load_task_spec, resolve_device


GROUPS = ("head", "medium", "tail")
DEFAULT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def frequency_group(count: int) -> str:
    if count > 1000:
        return "head"
    if count > 100:
        return "medium"
    return "tail"


def method_run(method: str, seed: int) -> Path:
    if method == "C3":
        return PROJECT_ROOT / "outputs/runs_long_tail" / (
            f"p9_tail_profile_balanced_seed{seed}_"
            "replay_distill_fixed_budget_uniform_mlptddi"
        )
    if method == "C4":
        return PROJECT_ROOT / "outputs/runs_p9_cl_methods" / f"p9_h8_tddi_c4_seed{seed}"
    raise ValueError(f"Unsupported method: {method}")


@dataclass(frozen=True)
class SharedData:
    raw_classes: list[int]
    class_map: dict[int, int]
    feature_columns: list[str]
    scaler: dict[str, object]
    config: dict[str, object]
    train_counts: np.ndarray
    group_indices: dict[str, np.ndarray]


def load_shared(seed: int = 0) -> SharedData:
    run_dir = method_run("C4", seed)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["arguments"]
    feature_columns = load_feature_columns(config["feature_cols"])
    scaler = load_scaler_payload(config["scaler"])
    task_spec = load_task_spec(Path(config["task_file"]))
    raw_classes = sorted(int(c) for task in task_spec["tasks"] for c in task["classes"])
    class_map = build_seen_class_map(raw_classes)
    counts_frame = pd.read_csv(PROJECT_ROOT / "outputs/class_distribution/class_counts_train.csv")
    count_by_class = dict(zip(counts_frame.class_id.astype(int), counts_frame["count"].astype(int)))
    train_counts = np.asarray([count_by_class[c] for c in raw_classes], dtype=np.float64)
    group_indices = {
        group: np.asarray(
            [i for i, count in enumerate(train_counts) if frequency_group(int(count)) == group],
            dtype=np.int64,
        )
        for group in GROUPS
    }
    return SharedData(
        raw_classes, class_map, feature_columns, scaler, config, train_counts, group_indices
    )


def load_arrays(shared: SharedData, split: str):
    arrays = load_split_arrays(
        shared.config[split], shared.feature_columns, class_ids=shared.raw_classes,
        scaler_payload=shared.scaler,
    )
    arrays.labels = remap_labels(arrays.labels, shared.class_map)
    return arrays


def load_final_model(shared: SharedData, method: str, seed: int, device: torch.device) -> nn.Module:
    run_dir = method_run(method, seed)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["arguments"]
    model = expand_model_for_seen_classes(
        None, None, shared.class_map, variant="tddi",
        input_dim=len(shared.feature_columns), dropout=float(config["dropout"]),
        activation=str(config["activation"]), norm=str(config["norm"]),
    )
    model.load_state_dict(torch.load(
        run_dir / "checkpoints/task_7_model.pt", map_location="cpu", weights_only=True
    ))
    return model.to(device).eval()


def load_task_model(
    shared: SharedData, method: str, seed: int, task_id: int,
    seen_map: dict[int, int], device: torch.device,
) -> nn.Module:
    run_dir = method_run(method, seed)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["arguments"]
    model = expand_model_for_seen_classes(
        None, None, seen_map, variant="tddi", input_dim=len(shared.feature_columns),
        dropout=float(config["dropout"]), activation=str(config["activation"]),
        norm=str(config["norm"]),
    )
    model.load_state_dict(torch.load(
        run_dir / f"checkpoints/task_{task_id}_model.pt", map_location="cpu", weights_only=True
    ))
    return model.to(device).eval()


def infer(model: nn.Module, features: np.ndarray, device: torch.device, *, latent: bool = False) -> np.ndarray:
    if not features.flags.writeable:
        features = features.copy()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)), batch_size=1024, shuffle=False,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            values = model.encode(batch) if latent else model(batch)
            chunks.append(values.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks)


def metric_rows(
    labels: np.ndarray,
    predictions: np.ndarray,
    group_indices: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    class_labels = np.arange(sum(len(v) for v in group_indices.values()))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_labels, zero_division=0
    )
    rows: list[dict[str, object]] = []
    for group in ("all", *GROUPS):
        indices = class_labels if group == "all" else group_indices[group]
        true_count = int(np.isin(labels, indices).sum())
        predicted_count = int(np.isin(predictions, indices).sum())
        mean_precision = float(precision[indices].mean()) if len(indices) else np.nan
        mean_recall = float(recall[indices].mean()) if len(indices) else np.nan
        mean_f1 = float(f1[indices].mean()) if len(indices) else np.nan
        rows.append({
            "group": group,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": mean_f1,
            "true_count": true_count,
            "predicted_count": predicted_count,
            "prediction_to_true_ratio": predicted_count / true_count if true_count else np.nan,
        })
    return rows


def group_direction(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    group_indices: dict[str, np.ndarray],
) -> np.ndarray:
    predictions = validation_logits.argmax(axis=1)
    offsets = np.empty(validation_logits.shape[1], dtype=np.float32)
    for group, indices in group_indices.items():
        true_mass = np.isin(validation_labels, indices).mean()
        predicted_mass = np.isin(predictions, indices).mean()
        direction = -np.log(max(predicted_mass, 1e-12) / max(true_mass, 1e-12))
        offsets[indices] = direction
    return offsets


def choose_strength(
    logits: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
    group_indices: dict[str, np.ndarray],
    grid: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for strength in grid:
        predictions = (logits + strength * direction).argmax(axis=1)
        score = metric_rows(labels, predictions, group_indices)[0]["f1"]
        rows.append({"strength": strength, "macro_f1": float(score)})
    best = max(rows, key=lambda row: (row["macro_f1"], -row["strength"]))
    return float(best["strength"]), rows


def run_calibration(output: Path, seeds: list[int], device: torch.device) -> None:
    shared = load_shared()
    validation = load_arrays(shared, "validation")
    test = load_arrays(shared, "test")
    class_direction = np.log(shared.train_counts / shared.train_counts.sum()).astype(np.float32)
    selection_rows: list[dict[str, object]] = []
    metric_output: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    for seed in seeds:
        model = load_final_model(shared, "C4", seed, device)
        validation_logits = infer(model, validation.features, device)
        test_logits = infer(model, test.features, device)
        directions = {
            "none": np.zeros(len(shared.raw_classes), dtype=np.float32),
            "group_prior": group_direction(
                validation_logits, validation.labels, shared.group_indices
            ),
            "class_prior": class_direction,
        }
        for scheme, direction in directions.items():
            if scheme == "none":
                strength, grid_rows = 0.0, [{"strength": 0.0, "macro_f1": metric_rows(
                    validation.labels, validation_logits.argmax(1), shared.group_indices
                )[0]["f1"]}]
            else:
                strength, grid_rows = choose_strength(
                    validation_logits, validation.labels, direction,
                    shared.group_indices, DEFAULT_GRID,
                )
            for row in grid_rows:
                selection_rows.append({"seed": seed, "scheme": scheme, **row})
            predictions = (test_logits + strength * direction).argmax(axis=1)
            for row in metric_rows(test.labels, predictions, shared.group_indices):
                metric_output.append({
                    "seed": seed, "scheme": scheme, "selected_strength": strength, **row
                })
            for index, raw_class in enumerate(shared.raw_classes):
                direction_rows.append({
                    "seed": seed, "scheme": scheme, "class_id": raw_class,
                    "group": next(g for g, ids in shared.group_indices.items() if index in ids),
                    "direction": float(direction[index]),
                })
        print(f"[calibration] seed={seed} complete", flush=True)
        del model, validation_logits, test_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(output / "calibration_validation_grid.csv", index=False)
    metrics = pd.DataFrame(metric_output)
    metrics.to_csv(output / "calibration_test_per_seed.csv", index=False)
    metrics.groupby(["scheme", "group"])[
        ["precision", "recall", "f1", "prediction_to_true_ratio"]
    ].agg(["count", "mean", "std"]).to_csv(output / "calibration_test_summary.csv")
    pd.DataFrame(direction_rows).to_csv(output / "calibration_directions.csv", index=False)


def train_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    group_indices: dict[str, np.ndarray],
    *, seed: int, device: torch.device, max_epochs: int = 20, patience: int = 5,
) -> tuple[nn.Linear, int, float]:
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[1], len(group_indices["head"]) +
                     len(group_indices["medium"]) + len(group_indices["tail"])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = FocalLoss(gamma=1.0)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
        batch_size=1024, shuffle=True, generator=generator,
        pin_memory=device.type == "cuda",
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score, best_epoch, stale = -1.0, 0, 0
    validation_tensor = torch.from_numpy(validation_features)
    for epoch in range(1, max_epochs + 1):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                head(features.to(device, non_blocking=True)),
                labels.to(device, non_blocking=True),
            )
            loss.backward()
            optimizer.step()
        head.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(validation_tensor), 4096):
                chunks.append(head(validation_tensor[start:start + 4096].to(device)).argmax(1).cpu().numpy())
        score = float(metric_rows(validation_labels, np.concatenate(chunks), group_indices)[0]["f1"])
        print(f"[probe] epoch={epoch} val_macro_f1={score:.6f}", flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    head.load_state_dict(best_state)
    return head.eval(), best_epoch, best_score


def load_memory(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(run_dir / "memory/memory_after_task_7.parquet")
    labels = frame.pop("raw_class_id").to_numpy(dtype=np.int64)
    return frame.to_numpy(dtype=np.float32, copy=False), labels


def run_probes(output: Path, seeds: list[int], device: torch.device) -> None:
    shared = load_shared()
    train = load_arrays(shared, "train")
    validation = load_arrays(shared, "validation")
    test = load_arrays(shared, "test")
    result_rows: list[dict[str, object]] = []
    for seed in seeds:
        for method in ("C3", "C4"):
            print(f"[probe] extracting method={method} seed={seed}", flush=True)
            model = load_final_model(shared, method, seed, device)
            validation_latent = infer(model, validation.features, device, latent=True)
            test_latent = infer(model, test.features, device, latent=True)
            for probe in ("oracle", "memory"):
                if probe == "oracle":
                    probe_features = infer(model, train.features, device, latent=True)
                    probe_labels = train.labels
                else:
                    memory_features, memory_raw_labels = load_memory(method_run(method, seed))
                    probe_features = infer(model, memory_features, device, latent=True)
                    probe_labels = remap_labels(memory_raw_labels, shared.class_map)
                print(f"[probe] training method={method} seed={seed} probe={probe}", flush=True)
                head, best_epoch, validation_f1 = train_probe(
                    probe_features, probe_labels, validation_latent, validation.labels,
                    shared.group_indices, seed=seed, device=device,
                )
                predictions: list[np.ndarray] = []
                test_tensor = torch.from_numpy(test_latent)
                with torch.no_grad():
                    for start in range(0, len(test_tensor), 4096):
                        predictions.append(head(test_tensor[start:start + 4096].to(device)).argmax(1).cpu().numpy())
                for row in metric_rows(test.labels, np.concatenate(predictions), shared.group_indices):
                    result_rows.append({
                        "seed": seed, "method": method, "probe": probe,
                        "best_epoch": best_epoch, "validation_macro_f1": validation_f1, **row,
                    })
                del probe_features, head
            del model, validation_latent, test_latent
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(output / "probe_test_per_seed.csv", index=False)
    results.groupby(["probe", "method", "group"])[
        ["precision", "recall", "f1", "prediction_to_true_ratio"]
    ].agg(["count", "mean", "std"]).to_csv(output / "probe_test_summary.csv")


def run_deployable_correction(output: Path, seeds: list[int], device: torch.device) -> None:
    """Fit group correction after every task using seen-class validation only."""
    shared = load_shared()
    validation = load_arrays(shared, "validation")
    test = load_arrays(shared, "test")
    metric_output: list[dict[str, object]] = []
    task_scores: list[dict[str, object]] = []
    calibrator_rows: list[dict[str, object]] = []
    for seed in seeds:
        config = json.loads(
            (method_run("C4", seed) / "run_config.json").read_text(encoding="utf-8")
        )["arguments"]
        tasks = load_task_spec(Path(config["task_file"]))["tasks"]
        seen_raw: list[int] = []
        for task_id, task in enumerate(tasks):
            seen_raw.extend(int(c) for c in task["classes"])
            seen_raw = sorted(set(seen_raw))
            seen_map = build_seen_class_map(seen_raw)
            global_indices = np.asarray([shared.class_map[c] for c in seen_raw], dtype=np.int64)
            validation_mask = np.isin(validation.labels, global_indices)
            test_mask = np.isin(test.labels, global_indices)
            local_lookup = np.full(len(shared.raw_classes), -1, dtype=np.int64)
            local_lookup[global_indices] = np.arange(len(seen_raw))
            validation_labels = local_lookup[validation.labels[validation_mask]]
            test_labels = local_lookup[test.labels[test_mask]]
            local_groups = {
                group: np.asarray(
                    [seen_map[c] for c in seen_raw if frequency_group(int(shared.train_counts[shared.class_map[c]])) == group],
                    dtype=np.int64,
                )
                for group in GROUPS
            }
            model = load_task_model(shared, "C4", seed, task_id, seen_map, device)
            validation_logits = infer(model, validation.features[validation_mask], device)
            test_logits = infer(model, test.features[test_mask], device)
            direction = group_direction(validation_logits, validation_labels, local_groups)
            strength, _ = choose_strength(
                validation_logits, validation_labels, direction, local_groups, DEFAULT_GRID
            )
            schemes = {
                "none": test_logits.argmax(1),
                "group_prior": (test_logits + strength * direction).argmax(1),
            }
            for scheme, predictions in schemes.items():
                for row in metric_rows(test_labels, predictions, local_groups):
                    metric_output.append({
                        "seed": seed, "train_task": task_id, "scheme": scheme,
                        "selected_strength": 0.0 if scheme == "none" else strength, **row,
                    })
                for eval_task_id, eval_task in enumerate(tasks[:task_id + 1]):
                    eval_raw = [int(c) for c in eval_task["classes"]]
                    eval_local = np.asarray([seen_map[c] for c in eval_raw], dtype=np.int64)
                    eval_mask = np.isin(test_labels, eval_local)
                    task_f1 = f1_score(
                        test_labels[eval_mask], predictions[eval_mask], labels=eval_local,
                        average="macro", zero_division=0,
                    )
                    task_scores.append({
                        "seed": seed, "scheme": scheme, "train_task": task_id,
                        "eval_task": eval_task_id, "task_f1": float(task_f1),
                    })
            for local_index, raw_class in enumerate(seen_raw):
                calibrator_rows.append({
                    "seed": seed, "task": task_id, "selected_strength": strength,
                    "class_id": raw_class,
                    "group": frequency_group(int(shared.train_counts[shared.class_map[raw_class]])),
                    "direction": float(direction[local_index]),
                    "offset": float(strength * direction[local_index]),
                })
            print(
                f"[deploy] seed={seed} task={task_id} strength={strength:.2f}", flush=True
            )
            del model, validation_logits, test_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_output)
    scores = pd.DataFrame(task_scores)
    metrics.to_csv(output / "deployable_metrics_per_task.csv", index=False)
    scores.to_csv(output / "deployable_task_matrix_long.csv", index=False)
    pd.DataFrame(calibrator_rows).to_csv(output / "deployable_calibrators.csv", index=False)
    retention_rows: list[dict[str, object]] = []
    for (seed, scheme), part in scores.groupby(["seed", "scheme"]):
        final_task = int(part.train_task.max())
        for eval_task in range(final_task + 1):
            trajectory = part[part.eval_task == eval_task].sort_values("train_task")
            immediate = float(trajectory.loc[trajectory.train_task == eval_task, "task_f1"].iloc[0])
            final = float(trajectory.loc[trajectory.train_task == final_task, "task_f1"].iloc[0])
            retention_rows.append({
                "seed": seed, "scheme": scheme, "eval_task": eval_task,
                "new_task_f1": immediate, "final_task_f1": final,
                "retention_ratio": final / immediate if immediate > 0 else np.nan,
                "forgetting": float(trajectory.task_f1.max() - final),
            })
    retention = pd.DataFrame(retention_rows)
    retention.to_csv(output / "deployable_retention_per_task.csv", index=False)
    retention.groupby("scheme")[[
        "new_task_f1", "final_task_f1", "retention_ratio", "forgetting"
    ]].agg(["count", "mean", "std"]).to_csv(output / "deployable_retention_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("calibrate", "probe", "deploy", "all"))
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--probe-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--deploy-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/runs_c4_classifier_diagnostics/analysis",
    )
    args = parser.parse_args()
    device = torch.device(resolve_device(args.device))
    if args.experiment in {"calibrate", "all"}:
        run_calibration(args.output, args.calibration_seeds, device)
    if args.experiment in {"probe", "all"}:
        run_probes(args.output, args.probe_seeds, device)
    if args.experiment in {"deploy", "all"}:
        run_deployable_correction(args.output, args.deploy_seeds, device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostic_contract.json").write_text(json.dumps({
        "selection_split": "validation",
        "evaluation_split": "test",
        "selection_metric": "macro_f1",
        "strength_grid": list(DEFAULT_GRID),
        "calibration_seeds": args.calibration_seeds,
        "probe_seeds": args.probe_seeds,
        "deploy_seeds": args.deploy_seeds,
        "probe_classifier": "linear_focal_gamma1_adamw_lr1e-3_wd1e-4",
        "test_used_for_selection": False,
    }, indent=2), encoding="utf-8")
    print(f"[done] outputs written to {args.output}")


if __name__ == "__main__":
    main()
