#!/usr/bin/env python3
"""Diagnose C3/C4 precision-recall, plasticity, retention and robustness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_mapping import build_seen_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.training.train_cil import expand_model_for_seen_classes, load_task_spec, resolve_device


GROUPS = ("head", "medium", "tail")


def frequency_group(count: int) -> str:
    if count > 1000:
        return "head"
    if count > 100:
        return "medium"
    return "tail"


def complete(run_dir: Path) -> bool:
    events = run_dir / "events.csv"
    return (
        (run_dir / "class_trajectory.csv").is_file()
        and (run_dir / "task_matrix.csv").is_file()
        and events.is_file()
        and "run_completed" in events.read_text(encoding="utf-8")
    )


def candidates(study_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "protocol": "JOINT", "horizon": 1, "method_id": "J0", "seed": 0,
            "run_dir": study_root / "j0_joint_natural_seed0",
        },
        {
            "protocol": "JOINT", "horizon": 1, "method_id": "J1", "seed": 0,
            "run_dir": study_root / "j1_joint_balanced_seed0",
        },
    ]
    for seed in range(5):
        rows.extend([
            {
                "protocol": "P9", "horizon": 8, "method_id": "C3", "seed": seed,
                "run_dir": PROJECT_ROOT / "outputs/runs_long_tail" /
                f"p9_tail_profile_balanced_seed{seed}_replay_distill_fixed_budget_uniform_mlptddi",
            },
            {
                "protocol": "P9", "horizon": 8, "method_id": "C4", "seed": seed,
                "run_dir": PROJECT_ROOT / "outputs/runs_p9_cl_methods" /
                f"p9_h8_tddi_c4_seed{seed}",
            },
            {
                "protocol": "P4", "horizon": 8, "method_id": "C3", "seed": seed,
                "run_dir": PROJECT_ROOT / "outputs/runs_backbones" /
                f"p4_constrained_mass_balanced_seed{seed}_replay_distill_fixed_budget_uniform_mlptddi",
            },
            {
                "protocol": "P4", "horizon": 8, "method_id": "C4", "seed": seed,
                "run_dir": study_root / f"p4_h8_tddi_c4_seed{seed}",
            },
        ])
    for seed in range(3):
        rows.extend([
            {
                "protocol": "P9", "horizon": 15, "method_id": "C3", "seed": seed,
                "run_dir": (
                    PROJECT_ROOT / "outputs/runs_h15_merging" /
                    "p9_tail_profile_balanced_h15_seed0_none_alpha0p5"
                    if seed == 0 else study_root / f"p9_h15_tddi_c3_seed{seed}"
                ),
            },
            {
                "protocol": "P9", "horizon": 15, "method_id": "C4", "seed": seed,
                "run_dir": study_root / f"p9_h15_tddi_c4_seed{seed}",
            },
        ])
    return [row for row in rows if complete(Path(row["run_dir"]))]


def summarize_runs(run_rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_rows: list[dict[str, object]] = []
    retention_rows: list[dict[str, object]] = []
    for meta in run_rows:
        run_dir = Path(meta["run_dir"])
        trajectory = pd.read_csv(run_dir / "class_trajectory.csv")
        matrix = pd.read_csv(run_dir / "task_matrix.csv")
        final_task = int(matrix.train_task_id.max())
        final = trajectory[trajectory.train_task.astype(int) == final_task].copy()
        final["group"] = final.train_count.astype(int).map(frequency_group)
        for group in ("all", *GROUPS):
            part = final if group == "all" else final[final.group == group]
            endpoint_rows.append({
                **meta, "group": group, "class_count": len(part),
                "macro_precision": part.precision.mean(),
                "macro_recall": part.recall.mean(),
                "macro_f1": part.f1.mean(),
            })

        final_matrix = matrix.loc[matrix.train_task_id.astype(int) == final_task].iloc[0]
        for task_id in range(final_task + 1):
            column = f"test_task_{task_id}"
            immediate = float(matrix.loc[matrix.train_task_id.astype(int) == task_id, column].iloc[0])
            final_f1 = float(final_matrix[column])
            history = matrix.loc[matrix.train_task_id.astype(int) >= task_id, column].dropna().astype(float)
            retention_rows.append({
                **meta, "task_id": task_id, "task_age": final_task - task_id,
                "new_task_f1": immediate, "final_task_f1": final_f1,
                "retention_ratio": final_f1 / immediate if immediate > 0 else np.nan,
                "forgetting": float(history.max() - final_f1),
            })
    return pd.DataFrame(endpoint_rows), pd.DataFrame(retention_rows)


def infer_distributions(run_rows: list[dict[str, object]], output: Path, device: str) -> None:
    selected = [
        row for row in run_rows
        if row["protocol"] == "P9" and row["horizon"] == 8
    ]
    if not selected:
        return
    cache = output / "prediction_distribution.csv"
    completed_keys: set[tuple[str, int]] = set()
    old_distribution = pd.DataFrame()
    old_confusion = pd.DataFrame()
    if cache.is_file():
        old_distribution = pd.read_csv(cache)
        completed_keys = set(zip(old_distribution.method_id, old_distribution.seed))
        confusion_path = output / "confusion_counts.csv"
        if confusion_path.is_file():
            old_confusion = pd.read_csv(confusion_path)

    pending = [row for row in selected if (row["method_id"], row["seed"]) not in completed_keys]
    if not pending:
        return
    first_config = json.loads((Path(pending[0]["run_dir"]) / "run_config.json").read_text())
    args = first_config["arguments"]
    feature_columns = load_feature_columns(args["feature_cols"])
    scaler = load_scaler_payload(args["scaler"])
    task_spec = load_task_spec(Path(args["task_file"]))
    raw_classes = sorted(int(c) for task in task_spec["tasks"] for c in task["classes"])
    class_map = build_seen_class_map(raw_classes)
    arrays = load_split_arrays(
        args["test"], feature_columns, class_ids=raw_classes, scaler_payload=scaler
    )
    local_labels = remap_labels(arrays.labels, class_map)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays.features), torch.from_numpy(local_labels)),
        batch_size=1024, shuffle=False,
    )
    true_counts = np.bincount(local_labels, minlength=len(raw_classes))
    group_by_class = {
        int(row.class_id): frequency_group(int(row.train_count))
        for row in pd.read_csv(Path(pending[0]["run_dir"]) / "class_trajectory.csv").itertuples()
    }
    distribution_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    resolved_device = resolve_device(device)
    for meta in pending:
        run_dir = Path(meta["run_dir"])
        config = json.loads((run_dir / "run_config.json").read_text())["arguments"]
        model = expand_model_for_seen_classes(
            None, None, class_map, variant="tddi", input_dim=arrays.features.shape[1],
            dropout=float(config["dropout"]), activation=str(config["activation"]),
            norm=str(config["norm"]),
        )
        model.load_state_dict(torch.load(
            run_dir / "checkpoints/task_7_model.pt", map_location="cpu", weights_only=True
        ))
        model.to(resolved_device).eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for features, _ in loader:
                predictions.append(model(features.to(resolved_device)).argmax(1).cpu().numpy())
        local_pred = np.concatenate(predictions)
        predicted_counts = np.bincount(local_pred, minlength=len(raw_classes))
        matrix = confusion_matrix(local_labels, local_pred, labels=np.arange(len(raw_classes)))
        for index, raw_class in enumerate(raw_classes):
            p = true_counts[index] / true_counts.sum()
            q = predicted_counts[index] / predicted_counts.sum()
            distribution_rows.append({
                "method_id": meta["method_id"], "seed": meta["seed"],
                "class_id": raw_class, "group": group_by_class[raw_class],
                "true_count": int(true_counts[index]),
                "predicted_count": int(predicted_counts[index]),
                "true_fraction": p, "predicted_fraction": q,
                "prediction_to_true_ratio": q / p if p else np.nan,
            })
        true_index, pred_index = np.nonzero(matrix)
        for ti, pi in zip(true_index, pred_index, strict=True):
            confusion_rows.append({
                "method_id": meta["method_id"], "seed": meta["seed"],
                "true_class": raw_classes[int(ti)], "predicted_class": raw_classes[int(pi)],
                "count": int(matrix[ti, pi]),
            })
        print(f"[inference] {meta['method_id']} seed={meta['seed']}", flush=True)
    pd.concat([old_distribution, pd.DataFrame(distribution_rows)], ignore_index=True).to_csv(
        cache, index=False
    )
    pd.concat([old_confusion, pd.DataFrame(confusion_rows)], ignore_index=True).to_csv(
        output / "confusion_counts.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root", type=Path,
        default=PROJECT_ROOT / "outputs/runs_c3_c4_diagnostics",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    output = args.study_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    run_rows = candidates(args.study_root)
    endpoint, retention = summarize_runs(run_rows)
    endpoint.to_csv(output / "precision_recall_f1_per_seed.csv", index=False)
    retention.to_csv(output / "plasticity_retention_per_task.csv", index=False)
    endpoint.groupby(["protocol", "horizon", "method_id", "group"])[
        ["macro_precision", "macro_recall", "macro_f1"]
    ].agg(["count", "mean", "std"]).to_csv(output / "precision_recall_f1_summary.csv")
    retention.groupby(["protocol", "horizon", "method_id"])[
        ["new_task_f1", "final_task_f1", "retention_ratio", "forgetting"]
    ].mean().to_csv(output / "plasticity_retention_summary.csv")
    if not args.skip_inference:
        infer_distributions(run_rows, output, args.device)
    print(f"[done] analyzed {len(run_rows)} complete runs under {output}")


if __name__ == "__main__":
    main()
