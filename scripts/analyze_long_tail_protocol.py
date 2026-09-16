#!/usr/bin/env python3
"""Derive P9-P12 long-tail and temporal metrics from an existing CIL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ("head", "medium", "tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--shock-task", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def frequency_group(count: int) -> str:
    if count > 1000:
        return "head"
    if count > 100:
        return "medium"
    return "tail"


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def protocol_metadata(run_dir: Path) -> tuple[str, int]:
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return str(config["resolved"]["task_protocol"]), int(config["arguments"]["seed"])


def grouped_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    frame = trajectory.copy()
    frame["frequency_group"] = frame["train_count"].astype(int).map(frequency_group)
    grouped = (
        frame.groupby(["train_task", "frequency_group"], sort=True)["f1"]
        .mean()
        .unstack("frequency_group")
        .reindex(columns=list(GROUPS))
        .reset_index()
        .rename(columns={group: f"f1_{group}" for group in GROUPS})
    )
    grouped["head_tail_gap"] = grouped["f1_head"] - grouped["f1_tail"]
    return grouped


def metric_at(frame: pd.DataFrame, task: int, column: str) -> float:
    row = frame.loc[frame["train_task"].astype(int) == task, column]
    return float(row.iloc[0]) if len(row) == 1 else float("nan")


def old_class_drop(trajectory: pd.DataFrame, shock_task: int) -> float:
    before = trajectory[
        (trajectory["train_task"].astype(int) == shock_task - 1)
        & (trajectory["first_task"].astype(int) < shock_task)
    ][["class_id", "f1"]].rename(columns={"f1": "before_f1"})
    after = trajectory[
        (trajectory["train_task"].astype(int) == shock_task)
        & (trajectory["first_task"].astype(int) < shock_task)
    ][["class_id", "f1"]].rename(columns={"f1": "after_f1"})
    paired = before.merge(after, on="class_id", validate="one_to_one")
    if paired.empty:
        return float("nan")
    # Match the user's definition: F1_old,t_s - F1_old,t_s-1.
    return float((paired["after_f1"] - paired["before_f1"]).mean())


def analyze_run(run_dir: Path, shock_task: int) -> tuple[dict[str, object], pd.DataFrame]:
    required = [
        run_dir / "run_summary.md",
        run_dir / "metrics.csv",
        run_dir / "forgetting.csv",
        run_dir / "task_matrix.csv",
        run_dir / "class_trajectory.csv",
        run_dir / "class_forgetting.csv",
        run_dir / "run_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}; missing: {missing}")

    protocol, seed = protocol_metadata(run_dir)
    metrics_path = run_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    require_columns(
        metrics, {"train_task_id", "eval_task_id", "macro_f1", "balanced_accuracy"}, metrics_path
    )
    seen = metrics[metrics["eval_task_id"].astype(str) == "seen_all"].copy()
    seen["train_task_id"] = seen["train_task_id"].astype(int)
    seen = seen.sort_values("train_task_id")
    final_task = int(seen["train_task_id"].max())
    final_seen = seen.iloc[-1]

    trajectory_path = run_dir / "class_trajectory.csv"
    trajectory = pd.read_csv(trajectory_path)
    require_columns(
        trajectory, {"train_task", "class_id", "first_task", "train_count", "f1"}, trajectory_path
    )
    group_by_task = grouped_trajectory(trajectory)
    group_by_task.insert(0, "seed", seed)
    group_by_task.insert(0, "protocol", protocol)

    class_forgetting_path = run_dir / "class_forgetting.csv"
    class_forgetting = pd.read_csv(class_forgetting_path)
    require_columns(
        class_forgetting, {"train_task", "train_count", "forgetting"}, class_forgetting_path
    )
    final_class = class_forgetting[
        class_forgetting["train_task"].astype(int) == final_task
    ].copy()
    final_class["frequency_group"] = final_class["train_count"].astype(int).map(frequency_group)
    forgetting_by_group = final_class.groupby("frequency_group")["forgetting"].mean()

    forgetting = pd.read_csv(run_dir / "forgetting.csv")
    task_forgetting_row = forgetting[forgetting["task_id"].astype(str) == "mean_old_tasks"]
    if len(task_forgetting_row) != 1:
        raise ValueError(f"Missing mean_old_tasks row in {run_dir / 'forgetting.csv'}")

    matrix = pd.read_csv(run_dir / "task_matrix.csv")
    matrix_columns = [f"test_task_{task}" for task in range(final_task + 1)]
    require_columns(matrix, {"train_task_id", *matrix_columns}, run_dir / "task_matrix.csv")
    matrix = matrix.sort_values("train_task_id")
    final_values = matrix.iloc[-1][matrix_columns].astype(float).to_numpy()
    diagonal = np.asarray(
        [float(matrix.iloc[task][f"test_task_{task}"]) for task in range(final_task + 1)]
    )
    bwt = float(np.mean(final_values[:-1] - diagonal[:-1])) if final_task else 0.0

    first_group = group_by_task.iloc[0]
    final_group = group_by_task.iloc[-1]
    summary: dict[str, object] = {
        "protocol": protocol,
        "seed": seed,
        "run_dir": str(run_dir),
        "final_macro_f1": float(final_seen["macro_f1"]),
        "final_balanced_accuracy": float(final_seen["balanced_accuracy"]),
        "task_forgetting": float(task_forgetting_row.iloc[0]["forgetting"]),
        "f1_head": float(final_group["f1_head"]),
        "f1_medium": float(final_group["f1_medium"]),
        "f1_tail": float(final_group["f1_tail"]),
        "head_tail_gap": float(final_group["head_tail_gap"]),
        "final_task_macro_f1_variance": float(np.var(final_values, ddof=0)),
        "bwt": bwt,
        "delta_tail": float(final_group["f1_tail"] - first_group["f1_tail"]),
        "delta_macro": float(final_seen["macro_f1"] - seen.iloc[0]["macro_f1"]),
        "head_forgetting": float(forgetting_by_group.get("head", np.nan)),
        "tail_forgetting": float(forgetting_by_group.get("tail", np.nan)),
    }
    summary["forgetting_imbalance"] = float(
        summary["tail_forgetting"] - summary["head_forgetting"]
    )

    if protocol == "tail_shock" and 0 < shock_task <= final_task:
        macro_before = float(seen.loc[seen["train_task_id"] == shock_task - 1, "macro_f1"].iloc[0])
        macro_shock = float(seen.loc[seen["train_task_id"] == shock_task, "macro_f1"].iloc[0])
        next_task = min(shock_task + 1, final_task)
        macro_next = float(seen.loc[seen["train_task_id"] == next_task, "macro_f1"].iloc[0])
        summary.update({
            "shock_task": shock_task,
            "shock_drop": macro_shock - macro_before,
            "recovery_next_task": macro_next - macro_shock,
            "recovery_final": float(final_seen["macro_f1"]) - macro_shock,
            "tail_gain_at_shock": metric_at(group_by_task, shock_task, "f1_tail")
            - metric_at(group_by_task, shock_task - 1, "f1_tail"),
            "old_drop_at_shock": old_class_drop(trajectory, shock_task),
        })
    return summary, group_by_task


def main() -> None:
    args = parse_args()
    summaries = []
    trajectories = []
    for run_dir in args.run_dirs:
        summary, trajectory = analyze_run(run_dir, args.shock_task)
        summaries.append(summary)
        trajectories.append(trajectory)
    summary_frame = pd.DataFrame(summaries)
    trajectory_frame = pd.concat(trajectories, ignore_index=True)
    output = args.output or args.run_dirs[0].parent / "long_tail_analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output / "long_tail_summary.csv", index=False)
    numeric_columns = [
        column for column in summary_frame.select_dtypes(include="number").columns
        if column != "seed"
    ]
    if numeric_columns:
        aggregate = summary_frame.groupby("protocol")[numeric_columns].agg(["mean", "std"])
        aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
        aggregate.reset_index().to_csv(output / "long_tail_aggregate_summary.csv", index=False)
    trajectory_frame.to_csv(output / "long_tail_metrics_by_task.csv", index=False)
    (output / "long_tail_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(f"[done] Wrote long-tail metrics to {output}")


if __name__ == "__main__":
    main()
