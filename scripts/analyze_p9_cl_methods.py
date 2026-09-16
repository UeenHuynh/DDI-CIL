#!/usr/bin/env python3
"""Summarize the separate P9-H8 T-DDI CL method tracks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_IDS = ("C0", "C1", "C2", "C3", "C4", "C6", "C0O", "C1O", "C5")
OFFLINE = ("C0", "C1", "C2", "C4", "C6")
ONLINE = ("C0O", "C1O", "C5")


def run_dir(method_id: str, seed: int, output_root: Path) -> Path:
    if method_id == "C3":
        return PROJECT_ROOT / "outputs/runs_long_tail" / (
            f"p9_tail_profile_balanced_seed{seed}_replay_distill_fixed_budget_uniform_mlptddi"
        )
    return output_root / f"p9_h8_tddi_{method_id.lower()}_seed{seed}"


def read_run(path: Path, method_id: str, seed: int) -> dict[str, object] | None:
    required = ["run_summary.md", "metrics.csv", "forgetting.csv", "class_trajectory.csv", "events.csv"]
    if any(not (path / name).is_file() for name in required):
        return None
    if "run_completed" not in (path / "events.csv").read_text():
        return None
    metrics = pd.read_csv(path / "metrics.csv")
    final = metrics[
        (metrics.train_task_id.astype(str) == "7")
        & (metrics.eval_task_id.astype(str) == "seen_all")
        & (metrics.split == "test_seen_all")
    ]
    if len(final) != 1:
        raise ValueError(f"Expected one final seen-class test row in {path}.")
    forgetting = pd.read_csv(path / "forgetting.csv")
    old = forgetting[forgetting.task_id.astype(str) == "mean_old_tasks"]
    if len(old) != 1:
        raise ValueError(f"Expected mean_old_tasks in {path}.")
    classes = pd.read_csv(path / "class_trajectory.csv")
    classes = classes[classes.train_task.astype(int) == 7].copy()
    if len(classes) != 178:
        raise ValueError(f"Expected 178 final class rows in {path}.")
    classes["group"] = np.select(
        [classes.train_count > 1000, classes.train_count > 100],
        ["head", "medium"], default="tail",
    )
    grouped = classes.groupby("group").f1.mean()
    record = final.iloc[0]
    return {
        "method_id": method_id,
        "track": "online" if method_id.endswith("O") or method_id == "C5" else "offline",
        "seed": seed,
        "run_dir": str(path),
        "macro_f1": float(record.macro_f1),
        "balanced_accuracy": float(record.balanced_accuracy),
        "task_forgetting": float(old.iloc[0].forgetting),
        "f1_head": float(grouped.get("head", np.nan)),
        "f1_medium": float(grouped.get("medium", np.nan)),
        "f1_tail": float(grouped.get("tail", np.nan)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs/runs_p9_cl_methods",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    rows = []
    for method_id in METHOD_IDS:
        for seed in range(5):
            record = read_run(run_dir(method_id, seed, output_root), method_id, seed)
            if record is not None:
                rows.append(record)
    if not rows:
        raise SystemExit("No complete P9-H8 method runs found.")
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(analysis_dir / "method_per_seed.csv", index=False)
    metrics = ["macro_f1", "balanced_accuracy", "task_forgetting", "f1_head", "f1_medium", "f1_tail"]
    summary = frame.groupby(["track", "method_id"])[metrics].agg(["count", "mean", "std"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(analysis_dir / "method_summary.csv", index=False)
    pairs = []
    for method_id in (*OFFLINE, *ONLINE):
        baseline = "C1O" if method_id == "C5" else "C3" if method_id in OFFLINE else "C0O"
        if method_id == baseline:
            continue
        left = frame[frame.method_id == method_id].set_index("seed")
        right = frame[frame.method_id == baseline].set_index("seed")
        for seed in sorted(set(left.index) & set(right.index)):
            pairs.append({
                "method_id": method_id, "baseline_id": baseline, "seed": seed,
                **{f"delta_{metric}": float(left.loc[seed, metric] - right.loc[seed, metric]) for metric in metrics},
            })
    pd.DataFrame(pairs).to_csv(analysis_dir / "paired_deltas.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
