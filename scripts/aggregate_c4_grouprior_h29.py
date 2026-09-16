#!/usr/bin/env python3
"""Aggregate the locked P9-H29 C4-GroupPrior-v1 evaluation across seeds 0-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "final_macro_f1",
    "final_macro_precision",
    "final_macro_recall_balanced_accuracy",
    "head_f1",
    "medium_f1",
    "tail_f1",
    "new_task_f1",
    "final_task_f1",
    "forgetting",
    "bwt",
    "retention_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True, type=Path)
    args = parser.parse_args()
    summary_paths = [
        args.analysis_root / "h29_seed0_summary.csv",
        args.analysis_root / "seed1/h29_seed0_summary.csv",
        args.analysis_root / "seed2/h29_seed0_summary.csv",
    ]
    endpoint_paths = [
        args.analysis_root / "endpoint_metrics_per_task.csv",
        args.analysis_root / "seed1/endpoint_metrics_per_task.csv",
        args.analysis_root / "seed2/endpoint_metrics_per_task.csv",
    ]
    missing = [str(path) for path in summary_paths + endpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing H29 analysis artifacts: {missing}")

    per_seed = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    if sorted(per_seed.seed.unique().tolist()) != [0, 1, 2]:
        raise ValueError(f"Expected seeds 0,1,2; found {sorted(per_seed.seed.unique())}")
    if set(per_seed.candidate) != {"C3", "C4", "C4+GroupPrior"}:
        raise ValueError("Expected exactly C3, C4, and C4+GroupPrior")
    per_seed.to_csv(args.analysis_root / "h29_three_seed_per_seed.csv", index=False)

    aggregate = per_seed.groupby("candidate", sort=False)[METRICS].agg(["count", "mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.reset_index().to_csv(
        args.analysis_root / "h29_three_seed_aggregate.csv", index=False
    )

    c3 = per_seed.loc[per_seed.candidate == "C3"].set_index("seed")
    paired_rows: list[dict[str, object]] = []
    for candidate in ("C4", "C4+GroupPrior"):
        current = per_seed.loc[per_seed.candidate == candidate].set_index("seed")
        for seed in (0, 1, 2):
            row: dict[str, object] = {"seed": seed, "candidate": candidate}
            for metric in METRICS:
                row[f"delta_{metric}_vs_c3"] = float(current.loc[seed, metric] - c3.loc[seed, metric])
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.analysis_root / "h29_three_seed_paired_deltas.csv", index=False)
    delta_columns = [column for column in paired if column.startswith("delta_")]
    delta_summary = paired.groupby("candidate", sort=False)[delta_columns].agg(
        ["mean", "std", lambda values: int((values > 0).sum())]
    )
    delta_summary.columns = [
        f"{metric}_{'wins' if stat == '<lambda_0>' else stat}"
        for metric, stat in delta_summary.columns
    ]
    delta_summary.reset_index().to_csv(
        args.analysis_root / "h29_three_seed_paired_delta_summary.csv", index=False
    )

    endpoints = pd.concat([pd.read_csv(path) for path in endpoint_paths], ignore_index=True)
    final = endpoints.loc[endpoints.train_task == 28]
    ratios = final[["seed", "candidate", "group", "prediction_to_true_ratio"]]
    ratios.to_csv(args.analysis_root / "h29_three_seed_prediction_ratios.csv", index=False)
    ratios.groupby(["candidate", "group"], sort=False).prediction_to_true_ratio.agg(
        ["count", "mean", "std"]
    ).reset_index().to_csv(
        args.analysis_root / "h29_three_seed_prediction_ratio_aggregate.csv", index=False
    )
    (args.analysis_root / "three_seed_contract.json").write_text(json.dumps({
        "setting": "P9-H29",
        "seeds": [0, 1, 2],
        "candidate": "C4-GroupPrior-v1",
        "lambda": 1.0,
        "lambda_tuned_on_h29": False,
        "test_used_for_calibration": False,
        "configuration_changed_between_seeds": False,
    }, indent=2), encoding="utf-8")
    print(aggregate.to_string(), flush=True)
    print(f"[done] {args.analysis_root}", flush=True)


if __name__ == "__main__":
    main()
