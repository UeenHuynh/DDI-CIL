#!/usr/bin/env python3
"""Aggregate locked H29 C4-GroupPrior-v1 evaluations across arbitrary seeds."""

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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--setting", default="P9-H29")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--summary-template",
        default="h29_seed0_summary.csv",
        help="Summary filename template; may contain {seed} for per-seed names.",
    )
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if not seeds or seeds[0] != 0:
        raise ValueError("Aggregation currently requires seed 0 in the analysis root")
    # Keep the historical three-seed filenames used by docs and downstream
    # scripts while allowing explicit names for newer seed sets.
    output_prefix = args.output_prefix or (
        "h29_three_seed" if seeds == [0, 1, 2] else f"h29_{len(seeds)}seed"
    )
    summary_paths = [
        args.analysis_root
        / ("" if seed == 0 else f"seed{seed}")
        / args.summary_template.format(seed=seed)
        for seed in seeds
    ]
    endpoint_paths = [
        args.analysis_root / ("endpoint_metrics_per_task.csv" if seed == 0 else f"seed{seed}/endpoint_metrics_per_task.csv")
        for seed in seeds
    ]
    missing = [str(path) for path in summary_paths + endpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing H29 analysis artifacts: {missing}")

    per_seed = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    if sorted(per_seed.seed.unique().tolist()) != seeds:
        raise ValueError(f"Expected seeds {seeds}; found {sorted(per_seed.seed.unique())}")
    if set(per_seed.setting) != {args.setting}:
        raise ValueError(f"Expected setting {args.setting}; found {sorted(per_seed.setting.unique())}")
    if set(per_seed.candidate) != {"C3", "C4", "C4+GroupPrior"}:
        raise ValueError("Expected exactly C3, C4, and C4+GroupPrior")
    per_seed.to_csv(args.analysis_root / f"{output_prefix}_per_seed.csv", index=False)

    aggregate = per_seed.groupby("candidate", sort=False)[METRICS].agg(["count", "mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.reset_index().to_csv(
        args.analysis_root / f"{output_prefix}_aggregate.csv", index=False
    )

    c3 = per_seed.loc[per_seed.candidate == "C3"].set_index("seed")
    paired_rows: list[dict[str, object]] = []
    for candidate in ("C4", "C4+GroupPrior"):
        current = per_seed.loc[per_seed.candidate == candidate].set_index("seed")
        for seed in seeds:
            row: dict[str, object] = {"seed": seed, "candidate": candidate}
            for metric in METRICS:
                row[f"delta_{metric}_vs_c3"] = float(current.loc[seed, metric] - c3.loc[seed, metric])
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.analysis_root / f"{output_prefix}_paired_deltas.csv", index=False)
    delta_columns = [column for column in paired if column.startswith("delta_")]
    delta_summary = paired.groupby("candidate", sort=False)[delta_columns].agg(
        ["mean", "std", lambda values: int((values > 0).sum())]
    )
    delta_summary.columns = [
        f"{metric}_{'wins' if stat == '<lambda_0>' else stat}"
        for metric, stat in delta_summary.columns
    ]
    delta_summary.reset_index().to_csv(
        args.analysis_root / f"{output_prefix}_paired_delta_summary.csv", index=False
    )

    endpoints = pd.concat([pd.read_csv(path) for path in endpoint_paths], ignore_index=True)
    final = endpoints.loc[endpoints.train_task == 28]
    ratios = final[["seed", "candidate", "group", "prediction_to_true_ratio"]]
    ratios.to_csv(args.analysis_root / f"{output_prefix}_prediction_ratios.csv", index=False)
    ratios.groupby(["candidate", "group"], sort=False).prediction_to_true_ratio.agg(
        ["count", "mean", "std"]
    ).reset_index().to_csv(
        args.analysis_root / f"{output_prefix}_prediction_ratio_aggregate.csv", index=False
    )
    contract_name = (
        "three_seed_contract.json"
        if args.output_prefix is None and seeds == [0, 1, 2]
        else f"{output_prefix}_contract.json"
    )
    (args.analysis_root / contract_name).write_text(json.dumps({
        "setting": args.setting,
        "seeds": seeds,
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
