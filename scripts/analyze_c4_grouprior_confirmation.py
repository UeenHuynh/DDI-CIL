#!/usr/bin/env python3
"""Aggregate the seed-0/1/2 C4+GroupPrior robustness confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = PROJECT_ROOT / "outputs/runs_c4_grouprior_screening/analysis"


def main() -> None:
    endpoints = pd.concat(
        [
            pd.read_csv(ANALYSIS / "seed0_endpoint_metrics.csv"),
            pd.read_csv(ANALYSIS / "seed1_2_endpoint_metrics.csv"),
        ],
        ignore_index=True,
    )
    grids = pd.concat(
        [
            pd.read_csv(ANALYSIS / "seed0_validation_grid.csv"),
            pd.read_csv(ANALYSIS / "seed1_2_validation_grid.csv"),
        ],
        ignore_index=True,
    )
    endpoints.to_csv(ANALYSIS / "three_seed_endpoint_metrics.csv", index=False)
    summary = endpoints.groupby(["setting", "candidate", "group"])[
        ["precision", "recall", "f1", "prediction_to_true_ratio"]
    ].agg(["count", "mean", "std"])
    summary.to_csv(ANALYSIS / "three_seed_endpoint_summary.csv")

    all_group = endpoints[endpoints.group == "all"].pivot(
        index=["setting", "seed"], columns="candidate", values="f1"
    ).reset_index()
    all_group["gain_gp_vs_c3"] = all_group["C4+GroupPrior"] - all_group["C3"]
    all_group["gain_gp_vs_c4"] = all_group["C4+GroupPrior"] - all_group["C4"]
    all_group.to_csv(ANALYSIS / "three_seed_paired_gains.csv", index=False)
    gain_summary = all_group.groupby("setting").agg(
        gain_gp_vs_c3_mean=("gain_gp_vs_c3", "mean"),
        gain_gp_vs_c3_std=("gain_gp_vs_c3", "std"),
        gain_gp_vs_c3_min=("gain_gp_vs_c3", "min"),
        gp_beats_c3_count=("gain_gp_vs_c3", lambda values: int((values > 0).sum())),
        gain_gp_vs_c4_mean=("gain_gp_vs_c4", "mean"),
    ).reset_index()
    gain_summary.to_csv(ANALYSIS / "three_seed_gain_summary.csv", index=False)

    selected = grids.sort_values(["setting", "seed", "strength"]).loc[
        grids.groupby(["setting", "seed"])["validation_macro_f1"].idxmax()
    ].sort_values(["setting", "seed"])
    selected.to_csv(ANALYSIS / "three_seed_selected_lambda.csv", index=False)
    locked = bool(
        (gain_summary.gp_beats_c3_count == 3).all()
        and (gain_summary.gain_gp_vs_c3_min > 0).all()
        and (selected.strength == 1.0).all()
    )
    (ANALYSIS / "three_seed_decision.json").write_text(json.dumps({
        "candidate": "C4-GroupPrior-v1",
        "locked": locked,
        "seeds": [0, 1, 2],
        "settings": ["P4-H8", "P9-H15", "P4-H15"],
        "wins_over_c3": int((all_group.gain_gp_vs_c3 > 0).sum()),
        "comparisons": int(len(all_group)),
        "selected_lambda_all_comparisons": selected.strength.astype(float).tolist(),
        "deferred": ["five-seed final table", "H29", "dynamic/random arrival"],
    }, indent=2), encoding="utf-8")
    print(gain_summary.to_string(index=False))
    print(f"locked={locked}")


if __name__ == "__main__":
    main()
