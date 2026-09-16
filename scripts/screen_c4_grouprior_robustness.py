#!/usr/bin/env python3
"""Seed-0 robustness screen for C3, C4, and validation-fit C4+GroupPrior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.training.train_cil import expand_model_for_seen_classes, load_task_spec, resolve_device


def settings_for_seed(seed: int) -> dict[str, dict[str, Path]]:
    p9_h15_c3 = (
        PROJECT_ROOT / "outputs/runs_h15_merging/"
        "p9_tail_profile_balanced_h15_seed0_none_alpha0p5"
        if seed == 0 else
        PROJECT_ROOT / f"outputs/runs_c3_c4_diagnostics/p9_h15_tddi_c3_seed{seed}"
    )
    p4_h15_c3 = (
        PROJECT_ROOT / "outputs/runs_h15_merging/"
        "p4_constrained_mass_balanced_h15_seed0_none_alpha0p5"
        if seed == 0 else
        PROJECT_ROOT / f"outputs/runs_c4_grouprior_screening/p4_h15_tddi_c3_seed{seed}"
    )
    return {
        "P4-H8": {
            "c3": PROJECT_ROOT / "outputs/runs_backbones" /
            f"p4_constrained_mass_balanced_seed{seed}_replay_distill_fixed_budget_uniform_mlptddi",
            "c4": PROJECT_ROOT / "outputs/runs_c3_c4_diagnostics" /
            f"p4_h8_tddi_c4_seed{seed}",
        },
        "P9-H15": {
            "c3": p9_h15_c3,
            "c4": PROJECT_ROOT / "outputs/runs_c3_c4_diagnostics" /
            f"p9_h15_tddi_c4_seed{seed}",
        },
        "P4-H15": {
            "c3": p4_h15_c3,
            "c4": PROJECT_ROOT / "outputs/runs_c4_grouprior_screening" /
            f"p4_h15_tddi_c4_seed{seed}",
        },
    }


def load_model(run_dir: Path, class_map: dict[int, int], input_dim: int, device: torch.device):
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["arguments"]
    tasks = load_task_spec(Path(config["task_file"]))["tasks"]
    final_task = len(tasks) - 1
    model = expand_model_for_seen_classes(
        None, None, class_map, variant="tddi", input_dim=input_dim,
        dropout=float(config["dropout"]), activation=str(config["activation"]),
        norm=str(config["norm"]),
    )
    model.load_state_dict(torch.load(
        run_dir / f"checkpoints/task_{final_task}_model.pt",
        map_location="cpu", weights_only=True,
    ))
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/runs_c4_grouprior_screening/analysis",
    )
    args = parser.parse_args()
    device = torch.device(resolve_device(args.device))
    reference_settings = settings_for_seed(args.seeds[0])
    reference_config = json.loads(
        (reference_settings["P4-H8"]["c4"] / "run_config.json").read_text(encoding="utf-8")
    )["arguments"]
    feature_columns = load_feature_columns(reference_config["feature_cols"])
    scaler = load_scaler_payload(reference_config["scaler"])
    task_spec = load_task_spec(Path(reference_config["task_file"]))
    raw_classes = sorted(int(c) for task in task_spec["tasks"] for c in task["classes"])
    class_map = build_seen_class_map(raw_classes)
    counts = pd.read_csv(PROJECT_ROOT / "outputs/class_distribution/class_counts_train.csv")
    count_by_class = dict(zip(counts.class_id.astype(int), counts["count"].astype(int)))
    groups = {
        group: np.asarray(
            [class_map[c] for c in raw_classes if frequency_group(count_by_class[c]) == group],
            dtype=np.int64,
        )
        for group in GROUPS
    }
    validation = load_split_arrays(
        reference_config["validation"], feature_columns, class_ids=raw_classes,
        scaler_payload=scaler,
    )
    test = load_split_arrays(
        reference_config["test"], feature_columns, class_ids=raw_classes,
        scaler_payload=scaler,
    )
    validation.labels = remap_labels(validation.labels, class_map)
    test.labels = remap_labels(test.labels, class_map)

    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for setting, run_dirs in settings_for_seed(seed).items():
            missing = [str(path) for path in run_dirs.values() if not (path / "run_summary.md").is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete runs for {setting} seed={seed}: {missing}")
            for method in ("c3", "c4"):
                model = load_model(run_dirs[method], class_map, len(feature_columns), device)
                if method == "c4":
                    validation_logits = infer(model, validation.features, device)
                test_logits = infer(model, test.features, device)
                predictions_by_candidate = {method.upper(): test_logits.argmax(1)}
                if method == "c4":
                    direction = group_direction(validation_logits, validation.labels, groups)
                    grid = []
                    for strength in DEFAULT_GRID:
                        val_predictions = (validation_logits + strength * direction).argmax(1)
                        score = float(metric_rows(validation.labels, val_predictions, groups)[0]["f1"])
                        grid.append((strength, score))
                        selection_rows.append({
                            "setting": setting, "seed": seed, "strength": strength,
                            "validation_macro_f1": score,
                        })
                    strength = max(grid, key=lambda item: (item[1], -item[0]))[0]
                    predictions_by_candidate["C4+GroupPrior"] = (
                        test_logits + strength * direction
                    ).argmax(1)
                else:
                    strength = 0.0
                for candidate, predictions in predictions_by_candidate.items():
                    for metric in metric_rows(test.labels, predictions, groups):
                        rows.append({
                            "setting": setting, "seed": seed, "candidate": candidate,
                            "selected_strength": strength if candidate == "C4+GroupPrior" else 0.0,
                            **metric,
                        })
                print(f"[screen] {setting} seed={seed} {method.upper()} complete", flush=True)
                del model, test_logits
                if method == "c4":
                    del validation_logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_tag = "seed" + "_".join(str(seed) for seed in args.seeds)
    pd.DataFrame(rows).to_csv(args.output / f"{seed_tag}_endpoint_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output / f"{seed_tag}_validation_grid.csv", index=False)
    (args.output / f"{seed_tag}_screening_contract.json").write_text(json.dumps({
        "seeds": args.seeds,
        "selection_split": "validation",
        "evaluation_split": "test",
        "selection_metric": "macro_f1",
        "strength_grid": list(DEFAULT_GRID),
        "test_used_for_selection": False,
        "deferred": ["seeds 3-4", "H29", "dynamic/random arrival"],
    }, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
