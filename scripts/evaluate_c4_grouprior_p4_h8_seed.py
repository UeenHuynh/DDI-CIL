#!/usr/bin/env python3
"""Evaluate locked C4+GroupPrior on one P4-H8 T-DDI seed."""

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

from scripts.run_c4_classifier_diagnostics import GROUPS, frequency_group, group_direction, infer, metric_rows
from scripts.screen_c4_grouprior_robustness import load_model, settings_for_seed
from src.data.class_mapping import build_seen_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.training.train_cil import load_task_spec, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c3-run", type=Path)
    parser.add_argument("--c4-run", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root / "configs/c4_grouprior_candidate.json").read_text())
    strength = float(candidate["correction"]["lambda"])
    if candidate["candidate_id"] != "C4-GroupPrior-v1" or strength != 1.0:
        raise ValueError("Unexpected locked GroupPrior candidate")

    runs = settings_for_seed(args.seed)["P4-H8"]
    if args.c3_run is not None:
        runs["c3"] = args.c3_run
    if args.c4_run is not None:
        runs["c4"] = args.c4_run
    for name, path in runs.items():
        if not (path / "run_summary.md").is_file():
            raise FileNotFoundError(f"Missing {name} run: {path}")
    config = json.loads((runs["c4"] / "run_config.json").read_text())["arguments"]
    feature_columns = load_feature_columns(config["feature_cols"])
    scaler = load_scaler_payload(config["scaler"])
    tasks = load_task_spec(Path(config["task_file"]))["tasks"]
    raw_classes = sorted(int(c) for task in tasks for c in task["classes"])
    class_map = build_seen_class_map(raw_classes)
    counts = pd.read_csv(root / "outputs/class_distribution/class_counts_train.csv")
    count_by_class = dict(zip(counts.class_id.astype(int), counts["count"].astype(int)))
    groups = {
        group: np.asarray(
            [class_map[c] for c in raw_classes if frequency_group(count_by_class[c]) == group],
            dtype=np.int64,
        )
        for group in GROUPS
    }
    validation = load_split_arrays(config["validation"], feature_columns, class_ids=raw_classes, scaler_payload=scaler)
    test = load_split_arrays(config["test"], feature_columns, class_ids=raw_classes, scaler_payload=scaler)
    validation.labels = remap_labels(validation.labels, class_map)
    test.labels = remap_labels(test.labels, class_map)
    device = torch.device(resolve_device(args.device))

    rows = []
    direction = None
    for method in ("c3", "c4"):
        model = load_model(runs[method], class_map, len(feature_columns), device)
        if method == "c4":
            validation_logits = infer(model, validation.features, device)
            direction = group_direction(validation_logits, validation.labels, groups)
        test_logits = infer(model, test.features, device)
        predictions = {method.upper(): test_logits.argmax(1)}
        if method == "c4":
            predictions["C4+GroupPrior"] = (test_logits + strength * direction).argmax(1)
        for candidate_name, predicted in predictions.items():
            for metric in metric_rows(test.labels, predicted, groups):
                rows.append({"setting": "P4-H8", "seed": args.seed, "candidate": candidate_name, **metric})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "endpoint_metrics.csv", index=False)
    (args.output / "contract.json").write_text(json.dumps({
        "setting": "P4-H8", "seed": args.seed, "backbone": "tddi",
        "candidate": candidate["candidate_id"], "lambda": strength,
        "direction_source": "validation", "evaluation_split": "test",
        "runs": {name: str(path) for name, path in runs.items()},
    }, indent=2))
    print(pd.DataFrame(rows).query("group == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
