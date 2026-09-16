#!/usr/bin/env python3
"""Evaluate C3, C4, and locked C4-GroupPrior-v1 throughout P9-H29."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_c4_grouprior_trajectories import (
    append_c3_scores,
    load_task_model,
    retention_table,
)
from scripts.run_c4_classifier_diagnostics import (
    GROUPS,
    frequency_group,
    group_direction,
    infer,
    metric_rows,
)
from src.data.class_mapping import build_seen_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.training.train_cil import load_task_spec, resolve_device


LOCKED_LAMBDA = 1.0
SETTING = "P9-H29"


def require_complete(run_dir: Path) -> None:
    required = [
        run_dir / "run_summary.md",
        run_dir / "events.csv",
        run_dir / "task_matrix.csv",
        run_dir / "run_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}; missing: {missing}")
    events = pd.read_csv(run_dir / "events.csv")
    if "event_type" not in events or not (events["event_type"] == "run_completed").any():
        raise RuntimeError(f"Run has no run_completed event: {run_dir}")


def bwt_table(retention: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate, part in retention.groupby("candidate", sort=False):
        final_task = int(part.eval_task.max())
        old = part.loc[part.eval_task < final_task]
        rows.append({
            "setting": SETTING,
            "seed": seed,
            "candidate": candidate,
            "bwt": float((old.final_task_f1 - old.new_task_f1).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3-run", required=True, type=Path)
    parser.add_argument("--c4-run", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/runs_c4_grouprior_h29/analysis",
    )
    args = parser.parse_args()
    require_complete(args.c3_run)
    require_complete(args.c4_run)

    candidate_contract = json.loads(
        (PROJECT_ROOT / "configs/c4_grouprior_candidate.json").read_text(encoding="utf-8")
    )
    if float(candidate_contract["correction"]["lambda"]) != LOCKED_LAMBDA:
        raise ValueError("Candidate contract no longer specifies locked lambda=1.0")

    device = torch.device(resolve_device(args.device))
    config = json.loads((args.c4_run / "run_config.json").read_text(encoding="utf-8"))["arguments"]
    tasks = load_task_spec(Path(config["task_file"]))["tasks"]
    if len(tasks) != 29:
        raise ValueError(f"Expected H29, found {len(tasks)} tasks")
    feature_columns = load_feature_columns(config["feature_cols"])
    scaler = load_scaler_payload(config["scaler"])
    all_raw_classes = sorted({int(c) for task in tasks for c in task["classes"]})
    if len(all_raw_classes) != 178:
        raise ValueError(f"Expected 178 classes, found {len(all_raw_classes)}")
    global_map = build_seen_class_map(all_raw_classes)

    counts = pd.read_csv(PROJECT_ROOT / "outputs/class_distribution/class_counts_train.csv")
    count_by_class = dict(zip(counts.class_id.astype(int), counts["count"].astype(int)))
    validation = load_split_arrays(
        config["validation"], feature_columns, class_ids=all_raw_classes,
        scaler_payload=scaler,
    )
    test = load_split_arrays(
        config["test"], feature_columns, class_ids=all_raw_classes,
        scaler_payload=scaler,
    )
    validation.labels = remap_labels(validation.labels, global_map)
    test.labels = remap_labels(test.labels, global_map)

    endpoint_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    calibrator_rows: list[dict[str, object]] = []
    append_c3_scores(score_rows, SETTING, args.seed, args.c3_run)

    seen_raw: list[int] = []
    for task_id, task in enumerate(tasks):
        seen_raw = sorted(set(seen_raw + [int(c) for c in task["classes"]]))
        seen_map = build_seen_class_map(seen_raw)
        global_indices = np.asarray([global_map[c] for c in seen_raw], dtype=np.int64)
        local_lookup = np.full(len(all_raw_classes), -1, dtype=np.int64)
        local_lookup[global_indices] = np.arange(len(seen_raw))
        validation_mask = np.isin(validation.labels, global_indices)
        test_mask = np.isin(test.labels, global_indices)
        validation_labels = local_lookup[validation.labels[validation_mask]]
        test_labels = local_lookup[test.labels[test_mask]]
        local_groups = {
            group: np.asarray([
                seen_map[c] for c in seen_raw
                if frequency_group(count_by_class[c]) == group
            ], dtype=np.int64)
            for group in GROUPS
        }
        model = load_task_model(args.c4_run, seen_map, len(feature_columns), task_id, device)
        validation_logits = infer(model, validation.features[validation_mask], device)
        test_logits = infer(model, test.features[test_mask], device)
        direction = group_direction(validation_logits, validation_labels, local_groups)
        predictions_by_candidate = {
            "C4": test_logits.argmax(1),
            "C4+GroupPrior": (test_logits + LOCKED_LAMBDA * direction).argmax(1),
        }
        for candidate, predictions in predictions_by_candidate.items():
            for row in metric_rows(test_labels, predictions, local_groups):
                endpoint_rows.append({
                    "setting": SETTING, "seed": args.seed, "candidate": candidate,
                    "train_task": task_id, "lambda": LOCKED_LAMBDA if candidate == "C4+GroupPrior" else 0.0,
                    **row,
                })
            for eval_task_id, eval_task in enumerate(tasks[:task_id + 1]):
                eval_local = np.asarray(
                    [seen_map[int(c)] for c in eval_task["classes"]], dtype=np.int64
                )
                eval_mask = np.isin(test_labels, eval_local)
                score_rows.append({
                    "setting": SETTING, "seed": args.seed, "candidate": candidate,
                    "train_task": task_id, "eval_task": eval_task_id,
                    "task_f1": float(f1_score(
                        test_labels[eval_mask], predictions[eval_mask], labels=eval_local,
                        average="macro", zero_division=0,
                    )),
                })
        for local_index, raw_class in enumerate(seen_raw):
            calibrator_rows.append({
                "setting": SETTING, "seed": args.seed, "task": task_id,
                "class_id": raw_class,
                "group": frequency_group(count_by_class[raw_class]),
                "validation_direction": float(direction[local_index]),
                "applied_offset": float(LOCKED_LAMBDA * direction[local_index]),
            })
        print(f"[H29 trajectory] task={task_id:02d}/28 lambda=1.0", flush=True)
        del model, validation_logits, test_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Infer the final C3 checkpoint once so endpoint P/R/F1 uses the same evaluator.
    full_groups = {
        group: np.asarray([
            global_map[c] for c in all_raw_classes
            if frequency_group(count_by_class[c]) == group
        ], dtype=np.int64)
        for group in GROUPS
    }
    c3_model = load_task_model(
        args.c3_run, global_map, len(feature_columns), len(tasks) - 1, device
    )
    c3_predictions = infer(c3_model, test.features, device).argmax(1)
    for row in metric_rows(test.labels, c3_predictions, full_groups):
        endpoint_rows.append({
            "setting": SETTING, "seed": args.seed, "candidate": "C3",
            "train_task": len(tasks) - 1, "lambda": 0.0, **row,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    endpoints = pd.DataFrame(endpoint_rows)
    scores = pd.DataFrame(score_rows)
    scores["task_age"] = scores.train_task - scores.eval_task
    retention = retention_table(scores)
    bwt = bwt_table(retention, args.seed)
    retention_summary = retention.groupby(
        ["setting", "seed", "candidate"], sort=False
    )[["new_task_f1", "final_task_f1", "retention_ratio", "forgetting"]].mean().reset_index()
    final_endpoints = endpoints.loc[endpoints.train_task == len(tasks) - 1].copy()
    all_metrics = final_endpoints.loc[final_endpoints.group == "all", [
        "setting", "seed", "candidate", "precision", "recall", "f1",
    ]].rename(columns={
        "precision": "final_macro_precision", "recall": "final_macro_recall_balanced_accuracy",
        "f1": "final_macro_f1",
    })
    group_f1 = final_endpoints.pivot_table(
        index=["setting", "seed", "candidate"], columns="group", values="f1"
    ).reset_index().rename(columns={"head": "head_f1", "medium": "medium_f1", "tail": "tail_f1"})
    summary = (
        all_metrics.merge(group_f1, on=["setting", "seed", "candidate"], validate="one_to_one")
        .merge(retention_summary, on=["setting", "seed", "candidate"], validate="one_to_one")
        .merge(bwt, on=["setting", "seed", "candidate"], validate="one_to_one")
    )

    endpoints.to_csv(args.output / "endpoint_metrics_per_task.csv", index=False)
    scores.to_csv(args.output / "task_matrix_long.csv", index=False)
    retention.to_csv(args.output / "retention_per_task.csv", index=False)
    bwt.to_csv(args.output / "bwt.csv", index=False)
    summary.to_csv(args.output / "h29_seed0_summary.csv", index=False)
    pd.DataFrame(calibrator_rows).to_csv(args.output / "calibrators.csv", index=False)
    scores.groupby(["candidate", "task_age"], sort=True).task_f1.agg(
        ["count", "mean", "std"]
    ).reset_index().to_csv(args.output / "performance_by_task_age.csv", index=False)
    (args.output / "evaluation_contract.json").write_text(json.dumps({
        "candidate": "C4-GroupPrior-v1",
        "setting": SETTING,
        "seed": args.seed,
        "lambda": LOCKED_LAMBDA,
        "lambda_tuned_on_h29": False,
        "direction_source": "validation q_g/p_g after each task",
        "test_used_for_calibration": False,
        "backbone_retrained_for_group_prior": False,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
