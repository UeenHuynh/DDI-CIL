#!/usr/bin/env python3
"""Evaluate locked C4-GroupPrior-v1 after every task on seeds 0-2."""

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

from scripts.run_c4_classifier_diagnostics import (
    DEFAULT_GRID,
    GROUPS,
    frequency_group,
    group_direction,
    infer,
    metric_rows,
)
from scripts.screen_c4_grouprior_robustness import settings_for_seed
from src.data.class_mapping import build_seen_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.training.train_cil import expand_model_for_seen_classes, load_task_spec, resolve_device


LOCKED_LAMBDA = 1.0


def load_task_model(
    run_dir: Path,
    seen_map: dict[int, int],
    input_dim: int,
    task_id: int,
    device: torch.device,
) -> torch.nn.Module:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["arguments"]
    model = expand_model_for_seen_classes(
        None, None, seen_map, variant="tddi", input_dim=input_dim,
        dropout=float(config["dropout"]), activation=str(config["activation"]),
        norm=str(config["norm"]),
    )
    model.load_state_dict(torch.load(
        run_dir / f"checkpoints/task_{task_id}_model.pt",
        map_location="cpu", weights_only=True,
    ))
    return model.to(device).eval()


def append_c3_scores(
    rows: list[dict[str, object]], setting: str, seed: int, run_dir: Path,
) -> None:
    matrix = pd.read_csv(run_dir / "task_matrix.csv")
    for record in matrix.itertuples(index=False):
        train_task = int(record.train_task_id)
        for eval_task in range(train_task + 1):
            rows.append({
                "setting": setting, "seed": seed, "candidate": "C3",
                "train_task": train_task, "eval_task": eval_task,
                "task_f1": float(getattr(record, f"test_task_{eval_task}")),
            })


def retention_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (setting, seed, candidate), part in scores.groupby(
        ["setting", "seed", "candidate"], sort=True
    ):
        final_task = int(part.train_task.max())
        for eval_task in range(final_task + 1):
            trajectory = part[part.eval_task == eval_task].sort_values("train_task")
            immediate = float(trajectory.loc[
                trajectory.train_task == eval_task, "task_f1"
            ].iloc[0])
            final = float(trajectory.loc[
                trajectory.train_task == final_task, "task_f1"
            ].iloc[0])
            rows.append({
                "setting": setting, "seed": seed, "candidate": candidate,
                "eval_task": eval_task, "task_age": final_task - eval_task,
                "new_task_f1": immediate, "final_task_f1": final,
                "retention_ratio": final / immediate if immediate > 0 else np.nan,
                "forgetting": float(trajectory.task_f1.max() - final),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/runs_c4_grouprior_screening/per_task_analysis",
    )
    args = parser.parse_args()
    device = torch.device(resolve_device(args.device))
    candidate = json.loads(
        (PROJECT_ROOT / "configs/c4_grouprior_candidate.json").read_text(encoding="utf-8")
    )
    if float(candidate["correction"]["lambda"]) != LOCKED_LAMBDA:
        raise ValueError("Locked candidate lambda does not match evaluator contract.")

    reference_run = settings_for_seed(args.seeds[0])["P4-H8"]["c4"]
    reference = json.loads(
        (reference_run / "run_config.json").read_text(encoding="utf-8")
    )["arguments"]
    feature_columns = load_feature_columns(reference["feature_cols"])
    scaler = load_scaler_payload(reference["scaler"])
    all_raw_classes = sorted(
        int(row.class_id)
        for row in pd.read_csv(
            PROJECT_ROOT / "outputs/class_distribution/class_counts_train.csv"
        ).itertuples()
    )
    global_map = build_seen_class_map(all_raw_classes)
    counts_frame = pd.read_csv(PROJECT_ROOT / "outputs/class_distribution/class_counts_train.csv")
    count_by_class = dict(zip(
        counts_frame.class_id.astype(int), counts_frame["count"].astype(int)
    ))
    validation = load_split_arrays(
        reference["validation"], feature_columns, class_ids=all_raw_classes,
        scaler_payload=scaler,
    )
    test = load_split_arrays(
        reference["test"], feature_columns, class_ids=all_raw_classes,
        scaler_payload=scaler,
    )
    validation.labels = remap_labels(validation.labels, global_map)
    test.labels = remap_labels(test.labels, global_map)

    endpoint_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    lambda_rows: list[dict[str, object]] = []
    calibrator_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for setting, run_dirs in settings_for_seed(seed).items():
            append_c3_scores(score_rows, setting, seed, run_dirs["c3"])
            c4_config = json.loads(
                (run_dirs["c4"] / "run_config.json").read_text(encoding="utf-8")
            )["arguments"]
            tasks = load_task_spec(Path(c4_config["task_file"]))["tasks"]
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
                model = load_task_model(
                    run_dirs["c4"], seen_map, len(feature_columns), task_id, device
                )
                validation_logits = infer(model, validation.features[validation_mask], device)
                test_logits = infer(model, test.features[test_mask], device)
                direction = group_direction(
                    validation_logits, validation_labels, local_groups
                )
                grid_scores: list[tuple[float, float]] = []
                for strength in DEFAULT_GRID:
                    predictions = (validation_logits + strength * direction).argmax(1)
                    score = float(metric_rows(
                        validation_labels, predictions, local_groups
                    )[0]["f1"])
                    grid_scores.append((strength, score))
                validation_best = max(grid_scores, key=lambda item: (item[1], -item[0]))[0]
                lambda_rows.append({
                    "setting": setting, "seed": seed, "task": task_id,
                    "locked_lambda": LOCKED_LAMBDA,
                    "validation_best_lambda": validation_best,
                    "locked_matches_validation_best": LOCKED_LAMBDA == validation_best,
                })
                predictions_by_candidate = {
                    "C4": test_logits.argmax(1),
                    "C4+GroupPrior": (
                        test_logits + LOCKED_LAMBDA * direction
                    ).argmax(1),
                }
                for candidate_id, predictions in predictions_by_candidate.items():
                    for row in metric_rows(test_labels, predictions, local_groups):
                        endpoint_rows.append({
                            "setting": setting, "seed": seed,
                            "candidate": candidate_id, "train_task": task_id, **row,
                        })
                    for eval_task_id, eval_task in enumerate(tasks[:task_id + 1]):
                        eval_local = np.asarray(
                            [seen_map[int(c)] for c in eval_task["classes"]], dtype=np.int64
                        )
                        eval_mask = np.isin(test_labels, eval_local)
                        task_f1 = f1_score(
                            test_labels[eval_mask], predictions[eval_mask],
                            labels=eval_local, average="macro", zero_division=0,
                        )
                        score_rows.append({
                            "setting": setting, "seed": seed,
                            "candidate": candidate_id, "train_task": task_id,
                            "eval_task": eval_task_id, "task_f1": float(task_f1),
                        })
                for local_index, raw_class in enumerate(seen_raw):
                    calibrator_rows.append({
                        "setting": setting, "seed": seed, "task": task_id,
                        "class_id": raw_class,
                        "group": frequency_group(count_by_class[raw_class]),
                        "direction": float(direction[local_index]),
                        "applied_offset": float(LOCKED_LAMBDA * direction[local_index]),
                    })
                print(
                    f"[trajectory] {setting} seed={seed} task={task_id} "
                    f"validation_best_lambda={validation_best:.2f}", flush=True,
                )
                del model, validation_logits, test_logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    endpoints = pd.DataFrame(endpoint_rows)
    scores = pd.DataFrame(score_rows)
    lambdas = pd.DataFrame(lambda_rows)
    retention = retention_table(scores)
    endpoints.to_csv(args.output / "endpoint_metrics_per_task.csv", index=False)
    scores.to_csv(args.output / "task_matrix_long.csv", index=False)
    lambdas.to_csv(args.output / "lambda_audit.csv", index=False)
    pd.DataFrame(calibrator_rows).to_csv(args.output / "calibrators.csv", index=False)
    retention.to_csv(args.output / "retention_per_task.csv", index=False)
    seed_summary = retention.groupby(
        ["setting", "seed", "candidate"]
    )[["new_task_f1", "final_task_f1", "retention_ratio", "forgetting"]].mean().reset_index()
    seed_summary.to_csv(args.output / "retention_per_seed.csv", index=False)
    seed_summary.groupby(["setting", "candidate"])[
        ["new_task_f1", "final_task_f1", "retention_ratio", "forgetting"]
    ].agg(["count", "mean", "std"]).to_csv(args.output / "retention_three_seed_summary.csv")
    retention.groupby(["setting", "candidate", "task_age"])[
        ["new_task_f1", "final_task_f1", "retention_ratio", "forgetting"]
    ].mean().to_csv(args.output / "retention_by_task_age.csv")
    final_endpoints = endpoints.loc[
        endpoints.train_task == endpoints.groupby(["setting", "seed"])["train_task"].transform("max")
    ]
    final_endpoints.groupby(["setting", "candidate", "group"])[
        ["precision", "recall", "f1", "prediction_to_true_ratio"]
    ].agg(["count", "mean", "std"]).to_csv(args.output / "final_endpoint_summary.csv")
    (args.output / "evaluation_contract.json").write_text(json.dumps({
        "candidate": "C4-GroupPrior-v1",
        "seeds": args.seeds,
        "settings": ["P4-H8", "P9-H15", "P4-H15"],
        "lambda": LOCKED_LAMBDA,
        "direction_source": "validation q_g/p_g after each task",
        "test_used_for_calibration": False,
        "backbone_retrained": False,
    }, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
