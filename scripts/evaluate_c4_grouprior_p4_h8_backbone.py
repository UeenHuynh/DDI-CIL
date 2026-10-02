#!/usr/bin/env python3
"""Evaluate C3, C4, and locked GroupPrior on a P4-H8 backbone pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_c4_classifier_diagnostics import GROUPS, frequency_group, group_direction, metric_rows
from src.data.backbone_inputs import load_ddi_gcn_split_arrays
from src.data.class_mapping import build_seen_class_map, remap_labels
from src.data.ddi_dataset import load_feature_columns, load_scaler_payload, load_split_arrays
from src.data.molecular_graphs import load_graph_bank
from src.training.train_cil import expand_model_for_seen_classes, load_task_spec, resolve_device


def infer(model: torch.nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if not features.flags.writeable:
        features = features.copy()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            chunks.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def load_model(
    run_dir: Path,
    class_map: dict[int, int],
    input_dim: int,
    graph_bank: object | None,
    final_task: int,
    device: torch.device,
) -> torch.nn.Module:
    config = json.loads((run_dir / "run_config.json").read_text())["arguments"]
    model = expand_model_for_seen_classes(
        None, None, class_map,
        variant=str(config["variant"]),
        input_dim=input_dim,
        dropout=float(config["dropout"]),
        activation=str(config["activation"]),
        norm=str(config["norm"]),
        graph_bank=graph_bank,
        tabm_k=int(config["tabm_k"]),
        tabm_blocks=int(config["tabm_blocks"]),
        tabm_d_block=int(config["tabm_d_block"]),
        tabm_dropout=float(config["tabm_dropout"]),
        ddi_gcn_depth=int(config["ddi_gcn_depth"]),
        ddi_gcn_width=int(config["ddi_gcn_width"]),
        ddi_gcn_attention_dim=int(config["ddi_gcn_attention_dim"]),
    )
    model.load_state_dict(torch.load(
        run_dir / f"checkpoints/task_{final_task}_model.pt",
        map_location="cpu", weights_only=True,
    ))
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=["tabm", "ddi_gcn"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c3-run", required=True, type=Path)
    parser.add_argument("--c4-run", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graph-cache", type=Path)
    parser.add_argument("--graph-mapping", type=Path)
    args = parser.parse_args()

    for name, path in (("C3", args.c3_run), ("C4", args.c4_run)):
        if not (path / "run_summary.md").is_file():
            raise FileNotFoundError(f"Missing complete {name} run: {path}")
    candidate = json.loads((PROJECT_ROOT / "configs/c4_grouprior_candidate.json").read_text())
    strength = float(candidate["correction"]["lambda"])
    if candidate["candidate_id"] != "C4-GroupPrior-v1" or strength != 1.0:
        raise ValueError("Unexpected locked GroupPrior candidate")

    config = json.loads((args.c4_run / "run_config.json").read_text())["arguments"]
    if config["variant"] != args.backbone:
        raise ValueError(f"C4 run uses {config['variant']}, expected {args.backbone}")
    feature_columns = load_feature_columns(config["feature_cols"])
    scaler = load_scaler_payload(config["scaler"])
    tasks = load_task_spec(Path(config["task_file"]))["tasks"]
    raw_classes = sorted(int(c) for task in tasks for c in task["classes"])
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

    graph_bank = None
    if args.backbone == "ddi_gcn":
        if args.graph_cache is None or args.graph_mapping is None:
            raise ValueError("DDI-GCN evaluation requires graph cache and mapping")
        graph_bank = load_graph_bank(args.graph_cache, args.graph_mapping)
        validation, _ = load_ddi_gcn_split_arrays(config["validation"], graph_bank, class_ids=raw_classes)
        test, _ = load_ddi_gcn_split_arrays(config["test"], graph_bank, class_ids=raw_classes)
    else:
        validation = load_split_arrays(
            config["validation"], feature_columns, class_ids=raw_classes, scaler_payload=scaler
        )
        test = load_split_arrays(
            config["test"], feature_columns, class_ids=raw_classes, scaler_payload=scaler
        )
    validation.labels = remap_labels(validation.labels, class_map)
    test.labels = remap_labels(test.labels, class_map)
    device = torch.device(resolve_device(args.device))
    batch_size = 256

    rows: list[dict[str, object]] = []
    direction = None
    for name, run_dir in (("C3", args.c3_run), ("C4", args.c4_run)):
        model = load_model(
            run_dir, class_map, len(feature_columns), graph_bank, len(tasks) - 1, device
        )
        if name == "C4":
            direction = group_direction(infer(model, validation.features, device, batch_size), validation.labels, groups)
        test_logits = infer(model, test.features, device, batch_size)
        predictions = {name: test_logits.argmax(1)}
        if name == "C4":
            predictions["C4+GroupPrior"] = (test_logits + strength * direction).argmax(1)
        for candidate_name, predicted in predictions.items():
            for metric in metric_rows(test.labels, predicted, groups):
                rows.append({
                    "setting": "P4-H8", "backbone": args.backbone, "seed": args.seed,
                    "candidate": candidate_name, **metric,
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "endpoint_metrics.csv", index=False)
    (args.output / "contract.json").write_text(json.dumps({
        "setting": "P4-H8", "backbone": args.backbone, "seed": args.seed,
        "candidate": candidate["candidate_id"], "lambda": strength,
        "direction_source": "validation", "evaluation_split": "test",
        "runs": {"C3": str(args.c3_run), "C4": str(args.c4_run)},
    }, indent=2))
    print(frame.query("group == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
