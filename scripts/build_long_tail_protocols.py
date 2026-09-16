#!/usr/bin/env python3
"""Build P9-P12 schedules by extending the existing CIL task utilities.

Only train-split class counts are used.  The generated files keep the locked
8-task layout and are accepted unchanged by ``src/training/train_cil.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_cil_tasks import (
    load_counts,
    summarize_tasks,
    task_sizes,
    validate_task_sizes,
    validate_tasks,
)


PROTOCOLS = {
    "P9": "tail_profile_balanced",
    "P10": "effective_number_balanced",
    "P11": "progressive_tail_drift",
    "P12": "tail_shock",
}
DEFAULT_BETAS = (0.9, 0.99, 0.999, 0.9999)
HMT_GROUPS = ("head", "medium", "tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-counts", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "all"], default="all")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--quantiles", type=int, default=4)
    parser.add_argument("--betas", nargs="*", type=float, default=list(DEFAULT_BETAS))
    parser.add_argument("--p11-head-max", type=float, default=0.40)
    parser.add_argument("--p11-head-min", type=float, default=0.05)
    parser.add_argument("--p11-tail-min", type=float, default=0.25)
    parser.add_argument("--p11-tail-max", type=float, default=0.65)
    parser.add_argument("--shock-task", type=int, default=3)
    parser.add_argument("--shock-profile", nargs=3, type=float, default=[0.10, 0.10, 0.80])
    parser.add_argument("--num-classes", type=int, default=178)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--base-task-classes", type=int, default=38)
    parser.add_argument("--increment-classes", type=int, default=20)
    parser.add_argument("--swap-iterations", type=int, default=10_000)
    return parser.parse_args()


def beta_slug(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def relative_mse(values: Iterable[float], targets: Iterable[float]) -> float:
    values_array = np.asarray(list(values), dtype=np.float64)
    targets_array = np.asarray(list(targets), dtype=np.float64)
    scale = np.maximum(np.abs(targets_array), 1e-12)
    return float(np.mean(((values_array - targets_array) / scale) ** 2))


def effective_number(counts: np.ndarray, beta: float) -> np.ndarray:
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}.")
    counts = np.asarray(counts, dtype=np.float64)
    # expm1 is stable when beta is close to one.
    return np.expm1(counts * math.log(beta)) / math.expm1(math.log(beta))


def add_log_quantiles(frame: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    if quantiles < 2 or quantiles > len(frame):
        raise ValueError("quantiles must be between 2 and the number of classes.")
    ordered = frame.sort_values(["count", "class_id"], ascending=[False, True]).copy()
    ordered["log_count"] = np.log(ordered["count"].astype(float))
    positions = np.arange(len(ordered), dtype=np.int64)
    ordered["profile_group"] = [
        f"Q{min((position * quantiles) // len(ordered), quantiles - 1) + 1}"
        for position in positions
    ]
    return ordered.sort_values("class_id", ignore_index=True)


def add_hmt_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["profile_group"] = np.select(
        [result["count"] > 1000, result["count"] > 100],
        ["head", "medium"],
        default="tail",
    )
    return result


def allocate_integer_quotas(
    sizes: list[int],
    group_counts: dict[str, int],
    target_profiles: list[dict[str, float]],
    *,
    seed: int,
) -> list[dict[str, int]]:
    """Round profile targets while preserving exact row and column margins."""

    groups = list(group_counts)
    if len(target_profiles) != len(sizes):
        raise ValueError("One target profile is required per task.")
    if sum(group_counts.values()) != sum(sizes):
        raise ValueError("Group counts must equal total task capacity.")
    quotas = [{group: 0 for group in groups} for _ in sizes]
    remaining_rows = sizes.copy()
    remaining_groups = dict(group_counts)
    rng = np.random.default_rng(seed)
    tie_break = {
        (task, group): float(rng.random())
        for task in range(len(sizes))
        for group in groups
    }

    while sum(remaining_rows):
        candidates: list[tuple[float, float, int, str]] = []
        for task, size in enumerate(sizes):
            if remaining_rows[task] <= 0:
                continue
            for group in groups:
                if remaining_groups[group] <= 0:
                    continue
                new_rows = remaining_rows.copy()
                new_groups = dict(remaining_groups)
                new_rows[task] -= 1
                new_groups[group] -= 1
                # Do not take a slot that makes any remaining group impossible.
                if any(value > sum(new_rows) for value in new_groups.values()):
                    continue
                target = size * float(target_profiles[task][group])
                current = quotas[task][group]
                scale = max(target, 1.0)
                marginal = (((current + 1 - target) / scale) ** 2) - (
                    ((current - target) / scale) ** 2
                )
                candidates.append((marginal, tie_break[(task, group)], task, group))
        if not candidates:
            raise RuntimeError("Unable to apportion integer profile quotas.")
        _, _, task, group = min(candidates)
        quotas[task][group] += 1
        remaining_rows[task] -= 1
        remaining_groups[group] -= 1

    return quotas


def assign_with_quotas(
    frame: pd.DataFrame,
    sizes: list[int],
    quotas: list[dict[str, int]],
    *,
    seed: int,
) -> list[list[int]]:
    """Use P4's mass-aware greedy idea under arbitrary profile quotas."""

    rng = np.random.default_rng(seed)
    groups = list(quotas[0])
    used = [{group: 0 for group in groups} for _ in sizes]
    buckets: list[list[int]] = [[] for _ in sizes]
    mass = np.zeros(len(sizes), dtype=np.float64)
    target_mass = float(frame["count"].sum()) / len(sizes)
    task_rank = {task: rank for rank, task in enumerate(rng.permutation(len(sizes)))}
    ordered = frame.assign(_tie=rng.random(len(frame))).sort_values(
        ["count", "_tie", "class_id"], ascending=[False, True, True]
    )
    for row in ordered.itertuples(index=False):
        group = str(row.profile_group)
        candidates = [
            task for task in range(len(sizes))
            if used[task][group] < quotas[task][group]
        ]
        if not candidates:
            raise RuntimeError(f"No quota remains for profile group {group}.")
        task = min(
            candidates,
            key=lambda candidate: (
                (mass[candidate] + float(row.count)) / target_mass,
                task_rank[candidate],
            ),
        )
        buckets[task].append(int(row.class_id))
        mass[task] += float(row.count)
        used[task][group] += 1
    return buckets


def assign_balanced_weight(
    frame: pd.DataFrame,
    sizes: list[int],
    *,
    weight_column: str,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in sizes]
    mass = np.zeros(len(sizes), dtype=np.float64)
    target = float(frame[weight_column].sum()) / len(sizes)
    rank = {task: position for position, task in enumerate(rng.permutation(len(sizes)))}
    ordered = frame.assign(_tie=rng.random(len(frame))).sort_values(
        [weight_column, "_tie", "class_id"], ascending=[False, True, True]
    )
    for row in ordered.itertuples(index=False):
        candidates = [task for task, size in enumerate(sizes) if len(buckets[task]) < size]
        task = min(
            candidates,
            key=lambda candidate: (
                (mass[candidate] + float(getattr(row, weight_column))) / target,
                rank[candidate],
            ),
        )
        buckets[task].append(int(row.class_id))
        mass[task] += float(getattr(row, weight_column))
    return buckets


def optimize_swaps(
    buckets: list[list[int]],
    frame: pd.DataFrame,
    objective: Callable[[list[list[int]]], float],
    *,
    seed: int,
    iterations: int,
    preserve_group: bool,
) -> list[list[int]]:
    rng = np.random.default_rng(seed + 200_003)
    group_by_class = frame.set_index("class_id")["profile_group"].to_dict()
    current = objective(buckets)
    for _ in range(iterations):
        left, right = rng.choice(len(buckets), size=2, replace=False).tolist()
        li = int(rng.integers(len(buckets[left])))
        if preserve_group:
            group = group_by_class[buckets[left][li]]
            eligible = [
                index for index, class_id in enumerate(buckets[right])
                if group_by_class[class_id] == group
            ]
            if not eligible:
                continue
            ri = int(rng.choice(eligible))
        else:
            ri = int(rng.integers(len(buckets[right])))
        buckets[left][li], buckets[right][ri] = buckets[right][ri], buckets[left][li]
        candidate = objective(buckets)
        if candidate + 1e-12 < current:
            current = candidate
        else:
            buckets[left][li], buckets[right][ri] = buckets[right][ri], buckets[left][li]
    return buckets


def mass_objective(frame: pd.DataFrame, column: str) -> Callable[[list[list[int]]], float]:
    data = frame.set_index("class_id")

    def objective(buckets: list[list[int]]) -> float:
        target = float(data[column].sum()) / len(buckets)
        values = [float(data.loc[bucket, column].sum()) for bucket in buckets]
        return relative_mse(values, [target] * len(buckets))

    return objective


def p11_profiles(num_tasks: int, args: argparse.Namespace) -> list[dict[str, float]]:
    profiles = []
    for task in range(num_tasks):
        alpha = task / max(num_tasks - 1, 1)
        head = args.p11_head_max + alpha * (args.p11_head_min - args.p11_head_max)
        tail = args.p11_tail_min + alpha * (args.p11_tail_max - args.p11_tail_min)
        medium = 1.0 - head - tail
        if min(head, medium, tail) < 0.0:
            raise ValueError("P11 profile contains a negative ratio.")
        profiles.append({"head": head, "medium": medium, "tail": tail})
    return profiles


def p12_quotas(
    sizes: list[int],
    group_counts: dict[str, int],
    *,
    shock_task: int,
    shock_profile: list[float],
    seed: int,
) -> tuple[list[dict[str, int]], list[dict[str, float]]]:
    if not 0 <= shock_task < len(sizes):
        raise ValueError("shock-task is outside the task layout.")
    if len(shock_profile) != 3 or not np.isclose(sum(shock_profile), 1.0):
        raise ValueError("shock-profile must contain three ratios summing to one.")
    shock_targets = [
        {group: float(value) for group, value in zip(HMT_GROUPS, shock_profile, strict=True)}
    ]
    shock_quota = allocate_integer_quotas(
        [sizes[shock_task]], group_counts={
            group: int(round(sizes[shock_task] * shock_targets[0][group]))
            for group in HMT_GROUPS
        }, target_profiles=shock_targets, seed=seed,
    )[0]
    # Repair any round-off mismatch (the default 20-class shock is already exact).
    difference = sizes[shock_task] - sum(shock_quota.values())
    if difference:
        shock_quota["tail"] += difference
    remaining_counts = {
        group: group_counts[group] - shock_quota[group] for group in HMT_GROUPS
    }
    if any(value < 0 for value in remaining_counts.values()):
        raise ValueError("Shock quota requests more classes than the dataset contains.")
    normal_total = sum(remaining_counts.values())
    normal_profile = {
        group: remaining_counts[group] / normal_total for group in HMT_GROUPS
    }
    normal_sizes = [size for task, size in enumerate(sizes) if task != shock_task]
    normal_quotas = allocate_integer_quotas(
        normal_sizes,
        remaining_counts,
        [normal_profile.copy() for _ in normal_sizes],
        seed=seed + 1,
    )
    quotas: list[dict[str, int]] = []
    profiles: list[dict[str, float]] = []
    cursor = 0
    for task in range(len(sizes)):
        if task == shock_task:
            quotas.append(shock_quota)
            profiles.append(shock_targets[0])
        else:
            quotas.append(normal_quotas[cursor])
            profiles.append(normal_profile.copy())
            cursor += 1
    return quotas, profiles


def buckets_to_tasks(buckets: list[list[int]]) -> list[dict[str, object]]:
    return [
        {"task_id": task, "classes": classes, "num_classes": len(classes)}
        for task, classes in enumerate(buckets)
    ]


def write_task_file(
    path: Path,
    *,
    protocol: str,
    seed: int,
    num_classes: int,
    tasks: list[dict[str, object]],
    construction: dict[str, object],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite task file: {path}")
    payload = {
        "protocol": protocol,
        "seed": seed,
        "num_classes": num_classes,
        "construction": construction,
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def audit_rows(
    protocol: str,
    seed: int,
    buckets: list[list[int]],
    frame: pd.DataFrame,
    *,
    beta: float | None = None,
) -> list[dict[str, object]]:
    data = frame.set_index("class_id")
    groups = sorted(frame["profile_group"].unique())
    rows = []
    for task, bucket in enumerate(buckets):
        row: dict[str, object] = {
            "protocol": protocol,
            "seed": seed,
            "beta": beta,
            "task_id": task,
            "num_classes": len(bucket),
            "train_samples": int(data.loc[bucket, "count"].sum()),
        }
        if "effective_number" in data:
            row["effective_mass"] = float(data.loc[bucket, "effective_number"].sum())
        for group in groups:
            row[f"classes_{group}"] = int(
                (data.loc[bucket, "profile_group"] == group).sum()
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    sizes = task_sizes(args.num_tasks, args.base_task_classes, args.increment_classes)
    validate_task_sizes(sizes, args.num_classes, args.base_task_classes, args.increment_classes)
    counts = load_counts(args.class_counts)
    if len(counts) != args.num_classes or counts["class_id"].duplicated().any():
        raise ValueError("Class counts must contain each expected class exactly once.")
    class_ids = sorted(counts["class_id"].astype(int).tolist())
    requested = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    generated: list[str] = []

    for protocol in requested:
        for seed in args.seeds:
            variants: list[tuple[str, pd.DataFrame, list[list[int]], dict[str, object], float | None]] = []
            if protocol == "P9":
                frame = add_log_quantiles(counts, args.quantiles)
                groups = [f"Q{index}" for index in range(1, args.quantiles + 1)]
                group_counts = {
                    group: int((frame["profile_group"] == group).sum()) for group in groups
                }
                global_profile = {
                    group: group_counts[group] / len(frame) for group in groups
                }
                quotas = allocate_integer_quotas(
                    sizes, group_counts, [global_profile.copy() for _ in sizes], seed=seed
                )
                buckets = assign_with_quotas(frame, sizes, quotas, seed=seed)
                buckets = optimize_swaps(
                    buckets, frame, mass_objective(frame, "count"), seed=seed,
                    iterations=args.swap_iterations, preserve_group=True,
                )
                variants.append((
                    f"tail_profile_balanced_seed{seed}_tasks.json", frame, buckets,
                    {"quantiles": args.quantiles, "quantile_basis": "rank(log(train_count))",
                     "lambda_mass": 1.0, "lambda_js": 1.0, "lambda_class_count": 1.0,
                     "target_profiles": [global_profile] * len(sizes), "integer_quotas": quotas},
                    None,
                ))
            elif protocol == "P10":
                for beta in args.betas:
                    frame = add_hmt_groups(counts)
                    frame["effective_number"] = effective_number(frame["count"].to_numpy(), beta)
                    buckets = assign_balanced_weight(
                        frame, sizes, weight_column="effective_number", seed=seed
                    )
                    buckets = optimize_swaps(
                        buckets, frame, mass_objective(frame, "effective_number"), seed=seed,
                        iterations=args.swap_iterations, preserve_group=False,
                    )
                    variants.append((
                        f"effective_number_balanced_beta{beta_slug(beta)}_seed{seed}_tasks.json",
                        frame, buckets,
                        {"beta": beta, "objective": "relative_effective_mass_mse",
                         "class_count_constraint": sizes}, beta,
                    ))
            elif protocol == "P11":
                frame = add_hmt_groups(counts)
                group_counts = {
                    group: int((frame["profile_group"] == group).sum()) for group in HMT_GROUPS
                }
                profiles = p11_profiles(len(sizes), args)
                quotas = allocate_integer_quotas(sizes, group_counts, profiles, seed=seed)
                buckets = assign_with_quotas(frame, sizes, quotas, seed=seed)
                buckets = optimize_swaps(
                    buckets, frame, mass_objective(frame, "count"), seed=seed,
                    iterations=args.swap_iterations, preserve_group=True,
                )
                variants.append((
                    f"progressive_tail_drift_seed{seed}_tasks.json", frame, buckets,
                    {"group_thresholds": {"head": ">1000", "medium": "101..1000", "tail": "<=100"},
                     "target_profiles": profiles, "integer_quotas": quotas}, None,
                ))
            elif protocol == "P12":
                frame = add_hmt_groups(counts)
                group_counts = {
                    group: int((frame["profile_group"] == group).sum()) for group in HMT_GROUPS
                }
                quotas, profiles = p12_quotas(
                    sizes, group_counts, shock_task=args.shock_task,
                    shock_profile=list(args.shock_profile), seed=seed,
                )
                buckets = assign_with_quotas(frame, sizes, quotas, seed=seed)
                buckets = optimize_swaps(
                    buckets, frame, mass_objective(frame, "count"), seed=seed,
                    iterations=args.swap_iterations, preserve_group=True,
                )
                variants.append((
                    f"tail_shock_task{args.shock_task}_seed{seed}_tasks.json", frame, buckets,
                    {"group_thresholds": {"head": ">1000", "medium": "101..1000", "tail": "<=100"},
                     "shock_task": args.shock_task, "target_profiles": profiles,
                     "integer_quotas": quotas}, None,
                ))

            for filename, frame, buckets, construction, beta in variants:
                tasks = buckets_to_tasks(buckets)
                validate_tasks(tasks, class_ids, sizes)
                path = args.outdir / filename
                write_task_file(
                    path, protocol=PROTOCOLS[protocol], seed=seed,
                    num_classes=args.num_classes, tasks=tasks, construction=construction,
                )
                summaries.append(summarize_tasks(
                    PROTOCOLS[protocol], seed, tasks, counts, None, None
                ))
                audits.extend(audit_rows(PROTOCOLS[protocol], seed, buckets, frame, beta=beta))
                generated.append(str(path))
                print(f"[tasks] Built {protocol} seed={seed}: {path}", flush=True)

    pd.concat(summaries, ignore_index=True).to_csv(
        args.outdir / "long_tail_task_summary.csv", index=False
    )
    pd.DataFrame(audits).to_csv(args.outdir / "long_tail_task_audit.csv", index=False)
    manifest = {
        "protocols": {key: PROTOCOLS[key] for key in requested},
        "seeds": args.seeds,
        "generated_files": generated,
        "train_counts_only": True,
        "test_split_used": False,
        "hmt_thresholds": {"head": "count > 1000", "medium": "101 <= count <= 1000", "tail": "count <= 100"},
        "p10_betas": args.betas,
        "p12_shock_task_zero_based": args.shock_task,
    }
    (args.outdir / "long_tail_protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[done] Wrote P9-P12 schedules to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
