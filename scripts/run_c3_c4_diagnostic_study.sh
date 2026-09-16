#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DIAG_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DIAG_DEVICE:-cuda}"
OUTPUT_ROOT="${DIAG_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_c3_c4_diagnostics}"
TASK_ROOT="${DIAG_TASK_ROOT:-${PROJECT_ROOT}/outputs/tasks_c3_c4_diagnostics}"
read -r -a SEEDS <<< "${DIAG_SEEDS:-0 1 2}"

common_args=(
  --train "${PROJECT_ROOT}/train_extracted.parquet"
  --validation "${PROJECT_ROOT}/validation_extracted.parquet"
  --test "${PROJECT_ROOT}/test_extracted.parquet"
  --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json"
  --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl"
  --variant tddi --batch-size 1024 --effective-batch-size 1024
  --epochs 20 --patience 5 --lr 0.001 --weight-decay 0.0001
  --device "${DEVICE}" --focal-gamma 1.0
)

is_complete() {
  local run_dir="$1" expected_tasks="$2"
  [[ -s "${run_dir}/run_summary.md" ]] &&
    [[ -s "${run_dir}/metrics.csv" ]] &&
    [[ -s "${run_dir}/class_trajectory.csv" ]] &&
    [[ -s "${run_dir}/events.csv" ]] &&
    [[ "$(find "${run_dir}/checkpoints" -maxdepth 1 -name 'task_*_model.pt' 2>/dev/null | wc -l)" -eq "${expected_tasks}" ]] &&
    rg -q 'run_completed' "${run_dir}/events.csv"
}

run_one() {
  local label="$1" task_file="$2" run_dir="$3" method="$4" seed="$5" tasks="$6" sampling="${7:-natural}"
  if is_complete "${run_dir}" "${tasks}"; then
    echo "[skip] ${label}"
    return
  fi
  if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
    exit 1
  fi
  echo "[run] ${label}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/training/train_cil.py" \
    "${common_args[@]}" --task-file "${task_file}" --outdir "${run_dir}" \
    --method "${method}" --sampling "${sampling}" --seed "${seed}" \
    --total-memory-budget 6800 --replay-draws-per-epoch 6800 \
    --distill-alpha 1.0 --temperature 2.0 --feature-distill-weight 0.5
}

mkdir -p "${OUTPUT_ROOT}" "${TASK_ROOT}/joint" "${TASK_ROOT}/h15_p9"

if [[ ! -s "${TASK_ROOT}/joint/random_seed0_tasks.json" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_cil_tasks.py" \
    --class-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_train.csv" \
    --validation-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_validation.csv" \
    --test-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_test.csv" \
    --outdir "${TASK_ROOT}/joint" --protocol random --num-classes 178 \
    --num-tasks 1 --base-task-classes 178 --increment-classes 1 --seeds 0
fi

missing_h15=0
for seed in "${SEEDS[@]}"; do
  [[ -s "${TASK_ROOT}/h15_p9/tail_profile_balanced_seed${seed}_tasks.json" ]] || missing_h15=1
done
if (( missing_h15 )); then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_long_tail_protocols.py" \
    --class-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_train.csv" \
    --outdir "${TASK_ROOT}/h15_p9" --protocol P9 --seeds "${SEEDS[@]}" \
    --num-classes 178 --num-tasks 15 --base-task-classes 38 \
    --increment-classes 10
fi

joint_task="${TASK_ROOT}/joint/random_seed0_tasks.json"
run_one "J0 joint natural seed=0" "${joint_task}" \
  "${OUTPUT_ROOT}/j0_joint_natural_seed0" sequential 0 1 natural
run_one "J1 joint balanced seed=0" "${joint_task}" \
  "${OUTPUT_ROOT}/j1_joint_balanced_seed0" sequential 0 1 class_balanced

for seed in "${SEEDS[@]}"; do
  h15_task="${TASK_ROOT}/h15_p9/tail_profile_balanced_seed${seed}_tasks.json"
  if [[ "${seed}" != "0" ]]; then
    run_one "C3 P9-H15 seed=${seed}" "${h15_task}" \
      "${OUTPUT_ROOT}/p9_h15_tddi_c3_seed${seed}" \
      replay_distill_fixed_budget_uniform "${seed}" 15
  fi
  run_one "C4 P9-H15 seed=${seed}" "${h15_task}" \
    "${OUTPUT_ROOT}/p9_h15_tddi_c4_seed${seed}" \
    xder_fixed_budget_uniform "${seed}" 15

  p4_task="${PROJECT_ROOT}/outputs/tasks/constrained_mass_balanced_seed${seed}_tasks.json"
  run_one "C4 P4-H8 seed=${seed}" "${p4_task}" \
    "${OUTPUT_ROOT}/p4_h8_tddi_c4_seed${seed}" \
    xder_fixed_budget_uniform "${seed}" 8
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/analyze_c3_c4_diagnostics.py" \
  --study-root "${OUTPUT_ROOT}"
echo "[done] C3/C4 diagnostic study complete: ${OUTPUT_ROOT}"
