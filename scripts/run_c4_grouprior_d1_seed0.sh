#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${GP_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${GP_DEVICE:-cuda}"
TASK_ROOT="${PROJECT_ROOT}/outputs/tasks_c4_grouprior_dynamic/d1"
RUN_ROOT="${PROJECT_ROOT}/outputs/runs_c4_grouprior_dynamic/d1"
TASK_FILE="${TASK_ROOT}/random_seed0_tasks.json"
C3_RUN="${RUN_ROOT}/d1_tddi_c3_seed0"
C4_RUN="${RUN_ROOT}/d1_tddi_c4_seed0"

mkdir -p "${TASK_ROOT}" "${RUN_ROOT}"

if [[ ! -s "${TASK_FILE}" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_cil_tasks.py" \
    --class-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_train.csv" \
    --validation-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_validation.csv" \
    --test-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_test.csv" \
    --outdir "${TASK_ROOT}" --protocol random --seeds 0 \
    --num-classes 178 --num-tasks 29 --base-task-classes 38 --increment-classes 5
fi

complete() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/events.csv" ]] && \
    grep -q 'run_completed' "${run_dir}/events.csv"
}

run_base() {
  local method_id="$1" method="$2" run_dir="$3"
  if complete "${run_dir}"; then
    echo "[skip] complete D1 ${method_id} seed=0"
    return
  fi
  if [[ -d "${run_dir}" && -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
    exit 1
  fi
  echo "[run] D1 ${method_id} seed=0"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/training/train_cil.py" \
    --train "${PROJECT_ROOT}/train_extracted.parquet" \
    --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
    --test "${PROJECT_ROOT}/test_extracted.parquet" \
    --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
    --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
    --variant tddi --batch-size 1024 --effective-batch-size 1024 \
    --epochs 20 --patience 20 --lr 0.001 --weight-decay 0.0001 \
    --device "${DEVICE}" --focal-gamma 1.0 \
    --task-file "${TASK_FILE}" --outdir "${run_dir}" \
    --method "${method}" --sampling natural --seed 0 \
    --total-memory-budget 6800 --replay-draws-per-epoch 6800 \
    --distill-alpha 1.0 --temperature 2.0 --feature-distill-weight 0.5
}

run_base c3 replay_distill_fixed_budget_uniform "${C3_RUN}"
run_base c4 xder_fixed_budget_uniform "${C4_RUN}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_c4_grouprior_h29.py" \
  --c3-run "${C3_RUN}" --c4-run "${C4_RUN}" --seed 0 \
  --setting D1-H29 --summary-name d1_seed0_summary.csv \
  --device "${DEVICE}" --output "${RUN_ROOT}/analysis"

echo "[done] D1 seed 0 and locked GroupPrior-v1 evaluation complete"
