#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/tmp/ydang_c4_p4_h8_seed3
TASK_ROOT=outputs/tasks_c4_grouprior_h29/p4_h29
RUN_ROOT=outputs/runs_c4_grouprior_p4_h29

if [[ "${1:-}" != allocated ]]; then
  tar -C "${PROJECT_ROOT}" -cf "${PROJECT_ROOT}/outputs/c4_p4h29_s14_stage.tar" \
    src scripts configs outputs/audit/feature_columns.json outputs/preprocess/scaler.pkl \
    outputs/class_distribution/class_counts_train.csv \
    "${TASK_ROOT}/constrained_mass_balanced_seed1_tasks.json" \
    "${TASK_ROOT}/constrained_mass_balanced_seed2_tasks.json" \
    "${TASK_ROOT}/constrained_mass_balanced_seed3_tasks.json" \
    "${TASK_ROOT}/constrained_mass_balanced_seed4_tasks.json"
  salloc -p LocalQ -w mantis-10 --gres=gpu:rtx_pro_4000:1 \
    -c 8 --mem=128G --time=24:00:00 bash "$0" allocated
  rm -f "${PROJECT_ROOT}/outputs/c4_p4h29_s14_stage.tar"
  exit 0
fi

srun --chdir=/tmp mkdir -p "${STAGE}/${TASK_ROOT}" "${STAGE}/${RUN_ROOT}"
sbcast -f "${PROJECT_ROOT}/outputs/c4_p4h29_s14_stage.tar" "${STAGE}/bundle_p4h29_s14.tar"
srun --chdir=/tmp tar -C "${STAGE}" -xf "${STAGE}/bundle_p4h29_s14.tar"
srun --chdir=/tmp bash -c "cd '${STAGE}' && bash scripts/worker_c4_grouprior_p4_h29_seeds1_4.sh"
srun --chdir=/tmp tar -C "${STAGE}" -cf - "${RUN_ROOT}" | tar -C "${PROJECT_ROOT}" -xf -
echo "[done] P4-H29 T-DDI seeds 1-4"
