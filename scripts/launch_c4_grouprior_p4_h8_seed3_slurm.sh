#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/tmp/ydang_c4_p4_h8_seed3
BUNDLE="${PROJECT_ROOT}/outputs/c4_p4h8_seed3_stage.tar"
RUN_REL=outputs/runs_c3_c4_diagnostics/p4_h8_tddi_c4_seed3
EVAL_REL=outputs/runs_c4_grouprior_seed3/p4_h8
C3_REL=outputs/runs_backbones/p4_constrained_mass_balanced_seed3_replay_distill_fixed_budget_uniform_mlptddi

if [[ -e "${PROJECT_ROOT}/${RUN_REL}" ]]; then
  echo "Refusing to overwrite ${PROJECT_ROOT}/${RUN_REL}" >&2
  exit 1
fi

tar -C "${PROJECT_ROOT}" -cf "${BUNDLE}" \
  src scripts configs \
  outputs/audit/feature_columns.json outputs/preprocess/scaler.pkl \
  outputs/class_distribution/class_counts_train.csv \
  outputs/tasks/constrained_mass_balanced_seed3_tasks.json \
  "${C3_REL}/run_config.json" "${C3_REL}/run_summary.md" \
  "${C3_REL}/checkpoints/task_7_model.pt"

if [[ "${1:-}" != allocated ]]; then
  salloc -p LocalQ -w mantis-10 --gres=gpu:rtx_pro_4000:1 \
    -c 8 --mem=128G --time=24:00:00 bash "$0" allocated
  rm -f "${BUNDLE}"
  exit 0
fi

srun --chdir=/tmp mkdir -p "${STAGE}"
sbcast -f "${BUNDLE}" "${STAGE}/bundle.tar"
sbcast -f "${PROJECT_ROOT}/train_extracted.parquet" "${STAGE}/train_extracted.parquet"
sbcast -f "${PROJECT_ROOT}/validation_extracted.parquet" "${STAGE}/validation_extracted.parquet"
sbcast -f "${PROJECT_ROOT}/test_extracted.parquet" "${STAGE}/test_extracted.parquet"
srun --chdir=/tmp tar -C "${STAGE}" -xf "${STAGE}/bundle.tar"
srun --chdir=/tmp bash -c "cd '${STAGE}' && bash scripts/worker_c4_grouprior_p4_h8_seed3.sh"
srun --chdir=/tmp tar -C "${STAGE}" -cf - "${RUN_REL}" "${EVAL_REL}" | tar -C "${PROJECT_ROOT}" -xf -
echo "[done] P4-H8 C4+GroupPrior seed 3"
