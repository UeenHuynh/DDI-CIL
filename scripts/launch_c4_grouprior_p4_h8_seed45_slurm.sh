#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/tmp/ydang_c4_p4_h8_seed3
BUNDLE="${PROJECT_ROOT}/outputs/c4_p4h8_seed45_stage.tar"
C3_4=outputs/runs_backbones/p4_constrained_mass_balanced_seed4_replay_distill_fixed_budget_uniform_mlptddi

for run in \
  outputs/runs_c3_c4_diagnostics/p4_h8_tddi_c4_seed4 \
  outputs/runs_c4_grouprior_extra_seed5; do
  if [[ -e "${PROJECT_ROOT}/${run}" ]]; then
    echo "Refusing to overwrite ${PROJECT_ROOT}/${run}" >&2
    exit 1
  fi
done

if [[ "${1:-}" != allocated ]]; then
  tar -C "${PROJECT_ROOT}" -cf "${BUNDLE}" \
    src scripts configs \
    outputs/audit/feature_columns.json outputs/preprocess/scaler.pkl \
    outputs/class_distribution/class_counts_train.csv \
    outputs/tasks/constrained_mass_balanced_seed4_tasks.json \
    outputs/tasks_extra_seed5/constrained_mass_balanced_seed5_tasks.json \
    "${C3_4}/run_config.json" "${C3_4}/run_summary.md" \
    "${C3_4}/checkpoints/task_7_model.pt"
  salloc -p LocalQ -w mantis-10 --gres=gpu:rtx_pro_4000:1 \
    -c 8 --mem=128G --time=24:00:00 bash "$0" allocated
  rm -f "${BUNDLE}"
  exit 0
fi

srun --chdir=/tmp test -x "${STAGE}/.venv/bin/python"
srun --chdir=/tmp test -s "${STAGE}/train_extracted.parquet"
sbcast -f "${BUNDLE}" "${STAGE}/bundle45.tar"
srun --chdir=/tmp tar -C "${STAGE}" -xf "${STAGE}/bundle45.tar"
srun --chdir=/tmp bash -c "cd '${STAGE}' && bash scripts/worker_c4_grouprior_p4_h8_seed45.sh"
srun --chdir=/tmp tar -C "${STAGE}" -cf - \
  outputs/runs_c3_c4_diagnostics/p4_h8_tddi_c4_seed4 \
  outputs/runs_c4_grouprior_seed4 \
  outputs/runs_c4_grouprior_extra_seed5 | tar -C "${PROJECT_ROOT}" -xf -
echo "[done] P4-H8 C4+GroupPrior seeds 4 and 5"
