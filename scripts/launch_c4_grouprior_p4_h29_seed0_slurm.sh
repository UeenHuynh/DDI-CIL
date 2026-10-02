#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/tmp/ydang_c4_p4_h8_seed3
TASK_REL=outputs/tasks_c4_grouprior_h29/p4_h29/constrained_mass_balanced_seed0_tasks.json
TASK_SRC="${PROJECT_ROOT}/${TASK_REL}"
WORKER="${PROJECT_ROOT}/scripts/worker_c4_grouprior_p4_h29_seed0.sh"

if [[ -e "${PROJECT_ROOT}/outputs/runs_c4_grouprior_p4_h29" ]]; then
  echo "Refusing to overwrite existing P4-H29 output root" >&2
  exit 1
fi

if [[ "${1:-}" != allocated ]]; then
  salloc -p LocalQ -w mantis-10 --gres=gpu:rtx_pro_4000:1 \
    -c 8 --mem=128G --time=24:00:00 bash "$0" allocated
  exit 0
fi

srun --chdir=/tmp mkdir -p "${STAGE}/outputs/tasks_c4_grouprior_h29/p4_h29" "${STAGE}/outputs/runs_c4_grouprior_p4_h29"
sbcast -f "${TASK_SRC}" "${STAGE}/${TASK_REL}"
sbcast -f "${WORKER}" "${STAGE}/scripts/worker_c4_grouprior_p4_h29_seed0.sh"
srun --chdir=/tmp bash -c "cd '${STAGE}' && bash scripts/worker_c4_grouprior_p4_h29_seed0.sh"
srun --chdir=/tmp tar -C "${STAGE}" -cf - \
  outputs/runs_c4_grouprior_p4_h29 | tar -C "${PROJECT_ROOT}" -xf -
echo "[done] P4-H29 C4+GroupPrior seed 0"
