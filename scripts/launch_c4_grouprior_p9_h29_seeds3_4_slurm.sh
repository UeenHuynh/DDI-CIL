#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/tmp/ydang_c4_p4_h8_seed3
TASK_ROOT=outputs/tasks_c4_grouprior_h29/p9_h29
RUN_ROOT=outputs/runs_c4_grouprior_h29
BUNDLE="${PROJECT_ROOT}/outputs/c4_p9h29_s34_stage.tar"

for seed in 3 4; do
  for method in c3 c4; do
    if [[ -e "${PROJECT_ROOT}/${RUN_ROOT}/p9_h29_tddi_${method}_seed${seed}" ]]; then
      echo "Refusing to overwrite P9-H29 ${method} seed ${seed}" >&2
      exit 1
    fi
  done
done

if [[ "${1:-}" != allocated ]]; then
  tar -C "${PROJECT_ROOT}" -cf "${BUNDLE}" src scripts configs \
    outputs/audit/feature_columns.json outputs/preprocess/scaler.pkl \
    outputs/class_distribution/class_counts_train.csv \
    "${TASK_ROOT}/tail_profile_balanced_seed3_tasks.json" \
    "${TASK_ROOT}/tail_profile_balanced_seed4_tasks.json"
  salloc -p LocalQ -w mantis-05 --gres=gpu:rtx_a6000:1 \
    -c 8 --mem=128G --time=24:00:00 bash "$0" allocated
  rm -f "${BUNDLE}"
  exit 0
fi

srun --chdir=/tmp mkdir -p "${STAGE}/${TASK_ROOT}" "${STAGE}/${RUN_ROOT}"
sbcast -f "${BUNDLE}" "${STAGE}/bundle_p9h29_s34.tar"
if ! srun --chdir=/tmp test -s "${STAGE}/train_extracted.parquet"; then
  sbcast -f "${PROJECT_ROOT}/train_extracted.parquet" "${STAGE}/train_extracted.parquet"
fi
if ! srun --chdir=/tmp test -s "${STAGE}/validation_extracted.parquet"; then
  sbcast -f "${PROJECT_ROOT}/validation_extracted.parquet" "${STAGE}/validation_extracted.parquet"
fi
if ! srun --chdir=/tmp test -s "${STAGE}/test_extracted.parquet"; then
  sbcast -f "${PROJECT_ROOT}/test_extracted.parquet" "${STAGE}/test_extracted.parquet"
fi
srun --chdir=/tmp tar -C "${STAGE}" -xf "${STAGE}/bundle_p9h29_s34.tar"
srun --chdir=/tmp bash -c "cd '${STAGE}' && bash scripts/worker_c4_grouprior_p9_h29_seeds3_4.sh"
srun --chdir=/tmp tar -C "${STAGE}" -cf - \
  "${RUN_ROOT}/p9_h29_tddi_c3_seed3" "${RUN_ROOT}/p9_h29_tddi_c4_seed3" \
  "${RUN_ROOT}/p9_h29_tddi_c3_seed4" "${RUN_ROOT}/p9_h29_tddi_c4_seed4" \
  "${RUN_ROOT}/analysis/seed3" "${RUN_ROOT}/analysis/seed4" | \
  tar -C "${PROJECT_ROOT}" -xf -

/mnt/beegfs/ydang/BRATS/envs/brats-py311/bin/python \
  "${PROJECT_ROOT}/scripts/aggregate_c4_grouprior_h29.py" \
  --analysis-root "${PROJECT_ROOT}/${RUN_ROOT}/analysis" \
  --seeds 0 1 2 3 4 --setting P9-H29 --output-prefix h29_five_seed
echo "[done] P9-H29 T-DDI seeds 3-4 and five-seed aggregate"
