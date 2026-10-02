#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/tmp/ydang_c4_p4_h8_seed3
ROOT=outputs/runs_c4_grouprior_dynamic/d1
TASK_ROOT=outputs/tasks_c4_grouprior_dynamic/d1
GROUP="${1:-}"

case "${GROUP}" in
  12) SEEDS=(1 2) ;;
  34) SEEDS=(3 4) ;;
  *) echo "Usage: $0 {12|34} [allocated]" >&2; exit 2 ;;
esac

BUNDLE="${PROJECT_ROOT}/outputs/c4_d1h29_seed${GROUP}_stage.tar"
for seed in "${SEEDS[@]}"; do
  for method in c3 c4; do
    if [[ -e "${PROJECT_ROOT}/${ROOT}/d1_tddi_${method}_seed${seed}" ]]; then
      echo "Refusing to overwrite D1-H29 ${method} seed ${seed}" >&2
      exit 1
    fi
  done
done

if [[ "${2:-}" != allocated ]]; then
  cd "${PROJECT_ROOT}"
  tar -cf "${BUNDLE}" src scripts configs \
    outputs/audit/feature_columns.json outputs/preprocess/scaler.pkl \
    outputs/class_distribution/class_counts_train.csv \
    "${TASK_ROOT}/random_seed1_tasks.json" \
    "${TASK_ROOT}/random_seed2_tasks.json" \
    "${TASK_ROOT}/random_seed3_tasks.json" \
    "${TASK_ROOT}/random_seed4_tasks.json"
  salloc -p LocalQ -w mantis-05 --gres=gpu:rtx_a6000:1 \
    -c 8 --mem=128G --time=24:00:00 bash "$0" "${GROUP}" allocated
  rm -f "${BUNDLE}"
  exit 0
fi

srun --chdir=/tmp mkdir -p "${STAGE}/${TASK_ROOT}" "${STAGE}/${ROOT}/analysis"
sbcast -f "${BUNDLE}" "${STAGE}/bundle_d1h29_seed${GROUP}.tar"
if ! srun --chdir=/tmp test -s "${STAGE}/train_extracted.parquet"; then
  sbcast -f "${PROJECT_ROOT}/train_extracted.parquet" "${STAGE}/train_extracted.parquet"
fi
if ! srun --chdir=/tmp test -s "${STAGE}/validation_extracted.parquet"; then
  sbcast -f "${PROJECT_ROOT}/validation_extracted.parquet" "${STAGE}/validation_extracted.parquet"
fi
if ! srun --chdir=/tmp test -s "${STAGE}/test_extracted.parquet"; then
  sbcast -f "${PROJECT_ROOT}/test_extracted.parquet" "${STAGE}/test_extracted.parquet"
fi
srun --chdir=/tmp tar -C "${STAGE}" -xf "${STAGE}/bundle_d1h29_seed${GROUP}.tar"
seed_string="${SEEDS[*]}"
srun --chdir=/tmp bash -c "cd '${STAGE}' && D1_SEEDS='${seed_string}' bash scripts/worker_c4_grouprior_d1_h29_multiseed.sh"

copy_paths=()
for seed in "${SEEDS[@]}"; do
  copy_paths+=(
    "${ROOT}/d1_tddi_c3_seed${seed}"
    "${ROOT}/d1_tddi_c4_seed${seed}"
    "${ROOT}/analysis/seed${seed}"
  )
done
srun --chdir=/tmp tar -C "${STAGE}" -cf - "${copy_paths[@]}" | \
  tar -C "${PROJECT_ROOT}" -xf -

if [[ "${GROUP}" == 34 ]]; then
  /mnt/beegfs/ydang/BRATS/envs/brats-py311/bin/python \
    "${PROJECT_ROOT}/scripts/aggregate_c4_grouprior_h29.py" \
    --analysis-root "${PROJECT_ROOT}/${ROOT}/analysis" \
    --seeds 0 1 2 3 4 --setting D1-H29 --output-prefix d1_five_seed \
    --summary-template 'd1_seed{seed}_summary.csv'
fi
echo "[done] D1-H29 T-DDI seeds ${SEEDS[*]}"
