#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${DIAG_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${DIAG_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
STAGE=/tmp/ydang_c4_p4_h8_seed3
MODE="${1:?Usage: launch_h29_ceiling_diagnostic_slurm.sh posthoc-or-unlimited [allocated]}"
ROOT=outputs/runs_c4_next_candidate_diagnostics
TASK=outputs/tasks_c4_grouprior_dynamic/d1/random_seed0_tasks.json
BUNDLE="${PROJECT_ROOT}/outputs/h29_ceiling_${MODE}_stage.tar"

case "${MODE}" in
  posthoc)
    OUTPUT_REL="${ROOT}/d1_h29_seed0_classifier_prior"
    WALLTIME=08:00:00
    ;;
  unlimited)
    OUTPUT_REL="${ROOT}/d1_h29_unlimited_replay_seed0"
    WALLTIME=48:00:00
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

if [[ "${2:-}" != allocated ]]; then
  if [[ -e "${PROJECT_ROOT}/${OUTPUT_REL}" ]]; then
    echo "Refusing to overwrite ${OUTPUT_REL}" >&2
    exit 1
  fi
  cd "${PROJECT_ROOT}"
  paths=(
    src scripts configs
    outputs/audit/feature_columns.json outputs/preprocess/scaler.pkl
    outputs/class_distribution/class_counts_train.csv "${TASK}"
  )
  if [[ "${MODE}" == posthoc ]]; then
    paths+=(
      outputs/runs_c4_grouprior_dynamic/d1/d1_tddi_c4_seed0/run_config.json
      outputs/runs_c4_grouprior_dynamic/d1/d1_tddi_c4_seed0/checkpoints/task_28_model.pt
    )
  fi
  tar -cf "${BUNDLE}" "${paths[@]}"
  submit_args=(
    --parsable --partition=LocalQ --nodelist=mantis-05
    --gres=gpu:rtx_a6000:1 --cpus-per-task=8 --mem=128G
    --time="${WALLTIME}" --job-name="h29-${MODE}"
    --output="${PROJECT_ROOT}/${ROOT}/slurm_${MODE}_%j.out"
    --error="${PROJECT_ROOT}/${ROOT}/slurm_${MODE}_%j.err"
    --export="ALL,DIAG_PROJECT_ROOT=${PROJECT_ROOT}"
  )
  if [[ -n "${DIAG_DEPENDENCY:-}" ]]; then
    submit_args+=(--dependency="afterok:${DIAG_DEPENDENCY}")
  fi
  mkdir -p "${PROJECT_ROOT}/${ROOT}"
  exec sbatch "${submit_args[@]}" "$0" "${MODE}" allocated
fi

srun --chdir=/tmp mkdir -p "${STAGE}/${ROOT}"
sbcast -f "${BUNDLE}" "${STAGE}/bundle_h29_ceiling_${MODE}.tar"
for split in train validation test; do
  if ! srun --chdir=/tmp test -s "${STAGE}/${split}_extracted.parquet"; then
    sbcast -f "${PROJECT_ROOT}/${split}_extracted.parquet" \
      "${STAGE}/${split}_extracted.parquet"
  fi
done
srun --chdir=/tmp tar -C "${STAGE}" -xf "${STAGE}/bundle_h29_ceiling_${MODE}.tar"
srun --chdir=/tmp bash -c "cd '${STAGE}' && bash scripts/worker_h29_ceiling_diagnostics.sh '${MODE}'"
srun --chdir=/tmp tar -C "${STAGE}" -cf - "${OUTPUT_REL}" | tar -C "${PROJECT_ROOT}" -xf -
rm -f "${BUNDLE}"
echo "[done] ${MODE}: ${OUTPUT_REL}"
