#!/usr/bin/env bash
set -euo pipefail

STAGE=/tmp/ydang_c4_p4_h8_seed3
cd "${STAGE}"
PY=.venv/bin/python
MODE="${1:?Usage: worker_h29_ceiling_diagnostics.sh posthoc-or-unlimited}"
ROOT=outputs/runs_c4_next_candidate_diagnostics
TASK=outputs/tasks_c4_grouprior_dynamic/d1/random_seed0_tasks.json

${PY} -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)'

case "${MODE}" in
  posthoc)
    ${PY} scripts/run_h29_classifier_prior_diagnostics.py \
      --c4-run outputs/runs_c4_grouprior_dynamic/d1/d1_tddi_c4_seed0 \
      --task-file "${TASK}" --device cuda --seed 0 \
      --output "${ROOT}/d1_h29_seed0_classifier_prior"
    ;;
  unlimited)
    ${PY} src/training/train_cil.py \
      --train train_extracted.parquet --validation validation_extracted.parquet \
      --test test_extracted.parquet --feature-cols outputs/audit/feature_columns.json \
      --scaler outputs/preprocess/scaler.pkl --variant tddi --batch-size 1024 \
      --effective-batch-size 1024 --epochs 20 --patience 20 --lr 0.001 \
      --weight-decay 0.0001 --device cuda --focal-gamma 1.0 \
      --task-file "${TASK}" --outdir "${ROOT}/d1_h29_unlimited_replay_seed0" \
      --method replay_distill --sampling natural --seed 0 \
      --memory-per-class 1000000 --skip-memory-snapshots \
      --distill-alpha 1.0 --temperature 2.0 --feature-distill-weight 0.5
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
