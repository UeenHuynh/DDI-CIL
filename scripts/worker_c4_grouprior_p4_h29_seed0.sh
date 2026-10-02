#!/usr/bin/env bash
set -euo pipefail

STAGE=/tmp/ydang_c4_p4_h8_seed3
cd "${STAGE}"
PY=.venv/bin/python
TASK=outputs/tasks_c4_grouprior_h29/p4_h29/constrained_mass_balanced_seed0_tasks.json
C3=outputs/runs_c4_grouprior_p4_h29/p4_h29_tddi_c3_seed0
C4=outputs/runs_c4_grouprior_p4_h29/p4_h29_tddi_c4_seed0

${PY} -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)'

run_one() {
  local method="$1" out="$2"
  if [[ -e "${out}" ]]; then
    echo "Refusing to overwrite ${out}" >&2
    exit 1
  fi
  ${PY} src/training/train_cil.py \
    --train train_extracted.parquet --validation validation_extracted.parquet \
    --test test_extracted.parquet --feature-cols outputs/audit/feature_columns.json \
    --scaler outputs/preprocess/scaler.pkl --variant tddi --batch-size 1024 \
    --effective-batch-size 1024 --epochs 20 --patience 5 --lr 0.001 \
    --weight-decay 0.0001 --device cuda --focal-gamma 1.0 \
    --task-file "${TASK}" --outdir "${out}" --method "${method}" \
    --sampling natural --seed 0 --total-memory-budget 6800 \
    --replay-draws-per-epoch 6800 --distill-alpha 1.0 --temperature 2.0 \
    --feature-distill-weight 0.5
}

run_one replay_distill_fixed_budget_uniform "${C3}"
run_one xder_fixed_budget_uniform "${C4}"
${PY} scripts/evaluate_c4_grouprior_h29.py --c3-run "${C3}" --c4-run "${C4}" \
  --seed 0 --setting P4-H29 --device cuda \
  --output outputs/runs_c4_grouprior_p4_h29/analysis
