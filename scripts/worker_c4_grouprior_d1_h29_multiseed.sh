#!/usr/bin/env bash
set -euo pipefail

STAGE=/tmp/ydang_c4_p4_h8_seed3
cd "${STAGE}"
PY=.venv/bin/python
read -r -a SEEDS <<< "${D1_SEEDS:?D1_SEEDS must contain one or more seeds}"
${PY} -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)'

run_one() {
  local seed="$1" method="$2" task="$3" out="$4"
  if [[ -e "${out}" ]]; then
    echo "Refusing to overwrite ${out}" >&2
    exit 1
  fi
  ${PY} src/training/train_cil.py \
    --train train_extracted.parquet --validation validation_extracted.parquet \
    --test test_extracted.parquet --feature-cols outputs/audit/feature_columns.json \
    --scaler outputs/preprocess/scaler.pkl --variant tddi --batch-size 1024 \
    --effective-batch-size 1024 --epochs 20 --patience 20 --lr 0.001 \
    --weight-decay 0.0001 --device cuda --focal-gamma 1.0 \
    --task-file "${task}" --outdir "${out}" --method "${method}" \
    --sampling natural --seed "${seed}" --total-memory-budget 6800 \
    --replay-draws-per-epoch 6800 --distill-alpha 1.0 --temperature 2.0 \
    --feature-distill-weight 0.5
}

for seed in "${SEEDS[@]}"; do
  task="outputs/tasks_c4_grouprior_dynamic/d1/random_seed${seed}_tasks.json"
  root="outputs/runs_c4_grouprior_dynamic/d1"
  c3="${root}/d1_tddi_c3_seed${seed}"
  c4="${root}/d1_tddi_c4_seed${seed}"
  run_one "${seed}" replay_distill_fixed_budget_uniform "${task}" "${c3}"
  run_one "${seed}" xder_fixed_budget_uniform "${task}" "${c4}"
  ${PY} scripts/evaluate_c4_grouprior_h29.py \
    --c3-run "${c3}" --c4-run "${c4}" --seed "${seed}" \
    --setting D1-H29 --summary-name "d1_seed${seed}_summary.csv" \
    --device cuda --output "${root}/analysis/seed${seed}"
done

