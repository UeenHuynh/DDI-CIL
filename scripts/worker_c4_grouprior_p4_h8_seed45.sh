#!/usr/bin/env bash
set -euo pipefail

STAGE=/tmp/ydang_c4_p4_h8_seed3
cd "${STAGE}"
test -x .venv/bin/python
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0), flush=True)'

.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('outputs/runs_backbones/p4_constrained_mass_balanced_seed4_replay_distill_fixed_budget_uniform_mlptddi/run_config.json')
config = json.loads(path.read_text())
config['arguments']['task_file'] = '/tmp/ydang_c4_p4_h8_seed3/outputs/tasks/constrained_mass_balanced_seed4_tasks.json'
path.write_text(json.dumps(config, indent=2))
PY

run_one() {
  local seed="$1" method="$2" task_file="$3" outdir="$4"
  if [[ -e "${outdir}" ]]; then
    echo "Refusing to overwrite ${outdir}" >&2
    exit 1
  fi
  .venv/bin/python src/training/train_cil.py \
    --train train_extracted.parquet --validation validation_extracted.parquet \
    --test test_extracted.parquet \
    --feature-cols outputs/audit/feature_columns.json \
    --scaler outputs/preprocess/scaler.pkl \
    --variant tddi --batch-size 1024 --effective-batch-size 1024 \
    --epochs 20 --patience 5 --lr 0.001 --weight-decay 0.0001 \
    --device cuda --focal-gamma 1.0 --task-file "${task_file}" \
    --outdir "${outdir}" --method "${method}" --sampling natural \
    --seed "${seed}" --total-memory-budget 6800 \
    --replay-draws-per-epoch 6800 --distill-alpha 1.0 \
    --temperature 2.0 --feature-distill-weight 0.5
}

run_one 4 xder_fixed_budget_uniform \
  outputs/tasks/constrained_mass_balanced_seed4_tasks.json \
  outputs/runs_c3_c4_diagnostics/p4_h8_tddi_c4_seed4
.venv/bin/python scripts/evaluate_c4_grouprior_p4_h8_seed.py \
  --seed 4 --device cuda --output outputs/runs_c4_grouprior_seed4/p4_h8

TASK5=outputs/tasks_extra_seed5/constrained_mass_balanced_seed5_tasks.json
C3_5=outputs/runs_c4_grouprior_extra_seed5/p4_h8_tddi_c3_seed5
C4_5=outputs/runs_c4_grouprior_extra_seed5/p4_h8_tddi_c4_seed5
run_one 5 replay_distill_fixed_budget_uniform "${TASK5}" "${C3_5}"
run_one 5 xder_fixed_budget_uniform "${TASK5}" "${C4_5}"
.venv/bin/python scripts/evaluate_c4_grouprior_p4_h8_seed.py \
  --seed 5 --device cuda --c3-run "${C3_5}" --c4-run "${C4_5}" \
  --output outputs/runs_c4_grouprior_extra_seed5/p4_h8_groupprior
