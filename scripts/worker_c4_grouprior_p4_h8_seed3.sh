#!/usr/bin/env bash
set -euo pipefail

STAGE=/tmp/ydang_c4_p4_h8_seed3
cd "${STAGE}"
tar -xf bundle.tar
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
.venv/bin/python get-pip.py >pip_bootstrap.log 2>&1
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 >torch_install.log 2>&1
.venv/bin/python -m pip install 'numpy<2' pandas pyarrow scikit-learn scipy >data_install.log 2>&1

.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('outputs/runs_backbones/p4_constrained_mass_balanced_seed3_replay_distill_fixed_budget_uniform_mlptddi/run_config.json')
config = json.loads(path.read_text())
config['arguments']['task_file'] = '/tmp/ydang_c4_p4_h8_seed3/outputs/tasks/constrained_mass_balanced_seed3_tasks.json'
path.write_text(json.dumps(config, indent=2))
PY

.venv/bin/python -c 'import torch; print("CUDA:", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)'
.venv/bin/python src/training/train_cil.py \
  --train train_extracted.parquet --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --feature-cols outputs/audit/feature_columns.json \
  --scaler outputs/preprocess/scaler.pkl \
  --variant tddi --batch-size 1024 --effective-batch-size 1024 \
  --epochs 20 --patience 5 --lr 0.001 --weight-decay 0.0001 \
  --device cuda --focal-gamma 1.0 \
  --task-file outputs/tasks/constrained_mass_balanced_seed3_tasks.json \
  --outdir outputs/runs_c3_c4_diagnostics/p4_h8_tddi_c4_seed3 \
  --method xder_fixed_budget_uniform --sampling natural --seed 3 \
  --total-memory-budget 6800 --replay-draws-per-epoch 6800 \
  --distill-alpha 1.0 --temperature 2.0 --feature-distill-weight 0.5

.venv/bin/python scripts/evaluate_c4_grouprior_p4_h8_seed.py \
  --seed 3 --device cuda --output outputs/runs_c4_grouprior_seed3/p4_h8
