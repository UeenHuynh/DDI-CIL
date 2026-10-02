#!/usr/bin/env bash
set -euo pipefail

STAGE=/tmp/ydang_c4_p4_h8_seed3
cd "${STAGE}"
PY=.venv/bin/python
${PY} -m pip install 'tabm>=0.0.3,<0.1' >tabm_install.log 2>&1
${PY} -c 'import torch, tabm; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)'

TASK=outputs/tasks/constrained_mass_balanced_seed0_tasks.json
ROOT=outputs/runs_c4_grouprior_backbones
GRAPH_CACHE=outputs/graph_cache/ddi_gcn/graphs.npz
GRAPH_MAPPING=outputs/graph_cache/ddi_gcn/drug_id_mapping.json

run_c4() {
  local backbone="$1" out="$2"
  shift 2
  if [[ -e "${out}" ]]; then
    echo "Refusing to overwrite ${out}" >&2
    exit 1
  fi
  ${PY} src/training/train_cil.py \
    --train train_extracted.parquet --validation validation_extracted.parquet \
    --test test_extracted.parquet --feature-cols outputs/audit/feature_columns.json \
    --scaler outputs/preprocess/scaler.pkl --variant "${backbone}" \
    --batch-size 256 --effective-batch-size 1024 --epochs 20 --patience 5 \
    --lr 0.001 --weight-decay 0.0001 --device cuda --focal-gamma 1.0 \
    --task-file "${TASK}" --outdir "${out}" --method xder_fixed_budget_uniform \
    --sampling natural --seed 0 --total-memory-budget 6800 \
    --replay-draws-per-epoch 6800 --distill-alpha 1.0 --temperature 2.0 \
    --feature-distill-weight 0.5 "$@"
}

for backbone in tabm ddi_gcn; do
  c3="outputs/runs_selected_backbones/${backbone}_p4_constrained_mass_balanced_seed0_replay_distill_fixed_budget_uniform"
  c4="${ROOT}/p4_h8_${backbone}_c4_seed0"
  analysis="${ROOT}/analysis/${backbone}_seed0"
  if [[ "${backbone}" == tabm ]]; then
    run_c4 tabm "${c4}" \
      --tabm-k 32 --tabm-blocks 3 --tabm-d-block 512 --tabm-dropout 0.1
    graph_args=()
  else
    run_c4 ddi_gcn "${c4}" \
      --graph-cache "${GRAPH_CACHE}" --graph-mapping "${GRAPH_MAPPING}" \
      --ddi-gcn-depth 8 --ddi-gcn-width 128 --ddi-gcn-attention-dim 65
    graph_args=(--graph-cache "${GRAPH_CACHE}" --graph-mapping "${GRAPH_MAPPING}")
  fi
  ${PY} scripts/evaluate_c4_grouprior_p4_h8_backbone.py \
    --backbone "${backbone}" --seed 0 --c3-run "${c3}" --c4-run "${c4}" \
    --device cuda --output "${analysis}" "${graph_args[@]}"
done

