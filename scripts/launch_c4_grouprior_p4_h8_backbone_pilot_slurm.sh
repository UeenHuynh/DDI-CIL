#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE=/tmp/ydang_c4_p4_h8_seed3
ROOT=outputs/runs_c4_grouprior_backbones
BUNDLE="${PROJECT_ROOT}/outputs/c4_backbone_pilot_stage.tar"

for backbone in tabm ddi_gcn; do
  if [[ -e "${PROJECT_ROOT}/${ROOT}/p4_h8_${backbone}_c4_seed0" ]]; then
    echo "Refusing to overwrite P4-H8 ${backbone} C4 seed 0" >&2
    exit 1
  fi
done

if [[ "${1:-}" != allocated ]]; then
  cd "${PROJECT_ROOT}"
  tar -cf "${BUNDLE}" src scripts configs \
    outputs/audit/feature_columns.json outputs/preprocess/scaler.pkl \
    outputs/class_distribution/class_counts_train.csv \
    outputs/tasks/constrained_mass_balanced_seed0_tasks.json \
    outputs/graph_cache/ddi_gcn \
    outputs/runs_selected_backbones/tabm_p4_constrained_mass_balanced_seed0_replay_distill_fixed_budget_uniform/run_config.json \
    outputs/runs_selected_backbones/tabm_p4_constrained_mass_balanced_seed0_replay_distill_fixed_budget_uniform/run_summary.md \
    outputs/runs_selected_backbones/tabm_p4_constrained_mass_balanced_seed0_replay_distill_fixed_budget_uniform/checkpoints/task_7_model.pt \
    outputs/runs_selected_backbones/ddi_gcn_p4_constrained_mass_balanced_seed0_replay_distill_fixed_budget_uniform/run_config.json \
    outputs/runs_selected_backbones/ddi_gcn_p4_constrained_mass_balanced_seed0_replay_distill_fixed_budget_uniform/run_summary.md \
    outputs/runs_selected_backbones/ddi_gcn_p4_constrained_mass_balanced_seed0_replay_distill_fixed_budget_uniform/checkpoints/task_7_model.pt
  salloc -p LocalQ -w mantis-05 --gres=gpu:rtx_a6000:1 \
    -c 8 --mem=128G --time=24:00:00 bash "$0" allocated
  rm -f "${BUNDLE}"
  exit 0
fi

srun --chdir=/tmp mkdir -p "${STAGE}/${ROOT}"
sbcast -f "${BUNDLE}" "${STAGE}/bundle_backbone_pilot.tar"
srun --chdir=/tmp tar -C "${STAGE}" -xf "${STAGE}/bundle_backbone_pilot.tar"
srun --chdir=/tmp bash -c "cd '${STAGE}' && bash scripts/worker_c4_grouprior_p4_h8_backbone_pilot.sh"
srun --chdir=/tmp tar -C "${STAGE}" -cf - \
  "${ROOT}/p4_h8_tabm_c4_seed0" "${ROOT}/p4_h8_ddi_gcn_c4_seed0" \
  "${ROOT}/analysis/tabm_seed0" "${ROOT}/analysis/ddi_gcn_seed0" | \
  tar -C "${PROJECT_ROOT}" -xf -
echo "[done] P4-H8 C4+GroupPrior cross-backbone seed-0 pilot"
