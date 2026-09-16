#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY_PYTHON="${STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
STUDY_DEVICE="${STUDY_DEVICE:-cuda}"
STUDY_OUTPUT_ROOT="${STUDY_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_p9_backbones}"
STUDY_MIN_AVAILABLE_GIB="${STUDY_MIN_AVAILABLE_GIB:-12}"
STUDY_MIN_DISK_GIB="${STUDY_MIN_DISK_GIB:-50}"
STUDY_MIN_GPU_FREE_MIB="${STUDY_MIN_GPU_FREE_MIB:-10000}"
FROZEN_COMMIT="6588915011ee3d0d95519aefc54c4d0a885a44e5"
FROZEN_HASH="643b757eecfc9457cdf9db7c746fb0d0c69426b94d839373fd35a6400339498d"
FROZEN_ROOT="${P9_FROZEN_ROOT:-/tmp/ddi_p9_backbone_6588915}"
read -r -a SEEDS <<< "${P9_BACKBONE_SEEDS:-0 1 2 3 4}"
read -r -a BACKBONES <<< "${P9_BACKBONES:-tabm ddi_gcn}"

METHOD="replay_distill_fixed_budget_uniform"
GRAPH_CACHE="${PROJECT_ROOT}/outputs/graph_cache/ddi_gcn/graphs.npz"
GRAPH_MAPPING="${PROJECT_ROOT}/outputs/graph_cache/ddi_gcn/drug_id_mapping.json"

prepare_frozen_source() {
  local actual_hash
  if [[ ! -f "${FROZEN_ROOT}/src/training/train_cil.py" ]]; then
    mkdir -p "${FROZEN_ROOT}"
    git -C "${PROJECT_ROOT}" archive "${FROZEN_COMMIT}" | tar -xf - -C "${FROZEN_ROOT}"
  fi
  actual_hash="$(sha256sum "${FROZEN_ROOT}/src/training/train_cil.py" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${FROZEN_HASH}" ]]; then
    echo "Frozen training source hash mismatch: ${actual_hash}" >&2
    exit 1
  fi
}

preflight_resource_guard() {
  local available_kb disk_kb gpu_free_mib
  mkdir -p "${STUDY_OUTPUT_ROOT}"
  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  disk_kb="$(df -Pk "${STUDY_OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kb < STUDY_MIN_AVAILABLE_GIB * 1024 * 1024 )); then
    echo "Resource guard: available RAM is below ${STUDY_MIN_AVAILABLE_GIB} GiB." >&2
    exit 1
  fi
  if (( disk_kb < STUDY_MIN_DISK_GIB * 1024 * 1024 )); then
    echo "Resource guard: free disk is below ${STUDY_MIN_DISK_GIB} GiB." >&2
    exit 1
  fi
  if [[ "${STUDY_DEVICE}" == cuda* ]]; then
    gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')"
    if [[ -z "${gpu_free_mib}" ]] || (( gpu_free_mib < STUDY_MIN_GPU_FREE_MIB )); then
      echo "Resource guard: free GPU memory is below ${STUDY_MIN_GPU_FREE_MIB} MiB." >&2
      exit 1
    fi
  fi
}

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/metrics.csv" ]] && \
    [[ -s "${run_dir}/forgetting.csv" ]] && \
    [[ -s "${run_dir}/class_trajectory.csv" ]] && \
    [[ -s "${run_dir}/class_forgetting.csv" ]] && \
    [[ -s "${run_dir}/training_audit.csv" ]] && \
    [[ -s "${run_dir}/replay_budget_audit.csv" ]] && \
    rg -q 'run_completed' "${run_dir}/events.csv"
}

prepare_frozen_source
mkdir -p "${STUDY_OUTPUT_ROOT}"

for backbone in "${BACKBONES[@]}"; do
  case "${backbone}" in
    tabm|ddi_gcn) ;;
    *) echo "Unsupported P9 backbone: ${backbone}" >&2; exit 2 ;;
  esac
  if [[ "${backbone}" == "ddi_gcn" ]] && \
     { [[ ! -s "${GRAPH_CACHE}" ]] || [[ ! -s "${GRAPH_MAPPING}" ]]; }; then
    echo "DDI-GCN graph cache is missing." >&2
    exit 1
  fi

  for seed in "${SEEDS[@]}"; do
    preflight_resource_guard
    task_file="${PROJECT_ROOT}/outputs/tasks/tail_profile_balanced_seed${seed}_tasks.json"
    run_dir="${STUDY_OUTPUT_ROOT}/${backbone}_p9_tail_profile_balanced_seed${seed}_${METHOD}"
    if [[ ! -s "${task_file}" ]]; then
      echo "Missing P9 task file: ${task_file}" >&2
      exit 1
    fi
    if is_complete_run "${run_dir}"; then
      echo "[skip] Complete: ${run_dir}"
      continue
    fi
    if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
      exit 1
    fi

    model_args=()
    if [[ "${backbone}" == "tabm" ]]; then
      model_args+=(--tabm-k 32 --tabm-blocks 3 --tabm-d-block 512 --tabm-dropout 0.1)
    else
      model_args+=(
        --graph-cache "${GRAPH_CACHE}"
        --graph-mapping "${GRAPH_MAPPING}"
        --ddi-gcn-depth 8
        --ddi-gcn-width 128
        --ddi-gcn-attention-dim 65
      )
    fi

    echo "[run] backbone=${backbone} protocol=P9 horizon=H8 seed=${seed} device=${STUDY_DEVICE} frozen_commit=${FROZEN_COMMIT}"
    "${STUDY_PYTHON}" "${FROZEN_ROOT}/src/training/train_cil.py" \
      --train "${PROJECT_ROOT}/train_extracted.parquet" \
      --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
      --test "${PROJECT_ROOT}/test_extracted.parquet" \
      --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
      --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
      --task-file "${task_file}" \
      --outdir "${run_dir}" \
      --method "${METHOD}" \
      --variant "${backbone}" \
      --batch-size 256 \
      --effective-batch-size 1024 \
      --epochs 20 \
      --lr 0.001 \
      --weight-decay 0.0001 \
      --patience 5 \
      --seed "${seed}" \
      --device "${STUDY_DEVICE}" \
      --total-memory-budget 6800 \
      --replay-draws-per-epoch 6800 \
      --distill-alpha 1.0 \
      --temperature 2.0 \
      --feature-distill-weight 0.5 \
      --focal-gamma 1.0 \
      "${model_args[@]}"
  done
done

echo "[done] P9 full-backbone runs are complete under ${STUDY_OUTPUT_ROOT}"
