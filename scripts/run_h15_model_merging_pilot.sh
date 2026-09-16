#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H15_PYTHON="${H15_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
H15_DEVICE="${H15_DEVICE:-auto}"
H15_OUTPUT_ROOT="${H15_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_h15_merging}"
H15_TASK_ROOT="${H15_TASK_ROOT:-${PROJECT_ROOT}/outputs/tasks_h15}"
H15_MIN_AVAILABLE_GIB="${H15_MIN_AVAILABLE_GIB:-12}"
H15_MIN_DISK_GIB="${H15_MIN_DISK_GIB:-50}"
H15_MERGE_ALPHA="${H15_MERGE_ALPHA:-0.5}"
read -r -a SEEDS <<< "${H15_SEEDS:-0}"
read -r -a PROTOCOLS <<< "${H15_PROTOCOLS:-P4 P9}"
read -r -a STRATEGIES <<< "${H15_STRATEGIES:-none ema_backbone selective}"

METHOD="replay_distill_fixed_budget_uniform"
VARIANT="tddi"

if [[ ! -x "${H15_PYTHON}" ]]; then
  echo "Python environment not found: ${H15_PYTHON}" >&2
  exit 1
fi

task_file_for_protocol() {
  case "$1" in
    P4) echo "${H15_TASK_ROOT}/p4/constrained_mass_balanced_seed${2}_tasks.json" ;;
    P9) echo "${H15_TASK_ROOT}/p9/tail_profile_balanced_seed${2}_tasks.json" ;;
    *) echo "Unsupported H15 protocol: $1" >&2; exit 2 ;;
  esac
}

slug_for_protocol() {
  case "$1" in
    P4) echo "p4_constrained_mass_balanced" ;;
    P9) echo "p9_tail_profile_balanced" ;;
  esac
}

preflight_resource_guard() {
  local available_kb disk_kb
  mkdir -p "${H15_OUTPUT_ROOT}"
  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  disk_kb="$(df -Pk "${H15_OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kb < H15_MIN_AVAILABLE_GIB * 1024 * 1024 )); then
    echo "Resource guard: available RAM is below ${H15_MIN_AVAILABLE_GIB} GiB." >&2
    exit 1
  fi
  if (( disk_kb < H15_MIN_DISK_GIB * 1024 * 1024 )); then
    echo "Resource guard: free disk is below ${H15_MIN_DISK_GIB} GiB." >&2
    exit 1
  fi
}

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/metrics.csv" ]] && \
    [[ -s "${run_dir}/forgetting.csv" ]] && \
    [[ -s "${run_dir}/class_trajectory.csv" ]] && \
    [[ -s "${run_dir}/class_forgetting.csv" ]] && \
    [[ -s "${run_dir}/merge_audit.csv" ]]
}

for protocol in "${PROTOCOLS[@]}"; do
  protocol_slug="$(slug_for_protocol "${protocol}")"
  for strategy in "${STRATEGIES[@]}"; do
    case "${strategy}" in
      none|ema_backbone|selective) ;;
      *) echo "Unsupported merge strategy: ${strategy}" >&2; exit 2 ;;
    esac
    for seed in "${SEEDS[@]}"; do
      preflight_resource_guard
      task_file="$(task_file_for_protocol "${protocol}" "${seed}")"
      if [[ ! -s "${task_file}" ]]; then
        echo "Missing H15 task file: ${task_file}" >&2
        exit 1
      fi
      run_dir="${H15_OUTPUT_ROOT}/${protocol_slug}_h15_seed${seed}_${strategy}_alpha${H15_MERGE_ALPHA//./p}"
      if is_complete_run "${run_dir}"; then
        echo "[skip] Complete: ${run_dir}"
        continue
      fi
      if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
        exit 1
      fi

      echo "[run] horizon=H15 protocol=${protocol} strategy=${strategy} alpha=${H15_MERGE_ALPHA} seed=${seed} device=${H15_DEVICE}"
      "${H15_PYTHON}" "${PROJECT_ROOT}/src/training/train_cil.py" \
        --train "${PROJECT_ROOT}/train_extracted.parquet" \
        --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
        --test "${PROJECT_ROOT}/test_extracted.parquet" \
        --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
        --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
        --task-file "${task_file}" \
        --outdir "${run_dir}" \
        --method "${METHOD}" \
        --variant "${VARIANT}" \
        --batch-size 1024 \
        --epochs 20 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --dropout 0.2 \
        --activation gelu \
        --norm layernorm \
        --patience 5 \
        --seed "${seed}" \
        --device "${H15_DEVICE}" \
        --total-memory-budget 6800 \
        --replay-draws-per-epoch 6800 \
        --distill-alpha 1.0 \
        --temperature 2.0 \
        --feature-distill-weight 0.5 \
        --focal-gamma 1.0 \
        --merge-strategy "${strategy}" \
        --merge-alpha "${H15_MERGE_ALPHA}"
    done
  done
done

echo "[done] H15 pilot runs are complete under ${H15_OUTPUT_ROOT}"
