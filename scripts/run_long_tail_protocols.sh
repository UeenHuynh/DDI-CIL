#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL_PYTHON="${PROTOCOL_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
PROTOCOL_DEVICE="${PROTOCOL_DEVICE:-auto}"
PROTOCOL_OUTPUT_ROOT="${PROTOCOL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_long_tail}"
PROTOCOL_MIN_AVAILABLE_GIB="${PROTOCOL_MIN_AVAILABLE_GIB:-12}"
PROTOCOL_MIN_DISK_GIB="${PROTOCOL_MIN_DISK_GIB:-50}"
P10_BETA="${P10_BETA:-0.999}"
P12_SHOCK_TASK="${P12_SHOCK_TASK:-3}"
METHOD="replay_distill_fixed_budget_uniform"
VARIANT="tddi"
read -r -a SEEDS <<< "${LONG_TAIL_SEEDS:-0}"

usage() {
  echo "Usage: $0 [P9|P10|P11|P12|all]" >&2
}

protocol_arg="${1:-all}"
case "${protocol_arg}" in
  P9|P10|P11|P12) protocols=("${protocol_arg}") ;;
  all) protocols=(P9 P10 P11 P12) ;;
  *) usage; exit 2 ;;
esac

if [[ ! -x "${PROTOCOL_PYTHON}" ]]; then
  echo "Python environment not found: ${PROTOCOL_PYTHON}" >&2
  exit 1
fi
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "LONG_TAIL_SEEDS must contain at least one seed." >&2
  exit 2
fi

beta_slug="${P10_BETA//./p}"

task_file_for_protocol() {
  case "$1" in
    P9) echo "${PROJECT_ROOT}/outputs/tasks/tail_profile_balanced_seed${2}_tasks.json" ;;
    P10) echo "${PROJECT_ROOT}/outputs/tasks/effective_number_balanced_beta${beta_slug}_seed${2}_tasks.json" ;;
    P11) echo "${PROJECT_ROOT}/outputs/tasks/progressive_tail_drift_seed${2}_tasks.json" ;;
    P12) echo "${PROJECT_ROOT}/outputs/tasks/tail_shock_task${P12_SHOCK_TASK}_seed${2}_tasks.json" ;;
  esac
}

slug_for_protocol() {
  case "$1" in
    P9) echo "p9_tail_profile_balanced" ;;
    P10) echo "p10_effective_number_balanced_beta${beta_slug}" ;;
    P11) echo "p11_progressive_tail_drift" ;;
    P12) echo "p12_tail_shock_task${P12_SHOCK_TASK}" ;;
  esac
}

preflight_resource_guard() {
  local available_kb disk_kb
  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  disk_kb="$(df -Pk "${PROTOCOL_OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kb < PROTOCOL_MIN_AVAILABLE_GIB * 1024 * 1024 )); then
    echo "Resource guard: available RAM is below ${PROTOCOL_MIN_AVAILABLE_GIB} GiB." >&2
    exit 1
  fi
  if (( disk_kb < PROTOCOL_MIN_DISK_GIB * 1024 * 1024 )); then
    echo "Resource guard: free disk is below ${PROTOCOL_MIN_DISK_GIB} GiB." >&2
    exit 1
  fi
}

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/metrics.csv" ]] && \
    [[ -s "${run_dir}/forgetting.csv" ]] && \
    [[ -s "${run_dir}/class_trajectory.csv" ]] && \
    [[ -s "${run_dir}/class_forgetting.csv" ]]
}

mkdir -p "${PROTOCOL_OUTPUT_ROOT}"

for protocol in "${protocols[@]}"; do
  protocol_slug="$(slug_for_protocol "${protocol}")"
  for seed in "${SEEDS[@]}"; do
    preflight_resource_guard
    task_file="$(task_file_for_protocol "${protocol}" "${seed}")"
    if [[ ! -s "${task_file}" ]]; then
      echo "Missing task file for ${protocol} seed=${seed}: ${task_file}" >&2
      echo "Build it with scripts/build_long_tail_protocols.py first." >&2
      exit 1
    fi
    run_dir="${PROTOCOL_OUTPUT_ROOT}/${protocol_slug}_seed${seed}_${METHOD}_mlp${VARIANT}"
    if is_complete_run "${run_dir}"; then
      echo "[skip] Complete: ${run_dir}"
      continue
    fi
    if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
      exit 1
    fi

    echo "[run] protocol=${protocol} variant=${VARIANT} method=${METHOD} seed=${seed} device=${PROTOCOL_DEVICE}"
    "${PROTOCOL_PYTHON}" "${PROJECT_ROOT}/src/training/train_cil.py" \
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
      --device "${PROTOCOL_DEVICE}" \
      --total-memory-budget 6800 \
      --replay-draws-per-epoch 6800 \
      --distill-alpha 1.0 \
      --temperature 2.0 \
      --feature-distill-weight 0.5 \
      --focal-gamma 1.0
  done
done

run_dirs=()
for protocol in "${protocols[@]}"; do
  protocol_slug="$(slug_for_protocol "${protocol}")"
  for seed in "${SEEDS[@]}"; do
    run_dirs+=("${PROTOCOL_OUTPUT_ROOT}/${protocol_slug}_seed${seed}_${METHOD}_mlp${VARIANT}")
  done
done
"${PROTOCOL_PYTHON}" "${PROJECT_ROOT}/scripts/analyze_long_tail_protocol.py" \
  --shock-task "${P12_SHOCK_TASK}" \
  --output "${PROTOCOL_OUTPUT_ROOT}/analysis" \
  "${run_dirs[@]}"
