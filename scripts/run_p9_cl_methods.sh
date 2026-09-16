#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${P9_CL_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${P9_CL_DEVICE:-cuda}"
OUTPUT_ROOT="${P9_CL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_p9_cl_methods}"
read -r -a SEEDS <<< "${P9_CL_SEEDS:-0 1 2 3 4}"
read -r -a METHODS <<< "${P9_CL_METHODS:-C0 C1 C2 C4 C6 C0O C1O C5}"

MIN_RAM_GIB="${P9_CL_MIN_RAM_GIB:-12}"
MIN_DISK_GIB="${P9_CL_MIN_DISK_GIB:-50}"
MIN_GPU_FREE_MIB="${P9_CL_MIN_GPU_FREE_MIB:-10000}"
BASELINE_ROOT="${PROJECT_ROOT}/outputs/runs_long_tail"
SOURCE_FINGERPRINT="$(sha256sum \
  "${PROJECT_ROOT}/src/training/train_cil.py" \
  "${PROJECT_ROOT}/src/data/fixed_budget_replay.py" \
  "${PROJECT_ROOT}/src/data/historical_logits.py" \
  "${PROJECT_ROOT}/src/methods/dark_replay.py" \
  "${PROJECT_ROOT}/src/methods/otc_mmot.py" \
  "${PROJECT_ROOT}/src/methods/perturb_merge.py" | sha256sum | awk '{print $1}')"

preflight() {
  local available_kb disk_kb gpu_free_mib
  mkdir -p "${OUTPUT_ROOT}"
  if [[ -f "${OUTPUT_ROOT}/code_fingerprint.sha256" ]]; then
    if [[ "$(cat "${OUTPUT_ROOT}/code_fingerprint.sha256")" != "${SOURCE_FINGERPRINT}" ]]; then
      echo "Study source has changed since the first run. Use a new output root." >&2
      exit 1
    fi
  else
    printf '%s\n' "${SOURCE_FINGERPRINT}" > "${OUTPUT_ROOT}/code_fingerprint.sha256"
  fi
  if [[ "$(sha256sum \
      "${PROJECT_ROOT}/src/training/train_cil.py" \
      "${PROJECT_ROOT}/src/data/fixed_budget_replay.py" \
      "${PROJECT_ROOT}/src/data/historical_logits.py" \
      "${PROJECT_ROOT}/src/methods/dark_replay.py" \
      "${PROJECT_ROOT}/src/methods/otc_mmot.py" \
      "${PROJECT_ROOT}/src/methods/perturb_merge.py" | sha256sum | awk '{print $1}')" != "${SOURCE_FINGERPRINT}" ]]; then
    echo "Study source changed while the driver was running." >&2
    exit 1
  fi
  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  disk_kb="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kb < MIN_RAM_GIB * 1024 * 1024 )); then
    echo "Resource guard: RAM available below ${MIN_RAM_GIB} GiB." >&2
    exit 1
  fi
  if (( disk_kb < MIN_DISK_GIB * 1024 * 1024 )); then
    echo "Resource guard: disk free below ${MIN_DISK_GIB} GiB." >&2
    exit 1
  fi
  if [[ "${DEVICE}" == cuda* ]]; then
    gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')"
    if [[ -z "${gpu_free_mib}" ]] || (( gpu_free_mib < MIN_GPU_FREE_MIB )); then
      echo "Resource guard: GPU free below ${MIN_GPU_FREE_MIB} MiB." >&2
      exit 1
    fi
  fi
}

is_complete() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] &&
    [[ -s "${run_dir}/metrics.csv" ]] &&
    [[ -s "${run_dir}/forgetting.csv" ]] &&
    [[ -s "${run_dir}/class_trajectory.csv" ]] &&
    [[ -s "${run_dir}/class_forgetting.csv" ]] &&
    [[ -s "${run_dir}/training_audit.csv" ]] &&
    [[ -s "${run_dir}/events.csv" ]] &&
    [[ "$(find "${run_dir}/checkpoints" -maxdepth 1 -name 'task_*_model.pt' | wc -l)" -eq 8 ]] &&
    rg -q 'run_completed' "${run_dir}/events.csv"
}

for seed in "${SEEDS[@]}"; do
  baseline="${BASELINE_ROOT}/p9_tail_profile_balanced_seed${seed}_replay_distill_fixed_budget_uniform_mlptddi"
  if ! is_complete "${baseline}"; then
    echo "C3 baseline is incomplete for seed ${seed}: ${baseline}" >&2
    exit 1
  fi
done

for method_id in "${METHODS[@]}"; do
  case "${method_id}" in
    C0) method=sequential; epochs=20 ;;
    C1) method=er_fixed_budget_uniform; epochs=20 ;;
    C2) method=derpp_fixed_budget_uniform; epochs=20 ;;
    C4) method=xder_fixed_budget_uniform; epochs=20 ;;
    C6) method=perturb_merge_full_network; epochs=20 ;;
    C0O) method=sequential; epochs=1 ;;
    C1O) method=er_fixed_budget_uniform; epochs=1 ;;
    C5) method=otc_mmot_online_adapted; epochs=1 ;;
    *) echo "Unsupported method ID: ${method_id}" >&2; exit 2 ;;
  esac
  for seed in "${SEEDS[@]}"; do
    preflight
    task_file="${PROJECT_ROOT}/outputs/tasks/tail_profile_balanced_seed${seed}_tasks.json"
    if [[ ! -s "${task_file}" ]]; then
      echo "Missing task file: ${task_file}" >&2
      exit 1
    fi
    run_dir="${OUTPUT_ROOT}/p9_h8_tddi_${method_id,,}_seed${seed}"
    if is_complete "${run_dir}"; then
      echo "[skip] Complete ${method_id} seed=${seed}"
      continue
    fi
    if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
      exit 1
    fi
    echo "[run] protocol=P9 horizon=H8 backbone=T-DDI method=${method_id} seed=${seed} epochs=${epochs} device=${DEVICE}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/training/train_cil.py" \
      --train "${PROJECT_ROOT}/train_extracted.parquet" \
      --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
      --test "${PROJECT_ROOT}/test_extracted.parquet" \
      --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
      --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
      --task-file "${task_file}" \
      --outdir "${run_dir}" \
      --method "${method}" --variant tddi \
      --batch-size 1024 --effective-batch-size 1024 \
      --epochs "${epochs}" --patience 5 \
      --lr 0.001 --weight-decay 0.0001 \
      --seed "${seed}" --device "${DEVICE}" \
      --total-memory-budget 6800 --replay-draws-per-epoch 6800 \
      --distill-alpha 1.0 --temperature 2.0 \
      --feature-distill-weight 0.5 --focal-gamma 1.0
  done
done

echo "[done] P9-H8 CL method study complete under ${OUTPUT_ROOT}"
