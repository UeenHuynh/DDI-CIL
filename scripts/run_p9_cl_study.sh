#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Complete all methods for one seed before committing to the remaining seeds.
P9_CL_SEEDS=0 bash scripts/run_p9_cl_methods.sh
P9_CL_SEEDS='1 2 3 4' bash scripts/run_p9_cl_methods.sh
"${P9_CL_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}" scripts/analyze_p9_cl_methods.py
