#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
PY_MULTIGRATE="/home/aqoku/miniforge3/envs/multigrate_env/bin/python"

cd "$ROOT"
PYTHONPATH=. "$PY_MULTIGRATE" runs/run_scarches.py --config configs/sln_208_d1_scarches.yaml
