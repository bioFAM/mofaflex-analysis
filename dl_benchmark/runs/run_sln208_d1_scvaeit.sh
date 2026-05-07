#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
PY_SCVAEIT="/home/aqoku/miniforge3/envs/scvaeit_env/bin/python"

cd "$ROOT"
PYTHONPATH=. "$PY_SCVAEIT" runs/run_scvaeit.py --config configs/sln_208_d1_scvaeit.yaml
