#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
PY_SCFOUNDATION="/home/aqoku/miniforge3/envs/scfoundation_env/bin/python"

cd "$ROOT"
PYTHONPATH=. "$PY_SCFOUNDATION" runs/run_scfoundation.py --config configs/sln_208_d1_scfoundation.yaml
