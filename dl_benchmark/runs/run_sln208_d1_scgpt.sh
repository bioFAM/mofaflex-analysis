#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
PY_SCGPT="/home/aqoku/miniforge3/envs/scgpt_env/bin/python"

cd "$ROOT"
PYTHONPATH=. "$PY_SCGPT" runs/run_scgpt.py --config configs/sln_208_d1_scgpt.yaml
