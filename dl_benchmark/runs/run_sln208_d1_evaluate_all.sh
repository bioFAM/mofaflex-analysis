#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "scvi_env" ]]; then
  echo "Please activate scvi_env first: conda activate scvi_env" >&2
  exit 1
fi

exec bash /home/aqoku/projects/code_new/mofaflex-benchmark/runs/run_sln208_d1_scib_benchmarker.sh
