#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"

if [[ "${CONDA_DEFAULT_ENV:-}" != "scvi_env" ]]; then
  echo "Please activate scvi_env first: conda activate scvi_env" >&2
  exit 1
fi

cd "$ROOT"

bash runs/run_sln208_d1_scib_benchmarker.sh
bash runs/run_sln208_d1_scgraph.sh

PYTHONPATH=. python runs/merge_metric_tables.py \
  --left-input outputs/sln_208_d1_scib_benchmarker.csv \
  --left-metric "Bio conservation" \
  --left-alias "Bio conservation" \
  --right-input outputs/sln_208_d1_scgraph.csv \
  --right-metric "Corr-Weighted" \
  --right-alias "Corr-Weighted" \
  --out-csv outputs/sln_208_d1_metric_comparison.csv \
  --out-json outputs/sln_208_d1_metric_comparison.json

echo "Done: wrote D1 scIB, scGraph, and merged metric-comparison outputs under outputs/"
