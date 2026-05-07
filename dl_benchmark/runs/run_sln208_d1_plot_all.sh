#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"

if [[ "${CONDA_DEFAULT_ENV:-}" != "scvi_env" ]]; then
  echo "Please activate scvi_env first: conda activate scvi_env" >&2
  exit 1
fi

cd "$ROOT"
PYTHONPATH=. python runs/plot_scib_results.py \
  --input outputs/sln_208_d1_scib_benchmarker.csv \
  --metric "Bio conservation" \
  --out-png outputs/sln_208_d1_scib_bio_conservation.png \
  --title "sln_208_d1 Bio Conservation"

echo "Done: wrote primary result plot to outputs/sln_208_d1_scib_bio_conservation.png"
