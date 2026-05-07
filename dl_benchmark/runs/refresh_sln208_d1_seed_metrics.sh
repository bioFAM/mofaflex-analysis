#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
DATA_D1="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1.h5ad"
PY_SCVI="/home/aqoku/miniforge3/envs/scvi_env/bin/python"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <seed>" >&2
  exit 1
fi

SEED="$1"
SEED_DIR="$ROOT/outputs/seed_sweeps/sln_208_d1/seed_${SEED}"

cd "$ROOT"

RUN_SPECS=(
  "mofaflex=$SEED_DIR/sln_208_d1_mofaflex"
  "mofaflex_informed_mca=$SEED_DIR/sln_208_d1_mofaflex_informed_mca"
  "totalvi=$SEED_DIR/sln_208_d1_totalvi"
  "multigrate=$SEED_DIR/sln_208_d1_multigrate"
  "scarches=$SEED_DIR/sln_208_d1_scarches"
  "pca=$SEED_DIR/sln_208_d1_pca"
  "nmf=$SEED_DIR/sln_208_d1_nmf"
  "scgpt=$SEED_DIR/sln_208_d1_scgpt"
  "scvaeit=$SEED_DIR/sln_208_d1_scvaeit"
  "scfoundation=$SEED_DIR/sln_208_d1_scfoundation"
  "mojitoo=$SEED_DIR/sln_208_d1_mojitoo"
  "scoit=$SEED_DIR/sln_208_d1_scoit"
)

SCIB_ARGS=()
SCGRAPH_ARGS=()
for spec in "${RUN_SPECS[@]}"; do
  out_dir="${spec#*=}"
  if [[ -f "$out_dir/latent.npy" ]]; then
    SCIB_ARGS+=(--run "$spec")
    SCGRAPH_ARGS+=(--run "$spec")
  fi
done

PYTHONPATH=. "$PY_SCVI" runs/evaluate_scib_benchmarker.py \
  --data "$DATA_D1" \
  --label-key "cell_types" \
  --batch-key "" \
  --bio-only \
  --bio-no-kmeans \
  "${SCIB_ARGS[@]}" \
  --out-csv "$SEED_DIR/sln_208_d1_scib_benchmarker.csv" \
  --out-json "$SEED_DIR/sln_208_d1_scib_benchmarker.json"

PYTHONPATH=. "$PY_SCVI" runs/plot_scib_results.py \
  --input "$SEED_DIR/sln_208_d1_scib_benchmarker.csv" \
  --metric "Bio conservation" \
  --out-png "$SEED_DIR/sln_208_d1_scib_bio_conservation.png" \
  --title "sln_208_d1 seed ${SEED} Bio Conservation"

PYTHONPATH=. "$PY_SCVI" runs/evaluate_scgraph.py \
  --data "$DATA_D1" \
  --label-key "cell_types" \
  --batch-key "batch" \
  "${SCGRAPH_ARGS[@]}" \
  --out-csv "$SEED_DIR/sln_208_d1_scgraph.csv" \
  --out-json "$SEED_DIR/sln_208_d1_scgraph.json"

PYTHONPATH=. "$PY_SCVI" runs/plot_scib_results.py \
  --input "$SEED_DIR/sln_208_d1_scgraph.csv" \
  --metric "Corr-Weighted" \
  --out-png "$SEED_DIR/sln_208_d1_scgraph_corr_weighted.png" \
  --title "sln_208_d1 seed ${SEED} scGraph Corr-Weighted"

PYTHONPATH=. "$PY_SCVI" runs/merge_metric_tables.py \
  --left-input "$SEED_DIR/sln_208_d1_scib_benchmarker.csv" \
  --left-metric "Bio conservation" \
  --left-alias "Bio conservation" \
  --right-input "$SEED_DIR/sln_208_d1_scgraph.csv" \
  --right-metric "Corr-Weighted" \
  --right-alias "Corr-Weighted" \
  --out-csv "$SEED_DIR/sln_208_d1_metric_comparison.csv" \
  --out-json "$SEED_DIR/sln_208_d1_metric_comparison.json"

"$PY_SCVI" runs/summarize_seed_sweep.py \
  --root "$ROOT/outputs/seed_sweeps/sln_208_d1" \
  --input-name "sln_208_d1_metric_comparison.csv" \
  --metric "Bio conservation" \
  --metric "Corr-Weighted" \
  --out-dir "$ROOT/outputs/seed_sweeps/sln_208_d1/summary"

"$PY_SCVI" runs/merge_seed_summaries.py \
  --base-combined "$ROOT/outputs/seed_sweeps/sln_208_d1/summary/seed_metrics_combined.csv" \
  --add-combined "$ROOT/outputs/seed_sweeps/sln_208_d1_mofaflex_paper/summary/seed_metrics_combined.csv" \
  --metric "Bio conservation" \
  --metric "Corr-Weighted" \
  --out-dir "$ROOT/outputs/seed_sweeps/sln_208_d1/summary_with_paper"
