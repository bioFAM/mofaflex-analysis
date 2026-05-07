#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
DATA_D1="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1.h5ad"
PY_MFL="/home/aqoku/miniforge3/envs/mfl_bench/bin/python"
PY_SCVI="/home/aqoku/miniforge3/envs/scvi_env/bin/python"

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(0 1 2 3 4)
fi

OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/seed_sweeps/sln_208_d1_mofaflex_paper}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$ROOT/outputs/seed_sweeps/sln_208_d1/summary_with_paper}"

cd "$ROOT"
rm -rf "$OUT_ROOT"
rm -rf "$SUMMARY_ROOT"
mkdir -p "$OUT_ROOT"

for seed in "${SEEDS[@]}"; do
  SEED_DIR="$OUT_ROOT/seed_${seed}"
  MODEL_DIR="$SEED_DIR/sln_208_d1_mofaflex_paper"
  METRIC_DIR="$SEED_DIR/metrics"
  mkdir -p "$METRIC_DIR"

  echo
  echo "========== mofaflex (paper) seed ${seed} =========="

  PYTHONPATH=. "$PY_MFL" runs/run_mofaflex.py \
    --config configs/sln_208_d1_mofaflex_informed.yaml \
    --n-factors 3 \
    --seed "$seed" \
    --out "$MODEL_DIR"

  PYTHONPATH=. "$PY_SCVI" runs/evaluate_scib_benchmarker.py \
    --data "$DATA_D1" \
    --label-key "cell_types" \
    --batch-key "" \
    --bio-only \
    --bio-no-kmeans \
    --run "mofaflex (paper)=$MODEL_DIR" \
    --out-csv "$SEED_DIR/sln_208_d1_metric_comparison.csv" \
    --out-json "$METRIC_DIR/scib.json"

  "$PY_SCVI" - <<PY
import pandas as pd
path = "$SEED_DIR/sln_208_d1_metric_comparison.csv"
df = pd.read_csv(path)
df = df[["Embedding", "Bio conservation"]]
df.to_csv(path, index=False)
PY

  PYTHONPATH=. "$PY_SCVI" runs/evaluate_scgraph.py \
    --data "$DATA_D1" \
    --label-key "cell_types" \
    --batch-key "batch" \
    --run "mofaflex (paper)=$MODEL_DIR" \
    --out-csv "$METRIC_DIR/scgraph.csv" \
    --out-json "$METRIC_DIR/scgraph.json"

  PYTHONPATH=. "$PY_SCVI" runs/merge_metric_tables.py \
    --left-input "$SEED_DIR/sln_208_d1_metric_comparison.csv" \
    --left-metric "Bio conservation" \
    --left-alias "Bio conservation" \
    --right-input "$METRIC_DIR/scgraph.csv" \
    --right-metric "Corr-Weighted" \
    --right-alias "Corr-Weighted" \
    --out-csv "$SEED_DIR/sln_208_d1_metric_comparison.csv" \
    --out-json "$SEED_DIR/sln_208_d1_metric_comparison.json"
done

PYTHONPATH=. "$PY_SCVI" runs/summarize_seed_sweep.py \
  --root "$OUT_ROOT" \
  --metric "Bio conservation" \
  --metric "Corr-Weighted" \
  --out-dir "$OUT_ROOT/summary" \
  --title-prefix "sln_208_d1 mofaflex (paper) "

PYTHONPATH=. "$PY_SCVI" runs/merge_seed_summaries.py \
  --base-combined "$ROOT/outputs/seed_sweeps/sln_208_d1/summary/seed_metrics_combined.csv" \
  --add-combined "$OUT_ROOT/summary/seed_metrics_combined.csv" \
  --metric "Bio conservation" \
  --metric "Corr-Weighted" \
  --out-dir "$SUMMARY_ROOT" \
  --title-prefix "sln_208_d1 seed sweep "

echo
echo "Done."
echo "Paper-only sweep: $OUT_ROOT"
echo "Merged summary:   $SUMMARY_ROOT"
