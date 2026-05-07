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

PYTHONPATH=. "$PY_SCVI" runs/evaluate_scib_benchmarker.py \
  --data "$DATA_D1" \
  --label-key "cell_types" \
  --batch-key "" \
  --bio-only \
  --bio-no-kmeans \
  --run "mojitoo=$SEED_DIR/sln_208_d1_mojitoo" \
  --out-csv "$SEED_DIR/mojitoo_scib_tmp.csv" \
  --out-json "$SEED_DIR/mojitoo_scib_tmp.json"

PYTHONPATH=. "$PY_SCVI" runs/evaluate_scgraph.py \
  --data "$DATA_D1" \
  --label-key "cell_types" \
  --batch-key "batch" \
  --run "mojitoo=$SEED_DIR/sln_208_d1_mojitoo" \
  --out-csv "$SEED_DIR/mojitoo_scgraph_tmp.csv" \
  --out-json "$SEED_DIR/mojitoo_scgraph_tmp.json"

python - <<PY
import csv
from pathlib import Path

root = Path(r"$SEED_DIR")
replacements = {
    "sln_208_d1_scib_benchmarker.csv": root / "mojitoo_scib_tmp.csv",
    "sln_208_d1_scgraph.csv": root / "mojitoo_scgraph_tmp.csv",
}

for target_name, tmp_path in replacements.items():
    target_path = root / target_name
    with tmp_path.open() as f:
        tmp_rows = list(csv.DictReader(f))
    tmp_map = {row["Embedding"]: row for row in tmp_rows}
    with target_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    new_rows = []
    for row in rows:
        emb = row["Embedding"]
        if emb in tmp_map:
            new_rows.append(tmp_map.pop(emb))
        else:
            new_rows.append(row)
    for row in tmp_map.values():
        new_rows.append(row)

    with target_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
PY

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
