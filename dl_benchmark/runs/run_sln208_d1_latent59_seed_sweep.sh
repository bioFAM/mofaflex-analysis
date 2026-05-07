#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
cd "$ROOT"

STAMP="$(date +%F_%H%M%S)"
RUN_NAME="${RUN_NAME:-sln_208_d1_latent59_seed_sweep_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/seed_sweeps/$RUN_NAME}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT/outputs/snapshots/${RUN_NAME}_preexisting}"

DATA_D1="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1.h5ad"
DATA_D1_MOFAMU="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1_mofaflex.h5mu"
DATA_TOTALVI="/home/aqoku/projects/data/mfl_bench/sln_208_totalvi.h5ad"

PY_SCVI="/home/aqoku/miniforge3/envs/scvi_env/bin/python"
PY_MFL="/home/aqoku/miniforge3/envs/mfl_bench/bin/python"
PY_MULTIGRATE="/home/aqoku/miniforge3/envs/multigrate_env/bin/python"
PY_SCVAEIT="/home/aqoku/miniforge3/envs/scvaeit_env/bin/python"
PY_MOJITOO="/home/aqoku/miniforge3/envs/mojitoo_env/bin/python"
PY_SCOIT="/home/aqoku/miniforge3/envs/scoit_env/bin/python"
PY_CONC="/home/aqoku/miniforge3/envs/conc_env/bin/python"
PY_SCGPT="/home/aqoku/miniforge3/envs/scgpt_env/bin/python"
PY_SCFOUND="/home/aqoku/miniforge3/envs/scFound_env/bin/python"

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(0 1 2 3 4)
fi

mkdir -p "$OUT_ROOT" "$SNAPSHOT_DIR" "$ROOT/outputs/snapshots"

echo "[snapshot] Saving current live D1 outputs into $SNAPSHOT_DIR"
shopt -s nullglob
SNAPSHOT_ITEMS=(
  "$ROOT"/outputs/sln_208_d1_*
  "$ROOT"/outputs/seed_sweeps/sln_208_d1_final
)
for item in "${SNAPSHOT_ITEMS[@]}"; do
  if [[ -e "$item" ]]; then
    cp -a "$item" "$SNAPSHOT_DIR/"
  fi
done
shopt -u nullglob

cat > "$OUT_ROOT/README.txt" <<EOF
sln_208_d1 latent-59 seed sweep
run_name: $RUN_NAME
created_at: $STAMP
seeds: ${SEEDS[*]}

Requested target setup:
- mofaflex: hallmark+mca + 3 dense = 59 total
- totalvi: 59
- multigrate: 59
- scarches: 59
- scvaeit: 59
- scoit: 59
- mojitoo: 59 RNA PCs + 59 protein PCs
- concerto: native dimensionality (current wrapper has no latent-dim override)
- scgpt: native checkpoint dimensionality (current wrapper has no latent-dim override)
- scfoundation: native checkpoint dimensionality (current wrapper has no latent-dim override)

Outputs:
- one folder per seed: seed_<n>/
- per-seed metric files:
  - sln_208_d1_scib_benchmarker.csv
  - sln_208_d1_scgraph.csv
  - sln_208_d1_metric_comparison.csv
- aggregate summaries:
  - summary/
  - scgraph_summary/
EOF

for seed in "${SEEDS[@]}"; do
  SEED_DIR="$OUT_ROOT/seed_${seed}"
  mkdir -p "$SEED_DIR"

  echo
  echo "========== seed ${seed} =========="

  echo "[1/13] mofaflex (hallmark+mca + 3 dense = 59 total)"
  PYTHONPATH=. "$PY_MFL" runs/run_mofaflex.py \
    --config configs/sln_208_d1_mofaflex_informed.yaml \
    --data "$DATA_D1_MOFAMU" \
    --out "$SEED_DIR/sln_208_d1_mofaflex" \
    --n-factors 3 \
    --seed "$seed"

  echo "[2/13] totalvi (59)"
  PYTHONPATH=. "$PY_SCVI" runs/run_totalvi.py \
    --config configs/sln_208_d1_totalvi.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_totalvi" \
    --latent-dim 59 \
    --seed "$seed"

  echo "[3/13] multigrate (59)"
  PYTHONPATH=. "$PY_MULTIGRATE" runs/run_multigrate.py \
    --config configs/sln_208_d1_multigrate.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_multigrate" \
    --latent-dim 59 \
    --seed "$seed"

  echo "[4/13] scarches (59)"
  PYTHONPATH=. "$PY_MULTIGRATE" runs/run_scarches.py \
    --config configs/sln_208_d1_scarches.yaml \
    --data "$DATA_TOTALVI" \
    --out "$SEED_DIR/sln_208_d1_scarches" \
    --latent-dim 59 \
    --seed "$seed"

  echo "[5/13] scvaeit (59)"
  PYTHONPATH=. "$PY_SCVAEIT" runs/run_scvaeit.py \
    --config configs/sln_208_d1_scvaeit.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_scvaeit" \
    --dim-latent 59 \
    --seed "$seed"

  echo "[6/13] scoit (59)"
  PYTHONPATH=. "$PY_SCOIT" runs/run_scoit.py \
    --config configs/sln_208_d1_scoit.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_scoit" \
    --k1 59 \
    --k2 59 \
    --k3 59 \
    --seed "$seed"

  echo "[7/13] mojitoo (59 RNA PCs + 59 protein PCs)"
  PYTHONPATH=. "$PY_MOJITOO" runs/run_mojitoo.py \
    --config configs/sln_208_d1_mojitoo.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_mojitoo" \
    --rna-pcs 59 \
    --prot-pcs 59 \
    --seed "$seed"

  echo "[8/13] concerto (native dimensionality)"
  PYTHONPATH=. "$PY_CONC" runs/run_concerto.py \
    --config configs/sln_208_d1_concerto.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_concerto" \
    --seed "$seed"

  echo "[9/13] scgpt (native checkpoint dimensionality)"
  PYTHONPATH=. "$PY_SCGPT" runs/run_scgpt.py \
    --config configs/sln_208_d1_scgpt.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_scgpt"

  echo "[10/13] scfoundation (native checkpoint dimensionality)"
  PYTHONPATH=. "$PY_SCFOUND" runs/run_scfoundation.py \
    --config configs/sln_208_d1_scfoundation.yaml \
    --data "$DATA_D1" \
    --out "$SEED_DIR/sln_208_d1_scfoundation"

  RUN_SPECS=(
    "mofaflex=$SEED_DIR/sln_208_d1_mofaflex"
    "totalvi=$SEED_DIR/sln_208_d1_totalvi"
    "multigrate=$SEED_DIR/sln_208_d1_multigrate"
    "scarches=$SEED_DIR/sln_208_d1_scarches"
    "scvaeit=$SEED_DIR/sln_208_d1_scvaeit"
    "scoit=$SEED_DIR/sln_208_d1_scoit"
    "mojitoo=$SEED_DIR/sln_208_d1_mojitoo"
    "concerto=$SEED_DIR/sln_208_d1_concerto"
    "scgpt=$SEED_DIR/sln_208_d1_scgpt"
    "scfoundation=$SEED_DIR/sln_208_d1_scfoundation"
  )

  SCIB_ARGS=()
  SCGRAPH_ARGS=()
  UMAP_ARGS=()
  for spec in "${RUN_SPECS[@]}"; do
    out_dir="${spec#*=}"
    if [[ -f "$out_dir/latent.npy" ]]; then
      SCIB_ARGS+=(--run "$spec")
      SCGRAPH_ARGS+=(--run "$spec")
      UMAP_ARGS+=(--run "$spec")
    fi
  done

  echo "[11/13] scib"
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
    --title "sln_208_d1 latent59 seed ${seed} Bio Conservation"

  echo "[12/13] scgraph"
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
    --title "sln_208_d1 latent59 seed ${seed} scGraph Corr-Weighted"

  echo "[13/13] merge + umaps"
  PYTHONPATH=. "$PY_SCVI" runs/merge_metric_tables.py \
    --left-input "$SEED_DIR/sln_208_d1_scib_benchmarker.csv" \
    --left-metric "Bio conservation" \
    --left-alias "Bio conservation" \
    --right-input "$SEED_DIR/sln_208_d1_scgraph.csv" \
    --right-metric "Corr-Weighted" \
    --right-alias "Corr-Weighted" \
    --out-csv "$SEED_DIR/sln_208_d1_metric_comparison.csv" \
    --out-json "$SEED_DIR/sln_208_d1_metric_comparison.json"

  PYTHONPATH=. "$PY_SCVI" runs/plot_latent_umap.py \
    "${UMAP_ARGS[@]}" \
    --data "$DATA_D1" \
    --input-format h5ad \
    --label-key "cell_types" \
    --seed "$seed" \
    --out-dir "$SEED_DIR/sln_208_d1_umaps_celltypes"
done

echo
echo "[summary] Bio conservation + Corr-Weighted"
PYTHONPATH=. "$PY_SCVI" runs/summarize_seed_sweep.py \
  --root "$OUT_ROOT" \
  --input-name sln_208_d1_metric_comparison.csv \
  --metric "Bio conservation" \
  --metric "Corr-Weighted" \
  --out-dir "$OUT_ROOT/summary" \
  --title-prefix "sln_208_d1 latent59 "

echo "[summary] Rank-PCA + Corr-PCA + Corr-Weighted"
PYTHONPATH=. "$PY_SCVI" runs/summarize_seed_sweep.py \
  --root "$OUT_ROOT" \
  --input-name sln_208_d1_scgraph.csv \
  --metric "Rank-PCA" \
  --metric "Corr-PCA" \
  --metric "Corr-Weighted" \
  --out-dir "$OUT_ROOT/scgraph_summary" \
  --title-prefix "sln_208_d1 latent59 "

echo "[summary] scatter plots"
PYTHONPATH=. "$PY_SCVI" runs/plot_seed_scatter.py \
  --input "$OUT_ROOT/summary/seed_metrics_summary.csv" \
  --x-metric "Bio conservation" \
  --y-metric "Corr-Weighted" \
  --out-png "$OUT_ROOT/summary/bio_vs_corr_scatter_mean_std.png" \
  --title "sln_208_d1 latent59 across seeds"

PYTHONPATH=. "$PY_SCVI" runs/plot_seed_scatter.py \
  --input "$OUT_ROOT/scgraph_summary/seed_metrics_summary.csv" \
  --x-metric "Rank-PCA" \
  --y-metric "Corr-PCA" \
  --out-png "$OUT_ROOT/scgraph_summary/rank_pca_vs_corr_pca_scatter_mean_std.png" \
  --title "sln_208_d1 latent59 scGraph across seeds"

echo
echo "Done."
echo "Snapshot saved under: $SNAPSHOT_DIR"
echo "Seed sweep written to: $OUT_ROOT"
