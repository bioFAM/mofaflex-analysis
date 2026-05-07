#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
DATA_D1="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1.h5ad"
DATA_D1_MOFAMU="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1_mofaflex.h5mu"
DATA_TOTALVI="/home/aqoku/projects/data/mfl_bench/sln_208_totalvi.h5ad"
DATA_MOFAFULL="/home/aqoku/projects/data/mfl_bench/sln_208_mofaflex.h5mu"

PY_SCVI="/home/aqoku/miniforge3/envs/scvi_env/bin/python"
PY_MFL="/home/aqoku/miniforge3/envs/mfl_bench/bin/python"
PY_MULTIGRATE="/home/aqoku/miniforge3/envs/multigrate_env/bin/python"
PY_SCGPT="/home/aqoku/miniforge3/envs/scgpt_env/bin/python"
PY_SCFOUND="/home/aqoku/miniforge3/envs/scFound_env/bin/python"
PY_SCVAEIT="/home/aqoku/miniforge3/envs/scvaeit_env/bin/python"
PY_MOJITOO="/home/aqoku/miniforge3/envs/mojitoo_env/bin/python"
PY_SCOIT="/home/aqoku/miniforge3/envs/scoit_env/bin/python"

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(0 1 2 3 4)
fi

OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/seed_sweeps/sln_208_d1}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

echo "[prep] Refresh D1 inputs"
PYTHONPATH=. "$PY_SCVI" runs/prepare_sln208_single_batch.py \
  --data "$DATA_TOTALVI" \
  --out "$DATA_D1" \
  --batch-key batch \
  --batch-value SLN208-D1

PYTHONPATH=. "$PY_MFL" runs/prepare_sln208_single_batch_h5mu.py \
  --data "$DATA_MOFAFULL" \
  --out "$DATA_D1_MOFAMU" \
  --batch-key batch \
  --batch-value SLN208-D1

for seed in "${SEEDS[@]}"; do
  SEED_DIR="$OUT_ROOT/seed_${seed}"
  mkdir -p "$SEED_DIR"

  echo
  echo "========== seed ${seed} =========="

  echo "[1/14] mofaflex"
  PYTHONPATH=. "$PY_MFL" runs/run_mofaflex.py \
    --config configs/sln_208_d1_mofaflex.yaml \
    --seed "$seed" \
    --out "$SEED_DIR/sln_208_d1_mofaflex"

  echo "[2/14] mofaflex_informed_mca"
  PYTHONPATH=. "$PY_MFL" runs/run_mofaflex.py \
    --config configs/sln_208_d1_mofaflex_informed_mca.yaml \
    --seed "$seed" \
    --out "$SEED_DIR/sln_208_d1_mofaflex_informed_mca"

  echo "[3/14] totalvi"
  PYTHONPATH=. "$PY_SCVI" runs/run_totalvi.py \
    --config configs/sln_208_d1_totalvi.yaml \
    --seed "$seed" \
    --out "$SEED_DIR/sln_208_d1_totalvi"

  echo "[4/14] multigrate"
  PYTHONPATH=. "$PY_MULTIGRATE" runs/run_multigrate.py \
    --config configs/sln_208_d1_multigrate.yaml \
    --seed "$seed" \
    --out "$SEED_DIR/sln_208_d1_multigrate"

  echo "[5/14] scarches"
  PYTHONPATH=. "$PY_MULTIGRATE" runs/run_scarches.py \
    --config configs/sln_208_d1_scarches.yaml \
    --seed "$seed" \
    --out "$SEED_DIR/sln_208_d1_scarches"

  echo "[6/14] pca + nmf"
  PYTHONPATH=. "$PY_SCVI" runs/run_baselines.py \
    --data "$DATA_D1" \
    --out-root "$SEED_DIR" \
    --name-prefix sln_208_d1 \
    --latent-dim 20 \
    --seed "$seed" \
    --nmf-max-iter 1000

  if [[ -x "$PY_SCGPT" ]]; then
    echo "[7/14] scgpt"
    PYTHONPATH=. "$PY_SCGPT" runs/run_scgpt.py \
      --config configs/sln_208_d1_scgpt.yaml \
      --out "$SEED_DIR/sln_208_d1_scgpt"
  else
    echo "[7/14] skip scgpt: missing $PY_SCGPT"
  fi

  if [[ -x "$PY_SCFOUND" ]]; then
    echo "[8/14] scfoundation"
    PYTHONPATH=. "$PY_SCFOUND" runs/run_scfoundation.py \
      --config configs/sln_208_d1_scfoundation.yaml \
      --out "$SEED_DIR/sln_208_d1_scfoundation"
  else
    echo "[8/14] skip scfoundation: missing $PY_SCFOUND"
  fi

  if [[ -x "$PY_SCVAEIT" ]]; then
    echo "[9/14] scvaeit"
    PYTHONPATH=. "$PY_SCVAEIT" runs/run_scvaeit.py \
      --config configs/sln_208_d1_scvaeit.yaml \
      --seed "$seed" \
      --out "$SEED_DIR/sln_208_d1_scvaeit"
  else
    echo "[9/14] skip scvaeit: missing $PY_SCVAEIT"
  fi

  if [[ -x "$PY_MOJITOO" ]]; then
    echo "[10/14] mojitoo"
    PYTHONPATH=. "$PY_MOJITOO" runs/run_mojitoo.py \
      --config configs/sln_208_d1_mojitoo.yaml \
      --seed "$seed" \
      --out "$SEED_DIR/sln_208_d1_mojitoo"
  else
    echo "[10/14] skip mojitoo: missing $PY_MOJITOO"
  fi

  if [[ -x "$PY_SCOIT" ]]; then
    echo "[11/14] scoit"
    PYTHONPATH=. "$PY_SCOIT" runs/run_scoit.py \
      --config configs/sln_208_d1_scoit.yaml \
      --seed "$seed" \
      --out "$SEED_DIR/sln_208_d1_scoit"
  else
    echo "[11/14] skip scoit: missing $PY_SCOIT"
  fi

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
  UMAP_ARGS=()
  for spec in "${RUN_SPECS[@]}"; do
    out_dir="${spec#*=}"
    if [[ -f "$out_dir/latent.npy" ]]; then
      SCIB_ARGS+=(--run "$spec")
      SCGRAPH_ARGS+=(--run "$spec")
      UMAP_ARGS+=(--run "$spec")
    fi
  done

  echo "[12/14] scib"
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
    --title "sln_208_d1 seed ${seed} Bio Conservation"

  echo "[13/14] scgraph"
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
    --title "sln_208_d1 seed ${seed} scGraph Corr-Weighted"

  echo "[14/14] merge + umaps"
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

  cat > "$SEED_DIR/README.txt" <<EOF
sln_208_d1 seed sweep result
seed: $seed
models:
- mofaflex (20 dense)
- mofaflex_informed_mca (10 informed + 10 dense)
- totalvi (20)
- multigrate (20)
- scarches (20)
- pca (20)
- nmf (20)
- scgpt (native checkpoint embedding)
- scfoundation (native checkpoint embedding)
- scvaeit (20)
- mojitoo (method-native dimensionality)
- scoit (20)
EOF
done

echo
echo "Done. Seed sweep outputs are under: $OUT_ROOT"
