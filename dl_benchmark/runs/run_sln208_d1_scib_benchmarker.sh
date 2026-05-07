#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
DATA="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1.h5ad"

if [[ "${CONDA_DEFAULT_ENV:-}" != "scvi_env" ]]; then
  echo "Please activate scvi_env first: conda activate scvi_env" >&2
  exit 1
fi

cd "$ROOT"

RUN_SPECS=(
  "mofaflex=outputs/sln_208_d1_mofaflex"
  "mofaflex_informed_mca=outputs/sln_208_d1_mofaflex_informed_mca"
  "totalvi=outputs/sln_208_d1_totalvi"
  "multigrate=outputs/sln_208_d1_multigrate"
  "scarches=outputs/sln_208_d1_scarches"
  "pca=outputs/sln_208_d1_pca"
  "nmf=outputs/sln_208_d1_nmf"
  "scgpt=outputs/sln_208_d1_scgpt"
  "scvaeit=outputs/sln_208_d1_scvaeit"
  "scfoundation=outputs/sln_208_d1_scfoundation"
  "mojitoo=outputs/sln_208_d1_mojitoo"
  "scoit=outputs/sln_208_d1_scoit"
  "scmm=outputs/sln_208_d1_scmm"
  "scmsi=outputs/sln_208_d1_scmsi"
  "concerto=outputs/sln_208_d1_concerto"
)

BENCH_ARGS=()
for spec in "${RUN_SPECS[@]}"; do
  name="${spec%%=*}"
  out_dir="${spec#*=}"
  if [[ -f "$out_dir/latent.npy" ]]; then
    BENCH_ARGS+=(--run "$spec")
  else
    echo "Skipping $name: missing $out_dir/latent.npy"
  fi
done

PYTHONPATH=. python runs/evaluate_scib_benchmarker.py \
  --data "$DATA" \
  --label-key "cell_types" \
  --batch-key "" \
  --bio-only \
  --bio-no-kmeans \
  "${BENCH_ARGS[@]}" \
  --out-csv outputs/sln_208_d1_scib_benchmarker.csv \
  --out-json outputs/sln_208_d1_scib_benchmarker.json

PYTHONPATH=. python runs/plot_scib_results.py \
  --input outputs/sln_208_d1_scib_benchmarker.csv \
  --metric "Bio conservation" \
  --out-png outputs/sln_208_d1_scib_bio_conservation.png \
  --title "sln_208_d1 Bio Conservation"

echo "Done: wrote outputs/sln_208_d1_scib_benchmarker.csv/.json and outputs/sln_208_d1_scib_bio_conservation.png"
