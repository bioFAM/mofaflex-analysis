#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aqoku/projects/code_new/mofaflex-benchmark"
DATA="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1.h5ad"
DATA_MOFAMU="/home/aqoku/projects/data/mfl_bench/sln_208_single_batch_d1_mofaflex.h5mu"
PY_SCVI="/home/aqoku/miniforge3/envs/scvi_env/bin/python"
PY_MFL="/home/aqoku/miniforge3/envs/mfl_bench/bin/python"
PY_MULTIGRATE="/home/aqoku/miniforge3/envs/multigrate_env/bin/python"
PY_SCGPT="/home/aqoku/miniforge3/envs/scgpt_env/bin/python"
PY_SCVAEIT="/home/aqoku/miniforge3/envs/scvaeit_env/bin/python"
PY_MOJITOO="/home/aqoku/miniforge3/envs/mojitoo_env/bin/python"
PY_SCOIT="/home/aqoku/miniforge3/envs/scoit_env/bin/python"
PY_SCMM="/home/aqoku/miniforge3/envs/scmm_env/bin/python"
PY_SCMSI="/home/aqoku/miniforge3/envs/scmsi_env/bin/python"
PY_CONC="/home/aqoku/miniforge3/envs/conc_env/bin/python"

echo "[0/8] Prepare single batch D1 dataset (.h5ad)"
cd "$ROOT"
PYTHONPATH=. "$PY_SCVI" runs/prepare_sln208_single_batch.py \
  --data /home/aqoku/projects/data/mfl_bench/sln_208_totalvi.h5ad \
  --out "$DATA" \
  --batch-key batch \
  --batch-value SLN208-D1

echo "[0b/8] Prepare single batch D1 dataset (.h5mu for informed MOFAFLEX)"
PYTHONPATH=. "$PY_MFL" runs/prepare_sln208_single_batch_h5mu.py \
  --data /home/aqoku/projects/data/mfl_bench/sln_208_mofaflex.h5mu \
  --out "$DATA_MOFAMU" \
  --batch-key batch \
  --batch-value SLN208-D1

echo "[1/6] Train MOFAFLEX (20 dense factors, mfl_bench)"
PYTHONPATH=. "$PY_MFL" runs/run_mofaflex.py --config configs/sln_208_d1_mofaflex.yaml

echo "[2/6] Train MOFAFLEX informed mca (10 dense + 10 informed, mfl_bench)"
PYTHONPATH=. "$PY_MFL" runs/run_mofaflex.py --config configs/sln_208_d1_mofaflex_informed_mca.yaml

echo "[3/6] Train totalVI (scvi_env)"
PYTHONPATH=. "$PY_SCVI" runs/run_totalvi.py --config configs/sln_208_d1_totalvi.yaml

echo "[4/7] Train Multigrate (multigrate_env)"
PYTHONPATH=. "$PY_MULTIGRATE" runs/run_multigrate.py --config configs/sln_208_d1_multigrate.yaml

echo "[4b/7] Train scArches totalVI surgery D2->D1 (multigrate_env)"
PYTHONPATH=. "$PY_MULTIGRATE" runs/run_scarches.py --config configs/sln_208_d1_scarches.yaml

echo "[5/7] Train PCA + NMF baselines (scvi_env)"
PYTHONPATH=. "$PY_SCVI" runs/run_baselines.py \
  --data "$DATA" \
  --out-root outputs \
  --name-prefix sln_208_d1 \
  --latent-dim 20 \
  --nmf-max-iter 1000

if [[ -x "$PY_SCGPT" ]]; then
  echo "[6/10] Generate scGPT embeddings (scgpt_env)"
  PYTHONPATH=. "$PY_SCGPT" runs/run_scgpt.py --config configs/sln_208_d1_scgpt.yaml
else
  echo "[6/10] Skip scGPT: missing $PY_SCGPT"
fi

if [[ -x "$PY_SCVAEIT" ]]; then
  echo "[7/10] Run scVAEIT (scvaeit_env)"
  PYTHONPATH=. "$PY_SCVAEIT" runs/run_scvaeit.py --config configs/sln_208_d1_scvaeit.yaml
else
  echo "[7/10] Skip scVAEIT: missing $PY_SCVAEIT"
fi

if [[ -x "$PY_MOJITOO" ]]; then
  echo "[8/10] Run MOJITOO (mojitoo_env)"
  PYTHONPATH=. "$PY_MOJITOO" runs/run_mojitoo.py --config configs/sln_208_d1_mojitoo.yaml
else
  echo "[8/10] Skip MOJITOO: missing $PY_MOJITOO"
fi

if [[ -x "$PY_SCOIT" ]]; then
  echo "[9/10] Run SCOIT (scoit_env)"
  PYTHONPATH=. "$PY_SCOIT" runs/run_scoit.py --config configs/sln_208_d1_scoit.yaml
else
  echo "[9/10] Skip SCOIT: missing $PY_SCOIT"
fi

if [[ -x "$PY_SCMM" ]]; then
  echo "[10/10] Run scMM (scmm_env)"
  PYTHONPATH=. "$PY_SCMM" runs/run_scmm.py --config configs/sln_208_d1_scmm.yaml
else
  echo "[10/10] Skip scMM: missing $PY_SCMM"
fi

if [[ -x "$PY_SCMSI" ]]; then
  echo "[11/12] Run scMSI (scmsi_env)"
  PYTHONPATH=. "$PY_SCMSI" runs/run_scmsi.py --config configs/sln_208_d1_scmsi.yaml
else
  echo "[11/12] Skip scMSI: missing $PY_SCMSI"
fi

if [[ -x "$PY_CONC" ]]; then
  echo "[12/12] Run Concerto (conc_env)"
  PYTHONPATH=. "$PY_CONC" runs/run_concerto.py --config configs/sln_208_d1_concerto.yaml
else
  echo "[12/12] Skip Concerto: missing $PY_CONC"
fi

echo "Done: trained sln_208 D1 models (mofaflex/mofaflex_informed/totalvi/multigrate/scarches/pca/nmf/scgpt/scvaeit/mojitoo/scoit/scmm/scmsi/concerto)."
