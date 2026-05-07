#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-scgpt_env}"
MODEL_DIR="${2:-/home/aqoku/models/scgpt}"
MODEL_URL="https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing"

echo "[1/5] Recreate env: $ENV_NAME"
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda create -n "$ENV_NAME" python=3.9 -y

echo "[2/5] Install PyTorch 1.13.1 + CUDA 11.7 wheel"
conda run -n "$ENV_NAME" pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1

echo "[3/5] Install scGPT-compatible Python packages"
conda run -n "$ENV_NAME" pip install \
  packaging \
  ninja \
  gdown \
  scgpt==0.2.4 --no-deps

conda run -n "$ENV_NAME" pip install \
  torchtext==0.14.1 \
  scvi-tools==0.20.3 \
  pandas==1.5.3 \
  scanpy==1.9.5 \
  leidenalg \
  numba==0.58.1 \
  scib==1.1.4 \
  scikit-misc \
  umap-learn \
  typing-extensions \
  "datasets>=2.3.0,<3.0.0" \
  "orbax<0.1.8" \
  "cell-gears<0.0.3"

echo "[4/5] Pin AnnData for scvi-tools/mudata compatibility"
conda run -n "$ENV_NAME" pip install anndata==0.9.2

echo "[5/5] Download continual-pretrained scGPT checkpoint"
mkdir -p "$MODEL_DIR"
conda run -n "$ENV_NAME" gdown --folder "$MODEL_URL" -O "$MODEL_DIR"

echo
echo "Sanity check"
conda run -n "$ENV_NAME" python -c "import scgpt, torchtext, scanpy, scvi; print('all good')"
echo "Done: env=$ENV_NAME model_dir=$MODEL_DIR"
