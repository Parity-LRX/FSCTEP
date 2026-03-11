#!/usr/bin/env bash
set -euo pipefail

# One-shot installer for:
#   - torch 2.7.1 + cu128 (+ matching torchvision/torchaudio)
#   - torch_scatter/torch_cluster/pyg_lib wheels matching pt27cu128
#   - cuEquivariance torch + CUDA ops
#
# Recommended usage:
#   bash scripts/install_pt271_cu128.sh
#
# Notes:
# - This script is Linux CUDA focused.
# - It uses extra index URLs (-f / --index-url) that cannot be encoded in setup.py.

python -m pip install -U pip

echo "[1/4] Installing PyTorch 2.7.1 + cu128 ..."
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128

echo "[2/4] Installing PyG wheels (pt27cu128) ..."
python -m pip install pyg-lib torch-scatter torch-cluster \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

echo "[3/4] Installing cuEquivariance (torch + ops) ..."
python -m pip install cuequivariance-torch==0.8.1 cuequivariance-ops-torch-cu12==0.8.1

echo "[4/4] Done. Verifying imports ..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
import torch_scatter
print("torch_scatter: OK")
import cuequivariance_ops_torch
print("cuequivariance_ops_torch: OK")
PY

echo "All good."
