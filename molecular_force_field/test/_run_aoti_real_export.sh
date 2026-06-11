#!/bin/bash
# Detached runner: AOTI-export today's REAL trained checkpoint to a .pt2 and verify
# numerics + equivariance on the loaded package. Launched via setsid so it survives
# SSH disconnects; writes to a log that the caller polls for DONE.
# Args: $1 = atom count to bake (default 250), $2 = output .pt2 path, $3 = log path.
set -u
ATOMS="${1:-250}"
OUT="${2:-/home/ylzhang/lrx/mff_md_work/core_real.pt2}"
LOG="${3:-/home/ylzhang/lrx/aoti_real_export.log}"
PY=/home/ylzhang/micromamba/envs/FSCETP/bin/python
CKPT=/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix.pth

cd /home/ylzhang/lrx/FSCETP || { echo "NO_REPO" > "$LOG"; exit 1; }
: > "$LOG"
echo "=== AOTI real-ckpt export start  atoms=$ATOMS out=$OUT ===" >> "$LOG"

setsid bash -c "
  export PATH=/home/ylzhang/micromamba/envs/FSCETP/bin:\$PATH
  '$PY' -m molecular_force_field.test.bench_aoti_export \
      --checkpoint '$CKPT' \
      --elements H,O \
      --atoms '$ATOMS' --degree 24 --vary-degree 18 \
      --device cuda --dynamic \
      --out '$OUT' >> '$LOG' 2>&1
  echo \"EXIT_CODE=\$?\" >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$! atoms=$ATOMS out=$OUT"
