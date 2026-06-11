#!/bin/bash
# Validate the trainer root-fix: run ONE epoch of the same ictd-fix slab config with the FIXED
# trainer, save a fresh checkpoint, then probe it -> the ictd_fix_* knobs + avg_num_neighbors
# should now be PRESENT (they are ABSENT in the old 09:55 checkpoint). Detached; poll for DONE.
PFX=/home/ylzhang/micromamba/envs/FSCETP
PY=$PFX/bin/python
OUT=/home/ylzhang/lrx/savetest
NEWCKPT=$OUT/checkpoint/model_pure_cartesian_ictd_fix.pth
OLDCKPT=/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix.pth
PROBE=/home/ylzhang/lrx/FSCETP/molecular_force_field/test/_probe_ckpt_params.py
LOG=/home/ylzhang/lrx/savetest.log
export PATH=$PFX/bin:$PATH
export LD_LIBRARY_PATH=$PFX/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$OUT/checkpoint"

: > "$LOG"
setsid bash -c "
  export PATH=$PFX/bin:\$PATH
  export LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-}
  cd '$OUT'
  echo '=== one-epoch train (fixed trainer) ===' >> '$LOG'
  '$PY' -m molecular_force_field.cli.train \
    --data-dir /home/ylzhang/lrx/lr_ablation/data_slab \
    --energy-key energy --force-key forces \
    --tensor-product-mode pure-cartesian-ictd-fix \
    --ictd-fix-route fusion --ictd-fix-interaction-attn-heads 1 \
    --ictd-fix-fusion-readout-mixed-channels \
    --irreps-output-conv-channels 64 --lmax 2 --num-interaction 2 --max-radius 6.0 \
    --epochs 1 --batch-size 1 --device cuda --dtype float32 \
    --learning-rate 1e-3 --energy-weight 1.0 --force-weight 10.0 --long-range-mode none \
    --checkpoint '$NEWCKPT' >> '$LOG' 2>&1
  echo \"TRAIN_RC=\$?\" >> '$LOG'
  echo '' >> '$LOG'; echo '##### OLD ckpt (before fix) #####' >> '$LOG'
  '$PY' '$PROBE' '$OLDCKPT' >> '$LOG' 2>&1 || true
  echo '' >> '$LOG'; echo '##### NEW ckpt (fixed trainer) #####' >> '$LOG'
  '$PY' '$PROBE' '$NEWCKPT' >> '$LOG' 2>&1 || true
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
