#!/bin/bash
# Validate the DEPLOYED model's LAMMPS energy against the dataset DFT energy: re-export the
# TorchScript core (all 4 elements, correct avg + E0), build a LAMMPS data file from a slab_val
# config, run pair_style mff/torch at the training cutoff 6.0, compare PE to the dataset energy.
PFX=/home/ylzhang/micromamba/envs/FSCETP
PY=$PFX/bin/python
LMP=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp
WORK=/home/ylzhang/lrx/mff_md_work
CKPT=/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix.pth
AVG=82.5075188429
IDX=${1:-0}
LOG=/home/ylzhang/lrx/energy_vs_dataset.log
export PATH=$PFX/bin:$PATH
export LD_LIBRARY_PATH=$PFX/lib:${LD_LIBRARY_PATH:-}

: > "$LOG"
setsid bash -c "
  export PATH=$PFX/bin:\$PATH
  export LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-}
  cd /home/ylzhang/lrx/FSCETP
  echo '=== re-export TS core (H O F K, avg+E0) ===' >> '$LOG'
  '$PY' -m molecular_force_field.cli.export_libtorch_core --checkpoint '$CKPT' --elements H O F K \
      --device cuda --avg-num-neighbors $AVG --out '$WORK/core_real_ts_hofk.pt' >> '$LOG' 2>&1
  echo \"TS_RC=\$?\" >> '$LOG'
  echo '=== build LAMMPS data from slab_val[$IDX] ===' >> '$LOG'
  '$PY' molecular_force_field/test/_make_lmp_data.py $IDX '$WORK/core_real_ts_hofk.pt' >> '$LOG' 2>&1
  echo \"MK_RC=\$?\" >> '$LOG'
  cd '$WORK'
  echo '=== LAMMPS run 0 (energy) ===' >> '$LOG'
  timeout 400 '$LMP' -in '$WORK/in.slabval' >> '$LOG' 2>&1
  echo \"LMP_RC=\$?\" >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
