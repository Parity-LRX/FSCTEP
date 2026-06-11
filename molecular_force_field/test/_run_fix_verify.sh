#!/bin/bash
# Verify the avg_num_neighbors + E0 fixes end to end: re-export the .pt2 AND the TorchScript core
# with the TRUE training avg_num_neighbors (82.5075188429) and E0 embedded, then re-run the LAMMPS
# A/B at the correct cutoff 6.0. Expect: bench self-check PASS, and eager(TS) PE == aoti(.pt2) PE.
PFX=/home/ylzhang/micromamba/envs/FSCETP
PY=$PFX/bin/python
LMP=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp
WORK=/home/ylzhang/lrx/mff_md_work
CKPT=/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix.pth
AVG=82.5075188429
LOG=/home/ylzhang/lrx/fix_verify.log
export PATH=$PFX/bin:$PATH
export LD_LIBRARY_PATH=$PFX/lib:${LD_LIBRARY_PATH:-}

: > "$LOG"
setsid bash -c "
  export PATH=$PFX/bin:\$PATH
  export LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-}
  cd /home/ylzhang/lrx/FSCETP
  echo '=== [1] re-export .pt2 (avg=$AVG + E0), N=762 ===' >> '$LOG'
  '$PY' -m molecular_force_field.test.bench_aoti_export --checkpoint '$CKPT' --elements H,O \
      --atoms 762 --degree 24 --device cuda --dynamic \
      --avg-num-neighbors $AVG --embed-e0 --out '$WORK/core_real762.pt2' >> '$LOG' 2>&1
  echo \"PT2_RC=\$?\" >> '$LOG'
  echo '=== [2] re-export TorchScript core (avg=$AVG, E0 default ON) ===' >> '$LOG'
  '$PY' -m molecular_force_field.cli.export_libtorch_core --checkpoint '$CKPT' --elements H O \
      --device cuda --avg-num-neighbors $AVG --out '$WORK/core_real_ts.pt' >> '$LOG' 2>&1
  echo \"TS_RC=\$?\" >> '$LOG'
  cd '$WORK'
  echo '=== [3] eager TorchScript @cut6 ===' >> '$LOG'
  timeout 400 '$LMP' -in '$WORK/in.run.ts' >> '$LOG' 2>&1; echo \"EAGER_RC=\$?\" >> '$LOG'
  echo '=== [4] aoti .pt2 @cut6 ===' >> '$LOG'
  timeout 400 '$LMP' -in '$WORK/in.run.aoti' >> '$LOG' 2>&1; echo \"AOTI_RC=\$?\" >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
