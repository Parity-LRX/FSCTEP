#!/bin/bash
# Re-run the matched A/B at the CORRECT cutoff 6.0 (= model training cutoff). eager uses the
# N-flexible TorchScript core (core_real_ts.pt); aoti uses the N=762 package (ntotal at cut 6.0).
PFX=/home/ylzhang/micromamba/envs/FSCETP
LMP=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp
WORK=/home/ylzhang/lrx/mff_md_work
LOG=/home/ylzhang/lrx/ab_cut6.log
export PATH=$PFX/bin:$PATH
export LD_LIBRARY_PATH=$PFX/lib:${LD_LIBRARY_PATH:-}

: > "$LOG"
setsid bash -c "
  export LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-}
  cd '$WORK'
  echo '=== eager TorchScript @cut6.0 ===' >> '$LOG'
  timeout 400 '$LMP' -in '$WORK/in.run.ts' >> '$LOG' 2>&1; echo \"EAGER_RC=\$?\" >> '$LOG'
  echo '=== aoti .pt2 (N=762) @cut6.0 ===' >> '$LOG'
  timeout 400 '$LMP' -in '$WORK/in.run.aoti' >> '$LOG' 2>&1; echo \"AOTI_RC=\$?\" >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
