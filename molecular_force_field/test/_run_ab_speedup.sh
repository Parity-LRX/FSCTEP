#!/bin/bash
# Matched LAMMPS A/B: export today's real ckpt to a TorchScript core (no E0, so it matches
# the bare-model core_real663.pt2), then run eager / cuda-graph / aoti on the SAME frozen
# 250-atom config and report PE + timing for each. Detached; poll log for DONE.
PFX=/home/ylzhang/micromamba/envs/FSCETP
PY=$PFX/bin/python
LMP=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp
WORK=/home/ylzhang/lrx/mff_md_work
CKPT=/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix.pth
LOG=/home/ylzhang/lrx/ab_speedup.log
export PATH=$PFX/bin:$PATH
export LD_LIBRARY_PATH=$PFX/lib:${LD_LIBRARY_PATH:-}

: > "$LOG"
setsid bash -c "
  export PATH=$PFX/bin:\$PATH
  export LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-}
  cd /home/ylzhang/lrx/FSCETP
  echo '=== export TorchScript core (--no-embed-e0, bare model = matches .pt2) ===' >> '$LOG'
  '$PY' -m molecular_force_field.cli.export_libtorch_core --checkpoint '$CKPT' --elements H O --device cuda --no-embed-e0 --out '$WORK/core_real_ts.pt' >> '$LOG' 2>&1
  echo \"TS_EXPORT_RC=\$?\" >> '$LOG'
  cd '$WORK'
  echo '=== RUN eager (TorchScript) ===' >> '$LOG'
  timeout 400 '$LMP' -in '$WORK/in.run.ts' >> '$LOG' 2>&1; echo \"EAGER_RC=\$?\" >> '$LOG'
  echo '=== RUN graph (TorchScript + MFF_CUDA_GRAPH) ===' >> '$LOG'
  MFF_CUDA_GRAPH=1 timeout 400 '$LMP' -in '$WORK/in.run.ts' >> '$LOG' 2>&1; echo \"GRAPH_RC=\$?\" >> '$LOG'
  echo '=== RUN aoti (.pt2) ===' >> '$LOG'
  timeout 400 '$LMP' -in '$WORK/in.run.aoti' >> '$LOG' 2>&1; echo \"AOTI_RC=\$?\" >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
