#!/bin/bash
# Rebuild kk lmp (conda toolchain) with the ghost->local edge fix, regenerate in.slabval (with
# atom_modify map yes), run slab_val[0], and report PE vs the DFT energy -189.830.
LOG=/home/ylzhang/lrx/build_validate.log
BDIR=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk
LMP=$BDIR/lmp
WORK=/home/ylzhang/lrx/mff_md_work
PFX=/home/ylzhang/micromamba/envs/FSCETP
PY=$PFX/bin/python
export CONDA_PREFIX=$PFX PREFIX=$PFX BUILD_PREFIX=$PFX
export PATH=$PFX/bin:$PATH
for s in activate-binutils_linux-64.sh activate-gcc_linux-64.sh activate-gxx_linux-64.sh '~cuda-nvcc_activate.sh'; do
  src="$PFX/etc/conda/activate.d/$s"; [ -f "$src" ] && source "$src" || true
done
export NVCC_WRAPPER_DEFAULT_COMPILER="${CXX:-$PFX/bin/x86_64-conda-linux-gnu-g++}"
export LD_LIBRARY_PATH=$PFX/lib:${LD_LIBRARY_PATH:-}
LINKTXT=$BDIR/CMakeFiles/lmp.dir/link.txt
[ -f "$LINKTXT" ] && ! grep -q 'allow-shlib-undefined' "$LINKTXT" && sed -i 's#nvcc_wrapper #nvcc_wrapper -Wl,--allow-shlib-undefined #' "$LINKTXT"

: > "$LOG"
cd "$BDIR" || { echo NO_BUILD_DIR >> "$LOG"; echo DONE >> "$LOG"; exit 1; }
setsid bash -c "
  cmake --build '$BDIR' --target lmp -j8 >> '$LOG' 2>&1
  echo \"BUILD_RC=\$?\" >> '$LOG'
  cd /home/ylzhang/lrx/FSCETP
  '$PY' molecular_force_field/test/_make_lmp_data.py 0 '$WORK/core_real_ts_hofk.pt' >> '$LOG' 2>&1
  cd '$WORK'
  echo '=== run slab_val[0] (ghost->local fix) ===' >> '$LOG'
  LD_LIBRARY_PATH=$PFX/lib timeout 400 '$LMP' -in '$WORK/in.slabval' >> '$LOG' 2>&1
  echo \"RUN_RC=\$?\" >> '$LOG'
  grep -E 'ref_energy|PotEng|^ *0 ' '$LOG' | tail -4 >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
