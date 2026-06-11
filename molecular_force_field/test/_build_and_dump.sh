#!/bin/bash
# Rebuild kk lmp (conda toolchain) then run slab_val[0] with MFF_DUMP_GRAPH=1 to dump the exact
# engine graph to /tmp/mff_graph_*.pt for Python replay.
LOG=/home/ylzhang/lrx/build_dump.log
BDIR=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk
LMP=$BDIR/lmp
WORK=/home/ylzhang/lrx/mff_md_work
PFX=/home/ylzhang/micromamba/envs/FSCETP
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
  cd '$WORK'
  rm -f /tmp/mff_graph_*.pt /tmp/mff_graph_meta.txt
  echo '=== run slab_val[0] with MFF_DUMP_GRAPH ===' >> '$LOG'
  MFF_DUMP_GRAPH=1 LD_LIBRARY_PATH=$PFX/lib timeout 400 '$LMP' -in '$WORK/in.slabval' >> '$LOG' 2>&1
  echo \"RUN_RC=\$?\" >> '$LOG'
  grep -E 'MFF_DUMP_GRAPH|PotEng|^ *0 ' '$LOG' | tail -5 >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
