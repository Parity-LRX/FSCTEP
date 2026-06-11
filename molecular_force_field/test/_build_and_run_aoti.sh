#!/bin/bash
# Incremental rebuild of the kk lmp (only mff_torch_engine.cpp changed) then run the AOTI
# .pt2 MD test. Two toolchain hazards, both handled:
#  1) the link needs conda's libstdc++ (CXXABI_1.3.15, pulled by libnccl) + ABI-consistent
#     objects -> use the conda host g++ (NVCC_WRAPPER_DEFAULT_COMPILER) for kokkos nvcc_wrapper;
#  2) the conda (cross-configured) linker won't defer SYSTEM OpenMPI's transitive opal_* symbols
#     to runtime -> add -Wl,--allow-shlib-undefined (what conda's own LDFLAGS do) so they resolve
#     at runtime via link.txt's existing -rpath (openmpi/lib + torch/lib).
# NO `set -u`: conda activate.d scripts reference unset vars (NVCC_PREPEND_FLAGS).
# NB: don't reuse var names BUILD/HOST/PREFIX -- conda activate.d exports them. Use BDIR.
LOG=/home/ylzhang/lrx/aoti_build_run.log
BDIR=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk
LMP=$BDIR/lmp
PFX=/home/ylzhang/micromamba/envs/FSCETP

# --- replicate the conda compiler env (activate.d sets CXX/sysroot) ---
export CONDA_PREFIX=$PFX
export PREFIX=$PFX
export BUILD_PREFIX=$PFX
export PATH=$PFX/bin:$PATH
for s in activate-binutils_linux-64.sh activate-gcc_linux-64.sh activate-gxx_linux-64.sh '~cuda-nvcc_activate.sh'; do
  src="$PFX/etc/conda/activate.d/$s"
  [ -f "$src" ] && source "$src" || true
done
export NVCC_WRAPPER_DEFAULT_COMPILER="${CXX:-$PFX/bin/x86_64-conda-linux-gnu-g++}"
export LD_LIBRARY_PATH=$PFX/lib:${LD_LIBRARY_PATH:-}

: > "$LOG"
{
  echo "=== toolchain ==="
  echo "NVCC_WRAPPER_DEFAULT_COMPILER=$NVCC_WRAPPER_DEFAULT_COMPILER"
  echo "CONDA_BUILD_SYSROOT=${CONDA_BUILD_SYSROOT:-UNSET}"
} >> "$LOG"

# --- inject --allow-shlib-undefined into the cached link command (idempotent) ---
LINKTXT=$BDIR/CMakeFiles/lmp.dir/link.txt
if [ -f "$LINKTXT" ] && ! grep -q 'allow-shlib-undefined' "$LINKTXT"; then
  sed -i 's#nvcc_wrapper #nvcc_wrapper -Wl,--allow-shlib-undefined #' "$LINKTXT"
  echo "patched link.txt with --allow-shlib-undefined" >> "$LOG"
fi
echo "=== build start ===" >> "$LOG"

cd "$BDIR" || { echo NO_BUILD_DIR >> "$LOG"; echo DONE >> "$LOG"; exit 1; }

setsid bash -c "
  cmake --build '$BDIR' --target lmp -j8 >> '$LOG' 2>&1
  BRC=\$?
  echo \"BUILD_RC=\$BRC\" >> '$LOG'
  if [ \$BRC -eq 0 ]; then
    echo '=== run aoti (in.run.aoti) ===' >> '$LOG'
    cd /home/ylzhang/lrx/mff_md_work
    LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-} timeout 400 '$LMP' -in /home/ylzhang/lrx/mff_md_work/in.run.aoti >> '$LOG' 2>&1
    echo \"RUN_RC=\$?\" >> '$LOG'
  fi
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
