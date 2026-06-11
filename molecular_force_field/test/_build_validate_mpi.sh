#!/bin/bash
# Rebuild kk lmp (conda toolchain) with option A (ghost-center edges + extended halo), then validate
# slab_val[0] energy on np1 (serial) AND np2 (system OpenMPI domain decomposition). Expect both to
# match DFT -189.830 (np2 was -182.85 with the single-subdomain fold).
LOG=/home/ylzhang/lrx/build_validate_mpi.log
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
LINKTXT=$BDIR/CMakeFiles/lmp.dir/link.txt
[ -f "$LINKTXT" ] && ! grep -q 'allow-shlib-undefined' "$LINKTXT" && sed -i 's#nvcc_wrapper #nvcc_wrapper -Wl,--allow-shlib-undefined #' "$LINKTXT"

: > "$LOG"
cd "$BDIR" || { echo NO_BUILD_DIR >> "$LOG"; echo DONE >> "$LOG"; exit 1; }
setsid bash -c "
  export LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-}
  # The .h change forces a cmake reconfigure that re-runs find_package(MPI). Two-step build:
  #  (1) reconfigure with the SYSTEM host g++ (NVCC_WRAPPER_DEFAULT_COMPILER unset) so the system-MPI
  #      compile test passes -> reconfigure completes and clears the pending state;
  #  (2) compile/link with conda g++ (libtorch needs conda libstdc++ CXXABI) + --allow-shlib-undefined.
  # Different toolchain between the two steps is fine: cmake's reconfigure check is file-mtime based,
  # not env-based, so step 2 won't reconfigure again. Drop any stale *_WORKS lines I added earlier.
  sed -i '/^MPI_CXX_WORKS:/d; /^MPI_C_WORKS:/d; /^MPI_CXX_HAVE_/d' '$BDIR/CMakeCache.txt' 2>/dev/null
  ( unset NVCC_WRAPPER_DEFAULT_COMPILER; cmake '$BDIR' >> '$LOG' 2>&1 ); echo \"RECONFIG_RC=\$?\" >> '$LOG'
  cmake --build '$BDIR' --target lmp -j8 >> '$LOG' 2>&1
  echo \"BUILD_RC=\$?\" >> '$LOG'
  cd /home/ylzhang/lrx/FSCETP
  '$PY' molecular_force_field/test/_make_lmp_data.py 0 '$WORK/core_real_ts_hofk.pt' >> '$LOG' 2>&1
  cd '$WORK'
  echo '=== np1 (serial) ===' >> '$LOG'
  LD_LIBRARY_PATH=$PFX/lib timeout 400 '$LMP' -in '$WORK/in.slabval' >> '$LOG' 2>&1; echo \"NP1_RC=\$?\" >> '$LOG'
  echo '=== np2 (system OpenMPI domain decomp) ===' >> '$LOG'
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$PFX/lib
  timeout 400 /usr/bin/mpirun -np 2 '$LMP' -in '$WORK/in.slabval' >> '$LOG' 2>&1; echo \"NP2_RC=\$?\" >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
