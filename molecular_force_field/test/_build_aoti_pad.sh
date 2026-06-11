#!/bin/bash
# Rebuild kk lmp with the AOTI padding + dual-core fallback, then test the FALLBACK path: the dense
# slab's refined-A ntotal (~8228) far exceeds core_real762.pt2's N_max=762, so loading the .pt2 with
# a fallback meta must dispatch to the N-flexible TorchScript core -> -189.856 (no crash).
LOG=/home/ylzhang/lrx/build_aoti_pad.log
BDIR=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk
LMP=$BDIR/lmp
WORK=/home/ylzhang/lrx/mff_md_work
SRC=/home/ylzhang/lrx/lammps-stable_22Jul2025/src/USER-MFFTORCH
PFX=/home/ylzhang/micromamba/envs/FSCETP
PY=$PFX/bin/python
export CONDA_PREFIX=$PFX PREFIX=$PFX BUILD_PREFIX=$PFX
export PATH=$PFX/bin:$PATH
for s in activate-binutils_linux-64.sh activate-gcc_linux-64.sh activate-gxx_linux-64.sh '~cuda-nvcc_activate.sh'; do
  src="$PFX/etc/conda/activate.d/$s"; [ -f "$src" ] && source "$src" || true
done
export NVCC_WRAPPER_DEFAULT_COMPILER="${CXX:-$PFX/bin/x86_64-conda-linux-gnu-g++}"
# back-date headers so the CONFIGURE_DEPENDS glob does NOT reconfigure...
touch -d '2020-01-01 00:00:00' "$SRC/mff_torch_engine.h" "$SRC/pair_mff_torch.h"
# ...but mff_torch_engine.h ADDED MEMBERS (changed the class layout), so EVERY TU that uses the
# engine must recompile or it accesses members at stale offsets -> segfault. The back-dated header
# won't trigger that, so force it: delete the USER-MFFTORCH objects so make rebuilds them all with
# the new layout (consistent ABI), still without a reconfigure.
rm -f "$BDIR"/CMakeFiles/lammps.dir/home/ylzhang/lrx/lammps-stable_22Jul2025/src/USER-MFFTORCH/*.o
LINKTXT=$BDIR/CMakeFiles/lmp.dir/link.txt
[ -f "$LINKTXT" ] && ! grep -q 'allow-shlib-undefined' "$LINKTXT" && sed -i 's#nvcc_wrapper #nvcc_wrapper -Wl,--allow-shlib-undefined #' "$LINKTXT"

# sidecar meta for the existing N=762 .pt2 -> fallback to the H/O/F/K TS core
printf 'nmax 762\npad_z 1\nfallback %s/core_real_ts_hofk.pt\n' "$WORK" > "$WORK/core_real762.pt2.meta"

: > "$LOG"
cd "$BDIR" || { echo NO_BUILD_DIR >> "$LOG"; echo DONE >> "$LOG"; exit 1; }
setsid bash -c "
  export LD_LIBRARY_PATH=$PFX/lib:\${LD_LIBRARY_PATH:-}
  cmake --build '$BDIR' --target lmp -j8 >> '$LOG' 2>&1
  echo \"BUILD_RC=\$?\" >> '$LOG'
  cd /home/ylzhang/lrx/FSCETP
  '$PY' molecular_force_field/test/_make_lmp_data.py 0 '$WORK/core_real762.pt2' >> '$LOG' 2>&1
  cd '$WORK'
  echo '=== fallback test: .pt2 (N_max 762) on dense slab (ntotal ~8228) -> TS fallback ===' >> '$LOG'
  LD_LIBRARY_PATH=$PFX/lib timeout 400 '$LMP' -in '$WORK/in.slabval' >> '$LOG' 2>&1; echo \"RUN_RC=\$?\" >> '$LOG'
  echo DONE >> '$LOG'
" </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid=$!"
