#!/usr/bin/env bash
# Kokkos (CUDA, Ada/sm_89) LAMMPS + USER-MFFTORCH build. Reuses the already
# extracted+patched LAMMPS tree from build_lammps_full.sh; only re-configures into a
# SEPARATE build dir (build-mfftorch-kk) so the non-Kokkos lmp is untouched.
# Prints BUILD_KK_DONE + BUILD_KK_OK/BUILD_KK_FAIL.
set -u
exec 9>/home/ylzhang/lrx/.lammps_kk_build.lock 2>/dev/null || true
flock -n 9 2>/dev/null || { echo "another KK build is already running; exiting"; exit 0; }
PY=/home/ylzhang/micromamba/envs/FSCETP/bin/python
LRX=/home/ylzhang/lrx
LOG=$LRX/lammps_kk_build.log
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

# Kokkos for Ada (sm_89) needs CUDA >= 11.8; the system /usr/bin/nvcc is 11.5 (too old
# and incompatible with gcc 12). Use the CUDA 12.4 toolkit that's installed.
for CUDADIR in /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
  if [ -x "$CUDADIR/bin/nvcc" ]; then
    export PATH="$CUDADIR/bin:$PATH"
    export CUDAToolkit_ROOT="$CUDADIR"
    export CUDACXX="$CUDADIR/bin/nvcc"
    log "using CUDA toolkit: $CUDADIR ($("$CUDADIR/bin/nvcc" --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1))"
    break
  fi
done

LDIR=$(ls -d "$LRX"/lammps-*/ 2>/dev/null | grep -v 'tar.gz' | head -1); LDIR=${LDIR%/}
log "LDIR=$LDIR"
if [ -z "$LDIR" ] || [ ! -d "$LDIR/cmake" ]; then log "no extracted LAMMPS tree; run build_lammps_full.sh first"; log "BUILD_KK_FAIL"; log "BUILD_KK_DONE"; exit 1; fi
if ! grep -q "USER-MFFTORCH" "$LDIR/cmake/CMakeLists.txt"; then log "CMakeLists not patched; run build_lammps_full.sh first"; log "BUILD_KK_FAIL"; log "BUILD_KK_DONE"; exit 1; fi

log "=== nvcc / CUDA toolkit ==="
which nvcc >>"$LOG" 2>&1; nvcc --version >>"$LOG" 2>&1; log "nvcc rc=$?"

LIBTORCH=$("$PY" -c "import torch; print(torch.utils.cmake_prefix_path)")
log "LIBTORCH=$LIBTORCH"
MKL_INC=$("$PY" -c "import sys,os; print(os.path.join(sys.prefix,'include'))")
[ -d "$MKL_INC" ] || MKL_INC=/usr/include
log "MKL_INCLUDE_DIR=$MKL_INC"

# Kokkos+CUDA puts nvcc-only flags (-Xcudafe) into CMAKE_CXX_FLAGS, so the CXX compiler
# MUST be Kokkos's nvcc_wrapper (otherwise torch's detect_cuda_version test compiles with
# plain g++ and chokes on -Xcudafe). nvcc_wrapper forwards device flags to nvcc (on PATH).
NVCCW="$LDIR/lib/kokkos/bin/nvcc_wrapper"
[ -x "$NVCCW" ] || NVCCW=$(find "$LDIR/lib" -name nvcc_wrapper -type f 2>/dev/null | head -1)
log "nvcc_wrapper=$NVCCW"
if [ -z "$NVCCW" ] || [ ! -e "$NVCCW" ]; then log "nvcc_wrapper not found under $LDIR/lib"; log "BUILD_KK_FAIL"; log "BUILD_KK_DONE"; exit 1; fi
chmod +x "$NVCCW" 2>/dev/null || true

# 4090 = Ada Lovelace sm_89 -> Kokkos_ARCH_ADA89. Fallback to AMPERE86 (PTX-JITs to
# sm_89) if the bundled Kokkos doesn't know ADA89.
for ARCH in ADA89 AMPERE86; do
  log "=== cmake configure (Kokkos CUDA, $ARCH) ==="
  rm -rf "$LDIR/build-mfftorch-kk"
  cmake -S "$LDIR/cmake" -B "$LDIR/build-mfftorch-kk" \
    -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_${ARCH}=ON \
    -D CMAKE_CXX_COMPILER="$NVCCW" \
    -D PKG_USER-MFFTORCH=ON -D MFF_ENABLE_VIRIAL=ON \
    -D BUILD_OMP=no \
    -D MKL_INCLUDE_DIR="$MKL_INC" \
    -D CMAKE_PREFIX_PATH="$LIBTORCH" -D CMAKE_BUILD_TYPE=Release >>"$LOG" 2>&1
  CFG_RC=$?
  log "configure($ARCH) rc=$CFG_RC"
  [ "$CFG_RC" -eq 0 ] && { log "configured with $ARCH"; break; }
  log "configure with $ARCH failed; trying next arch"
done
if [ "$CFG_RC" -ne 0 ]; then log "ALL CONFIGURE FAILED (tail)"; tail -n 40 "$LOG"; log "BUILD_KK_FAIL"; log "BUILD_KK_DONE"; exit 1; fi

log "=== build (-j40, Kokkos CUDA — slow) ==="
cmake --build "$LDIR/build-mfftorch-kk" -j 40 >>"$LOG" 2>&1
BLD_RC=$?
log "build rc=$BLD_RC"
if [ "$BLD_RC" -eq 0 ] && [ -x "$LDIR/build-mfftorch-kk/lmp" ]; then
  log "lmp(kk) = $LDIR/build-mfftorch-kk/lmp"
  log "BUILD_KK_OK"
else
  log "KK BUILD FAILED (tail)"; tail -n 60 "$LOG"; log "BUILD_KK_FAIL"
fi
log "BUILD_KK_DONE"
