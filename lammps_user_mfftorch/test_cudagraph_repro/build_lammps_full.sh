#!/usr/bin/env bash
# One-shot, resumable, NON-Kokkos LAMMPS + USER-MFFTORCH build on the 4090.
# Source priority: an existing valid /home/ylzhang/lrx/lammps-stable.tar.gz (e.g. one
# rsync'd up from the Mac) else github codeload. Prints BUILD_DONE + BUILD_OK/BUILD_FAIL.
set -u
# Single-instance lock (robust; avoids pgrep self-matching the launching ssh command).
exec 9>/home/ylzhang/lrx/.lammps_build.lock 2>/dev/null || true
flock -n 9 2>/dev/null || { echo "another LAMMPS build is already running; exiting"; exit 0; }
PY=/home/ylzhang/micromamba/envs/FSCETP/bin/python
LRX=/home/ylzhang/lrx
REPO=$LRX/FSCETP
SRC=$REPO/lammps_user_mfftorch/test_cudagraph_repro
LOG=$LRX/lammps_build.log
TARBALL=$LRX/lammps-stable.tar.gz
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }
cd "$LRX"

log "=== source tarball ==="
if [ -s "$TARBALL" ] && tar tzf "$TARBALL" >/dev/null 2>&1; then
  log "using existing tarball $(du -h "$TARBALL" | cut -f1)"
else
  rm -f "$TARBALL"
  for TAG in stable_22Jul2025 stable_29Aug2024 stable; do
    log "codeload $TAG ..."
    if curl -fL --connect-timeout 25 -o "$TARBALL" "https://codeload.github.com/lammps/lammps/tar.gz/refs/tags/$TAG" >>"$LOG" 2>&1 && tar tzf "$TARBALL" >/dev/null 2>&1; then
      log "downloaded $TAG"; break
    fi
    rm -f "$TARBALL"
  done
fi
if ! tar tzf "$TARBALL" >/dev/null 2>&1; then log "NO valid tarball"; log "BUILD_FAIL"; log "BUILD_DONE"; exit 1; fi
LDIR=$LRX/$(tar tzf "$TARBALL" 2>/dev/null | head -1 | cut -d/ -f1)
log "LDIR=$LDIR"

log "=== extract ==="
[ -d "$LDIR" ] || tar xzf "$TARBALL" >>"$LOG" 2>&1
log "extract rc=$? exists=$([ -d "$LDIR" ] && echo yes || echo no)"

log "=== install USER-MFFTORCH ==="
bash "$REPO/scripts/install_user_mfftorch_into_lammps.sh" "$LDIR" >>"$LOG" 2>&1
log "install rc=$?"

CML="$LDIR/cmake/CMakeLists.txt"
log "=== cmake structure (pre-patch) ==="
grep -n "STANDARD_PACKAGES\|PKG_WITH_INCL\|add_library(lammps" "$CML" | head -20 | tee -a "$LOG"

log "=== patch CMakeLists ==="
"$PY" "$SRC/patch_lammps_cmake.py" "$CML" 2>&1 | tee -a "$LOG"
log "post-patch grep USER-MFFTORCH:"; grep -n "USER-MFFTORCH" "$CML" | tee -a "$LOG"

log "=== cmake configure (non-Kokkos) ==="
LIBTORCH=$("$PY" -c "import torch; print(torch.utils.cmake_prefix_path)")
log "LIBTORCH=$LIBTORCH"
# torch's cmake config poisons its include dirs with MKL_INCLUDE_DIR-NOTFOUND when MKL
# headers aren't found; our code doesn't use MKL headers, so point it at any existing dir.
MKL_INC=$("$PY" -c "import sys,os; print(os.path.join(sys.prefix,'include'))")
[ -d "$MKL_INC" ] || MKL_INC=/usr/include
log "MKL_INCLUDE_DIR=$MKL_INC"
rm -rf "$LDIR/build-mfftorch"
cmake -S "$LDIR/cmake" -B "$LDIR/build-mfftorch" \
  -D PKG_USER-MFFTORCH=ON \
  -D BUILD_OMP=no \
  -D MKL_INCLUDE_DIR="$MKL_INC" \
  -D CMAKE_PREFIX_PATH="$LIBTORCH" \
  -D CMAKE_BUILD_TYPE=Release >>"$LOG" 2>&1
CFG_RC=$?
log "configure rc=$CFG_RC"
if [ "$CFG_RC" -ne 0 ]; then log "CONFIGURE FAILED (tail below)"; tail -n 40 "$LOG"; log "BUILD_FAIL"; log "BUILD_DONE"; exit 1; fi

log "=== build (-j40) ==="
cmake --build "$LDIR/build-mfftorch" -j 40 >>"$LOG" 2>&1
BLD_RC=$?
log "build rc=$BLD_RC"
if [ "$BLD_RC" -eq 0 ] && [ -x "$LDIR/build-mfftorch/lmp" ]; then
  log "lmp = $LDIR/build-mfftorch/lmp"
  "$LDIR/build-mfftorch/lmp" -h 2>/dev/null | grep -i "mff/torch" | tee -a "$LOG"
  log "BUILD_OK"
else
  log "BUILD FAILED (tail below)"; tail -n 50 "$LOG"; log "BUILD_FAIL"
fi
log "BUILD_DONE"
