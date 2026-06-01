#!/usr/bin/env bash
# Stage 1: fetch LAMMPS, extract, install USER-MFFTORCH, dump the cmake structure
# we need to patch. Writes to $LOG and prints STAGE1_DONE at the end.
set -u
LRX=/home/ylzhang/lrx
REPO=$LRX/FSCETP
LOG=$LRX/lammps_build.log
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }
cd "$LRX"

log "=== STAGE 1: download LAMMPS (github codeload — ~250x faster than lammps.org here) ==="
rm -f lammps-stable.tar.gz.partial
if [ ! -s lammps-stable.tar.gz ] || ! tar tzf lammps-stable.tar.gz >/dev/null 2>&1; then
  rm -f lammps-stable.tar.gz
  for TAG in stable_22Jul2025 stable_29Aug2024 stable; do
    log "trying tag $TAG ..."
    if curl -fL -o lammps-stable.tar.gz "https://codeload.github.com/lammps/lammps/tar.gz/refs/tags/$TAG" >>"$LOG" 2>&1 && tar tzf lammps-stable.tar.gz >/dev/null 2>&1; then
      log "downloaded tag $TAG"
      break
    fi
    rm -f lammps-stable.tar.gz
  done
fi
ls -la lammps-stable.tar.gz >>"$LOG" 2>&1
LDIR=$(tar tzf lammps-stable.tar.gz 2>/dev/null | head -1 | cut -d/ -f1)
log "LAMMPS dir = $LDIR"
if [ -z "$LDIR" ]; then log "BAD tarball"; log "STAGE1_DONE"; exit 1; fi

log "=== extract ==="
if [ ! -d "$LRX/$LDIR" ]; then
  tar xzf lammps-stable.tar.gz >>"$LOG" 2>&1
  log "extract rc=$?"
fi

log "=== install USER-MFFTORCH ==="
bash "$REPO/scripts/install_user_mfftorch_into_lammps.sh" "$LRX/$LDIR" >>"$LOG" 2>&1
log "install rc=$?"

CML="$LRX/$LDIR/cmake/CMakeLists.txt"
log "=== CMakeLists: STANDARD_PACKAGES occurrences ==="
grep -n "STANDARD_PACKAGES" "$CML" | tee -a "$LOG"
log "=== CMakeLists: PKG_WITH_INCL occurrences ==="
grep -n "PKG_WITH_INCL" "$CML" | tee -a "$LOG"
log "=== set(STANDARD_PACKAGES ... ) block (first 60 lines from match) ==="
awk '/set\(STANDARD_PACKAGES/{f=1} f{print NR": "$0} /\)/{if(f)c++} f&&c>=1{exit}' "$CML" | head -80 | tee -a "$LOG"
log "=== foreach(PKG_WITH_INCL ... ) context ==="
grep -n -A6 "foreach(PKG_WITH_INCL" "$CML" | tee -a "$LOG"
log "STAGE1_DONE"
