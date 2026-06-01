#!/usr/bin/env bash
# Kokkos /kk end-to-end MD: pair_style mff/torch on CUDA via Kokkos, eager vs
# MFF_CUDA_GRAPH=1. Reports Loop time / timesteps-per-s + final PE. Prints MD_DONE.
set -u
PY=/home/ylzhang/micromamba/envs/FSCETP/bin/python
LRX=/home/ylzhang/lrx
SRC=$LRX/FSCETP/lammps_user_mfftorch/test_cudagraph_repro
LDIR=$(ls -d "$LRX"/lammps-*/ 2>/dev/null | grep -v 'tar.gz' | head -1); LDIR=${LDIR%/}
LMP="$LDIR/build-mfftorch-kk/lmp"
CORE=$LRX/mff_cg_work/core.pt
WORK=$LRX/mff_md_kk_work
LOG=$WORK/md_kk.log
NSTEPS=${1:-300}
KK="-k on g 1 -sf kk -pk kokkos newton off neigh full"
mkdir -p "$WORK"; cd "$WORK"; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

log "lmp(kk)=$LMP  core=$CORE  nsteps=$NSTEPS"
[ -x "$LMP" ] || { log "no kk lmp"; log "MD_DONE"; exit 1; }
[ -f "$CORE" ] || { log "no core.pt"; log "MD_DONE"; exit 1; }
export LD_LIBRARY_PATH="$("$PY" -c "import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):${LD_LIBRARY_PATH:-}"
sed -e "s#__CORE_PT__#$CORE#" -e "s#__NSTEPS__#$NSTEPS#" "$SRC/in.mfftorch" > "$WORK/in.run"

log "=== KK RUN 1: EAGER ==="
"$LMP" $KK -in "$WORK/in.run" > "$WORK/eager.out" 2>&1
log "eager rc=$?"
log "=== KK RUN 2: MFF_CUDA_GRAPH=1 ==="
MFF_CUDA_GRAPH=1 "$LMP" $KK -in "$WORK/in.run" > "$WORK/graph.out" 2>&1
log "graph rc=$?"

log "=== EAGER timing/PE ==="
grep -E "Loop time|Performance:" "$WORK/eager.out" | tee -a "$LOG"
grep -E "^\s+[0-9]+\s" "$WORK/eager.out" | tail -1 | tee -a "$LOG"
log "=== GRAPH timing/PE ==="
grep -E "Loop time|Performance:" "$WORK/graph.out" | tee -a "$LOG"
grep -E "^\s+[0-9]+\s" "$WORK/graph.out" | tail -1 | tee -a "$LOG"
log "=== capture-failed warnings (graph run)? ==="
grep -E "CUDA Graph capture failed|Falling back to eager" "$WORK/graph.out" | tee -a "$LOG" || log "(no capture-failed line found)"
log "=== any LAMMPS ERROR? ==="
grep -iE "ERROR" "$WORK/eager.out" "$WORK/graph.out" | head -5 | tee -a "$LOG" || true
log "MD_DONE"
