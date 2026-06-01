#!/usr/bin/env bash
# End-to-end MD: run pair_style mff/torch on CUDA, eager vs MFF_CUDA_GRAPH=1.
# Reports per-run Loop time / timesteps-per-s + final PE (correctness). Prints MD_DONE.
set -u
PY=/home/ylzhang/micromamba/envs/FSCETP/bin/python
LRX=/home/ylzhang/lrx
SRC=$LRX/FSCETP/lammps_user_mfftorch/test_cudagraph_repro
LDIR=$(ls -d "$LRX"/lammps-*/ 2>/dev/null | grep -v 'tar.gz' | head -1)
LMP="${LDIR%/}/build-mfftorch/lmp"
CORE=$LRX/mff_cg_work/core.pt
WORK=$LRX/mff_md_work
LOG=$WORK/md.log
NSTEPS=${1:-300}
mkdir -p "$WORK"; cd "$WORK"; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }

log "lmp=$LMP"
log "core=$CORE  nsteps=$NSTEPS"
if [ ! -x "$LMP" ]; then log "NO lmp binary"; log "MD_DONE"; exit 1; fi
if [ ! -f "$CORE" ]; then log "NO core.pt"; log "MD_DONE"; exit 1; fi
export LD_LIBRARY_PATH="$("$PY" -c "import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):${LD_LIBRARY_PATH:-}"

sed -e "s#__CORE_PT__#$CORE#" -e "s#__NSTEPS__#$NSTEPS#" "$SRC/in.mfftorch" > "$WORK/in.run"

log "=== RUN 1: EAGER (no MFF_CUDA_GRAPH) ==="
"$LMP" -in "$WORK/in.run" > "$WORK/eager.out" 2>&1
log "eager rc=$?"

log "=== RUN 2: MFF_CUDA_GRAPH=1 ==="
MFF_CUDA_GRAPH=1 "$LMP" -in "$WORK/in.run" > "$WORK/graph.out" 2>&1
log "graph rc=$?"

log "=== EAGER timing/PE ==="
grep -E "Loop time|Performance:" "$WORK/eager.out" | tee -a "$LOG"
log "eager final thermo:"; grep -E "^\s+[0-9]+\s" "$WORK/eager.out" | tail -1 | tee -a "$LOG"
log "=== GRAPH timing/PE ==="
grep -E "Loop time|Performance:" "$WORK/graph.out" | tee -a "$LOG"
log "graph final thermo:"; grep -E "^\s+[0-9]+\s" "$WORK/graph.out" | tail -1 | tee -a "$LOG"
log "=== any capture-failed warnings (graph run)? ==="
grep -E "CUDA Graph capture failed|Falling back to eager" "$WORK/graph.out" | tee -a "$LOG" || log "(none -> capture succeeded)"
log "MD_DONE"
