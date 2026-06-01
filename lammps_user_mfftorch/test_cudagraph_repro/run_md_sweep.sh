#!/usr/bin/env bash
# Throughput sweep over atom count (fcc lattice, dense) on the Kokkos /kk GPU path:
# eager vs MFF_CUDA_GRAPH=1. Reports loop time, timesteps/s, katom-step/s, ave neighs/atom,
# speedup, and whether capture succeeded. Prints SWEEP_DONE.
set -u
PY=/home/ylzhang/micromamba/envs/FSCETP/bin/python
LRX=/home/ylzhang/lrx
SRC=$LRX/FSCETP/lammps_user_mfftorch/test_cudagraph_repro
LDIR=$(ls -d "$LRX"/lammps-*/ 2>/dev/null | grep -v 'tar.gz' | head -1); LDIR=${LDIR%/}
LMP="$LDIR/build-mfftorch-kk/lmp"
CORE=$LRX/mff_cg_work/core.pt
WORK=$LRX/mff_sweep_work
LOG=$WORK/sweep.log
A=${1:-3.6}
NSTEPS=${2:-400}
NCLIST="${3:-6 8 10}"
KK="-k on g 1 -sf kk -pk kokkos newton off neigh full"
mkdir -p "$WORK"; cd "$WORK"; : > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }
[ -x "$LMP" ] || { log "no kk lmp"; log "SWEEP_DONE"; exit 1; }
export LD_LIBRARY_PATH="$("$PY" -c "import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):${LD_LIBRARY_PATH:-}"

log "lattice=fcc a=${A}A  nsteps=$NSTEPS  NC=[$NCLIST]  lmp=$LMP"
val() { grep -E "Performance:" "$1" | grep -oE "[0-9.]+ $2" | awk '{print $1}'; }
loop() { grep -E "Loop time" "$1" | awk '{print $4}'; }

for NC in $NCLIST; do
  N=$((NC*NC*NC*4))
  log "===================== NC=$NC  (~$N atoms, fcc a=$A) ====================="
  sed -e "s#__CORE_PT__#$CORE#g" -e "s#__A__#$A#g" -e "s#__NC__#$NC#g" -e "s#__NSTEPS__#$NSTEPS#g" \
      "$SRC/in.mfftorch_lat" > "$WORK/in.$NC"
  "$LMP" $KK -in "$WORK/in.$NC" > "$WORK/eager.$NC" 2>&1
  MFF_CUDA_GRAPH=1 "$LMP" $KK -in "$WORK/in.$NC" > "$WORK/graph.$NC" 2>&1
  NB=$(grep -iE "ave neighs/atom" "$WORK/eager.$NC" | tail -1 | sed 's/^ *//')
  EL=$(loop "$WORK/eager.$NC"); GL=$(loop "$WORK/graph.$NC")
  ET=$(val "$WORK/eager.$NC" "timesteps/s"); GT=$(val "$WORK/graph.$NC" "timesteps/s")
  EK=$(val "$WORK/eager.$NC" "katom-step/s"); GK=$(val "$WORK/graph.$NC" "katom-step/s")
  EPE=$(grep -E "^\s+[0-9]+\s" "$WORK/eager.$NC" | tail -1 | awk '{print $3}')
  GPE=$(grep -E "^\s+[0-9]+\s" "$WORK/graph.$NC" | tail -1 | awk '{print $3}')
  CAP=$(grep -cE "capture failed|Falling back to eager" "$WORK/graph.$NC")
  ERR=$(grep -icE "ERROR" "$WORK/eager.$NC" "$WORK/graph.$NC" | awk -F: '{s+=$2} END{print s+0}')
  SU=$(awk "BEGIN{if(\"$GL\"!=\"\" && $GL>0) printf \"%.2f\", $EL/$GL; else print \"NA\"}")
  log "  $NB"
  log "  eager : loop=${EL}s  ${ET} steps/s  ${EK} katom-step/s  PE=$EPE"
  log "  graph : loop=${GL}s  ${GT} steps/s  ${GK} katom-step/s  PE=$GPE"
  log "  SPEEDUP=${SU}x   capture_failed=$CAP   lammps_errors=$ERR"
done
log "SWEEP_DONE"
