#!/usr/bin/env bash
# Build + run the standalone MFF_CUDA_GRAPH reproducer on the 4090.
# Writes progress to $WORK/run.log and prints ALL_DONE at the end.
set -u
PY=/home/ylzhang/micromamba/envs/FSCETP/bin/python
REPO=/home/ylzhang/lrx/FSCETP
SRC=$REPO/lammps_user_mfftorch/test_cudagraph_repro
WORK=/home/ylzhang/lrx/mff_cg_work
LOG=$WORK/run.log
mkdir -p "$WORK"
cd "$WORK"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== STAGE 1: core.pt ==="
if [ ! -f "$WORK/core.pt" ]; then
  "$PY" -c "import torch; from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd as m; m('$WORK/dummy.pth', device=torch.device('cpu')); print('dummy.pth ok')" >>"$LOG" 2>&1
  log "dummy.pth exit=$?"
  "$PY" -m molecular_force_field.cli.export_libtorch_core --checkpoint "$WORK/dummy.pth" --elements H O --device cuda --max-radius 5.0 --out "$WORK/core.pt" >>"$LOG" 2>&1
  log "export exit=$?"
fi
ls -la "$WORK/core.pt" >>"$LOG" 2>&1 || { log "NO core.pt - aborting"; log "ALL_DONE"; exit 1; }

log "=== STAGE 2: cmake configure + build ==="
LIBTORCH=$("$PY" -c "import torch; print(torch.utils.cmake_prefix_path)")
log "LIBTORCH=$LIBTORCH"
cmake -S "$SRC" -B "$WORK/build" -D CMAKE_PREFIX_PATH="$LIBTORCH" -D CMAKE_BUILD_TYPE=Release >>"$LOG" 2>&1
log "cmake configure exit=$?"
cmake --build "$WORK/build" -j 40 >>"$LOG" 2>&1
BUILD_RC=$?
log "cmake build exit=$BUILD_RC"
if [ "$BUILD_RC" -ne 0 ] || [ ! -x "$WORK/build/mff_cg_repro" ]; then
  log "BUILD FAILED - aborting"; log "ALL_DONE"; exit 1
fi

export LD_LIBRARY_PATH="$("$PY" -c "import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):${LD_LIBRARY_PATH:-}"

log "=== STAGE 3a: baseline (eager, no cuda graph) ==="
"$WORK/build/mff_cg_repro" "$WORK/core.pt" cuda 64 512 4 >>"$LOG" 2>&1
log "baseline exit=$?"

log "=== STAGE 3b: MFF_CUDA_GRAPH=1 ==="
MFF_CUDA_GRAPH=1 "$WORK/build/mff_cg_repro" "$WORK/core.pt" cuda 64 512 4 >>"$LOG" 2>&1
log "cudagraph exit=$?"

log "ALL_DONE"
