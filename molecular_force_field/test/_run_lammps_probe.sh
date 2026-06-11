#!/bin/bash
# Run the kk lmp binary on the probe input (dummy TorchScript core, run 0) and report
# Nlocal/Nghost + any error. Blocking but quick (run 0 = single force eval).
set -u
LMP=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp
IN=/home/ylzhang/lrx/mff_md_work/in.probe663
LOG=/home/ylzhang/lrx/md_probe.log
export PATH=/home/ylzhang/micromamba/envs/FSCETP/bin:$PATH
export LD_LIBRARY_PATH=/home/ylzhang/micromamba/envs/FSCETP/lib:${LD_LIBRARY_PATH:-}

cd /home/ylzhang/lrx/mff_md_work || exit 1
: > "$LOG"
timeout 300 "$LMP" -in "$IN" > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
echo "===GREP==="
grep -iE "Nlocal|Nghost|ghost atom cutoff|ERROR|error|aoti|AOTI|terminate|assert|^ *0 " "$LOG" 2>&1 | head -30
echo "===TAIL==="
tail -15 "$LOG"
