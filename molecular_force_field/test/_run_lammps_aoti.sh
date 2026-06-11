#!/bin/bash
# Run the kk lmp binary on the AOTI .pt2 deployment test. Blocking with a timeout;
# prints the key lines (aoti detection, Nghost, thermo PE, timing, any error).
set -u
LMP=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp
IN=/home/ylzhang/lrx/mff_md_work/in.run.aoti
LOG=/home/ylzhang/lrx/md_aoti.log
export PATH=/home/ylzhang/micromamba/envs/FSCETP/bin:$PATH
export LD_LIBRARY_PATH=/home/ylzhang/micromamba/envs/FSCETP/lib:${LD_LIBRARY_PATH:-}

cd /home/ylzhang/lrx/mff_md_work || exit 1
: > "$LOG"
timeout 400 "$LMP" -in "$IN" > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
echo "===GREP==="
grep -iE "aoti|AOTI|Nlocal|Nghost|ghost atom cutoff|ERROR|error|terminate|assert|nan|inf|Loop time|Performance|timesteps/s" "$LOG" 2>&1 | head -40
echo "===THERMO==="
grep -E "^ *(Step|0|20|40|60|80|100) " "$LOG" 2>&1 | head -20
echo "===TAIL==="
tail -8 "$LOG"
