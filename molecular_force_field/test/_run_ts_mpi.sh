#!/bin/bash
# Run the small periodic test (option A binary) on np1 (serial) and np2 (domain decomposition);
# the run-0 PE must be IDENTICAL if option A is domain-decomposition-correct.
LMP=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp
WORK=/home/ylzhang/lrx/mff_md_work
PFX=/home/ylzhang/micromamba/envs/FSCETP
IN=$WORK/in.tstest
export PATH=$PFX/bin:$PATH
cd "$WORK" || exit 1
echo "=== np1 (serial) ==="
LD_LIBRARY_PATH=$PFX/lib timeout 300 "$LMP" -in "$IN" 2>&1 | grep -E "processor grid|Nghost|^ *0 |ERROR|out of memory" | head -4
echo "=== np2 (system OpenMPI domain decomp) ==="
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$PFX/lib timeout 300 /usr/bin/mpirun -np 2 "$LMP" -in "$IN" 2>&1 | grep -E "processor grid|Nghost|^ *0 |ERROR|out of memory" | head -5
