#!/bin/bash
# Validate the AOTI PADDING path (ntotal < N_max -> pad up to N_max, slice back). The periodic slab's
# refined-A ntotal (~8228) always exceeds N_max=762 (-> fallback, already validated), so to land in
# (1, 762) we run the SAME 695-atom config NON-periodically (boundary s s s -> no ghosts -> ntotal=695).
# That exercises padding (695 -> 762) on the .pt2, and we compare against the TorchScript core on the
# IDENTICAL non-periodic input. Both cores come from the same checkpoint, so a match validates that
# padding does not perturb the real atoms' energy or forces. No rebuild -- pure runtime test.
LOG=/home/ylzhang/lrx/test_aoti_pad.log
BDIR=/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk
LMP=$BDIR/lmp
WORK=/home/ylzhang/lrx/mff_md_work
PFX=/home/ylzhang/micromamba/envs/FSCETP
PY=$PFX/bin/python

: > "$LOG"

# in.slabpad: stay PERIODIC but enlarge the box so every periodic image falls beyond the ghost cutoff
# (~12 Ang) -> zero ghosts -> ntotal == nlocal == 695 < N_max 762 -> PADDING path. (boundary s s s
# dropped atoms sitting on the periodic box faces; the big-box trick keeps every atom and creates no
# ghosts. Box is now [-50,100]^3 with the 16.9 x 14.6 x 41.5 slab inside -> nearest atom-to-face >= 50
# >> 12, image distance >= 108 >> 12.) Energy here is the isolated-in-vacuum slab, NOT the -189.856
# periodic value -- but .pt2-padded and the TS core see the IDENTICAL input, so they must still agree.
write_in () {  # $1 = output file, $2 = core path, $3 = force dump path
  cat > "$1" <<EOF
units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data $WORK/slab_val.data
change_box all x final -50 100 y final -50 100 z final -50 100 units box
mass 1 1.0080
mass 2 15.9990
mass 3 18.9984
mass 4 39.0983
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
pair_style mff/torch 6.0 cuda
pair_coeff * * $2 H O F K
thermo_style custom step pe
thermo 1
dump 1 all custom 1 $3 id fx fy fz
dump_modify 1 sort id format float %.10g
run 0
EOF
}

write_in "$WORK/in.slabpad"    "$WORK/core_real762.pt2"      "$WORK/force_pad.dump"
write_in "$WORK/in.slabpad_ts" "$WORK/core_real_ts_hofk.pt"  "$WORK/force_ts.dump"

cd "$WORK"
echo "=== PADDING run: .pt2 (N_max 762) on non-periodic 695-atom slab (ntotal 695 -> pad 762) ===" >> "$LOG"
LD_LIBRARY_PATH=$PFX/lib timeout 300 "$LMP" -in "$WORK/in.slabpad"    >> "$LOG" 2>&1; echo "PAD_RC=$?" >> "$LOG"
echo "=== REFERENCE run: TorchScript core on the SAME non-periodic 695-atom slab ===" >> "$LOG"
LD_LIBRARY_PATH=$PFX/lib timeout 300 "$LMP" -in "$WORK/in.slabpad_ts" >> "$LOG" 2>&1; echo "TS_RC=$?" >> "$LOG"

echo "=== PotEng (pad vs ts) ===" >> "$LOG"
grep -A1 "Step" "$LOG" | grep -E "^ *[0-9]" >> "$LOG"

# force diff: both dumps are sorted by id, same atom order -> compare fx/fy/fz numerically.
"$PY" - "$WORK/force_pad.dump" "$WORK/force_ts.dump" >> "$LOG" 2>&1 <<'PYEOF'
import sys
def load(p):
    rows=[]; started=False
    for ln in open(p):
        if ln.startswith("ITEM: ATOMS"): started=True; continue
        if ln.startswith("ITEM:"): started=False; continue
        if started:
            t=ln.split()
            if len(t)>=4: rows.append((int(t[0]), float(t[1]), float(t[2]), float(t[3])))
    rows.sort()
    return rows
a=load(sys.argv[1]); b=load(sys.argv[2])
print(f"pad atoms={len(a)} ts atoms={len(b)}")
assert len(a)==len(b) and len(a)>0, "force-dump length mismatch"
md=0.0; mi=-1
for (ia,*fa),(ib,*fb) in zip(a,b):
    assert ia==ib
    for x,y in zip(fa,fb):
        d=abs(x-y)
        if d>md: md=d; mi=ia
print(f"max |force_pad - force_ts| = {md:.3e} eV/Ang  (atom id {mi})")
print("PADDING FORCES MATCH" if md<1e-4 else "PADDING FORCES DIFFER")
PYEOF
echo DONE >> "$LOG"
