"""Build a LAMMPS data file + input from a dataset slab config, to check the deployed model's
energy against the dataset reference (DFT) energy. Uses the N-flexible TorchScript core (the .pt2
bakes N). Writes the pair_coeff/mass lines for exactly the elements present (sorted by Z)."""
import sys
import ase.io
import ase.data

XYZ = "/home/ylzhang/lrx/lr_ablation/data/slab_val.xyz"
IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CORE = sys.argv[2] if len(sys.argv) > 2 else "/home/ylzhang/lrx/mff_md_work/core_real_ts.pt"
DATA = "/home/ylzhang/lrx/mff_md_work/slab_val.data"
INFILE = "/home/ylzhang/lrx/mff_md_work/in.slabval"

at = ase.io.read(XYZ, index=IDX)
refE = at.info.get("energy", at.info.get("Energy"))
syms = sorted(set(at.get_chemical_symbols()), key=lambda s: ase.data.atomic_numbers[s])
# ASE default omits Masses; write atom_style atomic and set masses ourselves in the input.
ase.io.write(DATA, at, format="lammps-data", specorder=syms)
N = len(at)
mass_lines = "\n".join(
    f"mass {i + 1} {ase.data.atomic_masses[ase.data.atomic_numbers[s]]:.4f}" for i, s in enumerate(syms))
pc = " ".join(syms)
inp = f"""# auto-generated: dataset slab_val[{IDX}] -> LAMMPS energy vs DFT
units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data {DATA}
{mass_lines}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
pair_style mff/torch 6.0 cuda
pair_coeff * * {CORE} {pc}
thermo_style custom step pe
thermo 1
run 0
"""
with open(INFILE, "w") as f:
    f.write(inp)
print(f"N={N} ref_energy={refE} elements={syms} specorder={pc}")
print(f"wrote {DATA} and {INFILE}")
