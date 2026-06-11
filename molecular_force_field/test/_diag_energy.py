import ase.io
import ase.data
from collections import Counter

XYZ = "/home/ylzhang/lrx/lr_ablation/data/slab_val.xyz"
at = ase.io.read(XYZ, index=0)
e_info = at.info.get("energy")
try:
    e_calc = at.get_potential_energy()
except Exception as ex:
    e_calc = f"ERR {ex}"
comp = Counter(at.get_chemical_symbols())
E0 = {1: -0.17469873, 8: -0.08734936, 9: -4.62181598, 19: -4.62181598}
sumE0 = sum(E0[ase.data.atomic_numbers[s]] * n for s, n in comp.items())
LMP = -119.4183
print("N =", len(at), " comp =", dict(comp))
print("energy at.info =", e_info)
print("energy at.calc =", e_calc)
print("sum_E0 =", round(sumE0, 5))
print("LAMMPS PE =", LMP)
print("LMP + sumE0 =", round(LMP + sumE0, 5))
print("LMP - sumE0 =", round(LMP - sumE0, 5))
