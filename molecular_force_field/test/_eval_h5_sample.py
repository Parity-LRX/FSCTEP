"""Run the eager model on the EXACT preprocessed val graph (processed_val.h5 sample 0) and compare
to the stored target y (= E0-subtracted energy). Isolates model correctness from the LAMMPS/export
path: if E_pred ~= y, the model is correct and the +70 eV gap is in deployment (trace/engine)."""
import sys
import numpy as np
import torch
import h5py
from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
H5 = "/home/ylzhang/lrx/lr_ablation/data_slab/processed_val.h5"
CKPT = "/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix_repaired.pth"
DEV = "cuda"

obj = LAMMPS_MLIAP_MFF.from_checkpoint(checkpoint_path=CKPT, element_types=["H", "O", "F", "K"], device=DEV)
m = obj.wrapper.model
m.eval()
m.skip_input_validation = True
dt = next(m.parameters()).dtype

f = h5py.File(H5, "r")
g = f[f"sample_{IDX}"]
pos = torch.tensor(g["pos"][:], dtype=dt, device=DEV)
A = torch.tensor(g["A"][:], dtype=torch.long, device=DEV)
es = torch.tensor(g["edge_src"][:], dtype=torch.long, device=DEV)
ed = torch.tensor(g["edge_dst"][:], dtype=torch.long, device=DEV)
esh = torch.tensor(g["edge_shifts"][:], dtype=dt, device=DEV)
cell = torch.tensor(g["cell"][:], dtype=dt, device=DEV)
if cell.dim() == 2:
    cell = cell.unsqueeze(0)
y = float(np.array(g["y"][()]))
batch = torch.zeros(pos.shape[0], dtype=torch.long, device=DEV)

with torch.no_grad():
    out = m(pos, A, batch, es, ed, esh, cell)
    atom_e = out[0] if isinstance(out, tuple) else out
    E = float(atom_e.sum())

aek = obj.wrapper.atomic_energy_keys.tolist()
aev = obj.wrapper.atomic_energy_values.tolist()
E0 = {int(k): float(v) for k, v in zip(aek, aev)}
sumE0 = sum(E0[int(z)] for z in A.tolist())

print(f"N={pos.shape[0]} edges={es.shape[0]} avg_num_neighbors={m.avg_num_neighbors}")
print(f"E_pred (network, E0-subtracted) = {E:.6f}")
print(f"y (target correction)           = {y:.6f}")
print(f"  -> network err vs y           = {E - y:.6f} eV")
print(f"sumE0 = {sumE0:.6f}")
print(f"E_pred + sumE0 (absolute)       = {E + sumE0:.6f}")
print(f"dataset raw absolute            = -189.830392")
