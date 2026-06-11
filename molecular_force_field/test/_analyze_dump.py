"""Replay the EXACT engine graph (dumped to /tmp/mff_graph_*.pt) through the eager model.
Tests the current edge convention (es=local_center -> ed=neighbor) vs the swapped one
(neighbor -> center) to localize the +70 eV LAMMPS energy error. Sums LOCAL atom energies
(first nlocal), exactly like the pair_style."""
import torch
from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

CKPT = "/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix_repaired.pth"
DEV = "cuda"

def load(name):
    # C++ torch::save writes a TorchScript archive holding the tensor as parameter "0".
    return torch.jit.load(f"/tmp/mff_graph_{name}.pt").state_dict()["0"]

pos, A, es, ed, esh, cell = (load(n) for n in ("pos", "A", "es", "ed", "esh", "cell"))
nlocal, ntotal, E = (int(x) for x in open("/tmp/mff_graph_meta.txt").read().split())

obj = LAMMPS_MLIAP_MFF.from_checkpoint(checkpoint_path=CKPT, element_types=["H", "O", "F", "K"], device=DEV)
m = obj.wrapper.model
m.eval()
m.skip_input_validation = True
dt = next(m.parameters()).dtype
pos = pos.to(DEV, dt); A = A.to(DEV).long()
es = es.to(DEV).long(); ed = ed.to(DEV).long()
esh = esh.to(DEV, dt); cell = cell.to(DEV, dt)
if cell.dim() == 2:
    cell = cell.unsqueeze(0)
batch = torch.zeros(pos.shape[0], dtype=torch.long, device=DEV)

aek = obj.wrapper.atomic_energy_keys.tolist()
aev = obj.wrapper.atomic_energy_values.tolist()
E0 = {int(k): float(v) for k, v in zip(aek, aev)}
sumE0_local = sum(E0[int(z)] for z in A[:nlocal].tolist())

def run(src, dst, shifts):
    with torch.no_grad():
        out = m(pos, A, batch, src, dst, shifts, cell)
        ae = out[0] if isinstance(out, tuple) else out
        return float(ae[:nlocal].sum())

print(f"nlocal={nlocal} ntotal={ntotal} E={E} avg={m.avg_num_neighbors}")
print(f"edge_shifts abs max = {float(esh.abs().max()):.4g}  (≈0 means ghosts carry the image)")
net_cur = run(es, ed, esh)
net_swap = run(ed, es, -esh)
print(f"CURRENT  (es=center -> ed=neighbor): local network = {net_cur:10.4f}  abs(+E0) = {net_cur + sumE0_local:10.4f}")
print(f"SWAPPED  (neighbor -> center):       local network = {net_swap:10.4f}  abs(+E0) = {net_swap + sumE0_local:10.4f}")
print("reference: correct network +0.849, dataset abs -189.830; LAMMPS PE was -119.42")
