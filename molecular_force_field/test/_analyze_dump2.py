"""Confirm the FIX: collapse the engine's ghost-node graph back to the training topology
(local nodes + integer PBC shifts) by mapping each ghost neighbor to its local atom + cell
offset, and emit edges (neighbor_local -> center_local, shift). Run on nlocal atoms only.
Expect ~ +0.849 network / -189.83 absolute (matching the dataset)."""
import torch
import numpy as np
from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

CKPT = "/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix_repaired.pth"
DEV = "cuda"

def load(name):
    return torch.jit.load(f"/tmp/mff_graph_{name}.pt").state_dict()["0"]

pos, A, es, ed, cell = (load(n) for n in ("pos", "A", "es", "ed", "cell"))
nlocal, ntotal, E = (int(x) for x in open("/tmp/mff_graph_meta.txt").read().split())
posn = pos.double().numpy()
celln = cell.double().numpy().reshape(3, 3)
L = np.array([celln[0, 0], celln[1, 1], celln[2, 2]])  # orthogonal box

# map every atom index -> (local_idx, integer shift s) with pos[idx] = pos[local] + s*L
localpos = torch.tensor(posn[:nlocal])
wrapped_all = posn - np.floor(posn / L) * L                       # into [0,L)
d = torch.cdist(torch.tensor(wrapped_all), localpos)             # (ntotal, nlocal)
local_of = d.argmin(dim=1).numpy()
maxmatch = float(d.min(dim=1).values.max())
shift_of = np.round((posn - posn[local_of]) / L).astype(np.int64)  # s per atom

esn, edn = es.numpy(), ed.numpy()
# dumped edge: es=center i (local), ed=neighbor j (local/ghost). Training edge: src=neighbor_local,
# dst=center, shift carries the ghost's image. center i is already local.
src_local = local_of[edn]          # neighbor's local index
dst_local = esn                    # center (local)
neigh_shift = shift_of[edn]        # the ghost offset of the neighbor

Apar = A[:nlocal].to(DEV).long()
posL = pos[:nlocal].to(DEV)
obj = LAMMPS_MLIAP_MFF.from_checkpoint(checkpoint_path=CKPT, element_types=["H", "O", "F", "K"], device=DEV)
m = obj.wrapper.model
m.eval(); m.skip_input_validation = True
dt = next(m.parameters()).dtype
posL = posL.to(dt)
cellL = cell.to(DEV, dt)
if cellL.dim() == 2:
    cellL = cellL.unsqueeze(0)
batch = torch.zeros(nlocal, dtype=torch.long, device=DEV)
src = torch.tensor(src_local, dtype=torch.long, device=DEV)
dst = torch.tensor(dst_local, dtype=torch.long, device=DEV)
aek = obj.wrapper.atomic_energy_keys.tolist(); aev = obj.wrapper.atomic_energy_values.tolist()
E0 = {int(k): float(v) for k, v in zip(aek, aev)}
sumE0 = sum(E0[int(z)] for z in Apar.tolist())

def run(sh):
    shT = torch.tensor(sh, dtype=dt, device=DEV)
    with torch.no_grad():
        out = m(posL, Apar, batch, src, dst, shT, cellL)
        ae = out[0] if isinstance(out, tuple) else out
        return float(ae.sum())

print(f"nlocal={nlocal} ntotal={ntotal} E={E}  max ghost->local match dist={maxmatch:.2e}")
for name, sh in (("+s", neigh_shift), ("-s", -neigh_shift)):
    net = run(sh)
    print(f"shift={name}: network = {net:10.4f}  abs(+E0) = {net + sumE0:10.4f}")
print("target: network +0.849, abs -189.830")
