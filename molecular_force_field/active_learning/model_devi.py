"""Model deviation calculator for active learning (DPGen2-style)."""

import logging
import os
from typing import List, Optional

import numpy as np
import torch
from ase.io import read

from molecular_force_field.active_learning.model_loader import build_e3trans_from_checkpoint
from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu
from molecular_force_field.utils.tensor_utils import map_tensor_values

logger = logging.getLogger(__name__)


class ModelDeviCalculator:
    """
    Compute model deviation (std across ensemble) for structures.
    Output format compatible with DPGen2 model_devi.out.
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        device: torch.device,
        atomic_energy_file: Optional[str] = None,
        tensor_product_mode: Optional[str] = None,
        num_interaction: int = 2,
        external_field: Optional[list] = None,
    ):
        self.device = device
        self.models = []
        self.config = None
        for path in checkpoint_paths:
            e3trans, config = build_e3trans_from_checkpoint(
                path,
                device,
                atomic_energy_file=atomic_energy_file,
                tensor_product_mode=tensor_product_mode,
                num_interaction=num_interaction,
            )
            self.models.append(e3trans)
            if self.config is None:
                self.config = config
        self.max_radius = self.config.max_radius
        self.keys = self.config.atomic_energy_keys.to(device)
        self.values = self.config.atomic_energy_values.to(device)
        self.external_tensor = None
        if external_field is not None:
            from molecular_force_field.active_learning.data_merge import (
                external_field_tensor_shape,
            )

            shape = external_field_tensor_shape(len(external_field))
            self.external_tensor = torch.tensor(
                external_field, dtype=torch.float64, device=device
            ).reshape(shape)

    def _batch_inputs(self, atoms_batch) -> tuple:
        """Build one batched graph from a list of ASE Atoms."""
        pos_chunks = []
        a_chunks = []
        batch_chunks = []
        cell_chunks = []
        edge_src_chunks = []
        edge_dst_chunks = []
        edge_shift_chunks = []
        counts = []
        node_offset = 0

        for batch_id, atoms in enumerate(atoms_batch):
            pos_i = torch.tensor(
                atoms.get_positions(), dtype=torch.float64, device=self.device
            )
            a_i = torch.tensor(
                atoms.get_atomic_numbers(), dtype=torch.float64, device=self.device
            )
            if any(atoms.pbc):
                cell_i = torch.tensor(
                    atoms.get_cell().array, dtype=torch.float64, device=self.device
                ).unsqueeze(0)
                pbc = tuple(bool(x) for x in atoms.pbc)
            else:
                cell_i = (
                    torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0)
                    * 100.0
                )
                pbc = (False, False, False)

            edge_src_i, edge_dst_i, edge_shifts_i = radius_graph_pbc_gpu(
                pos_i, self.max_radius, cell_i, pbc=pbc
            )

            n_i = int(pos_i.shape[0])
            counts.append(n_i)
            pos_chunks.append(pos_i)
            a_chunks.append(a_i)
            batch_chunks.append(
                torch.full((n_i,), batch_id, dtype=torch.long, device=self.device)
            )
            cell_chunks.append(cell_i.squeeze(0))
            edge_src_chunks.append(edge_src_i + node_offset)
            edge_dst_chunks.append(edge_dst_i + node_offset)
            edge_shift_chunks.append(edge_shifts_i)
            node_offset += n_i

        pos = torch.cat(pos_chunks, dim=0)
        a = torch.cat(a_chunks, dim=0)
        batch_idx = torch.cat(batch_chunks, dim=0)
        cell = torch.stack(cell_chunks, dim=0)
        if edge_src_chunks:
            edge_src = torch.cat(edge_src_chunks, dim=0)
            edge_dst = torch.cat(edge_dst_chunks, dim=0)
            edge_shifts = torch.cat(edge_shift_chunks, dim=0)
        else:
            edge_src = torch.empty(0, dtype=torch.long, device=self.device)
            edge_dst = torch.empty(0, dtype=torch.long, device=self.device)
            edge_shifts = torch.empty((0, 3), dtype=torch.float64, device=self.device)
        return pos, a, batch_idx, cell, edge_src, edge_dst, edge_shifts, counts

    def _predict_many(self, atoms_batch, model_idx: int) -> tuple:
        """Return (energies, forces_list) for one model over a batch of structures."""
        (
            pos,
            a,
            batch_idx,
            cell,
            edge_src,
            edge_dst,
            edge_shifts,
            counts,
        ) = self._batch_inputs(atoms_batch)
        pos = pos.requires_grad_(True)

        model = self.models[model_idx]
        mapped_a = map_tensor_values(a, self.keys, self.values)
        e_offset = torch.zeros(len(atoms_batch), dtype=torch.float64, device=self.device)
        e_offset.index_add_(0, batch_idx, mapped_a)

        fwd_kwargs = {}
        if self.external_tensor is not None:
            fwd_kwargs["external_tensor"] = self.external_tensor
        atom_energies = model(
            pos, a, batch_idx, edge_src, edge_dst, edge_shifts, cell, **fwd_kwargs
        )
        if atom_energies.ndim == 2 and atom_energies.shape[1] == 1:
            atom_energies = atom_energies.squeeze(-1)
        graph_energy = torch.zeros(
            len(atoms_batch), dtype=torch.float64, device=self.device
        )
        graph_energy.index_add_(0, batch_idx, atom_energies)
        graph_energy = graph_energy + e_offset
        grads = torch.autograd.grad(graph_energy.sum(), pos)[0]
        if not torch.isfinite(graph_energy).all():
            raise FloatingPointError(
                f"Non-finite total energy during model deviation for model {model_idx}"
            )
        if not torch.isfinite(grads).all():
            raise FloatingPointError(
                f"Non-finite force gradients during model deviation for model {model_idx}"
            )

        forces_np = (-grads).detach().cpu().numpy()
        energies_np = graph_energy.detach().cpu().numpy()
        forces_list = []
        start = 0
        for count in counts:
            stop = start + count
            forces_list.append(forces_np[start:stop])
            start = stop
        return energies_np, forces_list

    def compute_devi(self, atoms) -> dict:
        """
        Compute model deviation for one structure.
        Returns dict with max_devi_f, min_devi_f, avg_devi_f, devi_e.
        """
        return self.compute_devi_batch([atoms])[0]

    def compute_devi_batch(self, atoms_batch) -> List[dict]:
        """Compute model deviation for a list of structures."""
        if not atoms_batch:
            return []

        n_models = len(self.models)
        n_structures = len(atoms_batch)
        energies = np.zeros((n_models, n_structures), dtype=np.float64)
        forces_by_model = []
        for i in range(n_models):
            e_batch, f_batch = self._predict_many(atoms_batch, i)
            energies[i] = e_batch
            forces_by_model.append(f_batch)

        results = []
        for struct_idx in range(n_structures):
            e_struct = energies[:, struct_idx]
            forces = np.stack(
                [forces_by_model[m][struct_idx] for m in range(n_models)], axis=0
            )
            if not np.isfinite(e_struct).all():
                raise FloatingPointError("Non-finite ensemble energies in model deviation")
            if not np.isfinite(forces).all():
                raise FloatingPointError("Non-finite ensemble forces in model deviation")
            n_atoms = forces.shape[1]
            if n_atoms == 0:
                results.append(
                    {
                        "max_devi_f": 0.0,
                        "min_devi_f": 0.0,
                        "avg_devi_f": 0.0,
                        "devi_e": 0.0,
                        "per_atom_f_std": np.array([], dtype=np.float64),
                    }
                )
                continue
            f_std_per_atom = np.std(forces, axis=0)
            f_std_mag = np.linalg.norm(f_std_per_atom, axis=1)
            devi_e = np.std(e_struct) / max(n_atoms, 1)
            if not np.isfinite(f_std_mag).all() or not np.isfinite(devi_e):
                raise FloatingPointError("Non-finite deviation statistics")
            results.append(
                {
                    "max_devi_f": float(np.max(f_std_mag)),
                    "min_devi_f": float(np.min(f_std_mag)),
                    "avg_devi_f": float(np.mean(f_std_mag)),
                    "devi_e": float(devi_e),
                    "per_atom_f_std": f_std_mag,
                }
            )
        return results

    def compute_from_trajectory(
        self,
        traj_path: str,
        output_path: str = "model_devi.out",
        batch_size: int = 8,
    ) -> List[dict]:
        """
        Compute model deviation for all frames in trajectory.
        Writes model_devi.out (DPGen2 format) and a companion
        ``*_per_atom.txt`` file with per-atom force deviations.
        """
        from molecular_force_field.active_learning.diversity_selector import (
            save_per_atom_devi,
        )

        atoms_list = read(traj_path, index=":")
        results = []
        per_atom_devis: List[np.ndarray] = []
        batch_size = max(1, int(batch_size))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write("# frame_id max_devi_f min_devi_f avg_devi_f devi_e\n")
            frame_idx = 0
            for start in range(0, len(atoms_list), batch_size):
                chunk = atoms_list[start : start + batch_size]
                for devi in self.compute_devi_batch(chunk):
                    per_atom_devis.append(devi.pop("per_atom_f_std"))
                    results.append(devi)
                    f.write(
                        f"{frame_idx} {devi['max_devi_f']:.6e} {devi['min_devi_f']:.6e} "
                        f"{devi['avg_devi_f']:.6e} {devi['devi_e']:.6e}\n"
                    )
                    frame_idx += 1

        per_atom_path = output_path.replace(".out", "_per_atom.txt")
        save_per_atom_devi(per_atom_devis, per_atom_path)

        logger.info(f"Wrote model_devi to {output_path} ({len(results)} frames)")
        return results
