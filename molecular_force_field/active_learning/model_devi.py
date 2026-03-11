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
            from molecular_force_field.active_learning.data_merge import external_field_tensor_shape
            shape = external_field_tensor_shape(len(external_field))
            self.external_tensor = torch.tensor(
                external_field, dtype=torch.float64, device=device
            ).reshape(shape)

    def _predict_one(self, atoms, model_idx: int) -> tuple:
        """Return (energy, forces) for one model."""
        pos = torch.tensor(
            atoms.get_positions(), dtype=torch.float64, device=self.device
        )
        A = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.float64, device=self.device
        )
        batch_idx = torch.zeros(len(pos), dtype=torch.long, device=self.device)
        if any(atoms.pbc):
            cell = torch.tensor(
                atoms.get_cell().array,
                dtype=torch.float64,
                device=self.device,
            ).unsqueeze(0)
            pbc = tuple(bool(x) for x in atoms.pbc)
        else:
            cell = torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0) * 100.0
            pbc = (False, False, False)

        edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(
            pos, self.max_radius, cell, pbc=pbc
        )
        pos = pos.requires_grad_(True)

        model = self.models[model_idx]
        mapped_A = map_tensor_values(A, self.keys, self.values)
        E_offset = mapped_A.sum()
        fwd_kwargs = {}
        if self.external_tensor is not None:
            fwd_kwargs["external_tensor"] = self.external_tensor
        atom_energies = model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell, **fwd_kwargs)
        E_total = atom_energies.sum() + E_offset
        grads = torch.autograd.grad(E_total, pos)[0]
        forces = -grads.detach().cpu().numpy()
        energy = E_total.item()
        return energy, forces

    def compute_devi(self, atoms) -> dict:
        """
        Compute model deviation for one structure.
        Returns dict with max_devi_f, min_devi_f, avg_devi_f, devi_e.
        """
        energies = []
        forces_list = []
        for i in range(len(self.models)):
            e, f = self._predict_one(atoms, i)
            energies.append(e)
            forces_list.append(f)
        energies = np.array(energies)
        forces = np.stack(forces_list, axis=0)  # [n_models, n_atoms, 3]
        n_atoms = forces.shape[1]
        if n_atoms == 0:
            return {
                "max_devi_f": 0.0,
                "min_devi_f": 0.0,
                "avg_devi_f": 0.0,
                "devi_e": 0.0,
                "per_atom_f_std": np.array([], dtype=np.float64),
            }
        f_std_per_atom = np.std(forces, axis=0)  # [n_atoms, 3]
        f_std_mag = np.linalg.norm(f_std_per_atom, axis=1)  # [n_atoms]
        devi_e = np.std(energies) / max(n_atoms, 1)
        return {
            "max_devi_f": float(np.max(f_std_mag)),
            "min_devi_f": float(np.min(f_std_mag)),
            "avg_devi_f": float(np.mean(f_std_mag)),
            "devi_e": float(devi_e),
            "per_atom_f_std": f_std_mag,
        }

    def compute_from_trajectory(
        self,
        traj_path: str,
        output_path: str = "model_devi.out",
    ) -> List[dict]:
        """
        Compute model deviation for all frames in trajectory.
        Writes model_devi.out (DPGen2 format) and a companion
        ``*_per_atom.txt`` file with per-atom force deviations (for
        diversity-based sub-selection).

        Returns list of devi dicts (without the per_atom_f_std arrays).
        """
        from molecular_force_field.active_learning.diversity_selector import (
            save_per_atom_devi,
        )

        atoms_list = read(traj_path, index=":")
        results = []
        per_atom_devis: List[np.ndarray] = []
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(
                "# frame_id max_devi_f min_devi_f avg_devi_f devi_e\n"
            )
            for i, atoms in enumerate(atoms_list):
                devi = self.compute_devi(atoms)
                per_atom_devis.append(devi.pop("per_atom_f_std"))
                results.append(devi)
                f.write(
                    f"{i} {devi['max_devi_f']:.6e} {devi['min_devi_f']:.6e} "
                    f"{devi['avg_devi_f']:.6e} {devi['devi_e']:.6e}\n"
                )

        per_atom_path = output_path.replace(".out", "_per_atom.txt")
        save_per_atom_devi(per_atom_devis, per_atom_path)

        logger.info(f"Wrote model_devi to {output_path} ({len(results)} frames)")
        return results
