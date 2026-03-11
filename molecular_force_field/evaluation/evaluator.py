"""Evaluation module for model assessment."""

import logging
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from molecular_force_field.utils.scatter import scatter

from molecular_force_field.utils.tensor_utils import map_tensor_values
from molecular_force_field.models.pure_cartesian_ictd_layers import PhysicalTensorICTDEmbedding


class Evaluator:
    """Evaluator for static model evaluation."""
    
    def __init__(self, model, dataset, device, atomic_energy_keys=None,
                 atomic_energy_values=None, force_shift_value=1,
                 output_physical_tensors: bool = False):
        """
        Initialize evaluator.
        
        Args:
            model: E3NN model (E3_TransformerLayer_multi)
            dataset: Dataset for evaluation
            device: Device to use
            atomic_energy_keys: Atomic energy keys tensor
            atomic_energy_values: Atomic energy values tensor
            force_shift_value: Force shift value
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.force_shift_value = force_shift_value
        
        if atomic_energy_keys is None:
            self.keys = torch.tensor([1, 6, 7, 8]).to(device)
        else:
            self.keys = atomic_energy_keys.to(device)
            
        if atomic_energy_values is None:
            self.values = torch.tensor([
                -430.53299511, -821.03326787, -1488.18856918, -2044.3509823
            ]).to(device)
        else:
            self.values = atomic_energy_values.to(device)
        self.output_physical_tensors = output_physical_tensors
        self._phys_label_embedders = {}
        self.physical_tensor_weights = {}

    def _compute_physical_tensor_loss(self, extras, phys_pred, batch_idx, model):
        if phys_pred is None or not isinstance(phys_pred, dict):
            return torch.tensor(0.0, device=self.device)
        lmax_model = int(getattr(model, "lmax", 2))
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
        phys_loss = torch.tensor(0.0, device=self.device)

        def _phys_weight(name: str) -> float:
            base = name.replace("_per_atom", "")
            return self.physical_tensor_weights.get(name, self.physical_tensor_weights.get(base, 1.0))

        def _get_embed(rank: int, *, include_trace_chain: bool):
            key = (rank, include_trace_chain, lmax_model)
            m = self._phys_label_embedders.get(key)
            if m is None:
                m = PhysicalTensorICTDEmbedding(
                    rank=rank,
                    lmax_out=lmax_model,
                    channels_in=1,
                    channels_out=1,
                    input_repr="cartesian",
                    include_trace_chain=include_trace_chain,
                ).to(self.device)
                self._phys_label_embedders[key] = m
            return m

        for name in ("charge", "charge_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                y = extras[name].view(-1)
                p0 = phys_pred[name].get(0) if isinstance(phys_pred[name], dict) else None
                if p0 is None:
                    continue
                if name.endswith("_per_atom"):
                    if p0.shape[0] != y.shape[0]:
                        continue
                    p = p0.view(y.shape[0], -1).mean(dim=1)
                else:
                    if p0.shape[0] != num_graphs:
                        continue
                    p = p0.view(num_graphs, -1).mean(dim=1)
                phys_loss = phys_loss + w * F.smooth_l1_loss(p, y, beta=0.5)

        for name in ("dipole", "dipole_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                y_blocks = _get_embed(1, include_trace_chain=False)(extras[name], return_blocks=True)
                p1 = phys_pred[name].get(1) if isinstance(phys_pred[name], dict) else None
                if p1 is not None and 1 in y_blocks:
                    phys_loss = phys_loss + w * F.smooth_l1_loss(p1.view(-1), y_blocks[1].view(-1), beta=0.5)

        for name in ("polarizability", "polarizability_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                y_blocks = _get_embed(2, include_trace_chain=True)(extras[name], return_blocks=True)
                for l in (0, 2):
                    pl = phys_pred[name].get(l) if isinstance(phys_pred[name], dict) else None
                    if pl is not None and l in y_blocks:
                        phys_loss = phys_loss + w * F.smooth_l1_loss(pl.view(-1), y_blocks[l].view(-1), beta=0.5)

        for name in ("quadrupole", "quadrupole_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                y_blocks = _get_embed(2, include_trace_chain=True)(extras[name], return_blocks=True)
                p2 = phys_pred[name].get(2) if isinstance(phys_pred[name], dict) else None
                if p2 is not None and 2 in y_blocks:
                    phys_loss = phys_loss + w * F.smooth_l1_loss(p2.view(-1), y_blocks[2].view(-1), beta=0.5)

        return phys_loss
    
    def evaluate(self, data_loader, output_prefix='test'):
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        val_E_preds = []
        val_E_targets = []
        val_E_avg_preds = []
        val_E_avg_targets = []
        val_F_preds = []
        val_F_targets = []
        val_total_loss_list = []
        val_phys_loss_list = []
        
        # Use SmoothL1Loss consistent with run_eval.py
        criterion = torch.nn.SmoothL1Loss(beta=0.5)
        
        for batch_idx_loader, batch in enumerate(data_loader):
            if batch is None:
                continue
            
            # Unpack
            extras = {}
            if isinstance(batch, (list, tuple)) and len(batch) == 11:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, _stress_ref, extras) = batch
            else:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, _stress_ref) = batch

            # Move to GPU
            pos = pos.to(self.device)
            A = A.to(self.device)
            batch_idx = batch_idx.to(self.device)
            force_ref = force_ref.to(self.device)
            target_energies = target_energies.to(self.device)
            edge_src = edge_src.to(self.device)
            edge_dst = edge_dst.to(self.device)
            edge_shifts = edge_shifts.to(self.device)
            cell = cell.to(self.device)
            extras = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in (extras or {}).items()}
            
            pos.requires_grad = True
            
            mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
            # E_offset_val shape: [Batch_Size]
            E_offset_val = scatter(mapped_A, batch_idx, dim=0, reduce='sum')
            
            # Forward
            forward_kwargs = {}
            if "external_field" in extras:
                forward_kwargs["external_tensor"] = extras["external_field"]
            want_phys = any(
                k in extras for k in (
                    "charge", "dipole", "polarizability", "quadrupole",
                    "charge_per_atom", "dipole_per_atom", "polarizability_per_atom", "quadrupole_per_atom",
                )
            )
            supports_phys = hasattr(self.model, "physical_tensor_heads") and self.model.physical_tensor_heads is not None
            if (self.output_physical_tensors or want_phys) and supports_phys:
                forward_kwargs["return_physical_tensors"] = True
            try:
                out = self.model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell, **forward_kwargs)
            except TypeError:
                out = self.model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
            E_per_atom = out[0] if isinstance(out, tuple) else out
            phys_pred = out[1] if isinstance(out, tuple) and len(out) >= 2 else None
            # E_conv_val shape: [Batch_Size, 1]
            E_conv_val = scatter(E_per_atom, batch_idx, dim=0, reduce='sum')
            
            # E_mean_val shape: [Batch_Size]
            E_mean_val = E_conv_val.view(-1) + E_offset_val
            
            # Backward (Force)
            grads = torch.autograd.grad(
                E_mean_val.sum(), pos, create_graph=False, retain_graph=False
            )[0]
            f_pred = -grads
            
            # Restore units
            fx_pred = self.dataset.restore_force(f_pred[:, 0])
            fy_pred = self.dataset.restore_force(f_pred[:, 1])
            fz_pred = self.dataset.restore_force(f_pred[:, 2])
            f_pred_final = torch.stack([fx_pred, fy_pred, fz_pred], dim=1)  # [N_atoms, 3]
            
            f_ref_final = force_ref * self.force_shift_value
            
            # Collect data (detach and move to CPU)
            with torch.no_grad():
                # Energy [Batch_Size]
                val_E_preds.append(E_mean_val.detach().cpu())
                val_E_targets.append(target_energies.detach().cpu())
                
                # Average atomic energy [Batch_Size]
                num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum')
                val_E_avg_preds.append((E_mean_val / num_atoms_per_mol).detach().cpu())
                val_E_avg_targets.append((target_energies / num_atoms_per_mol).detach().cpu())
                
                # Force [N_atoms, 3] (Note: keep structure to save as CSV)
                val_F_preds.append(f_pred_final.detach().cpu())
                val_F_targets.append(f_ref_final.detach().cpu())

                # Loss (only for Log)
                batch_e_loss = criterion(E_mean_val, target_energies)
                batch_f_loss = criterion(f_pred_final.view(-1), f_ref_final.view(-1))
                batch_phys_loss = torch.tensor(0.0, device=self.device)
                if want_phys and supports_phys and phys_pred is not None:
                    batch_phys_loss = self._compute_physical_tensor_loss(extras, phys_pred, batch_idx, self.model)
                    val_phys_loss_list.append(batch_phys_loss.item())
                val_total_loss_list.append((batch_e_loss + batch_f_loss + batch_phys_loss).item())

            # Fix: take [0] when printing log to avoid error when batch_size > 1
            logging.info(f"Batch Sample -> Pred: {self.dataset.restore_energy(E_mean_val[0].item()):.4f} , True: {self.dataset.restore_energy(target_energies[0].item()):.4f}")
        
        # Process data after loop ends
        # 1. Concatenate
        all_E_preds = torch.cat(val_E_preds)  # [Total_Mols]
        all_E_targets = torch.cat(val_E_targets)  # [Total_Mols]
        all_E_avg_preds = torch.cat(val_E_avg_preds)
        all_E_avg_targets = torch.cat(val_E_avg_targets)
        
        all_F_preds = torch.cat(val_F_preds)  # [Total_Atoms, 3]
        all_F_targets = torch.cat(val_F_targets)  # [Total_Atoms, 3]
        
        # 2. Calculate global metrics (Note: Force needs view(-1) here)
        # Energy
        val_energy_rmse = torch.sqrt(F.mse_loss(all_E_preds, all_E_targets)).item()
        val_energy_rmse = self.dataset.restore_energy(val_energy_rmse)
        
        val_energy_rmse_avg = torch.sqrt(F.mse_loss(all_E_avg_preds, all_E_avg_targets)).item()
        val_energy_rmse_avg = self.dataset.restore_energy(val_energy_rmse_avg)
        
        val_energy_mae_avg = F.l1_loss(all_E_avg_preds, all_E_avg_targets).item()
        val_energy_mae_avg = self.dataset.restore_energy(val_energy_mae_avg)
        
        val_energy_loss = F.mse_loss(all_E_avg_preds, all_E_avg_targets).item()

        # Force (Flatten to calculate metrics)
        val_force_rmse = torch.sqrt(F.mse_loss(all_F_preds.view(-1), all_F_targets.view(-1))).item()
        val_force_mae = F.l1_loss(all_F_preds.view(-1), all_F_targets.view(-1)).item()
        val_phys_loss = float(sum(val_phys_loss_list) / max(1, len(val_phys_loss_list))) if val_phys_loss_list else 0.0

        phys_log_line = f"""
                        "Phys Loss Val (weighted)": {val_phys_loss}""" if val_phys_loss_list else ""
        logging.info(f"""
                        "Energy Loss Val (MSE)": {val_energy_loss},
                        "Energy RMSE Val": {val_energy_rmse},
                        "Energy RMSE avg Val": {val_energy_rmse_avg},
                        "Energy MAE avg Val": {val_energy_mae_avg},
                        "Force MAE Val": {val_force_mae},
                        "Force RMSE Val":  {val_force_rmse},{phys_log_line}
                        """)
        
        # Save results
        loss_out = [{
            "Energy_RMSE_test": val_energy_rmse,
            "Energy_RMSE_avg_test": val_energy_rmse_avg,
            "Force_RMSE_test": val_force_rmse,
        }]
        
        loss_out_df = pd.DataFrame(loss_out)
        loss_out_df.to_csv(f"{output_prefix}_loss.csv", index=False)
        
        # Save energy (one molecule per row)
        df_energy_val = pd.DataFrame({
            "Target_Energy": all_E_targets.numpy(),
            "Predicted_Energy": all_E_preds.numpy(),
            "Delta": (all_E_preds - all_E_targets).numpy()
        })
        energy_save_path = f"{output_prefix}_energy.csv"
        df_energy_val.to_csv(energy_save_path, index=False)
        
        # Save force (one atom per row, includes x, y, z components)
        # Use numpy slicing to separate components
        f_pred_np = all_F_preds.numpy()
        f_true_np = all_F_targets.numpy()
        
        df_force_val = pd.DataFrame({
            "Fx_True": f_true_np[:, 0], "Fy_True": f_true_np[:, 1], "Fz_True": f_true_np[:, 2],
            "Fx_Pred": f_pred_np[:, 0], "Fy_Pred": f_pred_np[:, 1], "Fz_Pred": f_pred_np[:, 2]
        })
        force_save_path = f"{output_prefix}_force.csv"
        df_force_val.to_csv(force_save_path, index=False)
        
        logging.info(f"Saved test details to {energy_save_path} and {force_save_path}")
        
        return {
            'energy_rmse': val_energy_rmse,
            'energy_rmse_avg': val_energy_rmse_avg,
            'energy_mae_avg': val_energy_mae_avg,
            'energy_loss': val_energy_loss,
            'force_rmse': val_force_rmse,
            'force_mae': val_force_mae,
            'phys_loss': val_phys_loss,
        }
    
    def compute_hessian(self, pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell):
        """
        Compute Hessian matrix (second derivative of energy with respect to positions).
        
        Args:
            pos: Atomic positions [N_atoms, 3]
            A: Atomic numbers [N_atoms]
            batch_idx: Batch indices [N_atoms]
            edge_src: Edge source indices
            edge_dst: Edge destination indices
            edge_shifts: Edge shifts
            cell: Unit cell tensor
            
        Returns:
            hessian: Hessian matrix [N_atoms * 3, N_atoms * 3]
        """
        self.model.eval()
        
        pos = pos.to(self.device)
        A = A.to(self.device)
        batch_idx = batch_idx.to(self.device)
        edge_src = edge_src.to(self.device)
        edge_dst = edge_dst.to(self.device)
        edge_shifts = edge_shifts.to(self.device)
        cell = cell.to(self.device)
        
        pos.requires_grad_(True)
        
        n_atoms = pos.shape[0]
        hessian = torch.zeros(n_atoms * 3, n_atoms * 3, dtype=pos.dtype, device=self.device)
        
        # Compute energy
        mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
        E_offset = scatter(mapped_A, batch_idx, dim=0, reduce='sum')
        
        E_per_atom = self.model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
        E_conv = scatter(E_per_atom, batch_idx, dim=0, reduce='sum')
        E_total = E_conv.view(-1) + E_offset
        
        # Compute forces (first derivatives)
        forces = -torch.autograd.grad(
            E_total.sum(), pos, create_graph=True, retain_graph=True
        )[0]  # [N_atoms, 3]
        
        # Compute Hessian by differentiating forces
        forces_flat = forces.view(-1)  # [N_atoms * 3]
        
        logging.info(f"Computing Hessian matrix ({n_atoms * 3} x {n_atoms * 3})...")
        for i in range(n_atoms * 3):
            if (i + 1) % 10 == 0 or i == 0:
                logging.info(f"  Computing column {i + 1}/{n_atoms * 3}")
            
            # Compute gradient of force[i] with respect to all positions
            # Only retain graph if not the last iteration
            retain = (i < n_atoms * 3 - 1)
            grad_force_i = torch.autograd.grad(
                forces_flat[i],
                pos,
                retain_graph=retain,
                create_graph=False
            )[0]  # [N_atoms, 3]
            
            # Fill Hessian column
            hessian[:, i] = grad_force_i.view(-1)
        
        return hessian
    
    def compute_phonon_spectrum(self, hessian, masses, output_prefix='phonon'):
        """
        Compute phonon frequencies from Hessian matrix.
        
        Args:
            hessian: Hessian matrix [N_atoms * 3, N_atoms * 3] in eV/Å²
            masses: Atomic masses [N_atoms] in amu
            output_prefix: Prefix for output files
            
        Returns:
            frequencies: Phonon frequencies in cm⁻¹
        """
        import numpy as np
        try:
            from scipy.linalg import eigh
        except ImportError:
            raise ImportError("scipy is required for phonon calculation. Install it with: pip install scipy")
        
        hessian_np = hessian.detach().cpu().numpy()
        masses_np = masses.numpy() if isinstance(masses, torch.Tensor) else np.array(masses)
        
        n_atoms = len(masses_np)
        
        # Convert units: eV/Å² to J/m², then to dynamical matrix
        # 1 eV = 1.602176634e-19 J
        # 1 Å = 1e-10 m
        # So 1 eV/Å² = 1.602176634e-19 / (1e-10)² = 1.602176634e1 J/m²
        ev_to_j_per_m2 = 1.602176634e1
        
        # Mass in kg: 1 amu = 1.66053906660e-27 kg
        amu_to_kg = 1.66053906660e-27
        
        # Build mass matrix
        mass_matrix = np.zeros((n_atoms * 3, n_atoms * 3))
        for i in range(n_atoms):
            mass_val = masses_np[i] * amu_to_kg
            mass_matrix[3*i:3*i+3, 3*i:3*i+3] = np.eye(3) * mass_val
        
        # Compute dynamical matrix: D = M^(-1/2) * H * M^(-1/2)
        mass_inv_sqrt = np.linalg.inv(np.sqrt(mass_matrix))
        dynamical_matrix = mass_inv_sqrt @ (hessian_np * ev_to_j_per_m2) @ mass_inv_sqrt
        
        # Diagonalize dynamical matrix
        eigenvalues, eigenvectors = eigh(dynamical_matrix)
        
        # Convert eigenvalues to frequencies
        # ω² = λ, so ω = sqrt(λ)
        # Frequency in Hz: ω = sqrt(λ) rad/s
        # Frequency in cm⁻¹: ν = ω / (2π * c) where c = 2.99792458e10 cm/s
        c_cm_per_s = 2.99792458e10  # Speed of light in cm/s
        
        # Handle negative eigenvalues (unstable modes)
        frequencies_cm1 = np.zeros_like(eigenvalues)
        for i, eigval in enumerate(eigenvalues):
            if eigval >= 0:
                omega_rad_per_s = np.sqrt(eigval)
                frequencies_cm1[i] = omega_rad_per_s / (2 * np.pi * c_cm_per_s)
            else:
                # Imaginary frequency (unstable mode)
                omega_rad_per_s = np.sqrt(-eigval)
                frequencies_cm1[i] = -omega_rad_per_s / (2 * np.pi * c_cm_per_s)
        
        # Sort frequencies
        sorted_indices = np.argsort(frequencies_cm1)
        frequencies_sorted = frequencies_cm1[sorted_indices]
        
        # Save results
        np.save(f"{output_prefix}_hessian.npy", hessian_np)
        
        with open(f"{output_prefix}_frequencies.txt", 'w') as f:
            f.write("# Phonon frequencies (cm⁻¹)\n")
            f.write("# Negative values indicate imaginary frequencies (unstable modes)\n")
            f.write("# Index    Frequency (cm⁻¹)\n")
            for idx, freq in enumerate(frequencies_sorted):
                f.write(f"{idx:6d}    {freq:12.6f}\n")
        
        logging.info(f"Phonon calculation completed!")
        logging.info(f"  Total modes: {len(frequencies_sorted)}")
        logging.info(f"  Real modes: {np.sum(frequencies_sorted >= 0)}")
        logging.info(f"  Imaginary modes: {np.sum(frequencies_sorted < 0)}")
        logging.info(f"  Frequency range: {frequencies_sorted[frequencies_sorted >= 0].min():.2f} to {frequencies_sorted[frequencies_sorted >= 0].max():.2f} cm⁻¹")
        logging.info(f"  Saved Hessian to: {output_prefix}_hessian.npy")
        logging.info(f"  Saved frequencies to: {output_prefix}_frequencies.txt")
        
        return frequencies_sorted