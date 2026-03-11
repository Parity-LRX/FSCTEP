"""LAMMPS Python interface for FusedEquiTensorPot model.

This module provides a Python interface that can be called from LAMMPS
using the `fix python/invoke` or `python` command.

Usage in LAMMPS:
    python lammps_potential.py input
    fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential lammps_potential
    pair_style python 1
    pair_coeff * *

Or using the simpler python command:
    python lammps_potential.py input
    pair_style python 1 lammps_potential.py lammps_potential lammps_potential
    pair_coeff * *
"""

import sys
import os

# 在 import torch 之前设好线程限制，防止 LAMMPS 回调中多线程死锁
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import torch
# 必须在首次 torch 并行操作之前调用一次，之后不可重复调用（否则可能死锁）
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # 已经设置过或已有并行操作

import numpy as np
from typing import Dict, Tuple, Optional, Mapping, Any

# Add parent directory to path to import molecular_force_field modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu
from molecular_force_field.utils.tensor_utils import map_tensor_values

# 小体系 CPU 下用纯 PyTorch 邻居列表，避免 LAMMPS 回调中 torch_cluster 可能导致的死锁
_MAX_ATOMS_CPU_SIMPLE = 512


def _radius_graph_pbc_cpu_simple(pos: torch.Tensor, r: float, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """纯 PyTorch PBC 邻居列表，仅用于 CPU 且原子数较小时。"""
    cell_mat = cell.squeeze(0) if cell.dim() == 3 else cell
    n = pos.size(0)
    device = pos.device
    r2 = r * r
    all_src, all_dst, all_shifts = [], [], []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                s = torch.tensor([sx, sy, sz], dtype=torch.float64, device=device)
                shift = (s.unsqueeze(0) @ cell_mat).squeeze(0)
                pos_shifted = pos + shift
                for i in range(n):
                    d = pos_shifted - pos[i : i + 1]
                    dist2 = (d * d).sum(dim=1)
                    for j in range(n):
                        if sx == 0 and sy == 0 and sz == 0 and i == j:
                            continue
                        if dist2[j].item() <= r2 and dist2[j].item() > 1e-20:
                            all_src.append(i)
                            all_dst.append(j)
                            all_shifts.append([sx, sy, sz])
    if not all_src:
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, 3, device=device, dtype=torch.float64),
        )
    return (
        torch.tensor(all_src, device=device, dtype=torch.long),
        torch.tensor(all_dst, device=device, dtype=torch.long),
        torch.tensor(all_shifts, device=device, dtype=torch.float64),
    )


class LAMMPSPotential:
    """LAMMPS Python potential interface for FusedEquiTensorPot."""
    
    def __init__(self, checkpoint_path: str, config: Optional[ModelConfig] = None,
                 device: str = 'cuda', max_radius: float = 5.0,
                 atomic_energy_file: Optional[str] = None,
                 atomic_energy_keys: Optional[list] = None,
                 atomic_energy_values: Optional[list] = None,
                 embed_size: Optional[list] = None,
                 output_size: int = 8,
                 type_to_Z: Optional[Mapping[int, int]] = None):
        """
        Initialize LAMMPS potential calculator.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            config: ModelConfig object (if None, will be created from checkpoint or defaults)
            device: Device to use ('cuda' or 'cpu')
            max_radius: Maximum radius for neighbor search (Angstrom)
            atomic_energy_file: Path to CSV file with atomic energies (fitted_E0.csv)
            atomic_energy_keys: List of atomic numbers for custom E0
            atomic_energy_values: List of atomic energies (eV) corresponding to keys
            embed_size: Hidden layer sizes for readout MLP
            output_size: Output size for atom readout MLP
            type_to_Z: Optional mapping from LAMMPS atom type -> atomic number Z.
                IMPORTANT: In LAMMPS, `type` is usually just a category label (1..Ntypes),
                not the atomic number. If you don't provide this mapping, this interface
                assumes `type == Z`, which can silently produce incorrect energies/forces.
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.max_radius = max_radius
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize config
        if config is None:
            # Try to infer dtype from checkpoint or use float64
            dtype = checkpoint.get('dtype', torch.float64)
            if isinstance(dtype, str):
                dtype = torch.float64 if dtype in ['float64', 'double'] else torch.float32
            
            config = ModelConfig(dtype=dtype, embed_size=embed_size, output_size=output_size)
        
        # Load atomic energies
        if atomic_energy_keys is not None and atomic_energy_values is not None:
            config.atomic_energy_keys = torch.tensor(atomic_energy_keys, dtype=torch.long)
            config.atomic_energy_values = torch.tensor(atomic_energy_values, dtype=config.dtype)
        elif atomic_energy_file is not None:
            config.load_atomic_energies_from_file(atomic_energy_file)
        else:
            # Try default path
            config.load_atomic_energies_from_file('fitted_E0.csv')
        
        # Initialize model
        self.model = E3_TransformerLayer_multi(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            function_type_main=config.function_type,
            device=self.device
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['e3trans_state_dict'])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Atomic energy mapping
        self.atomic_energy_keys = config.atomic_energy_keys.to(self.device)
        self.atomic_energy_values = config.atomic_energy_values.to(self.device)

        # LAMMPS type -> atomic number Z mapping
        # Default: identity mapping (type == Z), which is only correct if user sets types as Z.
        self.type_to_Z: Dict[int, int] = {int(k): int(v) for k, v in (type_to_Z or {}).items()}
        
        # Unit conversion: LAMMPS uses kcal/mol and Angstrom
        # Our model uses eV and Angstrom, so we need to convert
        # 1 eV = 23.06035 kcal/mol
        self.ev_to_kcalmol = 23.06035
        
        print(f"LAMMPS Potential initialized on {self.device}")
        print(f"  Max radius: {self.max_radius} Angstrom")
        print(f"  Atomic energies loaded for: {self.atomic_energy_keys.cpu().tolist()}")
        if self.type_to_Z:
            print(f"  Using LAMMPS type->Z mapping: {self.type_to_Z}")
        else:
            print("  WARNING: No type->Z mapping provided; assuming LAMMPS type == atomic number Z.")
    
    def compute(self, nlocal: int, nall: int, tag: np.ndarray,
                type_array: np.ndarray, x: np.ndarray,
                boxlo: np.ndarray, boxhi: np.ndarray,
                pbc: np.ndarray,
                xy: float = 0.0, xz: float = 0.0, yz: float = 0.0) -> Tuple[float, np.ndarray]:
        """
        Compute energy and forces for LAMMPS.
        
        Args:
            nlocal: Number of local atoms
            nall: Total number of atoms (including ghosts)
            tag: Atom tags (1-indexed)
            type_array: Atom types (1-indexed, LAMMPS convention)
            x: Atomic positions [nall, 3] in Angstrom
            boxlo: Lower box bounds [3]
            boxhi: Upper box bounds [3]
            pbc: Periodic boundary conditions [3] (0=non-periodic, 1=periodic)
            xy/xz/yz: Triclinic tilt factors (LAMMPS convention). For orthogonal boxes, keep 0.
            
        Returns:
            Tuple of (energy in kcal/mol, forces in kcal/mol/Angstrom)
            Forces shape: [nlocal, 3]
        """
        # Multi-rank (MPI) correctness note:
        # LAMMPS provides nlocal (owned atoms) and nall (owned + ghost atoms).
        # For correct forces on owned atoms under domain decomposition, we must include
        # ghost atoms in neighbor construction and message passing. We therefore build
        # the graph on ALL atoms (nall) but only accumulate energy for owned atoms.
        #
        # This makes per-rank energies additive (LAMMPS will sum them), and forces on
        # owned atoms include interactions that go through ghost neighbors.

        # Convert to torch tensors (ALL atoms, including ghosts)
        pos_all = torch.tensor(x[:nall], dtype=torch.float64, device=self.device)
        lmp_types_all = torch.tensor(type_array[:nall], dtype=torch.long, device=self.device)

        # Map LAMMPS types -> atomic numbers Z (embedding expects integer indices)
        if self.type_to_Z:
            # Build a lookup table up to max type present (fallback identity if missing)
            max_type = int(lmp_types_all.max().item()) if lmp_types_all.numel() > 0 else 0
            lut = torch.arange(max_type + 1, device=self.device, dtype=torch.long)
            for t, z in self.type_to_Z.items():
                if 0 <= t <= max_type:
                    lut[t] = int(z)
            A_all = lut[lmp_types_all]
        else:
            # Identity: assume type == Z
            A_all = lmp_types_all
        
        # Build cell tensor from box
        # LAMMPS triclinic box vectors (common convention):
        #   a = (Lx, 0, 0)
        #   b = (xy, Ly, 0)
        #   c = (xz, yz, Lz)
        # where Lx = xhi-xlo, etc.
        box_size = boxhi - boxlo
        Lx, Ly, Lz = float(box_size[0]), float(box_size[1]), float(box_size[2])
        cell_np = np.array(
            [
                [Lx, 0.0, 0.0],
                [float(xy), Ly, 0.0],
                [float(xz), float(yz), Lz],
            ],
            dtype=np.float64,
        )
        cell = torch.tensor(cell_np, dtype=torch.float64, device=self.device).unsqueeze(0)
        
        # Handle PBC: if non-periodic, use large box (avoid neighbor issues)
        if not any(pbc):
            cell = torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0) * 100.0
        
        # Compute neighbor list（CPU 且原子数较小时用纯 PyTorch 实现，避免 LAMMPS 回调中 torch_cluster 死锁）
        if self.device.type == "cpu" and pos_all.size(0) <= _MAX_ATOMS_CPU_SIMPLE:
            edge_src, edge_dst, edge_shifts = _radius_graph_pbc_cpu_simple(
                pos_all, self.max_radius, cell
            )
        else:
            edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(
                pos_all, self.max_radius, cell, max_num_neighbors=100
            )
        
        # Batch index (all atoms in same batch for single structure)
        batch_idx = torch.zeros(len(pos_all), dtype=torch.long, device=self.device)
        
        # Compute atomic energy offset
        mapped_A_all = map_tensor_values(A_all, self.atomic_energy_keys, self.atomic_energy_values)
        E_offset_local = mapped_A_all[:nlocal].sum()
        
        # Forward pass
        pos_all.requires_grad_(True)
        atom_energies = self.model(
            pos_all, A_all, batch_idx,
            edge_src, edge_dst, edge_shifts, cell
        )
        # Sum energy over OWNED atoms only, to avoid double counting across MPI ranks.
        E_total_local = atom_energies[:nlocal].sum() + E_offset_local
        
        # Compute forces
        grads_all = torch.autograd.grad(E_total_local, pos_all, create_graph=False)[0]
        forces_all = -grads_all  # Negative gradient is force
        
        # Convert units: eV -> kcal/mol, eV/Ang -> kcal/mol/Ang
        energy_kcalmol = E_total_local.item() * self.ev_to_kcalmol
        forces_kcalmol_ang = forces_all[:nlocal].detach().cpu().numpy() * self.ev_to_kcalmol
        
        return energy_kcalmol, forces_kcalmol_ang


# Global instance (will be set by lammps_potential function)
_potential_instance: Optional[LAMMPSPotential] = None


def lammps_potential(nlocal: int, nall: int, tag: np.ndarray,
                     type_array: np.ndarray, x: np.ndarray,
                     boxlo: np.ndarray, boxhi: np.ndarray,
                     pbc: np.ndarray,
                     xy: float = 0.0, xz: float = 0.0, yz: float = 0.0,
                     **kwargs) -> Tuple[float, np.ndarray]:
    """
    LAMMPS Python potential function interface.
    
    This function is called by LAMMPS for each force calculation.
    
    Args:
        nlocal: Number of local atoms
        nall: Total number of atoms (including ghosts)
        tag: Atom tags (1-indexed)
        type_array: Atom types (1-indexed)
        x: Atomic positions [nall, 3] in Angstrom
        boxlo: Lower box bounds [3]
        boxhi: Upper box bounds [3]
        pbc: Periodic boundary conditions [3]
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Tuple of (energy, forces)
    """
    global _potential_instance
    
    if _potential_instance is None:
        raise RuntimeError(
            "LAMMPS potential not initialized. "
            "Call lammps_potential_init() first or set _potential_instance."
        )
    
    return _potential_instance.compute(
        nlocal, nall, tag, type_array, x, boxlo, boxhi, pbc, xy=xy, xz=xz, yz=yz
    )


def lammps_potential_init(checkpoint_path: str, device: str = 'cuda',
                          max_radius: float = 5.0,
                          atomic_energy_file: str = None,
                          atomic_energy_keys: list = None,
                          atomic_energy_values: list = None,
                          embed_size: list = None,
                          output_size: int = 8,
                          type_to_Z: dict = None,
                          **kwargs):
    """
    Initialize LAMMPS potential (called once at startup).
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to use ('cuda' or 'cpu')
        max_radius: Maximum radius for neighbor search (Angstrom)
        atomic_energy_file: Path to CSV file with atomic energies
        atomic_energy_keys: List of atomic numbers for custom E0
        atomic_energy_values: List of atomic energies (eV) corresponding to keys
        embed_size: Hidden layer sizes for readout MLP
        output_size: Output size for atom readout MLP
        type_to_Z: Optional mapping dict from LAMMPS atom type -> atomic number Z
        **kwargs: Additional arguments (ignored)
    """
    global _potential_instance
    
    _potential_instance = LAMMPSPotential(
        checkpoint_path,
        device=device,
        max_radius=max_radius,
        atomic_energy_file=atomic_energy_file,
        atomic_energy_keys=atomic_energy_keys,
        atomic_energy_values=atomic_energy_values,
        embed_size=embed_size,
        output_size=output_size,
        type_to_Z=type_to_Z
    )
    print(f"LAMMPS potential initialized with checkpoint: {checkpoint_path}")


# For direct testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LAMMPS potential interface')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max-radius', type=float, default=5.0, help='Max radius (Angstrom)')
    parser.add_argument('--atomic-energy-file', type=str, default=None,
                       help='Path to atomic energy CSV file')
    parser.add_argument('--embed-size', type=int, nargs='+', default=None,
                       help='Hidden layer sizes for readout MLP (default: 128 128 128)')
    parser.add_argument('--output-size', type=int, default=8,
                       help='Output size for atom readout MLP (default: 8)')
    
    args = parser.parse_args()
    
    # Initialize
    lammps_potential_init(
        args.checkpoint,
        device=args.device,
        max_radius=args.max_radius,
        atomic_energy_file=args.atomic_energy_file,
        embed_size=args.embed_size,
        output_size=args.output_size
    )
    
    # Test with dummy data
    nlocal = 10
    nall = 10
    tag = np.arange(1, nall + 1)
    type_array = np.ones(nall, dtype=np.int32)  # All type 1
    x = np.random.rand(nall, 3) * 10.0  # Random positions
    boxlo = np.array([0.0, 0.0, 0.0])
    boxhi = np.array([10.0, 10.0, 10.0])
    pbc = np.array([1, 1, 1])  # Periodic
    
    energy, forces = lammps_potential(nlocal, nall, tag, type_array, x, boxlo, boxhi, pbc)
    
    print(f"\nTest Results:")
    print(f"  Energy: {energy:.6f} kcal/mol")
    print(f"  Forces shape: {forces.shape}")
    print(f"  Force magnitude (first atom): {np.linalg.norm(forces[0]):.6f} kcal/mol/Ang")
