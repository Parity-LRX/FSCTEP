"""Dataset classes for molecular modeling."""

import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any


def _load_struct_property_file(file_path: str, *, kind: str) -> torch.Tensor:
    """
    Load per-structure (graph-level) properties from an external file.

    Supported formats:
      - .npy/.npz: numpy array
      - pandas HDF (.h5/.hdf/.hdf5): DataFrame (values are used)

    kind:
      - "scalar": (B,)
      - "vector3": (B, 3)
      - "mat33": (B, 3, 3) or (B, 9) row-major
    """
    import os

    if file_path is None:
        raise ValueError("file_path must not be None")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Property file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".npy", ".npz"):
        arr = np.load(file_path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # Choose a reasonable default key
            keys = list(arr.keys())
            if not keys:
                raise ValueError(f"Empty npz file: {file_path}")
            values = arr[keys[0]]
        else:
            values = arr
    else:
        df = pd.read_hdf(file_path)
        values = df.values

    values = np.asarray(values)
    if kind == "scalar":
        if values.ndim == 2 and values.shape[1] >= 1:
            values = values[:, 0]
        values = values.reshape(-1).astype(np.float64)
        return torch.from_numpy(values).double()

    if kind == "vector3":
        if values.ndim == 1 and values.shape[0] == 3:
            values = values.reshape(1, 3)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"Expected vector3 shape (B,3), got {values.shape} from {file_path}")
        return torch.from_numpy(values.astype(np.float64)).double()

    if kind == "mat33":
        if values.ndim == 2 and values.shape[1] == 9:
            values = values.reshape(-1, 3, 3)
        if values.ndim != 3 or values.shape[1:] != (3, 3):
            raise ValueError(f"Expected mat33 shape (B,3,3) or (B,9), got {values.shape} from {file_path}")
        return torch.from_numpy(values.astype(np.float64)).double()

    raise ValueError(f"Unknown kind={kind!r}")


def _split_cell_and_pbc_from_cell_df(cell_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Support both legacy cell_{prefix}.h5 formats and the newer format with pbc_x/pbc_y/pbc_z columns.

    Returns:
        cells_all: np.ndarray shape [N, 3, 3] float64
        pbcs_all: np.ndarray shape [N, 3] bool, or None if not present
    """
    cols = list(cell_df.columns)
    has_named_cell = all(c in cols for c in ['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz'])
    has_pbc = all(c in cols for c in ['pbc_x', 'pbc_y', 'pbc_z'])

    if has_named_cell:
        cell_mat = cell_df[['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']].values.astype(np.float64)
    else:
        # legacy: assume first 9 columns represent the 3x3 cell (row-major)
        cell_mat = cell_df.iloc[:, :9].values.astype(np.float64)
    cells_all = cell_mat.reshape(-1, 3, 3)

    if has_pbc:
        pbcs_all = cell_df[['pbc_x', 'pbc_y', 'pbc_z']].values.astype(bool)
    else:
        pbcs_all = None

    return cells_all, pbcs_all


def compute_graph_worker(args):
    """
    Worker function for computing graph structure in parallel.
    
    Args:
        args: Tuple of (idx, block, target, cell, pbc, max_radius[, stress])
        
    Returns:
        Dictionary with precomputed graph data
    """
    extras: Dict[str, Any] | None = None
    if len(args) == 8:
        idx, block, target, cell, pbc, max_radius, stress, extras = args
    elif len(args) == 7:
        idx, block, target, cell, pbc, max_radius, stress = args
    else:
        idx, block, target, cell, pbc, max_radius = args
        stress = np.zeros((3, 3), dtype=np.float64)
    
    # Extract coordinates and atom types (numpy)
    pos = block[:, 1:4].numpy()
    atom_types = block[:, 4].numpy()
    
    # Determine periodicity (prefer explicit pbc if provided)
    if pbc is None:
        is_periodic = (np.abs(cell).sum() > 1e-5)
        pbc_flags = [True, True, True] if is_periodic else [False, False, False]
    else:
        pbc_flags = [bool(pbc[0]), bool(pbc[1]), bool(pbc[2])]
        is_periodic = any(pbc_flags)

    if is_periodic:
        # Safety net: if lattice is missing/zero, use a large dummy cell to keep ASE happy
        current_cell = cell if (np.abs(cell).sum() > 1e-9) else (np.eye(3) * 100.0)
    else:
        pbc_flags = [False, False, False]
        current_cell = np.eye(3) * 100.0  # Virtual large box
        
    i, j, S = matscipy_neighbour_list(
        'ijS', positions=pos, cell=current_cell, pbc=pbc_flags, cutoff=max_radius
    )
    
    # Return cache dictionary (all converted to Tensor)
    out = {
        'read_tensor': block,
        'target_energy': target,
        'edge_src': torch.tensor(i, dtype=torch.long),
        'edge_dst': torch.tensor(j, dtype=torch.long),
        'edge_shifts': torch.tensor(S, dtype=torch.float64),
        'cell': torch.tensor(current_cell, dtype=torch.float64),
        'stress': torch.tensor(stress, dtype=torch.float64)
    }
    if extras:
        # Keep extras as plain tensors (already on CPU)
        out.update(extras)
    return out


class CustomDataset(Dataset):
    """Custom dataset with precomputed graph structures."""
    
    def __init__(
        self,
        read_file_path,
        energy_file_path,
        cell_file_path,
        stress_file_path=None,
        *,
        max_radius=5.0,
        num_workers=10,
        # Optional per-structure Cartesian labels / global tensors
        charge_file_path: str | None = None,            # scalar
        dipole_file_path: str | None = None,            # vector3
        polarizability_file_path: str | None = None,    # mat33
        quadrupole_file_path: str | None = None,        # mat33 (typically traceless, but stored Cartesian)
        external_field_file_path: str | None = None,    # vector3 (e.g., uniform E field)
    ):
        """
        Initialize custom dataset.
        
        Args:
            read_file_path: Path to read HDF5 file
            energy_file_path: Path to energy HDF5 file
            cell_file_path: Path to cell HDF5 file
            stress_file_path: Path to stress HDF5 file (optional)
            max_radius: Maximum radius for neighbor search
            num_workers: Number of worker processes for preprocessing
        """
        print(f"Loading data from {read_file_path}...")
        self.max_radius = max_radius
        
        # 1. Read base HDF5 data
        read_data = pd.read_hdf(read_file_path)
        energy_df = pd.read_hdf(energy_file_path)
        cell_df = pd.read_hdf(cell_file_path)

        # Optional: load extra per-structure tensors
        charges_all = _load_struct_property_file(charge_file_path, kind="scalar") if charge_file_path else None
        dipoles_all = _load_struct_property_file(dipole_file_path, kind="vector3") if dipole_file_path else None
        polars_all = _load_struct_property_file(polarizability_file_path, kind="mat33") if polarizability_file_path else None
        quads_all = _load_struct_property_file(quadrupole_file_path, kind="mat33") if quadrupole_file_path else None
        ext_fields_all = _load_struct_property_file(external_field_file_path, kind="vector3") if external_field_file_path else None
        
        # Load stress data (optional)
        import os
        if stress_file_path is not None and os.path.exists(stress_file_path):
            stress_df = pd.read_hdf(stress_file_path)
            stresses_all = stress_df.values.astype(np.float64).reshape(-1, 3, 3)
        else:
            stresses_all = None
        
        # 2. Process energy and Cell
        if 'RawEnergy' in energy_df.columns:
            targets = torch.tensor(energy_df['RawEnergy'].values, dtype=torch.float64)
        else:
            targets = torch.tensor(energy_df.iloc[:, 0].values, dtype=torch.float64)

        cells, pbcs = _split_cell_and_pbc_from_cell_df(cell_df)
        
        # 3. Vectorized block splitting
        values = read_data.values
        stop_value = 128128.0
        is_separator = (values == stop_value).any(axis=1)
        group_ids = is_separator.cumsum()
        clean_mask = ~is_separator
        
        clean_values = values[clean_mask]
        clean_group_ids = group_ids[clean_mask]
        _, unique_indices = np.unique(clean_group_ids, return_index=True)
        
        # Original block list
        blocks = [
            torch.tensor(block, dtype=torch.float64)
            for block in np.split(clean_values, unique_indices[1:])
        ]

        # 4. Multi-process precomputation of ASE neighbor lists
        print(f"Pre-calculating ASE neighbor lists using {num_workers} workers...")
        self.cache = []
        
        # Prepare parallel task parameters
        tasks = []
        for i in range(len(blocks)):
            pbc_i = pbcs[i] if pbcs is not None and i < len(pbcs) else None
            stress_i = stresses_all[i] if stresses_all is not None and i < len(stresses_all) else np.zeros((3, 3), dtype=np.float64)
            extras = {}
            if charges_all is not None:
                extras["charge"] = charges_all[i]
            if dipoles_all is not None:
                extras["dipole"] = dipoles_all[i]
            if polars_all is not None:
                extras["polarizability"] = polars_all[i]
            if quads_all is not None:
                extras["quadrupole"] = quads_all[i]
            if ext_fields_all is not None:
                extras["external_field"] = ext_fields_all[i]
            tasks.append((i, blocks[i], targets[i], cells[i], pbc_i, self.max_radius, stress_i, extras))
        
        # Use process pool for parallel computation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(compute_graph_worker, tasks, chunksize=20),
                total=len(tasks),
                ascii=True,
                dynamic_ncols=True,
                desc="Pre-calculating ASE"
            ))
        
        self.cache = results
        print(f"Pre-calculation complete. Cached {len(self.cache)} structures in RAM.")
    
    def restore_energy(self, normalized_energy):
        """Restore energy units (if normalized)."""
        return normalized_energy
    
    def restore_force(self, normalized_force):
        """Restore force units (if normalized)."""
        return normalized_force

    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        # Return precomputed results directly from memory list
        d = self.cache[idx]
        extras = {k: v for k, v in d.items() if k not in {
            "read_tensor", "target_energy", "edge_src", "edge_dst", "edge_shifts", "cell", "stress"
        }}
        return (
            d['read_tensor'],
            d['target_energy'],
            d['edge_src'],
            d['edge_dst'],
            d['edge_shifts'],
            d['cell'],
            d['stress'],
            extras,
        )


class H5Dataset(Dataset):
    """Dataset loading from preprocessed HDF5 files."""
    
    def __init__(
        self,
        prefix,
        data_dir='.',
        file_path=None,
        *,
        # Optional: external per-structure label files (Cartesian)
        extra_label_paths: Dict[str, str] | None = None,
        # Optional: per-node label file (HDF5 with sample_0, sample_1, ... each with charge_per_atom etc.)
        extra_per_node_label_path: str | None = None,
    ):
        """
        Initialize H5 dataset.
        
        Args:
            prefix: Prefix for HDF5 files (e.g., 'train' or 'val'), used when file_path is None
            data_dir: Directory containing the HDF5 files (default: current directory), used when file_path is None
            file_path: Optional full path to processed H5 file. If given, use this file and ignore prefix/data_dir
        """
        import os
        if file_path is not None:
            self.file_path = os.path.abspath(file_path)
        else:
            self.file_path = os.path.join(data_dir, f'processed_{prefix}.h5')
        self._h5_file = None
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Preprocessed data file not found: {self.file_path}\n"
                f"Please run 'mff-preprocess' first to generate the required data files."
            )
        
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys())

        self._extra_labels: Dict[str, torch.Tensor] = {}
        if extra_label_paths:
            for name, path in extra_label_paths.items():
                key = str(name)
                p = str(path)
                if key in ("charge",):
                    self._extra_labels[key] = _load_struct_property_file(p, kind="scalar")
                elif key in ("dipole", "external_field"):
                    self._extra_labels[key] = _load_struct_property_file(p, kind="vector3")
                elif key in ("polarizability", "quadrupole"):
                    self._extra_labels[key] = _load_struct_property_file(p, kind="mat33")
                else:
                    raise ValueError(f"Unknown extra label name {key!r}; supported: charge, dipole, polarizability, quadrupole, external_field")
            # Basic length validation (best-effort)
            for k, v in self._extra_labels.items():
                if v.shape[0] != self.num_samples:
                    raise ValueError(f"Extra label {k!r} length {v.shape[0]} != num_samples {self.num_samples}")

        self._extra_per_node: Dict[str, list] = {}
        if extra_per_node_label_path and os.path.exists(extra_per_node_label_path):
            with h5py.File(extra_per_node_label_path, 'r') as f:
                for i in range(self.num_samples):
                    key = f'sample_{i}'
                    if key not in f:
                        continue
                    g = f[key]
                    for k in ("charge_per_atom", "dipole_per_atom", "polarizability_per_atom", "quadrupole_per_atom"):
                        if k in g:
                            self._extra_per_node.setdefault(k, [None] * self.num_samples)[i] = torch.from_numpy(g[k][:]).double()
    
    def restore_energy(self, normalized_energy):
        """Restore energy units (if normalized)."""
        return normalized_energy
    
    def restore_force(self, normalized_force):
        """Restore force units (if normalized)."""
        return normalized_force

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self._h5_file is None:  # Multi-process safe initialization
            self._h5_file = h5py.File(self.file_path, 'r')
        
        g = self._h5_file[f'sample_{idx}']
        if 'stress' in g:
            stress = torch.from_numpy(g['stress'][:]).double()
        else:
            stress = torch.zeros((3, 3), dtype=torch.float64)
        out = {
            'pos': torch.from_numpy(g['pos'][:]).double(),
            'A': torch.from_numpy(g['A'][:]).long(),
            'y': torch.from_numpy(np.array([g['y'][()]])).double(),
            'force': torch.from_numpy(g['force'][:]).double(),
            'edge_src': torch.from_numpy(g['edge_src'][:]).long(),
            'edge_dst': torch.from_numpy(g['edge_dst'][:]).long(),
            'edge_shifts': torch.from_numpy(g['edge_shifts'][:]).double(),
            'cell': torch.from_numpy(g['cell'][:]).double(),
            'stress': stress,
        }
        # Prefer in-H5 datasets when present, else fall back to extra_label_paths.
        for k in ("charge", "dipole", "polarizability", "quadrupole", "external_field"):
            if k in g:
                out[k] = torch.from_numpy(g[k][:]).double()
            elif k in self._extra_labels:
                out[k] = self._extra_labels[k][idx]
        # Per-node labels (reduce="none"): shape (num_atoms,) or (num_atoms, 3) or (num_atoms, 3, 3)
        for k in ("charge_per_atom", "dipole_per_atom", "polarizability_per_atom", "quadrupole_per_atom"):
            if k in g:
                out[k] = torch.from_numpy(g[k][:]).double()
            elif hasattr(self, "_extra_per_node") and k in self._extra_per_node:
                out[k] = self._extra_per_node[k][idx]
        return out


class OnTheFlyDataset(Dataset):
    """Dataset that computes graph structure on-the-fly during loading."""
    
    def __init__(
        self,
        read_file_path,
        energy_file_path,
        cell_file_path,
        stress_file_path=None,
        *,
        max_radius=5.0,
        # Optional per-structure Cartesian labels / global tensors
        charge_file_path: str | None = None,            # scalar
        dipole_file_path: str | None = None,            # vector3
        polarizability_file_path: str | None = None,    # mat33
        quadrupole_file_path: str | None = None,        # mat33
        external_field_file_path: str | None = None,    # vector3
    ):
        """
        Initialize on-the-fly dataset.
        
        Args:
            read_file_path: Path to read HDF5 file
            energy_file_path: Path to energy HDF5 file
            cell_file_path: Path to cell HDF5 file
            stress_file_path: Path to stress HDF5 file (optional)
            max_radius: Maximum radius for neighbor search
        """
        print(f"Loading raw data indices from {read_file_path}...")
        self.max_radius = max_radius
        
        # 1. Only read raw data to memory (very fast, a few seconds)
        # No ASE computation
        self.read_data = pd.read_hdf(read_file_path)
        self.energy_df = pd.read_hdf(energy_file_path)
        self.cell_df = pd.read_hdf(cell_file_path)

        # Optional: load extra per-structure tensors
        self.charges_all = _load_struct_property_file(charge_file_path, kind="scalar") if charge_file_path else None
        self.dipoles_all = _load_struct_property_file(dipole_file_path, kind="vector3") if dipole_file_path else None
        self.polars_all = _load_struct_property_file(polarizability_file_path, kind="mat33") if polarizability_file_path else None
        self.quads_all = _load_struct_property_file(quadrupole_file_path, kind="mat33") if quadrupole_file_path else None
        self.ext_fields_all = _load_struct_property_file(external_field_file_path, kind="vector3") if external_field_file_path else None
        
        # Load stress data (optional)
        import os
        if stress_file_path is not None and os.path.exists(stress_file_path):
            stress_df = pd.read_hdf(stress_file_path)
            self.stresses = stress_df.values.astype(np.float64).reshape(-1, 3, 3)
        else:
            self.stresses = None
        
        # 2. Prepare data indices
        if 'RawEnergy' in self.energy_df.columns:
            self.targets = self.energy_df['RawEnergy'].values
        else:
            self.targets = self.energy_df.iloc[:, 0].values

        self.cells, self.pbcs = _split_cell_and_pbc_from_cell_df(self.cell_df)
        
        # 3. Vectorized block splitting (this step is fast)
        values = self.read_data.values
        stop_value = 128128.0
        is_separator = (values == stop_value).any(axis=1)
        group_ids = is_separator.cumsum()
        clean_mask = ~is_separator
        _, unique_indices = np.unique(group_ids[clean_mask], return_index=True)
        self.blocks = np.split(values[clean_mask], unique_indices[1:])

    def restore_energy(self, x):
        return x
    
    def restore_force(self, x):
        return x

    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        # Core of on-the-fly computation
        
        # 1. Get Numpy data
        block = self.blocks[idx]
        pos = block[:, 1:4]
        atom_types = block[:, 4]
        forces = block[:, 5:8]
        target = self.targets[idx]
        cell = self.cells[idx]
        
        # 2. Determine PBC (prefer explicit pbc flags if present)
        pbc_i = None
        if getattr(self, "pbcs", None) is not None and idx < len(self.pbcs):
            pbc_i = self.pbcs[idx]

        if pbc_i is None:
            is_periodic = (np.abs(cell).sum() > 1e-5)
            pbc_flags = [True, True, True] if is_periodic else [False, False, False]
        else:
            pbc_flags = [bool(pbc_i[0]), bool(pbc_i[1]), bool(pbc_i[2])]
            is_periodic = any(pbc_flags)

        if is_periodic:
            current_cell = cell if (np.abs(cell).sum() > 1e-9) else (np.eye(3) * 100.0)
        else:
            pbc_flags = [False, False, False]
            current_cell = np.eye(3) * 100.0
        
        i, j, S = matscipy_neighbour_list(
            'ijS', positions=pos, cell=current_cell, pbc=pbc_flags, cutoff=self.max_radius
        )
        
        # 4. Get stress
        if self.stresses is not None and idx < len(self.stresses):
            stress = self.stresses[idx]
        else:
            stress = np.zeros((3, 3), dtype=np.float64)

        # 5. Convert to Tensor and return
        out = {
            'pos': torch.tensor(pos, dtype=torch.float64),
            'A': torch.tensor(atom_types, dtype=torch.float64),
            'force': torch.tensor(forces, dtype=torch.float64),
            'target': torch.tensor(target, dtype=torch.float64),
            'edge_src': torch.tensor(i, dtype=torch.long),
            'edge_dst': torch.tensor(j, dtype=torch.long),
            'edge_shifts': torch.tensor(S, dtype=torch.float64),
            'cell': torch.tensor(current_cell, dtype=torch.float64),
            'stress': torch.tensor(stress, dtype=torch.float64),
        }
        if self.charges_all is not None:
            out["charge"] = self.charges_all[idx]
        if self.dipoles_all is not None:
            out["dipole"] = self.dipoles_all[idx]
        if self.polars_all is not None:
            out["polarizability"] = self.polars_all[idx]
        if self.quads_all is not None:
            out["quadrupole"] = self.quads_all[idx]
        if self.ext_fields_all is not None:
            out["external_field"] = self.ext_fields_all[idx]
        return out