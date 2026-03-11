"""Data preprocessing functions for molecular modeling."""

import os
import re
import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from ase.data import chemical_symbols, atomic_numbers
from matscipy.neighbours import neighbour_list as matscipy_neighbour_list


def _parse_pbc_from_comment(line: str):
    """
    Parse pbc flags from extended XYZ comment line.
    Accepts: pbc="F F F", pbc="T T F", pbc="1 1 0", etc.
    Returns: tuple(bool,bool,bool) or None if not found/invalid.
    """
    m = re.search(r'\bpbc\s*=\s*["\']([^"\']+)["\']', line)
    if not m:
        return None
    tokens = m.group(1).strip().split()
    if len(tokens) < 3:
        return None

    def _to_bool(tok: str) -> bool:
        t = tok.strip().lower()
        if t in {"t", "true", "1", "y", "yes"}:
            return True
        if t in {"f", "false", "0", "n", "no"}:
            return False
        # Fallback: anything else is invalid
        raise ValueError(tok)

    try:
        return (_to_bool(tokens[0]), _to_bool(tokens[1]), _to_bool(tokens[2]))
    except Exception:
        return None


def _parse_lattice_from_comment(line: str):
    """
    Parse lattice cell from extended XYZ comment line.
    Accepts: Lattice="9 numbers"
    Returns: list[float] length 9, or None if missing/invalid.
    """
    m = re.search(r'\bLattice\s*=\s*["\']([^"\']+)["\']', line)
    if not m:
        return None
    parts = m.group(1).strip().split()
    if len(parts) != 9:
        return None
    try:
        return [np.float64(x) for x in parts]
    except Exception:
        return None


def _parse_energy_from_comment(line: str):
    """
    Parse energy from extended XYZ comment line.
    Accepts: energy=-29654.7, Energy=..., etc.
    Returns: np.float64 or None
    """
    m = re.search(r'\benergy\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)', line, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return np.float64(m.group(1))
    except Exception:
        return None


def _parse_stress_from_comment(line: str, cell_9: list = None):
    """
    Parse stress tensor from extended XYZ comment line.

    Supported formats:
      - stress="xx yy zz yz xz xy"  (6 Voigt components, eV/Å³)
      - stress="xx xy xz yx yy yz zx zy zz"  (9 full 3×3 components, row-major, eV/Å³)
      - virial="xx xy xz yx yy yz zx zy zz"  (9 full 3×3 virial in eV; converted via stress = -virial / V)

    Returns: np.ndarray shape (3, 3), or None if not found.
    """
    # Try stress first
    m = re.search(r'\bstress\s*=\s*["\']([^"\']+)["\']', line, flags=re.IGNORECASE)
    if m:
        tokens = m.group(1).strip().split()
        try:
            vals = [np.float64(t) for t in tokens]
        except Exception:
            return None

        if len(vals) == 9:
            return np.array(vals, dtype=np.float64).reshape(3, 3)
        elif len(vals) == 6:
            # Voigt: xx yy zz yz xz xy
            s = np.zeros((3, 3), dtype=np.float64)
            s[0, 0] = vals[0]
            s[1, 1] = vals[1]
            s[2, 2] = vals[2]
            s[1, 2] = s[2, 1] = vals[3]
            s[0, 2] = s[2, 0] = vals[4]
            s[0, 1] = s[1, 0] = vals[5]
            return s
        return None

    # Try virial (convert to stress = -virial / volume)
    m = re.search(r'\bvirial\s*=\s*["\']([^"\']+)["\']', line, flags=re.IGNORECASE)
    if m:
        tokens = m.group(1).strip().split()
        try:
            vals = [np.float64(t) for t in tokens]
        except Exception:
            return None

        if len(vals) == 9:
            virial = np.array(vals, dtype=np.float64).reshape(3, 3)
        elif len(vals) == 6:
            virial = np.zeros((3, 3), dtype=np.float64)
            virial[0, 0] = vals[0]
            virial[1, 1] = vals[1]
            virial[2, 2] = vals[2]
            virial[1, 2] = virial[2, 1] = vals[3]
            virial[0, 2] = virial[2, 0] = vals[4]
            virial[0, 1] = virial[1, 0] = vals[5]
        else:
            return None

        if cell_9 is not None:
            cell_mat = np.array(cell_9, dtype=np.float64).reshape(3, 3)
            volume = abs(np.linalg.det(cell_mat))
            if volume > 1e-10:
                return -virial / volume
        return None

    return None


def _parse_properties_spec(line: str):
    """
    Parse Properties spec from extended XYZ comment line.
    Example: 'Properties = species:S:1:pos:R:3:force:R:3:Z:I:1 Generated=...'
    Returns: list of (name, dtype_char, count) or None.
    """
    if "Properties" not in line:
        return None
    # Get substring after the first '=' following 'Properties'
    try:
        after = line.split("Properties", 1)[1]
        if "=" not in after:
            return None
        after_eq = after.split("=", 1)[1].strip()
        # Tokens until first token that looks like key=value (metadata)
        tokens = after_eq.split()
        spec_token = None
        for tok in tokens:
            if "=" in tok:
                break
            spec_token = tok
            break
        if not spec_token:
            return None
        fields = spec_token.split(":")
        if len(fields) < 3 or len(fields) % 3 != 0:
            return None
        out = []
        for i in range(0, len(fields), 3):
            name = fields[i]
            dtype = fields[i + 1]
            count = int(fields[i + 2])
            out.append((name, dtype, count))
        return out
    except Exception:
        return None


def extract_data_blocks(file_path, elements=None):
    """
    Extract data blocks from XYZ file format.
    
    Args:
        file_path: Path to XYZ file
        elements: List of element symbols to recognize (default: None, uses all elements from periodic table)
                  If provided, only these elements will be recognized. If None, all elements from ASE's
                  chemical_symbols list (entire periodic table) will be recognized.
        
    Returns:
        Tuple of (data_blocks, energy_list, raw_energy_list, cell_list, pbc_list)
    """
    use_all_elements = elements is None
    # Use all elements from periodic table if not specified
    if use_all_elements:
        # ASE's chemical_symbols[1:] contains all element symbols (index 0 is 'X' for unknown)
        # Filter out empty strings and 'X'
        elements = [sym for sym in chemical_symbols[1:] if sym and sym != 'X']
    
    # Create a set for faster lookup
    elements_set = set(elements)
    
    print(f"Reading {file_path}...")
    # Avoid extremely verbose output when using the full periodic table
    if use_all_elements:
        print(f"Recognizing {len(elements_set)} element types (full periodic table).")
    else:
        if len(elements_set) <= 20:
            print(f"Recognizing {len(elements_set)} element types: {', '.join(sorted(elements_set))}")
        else:
            print(f"Recognizing {len(elements_set)} element types (custom list).")
    
    with open(file_path, 'r') as file:
        data = file.readlines()

    energy_list = []       # often per-atom energy if natoms is available
    raw_energy_list = []   # raw energy from comment line (typically total energy)
    cell_list = []         # 9 numbers (row-major) or zeros if non-periodic/unknown
    pbc_list = []          # list of (px,py,pz) booleans
    stress_list = []       # list of (3,3) stress tensors (eV/Å³), zeros if not present
    data_blocks = []
    
    current_block = []
    current_cell = [np.float64(0.0)] * 9
    current_pbc = (False, False, False)
    current_stress = np.zeros((3, 3), dtype=np.float64)

    current_natoms = None
    current_properties = None

    for line in data:
        line = line.strip()
        if not line:
            # Blank line may separate frames
            if current_block:
                data_blocks.append(current_block)
                cell_list.append(current_cell)
                pbc_list.append(current_pbc)
                stress_list.append(current_stress)
                current_block = []
                current_cell = [np.float64(0.0)] * 9
                current_pbc = (False, False, False)
                current_stress = np.zeros((3, 3), dtype=np.float64)
                current_natoms = None
                current_properties = None
            continue

        # Frame header: natoms line
        if line.isdigit():
            # If we already accumulated atoms for a previous frame, flush it before starting a new one.
            if current_block:
                data_blocks.append(current_block)
                cell_list.append(current_cell)
                pbc_list.append(current_pbc)
                stress_list.append(current_stress)
                current_block = []
                current_cell = [np.float64(0.0)] * 9
                current_pbc = (False, False, False)
                current_stress = np.zeros((3, 3), dtype=np.float64)
                current_properties = None
            current_natoms = int(line)
            continue
        
        # Comment line (extended XYZ) - contains energy/pbc/lattice/properties/stress/virial
        if "energy=" in line.lower() or "properties" in line or "lattice" in line or "pbc" in line or "stress" in line.lower() or "virial" in line.lower():
            # Properties spec (optional)
            spec = _parse_properties_spec(line)
            if spec is not None:
                current_properties = spec

            # Energy
            raw_e = _parse_energy_from_comment(line)
            if raw_e is None:
                raw_e = np.float64(0.0)
            raw_energy_list.append(np.float64(raw_e))
            if current_natoms is not None and current_natoms > 0:
                energy_list.append(np.float64(raw_e) / np.float64(current_natoms))
            else:
                energy_list.append(np.float64(raw_e))

            # Lattice + PBC
            lat = _parse_lattice_from_comment(line)
            if lat is not None:
                current_cell = lat
            else:
                # Leave as zeros by default; periodicity may still be encoded by pbc
                current_cell = [np.float64(0.0)] * 9

            pbc = _parse_pbc_from_comment(line)
            if pbc is not None:
                current_pbc = pbc
            else:
                # If lattice exists but pbc missing, assume periodic in all directions.
                # If lattice missing, assume non-periodic.
                current_pbc = (True, True, True) if lat is not None else (False, False, False)

            # Stress / virial
            parsed_stress = _parse_stress_from_comment(line, cell_9=current_cell)
            if parsed_stress is not None:
                current_stress = parsed_stress

            continue

        # Atom line
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        if symbol not in elements_set:
            continue

        # Parse based on Properties spec if available; otherwise use a tolerant positional fallback.
        try:
            if current_properties is not None:
                # Build field->slice mapping
                # Tokens in atom line correspond to concatenation of fields in Properties.
                # Common: species, pos(3), force(3), Z(1)
                total_needed = sum(c for _, _, c in current_properties)
                if len(parts) < total_needed:
                    # Not enough columns; fall back
                    raise ValueError("insufficient tokens for Properties spec")

                idx = 0
                vals = {}
                for name, _, count in current_properties:
                    vals[name] = parts[idx: idx + count]
                    idx += count

                # Normalize property keys to lower-case for tolerant matching
                vals_lower = {k.lower(): v for k, v in vals.items()}

                # Species/symbol already known
                pos_tokens = vals_lower.get("pos", None)
                if pos_tokens is None or len(pos_tokens) != 3:
                    raise ValueError("missing pos in Properties")
                x, y, z = (np.float64(pos_tokens[0]), np.float64(pos_tokens[1]), np.float64(pos_tokens[2]))

                force_tokens = vals_lower.get("force", None)
                if force_tokens is None:
                    force_tokens = vals_lower.get("forces", None)
                if force_tokens is None:
                    force_tokens = vals_lower.get("f", None)
                if force_tokens is not None and len(force_tokens) == 3:
                    fx, fy, fz = (np.float64(force_tokens[0]), np.float64(force_tokens[1]), np.float64(force_tokens[2]))
                else:
                    # Try split force components (fx, fy, fz)
                    fx_t = vals_lower.get("fx", None)
                    fy_t = vals_lower.get("fy", None)
                    fz_t = vals_lower.get("fz", None)
                    if fx_t is not None and fy_t is not None and fz_t is not None:
                        fx, fy, fz = (np.float64(fx_t[0]), np.float64(fy_t[0]), np.float64(fz_t[0]))
                    else:
                        fx = fy = fz = np.float64(0.0)

                z_tokens = vals_lower.get("z", None)
                if z_tokens is None:
                    z_tokens = vals_lower.get("atomic_number", None)
                if z_tokens is not None and len(z_tokens) == 1:
                    A = np.int64(z_tokens[0])
                else:
                    A = np.int64(atomic_numbers.get(symbol, 0))
            else:
                # Fallback formats:
                # sym x y z fx fy fz Z
                # sym x y z fx fy fz
                # sym x y z A fx fy fz
                # sym x y z Z
                x, y, z = np.float64(parts[1]), np.float64(parts[2]), np.float64(parts[3])
                fx = fy = fz = np.float64(0.0)
                A = np.int64(atomic_numbers.get(symbol, 0))

                def _int_like(token):
                    try:
                        v = float(token)
                        return abs(v - round(v)) < 1e-6
                    except Exception:
                        return False

                if len(parts) >= 8:
                    # Try: sym x y z fx fy fz Z
                    if _int_like(parts[7]):
                        fx = np.float64(parts[4])
                        fy = np.float64(parts[5])
                        fz = np.float64(parts[6])
                        A = np.int64(float(parts[7]))
                    # Try: sym x y z A fx fy fz
                    elif _int_like(parts[4]):
                        A = np.int64(float(parts[4]))
                        fx = np.float64(parts[5])
                        fy = np.float64(parts[6])
                        fz = np.float64(parts[7])
                elif len(parts) >= 7:
                    # sym x y z fx fy fz
                    fx = np.float64(parts[4])
                    fy = np.float64(parts[5])
                    fz = np.float64(parts[6])
                elif len(parts) == 5:
                    # sym x y z Z
                    A = np.int64(parts[4])

            current_block.append([x, y, z, np.float64(A), fx, fy, fz])
        except Exception:
            # Skip malformed atom line
            continue

    if current_block:
        data_blocks.append(current_block)
        cell_list.append(current_cell)
        pbc_list.append(current_pbc)
        stress_list.append(current_stress)

    return data_blocks, energy_list, raw_energy_list, cell_list, pbc_list, stress_list


def objective_function(new_values, keys, atom_indices_list, energy_list):
    """
    Objective function for fitting baseline energies.
    
    Args:
        new_values: New energy values to optimize
        keys: Atomic number keys
        atom_indices_list: List of atomic indices for each structure
        energy_list: List of energies
        
    Returns:
        Residuals array
    """
    residuals = []
    for i, atom_indices in enumerate(atom_indices_list):
        sorted_idx = np.searchsorted(keys, atom_indices)
        baseline = np.sum(new_values[sorted_idx])
        residuals.append(energy_list[i] - baseline)
    return np.array(residuals, dtype=np.float64)


def fit_baseline_energies(blocks, raw_energies, keys, initial_values=None):
    """
    Fit baseline atomic energies using training set only.
    
    Args:
        blocks: List of data blocks
        raw_energies: List of raw energies
        keys: Atomic number keys
        initial_values: Initial guess for energy values
        
    Returns:
        Fitted energy values
    """
    # We solve a linear least-squares problem:
    #   raw_energy[i] ≈ sum_j count(i, key_j) * E0[key_j]
    # This can be written as M @ x ≈ y, where M is the per-structure atom-count matrix.
    #
    # The original script used SciPy's least_squares with an upper bound (<= 0.0) for E0.
    # For robustness (and to avoid SciPy/OpenMP issues on some systems), we use NumPy lstsq
    # and then apply the same bound by clipping to <= 0.0.

    print("Fitting baseline energies on Training Set...")
    keys = np.asarray(keys, dtype=np.int64)
    y = np.asarray(raw_energies, dtype=np.float64).reshape(-1)
    n_samples = len(blocks)
    n_keys = len(keys)

    # Build count matrix M: shape [n_samples, n_keys]
    M = np.zeros((n_samples, n_keys), dtype=np.float64)
    key_to_col = {int(k): j for j, k in enumerate(keys)}
    for i, block in enumerate(blocks):
        # block rows: [x, y, z, A, fx, fy, fz]
        atoms = np.asarray([row[3] for row in block], dtype=np.int64)
        # Count occurrences of each atomic number in keys
        uniq, cnt = np.unique(atoms, return_counts=True)
        for a, c in zip(uniq, cnt):
            j = key_to_col.get(int(a), None)
            if j is not None:
                M[i, j] = float(c)

    # Solve least squares: minimize ||M x - y||^2
    x, *_ = np.linalg.lstsq(M, y, rcond=None)
    x = x.astype(np.float64)

    # Apply the same upper bound used previously: E0 <= 0.0
    x = np.minimum(x, 0.0)

    print("Fitted Baseline Energies (from Train Only):", x)
    return x


def compute_correction(blocks, raw_energies, keys, fitted_vals):
    """
    Compute correction energies.
    
    Args:
        blocks: List of data blocks
        raw_energies: List of raw energies
        keys: Atomic number keys
        fitted_vals: Fitted baseline energy values
        
    Returns:
        List of correction energies
    """
    corrections = []
    for i, block in enumerate(blocks):
        atoms = np.array([row[3] for row in block], dtype=np.int64)
        sorted_idx = np.searchsorted(keys, atoms)
        baseline = np.sum(fitted_vals[sorted_idx])
        corrections.append(np.float64(raw_energies[i] - baseline))
    return corrections


def save_set(prefix, indices, blocks, raw_E, correction_E, cell_list, pbc_list=None, stress_list=None, max_atom=1, output_dir='.'):
    """
    Save dataset to HDF5 and CSV files.
    
    Args:
        prefix: Prefix for output files (e.g., 'train' or 'val')
        indices: Indices of samples to save
        blocks: List of data blocks
        raw_E: List of raw energies
        correction_E: List of correction energies
        cell_list: List of cell matrices
        max_atom: Maximum number of atoms per structure (for padding)
        output_dir: Output directory for files (default: current directory)
    """
    # Create output directory if it doesn't exist
    if output_dir != '.' and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save raw energy (Raw Energy)
    df_raw_e = pd.DataFrame(raw_E, columns=['RawEnergy']).astype(np.float64)
    df_raw_e.to_hdf(os.path.join(output_dir, f'raw_energy_{prefix}.h5'), key='df', mode='w')
    df_raw_e.to_csv(os.path.join(output_dir, f'raw_energy_{prefix}.csv'), index=False)
    
    # 2. Save correction energy (Correction Energy)
    df_corr_e = pd.DataFrame(correction_E, columns=['CorrectionEnergy']).astype(np.float64)
    df_corr_e.to_hdf(os.path.join(output_dir, f'correction_energy_{prefix}.h5'), key='df', mode='w')
    df_corr_e.to_csv(os.path.join(output_dir, f'correction_energy_{prefix}.csv'), index=False)
    
    # 3. Save cell information (Cell) + PBC flags
    cells = [cell_list[i] for i in indices]
    if pbc_list is None:
        # Backward-compatible default: periodic iff cell is non-zero
        pbcs = []
        for c in cells:
            is_periodic = (np.abs(np.asarray(c, dtype=np.float64)).sum() > 1e-9)
            pbcs.append((is_periodic, is_periodic, is_periodic))
    else:
        pbcs = [pbc_list[i] for i in indices]

    df_cells = pd.DataFrame(
        cells, columns=['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']
    ).astype(np.float64)
    df_pbc = pd.DataFrame(pbcs, columns=['pbc_x', 'pbc_y', 'pbc_z']).astype(bool)
    df_cells = pd.concat([df_cells, df_pbc], axis=1)
    df_cells.to_hdf(os.path.join(output_dir, f'cell_{prefix}.h5'), key='df', mode='w')
    df_cells.to_csv(os.path.join(output_dir, f'cell_{prefix}.csv'), index=False)
    
    # 4. Save stress tensor (Stress)
    if stress_list is not None:
        stresses = [stress_list[i] for i in indices]
    else:
        stresses = [np.zeros((3, 3), dtype=np.float64) for _ in indices]
    stress_flat = np.array([s.flatten() for s in stresses], dtype=np.float64)
    df_stress = pd.DataFrame(
        stress_flat,
        columns=['sxx', 'sxy', 'sxz', 'syx', 'syy', 'syz', 'szx', 'szy', 'szz']
    ).astype(np.float64)
    df_stress.to_hdf(os.path.join(output_dir, f'stress_{prefix}.h5'), key='df', mode='w')
    df_stress.to_csv(os.path.join(output_dir, f'stress_{prefix}.csv'), index=False)

    # 5. Save atomic detailed data (Atom Data / read_*.h5)
    all_data = []
    for block in blocks:
        curr_len = len(block)
        if curr_len < max_atom:
            for _ in range(max_atom - curr_len):
                all_data.append([np.float64(0.0)] * 8)
        
        for r_idx, row in enumerate(block):
            # [Dimension, x, y, z, A, Fx, Fy, Fz]
            all_data.append([np.float64(r_idx)] + [np.float64(x) for x in row])
            
        # Insert separator 128128.0
        all_data.append([np.float64(128128.0)] + [np.float64(0.0)] * 7)
    
    # Remove last extra separator
    if all_data and all_data[-1][0] == 128128.0:
        all_data.pop()
        
    cols = ['Dimension', 'x', 'y', 'z', 'A', 'Fx', 'Fy', 'Fz']
    df_atoms = pd.DataFrame(all_data, columns=cols).astype(np.float64)
    df_atoms.to_hdf(os.path.join(output_dir, f'read_{prefix}.h5'), key='df', mode='w')
    df_atoms.to_csv(os.path.join(output_dir, f'read_{prefix}.csv'), index=False)
    
    print(f"Saved {prefix} set (H5 and CSV) to {output_dir}/ as float64.")


def process_single_frame(args):
    """
    Process a single frame in subprocess.
    Responsible for heavy ASE computation, no HDF5 writing.
    
    Args:
        args: Tuple of (idx, pos, atom_types, cell, max_radius)
        
    Returns:
        Dictionary with processed frame data
    """
    idx, pos, atom_types, cell, pbc, max_radius = args
    
    # Physical judgment
    # Prefer explicit pbc from input; fallback to cell magnitude
    if pbc is None:
        is_periodic = (np.abs(cell).sum() > 1e-5)
        pbc_flags = [True] * 3 if is_periodic else [False] * 3
    else:
        pbc_flags = [bool(pbc[0]), bool(pbc[1]), bool(pbc[2])]
        is_periodic = any(pbc_flags)

    if is_periodic:
        # If lattice is missing/zero, use a large dummy cell to avoid ASE errors.
        # (Neighbor list for truly periodic systems should provide Lattice; this is a safety net.)
        if np.abs(cell).sum() <= 1e-9:
            current_cell = np.eye(3) * 100.0
        else:
            current_cell = cell
    else:
        pbc_flags = [False, False, False]
        current_cell = np.eye(3) * 100.0
    
    i, j, S = matscipy_neighbour_list(
        'ijS', positions=pos, cell=current_cell, pbc=pbc_flags, cutoff=max_radius
    )
    pos_checksum = np.sum(pos)
    
    # Return pure data (dictionary form)
    return {
        'idx': idx,
        'checksum': pos_checksum,
        'edge_src': i.astype(np.int64),
        'edge_dst': j.astype(np.int64),
        'edge_shifts': S.astype(np.float64),
        'cell': current_cell.astype(np.float64)
    }


def validate_h5_integrity(prefix, h5_path, data_dir='.'):
    """
    Validate existing H5 file consistency with current raw dataset.
    
    Strategy:
    1. Check if sample counts match
    2. Sample check Energy values for first, middle, and last samples
    
    Args:
        prefix: Prefix for raw energy file
        h5_path: Path to H5 file to validate
        data_dir: Directory containing raw data files
        
    Returns:
        True if consistent, False otherwise
    """
    try:
        # 1. Read raw data "summary" (only read energy file as it's small)
        energy_file = os.path.join(data_dir, f'raw_energy_{prefix}.h5')
        if not os.path.exists(energy_file):
            print(f"❌ Raw file {energy_file} missing. Cannot validate.")
            return False
            
        df_energy = pd.read_hdf(energy_file)
        raw_targets = df_energy.values.flatten()
        raw_count = len(raw_targets)

        # 2. Open H5 file for comparison
        with h5py.File(h5_path, 'r') as f:
            h5_count = len(f.keys())
            
            # Check A: Count consistency
            if raw_count != h5_count:
                print(f"⚠️ Count mismatch! Raw: {raw_count}, H5: {h5_count}")
                return False
            
            # Check B: Sample value comparison (head, middle, tail)
            indices_to_check = [0, raw_count // 2, raw_count - 1]
            
            for idx in indices_to_check:
                key = f'sample_{idx}'
                if key not in f:
                    print(f"⚠️ Missing key {key} in H5!")
                    return False
                
                # Compare Target Energy (y)
                h5_y = f[key]['y'][()]  # Read scalar
                raw_y = raw_targets[idx]
                
                # Float comparison, tolerance set to 1e-6
                if not np.isclose(h5_y, raw_y, atol=1e-6):
                    print(f"⚠️ Value mismatch at index {idx}!")
                    print(f"   Raw: {raw_y}, H5: {h5_y}")
                    return False

        print(f"✅ Consistency Check Passed for {prefix}.")
        return True

    except Exception as e:
        print(f"⚠️ Validation failed with error: {e}")
        return False


def save_to_h5_parallel(prefix, max_radius, num_workers, data_dir='.'):
    """
    Save preprocessed data to H5 file in parallel.
    
    Args:
        prefix: Prefix for input/output files (e.g., 'train' or 'val')
        max_radius: Maximum radius for neighbor search
        num_workers: Number of worker processes
        data_dir: Directory containing input files and for output (default: current directory)
    """
    output_file = os.path.join(data_dir, f'processed_{prefix}.h5')
    # Check
    if os.path.exists(output_file):
        print(f"Found existing {output_file}, checking consistency...")
        
        # If validation passes, skip computation directly
        if validate_h5_integrity(prefix, output_file, data_dir):
            print(f"Data is up-to-date. Skipping preprocessing.")
            return
        else:
            print(f"Data inconsistency detected. Re-calculating...")
            os.remove(output_file)
    
    # Check if raw data files exist
    read_file = os.path.join(data_dir, f'read_{prefix}.h5')
    energy_file = os.path.join(data_dir, f'raw_energy_{prefix}.h5')
    cell_file = os.path.join(data_dir, f'cell_{prefix}.h5')
    stress_file = os.path.join(data_dir, f'stress_{prefix}.h5')
    
    if not os.path.exists(read_file):
        raise FileNotFoundError(
            f"Raw data file not found: {read_file}\n"
            f"Please run 'mff-preprocess' first to generate the required data files."
        )
            
    df_read = pd.read_hdf(read_file)
    df_energy = pd.read_hdf(energy_file)
    df_cell = pd.read_hdf(cell_file)

    # Load stress data (optional, zeros if not available)
    if os.path.exists(stress_file):
        df_stress = pd.read_hdf(stress_file)
        stress_all = df_stress.values.astype(np.float64).reshape(-1, 3, 3)
    else:
        stress_all = None
    
    targets = df_energy.values.flatten().astype(np.float64)

    # Support both legacy cell files (9 columns) and new format (9 cell + 3 pbc flags)
    cols = list(df_cell.columns)
    if all(c in cols for c in ['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']):
        cell_mat = df_cell[['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']].values.astype(np.float64)
    else:
        # legacy: assume first 9 columns are the cell
        cell_mat = df_cell.iloc[:, :9].values.astype(np.float64)
    cells_all = cell_mat.reshape(-1, 3, 3)

    if all(c in cols for c in ['pbc_x', 'pbc_y', 'pbc_z']):
        pbcs_all = df_cell[['pbc_x', 'pbc_y', 'pbc_z']].values.astype(bool)
    else:
        # legacy fallback: periodic iff cell non-zero
        pbcs_all = np.array([(np.abs(c).sum() > 1e-9, np.abs(c).sum() > 1e-9, np.abs(c).sum() > 1e-9) for c in cells_all], dtype=bool)
    
    # Split Blocks
    values = df_read.values
    is_sep = (values[:, 0] == 128128.0)
    group_ids = is_sep.cumsum()
    clean_values = values[~is_sep]
    clean_group_ids = group_ids[~is_sep]
    _, unique_indices = np.unique(clean_group_ids, return_index=True)
    blocks = np.split(clean_values, unique_indices[1:])
    
    total_frames = len(blocks)
    print(f"Total frames to process: {total_frames}")

    tasks = []
    for idx in range(total_frames):
        block = blocks[idx]
        pos = block[:, 1:4].astype(np.float64)
        atom_types = block[:, 4].astype(np.int64)
        cell = cells_all[idx]
        pbc = pbcs_all[idx].tolist() if idx < len(pbcs_all) else None
        tasks.append((idx, pos, atom_types, cell, pbc, max_radius))

    # CLI/tests may pass num_workers=0 to mean "no parallelism".
    # ProcessPoolExecutor requires max_workers >= 1 and we also avoid division by zero.
    if num_workers is None or num_workers <= 0:
        num_workers = 1

    # chunksize determines how many tasks to distribute to each process at once
    chunk_size = max(1, total_frames // (num_workers * 4))
    
    print(f"Starting Parallel Processing (Workers={num_workers})...")
    
    with h5py.File(output_file, 'w') as f:
        # Use ProcessPoolExecutor for parallel computation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # map returns results in order, which is important for HDF5 writing
            result_iterator = executor.map(process_single_frame, tasks, chunksize=chunk_size)
            
            print("Writing results to HDF5...")
            for res in tqdm(
                result_iterator,
                total=total_frames,
                ascii=True,
                dynamic_ncols=True,
                desc="Calculating & Writing"
            ):
                idx = res['idx']
                block = blocks[idx]
                # Validation
                pos_original = block[:, 1:4].astype(np.float64)
                pos = block[:, 1:4].astype(np.float64)
                atom_types = block[:, 4].astype(np.int64)
                forces = block[:, 5:8].astype(np.float64)
                original_checksum = np.sum(pos_original)
                returned_checksum = res['checksum']
                
                # If checksums don't match, data is corrupted, raise error and stop
                if not np.isclose(original_checksum, returned_checksum, atol=1e-5):
                    raise ValueError(f"Data Mismatch at index {idx}! Worker processed wrong data.")
                
                g = f.create_group(f'sample_{idx}')
                g.create_dataset('pos', data=pos)
                g.create_dataset('A', data=atom_types)
                g.create_dataset('y', data=targets[idx])
                g.create_dataset('force', data=forces)
                
                # Write ASE computation results
                g.create_dataset('edge_src', data=res['edge_src'])
                g.create_dataset('edge_dst', data=res['edge_dst'])
                g.create_dataset('edge_shifts', data=res['edge_shifts'])
                g.create_dataset('cell', data=res['cell'])
                if stress_all is not None and idx < len(stress_all):
                    g.create_dataset('stress', data=stress_all[idx])
                else:
                    g.create_dataset('stress', data=np.zeros((3, 3), dtype=np.float64))

    print(f"Success! Saved to {output_file}")