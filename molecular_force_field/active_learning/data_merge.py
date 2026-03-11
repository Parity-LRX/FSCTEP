"""Merge new labeled data into training set."""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from molecular_force_field.data.preprocessing import (
    extract_data_blocks,
    compute_correction,
    save_set,
    save_to_h5_parallel,
)

logger = logging.getLogger(__name__)


def _read_existing_blocks(data_dir: str, prefix: str = "train"):
    """Read existing blocks from read_*, raw_energy_*, cell_*, stress_*."""
    read_file = os.path.join(data_dir, f"read_{prefix}.h5")
    energy_file = os.path.join(data_dir, f"raw_energy_{prefix}.h5")
    cell_file = os.path.join(data_dir, f"cell_{prefix}.h5")
    stress_file = os.path.join(data_dir, f"stress_{prefix}.h5")
    if not os.path.exists(read_file) or not os.path.exists(energy_file):
        return None, None, None, None, None
    df_read = pd.read_hdf(read_file)
    df_energy = pd.read_hdf(energy_file)
    df_cell = pd.read_hdf(cell_file)
    values = df_read.values
    is_sep = values[:, 0] == 128128.0
    group_ids = is_sep.cumsum()
    clean_values = values[~is_sep]
    clean_group_ids = group_ids[~is_sep]
    _, unique_indices = np.unique(clean_group_ids, return_index=True)
    raw_blocks = np.split(clean_values, unique_indices[1:])
    blocks = []
    for blk in raw_blocks:
        block_list = []
        for row in blk:
            block_list.append([row[1], row[2], row[3], row[4], row[5], row[6], row[7]])
        blocks.append(block_list)
    raw_E = df_energy.values.flatten().tolist()
    cols = list(df_cell.columns)
    if all(c in cols for c in ["ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"]):
        cell_mat = df_cell[
            ["ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"]
        ].values.astype(np.float64)
    else:
        cell_mat = df_cell.iloc[:, :9].values.astype(np.float64)
    # save_set expects flat 9-element lists, not 3x3 nested lists
    cells = [cell_mat[i].tolist() for i in range(len(blocks))]
    if all(c in cols for c in ["pbc_x", "pbc_y", "pbc_z"]):
        pbcs = df_cell[["pbc_x", "pbc_y", "pbc_z"]].values.astype(bool).tolist()
    else:
        pbcs = [
            (np.abs(cell_mat[i]).sum() > 1e-9,) * 3
            for i in range(len(blocks))
        ]
    stresses = None
    if os.path.exists(stress_file):
        df_stress = pd.read_hdf(stress_file)
        stress_arr = df_stress.values.astype(np.float64).reshape(-1, 3, 3)
        stresses = [stress_arr[i] for i in range(len(blocks))]
    return blocks, raw_E, cells, pbcs, stresses


def external_field_tensor_shape(n_values: int) -> tuple:
    """Infer Cartesian tensor shape from the number of flat values.

    Returns
    -------
    tuple
        ``(1,)`` for rank-0 (1 value),
        ``(3,)`` for rank-1 (3 values),
        ``(3, 3)`` for rank-2 (9 values),
        ``(3, 3, 3)`` for rank-3 (27 values), etc.

    Raises
    ------
    ValueError
        If *n_values* is not a power of 3.
    """
    if n_values == 1:
        return (1,)
    rank = 0
    p = 1
    while p < n_values:
        rank += 1
        p *= 3
    if p != n_values:
        raise ValueError(
            f"External field has {n_values} values; "
            f"must be 3^L for some L >= 0 (1, 3, 9, 27, …)"
        )
    return (3,) * rank


def _inject_external_field_into_h5(
    h5_path: str,
    external_field: list,
) -> None:
    """Write a uniform external_field into every sample of an H5 file.

    The value is reshaped according to :func:`external_field_tensor_shape`
    (e.g. 3 values → ``(3,)`` for rank-1, 9 values → ``(3,3)`` for rank-2).
    """
    import h5py
    shape = external_field_tensor_shape(len(external_field))
    field = np.array(external_field, dtype=np.float64).reshape(shape)
    with h5py.File(h5_path, "a") as f:
        for key in f.keys():
            if key.startswith("sample_"):
                if "external_field" in f[key]:
                    del f[key]["external_field"]
                f[key].create_dataset("external_field", data=field)
    logger.info(f"Injected external_field shape={shape} into {h5_path}")


def merge_training_data(
    data_dir: str,
    new_xyz_path: str,
    train_prefix: str = "train",
    e0_csv_path: Optional[str] = None,
    max_radius: float = 5.0,
    num_workers: int = 8,
    max_atom: Optional[int] = None,
    external_field: Optional[list] = None,
) -> int:
    """
    Merge new labeled XYZ data into existing training set.

    Args:
        data_dir: Directory containing existing train data and for output
        new_xyz_path: Path to new labeled structures (extended XYZ with energy, force)
        train_prefix: Prefix for train files (default: train)
        e0_csv_path: Path to fitted_E0.csv (default: data_dir/fitted_E0.csv)
        max_radius: For save_to_h5_parallel
        num_workers: For save_to_h5_parallel
        max_atom: Max atoms per structure for padding (default: infer from data)

    Returns:
        Number of new structures added
    """
    e0_path = e0_csv_path or os.path.join(data_dir, "fitted_E0.csv")
    if not os.path.exists(e0_path):
        raise FileNotFoundError(
            f"fitted_E0.csv not found at {e0_path}. Required for merge."
        )
    (
        new_blocks,
        _,
        new_raw_energy,
        new_cells,
        new_pbcs,
        new_stresses,
    ) = extract_data_blocks(new_xyz_path)
    if not new_blocks:
        logger.warning("No structures in new_xyz_path")
        return 0
    keys = np.array(
        [int(row[3]) for block in new_blocks for row in block if row[3] > 0],
        dtype=np.int64,
    )
    keys = np.unique(keys)
    import pandas as pd
    e0_df = pd.read_csv(e0_path)
    if "Atom" in e0_df.columns and "E0" in e0_df.columns:
        e0_keys = e0_df["Atom"].values.astype(np.int64)
        e0_vals = e0_df["E0"].values.astype(np.float64)
    else:
        raise ValueError(f"fitted_E0.csv must have Atom and E0 columns")
    new_correction = compute_correction(new_blocks, new_raw_energy, e0_keys, e0_vals)

    existing = _read_existing_blocks(data_dir, train_prefix)
    if existing[0] is None:
        blocks = new_blocks
        raw_E = new_raw_energy
        cells = new_cells
        pbcs = new_pbcs
        stresses = new_stresses
        correction_E = new_correction
    else:
        old_blocks, old_raw_E, old_cells, old_pbcs, old_stresses = existing
        blocks = old_blocks + new_blocks
        raw_E = old_raw_E + new_raw_energy
        cells = old_cells + new_cells
        pbcs = old_pbcs + new_pbcs
        if old_stresses is not None and new_stresses is not None:
            stresses = old_stresses + new_stresses
        elif old_stresses is not None:
            stresses = old_stresses + [
                np.zeros((3, 3), dtype=np.float64) for _ in new_blocks
            ]
        elif new_stresses is not None:
            stresses = [
                np.zeros((3, 3), dtype=np.float64) for _ in old_blocks
            ] + new_stresses
        else:
            stresses = None
        old_correction = compute_correction(
            old_blocks, old_raw_E, e0_keys, e0_vals
        )
        correction_E = old_correction + new_correction

    if max_atom is None:
        max_atom = max(len(b) for b in blocks)

    indices = np.arange(len(blocks))
    save_set(
        train_prefix,
        indices,
        blocks,
        raw_E,
        correction_E,
        cells,
        pbc_list=pbcs,
        stress_list=stresses,
        max_atom=max_atom,
        output_dir=data_dir,
    )
    save_to_h5_parallel(train_prefix, max_radius, num_workers, data_dir=data_dir)

    if external_field is not None:
        h5_path = os.path.join(data_dir, f"processed_{train_prefix}.h5")
        if os.path.exists(h5_path):
            _inject_external_field_into_h5(h5_path, external_field)

    logger.info(f"Merged {len(new_blocks)} new structures into {train_prefix}")
    return len(new_blocks)
