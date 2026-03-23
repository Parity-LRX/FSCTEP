"""Geometry sanity checks for active-learning exploration/candidate filtering."""

from __future__ import annotations

import os
from collections import Counter
from typing import Iterable, List, Tuple

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read, write


def _periodic(atoms: Atoms) -> bool:
    return bool(np.any(atoms.pbc))


def _covalent_thresholds(numbers: np.ndarray, scale: float) -> np.ndarray:
    radii = np.array(
        [
            covalent_radii[int(z)] if 0 <= int(z) < len(covalent_radii) else 0.0
            for z in numbers
        ],
        dtype=float,
    )
    return scale * (radii[:, None] + radii[None, :])


def check_geometry(
    atoms: Atoms,
    *,
    min_dist: float = 0.5,
    covalent_scale: float = 0.75,
) -> Tuple[bool, str]:
    """Return ``(is_valid, reason)`` for one structure.

    Validation includes:
    - finite positions / cell
    - non-degenerate periodic cell
    - minimum pair distance
    - covalent-radius scaled pair-distance threshold
    """
    if len(atoms) == 0:
        return False, "empty_structure"

    pos = np.asarray(atoms.positions, dtype=float)
    if not np.isfinite(pos).all():
        return False, "nonfinite_positions"

    cell = np.asarray(atoms.cell.array, dtype=float)
    if not np.isfinite(cell).all():
        return False, "nonfinite_cell"

    is_periodic = _periodic(atoms)
    if is_periodic:
        lengths = np.linalg.norm(cell, axis=1)
        if np.any(lengths < 1e-8):
            return False, "degenerate_periodic_cell"
        if abs(np.linalg.det(cell)) < 1e-8:
            return False, "zero_periodic_volume"

    if len(atoms) == 1:
        return True, "ok"

    dists = atoms.get_all_distances(mic=is_periodic)
    if not np.isfinite(dists).all():
        return False, "nonfinite_pair_distance"
    np.fill_diagonal(dists, np.inf)

    thresholds = np.zeros_like(dists)
    if covalent_scale > 0:
        thresholds = np.maximum(
            thresholds,
            _covalent_thresholds(np.asarray(atoms.numbers, dtype=int), covalent_scale),
        )
    if min_dist > 0:
        thresholds = np.maximum(thresholds, float(min_dist))
    np.fill_diagonal(thresholds, -np.inf)

    min_margin = dists - thresholds
    if np.min(min_margin) < 0:
        return False, "min_distance_violation"
    return True, "ok"


def filter_atoms_by_geometry(
    atoms_list: Iterable[Atoms],
    *,
    min_dist: float = 0.5,
    covalent_scale: float = 0.75,
) -> Tuple[List[Atoms], List[Tuple[int, str]]]:
    kept: List[Atoms] = []
    rejected: List[Tuple[int, str]] = []
    for idx, atoms in enumerate(atoms_list):
        ok, reason = check_geometry(
            atoms,
            min_dist=min_dist,
            covalent_scale=covalent_scale,
        )
        if ok:
            kept.append(atoms)
        else:
            rejected.append((idx, reason))
    return kept, rejected


def filter_xyz_by_geometry(
    input_xyz: str,
    output_xyz: str,
    *,
    min_dist: float = 0.5,
    covalent_scale: float = 0.75,
) -> Tuple[int, int, Counter]:
    """Filter an XYZ/EXTXYZ trajectory file and write only valid frames."""
    atoms_list = read(input_xyz, index=":")
    kept, rejected = filter_atoms_by_geometry(
        atoms_list,
        min_dist=min_dist,
        covalent_scale=covalent_scale,
    )
    if kept:
        write(output_xyz, kept, format="extxyz")
    elif os.path.exists(output_xyz):
        os.remove(output_xyz)

    reason_counts = Counter(reason for _, reason in rejected)
    return len(kept), len(rejected), reason_counts
