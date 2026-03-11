"""Generate initial training data from one or more seed structures.

Workflow: perturb seed structures → DFT label → preprocess → ready-to-train.

Perturbation strategies
-----------------------
* **rattle** – Gaussian random displacements on all atoms.
* **cell_scale** – uniform random scaling of the cell (periodic only).
* **min_dist filter** – reject structures where any pair distance is too small.
"""

import logging
import os
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

logger = logging.getLogger(__name__)


def generate_perturbed_structures(
    atoms: Atoms,
    n_perturb: int = 10,
    rattle_std: float = 0.05,
    cell_scale_range: float = 0.0,
    min_dist: float = 0.5,
    seed: int = 42,
) -> List[Atoms]:
    """Generate perturbed copies of a structure.

    Parameters
    ----------
    atoms : Atoms
        Seed structure.
    n_perturb : int
        Number of perturbed copies to generate (original is always included).
    rattle_std : float
        Standard deviation (Å) of Gaussian atomic displacements.
    cell_scale_range : float
        ±range for uniform cell scaling factor (0 = disabled).
        E.g. 0.03 means cell is scaled by [0.97, 1.03].
    min_dist : float
        Minimum allowed interatomic distance (Å); structures violating
        this are discarded and regenerated.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    List[Atoms]
        Original + accepted perturbations.
    """
    rng = np.random.RandomState(seed)
    is_periodic = any(atoms.pbc)
    results = [atoms.copy()]

    attempts = 0
    max_attempts = n_perturb * 5
    while len(results) < n_perturb + 1 and attempts < max_attempts:
        attempts += 1
        a = atoms.copy()

        a.positions += rng.randn(*a.positions.shape) * rattle_std

        if cell_scale_range > 0 and is_periodic:
            scale = 1.0 + rng.uniform(-cell_scale_range, cell_scale_range)
            a.set_cell(a.get_cell() * scale, scale_atoms=True)

        if min_dist > 0 and len(a) > 1:
            dists = a.get_all_distances(mic=is_periodic)
            np.fill_diagonal(dists, np.inf)
            if dists.min() < min_dist:
                continue

        results.append(a)

    if len(results) < n_perturb + 1:
        logger.warning(
            f"Only generated {len(results) - 1}/{n_perturb} valid perturbations "
            f"(min_dist={min_dist} Å may be too strict)"
        )
    return results


def generate_init_dataset(
    structure_paths: List[str],
    n_perturb: int = 10,
    rattle_std: float = 0.05,
    cell_scale_range: float = 0.0,
    min_dist: float = 0.5,
    seed: int = 42,
) -> List[Atoms]:
    """Generate perturbations for multiple seed structures.

    Each structure gets the same number of perturbations.  A ``source``
    info tag is attached to every Atoms so downstream analysis can trace
    which seed a sample came from.
    """
    all_atoms: List[Atoms] = []
    for i, path in enumerate(structure_paths):
        base_atoms = ase_read(path)
        perturbed = generate_perturbed_structures(
            base_atoms,
            n_perturb=n_perturb,
            rattle_std=rattle_std,
            cell_scale_range=cell_scale_range,
            min_dist=min_dist,
            seed=seed + i * 1000,
        )
        for a in perturbed:
            a.info["source"] = os.path.basename(path)
        all_atoms.extend(perturbed)
        logger.info(
            f"Structure [{i}] {os.path.basename(path)}: "
            f"generated {len(perturbed)} frames (1 original + {len(perturbed) - 1} perturbed)"
        )
    logger.info(f"Total unlabeled frames: {len(all_atoms)}")
    return all_atoms
