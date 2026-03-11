"""PES (Potential Energy Surface) coverage evaluation using SOAP descriptors."""

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from ase.io import read

logger = logging.getLogger(__name__)


def _get_soap_descriptors(
    atoms_list: list,
    species: List[str],
    r_cut: float = 5.0,
    n_max: int = 8,
    l_max: int = 6,
    periodic: bool = False,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute SOAP descriptors for a list of ASE Atoms. Returns one vector per structure (averaged)."""
    try:
        from dscribe.descriptors import SOAP
    except ImportError as e:
        raise ImportError(
            "PES coverage evaluation requires dscribe. Install with: pip install dscribe"
        ) from e

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=periodic,
        average="outer",  # one descriptor per structure
    )
    desc = soap.create(atoms_list, n_jobs=n_jobs)
    return np.asarray(desc, dtype=np.float64)


def _atoms_list_from_xyz(path: str) -> list:
    """Load structures from XYZ file as list of ASE Atoms."""
    return read(path, index=":")


def _atoms_list_from_h5_or_dir(data_dir: str, prefix: str = "train") -> list:
    """Load structures from preprocessed H5 data. Falls back to XYZ if available."""
    xyz_path = os.path.join(data_dir, f"{prefix}.xyz")
    if os.path.exists(xyz_path):
        return _atoms_list_from_xyz(xyz_path)
    # Try to find any XYZ in data_dir
    for f in os.listdir(data_dir):
        if f.endswith(".xyz"):
            return _atoms_list_from_xyz(os.path.join(data_dir, f))
    raise FileNotFoundError(
        f"No XYZ file found in {data_dir}. PES coverage requires XYZ format. "
        "Use mff-preprocess output or provide --dataset-file path to XYZ."
    )


def _extract_species_from_atoms(atoms_list: list) -> List[str]:
    """Get unique element symbols from structures."""
    symbols = set()
    for a in atoms_list:
        symbols.update(a.get_chemical_symbols())
    return sorted(symbols)


def evaluate_pes_coverage(
    dataset_path: str,
    reference_path: Optional[str] = None,
    output_path: Optional[str] = None,
    soap_rcut: float = 5.0,
    soap_nmax: int = 8,
    soap_lmax: int = 6,
    r_cov: float = 0.5,
    periodic: bool = False,
    n_jobs: int = 1,
) -> dict:
    """
    Evaluate PES coverage of a dataset in descriptor (SOAP) space.

    Args:
        dataset_path: Path to dataset XYZ file (or directory containing train.xyz)
        reference_path: Optional path to reference structures XYZ (e.g. exploration trajectory)
        output_path: Optional path to save report JSON
        soap_rcut: SOAP cutoff radius (Å)
        soap_nmax: SOAP radial basis count
        soap_lmax: SOAP spherical harmonics max degree
        r_cov: Coverage threshold (distance in SOAP space below which a ref structure is "covered")
        periodic: Whether structures are periodic
        n_jobs: Number of parallel jobs for SOAP computation

    Returns:
        Dictionary with coverage metrics
    """
    if os.path.isdir(dataset_path):
        train_atoms = _atoms_list_from_h5_or_dir(dataset_path)
    else:
        train_atoms = _atoms_list_from_xyz(dataset_path)

    species = _extract_species_from_atoms(train_atoms)
    logger.info(f"Dataset: {len(train_atoms)} structures, species: {species}")

    train_desc = _get_soap_descriptors(
        train_atoms,
        species=species,
        r_cut=soap_rcut,
        n_max=soap_nmax,
        l_max=soap_lmax,
        periodic=periodic,
        n_jobs=n_jobs,
    )

    report = {
        "n_structures": len(train_atoms),
        "species": species,
        "soap_rcut": soap_rcut,
        "soap_nmax": soap_nmax,
        "soap_lmax": soap_lmax,
    }

    if reference_path and os.path.exists(reference_path):
        ref_atoms = _atoms_list_from_xyz(reference_path)
        ref_species = _extract_species_from_atoms(ref_atoms)
        all_species = sorted(set(species) | set(ref_species))
        ref_desc = _get_soap_descriptors(
            ref_atoms,
            species=all_species,
            r_cut=soap_rcut,
            n_max=soap_nmax,
            l_max=soap_lmax,
            periodic=periodic,
            n_jobs=n_jobs,
        )

        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as e:
            raise ImportError(
                "PES coverage with reference set requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from e

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(train_desc)
        distances, _ = nn.kneighbors(ref_desc)

        dist_flat = distances.flatten()
        n_covered = int(np.sum(dist_flat < r_cov))
        coverage = float(n_covered / len(ref_atoms)) if ref_atoms else 0.0
        fill_distance = float(np.max(dist_flat))

        report["reference_n_structures"] = len(ref_atoms)
        report["r_cov"] = r_cov
        report["coverage"] = coverage
        report["n_covered"] = n_covered
        report["fill_distance"] = fill_distance
        report["distances_min"] = float(np.min(dist_flat))
        report["distances_mean"] = float(np.mean(dist_flat))
        report["distances_max"] = fill_distance

        logger.info(
            f"Reference coverage: {coverage:.2%} ({n_covered}/{len(ref_atoms)}), "
            f"fill_distance={fill_distance:.4f}"
        )
    else:
        # Intrinsic: k-NN distances within training set
        k = min(5, len(train_desc) - 1)
        if k < 1:
            report["knn_mean_distance"] = 0.0
            report["knn_max_distance"] = 0.0
        else:
            try:
                from sklearn.neighbors import NearestNeighbors
            except ImportError as e:
                raise ImportError(
                    "PES coverage requires scikit-learn. pip install scikit-learn"
                ) from e
            nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
            nn.fit(train_desc)
            distances, _ = nn.kneighbors(train_desc)
            # Exclude self (distance 0)
            dist_to_neighbors = distances[:, 1:]
            report["knn_k"] = k
            report["knn_mean_distance"] = float(np.mean(dist_to_neighbors))
            report["knn_max_distance"] = float(np.max(dist_to_neighbors))
            logger.info(
                f"Intrinsic coverage: k-NN mean={report['knn_mean_distance']:.4f}, "
                f"max={report['knn_max_distance']:.4f}"
            )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")

    return report
