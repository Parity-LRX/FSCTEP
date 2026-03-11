"""Diversity-based sub-selection for active learning candidates.

Implements multi-metric fingerprinting + Farthest Point Sampling (FPS)
to select a maximally diverse subset from the uncertainty-filtered
candidate pool, reducing redundant DFT labels.

Supported metrics
-----------------
- ``soap``      : SOAP average descriptor (requires ``dscribe``).
                  Best general-purpose structural fingerprint;
                  rotationally / translationally / permutationally invariant.
- ``devi_hist`` : Per-atom force-deviation histogram.
                  Zero extra model inference; reuses ensemble deviation data.
- ``none``      : Pass-through, no diversity filtering.
"""

import logging
from typing import List, Optional

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def farthest_point_sampling(
    features: np.ndarray,
    n_select: int,
    seed_idx: int = 0,
) -> List[int]:
    """Greedy FPS in Euclidean feature space.

    Parameters
    ----------
    features : (N, D) array
    n_select : target number of points
    seed_idx : starting point index

    Returns
    -------
    List of selected indices (length ``min(n_select, N)``).
    """
    n = len(features)
    if n_select >= n:
        return list(range(n))

    selected = [seed_idx]
    min_dist = np.full(n, np.inf)
    dist = np.linalg.norm(features - features[seed_idx], axis=1)
    min_dist = np.minimum(min_dist, dist)

    for _ in range(n_select - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(features - features[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)

    return selected


# ---------------------------------------------------------------------------
# SOAP fingerprint
# ---------------------------------------------------------------------------

def compute_soap_fingerprints(
    atoms_list: List[Atoms],
    rcut: float = 5.0,
    nmax: int = 8,
    lmax: int = 6,
    sigma: float = 0.5,
) -> np.ndarray:
    """Average SOAP descriptor per structure.  Returns ``(N, D)``."""
    try:
        from dscribe.descriptors import SOAP
    except ImportError:
        raise ImportError(
            "dscribe is required for SOAP diversity selection. "
            "Install with:  pip install dscribe   (or  pip install molecular_force_field[al])"
        )

    all_species = sorted(
        {s for atoms in atoms_list for s in atoms.get_chemical_symbols()}
    )
    periodic = any(any(a.pbc) for a in atoms_list)

    soap = SOAP(
        species=all_species,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        sigma=sigma,
        average="outer",
        periodic=periodic,
    )
    descriptors = soap.create(atoms_list, n_jobs=1)
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)

    # L2-normalize so FPS distances are comparable across systems
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    descriptors /= norms
    return descriptors


# ---------------------------------------------------------------------------
# Deviation-histogram fingerprint
# ---------------------------------------------------------------------------

def compute_devi_hist_fingerprints(
    per_atom_devi: List[np.ndarray],
    n_bins: int = 32,
    log_scale: bool = True,
    atoms_list: Optional[List[Atoms]] = None,
) -> np.ndarray:
    """Histogram of per-atom force deviation as a fixed-length fingerprint.

    Parameters
    ----------
    per_atom_devi : list of 1-D arrays  ``(n_atoms_i,)``
    n_bins        : histogram resolution
    log_scale     : bin in log-space (recommended)
    atoms_list    : if given, compute per-element multi-channel histograms
    """
    eps = 1e-10
    all_vals = np.concatenate(per_atom_devi)
    if log_scale:
        all_vals = np.log(all_vals + eps)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    if atoms_list is not None:
        all_species = sorted(
            {s for atoms in atoms_list for s in atoms.get_chemical_symbols()}
        )
        sp2ch = {s: i for i, s in enumerate(all_species)}
        n_channels = len(all_species)
        fingerprints = np.zeros((len(per_atom_devi), n_channels * n_bins))
        for i, (devi, atoms) in enumerate(zip(per_atom_devi, atoms_list)):
            vals = np.log(devi + eps) if log_scale else devi.copy()
            for j, sym in enumerate(atoms.get_chemical_symbols()):
                ch = sp2ch[sym]
                b = int(np.searchsorted(bin_edges[1:], vals[j]))
                b = min(b, n_bins - 1)
                fingerprints[i, ch * n_bins + b] += 1.0
            for ch in range(n_channels):
                sl = slice(ch * n_bins, (ch + 1) * n_bins)
                total = fingerprints[i, sl].sum()
                if total > 0:
                    fingerprints[i, sl] /= total
    else:
        fingerprints = np.zeros((len(per_atom_devi), n_bins))
        for i, devi in enumerate(per_atom_devi):
            vals = np.log(devi + eps) if log_scale else devi.copy()
            hist, _ = np.histogram(vals, bins=bin_edges)
            total = hist.sum()
            fingerprints[i] = hist / total if total > 0 else hist

    return fingerprints


# ---------------------------------------------------------------------------
# Unified selector
# ---------------------------------------------------------------------------

class DiversitySelector:
    """Select a diverse subset from candidate structures via fingerprint + FPS.

    Parameters
    ----------
    metric : ``"soap"`` | ``"devi_hist"`` | ``"none"``
    max_select : maximum number of candidates to keep
    soap_rcut, soap_nmax, soap_lmax, soap_sigma : SOAP hyper-parameters
    devi_hist_bins : number of histogram bins for devi_hist
    devi_hist_log  : use log-scale binning
    devi_hist_per_element : separate histogram channels per element
    """

    METRICS = ("soap", "devi_hist", "none")

    def __init__(
        self,
        metric: str = "soap",
        max_select: int = 50,
        soap_rcut: float = 5.0,
        soap_nmax: int = 8,
        soap_lmax: int = 6,
        soap_sigma: float = 0.5,
        devi_hist_bins: int = 32,
        devi_hist_log: bool = True,
        devi_hist_per_element: bool = True,
    ):
        if metric not in self.METRICS:
            raise ValueError(
                f"Unknown diversity metric '{metric}'; choose from {self.METRICS}"
            )
        self.metric = metric
        self.max_select = max_select
        self.soap_rcut = soap_rcut
        self.soap_nmax = soap_nmax
        self.soap_lmax = soap_lmax
        self.soap_sigma = soap_sigma
        self.devi_hist_bins = devi_hist_bins
        self.devi_hist_log = devi_hist_log
        self.devi_hist_per_element = devi_hist_per_element

        # Validate SOAP availability at construction time
        if metric == "soap":
            try:
                import dscribe  # noqa: F401
            except ImportError:
                logger.warning(
                    "dscribe not installed — falling back to devi_hist metric. "
                    "Install with:  pip install dscribe"
                )
                self.metric = "devi_hist"

    def select(
        self,
        atoms_list: List[Atoms],
        max_devi_f: Optional[np.ndarray] = None,
        per_atom_devi: Optional[List[np.ndarray]] = None,
    ) -> List[int]:
        """Return indices of the diverse subset.

        Parameters
        ----------
        atoms_list     : candidate Atoms objects
        max_devi_f     : ``(N,)`` max force deviation per frame (seeds FPS)
        per_atom_devi  : list of ``(n_atoms_i,)`` arrays (required for devi_hist)
        """
        n = len(atoms_list)
        if n <= self.max_select or self.metric == "none":
            return list(range(n))

        if self.metric == "soap":
            fingerprints = compute_soap_fingerprints(
                atoms_list,
                rcut=self.soap_rcut,
                nmax=self.soap_nmax,
                lmax=self.soap_lmax,
                sigma=self.soap_sigma,
            )
        elif self.metric == "devi_hist":
            if per_atom_devi is None:
                raise ValueError(
                    "per_atom_devi is required for 'devi_hist' diversity metric."
                )
            fingerprints = compute_devi_hist_fingerprints(
                per_atom_devi,
                n_bins=self.devi_hist_bins,
                log_scale=self.devi_hist_log,
                atoms_list=atoms_list if self.devi_hist_per_element else None,
            )
        else:
            return list(range(n))

        seed = 0
        if max_devi_f is not None and len(max_devi_f) == n:
            seed = int(np.argmax(max_devi_f))

        selected = farthest_point_sampling(fingerprints, self.max_select, seed_idx=seed)
        logger.info(
            f"Diversity ({self.metric}): {n} candidates -> {len(selected)} selected (FPS)"
        )
        return selected


# ---------------------------------------------------------------------------
# Helpers for loading per-atom deviation data
# ---------------------------------------------------------------------------

def save_per_atom_devi(per_atom_devis: List[np.ndarray], path: str) -> None:
    """Save per-atom force deviations to a text file.

    Format::

        # frame_id n_atoms val_0 val_1 ... val_{n-1}
        0 5 0.023 0.045 ...
    """
    with open(path, "w") as f:
        f.write("# frame_id n_atoms f_std_mag_per_atom...\n")
        for i, d in enumerate(per_atom_devis):
            vals = " ".join(f"{v:.6e}" for v in d)
            f.write(f"{i} {len(d)} {vals}\n")


def load_per_atom_devi(path: str) -> List[np.ndarray]:
    """Load per-atom force deviations written by :func:`save_per_atom_devi`."""
    result: List[np.ndarray] = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            n_atoms = int(parts[1])
            vals = np.array([float(x) for x in parts[2 : 2 + n_atoms]])
            result.append(vals)
    return result


def parse_max_devi_f(model_devi_path: str) -> np.ndarray:
    """Parse model_devi.out and return ``max_devi_f`` array (all frames)."""
    values: List[float] = []
    with open(model_devi_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                values.append(float(parts[1]))
    return np.array(values)
