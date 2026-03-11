"""
Labeling interface for DFT or script-based labeling.

Available labelers
------------------
ScriptLabeler   -- call any user script
IdentityLabeler -- ML model (testing only, no DFT)
PySCFLabeler    -- PySCF (molecules: HF/DFT/MP2/CCSD, no external binary needed)
VaspLabeler     -- VASP via ASE (requires VASP binary + pseudopotentials)
CP2KLabeler     -- CP2K via ASE (requires CP2K-shell binary)
EspressoLabeler -- Quantum Espresso pw.x via ASE (requires QE binary + pseudopotentials)
GaussianLabeler -- Gaussian 16/09 via ASE (requires g16/g09 binary)
ORCALabeler     -- ORCA via ASE (requires orca binary)

All ASE-based labelers share the same ``ASECalculatorLabeler`` base class.
They each run every structure in a temporary sub-directory, write the
labeled extended-XYZ to ``output_path``, and clean up temp files when
``cleanup=True``.

Environment setup quick-reference
----------------------------------
VASP:
    export ASE_VASP_COMMAND="mpirun vasp_std"
    export VASP_PP_PATH=/path/to/vasp/pseudopotentials

CP2K:
    export ASE_CP2K_COMMAND="cp2k_shell.psmp"
    export CP2K_DATA_DIR=/path/to/cp2k/data

Quantum Espresso:
    # No env var needed; pass profile=EspressoProfile(command=..., pseudo_dir=...) to labeler.

Gaussian:
    # Put g16/g09 on PATH, or export ASE_GAUSSIAN_COMMAND="g16 < PREFIX.com > PREFIX.log"

ORCA:
    # Put orca on PATH, or pass orca_command='/full/path/to/orca' to labeler.
"""

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ase.io import read, write

logger = logging.getLogger(__name__)


class Labeler(ABC):
    """Abstract interface for labeling structures with DFT."""

    @abstractmethod
    def label(
        self,
        structures_xyz_path: str,
        output_path: str,
        work_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """Label structures and write to output_path. Returns output_path."""
        pass


class ScriptLabeler(Labeler):
    """Call user script: script_path input.xyz output.xyz"""

    def __init__(self, script_path: str):
        self.script_path = script_path

    def label(
        self,
        structures_xyz_path: str,
        output_path: str,
        work_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """Run script with input and output paths."""
        subprocess.run(
            [self.script_path, structures_xyz_path, output_path],
            cwd=work_dir,
            check=True,
        )
        return output_path


class IdentityLabeler(Labeler):
    """Use ML model to predict energy/forces (for testing, no DFT)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        max_radius: float = 5.0,
        atomic_energy_file: Optional[str] = None,
        external_field: Optional[list] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_radius = max_radius
        self.atomic_energy_file = atomic_energy_file
        self.external_field = external_field

    def label(
        self,
        structures_xyz_path: str,
        output_path: str,
        work_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """Predict with ML model and write extended XYZ."""
        import torch
        from molecular_force_field.evaluation.calculator import MyE3NNCalculator
        from molecular_force_field.active_learning.model_loader import build_e3trans_from_checkpoint

        dev = torch.device(self.device)
        ckpt = checkpoint_path or self.checkpoint_path
        e3trans, config = build_e3trans_from_checkpoint(
            ckpt,
            dev,
            atomic_energy_file=self.atomic_energy_file,
        )
        ref_dict = {
            k.item(): v.item()
            for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
        }
        ext_tensor = None
        if self.external_field is not None:
            from molecular_force_field.active_learning.data_merge import external_field_tensor_shape
            shape = external_field_tensor_shape(len(self.external_field))
            ext_tensor = torch.tensor(self.external_field, dtype=torch.float64, device=dev).reshape(shape)
        calc = MyE3NNCalculator(e3trans, ref_dict, dev, self.max_radius, external_tensor=ext_tensor)
        atoms_list = read(structures_xyz_path, index=":")
        for a in atoms_list:
            a.calc = calc
            _ = a.get_potential_energy()
            _ = a.get_forces()
        write(output_path, atoms_list, format="extxyz")
        logger.info(f"IdentityLabeler: wrote {len(atoms_list)} structures to {output_path}")
        return output_path


class PySCFLabeler(Labeler):
    """
    Label structures with PySCF (energy + forces via analytic gradients).

    Supports any method and basis set available in PySCF.
    Periodic systems are supported via PySCF-PBC (pbc=True).

    Parameters
    ----------
    method: str
        DFT functional or HF/MP2/CCSD, e.g. "b3lyp", "pbe", "hf", "mp2"
    basis: str
        Basis set, e.g. "6-31g*", "def2-svp", "sto-3g"
    charge: int
        Total charge (default 0)
    spin: int
        2S (number of unpaired electrons, default 0 for closed-shell)
    max_memory: int
        Max memory in MB for PySCF (default 4000)
    verbose: int
        PySCF verbosity level (0-9, default 0 = silent)
    conv_tol: float
        SCF convergence threshold (default 1e-9)
    """

    def __init__(
        self,
        method: str = "b3lyp",
        basis: str = "6-31g*",
        charge: int = 0,
        spin: int = 0,
        max_memory: int = 4000,
        verbose: int = 0,
        conv_tol: float = 1e-9,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        error_handling: str = "raise",
    ):
        self.method = method.lower()
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.max_memory = max_memory
        self.verbose = verbose
        self.conv_tol = conv_tol
        self.n_workers = n_workers
        # When n_workers > 1, default to 1 thread per worker to avoid
        # over-subscription (n_workers × threads_per_worker ≤ total cores).
        if threads_per_worker is None:
            self.threads_per_worker = 1 if n_workers > 1 else 0  # 0 = let PySCF choose
        else:
            self.threads_per_worker = threads_per_worker
        self.error_handling = error_handling

    def _label_one(self, atoms) -> tuple:
        """
        Compute energy (eV) and forces (eV/Å) for a single ASE Atoms object.
        Returns (energy_eV, forces_eV_Angstrom).
        """
        from pyscf import gto, scf, dft, grad
        import numpy as np

        HARTREE_TO_EV = 27.211386245988
        BOHR_TO_ANGSTROM = 0.529177210903
        # forces in Hartree/Bohr → eV/Å
        GRAD_CONV = HARTREE_TO_EV / BOHR_TO_ANGSTROM

        symbols = atoms.get_chemical_symbols()
        positions_ang = atoms.get_positions()

        # Build PySCF mol
        atom_spec = [
            (sym, tuple(pos)) for sym, pos in zip(symbols, positions_ang)
        ]
        mol = gto.Mole()
        mol.atom = atom_spec
        mol.basis = self.basis
        mol.charge = self.charge
        mol.max_memory = self.max_memory
        mol.verbose = self.verbose
        mol.unit = "Angstrom"
        # Auto-detect spin: if total electrons is odd and user passed spin=0, set spin=1
        import numpy as np
        from ase.data import atomic_numbers as _ase_anums
        z_sum = int(sum(atoms.get_atomic_numbers())) - self.charge
        spin = self.spin
        if spin == 0 and (z_sum % 2) == 1:
            spin = 1
            logger.debug(f"Auto-set spin=1 (odd electron count {z_sum})")
        mol.spin = spin
        mol.build()

        # Choose method
        m = self.method
        if m in ("hf", "rhf", "uhf"):
            if self.spin == 0:
                mf = scf.RHF(mol)
            else:
                mf = scf.UHF(mol)
        elif m == "mp2":
            from pyscf import mp
            if self.spin == 0:
                mf = scf.RHF(mol)
            else:
                mf = scf.UHF(mol)
            mf.conv_tol = self.conv_tol
            mf.kernel()
            mp2 = mp.MP2(mf)
            e_corr, _ = mp2.kernel()
            e_total_hart = mf.e_tot + e_corr
            # MP2 gradient
            g_obj = mp2.nuc_grad_method()
            grad_hart_bohr = g_obj.kernel()
            energy_ev = e_total_hart * HARTREE_TO_EV
            forces_ev_ang = -grad_hart_bohr * GRAD_CONV
            return energy_ev, forces_ev_ang
        else:
            # DFT
            if self.spin == 0:
                mf = dft.RKS(mol)
            else:
                mf = dft.UKS(mol)
            mf.xc = self.method

        mf.conv_tol = self.conv_tol
        mf.kernel()
        if not mf.converged:
            logger.warning("SCF did not converge for this structure; using unconverged result.")

        # Analytic gradient
        g_obj = mf.nuc_grad_method()
        grad_hart_bohr = g_obj.kernel()

        energy_ev = mf.e_tot * HARTREE_TO_EV
        forces_ev_ang = -grad_hart_bohr * GRAD_CONV
        return energy_ev, forces_ev_ang

    def _compute_one(self, idx: int, atoms):
        """Worker: compute (idx, atoms_out). Called in worker process."""
        from ase.calculators.singlepoint import SinglePointCalculator

        # Limit PySCF internal threads in each worker to avoid over-subscription.
        if self.threads_per_worker and self.threads_per_worker > 0:
            try:
                from pyscf import lib as pyscf_lib
                pyscf_lib.num_threads(self.threads_per_worker)
            except Exception:
                pass

        energy, forces = self._label_one(atoms)
        atoms_out = atoms.copy()
        atoms_out.calc = SinglePointCalculator(atoms_out, energy=energy, forces=forces)
        return idx, atoms_out

    def label(
        self,
        structures_xyz_path: str,
        output_path: str,
        work_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """Run PySCF on all structures, write labeled extended XYZ."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        atoms_list = read(structures_xyz_path, index=":")
        n = len(atoms_list)
        logger.info(
            f"PySCFLabeler: {n} structures  method={self.method}  basis={self.basis}"
            f"  n_workers={self.n_workers}"
        )

        results: Dict[int, Any] = {}

        if self.n_workers <= 1:
            for i, atoms in enumerate(atoms_list):
                logger.info(f"PySCFLabeler: [{i + 1}/{n}]")
                try:
                    _, atoms_out = self._compute_one(i, atoms)
                    results[i] = atoms_out
                except Exception as exc:
                    logger.error(f"PySCFLabeler: structure {i + 1} failed: {exc}")
                    if self.error_handling == "raise":
                        raise
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                future_to_idx = {
                    pool.submit(self._compute_one, i, atoms): i
                    for i, atoms in enumerate(atoms_list)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        _, atoms_out = future.result()
                        results[idx] = atoms_out
                        logger.info(
                            f"PySCFLabeler: [{idx + 1}/{n}] done  "
                            f"({len(results)}/{n} completed)"
                        )
                    except Exception as exc:
                        logger.error(f"PySCFLabeler: structure {idx + 1} failed: {exc}")
                        if self.error_handling == "raise":
                            for f in future_to_idx:
                                f.cancel()
                            raise

        labeled = [results[i] for i in sorted(results)]
        write(output_path, labeled, format="extxyz")
        logger.info(f"PySCFLabeler: wrote {len(labeled)}/{n} structures to {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# ASECalculatorLabeler – generic base for any ASE calculator
# ---------------------------------------------------------------------------

class ASECalculatorLabeler(Labeler):
    """
    Base class for DFT labelers backed by an ASE calculator.

    Sub-classes only need to implement ``_make_calculator(atoms, run_dir)``
    which returns a configured ASE calculator object.

    Each structure is computed in its own sub-directory
    ``<work_dir>/ase_calc_<i>/`` so that calculators that write many files
    (VASP, QE, …) do not pollute each other.

    Parallel execution
    ------------------
    Set ``n_workers > 1`` to run multiple structures concurrently via
    ``concurrent.futures.ProcessPoolExecutor``.  Each worker process is an
    independent Python interpreter so DFT codes (VASP, QE, ORCA …) that
    themselves use MPI are not affected — they each get their own directory
    and their own set of file handles.

    ``n_workers=1`` (default) runs serially in the current process, which is
    easier to debug and works correctly in environments that do not support
    ``fork`` (e.g. macOS with certain MKL builds).

    Failed structures
    -----------------
    By default any failed calculation raises immediately and stops the loop.
    Set ``error_handling="skip"`` to log the error and continue; the failed
    structure will be absent from the output file.

    Parameters
    ----------
    n_workers : int
        Number of parallel worker processes (default: 1 = serial).
    cleanup : bool
        Remove per-structure calculation directories after success.
    error_handling : str
        ``"raise"`` (default) or ``"skip"``.
    """

    def __init__(
        self,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
    ):
        self.n_workers = n_workers
        # Default: 1 thread per worker when parallel to avoid over-subscription.
        if threads_per_worker is None:
            self.threads_per_worker = 1 if n_workers > 1 else 0
        else:
            self.threads_per_worker = threads_per_worker
        self.cleanup = cleanup
        self.error_handling = error_handling

    def _make_calculator(self, atoms, run_dir: str):
        raise NotImplementedError

    def _get_label(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Internal: compute one structure (called in worker process)
    # ------------------------------------------------------------------

    def _compute_one(self, idx: int, atoms, run_dir: str):
        """
        Compute energy + forces for a single structure.
        Returns (idx, atoms_out) on success, raises on failure.
        """
        import shutil
        from ase.calculators.singlepoint import SinglePointCalculator

        # Inject thread-count env vars before the calculator starts any subprocess.
        # Covers OMP_NUM_THREADS (VASP, QE, OpenBLAS) and MKL_NUM_THREADS.
        t = self.threads_per_worker
        if t and t > 0:
            os.environ["OMP_NUM_THREADS"] = str(t)
            os.environ["MKL_NUM_THREADS"] = str(t)
            os.environ["OPENBLAS_NUM_THREADS"] = str(t)

        os.makedirs(run_dir, exist_ok=True)
        calc = self._make_calculator(atoms, run_dir)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        atoms_out = atoms.copy()
        atoms_out.calc = SinglePointCalculator(atoms_out, energy=energy, forces=forces)
        if self.cleanup:
            shutil.rmtree(run_dir, ignore_errors=True)
        return idx, atoms_out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(
        self,
        structures_xyz_path: str,
        output_path: str,
        work_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        os.makedirs(work_dir, exist_ok=True)
        atoms_list = read(structures_xyz_path, index=":")
        n = len(atoms_list)
        label = self._get_label()
        logger.info(f"{label}: {n} structures  n_workers={self.n_workers}")

        # Build task list: (idx, atoms, run_dir)
        tasks = [
            (i, atoms_list[i], os.path.join(work_dir, f"ase_calc_{i:04d}"))
            for i in range(n)
        ]

        results: Dict[int, Any] = {}

        if self.n_workers <= 1:
            # ---- serial path ----------------------------------------
            for idx, atoms, run_dir in tasks:
                logger.info(f"{label}: [{idx + 1}/{n}]  dir={run_dir}")
                try:
                    _, atoms_out = self._compute_one(idx, atoms, run_dir)
                    results[idx] = atoms_out
                except Exception as exc:
                    logger.error(f"{label}: structure {idx + 1} failed: {exc}")
                    if self.error_handling == "raise":
                        raise
        else:
            # ---- parallel path --------------------------------------
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                future_to_idx = {
                    pool.submit(self._compute_one, idx, atoms, run_dir): idx
                    for idx, atoms, run_dir in tasks
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        _, atoms_out = future.result()
                        results[idx] = atoms_out
                        logger.info(
                            f"{label}: [{idx + 1}/{n}] done  "
                            f"({len(results)}/{n} completed)"
                        )
                    except Exception as exc:
                        logger.error(f"{label}: structure {idx + 1} failed: {exc}")
                        if self.error_handling == "raise":
                            # cancel remaining futures cleanly
                            for f in future_to_idx:
                                f.cancel()
                            raise

        # Re-order by original index (as_completed returns in finish order)
        labeled = [results[i] for i in sorted(results)]
        write(output_path, labeled, format="extxyz")
        logger.info(f"{label}: wrote {len(labeled)}/{n} structures to {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# VaspLabeler
# ---------------------------------------------------------------------------

class VaspLabeler(ASECalculatorLabeler):
    """
    Label structures with VASP via the ASE Vasp calculator.

    Environment setup
    -----------------
    Set one of the following before running:

    .. code-block:: bash

        export ASE_VASP_COMMAND="mpirun vasp_std"
        export VASP_PP_PATH=/path/to/vasp/potpaw_PBE

    Parameters
    ----------
    xc : str
        Exchange-correlation functional, e.g. ``"PBE"`` (default).
    kpts : tuple or list
        k-point mesh, e.g. ``(4, 4, 4)`` for periodic solids or
        ``(1, 1, 1)`` / ``None`` for molecules in a box.
    encut : float
        Plane-wave cutoff in eV (default: VASP internal default).
    ediff : float
        Electronic convergence threshold in eV (default: 1e-6).
    ismear : int
        Smearing type.  0 = Gaussian, -5 = tetrahedron, 1 = Methfessel-Paxton.
        Use 0 for molecules; -5 for insulators; 1 for metals (default: 0).
    sigma : float
        Smearing width in eV (default: 0.05).
    command : str, optional
        Override ``ASE_VASP_COMMAND`` / ``VASP_COMMAND`` with a custom string,
        e.g. ``"mpiexec -np 8 vasp_std"``.
    vasp_kwargs : dict
        Any additional VASP INCAR keyword → value pairs accepted by ASE,
        e.g. ``{"ispin": 2, "algo": "Fast"}``.
    cleanup : bool
        Remove per-structure run directories after success.

    Example
    -------
    .. code-block:: python

        labeler = VaspLabeler(
            xc="PBE",
            kpts=(4, 4, 4),
            encut=500,
            ediff=1e-6,
            vasp_kwargs={"ispin": 2},
        )
    """

    def __init__(
        self,
        xc: str = "PBE",
        kpts=(1, 1, 1),
        encut: Optional[float] = None,
        ediff: float = 1e-6,
        ismear: int = 0,
        sigma: float = 0.05,
        command: Optional[str] = None,
        vasp_kwargs: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
    ):
        super().__init__(n_workers=n_workers, threads_per_worker=threads_per_worker,
                         cleanup=cleanup, error_handling=error_handling)
        self.xc = xc
        self.kpts = kpts
        self.encut = encut
        self.ediff = ediff
        self.ismear = ismear
        self.sigma = sigma
        self.command = command
        self.vasp_kwargs = vasp_kwargs or {}

    def _make_calculator(self, atoms, run_dir: str):
        from ase.calculators.vasp import Vasp

        kwargs: Dict[str, Any] = dict(
            xc=self.xc,
            kpts=self.kpts,
            ediff=self.ediff,
            ismear=self.ismear,
            sigma=self.sigma,
            directory=run_dir,
        )
        if self.encut is not None:
            kwargs["encut"] = self.encut
        if self.command is not None:
            kwargs["command"] = self.command
        kwargs.update(self.vasp_kwargs)
        return Vasp(**kwargs)

    def _get_label(self) -> str:
        return f"VaspLabeler(xc={self.xc})"


# ---------------------------------------------------------------------------
# CP2KLabeler
# ---------------------------------------------------------------------------

class CP2KLabeler(ASECalculatorLabeler):
    """
    Label structures with CP2K via the ASE CP2K calculator.

    CP2K communicates through a long-lived ``cp2k_shell`` subprocess; ASE
    manages the process lifecycle automatically.

    Environment setup
    -----------------
    .. code-block:: bash

        export ASE_CP2K_COMMAND="cp2k_shell.psmp"
        # Optional: point to CP2K's data directory so basis/potential files
        # are found automatically:
        export CP2K_DATA_DIR=/path/to/cp2k/data

    Parameters
    ----------
    xc : str
        XC functional accepted by CP2K / libxc, e.g. ``"PBE"`` (default).
    basis_set : str
        Gaussian basis set name, e.g. ``"DZVP-MOLOPT-SR-GTH"`` (default).
    pseudo_potential : str
        Pseudopotential name.  ``"auto"`` (default) lets CP2K choose based on
        the XC functional.
    cutoff : float
        Plane-wave cutoff for the finest grid in Ry (default: 400 Ry).
    max_scf : int
        Maximum SCF iterations (default: 50).
    charge : float
        Total system charge (default: 0).
    uks : bool
        Request unrestricted Kohn-Sham (spin-polarised).  Auto-detected
        from ``atoms.get_initial_magnetic_moments()`` if ``None``.
    inp : str, optional
        Raw CP2K input template string that overrides / extends the defaults.
        Useful for specifying k-points, smearing, etc.
    command : str, optional
        Override ``ASE_CP2K_COMMAND``.
    cp2k_kwargs : dict
        Any extra keyword arguments forwarded directly to
        ``ase.calculators.cp2k.CP2K``.
    cleanup : bool
        Remove per-structure run directories after success.

    Example
    -------
    .. code-block:: python

        labeler = CP2KLabeler(
            xc="PBE",
            basis_set="DZVP-MOLOPT-SR-GTH",
            cutoff=600,
        )
    """

    def __init__(
        self,
        xc: str = "PBE",
        basis_set: str = "DZVP-MOLOPT-SR-GTH",
        pseudo_potential: str = "auto",
        cutoff: float = 400.0,
        max_scf: int = 50,
        charge: float = 0.0,
        uks: Optional[bool] = None,
        inp: Optional[str] = None,
        command: Optional[str] = None,
        cp2k_kwargs: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
    ):
        super().__init__(n_workers=n_workers, threads_per_worker=threads_per_worker,
                         cleanup=cleanup, error_handling=error_handling)
        self.xc = xc
        self.basis_set = basis_set
        self.pseudo_potential = pseudo_potential
        self.cutoff = cutoff
        self.max_scf = max_scf
        self.charge = charge
        self.uks = uks
        self.inp = inp
        self.command = command
        self.cp2k_kwargs = cp2k_kwargs or {}

    def _make_calculator(self, atoms, run_dir: str):
        from ase.calculators.cp2k import CP2K
        from ase.units import Rydberg

        uks = self.uks
        if uks is None:
            magmoms = atoms.get_initial_magnetic_moments()
            uks = bool(any(m != 0 for m in magmoms))

        kwargs: Dict[str, Any] = dict(
            xc=self.xc,
            basis_set=self.basis_set,
            pseudo_potential=self.pseudo_potential,
            cutoff=self.cutoff * Rydberg,
            max_scf=self.max_scf,
            charge=self.charge,
            uks=uks,
            label=os.path.join(run_dir, "cp2k"),
        )
        if self.inp is not None:
            kwargs["inp"] = self.inp
        if self.command is not None:
            kwargs["command"] = self.command
        kwargs.update(self.cp2k_kwargs)
        return CP2K(**kwargs)

    def _get_label(self) -> str:
        return f"CP2KLabeler(xc={self.xc})"


# ---------------------------------------------------------------------------
# EspressoLabeler
# ---------------------------------------------------------------------------

class EspressoLabeler(ASECalculatorLabeler):
    """
    Label structures with Quantum Espresso ``pw.x`` via the ASE Espresso calculator.

    Parameters
    ----------
    pseudopotentials : dict
        Mapping from element symbol to pseudopotential filename,
        e.g. ``{"H": "H.pbe-rrkjus_psl.1.0.0.UPF", "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF"}``.
    pseudo_dir : str
        Directory containing the pseudopotential ``.UPF`` files.
    input_data : dict
        Flat or nested dictionary of QE ``pw.x`` input parameters, e.g.
        ``{"ecutwfc": 60, "ecutrho": 480, "occupations": "smearing",
           "smearing": "cold", "degauss": 0.02}``.
    kpts : tuple
        k-point mesh as ``(nk1, nk2, nk3)``, e.g. ``(4, 4, 4)`` (default: ``(1, 1, 1)``).
    koffset : tuple
        k-point grid offset ``(0|1, 0|1, 0|1)`` (default: ``(0, 0, 0)``).
    command : str, optional
        Command to run ``pw.x``, e.g. ``"mpirun -np 8 pw.x -in PREFIX.pwi > PREFIX.pwo"``.
        Alternatively set via a ``EspressoProfile``.
    espresso_kwargs : dict
        Any extra keyword arguments forwarded to ``ase.calculators.espresso.Espresso``.
    cleanup : bool
        Remove per-structure run directories after success.

    Example
    -------
    .. code-block:: python

        labeler = EspressoLabeler(
            pseudopotentials={"H": "H.pbe-rrkjus_psl.1.0.0.UPF",
                              "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF"},
            pseudo_dir="/path/to/pseudos",
            input_data={"ecutwfc": 60, "ecutrho": 480},
            kpts=(2, 2, 2),
        )
    """

    def __init__(
        self,
        pseudopotentials: Dict[str, str],
        pseudo_dir: str,
        input_data: Optional[Dict[str, Any]] = None,
        kpts=(1, 1, 1),
        koffset=(0, 0, 0),
        command: Optional[str] = None,
        espresso_kwargs: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
    ):
        super().__init__(n_workers=n_workers, threads_per_worker=threads_per_worker,
                         cleanup=cleanup, error_handling=error_handling)
        self.pseudopotentials = pseudopotentials
        self.pseudo_dir = pseudo_dir
        self.input_data = input_data or {}
        self.kpts = kpts
        self.koffset = koffset
        self.command = command
        self.espresso_kwargs = espresso_kwargs or {}

    def _make_calculator(self, atoms, run_dir: str):
        from ase.calculators.espresso import Espresso, EspressoProfile

        profile = None
        if self.command is not None:
            profile = EspressoProfile(
                command=self.command,
                pseudo_dir=self.pseudo_dir,
            )

        kwargs: Dict[str, Any] = dict(
            pseudopotentials=self.pseudopotentials,
            input_data=self.input_data,
            kpts=self.kpts,
            koffset=self.koffset,
            directory=run_dir,
        )
        if profile is not None:
            kwargs["profile"] = profile
        else:
            # Fall back to letting ASE locate pw.x via PATH;
            # still need to supply pseudo_dir
            kwargs["profile"] = EspressoProfile(pseudo_dir=self.pseudo_dir)
        kwargs.update(self.espresso_kwargs)
        return Espresso(**kwargs)

    def _get_label(self) -> str:
        return "EspressoLabeler"


# ---------------------------------------------------------------------------
# GaussianLabeler
# ---------------------------------------------------------------------------

class GaussianLabeler(ASECalculatorLabeler):
    """
    Label structures with Gaussian (g16 / g09) via the ASE Gaussian calculator.

    Environment setup
    -----------------
    Either put ``g16`` or ``g09`` on your ``PATH``, or set:

    .. code-block:: bash

        export ASE_GAUSSIAN_COMMAND="g16 < PREFIX.com > PREFIX.log"

    Parameters
    ----------
    method : str
        Level of theory, e.g. ``"b3lyp"`` (default), ``"pbe1pbe"``,
        ``"mp2"``, ``"ccsd"``.
    basis : str
        Basis set, e.g. ``"6-31+G*"`` (default), ``"def2tzvp"``.
    charge : int
        Molecular charge (default: 0).
    mult : int
        Spin multiplicity 2S+1 (default: 1 = singlet).
        Pass ``None`` to let ASE auto-detect from ``atoms.initial_magnetic_moments``.
    nproc : int
        Number of parallel CPU cores (written to the ``%nprocshared`` link-0 line).
    mem : str
        Memory allocation, e.g. ``"4GB"`` (default).
    command : str, optional
        Override the default Gaussian command string.
    gaussian_kwargs : dict
        Extra keyword arguments forwarded to ``ase.calculators.gaussian.Gaussian``,
        e.g. ``{"scf": "maxcycle=200"}``.
    cleanup : bool
        Remove per-structure run directories after success.

    Example
    -------
    .. code-block:: python

        labeler = GaussianLabeler(
            method="b3lyp",
            basis="6-31+G*",
            nproc=8,
            mem="8GB",
        )
    """

    def __init__(
        self,
        method: str = "b3lyp",
        basis: str = "6-31+G*",
        charge: int = 0,
        mult: Optional[int] = 1,
        nproc: int = 1,
        mem: str = "4GB",
        command: Optional[str] = None,
        gaussian_kwargs: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
    ):
        super().__init__(n_workers=n_workers, threads_per_worker=threads_per_worker,
                         cleanup=cleanup, error_handling=error_handling)
        self.method = method
        self.basis = basis
        self.charge = charge
        self.mult = mult
        self.nproc = nproc
        self.mem = mem
        self.command = command
        self.gaussian_kwargs = gaussian_kwargs or {}

    def _make_calculator(self, atoms, run_dir: str):
        from ase.calculators.gaussian import Gaussian

        label = os.path.join(run_dir, "gaussian")
        kwargs: Dict[str, Any] = dict(
            label=label,
            method=self.method,
            basis=self.basis,
            charge=self.charge,
            nprocshared=self.nproc,
            mem=self.mem,
            force="force",  # request analytic gradient
        )
        if self.mult is not None:
            kwargs["mult"] = self.mult
        if self.command is not None:
            kwargs["command"] = self.command
        kwargs.update(self.gaussian_kwargs)
        return Gaussian(**kwargs)

    def _get_label(self) -> str:
        return f"GaussianLabeler(method={self.method}/{self.basis})"


# ---------------------------------------------------------------------------
# ORCALabeler
# ---------------------------------------------------------------------------

class ORCALabeler(ASECalculatorLabeler):
    """
    Label structures with ORCA via the ASE ORCA calculator.

    Environment setup
    -----------------
    Either put ``orca`` on your ``PATH`` or pass ``orca_command`` explicitly.

    Parameters
    ----------
    simpleinput : str
        The ``!``-line of an ORCA input file, specifying method + basis set +
        additional keywords, e.g. ``"B3LYP def2-TZVP TightSCF"`` (default).
    blocks : str
        The ``%...end``-block section.  Use this to set parallelism:
        ``"%pal nprocs 8 end"`` (default: ``"%pal nprocs 1 end"``).
    charge : int
        Total charge (default: 0).
    mult : int
        Spin multiplicity 2S+1 (default: 1).
    orca_command : str, optional
        Full path to the ORCA executable, e.g. ``"/opt/orca/orca"``.
        If ``None``, ASE searches ``PATH``.
    orca_kwargs : dict
        Extra keyword arguments forwarded to ``ase.calculators.orca.ORCA``.
    cleanup : bool
        Remove per-structure run directories after success.

    Example
    -------
    .. code-block:: python

        labeler = ORCALabeler(
            simpleinput="B3LYP def2-TZVP TightSCF",
            blocks="%pal nprocs 8 end",
            charge=0,
            mult=1,
        )
    """

    def __init__(
        self,
        simpleinput: str = "B3LYP def2-TZVP TightSCF",
        blocks: str = "%pal nprocs 1 end",
        charge: int = 0,
        mult: int = 1,
        orca_command: Optional[str] = None,
        orca_kwargs: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
    ):
        super().__init__(n_workers=n_workers, threads_per_worker=threads_per_worker,
                         cleanup=cleanup, error_handling=error_handling)
        self.simpleinput = simpleinput
        self.blocks = blocks
        self.charge = charge
        self.mult = mult
        self.orca_command = orca_command
        self.orca_kwargs = orca_kwargs or {}

    def _make_calculator(self, atoms, run_dir: str):
        from ase.calculators.orca import ORCA, OrcaProfile

        profile = None
        if self.orca_command is not None:
            profile = OrcaProfile(command=self.orca_command)

        kwargs: Dict[str, Any] = dict(
            charge=self.charge,
            mult=self.mult,
            orcasimpleinput=self.simpleinput,
            orcablocks=self.blocks,
            directory=run_dir,
        )
        if profile is not None:
            kwargs["profile"] = profile
        kwargs.update(self.orca_kwargs)
        return ORCA(**kwargs)

    def _get_label(self) -> str:
        return f"ORCALabeler({self.simpleinput})"


# ---------------------------------------------------------------------------
# Shared template rendering utility
# ---------------------------------------------------------------------------

class _SafeFormatMap(dict):
    """
    A dict subclass for ``str.format_map`` that leaves unknown placeholders
    (including those with format specs like ``{val:.4f}`` or shell variables
    like ``${VAR}``) untouched instead of raising ``KeyError`` or
    ``ValueError``.

    For a known key the value is formatted normally.
    For an unknown key, the original ``{key}`` or ``{key:spec}`` token is
    reproduced verbatim in the output.
    """

    class _Preserve:
        """Returned for unknown keys; reproduces the original token."""
        def __init__(self, key: str):
            self._key = key

        def __format__(self, spec: str) -> str:
            return "{" + self._key + (":" + spec if spec else "") + "}"

        def __getattr__(self, name: str) -> "_SafeFormatMap._Preserve":
            return _SafeFormatMap._Preserve(self._key + "." + name)

        def __getitem__(self, item) -> "_SafeFormatMap._Preserve":
            return _SafeFormatMap._Preserve(self._key + "[" + str(item) + "]")

    def __missing__(self, key: str) -> "_Preserve":
        return _SafeFormatMap._Preserve(key)


def _render_template(template_text: str, **kwargs) -> str:
    """Render a template string using :class:`_SafeFormatMap`."""
    return template_text.format_map(_SafeFormatMap(kwargs))


# ---------------------------------------------------------------------------
# LocalScriptLabeler
# ---------------------------------------------------------------------------

class LocalScriptLabeler(Labeler):
    """
    Label structures by running a **local** script for each structure.

    Uses the **same template format** as :class:`SlurmLabeler` so that a
    single script works for both local testing and HPC (SLURM) production.
    The script is rendered with per-structure placeholders and executed
    directly via ``bash`` (no job scheduler needed).

    Supports ``n_workers > 1`` for multi-process parallel execution on a
    single node.

    Template placeholders
    ---------------------
    ``{run_dir}``    – absolute path to the per-structure working directory.
    ``{input_xyz}``  – absolute path to input extended-XYZ (single structure).
    ``{output_xyz}`` – path where the script **must** write the labeled result.
    ``{job_name}``   – auto-generated name (useful for logging inside script).
    Any other ``{key}`` tokens (e.g. ``${VAR}`` shell variables) are left
    unchanged.

    Example template (``dft_job.sh``)::

        #!/bin/bash
        cd {run_dir}
        python -c "from ase.io import read,write; write('POSCAR', read('{input_xyz}'))"
        # --- VASP parameters here ---
        cat > INCAR << 'EOF'
        ENCUT = 500
        EDIFF = 1E-6
        NSW   = 0
        ...
        EOF
        mpirun -np 8 vasp_std
        python -c "
        from ase.io import read, write
        write('{output_xyz}', read('OUTCAR'), format='extxyz')
        "

    Parameters
    ----------
    script_template : str
        Path to the script template file.
    n_workers : int
        Number of structures to run in parallel (default: 1).
    threads_per_worker : int, optional
        ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` injected into each worker
        environment.  Default: 1 when ``n_workers > 1``, otherwise unset.
    cleanup : bool
        Remove per-structure run directories after success (default: False).
    error_handling : str
        ``"raise"`` (default) or ``"skip"``.
    bash : str
        Path to bash interpreter (default: ``"bash"``).
    """

    def __init__(
        self,
        script_template: str,
        n_workers: int = 1,
        threads_per_worker: Optional[int] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
        bash: str = "bash",
    ):
        self.script_template = script_template
        self.n_workers = n_workers
        if threads_per_worker is None:
            self.threads_per_worker = 1 if n_workers > 1 else 0
        else:
            self.threads_per_worker = threads_per_worker
        self.cleanup = cleanup
        self.error_handling = error_handling
        self.bash = bash

    def _run_one(self, idx: int, atoms, run_dir: str, template_text: str):
        """Render and execute the script for one structure."""
        import shutil
        from ase.calculators.singlepoint import SinglePointCalculator

        os.makedirs(run_dir, exist_ok=True)
        input_xyz = os.path.join(run_dir, "input.xyz")
        output_xyz = os.path.join(run_dir, "output.xyz")
        write(input_xyz, atoms, format="extxyz")

        script_text = _render_template(
            template_text,
            run_dir=run_dir,
            input_xyz=input_xyz,
            output_xyz=output_xyz,
            job_name=f"mffal_{idx:04d}",
        )
        script_path = os.path.join(run_dir, "run.sh")
        with open(script_path, "w") as fh:
            fh.write(script_text)
        os.chmod(script_path, 0o755)

        env = os.environ.copy()
        t = self.threads_per_worker
        if t and t > 0:
            env["OMP_NUM_THREADS"] = str(t)
            env["MKL_NUM_THREADS"] = str(t)
            env["OPENBLAS_NUM_THREADS"] = str(t)

        result = subprocess.run(
            [self.bash, script_path],
            cwd=run_dir,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Script exited with code {result.returncode} "
                f"for structure {idx} in {run_dir}"
            )
        if not os.path.exists(output_xyz):
            raise FileNotFoundError(
                f"Script did not produce output.xyz for structure {idx}: "
                f"{output_xyz}"
            )

        labeled_atoms = read(output_xyz, index=":")
        if self.cleanup:
            shutil.rmtree(run_dir, ignore_errors=True)
        return idx, labeled_atoms

    def label(
        self,
        structures_xyz_path: str,
        output_path: str,
        work_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        os.makedirs(work_dir, exist_ok=True)
        with open(self.script_template) as fh:
            template_text = fh.read()

        atoms_list = read(structures_xyz_path, index=":")
        n = len(atoms_list)
        logger.info(
            f"LocalScriptLabeler: {n} structures  "
            f"n_workers={self.n_workers}  template={self.script_template}"
        )

        tasks = [
            (i, atoms_list[i], os.path.join(os.path.abspath(work_dir), f"local_{i:04d}"))
            for i in range(n)
        ]

        # Resume: skip structures whose output.xyz already exists
        pending_tasks = []
        results: Dict[int, Any] = {}
        for idx, atoms, run_dir in tasks:
            out_xyz = os.path.join(run_dir, "output.xyz")
            if os.path.exists(out_xyz):
                logger.info(f"LocalScriptLabeler: [{idx}] resuming from existing output")
                try:
                    results[idx] = read(out_xyz, index=":")
                except Exception as exc:
                    logger.warning(f"LocalScriptLabeler: [{idx}] cannot read existing output: {exc}")
                    pending_tasks.append((idx, atoms, run_dir))
            else:
                pending_tasks.append((idx, atoms, run_dir))

        if self.n_workers <= 1:
            for idx, atoms, run_dir in pending_tasks:
                logger.info(f"LocalScriptLabeler: [{idx + 1}/{n}]  dir={run_dir}")
                try:
                    _, labeled = self._run_one(idx, atoms, run_dir, template_text)
                    results[idx] = labeled
                except Exception as exc:
                    logger.error(f"LocalScriptLabeler: [{idx}] failed: {exc}")
                    if self.error_handling == "raise":
                        raise
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                future_to_idx = {
                    pool.submit(self._run_one, idx, atoms, run_dir, template_text): idx
                    for idx, atoms, run_dir in pending_tasks
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        _, labeled = future.result()
                        results[idx] = labeled
                        logger.info(
                            f"LocalScriptLabeler: [{idx + 1}/{n}] done  "
                            f"({len(results)}/{n} completed)"
                        )
                    except Exception as exc:
                        logger.error(f"LocalScriptLabeler: [{idx}] failed: {exc}")
                        if self.error_handling == "raise":
                            for f in future_to_idx:
                                f.cancel()
                            raise

        # Flatten and re-order
        labeled_flat: List = []
        for i in sorted(results):
            atoms_out = results[i]
            if isinstance(atoms_out, list):
                labeled_flat.extend(atoms_out)
            else:
                labeled_flat.append(atoms_out)

        write(output_path, labeled_flat, format="extxyz")
        logger.info(
            f"LocalScriptLabeler: wrote {len(labeled_flat)}/{n} structures "
            f"to {output_path}"
        )
        return output_path


# ---------------------------------------------------------------------------
# SlurmLabeler
# ---------------------------------------------------------------------------

class SlurmLabeler(Labeler):
    """
    Label structures by submitting one SLURM job per structure.

    Each structure is written to its own run directory.  The user provides a
    job script **template** that handles the actual DFT calculation and writes
    the result back as extended XYZ.  This class submits all jobs, polls their
    status, and assembles the final output file.

    Template placeholders
    ---------------------
    The following ``{key}`` tokens are substituted in the job script before
    submission.  Any key not listed here is left untouched (safe for literal
    braces inside the script).

    ``{run_dir}``
        Absolute path to the per-structure working directory.
    ``{input_xyz}``
        Absolute path to the input extended-XYZ file (single structure).
    ``{output_xyz}``
        Absolute path where the script **must** write the labeled result
        (extended XYZ with ``energy`` and ``forces`` in the info/arrays).
    ``{job_name}``
        Auto-generated SLURM job name (alphanumeric, safe for schedulers).
    ``{partition}``  ``{nodes}``  ``{ntasks}``  ``{time}``  ``{mem}``
        Values of the corresponding constructor parameters.

    Example template (``vasp_job.sh``)::

        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition={partition}
        #SBATCH --nodes={nodes}
        #SBATCH --ntasks-per-node={ntasks}
        #SBATCH --time={time}
        #SBATCH --mem={mem}
        #SBATCH --output={run_dir}/slurm.out
        #SBATCH --error={run_dir}/slurm.err

        cd {run_dir}
        # Convert input.xyz to POSCAR
        python -c "from ase.io import read,write; write('POSCAR', read('{input_xyz}'))"
        # Run VASP
        mpirun -np {ntasks} vasp_std
        # Convert OUTCAR → labeled extXYZ
        python -c "
        from ase.io import read, write
        atoms = read('OUTCAR')
        write('{output_xyz}', atoms, format='extxyz')
        "

    Parameters
    ----------
    job_script_template : str
        Path to the SLURM script template file described above.
    partition : str
        SLURM partition / queue (default: ``"cpu"``).
    nodes : int
        Nodes per job (default: 1).
    ntasks_per_node : int
        MPI tasks per node / ``--ntasks-per-node`` (default: 32).
    time_limit : str
        Wall-clock limit per job, e.g. ``"02:00:00"`` (default: ``"02:00:00"``).
    mem : str
        Memory per job, e.g. ``"64G"`` (default: ``"64G"``).
    max_concurrent : int
        Maximum jobs allowed in the SLURM queue at the same time.
        New submissions are held until the running count drops below this
        value (default: 200).
    poll_interval : int
        Seconds between ``squeue`` status polls (default: 30).
    sbatch_extra : list[str]
        Extra ``sbatch`` CLI arguments, e.g.
        ``["--account=myproject", "--qos=high"]``.
    cleanup : bool
        Remove per-structure run directories after success (default: False).
    error_handling : str
        ``"raise"`` — stop on first failed job (default).
        ``"skip"``  — log the failure and continue collecting other results.

    Notes
    -----
    **Resume / restart**: if ``output.xyz`` already exists in a run directory
    (from a previous interrupted run), that structure is not re-submitted and
    its result is reused directly.

    **sacct fallback**: when a job ID disappears from ``squeue`` (because it
    finished), the labeler queries ``sacct`` to verify the final state.
    If ``sacct`` is unavailable on the cluster, the labeler falls back to
    treating a missing-from-squeue job as ``COMPLETED`` after a brief
    grace-period check for the output file.
    """

    # SLURM states considered terminal
    _TERMINAL = frozenset({
        "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED",
        "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "REVOKED",
    })
    _SUCCESS = frozenset({"COMPLETED"})

    def __init__(
        self,
        job_script_template: str,
        partition: str = "cpu",
        nodes: int = 1,
        ntasks_per_node: int = 32,
        time_limit: str = "02:00:00",
        mem: str = "64G",
        max_concurrent: int = 200,
        poll_interval: int = 30,
        sbatch_extra: Optional[List[str]] = None,
        cleanup: bool = False,
        error_handling: str = "raise",
    ):
        self.job_script_template = job_script_template
        self.partition = partition
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.time_limit = time_limit
        self.mem = mem
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.sbatch_extra = sbatch_extra or []
        self.cleanup = cleanup
        self.error_handling = error_handling

    # ------------------------------------------------------------------
    # Template rendering  (delegates to module-level _render_template)
    # ------------------------------------------------------------------

    def _render_template(self, template_text: str, **kwargs) -> str:
        return _render_template(template_text, **kwargs)

    # ------------------------------------------------------------------
    # SLURM interaction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_cmd(cmd: List[str], retries: int = 3, delay: float = 5.0) -> str:
        """Run a shell command with retry on transient failures."""
        import time as _time
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                # Non-zero but maybe transient (e.g. squeue overloaded)
                last_exc = RuntimeError(
                    f"{cmd[0]} returned {result.returncode}: {result.stderr.strip()}"
                )
            except Exception as exc:
                last_exc = exc
            if attempt < retries - 1:
                _time.sleep(delay)
        raise RuntimeError(
            f"Command failed after {retries} attempts: {cmd}\n{last_exc}"
        )

    def _submit_job(self, script_path: str) -> str:
        """Submit a job and return its SLURM job ID (string)."""
        cmd = ["sbatch", "--parsable"] + self.sbatch_extra + [script_path]
        output = self._run_cmd(cmd)
        # --parsable prints "jobid" or "jobid;cluster"
        job_id = output.split(";")[0].strip()
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch output: {output!r}")
        return job_id

    def _query_squeue(self, job_ids: List[str]) -> Dict[str, str]:
        """
        Query squeue for a list of job IDs.
        Returns a dict {job_id: state_string} for jobs still in squeue.
        Jobs that have left the queue are absent from the returned dict.
        """
        if not job_ids:
            return {}
        try:
            output = self._run_cmd([
                "squeue",
                "--jobs=" + ",".join(job_ids),
                "--noheader",
                "--format=%i %T",
            ])
        except RuntimeError:
            return {}
        states: Dict[str, str] = {}
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
        return states

    def _query_sacct(self, job_id: str) -> Optional[str]:
        """
        Query sacct for the final state of a completed job.
        Returns the state string, or None if sacct is unavailable / returns nothing.
        """
        try:
            output = self._run_cmd([
                "sacct",
                "-j", job_id,
                "--noheader",
                "--format=State",
                "--parsable2",
            ])
        except RuntimeError:
            return None
        # sacct returns one line per job-step; use the first non-empty line
        for line in output.splitlines():
            state = line.strip().split()[0] if line.strip() else ""
            if state:
                # Strip trailing "+CANCELLED" modifiers
                return state.split("+")[0].upper()
        return None

    def _query_job_state(self, job_id: str, output_xyz: str) -> str:
        """
        Return the canonical state for a single job ID.
        Priority: squeue → sacct → file-exists heuristic.
        """
        sq = self._query_squeue([job_id])
        if job_id in sq:
            return sq[job_id].upper()

        # Job left squeue — check sacct
        sacct_state = self._query_sacct(job_id)
        if sacct_state and sacct_state in self._TERMINAL:
            return sacct_state

        # sacct unavailable or ambiguous — fall back to file check
        if os.path.exists(output_xyz):
            return "COMPLETED"
        return "FAILED"

    # ------------------------------------------------------------------
    # Main label loop
    # ------------------------------------------------------------------

    def label(
        self,
        structures_xyz_path: str,
        output_path: str,
        work_dir: str,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        import shutil
        import time as _time
        from ase.calculators.singlepoint import SinglePointCalculator

        os.makedirs(work_dir, exist_ok=True)
        atoms_list = read(structures_xyz_path, index=":")
        n = len(atoms_list)
        logger.info(f"SlurmLabeler: {n} structures  partition={self.partition}")

        # Load job script template
        with open(self.job_script_template) as fh:
            template_text = fh.read()

        # ------------------------------------------------------------------
        # Phase 1 – prepare directories and check for resumable results
        # ------------------------------------------------------------------
        run_dirs: List[str] = []
        output_xyzs: List[str] = []
        input_xyzs: List[str] = []

        for i, atoms in enumerate(atoms_list):
            run_dir = os.path.join(os.path.abspath(work_dir), f"slurm_{i:04d}")
            os.makedirs(run_dir, exist_ok=True)
            inp = os.path.join(run_dir, "input.xyz")
            out = os.path.join(run_dir, "output.xyz")
            write(inp, atoms, format="extxyz")
            run_dirs.append(run_dir)
            input_xyzs.append(inp)
            output_xyzs.append(out)

        # ------------------------------------------------------------------
        # Phase 2 – submit jobs (skip already-completed ones)
        # ------------------------------------------------------------------
        # Maps job_id → structure index
        pending: Dict[str, int] = {}
        # Structures whose output already exists (resume)
        completed: Dict[int, bool] = {}

        for i in range(n):
            if os.path.exists(output_xyzs[i]):
                logger.info(
                    f"SlurmLabeler: [{i}] output already exists, skipping submission"
                )
                completed[i] = True
                continue

            # Throttle: wait until below max_concurrent
            while len(pending) >= self.max_concurrent:
                _time.sleep(self.poll_interval)
                self._poll_once(pending, completed, output_xyzs)

            job_name = f"mffal_{i:04d}"
            script_text = self._render_template(
                template_text,
                run_dir=run_dirs[i],
                input_xyz=input_xyzs[i],
                output_xyz=output_xyzs[i],
                job_name=job_name,
                partition=self.partition,
                nodes=self.nodes,
                ntasks=self.ntasks_per_node,
                time=self.time_limit,
                mem=self.mem,
            )
            script_path = os.path.join(run_dirs[i], "job.sh")
            with open(script_path, "w") as fh:
                fh.write(script_text)
            os.chmod(script_path, 0o755)

            try:
                job_id = self._submit_job(script_path)
                pending[job_id] = i
                logger.info(
                    f"SlurmLabeler: [{i + 1}/{n}] submitted  job_id={job_id}"
                )
            except Exception as exc:
                logger.error(f"SlurmLabeler: [{i}] submission failed: {exc}")
                if self.error_handling == "raise":
                    raise
                completed[i] = False  # mark as failed

        # ------------------------------------------------------------------
        # Phase 3 – wait for all remaining pending jobs
        # ------------------------------------------------------------------
        logger.info(
            f"SlurmLabeler: all jobs submitted ({len(pending)} pending, "
            f"{len(completed)} already done).  Polling every {self.poll_interval}s …"
        )
        failed_indices: List[int] = [
            idx for idx, ok in completed.items() if not ok
        ]

        while pending:
            _time.sleep(self.poll_interval)
            newly_done, newly_failed = self._poll_once(
                pending, completed, output_xyzs
            )
            if newly_done:
                logger.info(
                    f"SlurmLabeler: {newly_done} job(s) completed  "
                    f"({len(completed)}/{n} total done,  {len(pending)} still running)"
                )
            if newly_failed:
                failed_indices.extend(newly_failed)
                logger.error(
                    f"SlurmLabeler: {len(newly_failed)} job(s) FAILED  "
                    f"(indices: {newly_failed})"
                )
                if self.error_handling == "raise":
                    raise RuntimeError(
                        f"SLURM jobs failed for structure indices: {newly_failed}"
                    )

        # ------------------------------------------------------------------
        # Phase 4 – collect results in original order
        # ------------------------------------------------------------------
        labeled: List = []
        failed_indices_set = set(failed_indices)

        for i, atoms in enumerate(atoms_list):
            if i in failed_indices_set:
                logger.warning(f"SlurmLabeler: [{i}] skipped (job failed)")
                continue
            out_xyz = output_xyzs[i]
            if not os.path.exists(out_xyz):
                msg = f"SlurmLabeler: [{i}] output file missing: {out_xyz}"
                logger.error(msg)
                if self.error_handling == "raise":
                    raise FileNotFoundError(msg)
                continue
            try:
                result = read(out_xyz, index=":")
                labeled.extend(result)
            except Exception as exc:
                logger.error(f"SlurmLabeler: [{i}] failed to read output: {exc}")
                if self.error_handling == "raise":
                    raise
                continue
            if self.cleanup:
                shutil.rmtree(run_dirs[i], ignore_errors=True)

        write(output_path, labeled, format="extxyz")
        logger.info(
            f"SlurmLabeler: wrote {len(labeled)}/{n} structures to {output_path}"
        )
        return output_path

    # ------------------------------------------------------------------
    # Polling helper
    # ------------------------------------------------------------------

    def _poll_once(
        self,
        pending: Dict[str, int],
        completed: Dict[int, bool],
        output_xyzs: List[str],
    ):
        """
        Check all pending jobs once.  Mutates ``pending`` and ``completed``.
        Returns (n_newly_done, list_of_newly_failed_indices).
        """
        if not pending:
            return 0, []

        job_ids = list(pending.keys())
        sq_states = self._query_squeue(job_ids)

        newly_done = 0
        newly_failed: List[int] = []

        for job_id in job_ids:
            state = sq_states.get(job_id)

            if state is None:
                # Job left squeue — resolve via sacct / file heuristic
                idx = pending[job_id]
                state = self._query_job_state(job_id, output_xyzs[idx])

            state = state.upper()

            if state in self._TERMINAL:
                idx = pending.pop(job_id)
                if state in self._SUCCESS:
                    completed[idx] = True
                    newly_done += 1
                else:
                    completed[idx] = False
                    newly_failed.append(idx)
                    logger.error(
                        f"SlurmLabeler: job {job_id} (struct {idx}) "
                        f"terminated with state={state}"
                    )

        return newly_done, newly_failed
