"""CLI for generating initial training data from seed structures (mff-init-data).

One-command cold-start: perturb → DFT label → preprocess → ready-to-train data.

Examples
--------
Molecules with PySCF (no external binary needed)::

    mff-init-data --structures water.xyz ethanol.xyz \\
        --n-perturb 15 --rattle-std 0.05 \\
        --label-type pyscf --pyscf-method b3lyp --pyscf-basis 6-31g* \\
        --output-dir data

Periodic systems with VASP::

    mff-init-data --structures POSCAR.vasp \\
        --n-perturb 20 --rattle-std 0.02 --cell-scale-range 0.03 \\
        --label-type vasp --vasp-xc PBE --vasp-encut 500 --vasp-kpts 4 4 4 \\
        --label-n-workers 8 --output-dir data

From a directory of structures::

    mff-init-data --structures structures/ \\
        --n-perturb 10 --label-type pyscf --output-dir data
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import shutil
from typing import List, Sequence, Tuple

import numpy as np

from molecular_force_field.active_learning.init_data import (
    generate_init_dataset,
)
from molecular_force_field.active_learning.geometry_filter import check_geometry

logger = logging.getLogger(__name__)


def _resolve_structures(paths):
    """Expand directories and validate paths; return list of files."""
    result = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith((".xyz", ".extxyz", ".cif", ".vasp", ".poscar")):
                    result.append(os.path.join(p, f))
        elif os.path.exists(p):
            result.append(p)
        else:
            raise FileNotFoundError(f"Structure not found: {p}")
    if not result:
        raise ValueError("No valid structure files found.")
    return result


def _build_labeler(args):
    """Instantiate the appropriate labeler from CLI args."""
    from molecular_force_field.active_learning.labeling import (
        CP2KLabeler,
        EspressoLabeler,
        GaussianLabeler,
        ORCALabeler,
        PySCFLabeler,
        VaspLabeler,
    )

    common = dict(
        n_workers=args.label_n_workers,
        threads_per_worker=args.label_threads_per_worker,
        error_handling=args.label_error_handling,
    )
    vasp_kwargs = {}
    if args.vasp_ncore is not None:
        vasp_kwargs["ncore"] = args.vasp_ncore

    if args.label_type == "pyscf":
        return PySCFLabeler(
            method=args.pyscf_method,
            basis=args.pyscf_basis,
            charge=args.pyscf_charge,
            spin=args.pyscf_spin,
            max_memory=args.pyscf_max_memory,
            conv_tol=args.pyscf_conv_tol,
            **common,
        )
    elif args.label_type == "vasp":
        return VaspLabeler(
            xc=args.vasp_xc,
            kpts=tuple(args.vasp_kpts),
            encut=args.vasp_encut,
            ediff=args.vasp_ediff,
            ismear=args.vasp_ismear,
            sigma=args.vasp_sigma,
            command=_resolve_mpi_command(args.vasp_command, args.vasp_mpi_ranks, args.vasp_mpi_launcher),
            vasp_kwargs=vasp_kwargs,
            cleanup=args.vasp_cleanup,
            **common,
        )
    elif args.label_type == "cp2k":
        return CP2KLabeler(
            xc=args.cp2k_xc,
            basis_set=args.cp2k_basis_set,
            pseudo_potential=args.cp2k_pseudo,
            cutoff=args.cp2k_cutoff,
            max_scf=args.cp2k_max_scf,
            charge=args.cp2k_charge,
            command=_resolve_mpi_command(args.cp2k_command, args.cp2k_mpi_ranks, args.cp2k_mpi_launcher),
            cleanup=args.cp2k_cleanup,
            **common,
        )
    elif args.label_type == "espresso":
        if not args.qe_pseudo_dir:
            raise ValueError("--qe-pseudo-dir required for espresso")
        if not args.qe_pseudopotentials:
            raise ValueError("--qe-pseudopotentials required for espresso")
        pseudopotentials = json.loads(args.qe_pseudopotentials)
        input_data = {"ecutwfc": args.qe_ecutwfc}
        if args.qe_ecutrho is not None:
            input_data["ecutrho"] = args.qe_ecutrho
        return EspressoLabeler(
            pseudopotentials=pseudopotentials,
            pseudo_dir=args.qe_pseudo_dir,
            input_data=input_data,
            kpts=tuple(args.qe_kpts),
            command=args.qe_command,
            cleanup=args.qe_cleanup,
            **common,
        )
    elif args.label_type == "gaussian":
        return GaussianLabeler(
            method=args.gaussian_method,
            basis=args.gaussian_basis,
            charge=args.gaussian_charge,
            mult=args.gaussian_mult,
            nproc=args.gaussian_nproc,
            mem=args.gaussian_mem,
            command=args.gaussian_command,
            cleanup=args.gaussian_cleanup,
            **common,
        )
    elif args.label_type == "orca":
        return ORCALabeler(
            simpleinput=args.orca_simpleinput,
            blocks=f"%pal nprocs {args.orca_nproc} end",
            charge=args.orca_charge,
            mult=args.orca_mult,
            orca_command=args.orca_command,
            cleanup=args.orca_cleanup,
            **common,
        )
    else:
        raise ValueError(f"Unsupported label type: {args.label_type}")


def _resolve_mpi_command(command, mpi_ranks, mpi_launcher):
    if command is None:
        return None
    if mpi_ranks is None or mpi_ranks <= 1:
        return command
    lowered = command.lower()
    if "mpirun" in lowered or "mpiexec" in lowered:
        return command
    return f"{mpi_launcher} -np {mpi_ranks} {command}"


def _relax_seed_structures(structure_paths, labeler, output_dir, fmax=0.05, steps=200):
    """Relax seed structures before perturbation for cold-start init data.

    Uses the same ASE-backed backend as the later single-point labeling stage.
    Non-ASE labelers are left untouched for compatibility.
    """
    if fmax <= 0:
        logger.info("Seed relaxation disabled (fmax <= 0); using input structures directly.")
        return structure_paths

    if not hasattr(labeler, "_make_calculator"):
        logger.warning(
            "Labeler %s does not expose an ASE calculator; skipping seed relaxation.",
            labeler.__class__.__name__,
        )
        return structure_paths

    from ase.io import read as ase_read, write as ase_write
    from ase.optimize import BFGS

    relax_dir = os.path.join(output_dir, "relaxed_seeds")
    if os.path.isdir(relax_dir):
        shutil.rmtree(relax_dir)
    os.makedirs(relax_dir, exist_ok=True)

    relaxed_paths = []
    for i, path in enumerate(structure_paths):
        atoms = ase_read(path)
        run_dir = os.path.join(relax_dir, f"seed_{i:04d}_work")
        os.makedirs(run_dir, exist_ok=True)
        calc = labeler._make_calculator(atoms, run_dir)
        atoms.calc = calc
        logger.info(
            "Relaxing seed [%d] %s with %s (fmax=%s, steps=%d)",
            i,
            os.path.basename(path),
            labeler.__class__.__name__,
            fmax,
            steps,
        )
        opt = BFGS(atoms, logfile=os.path.join(run_dir, "seed_relax.log"))
        opt.run(fmax=fmax, steps=steps)
        out_path = os.path.join(relax_dir, f"seed_{i:04d}.xyz")
        ase_write(out_path, atoms, format="extxyz")
        relaxed_paths.append(out_path)

    logger.info("Relaxed %d seed structure(s) into %s", len(relaxed_paths), relax_dir)
    return relaxed_paths


def _filter_labeled_atoms_by_force(atoms_list, max_force_filter: float):
    """Drop labeled frames whose max per-atom force norm exceeds the threshold."""
    if max_force_filter is None or max_force_filter <= 0:
        return list(atoms_list), []

    kept = []
    dropped = []
    for idx, atoms in enumerate(atoms_list):
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        if np.isfinite(max_force) and max_force <= max_force_filter:
            kept.append(atoms)
        else:
            dropped.append((idx, max_force))
    return kept, dropped


def _sample_uniform_indices(n_frames: int, sample_count: int) -> List[int]:
    """Return `sample_count` unique indices spread uniformly across a trajectory."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if n_frames < sample_count:
        raise ValueError(
            f"Cannot sample {sample_count} frames from only {n_frames} available frames."
        )

    raw = np.linspace(0, n_frames - 1, num=sample_count)
    indices: List[int] = []
    used = set()
    for value in raw:
        idx = int(round(float(value)))
        if idx not in used:
            used.add(idx)
            indices.append(idx)
            continue
        for delta in range(1, n_frames):
            left = idx - delta
            right = idx + delta
            if left >= 0 and left not in used:
                used.add(left)
                indices.append(left)
                break
            if right < n_frames and right not in used:
                used.add(right)
                indices.append(right)
                break
        else:
            raise RuntimeError("Failed to construct unique uniform sample indices.")
    return sorted(indices)


def _filter_valid_aimd_frames(
    frames: Sequence,
    min_dist: float,
    covalent_scale: float,
) -> Tuple[List, int, str]:
    """Keep the valid AIMD prefix and stop at the first invalid geometry."""
    valid = []
    for idx, atoms in enumerate(frames):
        ok, reason = check_geometry(
            atoms,
            min_dist=min_dist,
            covalent_scale=covalent_scale,
        )
        if not ok:
            return valid, idx, reason
        valid.append(atoms)
    return valid, -1, ""


def _run_ase_aimd_trajectory(
    atoms,
    labeler,
    run_dir: str,
    total_frames: int,
    temperature: float,
    timestep_fs: float,
    friction: float,
    min_dist: float,
    covalent_scale: float,
):
    """Fallback AIMD path using ASE MD with any ASE-backed calculator."""
    from ase import units
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    os.makedirs(run_dir, exist_ok=True)
    work_atoms = atoms.copy()
    work_atoms.calc = labeler._make_calculator(work_atoms, run_dir)
    MaxwellBoltzmannDistribution(work_atoms, temperature_K=temperature)
    dyn = Langevin(
        work_atoms,
        timestep_fs * units.fs,
        temperature_K=temperature,
        friction=friction,
    )

    frames = []

    def _record_frame():
        energy = work_atoms.get_potential_energy()
        forces = work_atoms.get_forces()
        frame = work_atoms.copy()
        frame.calc = SinglePointCalculator(frame, energy=energy, forces=forces)
        frames.append(frame)

    _record_frame()
    while len(frames) < total_frames:
        dyn.run(1)
        _record_frame()

    valid, invalid_idx, invalid_reason = _filter_valid_aimd_frames(
        frames,
        min_dist=min_dist,
        covalent_scale=covalent_scale,
    )
    if invalid_idx >= 0:
        logger.warning(
            "AIMD fallback stopped at invalid geometry frame=%d reason=%s kept=%d/%d",
            invalid_idx,
            invalid_reason,
            len(valid),
            len(frames),
        )
    return valid


def _run_vasp_aimd_trajectory(
    atoms,
    labeler,
    run_dir: str,
    total_frames: int,
    temperature: float,
    timestep_fs: float,
    min_dist: float,
    covalent_scale: float,
):
    """Run one internal VASP MD trajectory and parse all ionic frames."""
    from ase.io import read as ase_read

    os.makedirs(run_dir, exist_ok=True)
    work_atoms = atoms.copy()
    calc = labeler._make_calculator(work_atoms, run_dir)
    calc.set(
        ibrion=0,
        nsw=total_frames,
        potim=timestep_fs,
        tebeg=temperature,
        teend=temperature,
        smass=0,
        isif=2,
        isym=0,
    )
    work_atoms.calc = calc
    _ = work_atoms.get_potential_energy()

    vasprun_path = os.path.join(run_dir, "vasprun.xml")
    if not os.path.exists(vasprun_path):
        raise FileNotFoundError(f"VASP AIMD did not produce vasprun.xml: {vasprun_path}")

    frames = ase_read(vasprun_path, index=":")
    valid, invalid_idx, invalid_reason = _filter_valid_aimd_frames(
        frames,
        min_dist=min_dist,
        covalent_scale=covalent_scale,
    )
    if invalid_idx >= 0:
        logger.warning(
            "VASP AIMD hit invalid geometry frame=%d reason=%s kept=%d/%d",
            invalid_idx,
            invalid_reason,
            len(valid),
            len(frames),
        )
    return valid


def _generate_aimd_dataset(
    structure_paths,
    labeler,
    output_dir: str,
    total_frames: int,
    sample_count: int,
    temperature: float,
    timestep_fs: float,
    friction: float,
    min_dist: float,
    covalent_scale: float,
):
    """Generate cold-start data by AIMD, keeping the full trajectory and a sampled subset."""
    from ase.io import read as ase_read, write as ase_write

    if total_frames < sample_count:
        raise ValueError("--aimd-total-frames must be >= --aimd-sample-count")
    if not hasattr(labeler, "_make_calculator"):
        raise ValueError(
            f"Cold-start AIMD requires an ASE-backed labeler, got {labeler.__class__.__name__}."
        )

    aimd_dir = os.path.join(output_dir, "aimd_runs")
    os.makedirs(aimd_dir, exist_ok=True)

    full_frames = []
    sampled_frames = []

    for i, path in enumerate(structure_paths):
        atoms = ase_read(path)
        run_dir = os.path.join(aimd_dir, f"seed_{i:04d}")
        logger.info(
            "Running cold-start AIMD [%d] %s: total_frames=%d sample_count=%d T=%.1fK dt=%.3ffs",
            i,
            os.path.basename(path),
            total_frames,
            sample_count,
            temperature,
            timestep_fs,
        )
        if labeler.__class__.__name__ == "VaspLabeler":
            frames = _run_vasp_aimd_trajectory(
                atoms,
                labeler,
                run_dir,
                total_frames=total_frames,
                temperature=temperature,
                timestep_fs=timestep_fs,
                min_dist=min_dist,
                covalent_scale=covalent_scale,
            )
        else:
            frames = _run_ase_aimd_trajectory(
                atoms,
                labeler,
                run_dir,
                total_frames=total_frames,
                temperature=temperature,
                timestep_fs=timestep_fs,
                friction=friction,
                min_dist=min_dist,
                covalent_scale=covalent_scale,
            )

        if len(frames) < sample_count:
            raise RuntimeError(
                f"AIMD for {path} produced only {len(frames)} valid frame(s), "
                f"but {sample_count} are required."
            )

        sample_indices = _sample_uniform_indices(len(frames), sample_count)
        source = os.path.basename(path)
        for idx, frame in enumerate(frames):
            frame.info["source"] = source
            frame.info["aimd_frame"] = idx
        selected = [frames[idx] for idx in sample_indices]
        for rank, frame_idx in enumerate(sample_indices):
            selected[rank].info["aimd_sample_rank"] = rank
            selected[rank].info["aimd_sample_frame"] = frame_idx

        full_frames.extend(frames)
        sampled_frames.extend(selected)
        logger.info(
            "AIMD [%d] %s: valid_frames=%d sampled=%d first=%d last=%d",
            i,
            source,
            len(frames),
            len(selected),
            sample_indices[0],
            sample_indices[-1],
        )

    full_path = os.path.join(output_dir, "aimd_full.xyz")
    sampled_path = os.path.join(output_dir, "train.xyz")
    unlabeled_path = os.path.join(output_dir, "unlabeled.xyz")
    ase_write(full_path, full_frames, format="extxyz")
    ase_write(sampled_path, sampled_frames, format="extxyz")
    ase_write(unlabeled_path, [atoms.copy() for atoms in sampled_frames], format="extxyz")
    logger.info(
        "AIMD cold start ready: full=%d -> %s ; sampled=%d -> %s",
        len(full_frames),
        full_path,
        len(sampled_frames),
        sampled_path,
    )
    return sampled_frames, full_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial training data from seed structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- structures & perturbation ----
    parser.add_argument(
        "--structures", type=str, nargs="+", required=True,
        help="One or more seed structure files, or a directory of .xyz/.cif files.",
    )
    parser.add_argument(
        "--cold-start-mode",
        type=str,
        default="perturb",
        choices=["perturb", "aimd"],
        help="Cold-start source: perturbations or AIMD trajectory sampling (default: perturb).",
    )
    parser.add_argument(
        "--n-perturb", type=int, default=10,
        help="Number of perturbed copies per seed structure (default: 10)",
    )
    parser.add_argument(
        "--rattle-std", type=float, default=0.05,
        help="Gaussian displacement σ in Å (default: 0.05; use 0.01–0.03 for crystals)",
    )
    parser.add_argument(
        "--cell-scale-range", type=float, default=0.0,
        help="±range for random cell scaling (default: 0 = disabled; use 0.02–0.05 for crystals)",
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.5,
        help="Minimum interatomic distance filter in Å (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--seed-relax", dest="seed_relax", action="store_true",
        help="Relax each cold-start seed structure before perturbation (default: enabled).",
    )
    parser.add_argument(
        "--no-seed-relax", dest="seed_relax", action="store_false",
        help="Disable seed relaxation and perturb the raw input structures directly.",
    )
    parser.set_defaults(seed_relax=True)
    parser.add_argument(
        "--seed-relax-fmax", type=float, default=0.05,
        help="Force threshold for cold-start seed relaxation (default: 0.05 eV/Ang).",
    )
    parser.add_argument(
        "--seed-relax-steps", type=int, default=200,
        help="Maximum optimizer steps for cold-start seed relaxation (default: 200).",
    )
    parser.add_argument(
        "--aimd-total-frames", type=int, default=1000,
        help="Minimum AIMD frames generated per seed (default: 1000).",
    )
    parser.add_argument(
        "--aimd-sample-count", type=int, default=100,
        help="Uniformly sampled AIMD frames kept per seed for train.xyz (default: 100).",
    )
    parser.add_argument(
        "--aimd-temperature", type=float, default=300.0,
        help="AIMD temperature in K (default: 300).",
    )
    parser.add_argument(
        "--aimd-timestep", type=float, default=0.5,
        help="AIMD timestep in fs (default: 0.5).",
    )
    parser.add_argument(
        "--aimd-friction", type=float, default=0.02,
        help="ASE fallback AIMD Langevin friction (default: 0.02). Ignored by internal VASP AIMD.",
    )
    parser.add_argument(
        "--aimd-covalent-scale", type=float, default=0.75,
        help="AIMD geometry guard based on covalent radii (default: 0.75).",
    )

    # ---- output & preprocessing ----
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory for preprocessed data (default: data)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9,
        help="Fraction of data used for training vs validation (default: 0.9)",
    )
    parser.add_argument(
        "--max-force-filter", type=float, default=None,
        help=(
            "Optional cold-start filter: discard labeled frames whose max per-atom "
            "force norm exceeds this threshold in eV/Ang."
        ),
    )
    parser.add_argument("--max-radius", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Only generate labeled XYZ; skip preprocessing into H5 format",
    )

    # ---- labeling ----
    parser.add_argument(
        "--label-type", type=str, required=True,
        choices=["pyscf", "vasp", "cp2k", "espresso", "gaussian", "orca"],
        help="DFT backend for labeling",
    )
    parser.add_argument("--label-n-workers", type=int, default=1)
    parser.add_argument("--label-threads-per-worker", type=int, default=None)
    parser.add_argument(
        "--label-error-handling", type=str, default="skip",
        choices=["raise", "skip"],
        help="Default: skip (discard structures where DFT fails)",
    )

    # ---- PySCF ----
    parser.add_argument("--pyscf-method", type=str, default="b3lyp")
    parser.add_argument("--pyscf-basis", type=str, default="6-31g*")
    parser.add_argument("--pyscf-charge", type=int, default=0)
    parser.add_argument("--pyscf-spin", type=int, default=0)
    parser.add_argument("--pyscf-max-memory", type=int, default=4000)
    parser.add_argument("--pyscf-conv-tol", type=float, default=1e-9)

    # ---- VASP ----
    parser.add_argument("--vasp-xc", type=str, default="PBE")
    parser.add_argument("--vasp-encut", type=float, default=None)
    parser.add_argument("--vasp-kpts", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--vasp-ediff", type=float, default=1e-6)
    parser.add_argument("--vasp-ismear", type=int, default=0)
    parser.add_argument("--vasp-sigma", type=float, default=0.05)
    parser.add_argument("--vasp-command", type=str, default=None)
    parser.add_argument(
        "--vasp-mpi-ranks", type=int, default=1,
        help="MPI ranks per VASP job. >1 prefixes --vasp-command with the Intel MPI launcher.",
    )
    parser.add_argument(
        "--vasp-mpi-launcher", type=str,
        default="/opt/intel/oneapi/mpi/latest/bin/mpirun",
        help="MPI launcher used when --vasp-mpi-ranks > 1 (default: Intel MPI mpirun).",
    )
    parser.add_argument(
        "--vasp-ncore", type=int, default=None,
        help="Optional INCAR NCORE for VASP performance tuning.",
    )
    parser.add_argument("--vasp-cleanup", action="store_true")

    # ---- CP2K ----
    parser.add_argument("--cp2k-xc", type=str, default="PBE")
    parser.add_argument("--cp2k-basis-set", type=str, default="DZVP-MOLOPT-SR-GTH")
    parser.add_argument("--cp2k-pseudo", type=str, default="auto")
    parser.add_argument("--cp2k-cutoff", type=float, default=400.0)
    parser.add_argument("--cp2k-max-scf", type=int, default=50)
    parser.add_argument("--cp2k-charge", type=float, default=0.0)
    parser.add_argument("--cp2k-command", type=str, default=None)
    parser.add_argument(
        "--cp2k-mpi-ranks", type=int, default=1,
        help="MPI ranks per CP2K job. >1 prefixes --cp2k-command with the MPI launcher.",
    )
    parser.add_argument(
        "--cp2k-mpi-launcher", type=str,
        default="/opt/intel/oneapi/mpi/latest/bin/mpirun",
        help="MPI launcher used when --cp2k-mpi-ranks > 1 (default: Intel MPI mpirun).",
    )
    parser.add_argument("--cp2k-cleanup", action="store_true")

    # ---- QE ----
    parser.add_argument("--qe-pseudo-dir", type=str, default=None)
    parser.add_argument("--qe-pseudopotentials", type=str, default=None)
    parser.add_argument("--qe-ecutwfc", type=float, default=60.0)
    parser.add_argument("--qe-ecutrho", type=float, default=None)
    parser.add_argument("--qe-kpts", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--qe-command", type=str, default=None)
    parser.add_argument("--qe-cleanup", action="store_true")

    # ---- Gaussian ----
    parser.add_argument("--gaussian-method", type=str, default="b3lyp")
    parser.add_argument("--gaussian-basis", type=str, default="6-31+G*")
    parser.add_argument("--gaussian-charge", type=int, default=0)
    parser.add_argument("--gaussian-mult", type=int, default=1)
    parser.add_argument("--gaussian-nproc", type=int, default=1)
    parser.add_argument("--gaussian-mem", type=str, default="4GB")
    parser.add_argument("--gaussian-command", type=str, default=None)
    parser.add_argument("--gaussian-cleanup", action="store_true")

    # ---- ORCA ----
    parser.add_argument("--orca-simpleinput", type=str, default="B3LYP def2-TZVP TightSCF")
    parser.add_argument("--orca-nproc", type=int, default=1)
    parser.add_argument("--orca-charge", type=int, default=0)
    parser.add_argument("--orca-mult", type=int, default=1)
    parser.add_argument("--orca-command", type=str, default=None)
    parser.add_argument("--orca-cleanup", action="store_true")

    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ---- 1. Resolve structures ----
    structure_paths = _resolve_structures(args.structures)
    logger.info(f"Seed structures: {len(structure_paths)}")
    for i, p in enumerate(structure_paths):
        logger.info(f"  [{i}] {p}")

    os.makedirs(args.output_dir, exist_ok=True)
    labeler = _build_labeler(args)

    if args.seed_relax:
        structure_paths = _relax_seed_structures(
            structure_paths,
            labeler,
            args.output_dir,
            fmax=args.seed_relax_fmax,
            steps=args.seed_relax_steps,
        )

    from ase.io import write as ase_write
    labeled_path = os.path.join(args.output_dir, "train.xyz")
    if args.cold_start_mode == "aimd":
        labeled_atoms, _ = _generate_aimd_dataset(
            structure_paths,
            labeler,
            output_dir=args.output_dir,
            total_frames=args.aimd_total_frames,
            sample_count=args.aimd_sample_count,
            temperature=args.aimd_temperature,
            timestep_fs=args.aimd_timestep,
            friction=args.aimd_friction,
            min_dist=args.min_dist,
            covalent_scale=args.aimd_covalent_scale,
        )
    else:
        # ---- 2. Generate perturbations ----
        all_atoms = generate_init_dataset(
            structure_paths,
            n_perturb=args.n_perturb,
            rattle_std=args.rattle_std,
            cell_scale_range=args.cell_scale_range,
            min_dist=args.min_dist,
            seed=args.seed,
        )

        unlabeled_path = os.path.join(args.output_dir, "unlabeled.xyz")
        ase_write(unlabeled_path, all_atoms, format="extxyz")
        logger.info(f"Wrote {len(all_atoms)} unlabeled frames to {unlabeled_path}")

        # ---- 3. DFT labeling ----
        work_dir = os.path.join(args.output_dir, "_label_work")
        os.makedirs(work_dir, exist_ok=True)

        logger.info(f"Labeling {len(all_atoms)} structures with {args.label_type}...")
        labeler.label(unlabeled_path, labeled_path, work_dir)

        from ase.io import read as ase_read
        labeled_atoms = ase_read(labeled_path, index=":")
        logger.info(f"Successfully labeled {len(labeled_atoms)} structures -> {labeled_path}")

    if len(labeled_atoms) == 0:
        logger.error("No structures were labeled successfully. Aborting.")
        return

    if args.max_force_filter is not None and args.max_force_filter > 0:
        unfiltered_path = os.path.join(args.output_dir, "train_unfiltered.xyz")
        ase_write(unfiltered_path, labeled_atoms, format="extxyz")
        filtered_atoms, dropped = _filter_labeled_atoms_by_force(
            labeled_atoms,
            args.max_force_filter,
        )
        if not filtered_atoms:
            raise RuntimeError(
                "All labeled cold-start structures were removed by --max-force-filter. "
                "Relax the threshold or reduce perturbation strength."
            )
        ase_write(labeled_path, filtered_atoms, format="extxyz")
        logger.info(
            "Force filter applied: kept=%d dropped=%d threshold=%.3f eV/Ang "
            "(unfiltered copy: %s)",
            len(filtered_atoms),
            len(dropped),
            args.max_force_filter,
            unfiltered_path,
        )
        if dropped:
            preview = ", ".join(
                f"{idx}:{max_force:.3f}" for idx, max_force in dropped[:10]
            )
            logger.info("Dropped frame indices (first 10): %s", preview)
        labeled_atoms = filtered_atoms

    # ---- 4. Preprocess ----
    if args.skip_preprocess:
        logger.info(
            f"Skipping preprocessing. Run manually:\n"
            f"  mff-preprocess --input-file {labeled_path} --output-dir {args.output_dir}"
        )
        return

    atomic_numbers = set()
    for a in labeled_atoms:
        atomic_numbers.update(a.get_atomic_numbers().tolist())
    atomic_keys = sorted(atomic_numbers)

    cmd = [
        sys.executable, "-m", "molecular_force_field.cli.preprocess",
        "--input-file", labeled_path,
        "--output-dir", args.output_dir,
        "--max-radius", str(args.max_radius),
        "--num-workers", str(args.num_workers),
        "--atomic-energy-keys",
    ] + [str(k) for k in atomic_keys] + [
        "--train-ratio", str(args.train_ratio),
        "--seed", str(args.seed),
    ]
    logger.info(f"Preprocessing...")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Preprocessing failed (exit {ret.returncode})")

    logger.info(
        f"\nDone! Initial dataset ready in {args.output_dir}/\n"
        f"  Total labeled: {len(labeled_atoms)} structures\n"
        f"  Cold-start mode: {args.cold_start_mode}\n"
        f"\nNext step — start active learning:\n"
        f"  mff-active-learn --data-dir {args.output_dir} "
        f"--init-structure {' '.join(structure_paths)} "
        f"--explore-type ase --label-type {args.label_type} ..."
    )


if __name__ == "__main__":
    main()
