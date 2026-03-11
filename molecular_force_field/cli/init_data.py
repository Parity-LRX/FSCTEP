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

import numpy as np

from molecular_force_field.active_learning.init_data import (
    generate_init_dataset,
)

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
            command=args.vasp_command,
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
            command=args.cp2k_command,
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

    # ---- output & preprocessing ----
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory for preprocessed data (default: data)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9,
        help="Fraction of data used for training vs validation (default: 0.9)",
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
    parser.add_argument("--vasp-cleanup", action="store_true")

    # ---- CP2K ----
    parser.add_argument("--cp2k-xc", type=str, default="PBE")
    parser.add_argument("--cp2k-basis-set", type=str, default="DZVP-MOLOPT-SR-GTH")
    parser.add_argument("--cp2k-pseudo", type=str, default="auto")
    parser.add_argument("--cp2k-cutoff", type=float, default=400.0)
    parser.add_argument("--cp2k-max-scf", type=int, default=50)
    parser.add_argument("--cp2k-charge", type=float, default=0.0)
    parser.add_argument("--cp2k-command", type=str, default=None)
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

    # ---- 2. Generate perturbations ----
    from ase.io import write as ase_write
    all_atoms = generate_init_dataset(
        structure_paths,
        n_perturb=args.n_perturb,
        rattle_std=args.rattle_std,
        cell_scale_range=args.cell_scale_range,
        min_dist=args.min_dist,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    unlabeled_path = os.path.join(args.output_dir, "unlabeled.xyz")
    ase_write(unlabeled_path, all_atoms, format="extxyz")
    logger.info(f"Wrote {len(all_atoms)} unlabeled frames to {unlabeled_path}")

    # ---- 3. DFT labeling ----
    labeler = _build_labeler(args)
    labeled_path = os.path.join(args.output_dir, "train.xyz")
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
        f"  From {len(structure_paths)} seed structure(s), {args.n_perturb} perturbations each\n"
        f"\nNext step — start active learning:\n"
        f"  mff-active-learn --data-dir {args.output_dir} "
        f"--init-structure {' '.join(structure_paths)} "
        f"--explore-type ase --label-type {args.label_type} ..."
    )


if __name__ == "__main__":
    main()
