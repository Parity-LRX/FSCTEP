"""CLI for active learning (mff-active-learn).

Single-stage (backward-compatible):
  mff-active-learn --explore-type ase --explore-mode md --label-type identity \\
      --md-temperature 300 --md-steps 1000 --n-iterations 5

Long-range / slab checkpoint through ASE active learning:
  mff-active-learn --explore-type ase --explore-mode md --label-type identity \\
      --init-structure slab.xyz --identity-checkpoint model_0.pth \\
      --init-checkpoint model_0.pth model_1.pth \\
      --device cpu --n-models 2 --n-iterations 1 --md-steps 2

Recommended long-range training/export setup:
  --long-range-mode reciprocal-spectral-v1
  --long-range-reciprocal-backend mesh_fft
  --long-range-green-mode poisson
  --long-range-energy-partition potential
  --long-range-assignment cic
  # slab only:
  --long-range-boundary slab
  --long-range-slab-padding-factor 2

PySCF labeling:
  mff-active-learn --explore-type ase --label-type pyscf \\
      --pyscf-method b3lyp --pyscf-basis 6-31g* \\
      --md-steps 500 --n-iterations 3

VASP labeling:
  mff-active-learn --explore-type ase --label-type vasp \\
      --vasp-xc PBE --vasp-encut 500 --vasp-kpts 4 4 4

CP2K labeling:
  mff-active-learn --explore-type ase --label-type cp2k \\
      --cp2k-xc PBE --cp2k-cutoff 600

Quantum Espresso labeling:
  mff-active-learn --explore-type ase --label-type espresso \\
      --qe-pseudo-dir /path/to/pseudos \\
      --qe-pseudopotentials '{"H":"H.pbe.UPF","O":"O.pbe.UPF"}' \\
      --qe-ecutwfc 60

Gaussian labeling:
  mff-active-learn --explore-type ase --label-type gaussian \\
      --gaussian-method b3lyp --gaussian-basis 6-31+G* --gaussian-nproc 8

ORCA labeling:
  mff-active-learn --explore-type ase --label-type orca \\
      --orca-simpleinput "B3LYP def2-TZVP TightSCF" --orca-nproc 8

User-script labeling (any DFT code via wrapper script):
  mff-active-learn --explore-type ase --label-type script --label-script ./my_dft.sh

Multi-stage via JSON:
  mff-active-learn --explore-type ase --label-type identity --stages stages.json

stages.json example (list of dicts):
  [
    {"name":"300K", "temperature":300, "nsteps":500, "max_iters":3,
     "level_f_lo":0.05, "level_f_hi":0.5, "conv_accuracy":0.9},
    {"name":"600K", "temperature":600, "nsteps":1000, "max_iters":3,
     "level_f_lo":0.05, "level_f_hi":0.5, "conv_accuracy":0.9}
  ]
"""

import argparse
import json
import logging
import os

from molecular_force_field.active_learning.exploration import run_ase_md, run_ase_neb
from molecular_force_field.active_learning.labeling import (
    CP2KLabeler,
    EspressoLabeler,
    GaussianLabeler,
    IdentityLabeler,
    LocalScriptLabeler,
    ORCALabeler,
    PySCFLabeler,
    ScriptLabeler,
    SlurmLabeler,
    VaspLabeler,
)
from molecular_force_field.active_learning.diversity_selector import DiversitySelector
from molecular_force_field.active_learning.loop import run_active_learning_loop
from molecular_force_field.active_learning.stage_scheduler import (
    StageScheduler,
    make_single_stage_scheduler,
)


def _resolve_init_structs(args):
    """Resolve one or more initial structures from CLI args or data_dir.

    Returns a list of structure file paths.
    """
    from typing import List

    if args.init_structure:
        structures: List[str] = []
        for path in args.init_structure:
            if os.path.isdir(path):
                for f in sorted(os.listdir(path)):
                    if f.endswith((".xyz", ".extxyz", ".cif", ".vasp")):
                        structures.append(os.path.join(path, f))
            elif os.path.exists(path):
                structures.append(path)
            else:
                raise FileNotFoundError(f"Init structure not found: {path}")
        if structures:
            return structures

    train_xyz = os.path.join(args.data_dir, "train.xyz")
    if os.path.exists(train_xyz):
        from ase.io import read, write
        atoms_list = read(train_xyz, index=":")
        if atoms_list:
            init_struct = os.path.join(args.work_dir, "init.xyz")
            os.makedirs(args.work_dir, exist_ok=True)
            write(init_struct, atoms_list[0])
            return [init_struct]

    proc_h5 = os.path.join(args.data_dir, "processed_train.h5")
    if os.path.exists(proc_h5):
        import h5py
        from ase import Atoms
        from ase.io import write
        with h5py.File(proc_h5, "r") as f:
            s0 = f["sample_0"]
            pos = s0["pos"][:]
            A = s0["A"][:].astype(int)
            cell = s0["cell"][:] if "cell" in s0 else None
        atoms = Atoms(numbers=A, positions=pos, cell=cell)
        init_struct = os.path.join(args.work_dir, "init.xyz")
        os.makedirs(args.work_dir, exist_ok=True)
        write(init_struct, atoms)
        return [init_struct]

    raise ValueError(
        "Provide --init-structure or ensure data_dir has train.xyz / processed_train.h5"
    )


def _resolve_mpi_command(command: str | None, mpi_ranks: int, mpi_launcher: str) -> str | None:
    if command is None:
        return None
    if mpi_ranks <= 1:
        return command
    lowered = command.lower()
    if "mpirun" in lowered or "mpiexec" in lowered:
        return command
    return f"{mpi_launcher} -np {mpi_ranks} {command}"


def main():
    parser = argparse.ArgumentParser(
        description="DPGen2-style active learning for molecular force field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- common ----
    parser.add_argument("--work-dir", type=str, default="al_work")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--init-structure", type=str, nargs="+", default=None,
        help=(
            "One or more initial structure files for MD exploration, "
            "or a directory containing .xyz/.cif files. "
            "When multiple structures are given, each iteration explores "
            "all structures in parallel and merges the trajectories."
        ),
    )
    parser.add_argument("--n-models", type=int, default=4)
    parser.add_argument("--no-pre-eval", action="store_true")
    parser.add_argument("--explore-type", type=str, required=True, choices=["ase", "lammps"])
    parser.add_argument("--explore-mode", type=str, default="md", choices=["md", "neb"])
    parser.add_argument("--label-type", type=str, default="script",
                        choices=["script", "identity", "pyscf",
                                 "vasp", "cp2k", "espresso", "gaussian", "orca",
                                 "local-script", "slurm"])
    parser.add_argument(
        "--label-n-workers", type=int, default=1,
        help=(
            "Number of parallel worker processes for DFT labeling. "
            "Each worker handles one structure independently. "
            "Default: 1 (serial). Set to e.g. 8 to run 8 DFT jobs concurrently."
        ),
    )
    parser.add_argument(
        "--label-error-handling", type=str, default="raise",
        choices=["raise", "skip"],
        help=(
            "What to do when a DFT calculation fails. "
            "'raise' (default): stop immediately. "
            "'skip': log the error and continue with remaining structures."
        ),
    )
    parser.add_argument(
        "--label-threads-per-worker", type=int, default=None,
        metavar="T",
        help=(
            "Number of threads each worker process may use internally "
            "(e.g. PySCF linear algebra, QE OpenMP threads). "
            "Rule of thumb: n_workers × T ≤ total CPU cores. "
            "Default: 1 when n_workers > 1 (avoid over-subscription), "
            "0 / auto when n_workers = 1."
        ),
    )
    parser.add_argument("--label-script", type=str, default=None)
    parser.add_argument("--identity-checkpoint", type=str, default=None)
    parser.add_argument(
        "--init-checkpoint", type=str, nargs="+", default=None,
        help=(
            "One or more checkpoints used to bootstrap active learning. "
            "When provided, iteration 0 skips training and directly explores "
            "with these checkpoint(s). Provide either 1 checkpoint "
            "(bootstrap iteration 0 will skip uncertainty gating and promote "
            "explored frames directly), or exactly --n-models checkpoints "
            "(full ensemble deviation is available in iteration 0)."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume an interrupted active-learning run from work_dir/al_state.json "
            "and reuse existing checkpoints / trajectories / labeled files under "
            "iterations/iter_*."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-radius", type=float, default=5.0)
    parser.add_argument("--atomic-energy-file", type=str, default=None)
    parser.add_argument("--neb-initial", type=str, default=None)
    parser.add_argument("--neb-final", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--dtype", type=str, default=None,
        choices=["float32", "float64", "float", "double"],
        help="Training dtype forwarded to mff-train (e.g. float32).",
    )
    parser.add_argument(
        "--tensor-product-mode", type=str, default=None,
        help=(
            "Tensor product mode for training (e.g. pure-cartesian-ictd, spherical). "
            "Passed to mff-train. If not set, mff-train uses its default."
        ),
    )
    parser.add_argument(
        "--long-range-mode", type=str, default=None,
        choices=["none", "latent-coulomb", "isolated-far-field-v1", "isolated-far-field-v2", "reciprocal-spectral-v1"],
        help="Forwarded to mff-train. Use this to keep long-range architecture stable across AL retrains.",
    )
    parser.add_argument(
        "--long-range-boundary", type=str, default=None,
        choices=["nonperiodic", "periodic", "slab"],
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--long-range-backend", type=str, default=None,
        choices=["dense_pairwise", "tree_fmm"],
        help="Forwarded to mff-train for latent-coulomb runs.",
    )
    parser.add_argument("--long-range-hidden-dim", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-filter-hidden-dim", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-kmax", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-mesh-size", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-slab-padding-factor", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-source-channels", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument(
        "--long-range-reciprocal-backend", type=str, default=None,
        choices=["direct_kspace", "mesh_fft"],
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--long-range-energy-partition", type=str, default=None,
        choices=["potential", "uniform"],
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--long-range-green-mode", type=str, default=None,
        choices=["poisson", "learned_poisson"],
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--long-range-assignment", type=str, default=None,
        choices=["cic", "tsc", "pcs"],
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--long-range-mesh-fft-full-ewald", dest="long_range_mesh_fft_full_ewald", action="store_true",
        help="Forwarded to mff-train. Enables the slower full Ewald correction path for mesh_fft.",
    )
    parser.add_argument(
        "--no-long-range-mesh-fft-full-ewald", dest="long_range_mesh_fft_full_ewald", action="store_false",
        help="Forwarded to mff-train. Keeps mesh_fft on the faster reciprocal-only path.",
    )
    parser.set_defaults(long_range_mesh_fft_full_ewald=None)
    parser.add_argument("--long-range-theta", type=float, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-leaf-size", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-multipole-order", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-far-source-dim", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-far-num-shells", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-far-shell-growth", type=float, default=None, help="Forwarded to mff-train.")
    parser.add_argument(
        "--long-range-far-tail", dest="long_range_far_tail", action="store_true",
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--no-long-range-far-tail", dest="long_range_far_tail", action="store_false",
        help="Forwarded to mff-train.",
    )
    parser.set_defaults(long_range_far_tail=None)
    parser.add_argument("--long-range-far-tail-bins", type=int, default=None, help="Forwarded to mff-train.")
    parser.add_argument("--long-range-far-stats", type=str, default=None, help="Forwarded to mff-train.")
    parser.add_argument(
        "--long-range-far-source-norm", dest="long_range_far_source_norm", action="store_true",
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--no-long-range-far-source-norm", dest="long_range_far_source_norm", action="store_false",
        help="Forwarded to mff-train.",
    )
    parser.set_defaults(long_range_far_source_norm=None)
    parser.add_argument("--long-range-far-gate-init", type=float, default=None, help="Forwarded to mff-train.")
    parser.add_argument(
        "--long-range-neutralize", dest="long_range_neutralize", action="store_true",
        help="Forwarded to mff-train.",
    )
    parser.add_argument(
        "--no-long-range-neutralize", dest="long_range_neutralize", action="store_false",
        help="Forwarded to mff-train.",
    )
    parser.set_defaults(long_range_neutralize=None)
    parser.add_argument(
        "--feature-spectral-assignment", type=str, default=None,
        choices=["cic", "tsc", "pcs"],
        help="Forwarded to mff-train for feature_spectral_mode=fft.",
    )
    parser.add_argument(
        "--external-tensor-rank", type=int, default=None,
        help=(
            "External tensor rank for conv1 injection (e.g. 1 for electric field). "
            "Requires --external-field-file. Only for pure-cartesian-ictd. "
            "Passed to mff-train."
        ),
    )
    parser.add_argument(
        "--external-field-file", type=str, default=None,
        help=(
            "Per-structure external field file (.npy, shape N×3 for rank-1). "
            "Requires --external-tensor-rank. Passed to mff-train."
        ),
    )
    parser.add_argument(
        "--explore-external-field", type=float, nargs='+', default=None,
        metavar='V',
        help=(
            "Uniform external field applied during MD exploration, model "
            "deviation, identity labeling, and auto-injected into H5 for "
            "training. Number of values must equal 3^rank (Cartesian tensor, "
            "row-major). Auto-sets --external-tensor-rank if not given.\n"
            "  rank 0 (1 value):  scalar field strength\n"
            "  rank 1 (3 values): Fx  Fy  Fz  (Cartesian x/y/z, e.g. electric field V/Å)\n"
            "  rank 2 (9 values): Txx Txy Txz  Tyx Tyy Tyz  Tzx Tzy Tzz  (3×3 row-major)\n"
            "  rank L (3^L values): full rank-L Cartesian tensor, row-major"
        ),
    )
    parser.add_argument(
        "--explore-n-workers", type=int, default=1,
        help=(
            "Number of parallel workers for multi-structure exploration. "
            "1 (default): sequential. "
            ">1: launch that many concurrent threads via ThreadPoolExecutor. "
            "Only has effect when multiple --init-structure paths are given."
        ),
    )
    parser.add_argument(
        "--exploration-aggressiveness", type=float, default=1.0,
        help=(
            "High-level sampling knob for active learning. "
            "1.0 = keep user/stage settings unchanged. "
            "<1.0 = safer exploration: lowers temperature, timestep, steps, "
            "and trust window while increasing Langevin damping. "
            ">1.0 = more aggressive exploration."
        ),
    )
    parser.add_argument(
        "--train-n-gpu", type=int, default=1,
        help=(
            "Number of GPUs for training each ensemble model. "
            "1 (default): single-process training (CPU or single GPU). "
            ">1: launches torchrun --nproc_per_node=N with --distributed."
        ),
    )
    parser.add_argument(
        "--train-max-parallel", type=int, default=0,
        help=(
            "Max ensemble models trained simultaneously. "
            "0 (default): auto = available_gpus // train_n_gpu. "
            "1: sequential (one model at a time). "
            "E.g. 8 GPUs + --train-n-gpu 2 → auto parallel = 4 models."
        ),
    )
    parser.add_argument(
        "--train-nnodes", type=int, default=1,
        help=(
            "Number of nodes for multi-node DDP training. "
            "1 (default): single-node. "
            ">1: multi-node (uses torchrun rendezvous; auto-detects SLURM)."
        ),
    )
    parser.add_argument(
        "--train-master-addr", type=str, default="auto",
        help=(
            "Master/rendezvous address for multi-node DDP. "
            "'auto' (default): resolves from SLURM or local hostname."
        ),
    )
    parser.add_argument(
        "--train-master-port", type=int, default=29500,
        help="Base rendezvous port for DDP (default: 29500).",
    )
    parser.add_argument(
        "--train-launcher", type=str, default="auto",
        choices=["auto", "local", "slurm"],
        help=(
            "Launcher for multi-node training. "
            "'auto' (default): uses 'slurm' if SLURM detected + nnodes>1, else 'local'. "
            "'slurm': wraps torchrun with 'srun --nodes=N --ntasks-per-node=1'. "
            "'local': torchrun only (user must start workers on other nodes)."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    # ---- PySCF labeler options ----
    parser.add_argument("--pyscf-method", type=str, default="b3lyp",
                        help="PySCF method: b3lyp, pbe, hf, mp2, etc. (default: b3lyp)")
    parser.add_argument("--pyscf-basis", type=str, default="sto-3g",
                        help="PySCF basis set: sto-3g, 6-31g*, def2-svp, etc. (default: sto-3g)")
    parser.add_argument("--pyscf-charge", type=int, default=0)
    parser.add_argument("--pyscf-spin", type=int, default=0,
                        help="2S (number of unpaired electrons, default 0)")
    parser.add_argument("--pyscf-max-memory", type=int, default=4000,
                        help="Max memory in MB for PySCF (default: 4000)")
    parser.add_argument("--pyscf-conv-tol", type=float, default=1e-9)

    # ---- VASP labeler options ----
    parser.add_argument("--vasp-xc", type=str, default="PBE",
                        help="VASP XC functional, e.g. PBE, LDA, HSE06 (default: PBE)")
    parser.add_argument("--vasp-encut", type=float, default=None,
                        help="VASP plane-wave cutoff in eV")
    parser.add_argument("--vasp-kpts", type=int, nargs=3, default=[1, 1, 1],
                        metavar=("NK1", "NK2", "NK3"),
                        help="k-point mesh (default: 1 1 1)")
    parser.add_argument("--vasp-ediff", type=float, default=1e-6,
                        help="VASP SCF convergence threshold in eV (default: 1e-6)")
    parser.add_argument("--vasp-ismear", type=int, default=0,
                        help="VASP smearing type: 0=Gaussian, -5=tetrahedron (default: 0)")
    parser.add_argument("--vasp-sigma", type=float, default=0.05,
                        help="VASP smearing width in eV (default: 0.05)")
    parser.add_argument("--vasp-command", type=str, default=None,
                        help="Override ASE_VASP_COMMAND, e.g. 'mpiexec -np 8 vasp_std'")
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
    parser.add_argument("--vasp-cleanup", action="store_true",
                        help="Remove per-structure VASP run directories after success")

    # ---- CP2K labeler options ----
    parser.add_argument("--cp2k-xc", type=str, default="PBE",
                        help="CP2K XC functional (default: PBE)")
    parser.add_argument("--cp2k-basis-set", type=str, default="DZVP-MOLOPT-SR-GTH",
                        help="CP2K Gaussian basis set (default: DZVP-MOLOPT-SR-GTH)")
    parser.add_argument("--cp2k-pseudo", type=str, default="auto",
                        help="CP2K pseudopotential name (default: auto)")
    parser.add_argument("--cp2k-cutoff", type=float, default=400.0,
                        help="CP2K plane-wave cutoff in Ry (default: 400)")
    parser.add_argument("--cp2k-max-scf", type=int, default=50,
                        help="CP2K max SCF iterations (default: 50)")
    parser.add_argument("--cp2k-charge", type=float, default=0.0,
                        help="CP2K total system charge (default: 0)")
    parser.add_argument("--cp2k-command", type=str, default=None,
                        help="Override ASE_CP2K_COMMAND")
    parser.add_argument(
        "--cp2k-mpi-ranks", type=int, default=1,
        help="MPI ranks per CP2K job. >1 prefixes --cp2k-command with the MPI launcher.",
    )
    parser.add_argument(
        "--cp2k-mpi-launcher", type=str,
        default="/opt/intel/oneapi/mpi/latest/bin/mpirun",
        help="MPI launcher used when --cp2k-mpi-ranks > 1 (default: Intel MPI mpirun).",
    )
    parser.add_argument("--cp2k-cleanup", action="store_true",
                        help="Remove per-structure CP2K run directories after success")

    # ---- Quantum Espresso labeler options ----
    parser.add_argument("--qe-pseudo-dir", type=str, default=None,
                        help="Directory containing QE pseudopotential .UPF files")
    parser.add_argument(
        "--qe-pseudopotentials", type=str, default=None,
        help=(
            'JSON string mapping element → pseudopotential filename, e.g. '
            '\'{"H":"H.pbe.UPF","O":"O.pbe.UPF"}\''
        ),
    )
    parser.add_argument("--qe-ecutwfc", type=float, default=60.0,
                        help="QE wavefunction kinetic energy cutoff in Ry (default: 60)")
    parser.add_argument("--qe-ecutrho", type=float, default=None,
                        help="QE charge density cutoff in Ry (default: 4*ecutwfc)")
    parser.add_argument("--qe-kpts", type=int, nargs=3, default=[1, 1, 1],
                        metavar=("NK1", "NK2", "NK3"),
                        help="k-point mesh for QE (default: 1 1 1)")
    parser.add_argument("--qe-command", type=str, default=None,
                        help="QE pw.x command, e.g. 'mpirun -np 8 pw.x -in PREFIX.pwi > PREFIX.pwo'")
    parser.add_argument("--qe-cleanup", action="store_true",
                        help="Remove per-structure QE run directories after success")

    # ---- Gaussian labeler options ----
    parser.add_argument("--gaussian-method", type=str, default="b3lyp",
                        help="Gaussian level of theory (default: b3lyp)")
    parser.add_argument("--gaussian-basis", type=str, default="6-31+G*",
                        help="Gaussian basis set (default: 6-31+G*)")
    parser.add_argument("--gaussian-charge", type=int, default=0)
    parser.add_argument("--gaussian-mult", type=int, default=1,
                        help="Gaussian spin multiplicity 2S+1 (default: 1)")
    parser.add_argument("--gaussian-nproc", type=int, default=1,
                        help="Number of CPU cores for Gaussian %%nprocshared (default: 1)")
    parser.add_argument("--gaussian-mem", type=str, default="4GB",
                        help="Gaussian memory allocation (default: 4GB)")
    parser.add_argument("--gaussian-command", type=str, default=None,
                        help="Override Gaussian command, e.g. 'g16 < PREFIX.com > PREFIX.log'")
    parser.add_argument("--gaussian-cleanup", action="store_true",
                        help="Remove per-structure Gaussian run directories after success")

    # ---- ORCA labeler options ----
    parser.add_argument("--orca-simpleinput", type=str,
                        default="B3LYP def2-TZVP TightSCF",
                        help="ORCA simple-input line after '!' (default: 'B3LYP def2-TZVP TightSCF')")
    parser.add_argument("--orca-nproc", type=int, default=1,
                        help="Number of CPU cores for ORCA %%pal (default: 1)")
    parser.add_argument("--orca-charge", type=int, default=0)
    parser.add_argument("--orca-mult", type=int, default=1,
                        help="ORCA spin multiplicity 2S+1 (default: 1)")
    parser.add_argument("--orca-command", type=str, default=None,
                        help="Full path to ORCA executable")
    parser.add_argument("--orca-cleanup", action="store_true",
                        help="Remove per-structure ORCA run directories after success")

    # ---- local-script labeler options ----
    parser.add_argument(
        "--local-script-template", type=str, default=None,
        help=(
            "Path to a bash script template for --label-type local-script. "
            "Uses the same placeholder format as --slurm-template: "
            "{run_dir} {input_xyz} {output_xyz} {job_name}. "
            "The script is executed locally (no job scheduler). "
            "Compatible with --label-n-workers for parallel execution."
        ),
    )
    parser.add_argument(
        "--local-script-bash", type=str, default="bash",
        help="Bash interpreter to use (default: bash)",
    )
    parser.add_argument("--local-script-cleanup", action="store_true",
                        help="Remove per-structure run directories after success")

    # ---- SLURM labeler options ----
    parser.add_argument(
        "--slurm-template", type=str, default=None,
        help=(
            "Path to SLURM job script template (required for --label-type slurm). "
            "Placeholders: {run_dir} {input_xyz} {output_xyz} {job_name} "
            "{partition} {nodes} {ntasks} {time} {mem}. "
            "Any other {key} in the script is left unchanged."
        ),
    )
    parser.add_argument("--slurm-partition", type=str, default="cpu",
                        help="SLURM partition / queue (default: cpu)")
    parser.add_argument("--slurm-nodes", type=int, default=1,
                        help="Nodes per job (default: 1)")
    parser.add_argument("--slurm-ntasks", type=int, default=32,
                        help="--ntasks-per-node per job (default: 32)")
    parser.add_argument("--slurm-time", type=str, default="02:00:00",
                        help="Wall-clock time limit per job (default: 02:00:00)")
    parser.add_argument("--slurm-mem", type=str, default="64G",
                        help="Memory per job (default: 64G)")
    parser.add_argument("--slurm-max-concurrent", type=int, default=200,
                        help=(
                            "Max jobs in SLURM queue at once. "
                            "Submission is throttled when this limit is reached (default: 200)."
                        ))
    parser.add_argument("--slurm-poll-interval", type=int, default=30,
                        help="Seconds between squeue status polls (default: 30)")
    parser.add_argument(
        "--slurm-extra", type=str, nargs="*", default=None,
        metavar="ARG",
        help=(
            "Extra sbatch arguments, e.g. "
            "--slurm-extra --account=myproject --qos=high"
        ),
    )
    parser.add_argument("--slurm-cleanup", action="store_true",
                        help="Remove per-structure run directories after success")

    # ---- multi-stage: JSON file ----
    parser.add_argument(
        "--stages", type=str, default=None,
        help=(
            "Path to JSON file defining multiple exploration stages. "
            "When provided, single-stage flags (--md-*, --n-iterations, etc.) are ignored. "
            "See module docstring for format."
        ),
    )

    # ---- single-stage (kept for backward compat) ----
    parser.add_argument("--n-iterations", type=int, default=20,
                        help="Max iterations (single-stage mode only)")
    parser.add_argument("--level-f-lo", type=float, default=0.05)
    parser.add_argument("--level-f-hi", type=float, default=0.50)
    parser.add_argument("--conv-accuracy", type=float, default=0.9)
    parser.add_argument("--md-temperature", type=float, default=300.0)
    parser.add_argument("--md-steps", type=int, default=10000)
    parser.add_argument("--md-timestep", type=float, default=1.0)
    parser.add_argument("--md-friction", type=float, default=0.01)
    parser.add_argument(
        "--md-relax-fmax", type=float, default=0.0,
        help=(
            "Optional model-side pre-relaxation force threshold before MD. "
            "Default: 0.0 (disabled) to preserve PES sampling."
        ),
    )
    parser.add_argument("--md-log-interval", type=int, default=10)
    parser.add_argument(
        "--geometry-min-dist", type=float, default=0.5,
        help=(
            "Absolute minimum allowed interatomic distance in Angstrom during "
            "active-learning exploration/candidate filtering. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--geometry-covalent-scale", type=float, default=0.75,
        help=(
            "Chemistry-aware pair-distance threshold as "
            "scale * (r_cov_i + r_cov_j). Applied together with "
            "--geometry-min-dist; larger of the two wins. Set 0 to disable."
        ),
    )

    # ---- diversity selection (Layer 2) ----
    parser.add_argument(
        "--diversity-metric", type=str, default="soap",
        choices=["soap", "devi_hist", "none"],
        help=(
            "Fingerprint for diversity sub-selection of candidates. "
            "'soap' (default, requires dscribe): SOAP average descriptor + FPS. "
            "'devi_hist': per-atom force-deviation histogram + FPS (zero extra inference). "
            "'none': disable diversity filtering."
        ),
    )
    parser.add_argument(
        "--max-candidates-per-iter", type=int, default=50,
        help=(
            "Max candidates to keep per iteration after diversity selection. "
            "Only effective when --diversity-metric is not 'none'. (default: 50)"
        ),
    )
    parser.add_argument("--soap-rcut", type=float, default=5.0,
                        help="SOAP cutoff radius in Angstrom (default: 5.0)")
    parser.add_argument("--soap-nmax", type=int, default=8,
                        help="SOAP radial basis expansion order (default: 8)")
    parser.add_argument("--soap-lmax", type=int, default=6,
                        help="SOAP angular expansion order (default: 6)")
    parser.add_argument("--soap-sigma", type=float, default=0.5,
                        help="SOAP Gaussian smearing width (default: 0.5)")
    parser.add_argument("--devi-hist-bins", type=int, default=32,
                        help="Number of bins for devi_hist fingerprint (default: 32)")

    # ---- fail frame handling (Layer 0) ----
    parser.add_argument(
        "--fail-strategy", type=str, default="discard",
        choices=["discard", "sample_topk"],
        help=(
            "How to handle fail frames (max_devi_f >= level_f_hi). "
            "'discard' (default): drop all fail frames. "
            "'sample_topk': promote the least extreme fail frames into candidates."
        ),
    )
    parser.add_argument(
        "--fail-max-select", type=int, default=10,
        help="Number of fail frames to promote when --fail-strategy=sample_topk (default: 10)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    e0_path = args.atomic_energy_file or os.path.join(args.data_dir, "fitted_E0.csv")
    init_structs = _resolve_init_structs(args)

    # ------------------------------------------------------------------ #
    # Build StageScheduler
    # ------------------------------------------------------------------ #
    if args.stages:
        with open(args.stages) as f:
            stage_data = json.load(f)
        if isinstance(stage_data, dict) and "stages" in stage_data:
            stage_data = stage_data["stages"]
        scheduler = StageScheduler.from_dicts(stage_data)
        logging.info(f"Loaded {len(scheduler)} stage(s) from {args.stages}")
    else:
        scheduler = make_single_stage_scheduler(
            temperature=args.md_temperature,
            timestep=args.md_timestep,
            nsteps=args.md_steps,
            friction=args.md_friction,
            relax_fmax=args.md_relax_fmax,
            log_interval=args.md_log_interval,
            level_f_lo=args.level_f_lo,
            level_f_hi=args.level_f_hi,
            conv_accuracy=args.conv_accuracy,
            max_iters=args.n_iterations,
            name="stage_0",
        )

    if args.exploration_aggressiveness <= 0:
        raise ValueError("--exploration-aggressiveness must be > 0")
    if args.exploration_aggressiveness != 1.0:
        factor = args.exploration_aggressiveness
        for i, stage in enumerate(scheduler.stages):
            base_temperature = stage.temperature
            base_timestep = stage.timestep
            base_nsteps = stage.nsteps
            base_friction = stage.friction
            base_level_f_lo = stage.level_f_lo
            base_level_f_hi = stage.level_f_hi

            stage.temperature = base_temperature * factor
            stage.timestep = base_timestep * factor
            stage.nsteps = max(1, int(round(base_nsteps * factor)))
            stage.friction = base_friction / factor
            stage.level_f_lo = base_level_f_lo * factor
            stage.level_f_hi = max(stage.level_f_lo, base_level_f_hi * factor)

            logging.info(
                "Applied exploration aggressiveness %.3f to stage [%d] %s: "
                "T %.3f->%.3f K, dt %.4f->%.4f fs, steps %d->%d, "
                "friction %.4f->%.4f, level_f [%.4f, %.4f]->[%.4f, %.4f]",
                factor,
                i,
                stage.name or f"stage_{i}",
                base_temperature,
                stage.temperature,
                base_timestep,
                stage.timestep,
                base_nsteps,
                stage.nsteps,
                base_friction,
                stage.friction,
                base_level_f_lo,
                base_level_f_hi,
                stage.level_f_lo,
                stage.level_f_hi,
            )

    # ------------------------------------------------------------------ #
    # Build explore_fn
    # Accepts **kwargs so loop.py can pass input_structure / output_traj
    # for multi-structure parallel exploration.
    # ------------------------------------------------------------------ #
    efield = args.explore_external_field  # None or [fx, fy, fz]

    def explore_fn(iter_idx, checkpoint_path, stage, **kwargs):
        struct = kwargs.get("input_structure", init_structs[0])
        out_traj = kwargs.get(
            "output_traj",
            os.path.join(
                args.work_dir, "iterations", f"iter_{iter_idx}", "explore_traj.xyz"
            ),
        )
        os.makedirs(os.path.dirname(out_traj), exist_ok=True)
        if args.explore_type == "ase":
            if args.explore_mode == "md":
                return run_ase_md(
                    checkpoint_path=checkpoint_path,
                    input_structure=struct,
                    output_traj=out_traj,
                    device=args.device,
                    max_radius=args.max_radius,
                    atomic_energy_file=e0_path,
                    temperature=stage.temperature,
                    nsteps=stage.nsteps,
                    timestep=stage.timestep,
                    friction=stage.friction,
                    relax_fmax=stage.relax_fmax,
                    log_interval=stage.log_interval,
                    external_field=efield,
                    min_dist=args.geometry_min_dist,
                    covalent_scale=args.geometry_covalent_scale,
                )
            else:
                if not args.neb_initial or not args.neb_final:
                    raise ValueError("NEB requires --neb-initial and --neb-final")
                return run_ase_neb(
                    checkpoint_path=checkpoint_path,
                    initial_xyz=args.neb_initial,
                    final_xyz=args.neb_final,
                    output_traj=out_traj,
                    device=args.device,
                    max_radius=args.max_radius,
                    atomic_energy_file=e0_path,
                    external_field=efield,
                )
        else:
            raise NotImplementedError("LAMMPS: use run_lammps_md with custom template")

    # ------------------------------------------------------------------ #
    # Build label_fn
    # ------------------------------------------------------------------ #
    if args.label_type == "script":
        if not args.label_script:
            raise ValueError("--label-script required for --label-type script")
        labeler = ScriptLabeler(args.label_script)

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "pyscf":
        labeler = PySCFLabeler(
            method=args.pyscf_method,
            basis=args.pyscf_basis,
            charge=args.pyscf_charge,
            spin=args.pyscf_spin,
            max_memory=args.pyscf_max_memory,
            conv_tol=args.pyscf_conv_tol,
            n_workers=args.label_n_workers,
            threads_per_worker=args.label_threads_per_worker,
            error_handling=args.label_error_handling,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "vasp":
        vasp_kwargs = {}
        if args.vasp_ncore is not None:
            vasp_kwargs["ncore"] = args.vasp_ncore
        labeler = VaspLabeler(
            xc=args.vasp_xc,
            kpts=tuple(args.vasp_kpts),
            encut=args.vasp_encut,
            ediff=args.vasp_ediff,
            ismear=args.vasp_ismear,
            sigma=args.vasp_sigma,
            command=_resolve_mpi_command(
                args.vasp_command,
                args.vasp_mpi_ranks,
                args.vasp_mpi_launcher,
            ),
            vasp_kwargs=vasp_kwargs,
            n_workers=args.label_n_workers,
            threads_per_worker=args.label_threads_per_worker,
            cleanup=args.vasp_cleanup,
            error_handling=args.label_error_handling,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "cp2k":
        labeler = CP2KLabeler(
            xc=args.cp2k_xc,
            basis_set=args.cp2k_basis_set,
            pseudo_potential=args.cp2k_pseudo,
            cutoff=args.cp2k_cutoff,
            max_scf=args.cp2k_max_scf,
            charge=args.cp2k_charge,
            command=_resolve_mpi_command(
                args.cp2k_command,
                args.cp2k_mpi_ranks,
                args.cp2k_mpi_launcher,
            ),
            n_workers=args.label_n_workers,
            threads_per_worker=args.label_threads_per_worker,
            cleanup=args.cp2k_cleanup,
            error_handling=args.label_error_handling,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "espresso":
        if not args.qe_pseudo_dir:
            raise ValueError("--qe-pseudo-dir is required for --label-type espresso")
        if not args.qe_pseudopotentials:
            raise ValueError("--qe-pseudopotentials is required for --label-type espresso")
        pseudopotentials = json.loads(args.qe_pseudopotentials)
        input_data: dict = {"ecutwfc": args.qe_ecutwfc}
        if args.qe_ecutrho is not None:
            input_data["ecutrho"] = args.qe_ecutrho
        labeler = EspressoLabeler(
            pseudopotentials=pseudopotentials,
            pseudo_dir=args.qe_pseudo_dir,
            input_data=input_data,
            kpts=tuple(args.qe_kpts),
            command=args.qe_command,
            n_workers=args.label_n_workers,
            threads_per_worker=args.label_threads_per_worker,
            cleanup=args.qe_cleanup,
            error_handling=args.label_error_handling,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "gaussian":
        labeler = GaussianLabeler(
            method=args.gaussian_method,
            basis=args.gaussian_basis,
            charge=args.gaussian_charge,
            mult=args.gaussian_mult,
            nproc=args.gaussian_nproc,
            mem=args.gaussian_mem,
            command=args.gaussian_command,
            n_workers=args.label_n_workers,
            threads_per_worker=args.label_threads_per_worker,
            cleanup=args.gaussian_cleanup,
            error_handling=args.label_error_handling,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "orca":
        labeler = ORCALabeler(
            simpleinput=args.orca_simpleinput,
            blocks=f"%pal nprocs {args.orca_nproc} end",
            charge=args.orca_charge,
            mult=args.orca_mult,
            orca_command=args.orca_command,
            n_workers=args.label_n_workers,
            threads_per_worker=args.label_threads_per_worker,
            cleanup=args.orca_cleanup,
            error_handling=args.label_error_handling,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "local-script":
        if not args.local_script_template:
            raise ValueError("--local-script-template is required for --label-type local-script")
        labeler = LocalScriptLabeler(
            script_template=args.local_script_template,
            n_workers=args.label_n_workers,
            threads_per_worker=args.label_threads_per_worker,
            cleanup=args.local_script_cleanup,
            error_handling=args.label_error_handling,
            bash=args.local_script_bash,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    elif args.label_type == "slurm":
        if not args.slurm_template:
            raise ValueError("--slurm-template is required for --label-type slurm")
        labeler = SlurmLabeler(
            job_script_template=args.slurm_template,
            partition=args.slurm_partition,
            nodes=args.slurm_nodes,
            ntasks_per_node=args.slurm_ntasks,
            time_limit=args.slurm_time,
            mem=args.slurm_mem,
            max_concurrent=args.slurm_max_concurrent,
            poll_interval=args.slurm_poll_interval,
            sbatch_extra=args.slurm_extra or [],
            cleanup=args.slurm_cleanup,
            error_handling=args.label_error_handling,
        )

        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    else:
        def label_fn(candidate_path, output_path, work_dir, checkpoint_path=None):
            ckpt = checkpoint_path or args.identity_checkpoint or os.path.join(
                work_dir, "checkpoint", "model_0_spherical.pth"
            )
            labeler = IdentityLabeler(
                checkpoint_path=ckpt,
                device=args.device,
                max_radius=args.max_radius,
                atomic_energy_file=e0_path,
                external_field=efield,
            )
            return labeler.label(candidate_path, output_path, work_dir, checkpoint_path)

    # ------------------------------------------------------------------ #
    # External field validation & auto-inference
    # ------------------------------------------------------------------ #
    if args.explore_external_field:
        from molecular_force_field.active_learning.data_merge import (
            external_field_tensor_shape,
            _inject_external_field_into_h5,
        )
        shape = external_field_tensor_shape(len(args.explore_external_field))
        inferred_rank = len(shape) if shape != (1,) else 0
        if args.external_tensor_rank is None:
            args.external_tensor_rank = inferred_rank
            logging.info(
                f"Auto-set --external-tensor-rank={inferred_rank} from "
                f"--explore-external-field ({len(args.explore_external_field)} values)"
            )
        elif args.external_tensor_rank != inferred_rank:
            raise ValueError(
                f"--external-tensor-rank={args.external_tensor_rank} conflicts with "
                f"--explore-external-field ({len(args.explore_external_field)} values → rank {inferred_rank})"
            )
        for prefix in ("train", "val"):
            hp = os.path.join(args.data_dir, f"processed_{prefix}.h5")
            if os.path.exists(hp):
                _inject_external_field_into_h5(hp, args.explore_external_field)

    if args.external_field_file and not args.external_tensor_rank:
        raise ValueError("--external-field-file requires --external-tensor-rank (e.g. 1 for electric field)")
    if args.external_tensor_rank and args.tensor_product_mode not in {
        "pure-cartesian-ictd",
        "pure-cartesian-ictd-o3",
        "pure-cartesian-sparse",
        "pure-cartesian-sparse-save",
    }:
        raise ValueError(
            "--external-tensor-rank only supported for --tensor-product-mode "
            "pure-cartesian-ictd, pure-cartesian-ictd-o3, pure-cartesian-sparse, or pure-cartesian-sparse-save"
        )

    # ------------------------------------------------------------------ #
    # Extra train args
    # ------------------------------------------------------------------ #
    train_args = []
    if args.epochs is not None:
        train_args.extend(["--epochs", str(args.epochs)])
    if args.device is not None:
        train_args.extend(["--device", args.device])
    if args.dtype is not None:
        train_args.extend(["--dtype", args.dtype])
    if args.tensor_product_mode is not None:
        train_args.extend(["--tensor-product-mode", args.tensor_product_mode])
    if args.long_range_mode is not None:
        train_args.extend(["--long-range-mode", args.long_range_mode])
    if args.long_range_boundary is not None:
        train_args.extend(["--long-range-boundary", args.long_range_boundary])
    if args.long_range_backend is not None:
        train_args.extend(["--long-range-backend", args.long_range_backend])
    if args.long_range_hidden_dim is not None:
        train_args.extend(["--long-range-hidden-dim", str(args.long_range_hidden_dim)])
    if args.long_range_filter_hidden_dim is not None:
        train_args.extend(["--long-range-filter-hidden-dim", str(args.long_range_filter_hidden_dim)])
    if args.long_range_kmax is not None:
        train_args.extend(["--long-range-kmax", str(args.long_range_kmax)])
    if args.long_range_mesh_size is not None:
        train_args.extend(["--long-range-mesh-size", str(args.long_range_mesh_size)])
    if args.long_range_slab_padding_factor is not None:
        train_args.extend(["--long-range-slab-padding-factor", str(args.long_range_slab_padding_factor)])
    if args.long_range_source_channels is not None:
        train_args.extend(["--long-range-source-channels", str(args.long_range_source_channels)])
    if args.long_range_reciprocal_backend is not None:
        train_args.extend(["--long-range-reciprocal-backend", args.long_range_reciprocal_backend])
    if args.long_range_energy_partition is not None:
        train_args.extend(["--long-range-energy-partition", args.long_range_energy_partition])
    if args.long_range_green_mode is not None:
        train_args.extend(["--long-range-green-mode", args.long_range_green_mode])
    if args.long_range_assignment is not None:
        train_args.extend(["--long-range-assignment", args.long_range_assignment])
    if args.long_range_mesh_fft_full_ewald is True:
        train_args.append("--long-range-mesh-fft-full-ewald")
    elif args.long_range_mesh_fft_full_ewald is False:
        train_args.append("--no-long-range-mesh-fft-full-ewald")
    if args.long_range_theta is not None:
        train_args.extend(["--long-range-theta", str(args.long_range_theta)])
    if args.long_range_leaf_size is not None:
        train_args.extend(["--long-range-leaf-size", str(args.long_range_leaf_size)])
    if args.long_range_multipole_order is not None:
        train_args.extend(["--long-range-multipole-order", str(args.long_range_multipole_order)])
    if args.long_range_far_source_dim is not None:
        train_args.extend(["--long-range-far-source-dim", str(args.long_range_far_source_dim)])
    if args.long_range_far_num_shells is not None:
        train_args.extend(["--long-range-far-num-shells", str(args.long_range_far_num_shells)])
    if args.long_range_far_shell_growth is not None:
        train_args.extend(["--long-range-far-shell-growth", str(args.long_range_far_shell_growth)])
    if args.long_range_far_tail is True:
        train_args.append("--long-range-far-tail")
    elif args.long_range_far_tail is False:
        train_args.append("--no-long-range-far-tail")
    if args.long_range_far_tail_bins is not None:
        train_args.extend(["--long-range-far-tail-bins", str(args.long_range_far_tail_bins)])
    if args.long_range_far_stats is not None:
        train_args.extend(["--long-range-far-stats", args.long_range_far_stats])
    if args.long_range_far_source_norm is True:
        train_args.append("--long-range-far-source-norm")
    elif args.long_range_far_source_norm is False:
        train_args.append("--no-long-range-far-source-norm")
    if args.long_range_far_gate_init is not None:
        train_args.extend(["--long-range-far-gate-init", str(args.long_range_far_gate_init)])
    if args.long_range_neutralize is True:
        train_args.append("--long-range-neutralize")
    elif args.long_range_neutralize is False:
        train_args.append("--no-long-range-neutralize")
    if args.feature_spectral_assignment is not None:
        train_args.extend(["--feature-spectral-assignment", args.feature_spectral_assignment])
    if args.external_tensor_rank is not None:
        train_args.extend(["--external-tensor-rank", str(args.external_tensor_rank)])
    if args.external_field_file is not None:
        train_args.extend(["--external-field-file", os.path.abspath(args.external_field_file)])

    # ------------------------------------------------------------------ #
    # Build DiversitySelector (Layer 2)
    # ------------------------------------------------------------------ #
    if args.diversity_metric != "none":
        diversity_selector = DiversitySelector(
            metric=args.diversity_metric,
            max_select=args.max_candidates_per_iter,
            soap_rcut=args.soap_rcut,
            soap_nmax=args.soap_nmax,
            soap_lmax=args.soap_lmax,
            soap_sigma=args.soap_sigma,
            devi_hist_bins=args.devi_hist_bins,
        )
    else:
        diversity_selector = None

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    run_active_learning_loop(
        work_dir=args.work_dir,
        data_dir=args.data_dir,
        explore_fn=explore_fn,
        label_fn=label_fn,
        n_models=args.n_models,
        pre_eval=not args.no_pre_eval,
        explore_structure=init_structs[0],
        device=args.device,
        atomic_energy_file=e0_path,
        max_radius=args.max_radius,
        train_args=train_args if train_args else None,
        scheduler=scheduler,
        diversity_selector=diversity_selector,
        fail_strategy=args.fail_strategy,
        fail_max_select=args.fail_max_select,
        explore_structures=init_structs,
        explore_n_workers=args.explore_n_workers,
        initial_checkpoint_paths=args.init_checkpoint,
        resume=args.resume,
        n_gpu=args.train_n_gpu,
        max_parallel=args.train_max_parallel,
        nnodes=args.train_nnodes,
        master_addr=args.train_master_addr,
        master_port=args.train_master_port,
        launcher=args.train_launcher,
        external_field=efield,
        geometry_min_dist=args.geometry_min_dist,
        geometry_covalent_scale=args.geometry_covalent_scale,
    )


if __name__ == "__main__":
    main()
