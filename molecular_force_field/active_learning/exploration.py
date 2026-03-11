"""Exploration module: ASE or LAMMPS (USER-MFFTORCH) for MD/NEB."""

import logging
import os
import subprocess
import sys
from typing import Optional

from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase import units

logger = logging.getLogger(__name__)


def run_ase_md(
    checkpoint_path: str,
    input_structure: str,
    output_traj: str,
    device: str = "cuda",
    max_radius: float = 5.0,
    atomic_energy_file: Optional[str] = None,
    temperature: float = 300.0,
    timestep: float = 1.0,
    nsteps: int = 10000,
    friction: float = 0.01,
    relax_fmax: float = 0.05,
    log_interval: int = 10,
    external_field: Optional[list] = None,
) -> str:
    """
    Run MD with ASE + MyE3NNCalculator. Returns path to output trajectory.

    Parameters
    ----------
    external_field : list or None
        Global external field vector (e.g. ``[0, 0, 0.01]`` for a uniform
        electric field in V/Å).  Passed to ``MyE3NNCalculator`` so the model
        evaluates energy/forces under this field at every MD step.
    """
    import torch
    from molecular_force_field.evaluation.calculator import MyE3NNCalculator
    from molecular_force_field.active_learning.model_loader import build_e3trans_from_checkpoint

    dev = torch.device(device)
    e3trans, config = build_e3trans_from_checkpoint(
        checkpoint_path, dev, atomic_energy_file=atomic_energy_file
    )
    ref_dict = {
        k.item(): v.item()
        for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
    }
    ext_tensor = None
    if external_field is not None:
        from molecular_force_field.active_learning.data_merge import external_field_tensor_shape
        shape = external_field_tensor_shape(len(external_field))
        ext_tensor = torch.tensor(external_field, dtype=torch.float64, device=dev).reshape(shape)
    calc = MyE3NNCalculator(e3trans, ref_dict, dev, max_radius, external_tensor=ext_tensor)
    atoms = read(input_structure)
    atoms.calc = calc
    if relax_fmax > 0:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=relax_fmax)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Langevin(
        atoms,
        timestep * units.fs,
        temperature_K=temperature,
        friction=friction,
    )
    dyn.attach(lambda: write(output_traj, atoms, append=True), interval=log_interval)
    dyn.run(nsteps)
    logger.info(f"ASE MD done: {output_traj}")
    return output_traj


def run_ase_neb(
    checkpoint_path: str,
    initial_xyz: str,
    final_xyz: str,
    output_traj: str,
    device: str = "cuda",
    max_radius: float = 5.0,
    atomic_energy_file: Optional[str] = None,
    n_images: int = 10,
    fmax: float = 0.05,
    external_field: Optional[list] = None,
) -> str:
    """Run NEB with ASE. Returns path to output trajectory."""
    import torch
    from ase.mep import NEB
    from ase.optimize import FIRE
    from molecular_force_field.evaluation.calculator import MyE3NNCalculator
    from molecular_force_field.active_learning.model_loader import build_e3trans_from_checkpoint

    dev = torch.device(device)
    e3trans, config = build_e3trans_from_checkpoint(
        checkpoint_path, dev, atomic_energy_file=atomic_energy_file
    )
    ref_dict = {
        k.item(): v.item()
        for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
    }
    ext_tensor = None
    if external_field is not None:
        from molecular_force_field.active_learning.data_merge import external_field_tensor_shape
        shape = external_field_tensor_shape(len(external_field))
        ext_tensor = torch.tensor(external_field, dtype=torch.float64, device=dev).reshape(shape)
    calc = MyE3NNCalculator(e3trans, ref_dict, dev, max_radius, external_tensor=ext_tensor)
    initial = read(initial_xyz)
    final = read(final_xyz)
    images = [initial] + [initial.copy() for _ in range(n_images)] + [final]
    neb = NEB(images, climb=True, method="improvedtangent", allow_shared_calculator=True)
    neb.interpolate()
    for img in images:
        img.calc = calc
        if initial.cell.any():
            img.set_cell(initial.cell)
            img.set_pbc(initial.pbc)
        else:
            img.set_cell([100, 100, 100])
            img.set_pbc(False)
    opt = FIRE(neb, trajectory=output_traj, logfile=None)
    opt.run(fmax=fmax)
    logger.info(f"ASE NEB done: {output_traj}")
    return output_traj


def run_lammps_md(
    checkpoint_path: str,
    input_structure: str,
    output_traj: str,
    lammps_exec: str,
    work_dir: str,
    elements: list,
    lammps_in_template: Optional[str] = None,
    max_radius: float = 5.0,
    atomic_energy_file: Optional[str] = None,
) -> str:
    """
    Run MD with LAMMPS USER-MFFTORCH (LibTorch + Kokkos).
    Exports core.pt via mff-export-core, then runs LAMMPS.
    User must provide lammps_in_template with placeholders {core_pt}, {elements}, etc.
    Or use a pre-written in.mfftorch. Output trajectory must be converted from LAMMPS dump to XYZ.
    """
    os.makedirs(work_dir, exist_ok=True)
    core_pt = os.path.join(work_dir, "core.pt")
    if not os.path.exists(core_pt):
        cmd_export = [
            sys.executable,
            "-m",
            "molecular_force_field.cli.export_libtorch_core",
            "--checkpoint",
            checkpoint_path,
            "--elements",
        ] + elements + [
            "--max-radius",
            str(max_radius),
            "--out",
            core_pt,
        ]
        if atomic_energy_file and os.path.exists(atomic_energy_file):
            cmd_export.extend(["--embed-e0", "--e0-csv", atomic_energy_file])
        subprocess.run(cmd_export, check=True)
    lmp_in = lammps_in_template or os.path.join(work_dir, "in.mfftorch")
    if not os.path.exists(lmp_in):
        raise FileNotFoundError(
            f"LAMMPS input not found: {lmp_in}. "
            "Provide --lammps-in-template or create in.mfftorch with pair_style mff/torch."
        )
    cmd_lmp = [
        lammps_exec,
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", "kokkos", "newton", "off", "neigh", "full",
        "-in", lmp_in,
    ]
    subprocess.run(cmd_lmp, cwd=work_dir, check=True)
    logger.info(f"LAMMPS MD done. Convert dump to XYZ for {output_traj}")
    return output_traj
