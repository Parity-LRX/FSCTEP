from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from tqdm import tqdm

from molecular_force_field.thermal.callaway import (
    CallawayParameters,
    VOIGT_COMPONENTS,
    apply_engineering_scattering,
    extract_component,
    interpolate_component,
    load_phono3py_kappa_hdf5,
    save_callaway_summary,
)
from molecular_force_field.thermal.model_loader import ModelLoadOptions, load_model_and_calculator


def _parse_matrix(values: Sequence[str], name: str) -> np.ndarray:
    if len(values) not in {3, 9}:
        raise ValueError(f"{name} expects either 3 diagonal values or 9 full matrix values.")
    matrix_values = [float(v) for v in values]
    if len(matrix_values) == 3:
        return np.diag(matrix_values)
    return np.array(matrix_values, dtype=np.float64).reshape(3, 3)


def _parse_primitive_matrix(value: str):
    if value.lower() == "auto":
        return "auto"
    parts = value.replace(",", " ").split()
    if len(parts) != 9:
        raise ValueError("--primitive-matrix must be 'auto' or 9 numeric values.")
    return np.array([float(v) for v in parts], dtype=np.float64).reshape(3, 3)


def _load_temperatures(values: Sequence[float], t_min: float | None, t_max: float | None, t_step: float | None) -> np.ndarray:
    if values:
        return np.array(values, dtype=np.float64)
    if t_min is None or t_max is None or t_step is None:
        raise ValueError("Provide either explicit --temperatures or the --t-min/--t-max/--t-step triplet.")
    n_steps = int(round((t_max - t_min) / t_step))
    temperatures = t_min + np.arange(n_steps + 1, dtype=np.float64) * t_step
    if temperatures[-1] < t_max - 1e-9:
        temperatures = np.append(temperatures, t_max)
    return temperatures


def _ase_to_phonopy_atoms(atoms: Atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def _phonopy_to_ase_atoms(ph_atoms) -> Atoms:
    return Atoms(
        symbols=ph_atoms.symbols,
        cell=np.array(ph_atoms.cell, dtype=np.float64),
        scaled_positions=np.array(ph_atoms.scaled_positions, dtype=np.float64),
        pbc=True,
    )


def _get_displaced_supercells(obj, attr_name: str, getter_name: str):
    if hasattr(obj, attr_name):
        value = getattr(obj, attr_name)
        if value is not None:
            return value
    if hasattr(obj, getter_name):
        return getattr(obj, getter_name)()
    raise AttributeError(f"Cannot find displaced supercells using {attr_name} or {getter_name}.")


def _ensure_force_hdf5(path: Path, dataset_name: str, forces: list[np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset_name, data=np.array(forces, dtype=np.float64))


def _write_force_dump(path: Path, forces: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(forces)}\n")
        for force_set in forces:
            handle.write(f"{force_set.shape[0]}\n")
            for vec in force_set:
                handle.write(f"{vec[0]:22.15f} {vec[1]:22.15f} {vec[2]:22.15f}\n")


def _evaluate_forces(
    supercells: Iterable,
    calculator,
    *,
    output_dir: Path,
    prefix: str,
    save_supercells: bool,
) -> list[np.ndarray]:
    forces = []
    for idx, supercell in enumerate(tqdm(list(supercells), desc=f"Evaluating {prefix}", unit="scell")):
        atoms = _phonopy_to_ase_atoms(supercell)
        atoms.calc = calculator
        force = atoms.get_forces()
        forces.append(np.array(force, dtype=np.float64))
        if save_supercells:
            write(output_dir / f"{prefix}_{idx:04d}.extxyz", atoms, format="extxyz")
    return forces


def _find_latest_kappa_file(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("kappa-*.hdf5"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _save_fc_hdf5(path: Path, dataset_name: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset_name, data=np.array(data, dtype=np.float64))


@contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _build_model_options(args) -> ModelLoadOptions:
    return ModelLoadOptions(
        checkpoint=args.checkpoint,
        device=args.device,
        atomic_energy_file=args.atomic_energy_file,
        atomic_energy_keys=args.atomic_energy_keys,
        atomic_energy_values=args.atomic_energy_values,
        max_radius=args.max_radius,
        tensor_product_mode=args.tensor_product_mode,
        dtype=args.dtype,
        max_atomvalue=args.max_atomvalue,
        embedding_dim=args.embedding_dim,
        embed_size=args.embed_size,
        output_size=args.output_size,
        lmax=args.lmax,
        irreps_output_conv_channels=args.irreps_output_conv_channels,
        function_type=args.function_type,
        num_interaction=args.num_interaction,
    )


def run_bte(args) -> None:
    try:
        from phono3py import Phono3py
    except Exception as exc:
        raise ImportError(
            "The BTE workflow requires phono3py. Install it with `pip install phono3py phonopy spglib`."
        ) from exc

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dtype:
        torch.set_default_dtype(torch.float64 if args.dtype.lower() in {"float64", "double", "fp64"} else torch.float32)

    _, calculator, atomic_energies, model_meta = load_model_and_calculator(_build_model_options(args))
    atoms = read(args.structure)
    atoms.calc = calculator

    if args.relax_fmax is not None and args.relax_fmax > 0:
        logging.info("Relaxing the input structure with MLFF before IFC generation.")
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=args.relax_fmax)
        write(output_dir / "relaxed_structure.extxyz", atoms, format="extxyz")
    else:
        max_force = np.abs(atoms.get_forces()).max() if len(atoms) > 0 else 0.0
        logging.info("Input structure max force before IFC workflow: %.6f eV/Ang", max_force)

    temperatures = _load_temperatures(args.temperatures, args.t_min, args.t_max, args.t_step)
    primitive_matrix = _parse_primitive_matrix(args.primitive_matrix)
    supercell_matrix = _parse_matrix(args.supercell, "--supercell").astype(int)
    phonon_supercell_matrix = (
        _parse_matrix(args.phonon_supercell, "--phonon-supercell").astype(int)
        if args.phonon_supercell
        else None
    )

    ph3 = Phono3py(
        _ase_to_phonopy_atoms(atoms),
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
    )
    generate_kwargs = {"distance": args.displacement_distance}
    if args.fc3_cutoff_pair_distance is not None:
        generate_kwargs["cutoff_pair_distance"] = args.fc3_cutoff_pair_distance
    ph3.generate_displacements(**generate_kwargs)

    fc3_supercells = _get_displaced_supercells(
        ph3,
        "supercells_with_displacements",
        "get_supercells_with_displacements",
    )
    fc2_supercells = None
    if phonon_supercell_matrix is not None:
        try:
            fc2_supercells = _get_displaced_supercells(
                ph3,
                "phonon_supercells_with_displacements",
                "get_phonon_supercells_with_displacements",
            )
        except AttributeError:
            fc2_supercells = None

    if not fc3_supercells:
        raise RuntimeError("phono3py did not generate any FC3 displaced supercells.")

    logging.info("Evaluating %d FC3 displaced supercells with the MLFF.", len(fc3_supercells))
    fc3_forces = _evaluate_forces(
        fc3_supercells,
        calculator,
        output_dir=output_dir,
        prefix="fc3_supercell",
        save_supercells=args.save_displaced_supercells,
    )

    if fc2_supercells is not None and len(fc2_supercells) > 0:
        logging.info("Evaluating %d FC2 displaced supercells with the MLFF.", len(fc2_supercells))
        fc2_forces = _evaluate_forces(
            fc2_supercells,
            calculator,
            output_dir=output_dir,
            prefix="fc2_supercell",
            save_supercells=args.save_displaced_supercells,
        )
    else:
        logging.info("Using FC3 displacements for IFC2 because no separate phonon supercell matrix was provided.")
        fc2_forces = fc3_forces

    ph3.forces = fc3_forces
    if fc2_supercells is not None and len(fc2_supercells) > 0:
        ph3.phonon_forces = fc2_forces

    produce_fc3_kwargs = {}
    produce_fc2_kwargs = {}
    if args.symmetrize_fc3:
        produce_fc3_kwargs["symmetrize_fc3r"] = True
    if args.symmetrize_fc2:
        produce_fc2_kwargs["symmetrize_fc2"] = True

    ph3.produce_fc3(**produce_fc3_kwargs)
    if hasattr(ph3, "produce_fc2"):
        ph3.produce_fc2(**produce_fc2_kwargs)

    fc2 = np.array(ph3.fc2, dtype=np.float64)
    fc3 = np.array(ph3.fc3, dtype=np.float64)
    _save_fc_hdf5(output_dir / "fc2.hdf5", "force_constants", fc2)
    _save_fc_hdf5(output_dir / "fc3.hdf5", "fc3", fc3)
    _ensure_force_hdf5(output_dir / "fc2_forces.hdf5", "forces", fc2_forces)
    _ensure_force_hdf5(output_dir / "fc3_forces.hdf5", "forces", fc3_forces)
    _write_force_dump(output_dir / "fc2_forces.txt", fc2_forces)
    _write_force_dump(output_dir / "fc3_forces.txt", fc3_forces)

    mesh = [int(v) for v in args.mesh]
    logging.info(
        "Running %s intrinsic BTE on mesh %s and temperatures %s",
        "LBTE" if args.lbte else "RTA",
        mesh,
        temperatures.tolist(),
    )
    with _pushd(output_dir):
        ph3.mesh_numbers = mesh
        if hasattr(ph3, "init_phph_interaction"):
            ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(
            temperatures=temperatures,
            is_LBTE=args.lbte,
            is_isotope=args.isotope,
            write_kappa=True,
        )

    kappa_hdf5 = _find_latest_kappa_file(output_dir)
    metadata = {
        "structure": str(Path(args.structure).resolve()),
        "output_dir": str(output_dir),
        "temperatures_K": temperatures.tolist(),
        "mesh": mesh,
        "supercell_matrix": supercell_matrix.tolist(),
        "phonon_supercell_matrix": phonon_supercell_matrix.tolist() if phonon_supercell_matrix is not None else None,
        "primitive_matrix": primitive_matrix if isinstance(primitive_matrix, str) else primitive_matrix.tolist(),
        "fc2_shape": list(fc2.shape),
        "fc3_shape": list(fc3.shape),
        "n_fc2_supercells": len(fc2_forces),
        "n_fc3_supercells": len(fc3_forces),
        "kappa_hdf5": str(kappa_hdf5) if kappa_hdf5 is not None else None,
        "atomic_energies": atomic_energies,
        "model": model_meta,
        "lbte": bool(args.lbte),
        "isotope": bool(args.isotope),
    }
    with (output_dir / "thermal_workflow_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    logging.info("Saved IFC2/IFC3 and intrinsic BTE artifacts to %s", output_dir)
    if kappa_hdf5 is not None:
        logging.info("Intrinsic conductivity file: %s", kappa_hdf5)


def _default_fit_bounds(param_name: str):
    if param_name == "grain_size_nm":
        return 1e-3, 1e9
    if param_name == "specularity":
        return 0.0, 0.999999
    return 0.0, np.inf


def run_callaway(args) -> None:
    data = load_phono3py_kappa_hdf5(args.kappa_hdf5)
    params = CallawayParameters(
        grain_size_nm=args.grain_size_nm,
        specularity=args.specularity,
        point_defect_coeff=args.point_defect_coeff,
        dislocation_coeff=args.dislocation_coeff,
        interface_coeff=args.interface_coeff,
    )

    if args.fit_experiment_csv:
        try:
            from scipy.optimize import least_squares
        except Exception as exc:
            raise ImportError("Fitting requires scipy. Install it with `pip install scipy`.") from exc

        fit_df = pd.read_csv(args.fit_experiment_csv)
        if "temperature" not in fit_df.columns:
            raise ValueError("Experiment CSV must contain a 'temperature' column.")
        fit_component = args.fit_component
        target_column = args.fit_column or fit_component
        if target_column not in fit_df.columns:
            raise ValueError(
                f"Experiment CSV must contain a '{target_column}' column for fitting."
            )

        fit_params = [name.strip() for name in args.fit_parameters.split(",") if name.strip()]
        if not fit_params:
            raise ValueError("No fit parameters were selected.")
        for name in fit_params:
            if not hasattr(params, name):
                raise ValueError(f"Unknown fit parameter {name!r}.")

        x0 = np.array([getattr(params, name) if getattr(params, name) is not None else 100.0 for name in fit_params], dtype=np.float64)
        lower, upper = zip(*[_default_fit_bounds(name) for name in fit_params])

        target_temperatures = fit_df["temperature"].to_numpy(dtype=np.float64)
        target_values = fit_df[target_column].to_numpy(dtype=np.float64)

        def residual(vector: np.ndarray) -> np.ndarray:
            trial = CallawayParameters(**params.__dict__)
            for value, name in zip(vector, fit_params):
                setattr(trial, name, float(value))
            result = apply_engineering_scattering(data, trial)
            prediction = interpolate_component(result, fit_component, target_temperatures, engineered=True)
            return prediction - target_values

        fit_result = least_squares(residual, x0=x0, bounds=(np.array(lower), np.array(upper)))
        for value, name in zip(fit_result.x, fit_params):
            setattr(params, name, float(value))
        logging.info("Fitted Callaway parameters: %s", {name: getattr(params, name) for name in fit_params})

    result = apply_engineering_scattering(data, params)
    output_paths = save_callaway_summary(result, args.output_prefix, component=args.component)

    summary = {
        "parameters": params.__dict__,
        "component": args.component,
        "component_intrinsic": extract_component(result, args.component, engineered=False).tolist(),
        "component_engineered": extract_component(result, args.component, engineered=True).tolist(),
        "temperature": result["temperature"].tolist(),
        "csv": str(output_paths["csv"]),
        "json": str(output_paths["json"]),
    }
    with Path(str(args.output_prefix) + "_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    logging.info("Saved Callaway post-processing outputs to %s and %s", output_paths["csv"], output_paths["json"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thermal transport workflow: MLFF -> IFC2/IFC3 -> intrinsic BTE -> Callaway engineering scattering"
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    bte = subparsers.add_parser("bte", help="Run IFC2/IFC3 generation and intrinsic BTE with phono3py")
    bte.add_argument("--checkpoint", required=True, help="Path to the trained MLFF checkpoint")
    bte.add_argument("--structure", required=True, help="Relaxed crystal structure readable by ASE")
    bte.add_argument("--output-dir", required=True, help="Directory for IFC/BTE outputs")
    bte.add_argument("--supercell", nargs="+", required=True, help="3 or 9 integers defining the FC3 supercell")
    bte.add_argument("--phonon-supercell", nargs="+", default=None, help="Optional FC2-only supercell (3 or 9 integers)")
    bte.add_argument("--primitive-matrix", default="auto", help="Primitive matrix as 'auto' or 9 numbers")
    bte.add_argument("--mesh", nargs=3, required=True, help="q-point mesh, e.g. 16 16 16")
    bte.add_argument("--temperatures", nargs="*", type=float, default=[], help="Explicit temperature list in K")
    bte.add_argument("--t-min", type=float, default=None, help="Minimum temperature in K")
    bte.add_argument("--t-max", type=float, default=None, help="Maximum temperature in K")
    bte.add_argument("--t-step", type=float, default=None, help="Temperature step in K")
    bte.add_argument("--displacement-distance", type=float, default=0.03, help="Finite-displacement amplitude in Angstrom")
    bte.add_argument("--fc3-cutoff-pair-distance", type=float, default=None, help="Optional phono3py FC3 pair cutoff in Angstrom")
    bte.add_argument("--lbte", action="store_true", help="Use iterative LBTE instead of RTA")
    bte.add_argument("--isotope", action="store_true", help="Include isotope scattering in the intrinsic BTE solve")
    bte.add_argument("--relax-fmax", type=float, default=None, help="Optional MLFF relaxation threshold before IFC generation")
    bte.add_argument("--save-displaced-supercells", action="store_true", help="Write each displaced supercell to extxyz for debugging")
    bte.add_argument("--symmetrize-fc2", action="store_true", help="Symmetrize IFC2 when producing force constants")
    bte.add_argument("--symmetrize-fc3", action="store_true", help="Symmetrize IFC3 when producing force constants")
    _add_model_loading_args(bte)
    bte.set_defaults(func=run_bte)

    callaway = subparsers.add_parser("callaway", help="Apply Callaway-style engineering scattering on top of intrinsic BTE")
    callaway.add_argument("--kappa-hdf5", required=True, help="phono3py kappa-*.hdf5 file")
    callaway.add_argument("--output-prefix", required=True, help="Output prefix for CSV/JSON summaries")
    callaway.add_argument("--component", default="xx", choices=["isotropic", *VOIGT_COMPONENTS], help="Tensor component to summarize")
    callaway.add_argument("--grain-size-nm", type=float, default=None, help="Effective grain size for boundary scattering")
    callaway.add_argument("--specularity", type=float, default=0.0, help="Boundary specularity factor in [0, 1)")
    callaway.add_argument("--point-defect-coeff", type=float, default=0.0, help="Point-defect scattering coefficient A in native phono3py units")
    callaway.add_argument("--dislocation-coeff", type=float, default=0.0, help="Dislocation scattering coefficient in native phono3py units")
    callaway.add_argument("--interface-coeff", type=float, default=0.0, help="Interface scattering coefficient in native phono3py units")
    callaway.add_argument("--fit-experiment-csv", default=None, help="Optional experimental CSV for fitting extrinsic parameters")
    callaway.add_argument("--fit-component", default="xx", choices=["isotropic", *VOIGT_COMPONENTS], help="Tensor component used during fitting")
    callaway.add_argument("--fit-column", default=None, help="Column name inside the experiment CSV. Defaults to --fit-component")
    callaway.add_argument(
        "--fit-parameters",
        default="grain_size_nm,point_defect_coeff",
        help="Comma-separated parameter names to fit. Choices: grain_size_nm,specularity,point_defect_coeff,dislocation_coeff,interface_coeff",
    )
    callaway.set_defaults(func=run_callaway)

    return parser


def _add_model_loading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda", help="Torch device for MLFF inference")
    parser.add_argument("--dtype", default=None, choices=["float32", "float64", "float", "double"], help="Override checkpoint dtype")
    parser.add_argument("--atomic-energy-file", default=None, help="CSV with Atom,E0 columns")
    parser.add_argument("--atomic-energy-keys", nargs="+", type=int, default=None, help="Atomic numbers for custom E0 values")
    parser.add_argument("--atomic-energy-values", nargs="+", type=float, default=None, help="Atomic reference energies in eV")
    parser.add_argument("--max-radius", type=float, default=None, help="Override checkpoint max radius")
    parser.add_argument(
        "--tensor-product-mode",
        default=None,
        choices=[
            "spherical",
            "spherical-save",
            "spherical-save-cue",
            "partial-cartesian",
            "partial-cartesian-loose",
            "pure-cartesian",
            "pure-cartesian-sparse",
            "pure-cartesian-ictd",
            "pure-cartesian-ictd-save",
        ],
        help="Override checkpoint tensor_product_mode",
    )
    parser.add_argument("--num-interaction", type=int, default=None, help="Override checkpoint num_interaction")
    parser.add_argument("--max-atomvalue", type=int, default=None, help="Override checkpoint max_atomvalue")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Override checkpoint embedding_dim")
    parser.add_argument("--embed-size", nargs="+", type=int, default=None, help="Override checkpoint embed_size")
    parser.add_argument("--output-size", type=int, default=None, help="Override checkpoint output_size")
    parser.add_argument("--lmax", type=int, default=None, help="Override checkpoint lmax")
    parser.add_argument("--irreps-output-conv-channels", type=int, default=None)
    parser.add_argument("--function-type", default=None, choices=["gaussian", "bessel", "fourier", "cosine", "smooth_finite"], help="Override checkpoint function_type")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
