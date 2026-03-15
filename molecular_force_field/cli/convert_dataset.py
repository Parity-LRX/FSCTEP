"""Convert common ML force-field datasets into extxyz for this repository."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import h5py
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write


KCAL_MOL_TO_EV = 0.0433641153087705
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


@dataclass(frozen=True)
class FormatConfig:
    species_keys: tuple[str, ...]
    coord_keys: tuple[str, ...]
    energy_keys: tuple[str, ...]
    force_keys: tuple[str, ...]
    stress_keys: tuple[str, ...] = ()
    distance_unit: str = "ang"
    energy_unit: str = "ev"
    force_unit: str = "ev/ang"
    force_is_gradient: bool = False


FORMAT_PRESETS: dict[str, FormatConfig] = {
    "ani1x": FormatConfig(
        species_keys=("atomic_numbers", "species", "Z"),
        coord_keys=("coordinates", "coords", "positions"),
        energy_keys=("wb97x_dz.energy", "energies", "energy"),
        force_keys=("wb97x_dz.forces", "forces", "force"),
        distance_unit="ang",
        energy_unit="hartree",
        force_unit="hartree/ang",
    ),
    "spice": FormatConfig(
        species_keys=("atomic_numbers", "species", "Z"),
        coord_keys=("conformations", "coordinates", "positions"),
        energy_keys=("dft_total_energy", "formation_energy", "energies", "energy"),
        force_keys=("dft_total_gradient", "gradient", "gradients", "forces", "force"),
        distance_unit="bohr",
        energy_unit="hartree",
        force_unit="hartree/bohr",
        force_is_gradient=True,
    ),
    "qm7x": FormatConfig(
        species_keys=("atNUM", "atomic_numbers", "Z"),
        coord_keys=("atXYZ", "coordinates", "positions"),
        energy_keys=("ePBE0+MBD", "energy", "energies"),
        force_keys=("pbe0FOR", "forces", "force"),
        distance_unit="ang",
        energy_unit="ev",
        force_unit="ev/ang",
    ),
    "generic-h5": FormatConfig(
        species_keys=("atomic_numbers", "species", "Z", "atNUM"),
        coord_keys=("coordinates", "positions", "conformations", "atXYZ", "coords"),
        energy_keys=("energy", "energies", "ePBE0+MBD", "dft_total_energy"),
        force_keys=("forces", "force", "dft_total_gradient", "pbe0FOR"),
        stress_keys=("stress", "virial"),
        distance_unit="ang",
        energy_unit="ev",
        force_unit="ev/ang",
    ),
}


def _flatten_scalar(value: np.ndarray) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar array, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _find_dataset(group: h5py.Group, candidates: Iterable[str]) -> h5py.Dataset | None:
    for key in candidates:
        if key in group and isinstance(group[key], h5py.Dataset):
            return group[key]
    lower_to_name = {name.lower(): name for name in group.keys()}
    for key in candidates:
        name = lower_to_name.get(key.lower())
        if name is not None and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def _convert_distances(array: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.lower()
    if unit in {"ang", "angstrom", "angstroms"}:
        return np.asarray(array, dtype=np.float64)
    if unit == "bohr":
        return np.asarray(array, dtype=np.float64) * BOHR_TO_ANG
    raise ValueError(f"Unsupported distance unit: {unit}")


def _convert_energies(array: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.lower()
    if unit == "ev":
        return np.asarray(array, dtype=np.float64)
    if unit in {"hartree", "ha"}:
        return np.asarray(array, dtype=np.float64) * HARTREE_TO_EV
    if unit in {"kcal/mol", "kcalmol", "kcal_mol"}:
        return np.asarray(array, dtype=np.float64) * KCAL_MOL_TO_EV
    raise ValueError(f"Unsupported energy unit: {unit}")


def _convert_forces(array: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.lower()
    arr = np.asarray(array, dtype=np.float64)
    if unit == "ev/ang":
        return arr
    if unit in {"hartree/ang", "ha/ang"}:
        return arr * HARTREE_TO_EV
    if unit in {"hartree/bohr", "ha/bohr"}:
        return arr * HARTREE_TO_EV / BOHR_TO_ANG
    if unit in {"kcal/mol/ang", "kcalmol/ang"}:
        return arr * KCAL_MOL_TO_EV
    raise ValueError(f"Unsupported force unit: {unit}")


def _symbols_from_atomic_numbers(atomic_numbers: np.ndarray) -> list[str]:
    out = []
    for z in np.asarray(atomic_numbers).reshape(-1):
        z_int = int(z)
        if z_int <= 0 or z_int >= len(chemical_symbols):
            raise ValueError(f"Invalid atomic number: {z_int}")
        out.append(chemical_symbols[z_int])
    return out


def _make_atoms(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energy: float,
    forces: np.ndarray,
    stress: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
) -> Atoms:
    atoms = Atoms(symbols=_symbols_from_atomic_numbers(atomic_numbers), positions=positions, pbc=pbc)
    atoms.arrays["forces"] = np.asarray(forces, dtype=np.float64)
    atoms.arrays["Z"] = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    atoms.info["energy"] = float(energy)
    if stress is not None:
        stress_arr = np.asarray(stress, dtype=np.float64)
        if stress_arr.shape == (3, 3):
            atoms.info["stress"] = stress_arr.reshape(-1)
    return atoms


def inspect_input(path: str, max_items: int = 200) -> None:
    if path.endswith(".npz"):
        with np.load(path) as data:
            print(f"NPZ keys in {path}:")
            for key in sorted(data.files):
                arr = data[key]
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        return

    with h5py.File(path, "r") as h5:
        print(f"HDF5 tree for {path}:")
        count = 0

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            nonlocal count
            if count >= max_items:
                return
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {name}/")
            count += 1

        h5.visititems(visitor)
        if count >= max_items:
            print(f"... truncated after {max_items} items")


def convert_rmd17(args: argparse.Namespace) -> int:
    data = np.load(args.input)
    coords = _convert_distances(data["coords"], "ang")
    energies = _convert_energies(data["energies"], "kcal/mol").reshape(-1)
    forces = _convert_forces(data["forces"], "kcal/mol/ang")
    atomic_numbers = np.asarray(data["nuclear_charges"], dtype=np.int32)

    frames: list[Atoms] = []
    n_frames = coords.shape[0]
    for i in range(0, min(n_frames, args.limit or n_frames), args.stride):
        frames.append(
            _make_atoms(
                atomic_numbers=atomic_numbers,
                positions=coords[i],
                energy=float(energies[i]),
                forces=forces[i],
                stress=None,
                pbc=(False, False, False),
            )
        )

    write(args.output, frames, format="extxyz")
    print(f"Wrote {len(frames)} frames to {args.output}")
    return len(frames)


def _extract_group_frames(
    group: h5py.Group,
    cfg: FormatConfig,
    args: argparse.Namespace,
) -> list[Atoms]:
    species_ds = _find_dataset(group, (args.species_key,) if args.species_key else cfg.species_keys)
    coord_ds = _find_dataset(group, (args.coord_key,) if args.coord_key else cfg.coord_keys)
    energy_ds = _find_dataset(group, (args.energy_key,) if args.energy_key else cfg.energy_keys)
    force_ds = _find_dataset(group, (args.force_key,) if args.force_key else cfg.force_keys)
    stress_ds = _find_dataset(group, cfg.stress_keys)

    if not (species_ds and coord_ds and energy_ds and force_ds):
        return []

    atomic_numbers = np.asarray(species_ds[...], dtype=np.int32).reshape(-1)
    coords_raw = np.asarray(coord_ds[...])
    energies_raw = np.asarray(energy_ds[...])
    forces_raw = np.asarray(force_ds[...])

    if coords_raw.ndim == 2 and coords_raw.shape[-1] == 3:
        coords = _convert_distances(coords_raw[None, ...], args.distance_unit or cfg.distance_unit)
        energies = _convert_energies(np.asarray([_flatten_scalar(energies_raw)]), args.energy_unit or cfg.energy_unit)
        forces = _convert_forces(forces_raw[None, ...], args.force_unit or cfg.force_unit)
        stress_values = None
        if stress_ds is not None:
            stress_arr = np.asarray(stress_ds[...], dtype=np.float64)
            if stress_arr.shape == (3, 3):
                stress_values = stress_arr[None, ...]
            elif stress_arr.size == 9:
                stress_values = stress_arr.reshape(1, 3, 3)
    elif coords_raw.ndim == 3 and coords_raw.shape[-1] == 3:
        coords = _convert_distances(coords_raw, args.distance_unit or cfg.distance_unit)
        energies = _convert_energies(energies_raw.reshape(-1), args.energy_unit or cfg.energy_unit)
        forces = _convert_forces(forces_raw, args.force_unit or cfg.force_unit)
        stress_values = None
        if stress_ds is not None:
            stress_arr = np.asarray(stress_ds[...], dtype=np.float64)
            if stress_arr.ndim == 3 and stress_arr.shape[1:] == (3, 3):
                stress_values = stress_arr
    else:
        return []

    if coords.shape[0] != energies.shape[0] or coords.shape[0] != forces.shape[0]:
        return []

    if args.force_is_gradient or cfg.force_is_gradient:
        forces = -forces

    pbc = tuple(bool(x) for x in args.pbc)
    frames = []
    for i in range(coords.shape[0]):
        stress_i = None if stress_values is None else stress_values[i]
        frames.append(
            _make_atoms(
                atomic_numbers=atomic_numbers,
                positions=coords[i],
                energy=float(energies[i]),
                forces=forces[i],
                stress=stress_i,
                pbc=pbc,
            )
        )
    return frames


def convert_hdf5(args: argparse.Namespace) -> int:
    cfg = FORMAT_PRESETS[args.format]
    written = 0
    frames: list[Atoms] = []

    with h5py.File(args.input, "r") as h5:
        root_frames = _extract_group_frames(h5, cfg, args)
        for frame in root_frames[:: args.stride]:
            if args.limit is not None and written >= args.limit:
                break
            frames.append(frame)
            written += 1

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            nonlocal written
            if not isinstance(obj, h5py.Group):
                return
            if args.group_prefix and not name.startswith(args.group_prefix):
                return
            extracted = _extract_group_frames(obj, cfg, args)
            if not extracted:
                return
            for frame in extracted[:: args.stride]:
                if args.limit is not None and written >= args.limit:
                    return
                frames.append(frame)
                written += 1

        h5.visititems(visitor)

    if not frames:
        raise RuntimeError(
            "No structures were extracted. Run with --inspect first, then override "
            "--species-key/--coord-key/--energy-key/--force-key or units if needed."
        )

    write(args.output, frames, format="extxyz")
    print(f"Wrote {len(frames)} frames to {args.output}")
    return len(frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert common ML force-field datasets into extxyz compatible with mff-preprocess."
    )
    parser.add_argument("--format", required=True, choices=["rmd17", "ani1x", "qm7x", "spice", "generic-h5"])
    parser.add_argument("--input", required=True, help="Input dataset path (.npz/.h5/.hdf5)")
    parser.add_argument("--output", help="Output extxyz path")
    parser.add_argument("--inspect", action="store_true", help="Inspect keys/tree and exit")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of frames to write")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--species-key", type=str, default=None, help="Override atomic-number dataset key")
    parser.add_argument("--coord-key", type=str, default=None, help="Override coordinate dataset key")
    parser.add_argument("--energy-key", type=str, default=None, help="Override energy dataset key")
    parser.add_argument("--force-key", type=str, default=None, help="Override force/gradient dataset key")
    parser.add_argument("--distance-unit", type=str, default=None, help="Override coordinate unit: ang or bohr")
    parser.add_argument("--energy-unit", type=str, default=None, help="Override energy unit: ev, hartree, kcal/mol")
    parser.add_argument(
        "--force-unit",
        type=str,
        default=None,
        help="Override force unit: ev/ang, hartree/ang, hartree/bohr, kcal/mol/ang",
    )
    parser.add_argument("--force-is-gradient", action="store_true", help="Interpret force_key data as gradient and negate it")
    parser.add_argument("--group-prefix", type=str, default=None, help="Only inspect/convert HDF5 groups under this prefix")
    parser.add_argument("--pbc", nargs=3, type=int, default=[0, 0, 0], help="PBC flags for output extxyz, e.g. --pbc 1 1 1")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.inspect:
        inspect_input(args.input)
        return

    if not args.output:
        parser.error("--output is required unless --inspect is used")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    if args.format == "rmd17":
        convert_rmd17(args)
    else:
        convert_hdf5(args)


if __name__ == "__main__":
    main()
