"""Command-line interface for data preprocessing."""

import argparse
import os
import numpy as np
import pandas as pd
from molecular_force_field.data.preprocessing import (
    extract_data_blocks,
    fit_baseline_energies,
    compute_correction,
    save_set,
    save_to_h5_parallel,
)


def _random_split_indices(data_size: int, train_ratio: float, seed: int):
    indices = np.arange(data_size)
    np.random.seed(seed)
    train_size = int(train_ratio * data_size)
    val_size = data_size - train_size
    val_indices = np.random.choice(indices, size=val_size, replace=False) if val_size > 0 else np.array([], dtype=int)
    train_mask = ~np.isin(indices, val_indices)
    train_indices = indices[train_mask]
    return train_indices, val_indices


def _source_tail_split_indices(input_file: str, data_size: int, train_ratio: float, seed: int):
    from ase.io import read as ase_read

    atoms_list = ase_read(input_file, index=":")
    if len(atoms_list) != data_size or data_size < 2:
        return None

    source_keys = [atoms.info.get("source") for atoms in atoms_list]
    if not source_keys or any(key is None for key in source_keys):
        return None

    train_size = int(train_ratio * data_size)
    target_val_size = data_size - train_size
    if target_val_size <= 0:
        return np.arange(data_size), np.array([], dtype=int)

    anchor_by_source = {}
    scored_indices = []
    rng = np.random.default_rng(seed)
    for idx, (atoms, source_key) in enumerate(zip(atoms_list, source_keys)):
        source_key = str(source_key)
        if source_key not in anchor_by_source:
            anchor_by_source[source_key] = idx
            continue
        ref_atoms = atoms_list[anchor_by_source[source_key]]
        if (
            len(atoms) != len(ref_atoms)
            or not np.array_equal(atoms.get_atomic_numbers(), ref_atoms.get_atomic_numbers())
        ):
            score = -np.inf
        else:
            diff = atoms.get_positions() - ref_atoms.get_positions()
            score = float(np.sqrt(np.mean(diff * diff)))
        scored_indices.append((score, float(rng.random()), idx))

    if not scored_indices:
        return None

    scored_indices.sort(key=lambda item: (item[0], item[1]), reverse=True)
    target_val_size = min(target_val_size, len(scored_indices))
    val_indices = np.array(sorted(item[2] for item in scored_indices[:target_val_size]), dtype=int)
    indices = np.arange(data_size)
    train_mask = ~np.isin(indices, val_indices)
    train_indices = indices[train_mask]
    return train_indices, val_indices


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess molecular data')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input XYZ file')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for preprocessed files (default: data)')
    parser.add_argument('--max-atom', type=int, default=None,
                        help='Legacy padded raw-storage size. Leave unset to use variable-length read_{train,val}.h5.')
    parser.add_argument('--train-ratio', type=float, default=0.95,
                        help='Ratio of training data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--atomic-energy-keys', type=int, nargs='+', default=[1, 6, 7, 8],
                        help='Atomic number keys for energy fitting')
    parser.add_argument('--initial-energy-values', type=float, nargs='+', default=None,
                        help='Initial guess for atomic energies')
    parser.add_argument('--elements', type=str, nargs='+', default=None,
                        help='Element symbols to recognize (default: None, recognizes all elements from periodic table). '
                             'If specified, only these elements will be recognized. Example: --elements C H O N Fe')
    parser.add_argument('--energy-key', type=str, default=None,
                        help='Override the structure-level energy metadata key in extxyz comments (default: energy)')
    parser.add_argument('--force-key', type=str, default=None,
                        help='Override the per-atom vector force property key in extxyz Properties (default search: force/forces/f)')
    parser.add_argument('--species-key', type=str, default=None,
                        help='Override the per-atom species property key in extxyz Properties (default search: species/symbol/element)')
    parser.add_argument('--coord-key', type=str, default=None,
                        help='Override the per-atom coordinate property key in extxyz Properties (default: pos)')
    parser.add_argument('--atomic-number-key', type=str, default=None,
                        help='Override the per-atom atomic-number property key in extxyz Properties (default search: Z/atomic_number)')
    parser.add_argument('--skip-h5', action='store_true',
                        help='Skip neighbor list preprocessing (only save raw data)')
    parser.add_argument('--max-radius', type=float, default=5.0,
                        help='Maximum radius for neighbor search (for H5 preprocessing)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers (for H5 preprocessing)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    print(f"Reading {args.input_file}...")
    
    # Extract data blocks
    all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs, all_stresses = extract_data_blocks(
        args.input_file,
        elements=args.elements,
        energy_key=args.energy_key,
        force_key=args.force_key,
        species_key=args.species_key,
        coord_key=args.coord_key,
        atomic_number_key=args.atomic_number_key,
    )
    print(f"Total frames: {len(all_blocks)}")
    
    # Split train/val
    data_size = len(all_blocks)
    split_result = _source_tail_split_indices(args.input_file, data_size, args.train_ratio, args.seed)
    if split_result is None:
        train_indices, val_indices = _random_split_indices(data_size, args.train_ratio, args.seed)
        print("Split mode: random")
    else:
        train_indices, val_indices = split_result
        print("Split mode: source-tail holdout")
    
    print(f"Split: {len(train_indices)} Train, {len(val_indices)} Val")

    # Save split indices for aligning external labels (e.g. dipole, polarizability)
    # train_indices[i] = original extxyz frame index for processed_train.h5 sample_i
    # val_indices[i] = original extxyz frame index for processed_val.h5 sample_i
    train_indices_path = os.path.join(args.output_dir, 'train_indices.npy')
    val_indices_path = os.path.join(args.output_dir, 'val_indices.npy')
    np.save(train_indices_path, train_indices)
    np.save(val_indices_path, val_indices)
    print(f"Saved {train_indices_path}, {val_indices_path}")
    
    train_blocks = [all_blocks[i] for i in train_indices]
    train_raw_E = [all_raw_energy[i] for i in train_indices]
    val_blocks = [all_blocks[i] for i in val_indices]
    val_raw_E = [all_raw_energy[i] for i in val_indices]
    
    # Fit baseline energies
    keys = np.array(args.atomic_energy_keys, dtype=np.int64)
    if args.initial_energy_values is None:
        initial_values = np.array([-0.01] * len(keys), dtype=np.float64)
    else:
        initial_values = np.array(args.initial_energy_values, dtype=np.float64)
    
    fitted_values = fit_baseline_energies(train_blocks, train_raw_E, keys, initial_values)
    
    # Save fitted energies
    fitted_e0_path = os.path.join(args.output_dir, 'fitted_E0.csv')
    pd.DataFrame({'Atom': keys, 'E0': fitted_values}).to_csv(fitted_e0_path, index=False)
    print(f"Saved {fitted_e0_path}")
    
    # Compute corrections
    print("Computing correction energies...")
    train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)
    val_correction = compute_correction(val_blocks, val_raw_E, keys, fitted_values)
    
    # Save sets
    print("Saving files...")
    save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs,
             stress_list=all_stresses, max_atom=args.max_atom, output_dir=args.output_dir)
    save_set('val', val_indices, val_blocks, val_raw_E, val_correction, all_cells, pbc_list=all_pbcs,
             stress_list=all_stresses, max_atom=args.max_atom, output_dir=args.output_dir)
    
    print(f"Raw data saved to {args.output_dir}/")
    
    # Preprocess H5 files (neighbor list computation) - enabled by default
    if not args.skip_h5:
        print("\nComputing neighbor lists (this may take a while)...")
        save_to_h5_parallel('train', args.max_radius, args.num_workers, data_dir=args.output_dir)
        save_to_h5_parallel('val', args.max_radius, args.num_workers, data_dir=args.output_dir)
        print(f"\nDone! All preprocessed files saved in {args.output_dir}/")
        print("You can now run distributed training with:")
        print(f"  torchrun --nproc_per_node=2 -m molecular_force_field.cli.train --distributed --data-dir {args.output_dir}")
    else:
        print("\nSkipped neighbor list computation (--skip-h5 was set).")
        print("To complete preprocessing, run:")
        print(f"  mff-preprocess --input-file {args.input_file} --output-dir {args.output_dir}")


if __name__ == '__main__':
    main()
