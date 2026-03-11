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


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess molecular data')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input XYZ file')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for preprocessed files (default: data)')
    parser.add_argument('--max-atom', type=int, default=1,
                        help='Maximum number of atoms (for padding)')
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
        args.input_file, elements=args.elements
    )
    print(f"Total frames: {len(all_blocks)}")
    
    # Split train/val
    data_size = len(all_blocks)
    indices = np.arange(data_size)
    
    np.random.seed(args.seed)
    
    train_size = int(args.train_ratio * data_size)
    val_size = data_size - train_size
    val_indices = np.random.choice(indices, size=val_size, replace=False)
    train_mask = ~np.isin(indices, val_indices)
    train_indices = indices[train_mask]
    
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