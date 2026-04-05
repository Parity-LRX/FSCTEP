"""Command-line interface for training."""

import argparse
import math
import os
import random
import sys
import torch
import torch.distributed as dist
import numpy as np
import logging
import time
from datetime import timedelta
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from logging.handlers import RotatingFileHandler

from molecular_force_field.models import (
    E3_TransformerLayer_multi,
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    PureCartesianTransformerLayer,
    PureCartesianSparseTransformerLayer,
    PureCartesianSparseTransformerLayerSave,
    PureCartesianICTDTransformerLayer,
    PureCartesianICTDO3TransformerLayer,
    MainNet,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.data import H5Dataset
from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.data.preprocessing import save_to_h5_parallel
from molecular_force_field.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
    get_checkpoint_atomic_energies,
    maybe_load_checkpoint,
    resolve_model_architecture,
)
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.external_tensor_specs import (
    build_standard_external_tensor_specs,
    normalize_external_tensor_specs,
)
from molecular_force_field.utils.fidelity import parse_fidelity_loss_weights
from molecular_force_field.models.zbl import maybe_wrap_model_with_zbl


def check_and_preprocess_data(data_dir, train_prefix, val_prefix, max_radius, num_workers,
                              input_file=None, train_input_file=None, valid_input_file=None, seed=42):
    """
    Ensure train/val data exist: use existing H5, run H5-only preprocessing, or full pipeline from XYZ.

    Data resolution order:
        1. If processed_<prefix>.h5 exist in data_dir -> use as-is.
        2. If read_<prefix>.h5 exist -> run H5 preprocessing only.
        3. If train_input_file and valid_input_file are both given -> use as train/val, no split; run full preprocessing.
        4. If input_file is given -> split 90/10 train/val, then run full preprocessing.
        5. Otherwise -> return False and log error.

    Args:
        data_dir: Directory for data files (and fitted_E0.csv).
        train_prefix: Filename prefix for training set (e.g. 'train' -> processed_train.h5).
        val_prefix: Filename prefix for validation set (e.g. 'val' -> processed_val.h5).
        max_radius: Max radius for neighbor search in H5 preprocessing.
        num_workers: Number of workers for H5 preprocessing.
        input_file: Optional single XYZ path; triggers 90/10 train/val split then preprocessing.
        train_input_file: Optional training XYZ path; must be used together with valid_input_file (no split).
        valid_input_file: Optional validation XYZ path; must be used together with train_input_file.
        seed: Random seed for train/val split when input_file is used (default: 42).

    Returns:
        True if train/val data are ready, False otherwise.
    """
    train_processed = os.path.join(data_dir, f'processed_{train_prefix}.h5')
    val_processed = os.path.join(data_dir, f'processed_{val_prefix}.h5')
    train_raw = os.path.join(data_dir, f'read_{train_prefix}.h5')
    val_raw = os.path.join(data_dir, f'read_{val_prefix}.h5')

    # Case 1: Preprocessed train/val H5 already exist in data_dir
    if os.path.exists(train_processed) and os.path.exists(val_processed):
        logging.info(f"Found preprocessed data in {data_dir}/")
        return True

    # Case 2: Raw read_* H5 exist; run H5 preprocessing only (no XYZ extraction)
    if os.path.exists(train_raw) and os.path.exists(val_raw):
        logging.info(f"Found raw data in {data_dir}/, running H5 preprocessing...")
        save_to_h5_parallel(train_prefix, max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel(val_prefix, max_radius, num_workers, data_dir=data_dir)
        return True

    # Case 3a: Two XYZ files given (train + valid); use as-is, no split; run full preprocessing
    if train_input_file and valid_input_file and os.path.exists(train_input_file) and os.path.exists(valid_input_file):
        logging.info(f"Using specified train/valid files (no auto split): train={train_input_file}, valid={valid_input_file}")
        from molecular_force_field.data.preprocessing import (
            extract_data_blocks,
            fit_baseline_energies,
            compute_correction,
            save_set
        )
        import pandas as pd
        import numpy as np
        os.makedirs(data_dir, exist_ok=True)
        train_blocks, train_energy, train_raw_energy, train_cells, train_pbcs, train_stresses = extract_data_blocks(train_input_file)
        val_blocks, val_energy, val_raw_energy, val_cells, val_pbcs, val_stresses = extract_data_blocks(valid_input_file)
        logging.info(f"Train frames: {len(train_blocks)}, Valid frames: {len(val_blocks)}")
        train_atoms = []
        for block in train_blocks:
            train_atoms.extend([int(row[3]) for row in block])
        uniq = sorted({a for a in train_atoms if a > 0})
        if not uniq:
            raise ValueError("No valid atomic numbers found in training blocks; cannot fit baseline energies.")
        keys = np.asarray(uniq, dtype=np.int64)
        initial_values = np.array([-0.01] * len(keys), dtype=np.float64)
        fitted_values = fit_baseline_energies(train_blocks, train_raw_energy, keys, initial_values)
        fitted_e0_path = os.path.join(data_dir, 'fitted_E0.csv')
        pd.DataFrame({'Atom': keys, 'E0': fitted_values}).to_csv(fitted_e0_path, index=False)
        logging.info(f"Saved {fitted_e0_path}")
        train_correction = compute_correction(train_blocks, train_raw_energy, keys, fitted_values)
        val_correction = compute_correction(val_blocks, val_raw_energy, keys, fitted_values)
        save_set(train_prefix, np.arange(len(train_blocks)), train_blocks, train_raw_energy,
                 train_correction, train_cells, pbc_list=train_pbcs, stress_list=train_stresses, output_dir=data_dir)
        save_set(val_prefix, np.arange(len(val_blocks)), val_blocks, val_raw_energy,
                 val_correction, val_cells, pbc_list=val_pbcs, stress_list=val_stresses, output_dir=data_dir)
        save_to_h5_parallel(train_prefix, max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel(val_prefix, max_radius, num_workers, data_dir=data_dir)
        logging.info(f"Preprocessing completed! Data saved to {data_dir}/")
        return True

    # Case 3b: Single XYZ file given; 90/10 train/val split then full preprocessing
    if input_file and os.path.exists(input_file):
        logging.info(f"No preprocessed data found. Running preprocessing on {input_file} (auto 90/10 split)...")
        from molecular_force_field.data.preprocessing import (
            extract_data_blocks,
            fit_baseline_energies,
            compute_correction,
            save_set
        )
        import numpy as np
        import pandas as pd

        os.makedirs(data_dir, exist_ok=True)
        all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs, all_stresses = extract_data_blocks(input_file)
        logging.info(f"Total frames: {len(all_blocks)}")

        # 90/10 train/val split (fixed ratio)
        data_size = len(all_blocks)
        indices = np.arange(data_size)
        np.random.seed(seed)
        train_size = int(0.90 * data_size)
        val_size = data_size - train_size
        val_indices = np.random.choice(indices, size=val_size, replace=False)
        train_mask = ~np.isin(indices, val_indices)
        train_indices = indices[train_mask]
        logging.info(f"Split: {len(train_indices)} Train, {len(val_indices)} Val")

        train_blocks = [all_blocks[i] for i in train_indices]
        train_raw_E = [all_raw_energy[i] for i in train_indices]
        val_blocks = [all_blocks[i] for i in val_indices]
        val_raw_E = [all_raw_energy[i] for i in val_indices]

        # Fit E0 on train set only; keys derived from train atoms (no hardcoding)
        train_atoms = []
        for block in train_blocks:
            train_atoms.extend([int(row[3]) for row in block])  # row[3] = atomic number
        uniq = sorted({a for a in train_atoms if a > 0})
        if not uniq:
            raise ValueError("No valid atomic numbers found in training blocks; cannot fit baseline energies.")
        keys = np.asarray(uniq, dtype=np.int64)
        initial_values = np.array([-0.01] * len(keys), dtype=np.float64)
        fitted_values = fit_baseline_energies(train_blocks, train_raw_E, keys, initial_values)

        fitted_e0_path = os.path.join(data_dir, 'fitted_E0.csv')
        pd.DataFrame({'Atom': keys, 'E0': fitted_values}).to_csv(fitted_e0_path, index=False)
        logging.info(f"Saved {fitted_e0_path}")

        train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)
        val_correction = compute_correction(val_blocks, val_raw_E, keys, fitted_values)
        save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs, stress_list=all_stresses, output_dir=data_dir)
        save_set('val', val_indices, val_blocks, val_raw_E, val_correction, all_cells, pbc_list=all_pbcs, stress_list=all_stresses, output_dir=data_dir)
        save_to_h5_parallel('train', max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel('val', max_radius, num_workers, data_dir=data_dir)
        logging.info(f"Preprocessing completed! Data saved to {data_dir}/")
        return True

    # Case 4: No data and no input file; cannot proceed
    logging.error(
        f"No data files found in {data_dir}/ and no input file specified.\n"
        f"Please either:\n"
        f"  1. Run 'mff-preprocess --input-file <xyz_file> --output-dir {data_dir}' first, or\n"
        f"  2. Use '--input-file <xyz_file>' to automatically preprocess data."
    )
    return False


def setup_logging():
    """Configure logging: console (filtered) and rotating file (full)."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    class ConsoleFilter(logging.Filter):
        """Suppress per-epoch training logs on console; keep validation and lifecycle messages."""
        def filter(self, record):
            msg = record.getMessage()
            if 'Epoch' in msg and 'Validation' not in msg and 'Training started' not in msg and 'Training completed' not in msg and 'Early stopping' not in msg:
                return False
            return True

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleFilter())
    
    log_filename = f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = RotatingFileHandler(
        filename=log_filename,
        maxBytes=1000 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress cuequivariance DEBUG logs (TensorProductUniform3x1d etc.) during training
    for name in ("cuequivariance", "cuequivariance_torch"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _relaunch_with_torchrun(args):
    """Re-launch the current process via ``torchrun`` (and optionally ``srun``).

    Called when ``--n-gpu > 1`` or ``--nnodes > 1`` is given without
    ``--distributed``.  The function rebuilds ``sys.argv`` stripping out
    the launcher-specific flags, adds ``--distributed``, and wraps the
    command with ``torchrun`` / ``srun`` as needed.
    """
    import socket
    import subprocess

    # ---- Resolve launcher ----
    launcher = args.launcher
    if launcher == "auto":
        if args.nnodes > 1 and os.environ.get("SLURM_JOB_ID"):
            launcher = "slurm"
        else:
            launcher = "local"

    # ---- Resolve master address ----
    master_addr = args.master_addr
    if master_addr == "auto":
        nodelist = os.environ.get("SLURM_NODELIST", "")
        if nodelist:
            try:
                r = subprocess.run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True, text=True,
                )
                if r.returncode == 0 and r.stdout.strip():
                    master_addr = r.stdout.strip().split("\n")[0]
            except FileNotFoundError:
                pass
        if master_addr == "auto":
            master_addr = socket.gethostname()

    # ---- Rebuild argv: strip launcher-only flags, add --distributed ----
    _launcher_flags = {
        "--n-gpu", "--nnodes", "--master-addr", "--master-port", "--launcher",
    }
    new_argv = []
    skip_next = False
    for i, tok in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        bare = tok.split("=")[0]
        if bare in _launcher_flags:
            if "=" not in tok:
                skip_next = True
            continue
        new_argv.append(tok)
    if "--distributed" not in new_argv:
        new_argv.append("--distributed")

    # ---- Build torchrun command ----
    torchrun_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(args.n_gpu),
        "--nnodes", str(args.nnodes),
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", f"{master_addr}:{args.master_port}",
        "-m", "molecular_force_field.cli.train",
    ] + new_argv

    # ---- Wrap with srun for multi-node SLURM ----
    if launcher == "slurm" and args.nnodes > 1:
        torchrun_cmd = [
            "srun",
            "--nodes", str(args.nnodes),
            "--ntasks-per-node", "1",
        ] + torchrun_cmd

    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Auto-launching DDP: {args.n_gpu} GPU(s) × {args.nnodes} node(s), "
        f"launcher={launcher}"
    )
    logging.info(f"Command: {' '.join(torchrun_cmd)}")

    ret = subprocess.run(torchrun_cmd)
    sys.exit(ret.returncode)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train molecular force field model. When --checkpoint already exists, '
                    'model-structure hyperparameters default to checkpoint metadata and explicit CLI values override.'
    )
    parser.add_argument('--train-prefix', type=str, default='train',
                        help='Prefix for training data files')
    parser.add_argument('--val-prefix', type=str, default='val',
                        help='Prefix for validation data files')
    parser.add_argument('--max-radius', type=float, default=None,
                        help='Maximum radius for neighbor search. '
                             'If not set, prefer checkpoint metadata when --checkpoint exists; otherwise use the built-in default.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--min-learning-rate', type=float, default=2e-5,
                        help='Minimum learning rate (eta_min for cosine / hard floor for step, default: 2e-5)')
    parser.add_argument('--warmup-batches', type=int, default=1000,
                        help='Number of warmup batches for learning rate (default: 1000)')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=['cosine', 'step'],
                        help='LR scheduler after warmup: "cosine" for CosineAnnealingLR, '
                             '"step" for StepLR (default: cosine)')
    parser.add_argument('--lr-decay-patience', type=int, default=1000,
                        help='Step size (in batches) for StepLR scheduler (default: 1000)')
    parser.add_argument('--lr-decay-factor', type=float, default=0.98,
                        help='Decay factor (gamma) for StepLR scheduler (default: 0.98)')
    parser.add_argument('--warmup-start-ratio', type=float, default=0.1,
                        help='Starting learning rate ratio during warmup (0.1 means start at 10%% of target LR, default: 0.1)')
    parser.add_argument('--checkpoint', type=str, default='combined_model.pth',
                        help='Checkpoint path. If the file already exists, training resumes from it and model-structure '
                             'hyperparameters are restored with priority: explicit CLI > checkpoint metadata > defaults.')
    parser.add_argument('--reset-loss-weights', action='store_true', default=False,
                        help='When loading checkpoint, ignore saved loss weights (a, b) and use '
                             'values from command line arguments (--a, --b) instead. '
                             'Default: False (use checkpoint weights)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for data preprocessing')
    parser.add_argument('--mp-context', type=str, default='auto',
                        choices=['auto', 'fork', 'spawn'],
                        help='Multiprocessing start method for DataLoader workers. '
                             '"auto" forces "spawn" when validation compile is enabled (safer with CUDA/compile), '
                             'otherwise uses the default OS method (often "fork" on Linux, faster).')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'float64', 'float', 'double'],
                        help='Default dtype for tensors (float32 or float64, default: float64)')
    parser.add_argument('--matmul-precision', type=str, default='high',
                        choices=['highest', 'high', 'medium'],
                        help='Float32 matmul precision. "high" (default) enables TF32 on Ampere+ GPUs for ~2x matmul speedup. Use "highest" for strict FP32.')
    parser.add_argument('--dump-frequency', type=int, default=250,
                        help='Frequency (in batches) for validation and model saving (default: 250)')
    parser.add_argument('--train-eval-sample-ratio', type=float, default=0.0,
                        help='Ratio of training set to evaluate during validation (0.0-1.0, default: 0.0). '
                             'When 0.0 (default), only the current batch loss is recorded as train loss (no extra forward pass). '
                             'Set to e.g. 0.01 (1%%) to sample training set for more accurate train metrics, '
                             'or 1.0 for full evaluation. Recommended: 0.01 for a good speed/accuracy trade-off.')
    parser.add_argument('--energy-log-frequency', type=int, default=100,
                        help='Frequency (in batches) to log energy predictions (default: 100)')
    parser.add_argument('--energy-weight', '-a', type=float, default=1.0,
                        help='Initial weight for energy loss (default: 1.0)')
    parser.add_argument('--force-weight', '-b', type=float, default=10.0,
                        help='Initial weight for force loss (default: 10.0)')
    parser.add_argument('--update-param', type=int, default=1000,
                        help='Interval (in batches) to update loss weights a and b (default: 1000)')
    parser.add_argument('--weight-a-growth', type=float, default=1.05,
                        help='Growth factor for energy weight a at each update (default: 1.05, meaning 5%% growth). '
                             'Recommended: 1.005 (slow), 1.01 (medium), 1.02 (fast), 1.05 (very fast)')
    parser.add_argument('--weight-b-decay', type=float, default=0.98,
                        help='Decay factor for force weight b at each update (default: 0.98, meaning 2%% decay). '
                             'Recommended: 0.995 (slow), 0.99 (medium), 0.98 (fast)')
    parser.add_argument('--a-min', type=float, default=1.0,
                        help='Minimum clamp for dynamic energy weight a (default: 1.0).')
    parser.add_argument('--a-max', type=float, default=1000.0,
                        help='Maximum clamp for dynamic energy weight a (default: 1000.0).')
    parser.add_argument('--b-min', type=float, default=1.0,
                        help='Minimum clamp for dynamic force weight b (default: 1.0).')
    parser.add_argument('--b-max', type=float, default=1000.0,
                        help='Maximum clamp for dynamic force weight b (default: 1000.0).')
    parser.add_argument('--swa-start-epoch', type=int, default=None,
                        help='Epoch to start SWA (Stochastic Weight Averaging) for loss weights. '
                             'After this epoch, a and b will be set to --swa-a and --swa-b values directly. '
                             'If not set, continuous linear growth/decay will be used (default: None).')
    parser.add_argument('--swa-a', type=float, default=None,
                        help='Energy weight a after SWA starts (default: None, must be set if --swa-start-epoch is set)')
    parser.add_argument('--swa-b', type=float, default=None,
                        help='Force weight b after SWA starts (default: None, must be set if --swa-start-epoch is set)')
    parser.add_argument('--ema-start-epoch', type=int, default=None,
                        help='Epoch to start EMA (Exponential Moving Average) for e3trans weights. '
                             'If not set, EMA is disabled (default: None). Recommended: start at ~60%%-80%% of total epochs.')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay factor in (0, 1). Larger -> smoother but slower (default: 0.999).')
    parser.add_argument('--use-ema-for-validation', action='store_true',
                        help='Use EMA weights for validation forward pass. Auto-enabled when --ema-start-epoch is set.')
    parser.add_argument('--save-ema-model', action='store_true',
                        help='Save EMA model weights into checkpoints. Auto-enabled when --ema-start-epoch is set.')
    parser.add_argument('--save-val-csv', action='store_true', default=False,
                        help='Save validation energy and force predictions to CSV files (default: True). '
                             'Files: val_energy_epoch{N}_batch{M}.csv and val_force_epoch{N}_batch{M}.csv')
    parser.add_argument('--no-save-val-csv', dest='save_val_csv', action='store_false',
                        help='Disable saving validation CSV files to reduce I/O overhead.')
    parser.add_argument('--log-val-batch-energy', action='store_true', default=False,
                        help='Log validation batch energy predictions to console (default: False). '
                             'If False, energy predictions are only logged to file, not console.')
    parser.add_argument('--force-shift-value', type=float, default=1.0,
                        help='Scaling factor for force labels (default: 1.0)')
    parser.add_argument('--stress-weight', '-c', type=float, default=0.0,
                        help='Weight for stress loss (default: 0.0, disabled). '
                             'Set > 0 to enable stress training via cell strain derivative. '
                             'Requires stress/virial data in the training XYZ files.')
    parser.add_argument('--c-min', type=float, default=0.0,
                        help='Minimum clamp for stress weight c (default: 0.0).')
    parser.add_argument('--c-max', type=float, default=1000.0,
                        help='Maximum clamp for stress weight c (default: 1000.0).')

    # Atomic reference energies (E0)
    parser.add_argument('--atomic-energy-file', type=str, default=None,
                        help='CSV file with columns Atom,E0 to load atomic reference energies. '
                             'If set, this explicit CLI value overrides checkpoint E0. '
                             'If not set, prefer checkpoint E0 when available; otherwise default to <data-dir>/fitted_E0.csv.')
    parser.add_argument('--atomic-energy-keys', type=int, nargs='+', default=None,
                        help='Atomic numbers for custom atomic reference energies (must match --atomic-energy-values length). '
                             'Highest priority override. Example: --atomic-energy-keys 1 6 7 8')
    parser.add_argument('--atomic-energy-values', type=float, nargs='+', default=None,
                        help='Atomic reference energies (E0) in eV corresponding to --atomic-energy-keys. '
                             'Highest priority override. Example: --atomic-energy-values -430.53 -821.03 -1488.19 -2044.35')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience in validation checks (not epochs; default: 20)')
    parser.add_argument('--vhat-clamp-interval', type=int, default=2000,
                        help='Interval (in batches) to clamp v_hat (default: 2000)')
    parser.add_argument('--max-vhat-growth', type=float, default=5.0,
                        help='Maximum growth factor for v_hat (default: 5.0)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Maximum gradient norm (default: 0.5)')
    parser.add_argument('--grad-log-interval', type=int, default=500,
                        help='Interval (in batches) to log gradient statistics (default: 100)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing preprocessed data files (default: data)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Single XYZ file for automatic preprocessing; program will split 90/10 train/val (optional)')
    parser.add_argument('--train-input-file', type=str, default=None,
                        help='Training set XYZ file. If both --train-input-file and --valid-input-file are set, '
                             'use them as train/valid directly (no auto split).')
    parser.add_argument('--valid-input-file', type=str, default=None,
                        help='Validation set XYZ file. Use together with --train-input-file to specify train/valid datasets.')
    parser.add_argument('--train-data', type=str, default=None,
                        help='Path to preprocessed training H5 file (e.g. processed_train.h5). '
                             'If both --train-data and --valid-data are set, use these files and skip preprocessing.')
    parser.add_argument('--valid-data', type=str, default=None,
                        help='Path to preprocessed validation H5 file. Use together with --train-data.')

    # 外部张量 / 物理张量（仅 pure-cartesian-ictd 支持）
    parser.add_argument('--external-tensor-rank', type=int, default=None,
                        help='External tensor rank for conv1 injection (e.g. 1=electric field). '
                             'Supported by pure-cartesian-ictd, pure-cartesian-sparse, and pure-cartesian-sparse-save. '
                             'Used with --external-field-file or --external-field-value.')
    parser.add_argument('--charge-file', type=str, default=None,
                        help='Per-structure charge label file (.npy/.npz/.h5, scalar per sample)')
    parser.add_argument('--dipole-file', type=str, default=None,
                        help='Per-structure dipole label file (.npy/.npz/.h5, shape Bx3)')
    parser.add_argument('--polarizability-file', type=str, default=None,
                        help='Per-structure polarizability label file (.npy/.npz/.h5, shape Bx3x3)')
    parser.add_argument('--quadrupole-file', type=str, default=None,
                        help='Per-structure quadrupole label file (.npy/.npz/.h5, shape Bx3x3)')
    parser.add_argument('--external-field-file', type=str, default=None,
                        help='Per-structure external field file (.npy/.npz/.h5, shape Bx3). '
                             'Requires --external-tensor-rank. Mutually exclusive with --external-field-value.')
    parser.add_argument('--external-field-value', type=float, nargs='+', default=None,
                        metavar='V',
                        help=(
                            'Uniform external field applied to ALL samples (injected into H5). '
                            'Number of values must equal 3^rank (Cartesian tensor, row-major). '
                            'Mutually exclusive with --external-field-file.\n'
                            '  rank 0 (1 value):  scalar field strength\n'
                            '  rank 1 (3 values): Fx  Fy  Fz  (Cartesian x/y/z, e.g. electric field V/Å)\n'
                            '  rank 2 (9 values): Txx Txy Txz  Tyx Tyy Tyz  Tzx Tzy Tzz  (3×3 row-major)\n'
                            '  rank L (3^L values): full rank-L Cartesian tensor, row-major\n'
                            'Auto-sets --external-tensor-rank if not given.'
                        ))
    parser.add_argument('--magnetic-field-file', type=str, default=None,
                        help='Per-structure magnetic field file (.npy/.npz/.h5, shape Bx3). '
                             'Can be used together with --external-field-file/--external-field-value '
                             'to inject electric and magnetic fields simultaneously.')
    parser.add_argument('--magnetic-field-value', type=float, nargs=3, default=None,
                        metavar=('BX', 'BY', 'BZ'),
                        help='Uniform magnetic field applied to ALL samples (shape Bx3 after H5 injection). '
                             'Mutually exclusive with --magnetic-field-file.')
    parser.add_argument('--fidelity-id-file', type=str, default=None,
                        help='Per-structure multi-fidelity level ids (.npy/.npz/.h5, integer scalar per sample).')
    parser.add_argument('--num-fidelity-levels', type=int, default=None,
                        help='Number of discrete fidelity levels for graph-level fidelity conditioning. '
                             'Required for training new multi-fidelity models unless restored from checkpoint.')
    parser.add_argument('--multi-fidelity-mode', type=str, default=None, choices=['conditioning', 'delta-baseline'],
                        help='Multi-fidelity architecture mode. "conditioning" keeps a shared readout, '
                             '"delta-baseline" uses fidelity 0 as baseline and learns residual energy heads for higher fidelities.')
    parser.add_argument('--fidelity-loss-weights', type=str, default=None,
                        help='Optional per-fidelity loss weights for training, e.g. "0:1.0,1:3.0". '
                             'Applied to energy/force/stress SmoothL1 losses; unspecified fidelity ids default to 1.0.')
    parser.add_argument('--delta-regularization-weight', type=float, default=0.0,
                        help='Optional L2 regularization weight applied to multi-fidelity delta heads '
                             'when --multi-fidelity-mode=delta-baseline.')
    parser.add_argument('--external-tensor-irrep', type=str, default=None,
                        help="Optional parity-aware irrep tag for external tensor injection, e.g. '1o' or '1e'.")
    parser.add_argument('--external-tensor-parity', type=str, default=None, choices=['e', 'o'],
                        help='Optional parity shorthand for external tensor injection. Ignored when --external-tensor-irrep is set.')
    parser.add_argument('--o3-irrep-preset', type=str, default=None,
                        choices=['auto', 'minimal', 'balanced', 'full'],
                        help='Active-irrep preset for pure-cartesian-ictd-o3. '
                             '"auto" keeps canonical SO(3) irreps plus parity-sensitive inputs/outputs, '
                             '"minimal" keeps only required irreps, '
                             '"balanced" adds one extra coupling shell around auto, '
                             '"full" keeps every (l,e/o) block.')
    parser.add_argument('--o3-active-irreps', type=str, default=None,
                        help="Explicit active irrep override for pure-cartesian-ictd-o3, e.g. '0e,1e,2e'. "
                             'When set, this overrides --o3-irrep-preset but still auto-includes required inputs/outputs.')
    parser.add_argument('--extra-per-node-file', type=str, default=None,
                        help='Per-node label HDF5 (sample_0, sample_1, ... with charge_per_atom, dipole_per_atom, etc.)')
    parser.add_argument('--physical-tensors', type=str, default=None,
                        help='Comma-separated physical tensor outputs: charge,dipole,magnetic_moment,polarizability,quadrupole. '
                             'Requires corresponding label files. Supported by pure-cartesian-ictd, '
                             'pure-cartesian-sparse, and pure-cartesian-sparse-save.')
    parser.add_argument('--physical-tensors-per-node', type=str, default=None,
                        help='Comma-separated per-node outputs: charge_per_atom,dipole_per_atom,magnetic_moment_per_atom,polarizability_per_atom,quadrupole_per_atom,born_effective_charge_per_atom. '
                             'Requires --extra-per-node-file. Supported by pure-cartesian-ictd, '
                             'pure-cartesian-sparse, and pure-cartesian-sparse-save.')
    parser.add_argument('--physical-tensor-reduce', type=str, default='sum',
                        choices=['sum', 'mean', 'none'],
                        help='Reduce mode for graph-level physical tensors (default: sum). '
                             'Per-node tensors always use reduce=none.')
    parser.add_argument('--physical-tensor-weights', type=str, default=None,
                        help='Loss weights for physical tensors: "charge:1.0,dipole:2.0,magnetic_moment:1.0,polarizability:1.0,quadrupole:1.0,born_effective_charge:1.0". '
                             'Per-node (charge_per_atom) uses charge weight. Unspecified default 1.0.')
    parser.add_argument('--bec-derivative-weight', type=float, default=1.0,
                        help='Additional loss weight for dF/dE-derived Born effective charge supervision when born_effective_charge_per_atom labels and rank-1 external_field are present.')
    parser.add_argument('--bec-consistency-weight', type=float, default=0.25,
                        help='Additional loss weight tying the explicit born_effective_charge_per_atom head to the dF/dE-derived Born effective charge.')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training (DDP)')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training (set automatically by torchrun)')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend (default: nccl for GPU, gloo for CPU)')
    parser.add_argument('--init-method', type=str, default='env://',
                        help='Distributed initialization method (default: env://)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Model architecture hyperparameters
    parser.add_argument('--max-atomvalue', type=int, default=None,
                        help='Maximum atomic number in atom embedding. If not set, restore from checkpoint when available, else use 10.')
    parser.add_argument('--embedding-dim', type=int, default=None,
                        help='Atom embedding dimension. If not set, restore from checkpoint when available, else use 16.')
    parser.add_argument('--embed-size', type=int, nargs='+', default=None,
                        help='Hidden layer sizes for readout MLP. If not set, restore from checkpoint when available, else use 128 128 128.')
    parser.add_argument('--output-size', type=int, default=None,
                        help='Output size for atom readout MLP. If not set, restore from checkpoint when available, else use 8.')
    parser.add_argument('--lmax', type=int, default=None,
                        help='Maximum L value for spherical harmonics in irreps. If not set, restore from checkpoint when available, else use 2.')
    parser.add_argument('--irreps-output-conv-channels', type=int, default=None,
                        help='Number of channels for irreps_output_conv (e.g., 64 for lmax=2 gives "64x0e + 64x1o + 64x2e"). If not set, uses channel_in from config (default: 64)')
    parser.add_argument('--function-type', type=str, default=None,
                        choices=['gaussian', 'bessel', 'fourier', 'cosine', 'smooth_finite'],
                        help='Basis function type for radial basis. If not set, restore from checkpoint when available, else use gaussian.')
    parser.add_argument('--tensor-product-mode', type=str, default=None,
                        choices=['spherical', 'spherical-save', 'spherical-save-cue', 'partial-cartesian', 'partial-cartesian-loose', 'pure-cartesian', 'pure-cartesian-sparse', 'pure-cartesian-sparse-save', 'pure-cartesian-ictd', 'pure-cartesian-ictd-o3', 'pure-cartesian-ictd-save'],
                        help='Tensor product mode. If not set, restore from checkpoint when available, else use spherical. '
                             '"spherical" uses e3nn spherical harmonics (default), '
                             '"spherical-save" uses channelwise edge convolution (e3nn backend; fewer params, same irreps), '
                             '"spherical-save-cue" uses channelwise edge convolution (cuEquivariance backend; requires cuequivariance-torch), '
                             '"partial-cartesian" uses Cartesian tensor products with EquivariantTensorProduct (strictly equivariant), '
                             '"partial-cartesian-loose" uses non-strictly-equivariant Cartesian tensor products (norm product approximation, not strictly equivariant), '
                             '"pure-cartesian" uses full rank Cartesian tensors (3^L) with delta/epsilon contractions (most pure), '
                             '"pure-cartesian-sparse" uses a sparse pure-cartesian delta/epsilon tensor product (O(3) strict) by restricting rank-rank paths, '
                             '"pure-cartesian-sparse-save" uses the same sparse pure-cartesian implementation under a dedicated save-mode name, '
                             '"pure-cartesian-ictd" uses pure_cartesian_ictd_layers_full (ICTD, DDP supported). '
                             '"pure-cartesian-ictd-o3" uses pure_cartesian_ictd_layers_o3 (strict parity-aware O(3) ICTD). '
                             '"pure-cartesian-ictd-save" uses pure_cartesian_ictd_layers (original ICTD, same readout, DDP supported). '
                             'Note: ICTD inference is typically ~3x faster than spherical-save.')
    parser.add_argument('--max-rank-other', type=int, default=None,
                        help='Max rank for sparse tensor product in pure-cartesian-sparse / pure-cartesian-sparse-save mode. '
                             'If not set, restore from checkpoint when available, else use 1. '
                             'Only interactions where min(L1, L2) <= max_rank_other are allowed. '
                             'Larger values allow more interactions but increase parameters and computation.')
    parser.add_argument('--k-policy', type=str, default=None,
                        choices=['k0', 'k1', 'both'],
                        help='K policy for sparse tensor product in pure-cartesian-sparse / pure-cartesian-sparse-save mode. '
                             'If not set, restore from checkpoint when available, else use k0. '
                             'k0: only k=0 (promotes higher rank), k1: only k=1 (contracts to lower rank), both: keep both')
    parser.add_argument('--num-interaction', type=int, default=None,
                        help='Number of message-passing steps (conv layers) per block. '
                             'If not set, restore from checkpoint when available, else use 2. '
                             'Used by: pure-cartesian, pure-cartesian-ictd, pure-cartesian-ictd-save, pure-cartesian-sparse, pure-cartesian-sparse-save, '
                             'partial-cartesian, partial-cartesian-loose, spherical, spherical-save, spherical-save-cue. Must be >= 2.')
    parser.add_argument('--invariant-channels', type=int, default=None,
                        help='Per-interaction invariant readout channels used by product_3-style scalar invariant blocks. '
                             'If not set, restore from checkpoint when available, else use 32. '
                             'Applies to: spherical, spherical-save, partial-cartesian, partial-cartesian-loose, '
                             'pure-cartesian, pure-cartesian-sparse, pure-cartesian-sparse-save, pure-cartesian-ictd, pure-cartesian-ictd-save.')

    # ICTD path pruning controls (pure-cartesian-ictd and pure-cartesian-ictd-save)
    parser.add_argument('--ictd-tp-path-policy', type=str, default=None,
                        choices=['full', 'max_rank_other'],
                        help='Path policy for ICTD tensor products in pure-cartesian-ictd mode. '
                             'If not set, restore from checkpoint when available, else use full. '
                             'full: keep all CG-allowed (l1,l2->l3) paths; '
                             'max_rank_other: keep only paths with min(l1,l2) <= --ictd-tp-max-rank-other.')
    parser.add_argument('--ictd-tp-max-rank-other', type=int, default=None,
                        help='Used when --ictd-tp-path-policy=max_rank_other. '
                             'If not set, prefer checkpoint metadata when available. '
                             'Keeps only paths with min(l1,l2) <= this value (e.g. 1 keeps scalar/vector couplings).')
    parser.add_argument('--long-range-mode', type=str, default=None,
                        choices=['none', 'latent-coulomb', 'isolated-far-field-v1', 'isolated-far-field-v2', 'reciprocal-spectral-v1'],
                        help='Optional long-range head. '
                             'If not set, restore from checkpoint when available, else use none. '
                             '"latent-coulomb" predicts latent charges from invariant features and adds a screened O(N^2) real-space energy. '
                             '"isolated-far-field-v1" adds a lightweight nonperiodic isolated-system far-field correction from invariant features. '
                             '"isolated-far-field-v2" adds an explicit-shell nonperiodic isolated-system far-field correction with optional tail shell. '
                             '"reciprocal-spectral-v1" predicts latent reciprocal-space sources and applies either '
                             'the legacy direct_kspace kernel or the LES-style mesh_fft reciprocal backend (recommended).')
    parser.add_argument('--long-range-hidden-dim', type=int, default=None,
                        help='Hidden size of the latent long-range source head. '
                             'If not set, restore from checkpoint when available, else use 64.')
    parser.add_argument('--long-range-boundary', type=str, default=None,
                        choices=['nonperiodic', 'periodic', 'slab'],
                        help='Boundary condition for the long-range head. '
                             '"periodic" = fully periodic 3D cell. '
                             '"slab" = x/y periodic + z vacuum padding. '
                             'If not set, restore from checkpoint when available, else use nonperiodic.')
    parser.add_argument('--long-range-neutralize', dest='long_range_neutralize', action='store_true',
                        help='Neutralize latent charges per graph before pair summation (default).')
    parser.add_argument('--no-long-range-neutralize', dest='long_range_neutralize', action='store_false',
                        help='Disable per-graph latent charge neutralization.')
    parser.set_defaults(long_range_neutralize=None)
    parser.add_argument('--long-range-filter-hidden-dim', type=int, default=None,
                        help='Hidden size of the reciprocal spectral filter MLP. '
                             'If not set, restore from checkpoint when available, else use 64.')
    parser.add_argument('--long-range-kmax', type=int, default=None,
                        help='Maximum reciprocal lattice index in each direction for reciprocal-spectral-v1 '
                             'when --long-range-reciprocal-backend=direct_kspace. '
                             'If not set, restore from checkpoint when available, else use 2.')
    parser.add_argument('--long-range-mesh-size', type=int, default=None,
                        help='Uniform FFT mesh size per lattice direction for reciprocal-spectral-v1 '
                             'when --long-range-reciprocal-backend=mesh_fft. '
                             'If not set, restore from checkpoint when available, else use 16.')
    parser.add_argument('--long-range-slab-padding-factor', type=int, default=None,
                        help='Vacuum padding multiplier along the slab-normal direction when '
                             '--long-range-boundary=slab and --long-range-reciprocal-backend=mesh_fft. '
                             'If not set, restore from checkpoint when available, else use 2.')
    parser.add_argument('--long-range-include-k0', dest='long_range_include_k0', action='store_true',
                        help='Include the k=0 term in reciprocal-spectral-v1 (default: disabled).')
    parser.add_argument('--no-long-range-include-k0', dest='long_range_include_k0', action='store_false',
                        help='Exclude the k=0 term in reciprocal-spectral-v1 (default).')
    parser.set_defaults(long_range_include_k0=None)
    parser.add_argument('--long-range-source-channels', type=int, default=None,
                        help='Number of latent reciprocal source channels for reciprocal-spectral-v1. '
                             'If not set, restore from checkpoint when available, else use 1.')
    parser.add_argument('--long-range-backend', type=str, default=None,
                        choices=['dense_pairwise', 'tree_fmm'],
                        help='Real-space backend for --long-range-mode latent-coulomb. '
                             '"dense_pairwise" keeps the legacy O(N^2) all-pairs path. '
                             '"tree_fmm" enables the new open-boundary Barnes-Hut style tree backend '
                             '(currently nonperiodic only). '
                             'If not set, restore from checkpoint when available, else use dense_pairwise.')
    parser.add_argument('--long-range-reciprocal-backend', type=str, default=None,
                        choices=['direct_kspace', 'mesh_fft'],
                        help='Reciprocal backend implementation for reciprocal-spectral-v1. '
                             '"direct_kspace" = legacy direct k-space kernel (currently periodic only). '
                             '"mesh_fft" = LES-style FFT/mesh backend (recommended; supports periodic and slab). '
                             'If not set, restore from checkpoint when available, else use direct_kspace.')
    parser.add_argument('--long-range-energy-partition', type=str, default=None,
                        choices=['potential', 'uniform'],
                        help='How reciprocal graph energy is partitioned back to atoms. '
                             'Recommended: potential. '
                             'If not set, restore from checkpoint when available, else use potential.')
    parser.add_argument('--long-range-green-mode', type=str, default=None,
                        choices=['poisson', 'learned_poisson'],
                        help='Reciprocal Green-kernel family for reciprocal-spectral-v1. '
                             'Recommended first-stage choice: poisson. '
                             'If not set, restore from checkpoint when available, else use poisson.')
    parser.add_argument('--long-range-assignment', type=str, default=None,
                        choices=['cic', 'tsc', 'pcs'],
                        help='Particle-to-mesh assignment scheme for mesh_fft reciprocal long-range. '
                             'Current recommended choices: cic (fast baseline), tsc (better accuracy), pcs (highest-order). '
                             'If not set, restore from checkpoint when available, else use cic.')
    parser.add_argument('--long-range-mesh-fft-full-ewald', dest='long_range_mesh_fft_full_ewald', action='store_true',
                        help='Enable the complete Ewald correction stack for --long-range-reciprocal-backend=mesh_fft. '
                             'This adds real-space, self, and background terms and is disabled by default.')
    parser.add_argument('--no-long-range-mesh-fft-full-ewald', dest='long_range_mesh_fft_full_ewald', action='store_false',
                        help='Keep mesh_fft on the faster reciprocal-only path (default).')
    parser.set_defaults(long_range_mesh_fft_full_ewald=None)
    parser.add_argument('--long-range-theta', type=float, default=None,
                        help='Opening-angle parameter for --long-range-backend tree_fmm. '
                             'Smaller values are more accurate and more expensive. '
                             'If not set, restore from checkpoint when available, else use 0.5.')
    parser.add_argument('--long-range-leaf-size', type=int, default=None,
                        help='Leaf occupancy threshold for --long-range-backend tree_fmm. '
                             'If not set, restore from checkpoint when available, else use 32.')
    parser.add_argument('--long-range-multipole-order', type=int, default=None,
                        help='Multipole order for --long-range-backend tree_fmm. '
                             'The current correctness-first implementation supports only 0. '
                             'If not set, restore from checkpoint when available, else use 0.')
    parser.add_argument('--long-range-far-source-dim', type=int, default=None,
                        help='Explicit-shell far-field latent source width for --long-range-mode isolated-far-field-v2. '
                             'If not set, restore from checkpoint when available, else use 16.')
    parser.add_argument('--long-range-far-num-shells', type=int, default=None,
                        help='Number of explicit far-field shells for --long-range-mode isolated-far-field-v2. '
                             'If not set, restore from checkpoint when available, else use 3.')
    parser.add_argument('--long-range-far-shell-growth', type=float, default=None,
                        help='Geometric growth factor between far shells for --long-range-mode isolated-far-field-v2. '
                             'The explicit far radius is derived internally as growth ** num_shells. '
                             'If not set, restore from checkpoint when available, else use 2.0.')
    parser.add_argument('--long-range-far-tail', dest='long_range_far_tail', action='store_true',
                        help='Enable tail shell beyond the explicit far-field radius for isolated-far-field-v2 (default).')
    parser.add_argument('--no-long-range-far-tail', dest='long_range_far_tail', action='store_false',
                        help='Disable the tail shell for isolated-far-field-v2.')
    parser.set_defaults(long_range_far_tail=None)
    parser.add_argument('--long-range-far-tail-bins', type=int, default=None,
                        help='Number of coarse tail bins used by isolated-far-field-v2 when tail is enabled. '
                             'If not set, restore from checkpoint when available, else use 2.')
    parser.add_argument('--long-range-far-stats', type=str, default=None,
                        help='Comma-separated explicit-shell statistics for isolated-far-field-v2. '
                             'Current implementation supports only mean,count,mean_r,rms_r. '
                             'If not set, restore from checkpoint when available, else use mean,count,mean_r,rms_r.')
    parser.add_argument('--long-range-far-max-radius-multiplier', type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--long-range-far-source-norm', dest='long_range_far_source_norm', action='store_true',
                        help='Apply LayerNorm to the projected far-field latent source in isolated-far-field-v2 (default).')
    parser.add_argument('--no-long-range-far-source-norm', dest='long_range_far_source_norm', action='store_false',
                        help='Disable LayerNorm on the projected far-field latent source in isolated-far-field-v2.')
    parser.set_defaults(long_range_far_source_norm=None)
    parser.add_argument('--long-range-far-gate-init', type=float, default=None,
                        help='Initial scalar gate on shell-encoded far features for isolated-far-field-v2. '
                             'If not set, restore from checkpoint when available, else use 0.0.')
    parser.add_argument('--feature-spectral-mode', type=str, default=None,
                        choices=['none', 'fft'],
                        help='Optional low-rank feature-space spectral residual block before proj_total. '
                             'If not set, restore from checkpoint when available, else use none.')
    parser.add_argument('--feature-spectral-bottleneck-dim', type=int, default=None,
                        help='Bottleneck channel count C_lr for the feature spectral block.')
    parser.add_argument('--feature-spectral-mesh-size', type=int, default=None,
                        help='Uniform FFT mesh size per lattice direction for the feature spectral block.')
    parser.add_argument('--feature-spectral-filter-hidden-dim', type=int, default=None,
                        help='Hidden size of the spectral filter MLP used on the FFT mesh.')
    parser.add_argument('--feature-spectral-boundary', type=str, default=None,
                        choices=['periodic', 'slab'],
                        help='Boundary model for the feature spectral block. '
                             'If not set, restore from checkpoint when available, else use periodic. '
                             '"slab" means x/y periodic with z vacuum padding.')
    parser.add_argument('--feature-spectral-slab-padding-factor', type=int, default=None,
                        help='Vacuum padding multiplier along the nonperiodic slab direction for the feature spectral block. '
                             'If not set, restore from checkpoint when available, else use 2.')
    parser.add_argument('--feature-spectral-neutralize', dest='feature_spectral_neutralize', action='store_true',
                        help='Subtract per-graph bottleneck mean before FFT filtering (default).')
    parser.add_argument('--no-feature-spectral-neutralize', dest='feature_spectral_neutralize', action='store_false',
                        help='Disable per-graph bottleneck neutralization before FFT filtering.')
    parser.set_defaults(feature_spectral_neutralize=None)
    parser.add_argument('--feature-spectral-include-k0', dest='feature_spectral_include_k0', action='store_true',
                        help='Include the k=0 mode in the feature spectral block.')
    parser.add_argument('--no-feature-spectral-include-k0', dest='feature_spectral_include_k0', action='store_false',
                        help='Exclude the k=0 mode in the feature spectral block (default).')
    parser.set_defaults(feature_spectral_include_k0=None)
    parser.add_argument('--feature-spectral-assignment', type=str, default=None,
                        choices=['cic', 'tsc', 'pcs'],
                        help='Particle-to-mesh assignment scheme for the feature spectral FFT block. '
                             'Recommended choices: cic (fast baseline), tsc (better accuracy), pcs (highest-order). '
                             'If not set, restore from checkpoint when available, else use cic.')
    parser.add_argument('--feature-spectral-gate-init', type=float, default=None,
                        help='Initial residual gate value for the feature spectral block.')
    parser.add_argument('--zbl-enabled', dest='zbl_enabled', action='store_true',
                        help='Enable ZBL short-range repulsion and save it into checkpoint metadata.')
    parser.add_argument('--no-zbl-enabled', dest='zbl_enabled', action='store_false',
                        help='Disable ZBL short-range repulsion.')
    parser.set_defaults(zbl_enabled=None)
    parser.add_argument('--zbl-inner-cutoff', type=float, default=None,
                        help='Inner cutoff in Angstrom where full ZBL is applied.')
    parser.add_argument('--zbl-outer-cutoff', type=float, default=None,
                        help='Outer cutoff in Angstrom where ZBL decays to zero.')
    parser.add_argument('--zbl-exponent', type=float, default=None,
                        help='Screening exponent in the ZBL screening length formula.')
    parser.add_argument('--zbl-energy-scale', type=float, default=None,
                        help='Global multiplicative scale for the ZBL energy term.')

    # Validation acceleration (evaluation-only; training uses double backward and is NOT compiled)
    parser.add_argument('--compile-val', type=str, default='none',
                        choices=['none', 'e3trans'],
                        help='Enable torch.compile during validation only. '
                             '"e3trans" compiles the eval forward used in validate(). '
                             'Training forward/backward is NOT compiled (double backward unsupported).')
    parser.add_argument('--compile-val-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode for validation (default: reduce-overhead)')
    parser.add_argument('--compile-val-fullgraph', action='store_true',
                        help='Pass fullgraph=True to torch.compile for validation (may fail more often).')
    parser.add_argument('--compile-val-dynamic', action='store_true',
                        help='Pass dynamic=True to torch.compile for validation.')
    parser.add_argument('--compile-val-precache', action='store_true',
                        help='Run one eager forward on the first validation batch before compiling (recommended for ICTD).')

    # 推理模式：保存到 checkpoint，供 evaluate/inference_ddp 使用；TorchScript/LAMMPS 导出始终只输出能量和力
    parser.add_argument('--inference-output-physical-tensors', action='store_true',
                        help='Save in checkpoint: inference should output physical tensors (dipole, polarizability, etc.). '
                             'Default False: MD/LAMMPS 接口仅需能量和力，不输出物理张量。')

    # ---- Auto-launch parameters (self-relaunch via torchrun / srun) ----
    parser.add_argument('--n-gpu', type=int, default=1,
                        help='Number of GPUs to use. 1 (default): single-process training. '
                             '>1: auto-relaunch via torchrun --nproc_per_node=N --distributed. '
                             'Backward compatible: manual "torchrun ... --distributed" still works.')
    parser.add_argument('--nnodes', type=int, default=1,
                        help='Number of nodes for multi-node DDP. 1 (default): single-node. '
                             '>1: multi-node via torchrun rendezvous.')
    parser.add_argument('--master-addr', type=str, default='auto',
                        help='Rendezvous address for DDP. "auto" (default): resolve from SLURM or hostname.')
    parser.add_argument('--master-port', type=int, default=29500,
                        help='Rendezvous port for DDP (default: 29500).')
    parser.add_argument('--launcher', type=str, default='auto',
                        choices=['auto', 'local', 'slurm'],
                        help='Multi-node launcher. "auto": detect SLURM. "slurm": wrap with srun. '
                             '"local": torchrun only.')

    args = parser.parse_args()

    # ---- Auto-relaunch via torchrun when --n-gpu > 1 or --nnodes > 1 ----
    # Skip if already inside a torchrun worker (--distributed is set by the
    # relaunched command, so we never relaunch twice).
    if (args.n_gpu > 1 or args.nnodes > 1) and not args.distributed:
        _relaunch_with_torchrun(args)
        return  # unreachable when _relaunch exits, but defensive

    checkpoint_arch = maybe_load_checkpoint(args.checkpoint, map_location='cpu')
    user_far_max_radius_multiplier = args.long_range_far_max_radius_multiplier
    resolved_arch = resolve_model_architecture(
        checkpoint_arch,
        overrides={
            "dtype": args.dtype,
            "max_radius": args.max_radius,
            "max_atomvalue": args.max_atomvalue,
            "embedding_dim": args.embedding_dim,
            "embed_size": args.embed_size,
            "output_size": args.output_size,
            "lmax": args.lmax,
            "irreps_output_conv_channels": args.irreps_output_conv_channels,
            "function_type": args.function_type,
            "tensor_product_mode": args.tensor_product_mode,
            "num_interaction": args.num_interaction,
            "invariant_channels": args.invariant_channels,
            "o3_irrep_preset": args.o3_irrep_preset,
            "o3_active_irreps": args.o3_active_irreps,
            "max_rank_other": args.max_rank_other,
            "k_policy": args.k_policy,
            "ictd_tp_path_policy": args.ictd_tp_path_policy,
            "ictd_tp_max_rank_other": args.ictd_tp_max_rank_other,
            "long_range_mode": args.long_range_mode,
            "long_range_hidden_dim": args.long_range_hidden_dim,
            "long_range_boundary": args.long_range_boundary,
            "long_range_neutralize": args.long_range_neutralize,
            "long_range_filter_hidden_dim": args.long_range_filter_hidden_dim,
            "long_range_kmax": args.long_range_kmax,
            "long_range_mesh_size": args.long_range_mesh_size,
            "long_range_slab_padding_factor": args.long_range_slab_padding_factor,
            "long_range_include_k0": args.long_range_include_k0,
            "long_range_source_channels": args.long_range_source_channels,
            "long_range_backend": args.long_range_backend,
            "long_range_reciprocal_backend": args.long_range_reciprocal_backend,
            "long_range_energy_partition": args.long_range_energy_partition,
            "long_range_green_mode": args.long_range_green_mode,
            "long_range_assignment": args.long_range_assignment,
            "long_range_mesh_fft_full_ewald": args.long_range_mesh_fft_full_ewald,
            "long_range_theta": args.long_range_theta,
            "long_range_leaf_size": args.long_range_leaf_size,
            "long_range_multipole_order": args.long_range_multipole_order,
            "long_range_far_source_dim": args.long_range_far_source_dim,
            "long_range_far_num_shells": args.long_range_far_num_shells,
            "long_range_far_shell_growth": args.long_range_far_shell_growth,
            "long_range_far_tail": args.long_range_far_tail,
            "long_range_far_tail_bins": args.long_range_far_tail_bins,
            "long_range_far_stats": args.long_range_far_stats,
            "long_range_far_max_radius_multiplier": args.long_range_far_max_radius_multiplier,
            "long_range_far_source_norm": args.long_range_far_source_norm,
            "long_range_far_gate_init": args.long_range_far_gate_init,
            "feature_spectral_mode": args.feature_spectral_mode,
            "feature_spectral_bottleneck_dim": args.feature_spectral_bottleneck_dim,
            "feature_spectral_mesh_size": args.feature_spectral_mesh_size,
            "feature_spectral_filter_hidden_dim": args.feature_spectral_filter_hidden_dim,
            "feature_spectral_boundary": args.feature_spectral_boundary,
            "feature_spectral_slab_padding_factor": args.feature_spectral_slab_padding_factor,
            "feature_spectral_neutralize": args.feature_spectral_neutralize,
            "feature_spectral_include_k0": args.feature_spectral_include_k0,
            "feature_spectral_assignment": args.feature_spectral_assignment,
            "feature_spectral_gate_init": args.feature_spectral_gate_init,
            "zbl_enabled": args.zbl_enabled,
            "zbl_inner_cutoff": args.zbl_inner_cutoff,
            "zbl_outer_cutoff": args.zbl_outer_cutoff,
            "zbl_exponent": args.zbl_exponent,
            "zbl_energy_scale": args.zbl_energy_scale,
        },
    )
    args.dtype = resolved_arch["dtype"]
    args.max_radius = resolved_arch["max_radius"]
    args.max_atomvalue = resolved_arch["max_atomvalue"]
    args.embedding_dim = resolved_arch["embedding_dim"]
    args.embed_size = resolved_arch["embed_size"]
    args.output_size = resolved_arch["output_size"]
    args.lmax = resolved_arch["lmax"]
    args.irreps_output_conv_channels = resolved_arch["irreps_output_conv_channels"]
    args.function_type = resolved_arch["function_type"]
    args.tensor_product_mode = resolved_arch["tensor_product_mode"]
    args.num_interaction = resolved_arch["num_interaction"]
    args.invariant_channels = resolved_arch["invariant_channels"]
    args.max_rank_other = resolved_arch["max_rank_other"]
    args.k_policy = resolved_arch["k_policy"]
    args.ictd_tp_path_policy = resolved_arch["ictd_tp_path_policy"]
    args.ictd_tp_max_rank_other = resolved_arch["ictd_tp_max_rank_other"]
    args.long_range_mode = resolved_arch["long_range_mode"]
    args.long_range_hidden_dim = resolved_arch["long_range_hidden_dim"]
    args.long_range_boundary = resolved_arch["long_range_boundary"]
    args.long_range_neutralize = resolved_arch["long_range_neutralize"]
    args.long_range_filter_hidden_dim = resolved_arch["long_range_filter_hidden_dim"]
    args.long_range_kmax = resolved_arch["long_range_kmax"]
    args.long_range_mesh_size = resolved_arch["long_range_mesh_size"]
    args.long_range_slab_padding_factor = resolved_arch["long_range_slab_padding_factor"]
    args.long_range_include_k0 = resolved_arch["long_range_include_k0"]
    args.long_range_source_channels = resolved_arch["long_range_source_channels"]
    args.long_range_backend = resolved_arch["long_range_backend"]
    args.long_range_reciprocal_backend = resolved_arch["long_range_reciprocal_backend"]
    args.long_range_energy_partition = resolved_arch["long_range_energy_partition"]
    args.long_range_green_mode = resolved_arch["long_range_green_mode"]
    args.long_range_assignment = resolved_arch["long_range_assignment"]
    args.long_range_mesh_fft_full_ewald = resolved_arch["long_range_mesh_fft_full_ewald"]
    args.long_range_theta = resolved_arch["long_range_theta"]
    args.long_range_leaf_size = resolved_arch["long_range_leaf_size"]
    args.long_range_multipole_order = resolved_arch["long_range_multipole_order"]
    args.long_range_far_source_dim = resolved_arch["long_range_far_source_dim"]
    args.long_range_far_num_shells = resolved_arch["long_range_far_num_shells"]
    args.long_range_far_shell_growth = resolved_arch["long_range_far_shell_growth"]
    args.long_range_far_tail = resolved_arch["long_range_far_tail"]
    args.long_range_far_tail_bins = resolved_arch["long_range_far_tail_bins"]
    args.long_range_far_stats = resolved_arch["long_range_far_stats"]
    args.long_range_far_max_radius_multiplier = resolved_arch["long_range_far_max_radius_multiplier"]
    args.long_range_far_source_norm = resolved_arch["long_range_far_source_norm"]
    args.long_range_far_gate_init = resolved_arch["long_range_far_gate_init"]
    args.feature_spectral_mode = resolved_arch["feature_spectral_mode"]
    args.feature_spectral_bottleneck_dim = resolved_arch["feature_spectral_bottleneck_dim"]
    args.feature_spectral_mesh_size = resolved_arch["feature_spectral_mesh_size"]
    args.feature_spectral_filter_hidden_dim = resolved_arch["feature_spectral_filter_hidden_dim"]
    args.feature_spectral_boundary = resolved_arch["feature_spectral_boundary"]
    args.feature_spectral_slab_padding_factor = resolved_arch["feature_spectral_slab_padding_factor"]
    args.feature_spectral_neutralize = resolved_arch["feature_spectral_neutralize"]
    args.feature_spectral_include_k0 = resolved_arch["feature_spectral_include_k0"]
    args.feature_spectral_assignment = resolved_arch["feature_spectral_assignment"]
    args.feature_spectral_gate_init = resolved_arch["feature_spectral_gate_init"]
    args.zbl_enabled = resolved_arch["zbl_enabled"]
    args.zbl_inner_cutoff = resolved_arch["zbl_inner_cutoff"]
    args.zbl_outer_cutoff = resolved_arch["zbl_outer_cutoff"]
    args.zbl_exponent = resolved_arch["zbl_exponent"]
    args.zbl_energy_scale = resolved_arch["zbl_energy_scale"]
    if args.external_tensor_rank is None:
        args.external_tensor_rank = resolved_arch["external_tensor_rank"]
    if args.external_tensor_irrep is None:
        args.external_tensor_irrep = resolved_arch.get("external_tensor_irrep")
    restored_external_tensor_specs = resolved_arch.get("external_tensor_specs")
    if args.o3_irrep_preset is None:
        args.o3_irrep_preset = resolved_arch.get("o3_irrep_preset")
    if args.o3_active_irreps is None:
        args.o3_active_irreps = resolved_arch.get("o3_active_irreps")
    restored_physical_tensor_outputs = resolved_arch["physical_tensor_outputs"]

    # --- Dataset option validation (pairs must be both set or both unset) ---
    if (args.train_data is None) != (args.valid_data is None):
        raise ValueError("Must specify both --train-data and --valid-data together, or neither.")
    if (args.train_input_file is None) != (args.valid_input_file is None):
        raise ValueError("Must specify both --train-input-file and --valid-input-file together, or neither.")

    # --- External tensor / physical tensor validation ---
    if args.external_field_file and args.external_field_value:
        raise ValueError("--external-field-file and --external-field-value are mutually exclusive")
    if args.magnetic_field_file and args.magnetic_field_value:
        raise ValueError("--magnetic-field-file and --magnetic-field-value are mutually exclusive")
    if args.external_field_value:
        from molecular_force_field.active_learning.data_merge import external_field_tensor_shape
        shape = external_field_tensor_shape(len(args.external_field_value))
        inferred_rank = len(shape) if shape != (1,) else 0
        if args.external_tensor_rank is None:
            args.external_tensor_rank = inferred_rank
            logging.info(f"Auto-set --external-tensor-rank={inferred_rank} from --external-field-value ({len(args.external_field_value)} values)")
        elif args.external_tensor_rank != inferred_rank:
            raise ValueError(
                f"--external-tensor-rank={args.external_tensor_rank} conflicts with "
                f"--external-field-value ({len(args.external_field_value)} values → rank {inferred_rank})"
            )
    if args.magnetic_field_file or args.magnetic_field_value:
        if args.external_tensor_rank is not None and int(args.external_tensor_rank) != 1 and (args.external_field_file or args.external_field_value):
            raise ValueError("Simultaneous electric + magnetic field embedding currently requires rank-1 electric field input")
        if args.external_tensor_irrep is not None and str(args.external_tensor_irrep).strip() not in {"1o"} and (args.external_field_file or args.external_field_value):
            raise ValueError("When combining electric and magnetic fields, the legacy --external-tensor-irrep should describe the electric field and must be '1o'")
    if args.external_field_file and not args.external_tensor_rank:
        raise ValueError("--external-field-file requires --external-tensor-rank (e.g. 1 for electric field)")
    resolved_num_fidelity_levels = int(resolved_arch.get("num_fidelity_levels") or 0)
    num_fidelity_levels = int(args.num_fidelity_levels) if args.num_fidelity_levels is not None else resolved_num_fidelity_levels
    resolved_multi_fidelity_mode = str(resolved_arch.get("multi_fidelity_mode") or "conditioning")
    multi_fidelity_mode = str(args.multi_fidelity_mode or resolved_multi_fidelity_mode).strip().lower()
    fidelity_supported_modes = {
        "spherical-save-cue",
        "pure-cartesian-ictd",
        "pure-cartesian-ictd-o3",
        "pure-cartesian-sparse",
        "pure-cartesian-sparse-save",
    }
    if num_fidelity_levels < 0:
        raise ValueError("--num-fidelity-levels must be >= 0")
    if num_fidelity_levels > 0 and args.tensor_product_mode not in fidelity_supported_modes:
        raise ValueError(
            "--num-fidelity-levels is currently supported only for --tensor-product-mode "
            "spherical-save-cue, pure-cartesian-ictd, pure-cartesian-ictd-o3, pure-cartesian-sparse, or pure-cartesian-sparse-save"
        )
    if multi_fidelity_mode not in {"conditioning", "delta-baseline"}:
        raise ValueError("--multi-fidelity-mode must be one of conditioning|delta-baseline")
    if args.external_tensor_rank and not args.external_field_file and not args.external_field_value and not args.magnetic_field_file and not args.magnetic_field_value:
        logging.warning(
            "--external-tensor-rank is set but --external-field-file/--external-field-value is not. "
            "The model will include external field architecture. If 'external_field' "
            "is not embedded in the H5 dataset, the field will be zero at runtime."
        )
    phys_supported_modes = {"pure-cartesian-ictd", "pure-cartesian-ictd-o3", "pure-cartesian-sparse", "pure-cartesian-sparse-save"}
    has_external_tensor_request = bool(
        args.external_tensor_rank
        or args.external_tensor_irrep
        or args.external_tensor_parity
        or args.external_field_file
        or args.external_field_value
        or args.magnetic_field_file
        or args.magnetic_field_value
        or restored_external_tensor_specs
    )
    if has_external_tensor_request and args.tensor_product_mode not in phys_supported_modes:
        raise ValueError(
            "External tensor embedding is only supported for --tensor-product-mode "
            "pure-cartesian-ictd, pure-cartesian-ictd-o3, pure-cartesian-sparse, or pure-cartesian-sparse-save"
        )
    if (args.physical_tensors or args.physical_tensors_per_node) and args.tensor_product_mode not in phys_supported_modes:
        raise ValueError(
            "--physical-tensors and --physical-tensors-per-node only supported for --tensor-product-mode "
            "pure-cartesian-ictd, pure-cartesian-ictd-o3, pure-cartesian-sparse, or pure-cartesian-sparse-save"
        )
    if args.physical_tensors_per_node and not args.extra_per_node_file:
        raise ValueError("--physical-tensors-per-node requires --extra-per-node-file")

    external_tensor_specs = normalize_external_tensor_specs(restored_external_tensor_specs) if restored_external_tensor_specs else None
    if has_external_tensor_request:
        specs_by_name = {
            str(spec["name"]): dict(spec)
            for spec in (external_tensor_specs or [])
        }
        if args.external_tensor_rank is not None:
            electric_specs = build_standard_external_tensor_specs(
                external_tensor_rank=args.external_tensor_rank,
                external_tensor_irrep=args.external_tensor_irrep,
                external_tensor_parity=args.external_tensor_parity,
                include_magnetic_field=False,
            ) or []
            for spec in electric_specs:
                specs_by_name[str(spec["name"])] = dict(spec)
        elif args.external_field_file or args.external_field_value:
            raise ValueError("Electric-field data was provided but --external-tensor-rank is not set")
        if args.magnetic_field_file or args.magnetic_field_value:
            for spec in (
                normalize_external_tensor_specs(
                    None,
                    external_tensor_rank=1,
                    external_tensor_irrep="1e",
                    default_name="magnetic_field",
                )
                or []
            ):
                specs_by_name[str(spec["name"])] = dict(spec)
        external_tensor_specs = list(specs_by_name.values()) or None

    # Parse physical_tensor_weights: "charge:1.0,dipole:2.0,..." -> {"charge": 1.0, "dipole": 2.0, ...}
    physical_tensor_weights = {}
    if args.physical_tensor_weights:
        for part in args.physical_tensor_weights.split(","):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                k = k.strip()
                try:
                    physical_tensor_weights[k] = float(v.strip())
                except ValueError:
                    raise ValueError(f"Invalid --physical-tensor-weights part {part!r}; expected name:weight")
            else:
                raise ValueError(f"Invalid --physical-tensor-weights part {part!r}; expected name:weight")
    fidelity_loss_weights = parse_fidelity_loss_weights(args.fidelity_loss_weights)
    if fidelity_loss_weights and num_fidelity_levels <= 0:
        raise ValueError("--fidelity-loss-weights requires --num-fidelity-levels > 0")
    if fidelity_loss_weights:
        max_fid = max(fidelity_loss_weights)
        if max_fid >= num_fidelity_levels:
            raise ValueError(
                f"--fidelity-loss-weights references fidelity id {max_fid}, but --num-fidelity-levels={num_fidelity_levels}"
            )

    # --- Rank / world size (updated later if --distributed) ---
    rank = 0
    world_size = 1
    local_rank = 0

    # --- Loss weight bounds ---
    if args.a_min is not None and args.a_max is not None and args.a_min > args.a_max:
        raise ValueError("--a-min must be <= --a-max")
    if args.b_min is not None and args.b_max is not None and args.b_min > args.b_max:
        raise ValueError("--b-min must be <= --b-max")
    if args.num_interaction < 2:
        raise ValueError(f"--num-interaction must be >= 2, got {args.num_interaction}")
    if args.long_range_mode == "latent-coulomb":
        if args.long_range_boundary not in {"nonperiodic", "periodic"}:
            raise ValueError("--long-range-mode latent-coulomb requires --long-range-boundary nonperiodic or periodic")
        if args.long_range_backend == "tree_fmm" and args.long_range_boundary != "nonperiodic":
            raise ValueError("--long-range-backend tree_fmm currently requires --long-range-boundary nonperiodic")
        if args.long_range_backend == "tree_fmm" and args.long_range_multipole_order != 0:
            raise ValueError("--long-range-backend tree_fmm currently supports only --long-range-multipole-order 0")
    elif args.long_range_mode == "isolated-far-field-v1":
        if args.long_range_boundary != "nonperiodic":
            raise ValueError("--long-range-mode isolated-far-field-v1 currently requires --long-range-boundary nonperiodic")
        if args.long_range_energy_partition not in {"potential", "uniform"}:
            raise ValueError("--long-range-mode isolated-far-field-v1 requires --long-range-energy-partition potential or uniform")
        if args.long_range_backend != "dense_pairwise":
            raise ValueError("--long-range-backend is not used with --long-range-mode isolated-far-field-v1")
    elif args.long_range_mode == "isolated-far-field-v2":
        if args.long_range_boundary != "nonperiodic":
            raise ValueError("--long-range-mode isolated-far-field-v2 currently requires --long-range-boundary nonperiodic")
        if args.long_range_energy_partition not in {"potential", "uniform"}:
            raise ValueError("--long-range-mode isolated-far-field-v2 requires --long-range-energy-partition potential or uniform")
        if args.long_range_backend != "dense_pairwise":
            raise ValueError("--long-range-backend is not used with --long-range-mode isolated-far-field-v2")
        if args.long_range_far_source_dim < 1:
            raise ValueError("--long-range-far-source-dim must be >= 1")
        if args.long_range_far_num_shells < 1:
            raise ValueError("--long-range-far-num-shells must be >= 1")
        if args.long_range_far_shell_growth <= 1.0:
            raise ValueError("--long-range-far-shell-growth must be > 1")
        if args.long_range_far_tail_bins < 1:
            raise ValueError("--long-range-far-tail-bins must be >= 1")
        if args.long_range_far_stats != "mean,count,mean_r,rms_r":
            raise ValueError("--long-range-far-stats currently supports only mean,count,mean_r,rms_r")
        if args.long_range_far_max_radius_multiplier <= 1.0:
            raise ValueError("--long-range-far-max-radius-multiplier must be > 1")
        derived_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
            args.long_range_far_num_shells,
            args.long_range_far_shell_growth,
        )
        if user_far_max_radius_multiplier is not None and not math.isclose(
            user_far_max_radius_multiplier,
            derived_far_max_radius_multiplier,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "--long-range-far-max-radius-multiplier is derived internally as "
                "--long-range-far-shell-growth ** --long-range-far-num-shells; "
                "please omit it or pass the derived value exactly"
            )
    elif args.long_range_backend != "dense_pairwise":
        raise ValueError("--long-range-backend is currently only used with --long-range-mode latent-coulomb")
    if args.long_range_mode == "reciprocal-spectral-v1":
        if args.long_range_reciprocal_backend == "direct_kspace" and args.long_range_boundary != "periodic":
            raise ValueError(
                "--long-range-mode reciprocal-spectral-v1 with --long-range-reciprocal-backend direct_kspace "
                "currently requires --long-range-boundary periodic"
            )
        if args.long_range_reciprocal_backend == "mesh_fft" and args.long_range_boundary not in {"periodic", "slab"}:
            raise ValueError(
                "--long-range-mode reciprocal-spectral-v1 with --long-range-reciprocal-backend mesh_fft "
                "requires --long-range-boundary periodic or slab"
            )
        if args.long_range_mesh_fft_full_ewald and args.long_range_reciprocal_backend != "mesh_fft":
            raise ValueError(
                "--long-range-mesh-fft-full-ewald requires --long-range-reciprocal-backend mesh_fft"
            )
    
    # --- SWA (Stochastic Weight Averaging) ---
    if args.swa_start_epoch is not None:
        if args.swa_a is None or args.swa_b is None:
            raise ValueError("--swa-a and --swa-b must be set when --swa-start-epoch is set")
        if args.swa_start_epoch < 1:
            raise ValueError("--swa-start-epoch must be >= 1")
        if rank == 0:
            logging.info(f"SWA enabled: Will switch to a={args.swa_a}, b={args.swa_b} at epoch {args.swa_start_epoch}")

    # --- EMA (Exponential Moving Average) ---
    if args.ema_start_epoch is not None:
        if args.ema_start_epoch < 1:
            raise ValueError("--ema-start-epoch must be >= 1")
        if not (0.0 < args.ema_decay < 1.0):
            raise ValueError("--ema-decay must be in (0, 1)")
        args.use_ema_for_validation = True
        args.save_ema_model = True
        if rank == 0:
            logging.info(f"EMA enabled: Will start at epoch {args.ema_start_epoch} with decay={args.ema_decay}")
            logging.info("  Using EMA model for validation")
            logging.info("  Will save EMA model in checkpoint")
    
    # --- Random seed ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32: faster float32 matmul on Ampere+ (removes inductor warning, ~2x matmul speed)
        torch.set_float32_matmul_precision(args.matmul_precision)
    
    # --- Distributed training (DDP) ---
    if args.distributed:
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        if args.local_rank == -1:
            raise ValueError("--local-rank must be set when --distributed is enabled. "
                           "Use 'torchrun' or set LOCAL_RANK environment variable.")
        
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            timeout=timedelta(hours=2)
        )
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = args.local_rank
        
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if local_rank >= n:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {n} GPU(s) available. "
                    f"Use fewer processes (e.g. torchrun --nproc_per_node={n}) or set CUDA_VISIBLE_DEVICES."
                )
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        
        if rank == 0:
            setup_logging()
            logging.info(f"Distributed training enabled: {world_size} GPUs")
            logging.info(f"Using device: {device} (rank {rank}, local_rank {local_rank})")
            logging.info(f"Data directory: {args.data_dir}")
        else:
            logging.basicConfig(level=logging.WARNING)
    else:
        setup_logging()

    # Log scatter backend (helps diagnose performance regressions when torch_scatter is broken).
    try:
        from molecular_force_field.utils.scatter import scatter_backend, require_torch_scatter

        if not args.distributed or rank == 0:
            logging.info("Scatter backend: %s", scatter_backend())

        # For cuEquivariance backend, torch_scatter is strongly recommended for speed.
        if args.tensor_product_mode == "spherical-save-cue":
            require_torch_scatter(reason="tensor_product_mode='spherical-save-cue' aims for maximum speed.")
    except Exception:
        pass

    # --- Default dtype ---
    if args.dtype == 'float64' or args.dtype == 'double':
        torch.set_default_dtype(torch.float64)
        if rank == 0:
            logging.info("Using dtype: float64")
    elif args.dtype == 'float32' or args.dtype == 'float':
        torch.set_default_dtype(torch.float32)
        if rank == 0:
            logging.info("Using dtype: float32")
    
    # --- Device (non-distributed) ---
    if not args.distributed:
        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
            # Validate CUDA device ordinal to avoid "invalid device ordinal"
            if device.type == "cuda":
                n = torch.cuda.device_count()
                idx = device.index if device.index is not None else 0
                if n == 0:
                    logging.warning("CUDA requested but no GPU available; falling back to CPU")
                    device = torch.device("cpu")
                elif idx >= n:
                    logging.warning(f"CUDA device {idx} does not exist (only {n} GPU(s)); using cuda:0")
                    device = torch.device("cuda:0")
        
        logging.info(f"Using device: {device}")
        logging.info(f"Data directory: {args.data_dir}")
    
    # --- Data: resolve source (custom H5 paths vs data_dir + prefix) and preprocess if needed ---
    use_custom_data_paths = (args.train_data is not None and args.valid_data is not None)
    if use_custom_data_paths:
        if not os.path.exists(args.train_data):
            logging.error(f"Training data file not found: {args.train_data}")
            return
        if not os.path.exists(args.valid_data):
            logging.error(f"Validation data file not found: {args.valid_data}")
            return
        if rank == 0:
            logging.info(f"Using custom datasets: train={args.train_data}, valid={args.valid_data}")
        if args.distributed:
            dist.barrier()
    elif args.distributed:
        train_processed = os.path.join(args.data_dir, f'processed_{args.train_prefix}.h5')
        val_processed = os.path.join(args.data_dir, f'processed_{args.val_prefix}.h5')
        
        if not (os.path.exists(train_processed) and os.path.exists(val_processed)):
            if rank == 0:
                logging.error(
                    f"Preprocessed data not found in {args.data_dir}/\n"
                    f"In distributed mode, you must preprocess data first:\n"
                    f"  mff-preprocess --input-file {args.input_file or '<your_xyz_file>'} --output-dir {args.data_dir}\n"
                    f"Or run single-GPU training first (without --distributed) to auto-preprocess."
                )
            dist.destroy_process_group()
            return
        
        if rank == 0:
            logging.info(f"Found preprocessed data in {args.data_dir}/")
        dist.barrier()
    else:
        data_ready = check_and_preprocess_data(
            args.data_dir,
            args.train_prefix,
            args.val_prefix,
            args.max_radius,
            args.num_workers,
            input_file=args.input_file,
            train_input_file=args.train_input_file,
            valid_input_file=args.valid_input_file,
            seed=args.seed
        )
        if not data_ready:
            logging.error("Data preparation failed. Exiting.")
            return

    # --- Inject uniform external field / magnetic field into H5 ---
    if args.external_field_value or args.magnetic_field_value:
        from molecular_force_field.active_learning.data_merge import _inject_external_field_into_h5
        if use_custom_data_paths:
            h5_paths = [args.train_data, args.valid_data]
        else:
            h5_paths = [
                os.path.join(args.data_dir, f"processed_{args.train_prefix}.h5"),
                os.path.join(args.data_dir, f"processed_{args.val_prefix}.h5"),
            ]
        for hp in h5_paths:
            if os.path.exists(hp):
                if args.external_field_value:
                    _inject_external_field_into_h5(hp, args.external_field_value, dataset_name="external_field")
                if args.magnetic_field_value:
                    _inject_external_field_into_h5(hp, args.magnetic_field_value, dataset_name="magnetic_field")

    # --- Build extra_label_paths and extra_per_node for H5Dataset ---
    extra_label_paths = {}
    if args.charge_file:
        extra_label_paths["charge"] = args.charge_file
    if args.dipole_file:
        extra_label_paths["dipole"] = args.dipole_file
    if args.polarizability_file:
        extra_label_paths["polarizability"] = args.polarizability_file
    if args.quadrupole_file:
        extra_label_paths["quadrupole"] = args.quadrupole_file
    if args.external_field_file:
        extra_label_paths["external_field"] = args.external_field_file
    if args.magnetic_field_file:
        extra_label_paths["magnetic_field"] = args.magnetic_field_file
    if args.fidelity_id_file:
        extra_label_paths["fidelity_id"] = args.fidelity_id_file
    extra_label_paths = extra_label_paths if extra_label_paths else None

    # --- Build datasets (from custom paths or data_dir + prefix) ---
    if use_custom_data_paths:
        train_dataset = H5Dataset(
            'train', data_dir=args.data_dir, file_path=args.train_data,
            extra_label_paths=extra_label_paths,
            extra_per_node_label_path=args.extra_per_node_file,
        )
        val_dataset = H5Dataset(
            'val', data_dir=args.data_dir, file_path=args.valid_data,
            extra_label_paths=extra_label_paths,
            extra_per_node_label_path=args.extra_per_node_file,
        )
    else:
        train_dataset = H5Dataset(
            args.train_prefix, data_dir=args.data_dir,
            extra_label_paths=extra_label_paths,
            extra_per_node_label_path=args.extra_per_node_file,
        )
        val_dataset = H5Dataset(
            args.val_prefix, data_dir=args.data_dir,
            extra_label_paths=extra_label_paths,
            extra_per_node_label_path=args.extra_per_node_file,
        )

    # --- DataLoaders and distributed samplers ---
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_num_workers = max(1, args.num_workers // 2)
    val_num_workers = max(1, args.num_workers // 4)
    if args.mp_context == "spawn":
        mp_ctx = "spawn"
    elif args.mp_context == "fork":
        mp_ctx = "fork"
    else:
        # auto: only force spawn when validation compile is enabled on CUDA to avoid fork deadlocks
        mp_ctx = "spawn" if (args.compile_val != "none" and torch.cuda.is_available() and train_num_workers > 0) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn_h5,
        num_workers=train_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        multiprocessing_context=mp_ctx,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn_h5,
        num_workers=val_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        multiprocessing_context=mp_ctx,
    )
    
    # --- Model config ---
    config_dtype = torch.float64 if args.dtype in ['float64', 'double'] else torch.float32
    config = ModelConfig(
        dtype=config_dtype,
        max_atomvalue=args.max_atomvalue,
        embedding_dim=args.embedding_dim,
        embed_size=args.embed_size,
        output_size=args.output_size,
        lmax=args.lmax,
        irreps_output_conv_channels=args.irreps_output_conv_channels,
        function_type=args.function_type,
        max_radius=args.max_radius
    )
    checkpoint_atomic_energies = get_checkpoint_atomic_energies(checkpoint_arch, dtype=config.dtype)
    
    # --- Atomic reference energies (E0) ---
    if args.atomic_energy_keys is not None or args.atomic_energy_values is not None:
        if args.atomic_energy_keys is None or args.atomic_energy_values is None:
            raise ValueError("Both --atomic-energy-keys and --atomic-energy-values must be provided together.")
        if len(args.atomic_energy_keys) != len(args.atomic_energy_values):
            raise ValueError("--atomic-energy-keys and --atomic-energy-values must have the same length.")
        config.atomic_energy_keys = torch.tensor(args.atomic_energy_keys, dtype=torch.long)
        config.atomic_energy_values = torch.tensor(args.atomic_energy_values, dtype=config.dtype)
        if rank == 0:
            logging.info("Using custom atomic reference energies from CLI:")
            for k, v in zip(args.atomic_energy_keys, args.atomic_energy_values):
                logging.info(f"  Atom {k}: {v:.8f} eV")
    else:
        if args.atomic_energy_file is not None:
            config.load_atomic_energies_from_file(args.atomic_energy_file)
            if rank == 0:
                logging.info(f"Loaded atomic reference energies from file: {args.atomic_energy_file}")
        elif checkpoint_atomic_energies is not None:
            config.atomic_energy_keys, config.atomic_energy_values = checkpoint_atomic_energies
            if rank == 0:
                logging.info("Loaded atomic reference energies from checkpoint.")
        else:
            # Default behavior: load least-squares fitted E0 from fitted_E0.csv
            e0_path = os.path.join(args.data_dir, 'fitted_E0.csv')
            config.load_atomic_energies_from_file(e0_path)
    
    # --- Log hyperparameters ---
    if rank == 0:
        logging.info("=" * 80)
        logging.info("Model Hyperparameters:")
        logging.info(f"  max_atomvalue: {config.max_atomvalue}")
        logging.info(f"  embedding_dim: {config.embedding_dim}")
        logging.info(f"  embed_size: {config.embed_size}")
        logging.info(f"  output_size: {config.output_size}")
        logging.info(f"  lmax: {config.lmax}")
        logging.info(f"  irreps_output_conv: {config.get_irreps_output_conv()}")
        logging.info(f"  function_type: {config.function_type}")
        logging.info(f"  max_radius: {config.max_radius}")
    logging.info(f"  dtype: {config.dtype}")
    logging.info("=" * 80)

    common_long_range_kwargs = dict(
        long_range_mode=args.long_range_mode,
        long_range_hidden_dim=args.long_range_hidden_dim,
        long_range_boundary=args.long_range_boundary,
        long_range_neutralize=args.long_range_neutralize,
        long_range_filter_hidden_dim=args.long_range_filter_hidden_dim,
        long_range_kmax=args.long_range_kmax,
        long_range_mesh_size=args.long_range_mesh_size,
        long_range_slab_padding_factor=args.long_range_slab_padding_factor,
        long_range_include_k0=args.long_range_include_k0,
        long_range_source_channels=args.long_range_source_channels,
        long_range_backend=args.long_range_backend,
        long_range_reciprocal_backend=args.long_range_reciprocal_backend,
        long_range_energy_partition=args.long_range_energy_partition,
        long_range_green_mode=args.long_range_green_mode,
        long_range_assignment=args.long_range_assignment,
        long_range_mesh_fft_full_ewald=args.long_range_mesh_fft_full_ewald,
        long_range_theta=args.long_range_theta,
        long_range_leaf_size=args.long_range_leaf_size,
        long_range_multipole_order=args.long_range_multipole_order,
        long_range_far_source_dim=args.long_range_far_source_dim,
        long_range_far_num_shells=args.long_range_far_num_shells,
        long_range_far_shell_growth=args.long_range_far_shell_growth,
        long_range_far_tail=args.long_range_far_tail,
        long_range_far_tail_bins=args.long_range_far_tail_bins,
        long_range_far_stats=args.long_range_far_stats,
        long_range_far_max_radius_multiplier=args.long_range_far_max_radius_multiplier,
        long_range_far_source_norm=args.long_range_far_source_norm,
        long_range_far_gate_init=args.long_range_far_gate_init,
        feature_spectral_mode=args.feature_spectral_mode,
        feature_spectral_bottleneck_dim=args.feature_spectral_bottleneck_dim,
        feature_spectral_mesh_size=args.feature_spectral_mesh_size,
        feature_spectral_filter_hidden_dim=args.feature_spectral_filter_hidden_dim,
        feature_spectral_boundary=args.feature_spectral_boundary,
        feature_spectral_slab_padding_factor=args.feature_spectral_slab_padding_factor,
        feature_spectral_neutralize=args.feature_spectral_neutralize,
        feature_spectral_include_k0=args.feature_spectral_include_k0,
        feature_spectral_assignment=args.feature_spectral_assignment,
        feature_spectral_gate_init=args.feature_spectral_gate_init,
    )
    common_invariant_kwargs = dict(invariant_channels=args.invariant_channels)
    
    # Initialize models
    model = MainNet(
        input_size=config.input_dim_weight,
        hidden_sizes=config.main_hidden_sizes4,
        output_size=1
    ).to(device)
    
    # Initialize model based on tensor product mode
    if args.tensor_product_mode == 'pure-cartesian':
        logging.info("Using PURE Cartesian mode (rank tensors 3^L with delta/epsilon contractions), num_interaction=%d", args.num_interaction)
        e3trans = PureCartesianTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
            **common_invariant_kwargs,
            **common_long_range_kwargs,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-ictd':
        logging.info("Using PURE Cartesian ICTD mode (pure_cartesian_ictd_layers_full, DDP sync), num_interaction=%d", args.num_interaction)
        logging.info(f"  ictd_tp_path_policy={args.ictd_tp_path_policy}, ictd_tp_max_rank_other={args.ictd_tp_max_rank_other}")
        if args.long_range_mode != "none":
            logged_long_range_backend = (
                args.long_range_backend
                if args.long_range_mode == "latent-coulomb"
                else args.long_range_mode
                if args.long_range_mode in {"isolated-far-field-v1", "isolated-far-field-v2"}
                else args.long_range_reciprocal_backend
            )
            logging.info(
                "  long_range: mode=%s hidden_dim=%d boundary=%s neutralize=%s "
                "filter_hidden_dim=%d kmax=%d source_channels=%d backend=%s partition=%s include_k0=%s",
                args.long_range_mode,
                args.long_range_hidden_dim,
                args.long_range_boundary,
                args.long_range_neutralize,
                args.long_range_filter_hidden_dim,
                args.long_range_kmax,
                args.long_range_source_channels,
                logged_long_range_backend,
                args.long_range_energy_partition,
                args.long_range_include_k0,
            )

        # Build physical_tensor_outputs from CLI, or restore them from checkpoint when resuming.
        _phys_specs = {"charge": [0], "dipole": [1], "magnetic_moment": [1], "polarizability": [0, 2], "quadrupole": [2], "born_effective_charge": [0, 1, 2]}
        _phys_irreps = {
            "charge": ["0e"],
            "dipole": ["1o"],
            "magnetic_moment": ["1e"],
            "polarizability": ["0e", "2e"],
            "quadrupole": ["2e"],
            "born_effective_charge": ["0e", "1e", "2e"],
        }
        physical_tensor_outputs = {}
        if args.physical_tensors:
            for name in (s.strip() for s in args.physical_tensors.split(",") if s.strip()):
                if name == "born_effective_charge":
                    raise ValueError(
                        "born_effective_charge is currently supported only as per-node "
                        "born_effective_charge_per_atom via --physical-tensors-per-node"
                    )
                if name in _phys_specs:
                    physical_tensor_outputs[name] = {
                        "ls": _phys_specs[name],
                        "irreps": _phys_irreps.get(name),
                        "channels_out": 1,
                        "reduce": args.physical_tensor_reduce,
                    }
                else:
                    raise ValueError(f"Unknown --physical-tensors name {name!r}; supported: charge, dipole, magnetic_moment, polarizability, quadrupole")
        if args.physical_tensors_per_node:
            for name in (s.strip() for s in args.physical_tensors_per_node.split(",") if s.strip()):
                base = name.replace("_per_atom", "")
                if base in _phys_specs:
                    physical_tensor_outputs[name] = {
                        "ls": _phys_specs[base],
                        "irreps": _phys_irreps.get(base),
                        "channels_out": 1,
                        "reduce": "none",
                    }
                else:
                    raise ValueError(
                        f"Unknown --physical-tensors-per-node {name!r}; supported: charge_per_atom, dipole_per_atom, "
                        "magnetic_moment_per_atom, polarizability_per_atom, quadrupole_per_atom, born_effective_charge_per_atom"
                    )
        physical_tensor_outputs = physical_tensor_outputs if physical_tensor_outputs else restored_physical_tensor_outputs

        if physical_tensor_outputs and rank == 0:
            logging.info("  physical_tensor_outputs: %s", list(physical_tensor_outputs.keys()))
        if external_tensor_specs is not None and rank == 0:
            logging.info("  external_tensor_specs=%s", external_tensor_specs)

        e3trans = PureCartesianICTDTransformerLayerFull(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=args.ictd_tp_path_policy,
            ictd_tp_max_rank_other=args.ictd_tp_max_rank_other,
            internal_compute_dtype=config.internal_compute_dtype,
            device=device,
            **common_invariant_kwargs,
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=args.external_tensor_rank,
            external_tensor_irrep=args.external_tensor_irrep,
            external_tensor_parity=args.external_tensor_parity,
            external_tensor_specs=external_tensor_specs,
            num_fidelity_levels=num_fidelity_levels,
            multi_fidelity_mode=multi_fidelity_mode,
            long_range_mode=args.long_range_mode,
            long_range_hidden_dim=args.long_range_hidden_dim,
            long_range_boundary=args.long_range_boundary,
            long_range_neutralize=args.long_range_neutralize,
            long_range_filter_hidden_dim=args.long_range_filter_hidden_dim,
            long_range_kmax=args.long_range_kmax,
            long_range_mesh_size=args.long_range_mesh_size,
            long_range_slab_padding_factor=args.long_range_slab_padding_factor,
            long_range_include_k0=args.long_range_include_k0,
            long_range_source_channels=args.long_range_source_channels,
            long_range_backend=args.long_range_backend,
            long_range_reciprocal_backend=args.long_range_reciprocal_backend,
            long_range_energy_partition=args.long_range_energy_partition,
            long_range_green_mode=args.long_range_green_mode,
            long_range_assignment=args.long_range_assignment,
            long_range_mesh_fft_full_ewald=args.long_range_mesh_fft_full_ewald,
            long_range_theta=args.long_range_theta,
            long_range_leaf_size=args.long_range_leaf_size,
            long_range_multipole_order=args.long_range_multipole_order,
            long_range_far_source_dim=args.long_range_far_source_dim,
            long_range_far_num_shells=args.long_range_far_num_shells,
            long_range_far_shell_growth=args.long_range_far_shell_growth,
            long_range_far_tail=args.long_range_far_tail,
            long_range_far_tail_bins=args.long_range_far_tail_bins,
            long_range_far_stats=args.long_range_far_stats,
            long_range_far_max_radius_multiplier=args.long_range_far_max_radius_multiplier,
            long_range_far_source_norm=args.long_range_far_source_norm,
            long_range_far_gate_init=args.long_range_far_gate_init,
            feature_spectral_mode=args.feature_spectral_mode,
            feature_spectral_bottleneck_dim=args.feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=args.feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=args.feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=args.feature_spectral_boundary,
            feature_spectral_slab_padding_factor=args.feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=args.feature_spectral_neutralize,
            feature_spectral_include_k0=args.feature_spectral_include_k0,
            feature_spectral_gate_init=args.feature_spectral_gate_init,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-ictd-o3':
        logging.info("Using PURE Cartesian ICTD O(3) mode (strict parity-aware ICTD), num_interaction=%d", args.num_interaction)
        logging.info(f"  ictd_tp_path_policy={args.ictd_tp_path_policy}, ictd_tp_max_rank_other={args.ictd_tp_max_rank_other}")
        _phys_specs = {"charge": [0], "dipole": [1], "magnetic_moment": [1], "polarizability": [0, 2], "quadrupole": [2], "born_effective_charge": [0, 1, 2]}
        _phys_irreps = {
            "charge": ["0e"],
            "dipole": ["1o"],
            "magnetic_moment": ["1e"],
            "polarizability": ["0e", "2e"],
            "quadrupole": ["2e"],
            "born_effective_charge": ["0e", "1e", "2e"],
        }
        physical_tensor_outputs = {}
        if args.physical_tensors:
            for name in (s.strip() for s in args.physical_tensors.split(",") if s.strip()):
                if name == "born_effective_charge":
                    raise ValueError("born_effective_charge is currently supported only as per-node born_effective_charge_per_atom via --physical-tensors-per-node")
                if name in _phys_specs:
                    physical_tensor_outputs[name] = {"ls": _phys_specs[name], "irreps": _phys_irreps.get(name), "channels_out": 1, "reduce": args.physical_tensor_reduce}
                else:
                    raise ValueError(f"Unknown --physical-tensors name {name!r}; supported: charge, dipole, magnetic_moment, polarizability, quadrupole")
        if args.physical_tensors_per_node:
            for name in (s.strip() for s in args.physical_tensors_per_node.split(",") if s.strip()):
                base = name.replace("_per_atom", "")
                if base in _phys_specs:
                    physical_tensor_outputs[name] = {"ls": _phys_specs[base], "irreps": _phys_irreps.get(base), "channels_out": 1, "reduce": "none"}
                else:
                    raise ValueError(
                        f"Unknown --physical-tensors-per-node {name!r}; supported: charge_per_atom, dipole_per_atom, "
                        "magnetic_moment_per_atom, polarizability_per_atom, quadrupole_per_atom, born_effective_charge_per_atom"
                    )
        physical_tensor_outputs = physical_tensor_outputs if physical_tensor_outputs else restored_physical_tensor_outputs
        e3trans = PureCartesianICTDO3TransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=args.ictd_tp_path_policy,
            ictd_tp_max_rank_other=args.ictd_tp_max_rank_other,
            internal_compute_dtype=config.internal_compute_dtype,
            device=device,
            **common_invariant_kwargs,
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=args.external_tensor_rank,
            external_tensor_irrep=args.external_tensor_irrep,
            external_tensor_parity=args.external_tensor_parity,
            external_tensor_specs=external_tensor_specs,
            o3_irrep_preset=args.o3_irrep_preset,
            o3_active_irreps=args.o3_active_irreps,
            num_fidelity_levels=num_fidelity_levels,
            multi_fidelity_mode=multi_fidelity_mode,
            long_range_mode=args.long_range_mode,
            long_range_hidden_dim=args.long_range_hidden_dim,
            long_range_boundary=args.long_range_boundary,
            long_range_neutralize=args.long_range_neutralize,
            long_range_filter_hidden_dim=args.long_range_filter_hidden_dim,
            long_range_kmax=args.long_range_kmax,
            long_range_mesh_size=args.long_range_mesh_size,
            long_range_slab_padding_factor=args.long_range_slab_padding_factor,
            long_range_include_k0=args.long_range_include_k0,
            long_range_source_channels=args.long_range_source_channels,
            long_range_backend=args.long_range_backend,
            long_range_reciprocal_backend=args.long_range_reciprocal_backend,
            long_range_energy_partition=args.long_range_energy_partition,
            long_range_green_mode=args.long_range_green_mode,
            long_range_assignment=args.long_range_assignment,
            long_range_mesh_fft_full_ewald=args.long_range_mesh_fft_full_ewald,
            long_range_theta=args.long_range_theta,
            long_range_leaf_size=args.long_range_leaf_size,
            long_range_multipole_order=args.long_range_multipole_order,
            long_range_far_source_dim=args.long_range_far_source_dim,
            long_range_far_num_shells=args.long_range_far_num_shells,
            long_range_far_shell_growth=args.long_range_far_shell_growth,
            long_range_far_tail=args.long_range_far_tail,
            long_range_far_tail_bins=args.long_range_far_tail_bins,
            long_range_far_stats=args.long_range_far_stats,
            long_range_far_max_radius_multiplier=args.long_range_far_max_radius_multiplier,
            long_range_far_source_norm=args.long_range_far_source_norm,
            long_range_far_gate_init=args.long_range_far_gate_init,
            feature_spectral_mode=args.feature_spectral_mode,
            feature_spectral_bottleneck_dim=args.feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=args.feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=args.feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=args.feature_spectral_boundary,
            feature_spectral_slab_padding_factor=args.feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=args.feature_spectral_neutralize,
            feature_spectral_include_k0=args.feature_spectral_include_k0,
            feature_spectral_gate_init=args.feature_spectral_gate_init,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-ictd-save':
        logging.info("Using PURE Cartesian ICTD mode (pure_cartesian_ictd_layers, save/original), num_interaction=%d", args.num_interaction)
        logging.info(f"  ictd_tp_path_policy={args.ictd_tp_path_policy}, ictd_tp_max_rank_other={args.ictd_tp_max_rank_other}")
        e3trans = PureCartesianICTDTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=args.ictd_tp_path_policy,
            ictd_tp_max_rank_other=args.ictd_tp_max_rank_other,
            internal_compute_dtype=config.internal_compute_dtype,
            device=device,
            **common_invariant_kwargs,
            **common_long_range_kwargs,
        ).to(device)
    elif args.tensor_product_mode in {'pure-cartesian-sparse', 'pure-cartesian-sparse-save'}:
        logging.info(
            "Using PURE Cartesian SPARSE%s mode (δ/ε path-sparse within 3^L, O(3) strict)",
            " SAVE" if args.tensor_product_mode == "pure-cartesian-sparse-save" else "",
        )
        logging.info(f"  max_rank_other={args.max_rank_other}, k_policy={args.k_policy}, num_interaction={args.num_interaction}")
        physical_tensor_outputs = restored_physical_tensor_outputs or {}
        sparse_cls = (
            PureCartesianSparseTransformerLayerSave
            if args.tensor_product_mode == "pure-cartesian-sparse-save"
            else PureCartesianSparseTransformerLayer
        )
        e3trans = sparse_cls(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            max_rank_other=args.max_rank_other,
            k_policy=args.k_policy,
            physical_tensor_outputs=physical_tensor_outputs,
            external_tensor_rank=args.external_tensor_rank,
            external_tensor_specs=external_tensor_specs,
            num_fidelity_levels=num_fidelity_levels,
            device=device,
            **common_invariant_kwargs,
            **common_long_range_kwargs,
        ).to(device)
    elif args.tensor_product_mode == 'partial-cartesian':
        logging.info("Using Partial-Cartesian tensor product mode (strict), num_interaction=%d", args.num_interaction)
        e3trans = CartesianTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
            **common_invariant_kwargs,
            **common_long_range_kwargs,
        ).to(device)
    elif args.tensor_product_mode == 'partial-cartesian-loose':
        logging.info("Using Partial-Cartesian LOOSE mode (non-strictly-equivariant, norm product approximation), num_interaction=%d", args.num_interaction)
        e3trans = CartesianTransformerLayerLoose(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
            **common_invariant_kwargs,
            **common_long_range_kwargs,
        ).to(device)
    elif args.tensor_product_mode == 'spherical-save':
        logging.info("Using Spherical (channelwise conv) tensor product mode (e3nn_layers_channelwise), num_interaction=%d", args.num_interaction)
        e3trans = E3_TransformerLayer_multi_channelwise(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            device=device,
            **common_invariant_kwargs,
            **common_long_range_kwargs,
        ).to(device)
    elif args.tensor_product_mode == 'spherical-save-cue':
        logging.info("Using Spherical (channelwise conv) tensor product mode (cuEquivariance backend), num_interaction=%d", args.num_interaction)
        if args.long_range_mode != "none":
            logged_long_range_backend = (
                args.long_range_backend
                if args.long_range_mode == "latent-coulomb"
                else args.long_range_mode
                if args.long_range_mode in {"isolated-far-field-v1", "isolated-far-field-v2"}
                else args.long_range_reciprocal_backend
            )
            logging.info(
                "  long_range: mode=%s hidden_dim=%d boundary=%s neutralize=%s "
                "filter_hidden_dim=%d kmax=%d source_channels=%d backend=%s partition=%s include_k0=%s",
                args.long_range_mode,
                args.long_range_hidden_dim,
                args.long_range_boundary,
                args.long_range_neutralize,
                args.long_range_filter_hidden_dim,
                args.long_range_kmax,
                args.long_range_source_channels,
                logged_long_range_backend,
                args.long_range_energy_partition,
                args.long_range_include_k0,
            )
        # Detect optional dependency early with a clear message.
        try:
            import cuequivariance_torch  # noqa: F401
        except Exception as e:
            raise ImportError(
                "tensor_product_mode='spherical-save-cue' requires cuEquivariance.\n"
                "Install one of:\n"
                "  pip install -e \".[cue]\"\n"
                "  pip install -r requirements-cue.txt\n"
                "Notes: CUDA kernels package (cuequivariance-ops-torch-cu12) is Linux CUDA only.\n"
                f"Original import error: {e}"
            ) from e
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise_cue,
        )
        e3trans = E3_TransformerLayer_multi_channelwise_cue(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            device=device,
            long_range_mode=args.long_range_mode,
            long_range_hidden_dim=args.long_range_hidden_dim,
            long_range_boundary=args.long_range_boundary,
            long_range_neutralize=args.long_range_neutralize,
            long_range_filter_hidden_dim=args.long_range_filter_hidden_dim,
            long_range_kmax=args.long_range_kmax,
            long_range_mesh_size=args.long_range_mesh_size,
            long_range_slab_padding_factor=args.long_range_slab_padding_factor,
            long_range_include_k0=args.long_range_include_k0,
            long_range_source_channels=args.long_range_source_channels,
            long_range_backend=args.long_range_backend,
            long_range_reciprocal_backend=args.long_range_reciprocal_backend,
            long_range_energy_partition=args.long_range_energy_partition,
            long_range_green_mode=args.long_range_green_mode,
            long_range_assignment=args.long_range_assignment,
            long_range_mesh_fft_full_ewald=args.long_range_mesh_fft_full_ewald,
            long_range_theta=args.long_range_theta,
            long_range_leaf_size=args.long_range_leaf_size,
            long_range_multipole_order=args.long_range_multipole_order,
            long_range_far_source_dim=args.long_range_far_source_dim,
            long_range_far_num_shells=args.long_range_far_num_shells,
            long_range_far_shell_growth=args.long_range_far_shell_growth,
            long_range_far_tail=args.long_range_far_tail,
            long_range_far_tail_bins=args.long_range_far_tail_bins,
            long_range_far_stats=args.long_range_far_stats,
            long_range_far_max_radius_multiplier=args.long_range_far_max_radius_multiplier,
            long_range_far_source_norm=args.long_range_far_source_norm,
            long_range_far_gate_init=args.long_range_far_gate_init,
            feature_spectral_mode=args.feature_spectral_mode,
            feature_spectral_bottleneck_dim=args.feature_spectral_bottleneck_dim,
            feature_spectral_mesh_size=args.feature_spectral_mesh_size,
            feature_spectral_filter_hidden_dim=args.feature_spectral_filter_hidden_dim,
            feature_spectral_boundary=args.feature_spectral_boundary,
            feature_spectral_slab_padding_factor=args.feature_spectral_slab_padding_factor,
            feature_spectral_neutralize=args.feature_spectral_neutralize,
            feature_spectral_include_k0=args.feature_spectral_include_k0,
            feature_spectral_gate_init=args.feature_spectral_gate_init,
            num_fidelity_levels=num_fidelity_levels,
            multi_fidelity_mode=multi_fidelity_mode,
        ).to(device)
    else:  # spherical (default)
        logging.info("Using Spherical harmonics tensor product mode (e3nn), num_interaction=%d", args.num_interaction)
        e3trans = E3_TransformerLayer_multi(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            device=device,
            **common_invariant_kwargs,
            **common_long_range_kwargs,
        ).to(device)
    e3trans = maybe_wrap_model_with_zbl(e3trans, vars(args))

    # Initialize trainer
    trainer = Trainer(
        model=model,
        e3trans=e3trans,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        initial_learning_rate_for_weight=args.warmup_start_ratio,
        warmup_batches=args.warmup_batches,
        patience_opim=args.lr_decay_patience,
        gamma_value=args.lr_decay_factor,
        lr_scheduler_type=args.lr_scheduler,
        epoch_numbers=args.epochs,
        checkpoint_path=args.checkpoint,
        use_checkpoint_loss_weights=not args.reset_loss_weights,
        dump_frequency=args.dump_frequency,
        energy_log_frequency=args.energy_log_frequency,
        vhat_clamp_interval=args.vhat_clamp_interval,
        max_vhat_growth_factor=args.max_vhat_growth,
        max_norm_value=args.max_grad_norm,
        gradient_log_interval=args.grad_log_interval,
        a=args.energy_weight,
        b=args.force_weight,
        update_param=args.update_param,
        weight_a_growth=args.weight_a_growth,
        weight_b_decay=args.weight_b_decay,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        swa_start_epoch=args.swa_start_epoch,
        swa_a=args.swa_a,
        swa_b=args.swa_b,
        ema_start_epoch=args.ema_start_epoch,
        ema_decay=args.ema_decay,
        use_ema_for_validation=args.use_ema_for_validation,
        save_ema_model=args.save_ema_model,
        force_shift_value=args.force_shift_value,
        c=args.stress_weight,
        c_min=args.c_min,
        c_max=args.c_max,
        patience=args.patience,
        atomic_energy_keys=config.atomic_energy_keys,
        atomic_energy_values=config.atomic_energy_values,
        distributed=args.distributed,
        rank=rank,
        world_size=world_size,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        save_val_csv=args.save_val_csv,
        train_eval_sample_ratio=args.train_eval_sample_ratio,
        log_val_batch_energy_to_console=args.log_val_batch_energy,
        tensor_product_mode=args.tensor_product_mode,
        compile_val=args.compile_val,
        compile_val_mode=args.compile_val_mode,
        compile_val_fullgraph=args.compile_val_fullgraph,
        compile_val_dynamic=args.compile_val_dynamic,
        compile_val_precache=args.compile_val_precache,
        inference_output_physical_tensors=args.inference_output_physical_tensors,
        physical_tensor_weights=physical_tensor_weights,
        fidelity_loss_weights=fidelity_loss_weights,
        delta_regularization_weight=args.delta_regularization_weight,
        bec_derivative_weight=args.bec_derivative_weight,
        bec_consistency_weight=args.bec_consistency_weight,
    )
    
    # Start training
    if rank == 0:
        logging.info("Starting training...")
    trainer.run_training()
    if rank == 0:
        logging.info("Training completed!")
    
    # Cleanup distributed training
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
