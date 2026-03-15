# Usage Guide

FusedSCEquiTensorPot supports multiple equivariant tensor product implementation modes, and in `pure-cartesian-ictd`, `pure-cartesian-ictd-o3`, `pure-cartesian-sparse`, and `pure-cartesian-sparse-save` can **embed external fields** (electric field, magnetic field, and other rank-aware tensors) and **train physical tensors** (charge, dipole, polarizability, quadrupole, BEC, magnetic moment). Including:
- `spherical`: e3nn-based spherical harmonics method (default)
- `spherical-save`: channelwise edge conv (e3nn backend, fewer params)
- `spherical-save-cue`: channelwise edge conv (cuEquivariance backend, optional dependency, GPU accelerated)
- `partial-cartesian`: Cartesian coordinates + e3nn CG coefficients (strictly equivariant)
- `partial-cartesian-loose`: Approximate equivariant (norm product approximation)
- `pure-cartesian`: Pure Cartesian \(3^L\) representation (strictly equivariant, very slow, not recommended)
- `pure-cartesian-sparse`: Sparse pure Cartesian (strictly equivariant, parameter-optimized)
- `pure-cartesian-ictd`: ICTD irreps internal representation (strictly equivariant, fastest, fewest parameters)
- `pure-cartesian-ictd-o3`: Full parity-aware O(3) ICTD trunk that keeps `1e/1o` distinct through the whole network

All modes maintain O(3) equivariance (including rotation and reflection). For detailed performance comparison, see the [Tensor Product Mode Comparison](#tensor-product-mode-comparison) section.

## Installation

### 1. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install the package directly (dependencies will be installed automatically)
pip install -e .
```

### 2. Verify Installation

After installation, you can verify that the CLI is available using the following commands:

```bash
mff-preprocess --help
mff-train --help
mff-evaluate --help
mff-export-core --help   # LAMMPS LibTorch export
mff-lammps --help        # LAMMPS fix external interface
mff-init-data --help     # Initial dataset generation (cold start)
mff-active-learn --help  # Active learning workflow
python -m molecular_force_field.cli.export_mliap --help  # LAMMPS ML-IAP export
```

### 3. LAMMPS Integration

This framework supports three LAMMPS integration methods. See [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md) for details:

| Method | Speed | Requirements | Use Case |
|--------|-------|--------------|----------|
| **USER-MFFTORCH (LibTorch pure C++)** | Fastest, no Python | LAMMPS built with KOKKOS + USER-MFFTORCH | HPC, clusters, production |
| **ML-IAP unified** | Faster (~1.7x vs fix external) | LAMMPS built with ML-IAP | Recommended, GPU support |
| **fix external / pair_style python** | Slower | Standard LAMMPS + Python | Quick validation |

Python API examples: [Example 5a: LAMMPS LibTorch](#example-5a-lammps-libtorch-interface-usermfftorch-hpc-recommended) and [Example 5b: LAMMPS ML-IAP](#example-5b-lammps-mliap-unified-interface).

## Complete Workflow

### Step 1: Data Preprocessing

First, you need to preprocess your raw XYZ file into a format that the library can use.

```bash
mff-preprocess \
    --input-file 2000.xyz \
    --output-dir data \
    --train-ratio 0.95 \
    --max-radius 5.0 \
    --num-workers 8
```

**Parameter Description:**
- `--input-file`: Path to input XYZ file
- `--output-dir`: Output directory (default 'data'), all preprocessed files will be saved in this directory
- `--max-atom`: Optional legacy padded raw-storage size; by default `read_{train,val}.h5` uses variable-length per-sample groups
- `--train-ratio`: Training set ratio (default 0.95)
- `--max-radius`: Maximum radius for neighbor search
- `--num-workers`: Number of parallel processing workers

**Output Files (in `data/` directory):**
- `read_train.h5`, `read_val.h5` - Raw atomic data (variable-length `sample_i` groups by default; legacy padded format if `--max-atom` is set)
- `raw_energy_train.h5`, `raw_energy_val.h5` - Raw energies
- `correction_energy_train.h5`, `correction_energy_val.h5` - Correction energies
- `cell_train.h5`, `cell_val.h5` - Cell information
- `processed_train.h5`, `processed_val.h5` - Preprocessed data (if --preprocess-h5 is used)
- `fitted_E0.csv` - Fitted atomic reference energies (default: fitted from training set using least squares)

**Override custom extxyz keys**:

If your `extxyz` does not use the default `energy=...` plus `Properties=species:...:pos:...:force:...:Z:...` naming, you can override the keys explicitly:

```bash
mff-preprocess \
    --input-file custom.extxyz \
    --output-dir data \
    --energy-key REF_energy \
    --force-key REF_force \
    --species-key elem \
    --coord-key coords \
    --atomic-number-key atomic_number
```

Default search rules:
- `energy`: `energy=...` in the comment line
- `force`: `force` / `forces` / `f`
- `species`: `species` / `symbol` / `element`
- `coord`: `pos`
- `atomic number`: `Z` / `atomic_number`

### Step 2: Train Model

**Method 1: Using Preprocessed Data**
```bash
mff-train \
    --data-dir data \
    --train-prefix train \
    --val-prefix val \
    --epochs 1000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 2e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --weight-a-growth 1.05 \
    --weight-b-decay 0.98 \
    --force-shift-value 1.0 \
    --patience 20 \
    --vhat-clamp-interval 2000 \
    --max-vhat-growth 5.0 \
    --max-grad-norm 0.5 \
    --grad-log-interval 500 \
    --dump-frequency 250 \
    --energy-log-frequency 10 \
    --device cuda \
    --dtype float64 \
    --num-workers 8 \
    --max-radius 5.0
```

**Stress Training** (Optional, for periodic systems):
If XYZ contains stress/virial and is a PBC system, enable stress loss:
```bash
mff-train \
    --data-dir data \
    --stress-weight 0.1
```

**Custom Atomic Energies (E0)** (Optional):
- **Default behavior**: Use `fitted_E0.csv` under `--data-dir` (fitted from training set using least squares)
- **Method 1: Specify from CSV**:

```bash
mff-train \
    --data-dir data \
    --atomic-energy-file data/fitted_E0.csv
```

- **Method 2: Specify directly in CLI**:

```bash
mff-train \
    --data-dir data \
    --atomic-energy-keys 1 6 7 8 \
    --atomic-energy-values -430.53299511 -821.03326787 -1488.18856918 -2044.3509823
```

**ZBL Short-Range Repulsion** (Optional, recommended for high-energy close-contact configurations):

ZBL (`Ziegler-Biersack-Littmark`) is a classical short-range nucleus-nucleus repulsive potential. FSCETP can now add it as a unified short-range correction on top of the learned energy output to improve physical behavior at very small interatomic distances. Typical use cases include:

- irradiation damage, collision cascades, and ion implantation
- very close atomic contacts at high temperature or high pressure
- preventing unphysical short-range attraction in regions not covered by training data

Example: enable ZBL during training

```bash
mff-train \
    --data-dir data \
    --tensor-product-mode pure-cartesian-ictd \
    --zbl-enabled \
    --zbl-inner-cutoff 0.6 \
    --zbl-outer-cutoff 1.2 \
    --zbl-exponent 0.23 \
    --zbl-energy-scale 1.0
```

Parameter summary:

| Parameter | Description |
|-----------|-------------|
| `--zbl-enabled` | Enable ZBL short-range repulsion |
| `--zbl-inner-cutoff` | Inner cutoff in Angstrom. Full ZBL is applied below this distance |
| `--zbl-outer-cutoff` | Outer cutoff in Angstrom. ZBL is smoothly switched to zero beyond this distance |
| `--zbl-exponent` | Exponent used in the screening-length expression; the common default is `0.23` |
| `--zbl-energy-scale` | Global multiplicative scale for the ZBL energy term; `1.0` means standard strength |

Practical notes:

- You must keep `--zbl-inner-cutoff < --zbl-outer-cutoff`
- If your training set does not contain ultra-short contacts, a good starting point is `0.6 / 1.2 / 0.23 / 1.0`
- If your training data already covers the short-range collision regime explicitly, tune the cutoffs carefully to avoid double-counting the learned short-range behavior
- ZBL is attached as a unified correction layer outside the tensor-product backbone, so it works across the Python training/evaluation/ASE/ML-IAP stack; `core.pt` export still follows the existing TorchScript-supported mode restrictions

Where the constants come from:

- `14.399645...` is \(\frac{e^2}{4\pi\varepsilon_0}\) expressed in `eV·Angstrom`
- `0.529177... Angstrom` is the Bohr radius \(a_0\)
- The `0.8854` prefactor and default exponent `0.23` in the screening length come from the standard ZBL parameterization:
  \[
  a_{ij} = \frac{0.8854 a_0}{Z_i^{p} + Z_j^{p}}, \quad p=0.23
  \]
- The screening function
  \[
  \phi(x)=0.1818e^{-3.2x}+0.5099e^{-0.9423x}+0.2802e^{-0.4029x}+0.02817e^{-0.2016x}
  \]
  uses the standard universal ZBL fitted coefficients and exponents
- The quintic smoothstep
  \[
  1-10t^3+15t^4-6t^5
  \]
  is not an original ZBL physical constant; it is a numerical switching function used to keep both energy and forces smooth between the inner and outer cutoffs

In the current implementation, the pair potential is:

\[
E_{ij}^{\mathrm{ZBL}}(r)
=
s(r)\,\lambda\,
14.3996454784255
\frac{Z_i Z_j}{r}
\phi\!\left(\frac{r}{a_{ij}}\right)
\]

where \(\lambda = \texttt{zbl\_energy\_scale}\) and \(s(r)\) is the smooth switching function defined by `--zbl-inner-cutoff` and `--zbl-outer-cutoff`.

Checkpoint and downstream behavior:

- ZBL settings are saved into checkpoint metadata
- `mff-evaluate` restores the saved ZBL configuration by default
- `mff-export-core` and `python -m molecular_force_field.cli.export_mliap` also inherit the ZBL configuration from the checkpoint automatically
- In normal workflows you usually enable ZBL once during training, then reuse the saved checkpoint for evaluation and export without retyping the ZBL arguments

`--invariant-channels` (scalar invariant readout width):

- This parameter controls the scalar invariant channel width used by the product-3 style readout blocks in several tensor-product backbones; the default is `32`
- Supported modes: `spherical`, `spherical-save`, `partial-cartesian`, `partial-cartesian-loose`, `pure-cartesian`, `pure-cartesian-sparse`, `pure-cartesian-ictd`, `pure-cartesian-ictd-save`
- `spherical-save-cue` does not use this parameter; even if a checkpoint stores `invariant_channels`, that field is ignored by the cuEquivariance channelwise backend
- The previous behavior was equivalent to `--invariant-channels 32`; it is now exposed through the CLI, for example:

```bash
mff-train \
    --data-dir data \
    --tensor-product-mode pure-cartesian-ictd \
    --invariant-channels 64
```

- After training, `invariant_channels` is written into checkpoint `model_hyperparameters`
- Resume training, `mff-evaluate`, active learning loaders, thermal loaders, `mff-export-core`, and `python -m molecular_force_field.cli.export_mliap` all restore it from the checkpoint automatically
- In practice, active learning usually does not need `--invariant-channels` to be typed again unless you intentionally want to override the checkpoint architecture

Example: explicit override during evaluation

```bash
mff-evaluate \
    --checkpoint checkpoint/combined_model_pure_cartesian_ictd.pth \
    --input-file test.xyz \
    --tensor-product-mode pure-cartesian-ictd \
    --zbl-enabled \
    --zbl-inner-cutoff 0.6 \
    --zbl-outer-cutoff 1.2
```

**External Fields & Physical Tensor Training** (`pure-cartesian-ictd`, `pure-cartesian-ictd-o3`, `pure-cartesian-sparse`, `pure-cartesian-sparse-save`):

- **External field embedding**: Inject global tensors (e.g., electric field, rank=1) into conv1 for field-dependent potentials
- **Full-parity path**: any task with explicit parity semantics must use `pure-cartesian-ictd-o3`
- **Parity-sensitive tasks**: this includes magnetic-field (`1e`) inputs, magnetic moment (`1e`) outputs, and any pseudo/axial tensor task where `1e` and `1o` must remain distinct through the trunk
- **Physical tensor training**: Supervised outputs for charge, dipole, polarizability, quadrupole, and per-atom Born effective charge (BEC)

```bash
# External field (electric) + dipole/polarizability training
mff-train --data-dir data --tensor-product-mode pure-cartesian-sparse \
  --external-tensor-rank 1 --external-field-file data/efield.npy \
  --physical-tensors dipole,polarizability \
  --dipole-file data/dipole.npy --polarizability-file data/pol.npy \
  --physical-tensor-weights "dipole:2.0,polarizability:1.0"

# Per-atom physical tensors (requires per-node label HDF5)
mff-train --data-dir data --tensor-product-mode pure-cartesian-sparse-save \
  --extra-per-node-file data/per_atom_labels.h5 \
  --physical-tensors-per-node charge_per_atom,dipole_per_atom,born_effective_charge_per_atom
```

**External field & physical tensor parameters:**

| Parameter | Description |
|-----------|--------------|
| `--external-tensor-rank` | External tensor rank (e.g., 1=electric field), requires `--external-field-file` |
| `--external-tensor-irrep` | External-field irrep semantics; common values are `1o` (electric field) and `1e` (magnetic field). Any explicitly parity-sensitive irrep task should use `pure-cartesian-ictd-o3` |
| `--external-field-file` | External field label file (.npy/.npz/.h5, shape B×3) |
| `--charge-file` | Per-structure charge labels (scalar per sample) |
| `--dipole-file` | Per-structure dipole labels (B×3) |
| `--magnetic-moment-file` | Per-structure magnetic moment labels (B×3, `1e`; should be trained under `pure-cartesian-ictd-o3`) |
| `--polarizability-file` | Per-structure polarizability labels (B×3×3) |
| `--quadrupole-file` | Per-structure quadrupole labels (B×3×3) |
| `--extra-per-node-file` | Per-atom label HDF5 (sample_0, sample_1, ... with charge_per_atom, born_effective_charge_per_atom, etc.) |
| `--physical-tensors` | Per-structure outputs: charge,dipole,magnetic_moment,polarizability,quadrupole |
| `--physical-tensors-per-node` | Per-atom outputs: charge_per_atom,dipole_per_atom,magnetic_moment_per_atom,polarizability_per_atom,quadrupole_per_atom,born_effective_charge_per_atom |
| `--o3-irrep-preset` | Active-irrep preset for `pure-cartesian-ictd-o3`: `minimal` / `auto` / `balanced` / `full` |
| `--o3-active-irreps` | Explicit active-irrep override for `pure-cartesian-ictd-o3`, e.g. `0e,1e,2e` |
| `--physical-tensor-reduce` | Per-structure reduce: sum (default), mean, none |
| `--physical-tensor-weights` | Loss weights: `charge:1.0,dipole:2.0,...` |
| `--bec-derivative-weight` | Loss weight for BEC supervision derived from `dF/dE` (requires rank-1 external field + BEC labels) |
| `--bec-consistency-weight` | Loss weight tying the explicit BEC head to the `dF/dE`-derived BEC |
| `--inference-output-physical-tensors` | Save to checkpoint: inference outputs physical tensors (default: no; MD/LAMMPS only needs energy and forces) |

```bash
# Full-parity O(3) ICTD path: magnetic field (1e) + magnetic moment
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd-o3 \
  --external-tensor-rank 1 --external-tensor-irrep 1e \
  --o3-irrep-preset auto \
  --o3-active-irreps '0e,1e,2e' \
  --external-field-file data/bfield.npy \
  --physical-tensors magnetic_moment \
  --magnetic-moment-file data/magnetic_moment.npy

# Full-parity O(3) ICTD path: simultaneous electric field (1o) + magnetic field (1e)
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd-o3 \
  --external-tensor-rank 1 --external-tensor-irrep 1o \
  --external-field-file data/efield.npy \
  --magnetic-field-file data/bfield.npy \
  --o3-irrep-preset auto \
  --o3-active-irreps '0e,1e,1o,2e'
```

- `pure-cartesian-ictd-o3` is the strict full-parity path. Any explicitly parity-sensitive task should run in this mode.
- Other external fields can be embedded by pairing `--external-tensor-rank` with `--external-tensor-irrep`. Typical rank-1 cases are `1o` for electric field and `1e` for magnetic field. If the chosen irrep carries explicit parity semantics, use `pure-cartesian-ictd-o3`. Simultaneous rank-1 electric + magnetic conditioning is supported via `--external-field-file` together with `--magnetic-field-file`.

Note: `born_effective_charge_per_atom` is now wired through training, evaluation, Python inference, and the `mff/torch` export schema. To make LAMMPS read the new columns, rebuild `USER-MFFTORCH` with the updated C++ sources.

**Multi-fidelity training** (supported in `spherical-save-cue`, `pure-cartesian-ictd`, `pure-cartesian-ictd-o3`, `pure-cartesian-sparse`, `pure-cartesian-sparse-save`):

- `--num-fidelity-levels`: total number of fidelity levels
- `--fidelity-id-file`: one integer `fidelity_id` per structure
- `--multi-fidelity-mode conditioning`: shared trunk + shared head, with conv1 conditioning only
- `--multi-fidelity-mode delta-baseline`: shared trunk + baseline head + per-fidelity delta heads
- `--fidelity-loss-weights`: fidelity-specific loss weights such as `0:1.0,1:3.0`
- `--delta-regularization-weight`: regularization strength for delta heads, used with `delta-baseline`

```bash
# Conditioning-only multi-fidelity
mff --train \
  --data-dir data \
  --tensor-product-mode pure-cartesian-ictd \
  --num-fidelity-levels 2 \
  --fidelity-id-file data/train_fidelity_id.npy \
  --fidelity-loss-weights '0:1.0,1:3.0'

# Delta-learning multi-fidelity
mff --train \
  --data-dir data \
  --tensor-product-mode pure-cartesian-ictd-o3 \
  --num-fidelity-levels 2 \
  --multi-fidelity-mode delta-baseline \
  --fidelity-id-file data/train_fidelity_id.npy \
  --fidelity-loss-weights '0:1.0,1:3.0' \
  --delta-regularization-weight 1e-4
```

You can merge multiple processed HDF5 files into one multi-fidelity dataset:

```bash
mff --merge-multifidelity \
  --inputs data/processed_pbe.h5 data/processed_hse.h5 \
  --fidelity-ids 0 1 \
  --output-h5 data/processed_train_mf.h5 \
  --output-fidelity-npy data/train_fidelity_id.npy
```

**Method 2: Auto Preprocess and Train (One Step)**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --device cuda
```
If there are no preprocessed files in the `data/` directory, it will automatically preprocess from the XYZ file specified by `--input-file`.

**Quick Start Example (Using Default Parameters):**
```bash
# Simplest training command (using all default values)
mff-train --input-file 2000.xyz --device cuda
```

**Complete Parameter Example:**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 2000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --patience 20 \
    --dump-frequency 250 \
    --device cuda \
    --dtype float64
```

**Parameter Description:**

**Data Parameters:**
- `--data-dir`: Data directory containing preprocessed files (default 'data'). All preprocessed data files (e.g., processed_train.h5, processed_val.h5) should be in this directory
- `--input-file`: Path to input XYZ file for auto preprocessing (optional). If specified and no preprocessed files exist in the data directory, it will automatically preprocess from the XYZ file
- `--train-prefix`: Training data file prefix (default 'train'). Used to locate files like `processed_{train-prefix}.h5`
- `--val-prefix`: Validation data file prefix (default 'val'). Used to locate files like `processed_{val-prefix}.h5`
- `--max-radius`: Maximum radius for neighbor search (default 5.0 Å). Atoms beyond this distance will not be considered as neighbors. This parameter is required for both preprocessing and training

**Basic Training Parameters:**
- `--epochs`: Number of training epochs (default 1000). Training will run for the specified number of epochs unless early stopping is triggered
- `--batch-size`: Batch size (default 8). If GPU memory is insufficient, you can reduce this value (e.g., 4 or 2). Note: validation batch-size is fixed at 1
- `--checkpoint`: Model save path (default 'combined_model.pth'). After training, the final model (best validation performance model) will be saved to the `checkpoint/` directory, with filename including tensor product mode suffix (e.g., `combined_model_pure_cartesian.pth`). Intermediate checkpoints are also saved periodically to the `checkpoint/` directory during training
- `--reset-loss-weights`: When resuming from checkpoint, ignore saved loss weights (a, b) and use values from command line arguments (default False, i.e., use checkpoint weights)
- `--device`: Device ('cuda' or 'cpu', default auto-detect). If GPU is available, strongly recommend using 'cuda'. In distributed training, each process is automatically assigned to the corresponding GPU
- `--dtype`: Tensor data type ('float32' or 'float64', default 'float64'). Options: 'float32'/'float' (32-bit float, faster but slightly lower precision) or 'float64'/'double' (64-bit float, higher precision but slower)
- `--seed`: Random seed (default 42). Used to ensure experiment reproducibility. Affects random operations like data shuffle, train/val split, etc.

**Validation and Logging Parameters:**
- `--dump-frequency`: Frequency of validation and model saving (every N batches, default 250). Validation and model checkpoint saving will occur every N batches. Note: validation is triggered based on batch_count, not fixed at the end of each epoch
- `--train-eval-sample-ratio`: Ratio of training set to sample during validation (0.0-1.0, default 0.2). Set to 1.0 to evaluate the entire training set, or lower (e.g., 0.2 for 20%) for faster validation. Recommended for large datasets
- `--energy-log-frequency`: Frequency of energy prediction logging (every N batches, default 10). These logs are only written to files, not output to console
- `--log-val-batch-energy`: Output validation batch energy predictions to console (default False). If False, validation batch energy information is only logged to file, not output to console; if True, it will be output to both console and log file
- `--save-val-csv`: Save validation energy and force predictions to CSV files (default False). If enabled, validation results will be saved to the `validation/` directory with filenames `val_energy_epoch{epoch}_batch{batch_count}.csv` and `val_force_epoch{epoch}_batch{batch_count}.csv`
- `--no-save-val-csv`: Disable saving validation CSV files (mutually exclusive with `--save-val-csv`). Used to reduce I/O overhead

**Early Stopping Parameters:**
- `--patience`: Early stopping patience value (default 20, unit: epochs). If validation loss (energy + force + stress, unweighted) does not improve for N consecutive epochs, training will automatically stop. Note: early stopping uses unweighted validation loss, not weighted loss (`a × energy + b × force + c × stress`)

**Learning Rate Parameters:**
- `--learning-rate`: Target learning rate (default 1e-3). This is the learning rate after warmup and the main learning rate during training
- `--min-learning-rate`: Minimum learning rate (default 2e-5). Learning rate will not go below this value, even after multiple decays
- `--warmup-batches`: Number of batches for learning rate warmup (default 1000). For the first N batches, learning rate linearly increases from `learning_rate × warmup_start_ratio` to `learning_rate`
- `--warmup-start-ratio`: Starting ratio of learning rate during warmup (default 0.1). For example, if `--learning-rate 1e-3` and `--warmup-start-ratio 0.1`, learning rate will linearly increase from `1e-4` to `1e-3`
- `--lr-decay-patience`: Interval in batches for learning rate decay (default 1000). Every N batches, if validation metrics do not improve, learning rate will be multiplied by `--lr-decay-factor`
- `--lr-decay-factor`: Learning rate decay factor (default 0.98). Learning rate is multiplied by this value each time it decays (0.98 means 2% reduction). Learning rate scheduling: if validation loss does not improve within `lr-decay-patience` batches, learning rate is multiplied by this factor

**Loss Weight Parameters:**
- `--energy-weight` (or `-a`): Initial weight for energy loss (default 1.0). Total loss = `a × energy loss + b × force loss + c × stress loss`. During training, `a` will automatically increase according to `--weight-a-growth`
- `--force-weight` (or `-b`): Initial weight for force loss (default 10.0). Force loss typically needs larger weight because there are far more forces than energies (each atom has 3 force components). During training, `b` will automatically decay according to `--weight-b-decay`
- `--stress-weight` (or `-c`): Weight for stress loss (default 0.0, disabled). Set > 0 to enable stress training. Requires XYZ with stress/virial and PBC. Stress unit: eV/Å³
- `--update-param`: Frequency of automatic adjustment of weights `a` and `b` (every N batches, default 1000). Every N batches, weights will be adjusted according to `--weight-a-growth` and `--weight-b-decay`. Adjustment formula: `a = a × weight_a_growth`, `b = b × weight_b_decay`
- `--weight-a-growth`: Growth rate of energy weight `a` (default 1.05, i.e., 5% increase each time). Recommended values: 1.005 (slow, suitable for very long training), 1.01 (medium, more stable), 1.02 (fast), 1.05 (very fast)
- `--weight-b-decay`: Decay rate of force weight `b` (default 0.98, i.e., 2% decrease each time). Recommended values: 0.995 (slow), 0.99 (medium, more stable), 0.98 (fast)
- `--a-min`: Minimum value for energy weight `a` (default 1.0). `a` will not go below this value during growth
- `--a-max`: Maximum value for energy weight `a` (default 1000.0). `a` will not exceed this value during growth
- `--b-min`: Minimum value for force weight `b` (default 1.0). `b` will not go below this value during decay
- `--b-max`: Maximum value for force weight `b` (default 1000.0). `b` will not exceed this value during decay
- `--force-shift-value`: Scaling factor for force labels (default 1.0). Adjust this value if force units need conversion. Note: This parameter is currently not used in the code, reserved for future extensions

**Optimizer Parameters:**
- `--vhat-clamp-interval`: Frequency of optimizer `v_hat` clamping (every N batches, default 2000). Used to prevent Adam optimizer's second moment estimate (`v_hat`) from becoming too large. Every N batches, `v_hat` growth is checked and limited
- `--max-vhat-growth`: Maximum growth factor for `v_hat` (default 5.0). Limits `v_hat` to not exceed N times the historical maximum value, preventing optimizer instability
- `--max-grad-norm`: Gradient clipping threshold (default 0.5). If gradient norm exceeds this value, it will be clipped to this value to prevent gradient explosion. Implemented using `torch.nn.utils.clip_grad_norm_`
- `--grad-log-interval`: Frequency of gradient statistics logging (every N batches, default 500). Used to monitor training stability. Gradient statistics (norm, max, min, mean) are logged to the log file

**Data Processing Parameters:**
- `--num-workers`: Number of parallel processes for data processing (default 8). Uses all processes during preprocessing. During training, DataLoader automatically allocates: training DataLoader uses `max(1, num_workers // 2)`, validation DataLoader uses `max(1, num_workers // 4)`

**Model Architecture Hyperparameters:**
- `--max-atomvalue`: Maximum atomic number for atom embedding (default 10). If dataset contains elements with atomic number > 10, need to increase this value. For example, if dataset contains Cl (atomic number 17), need to set `--max-atomvalue 17`
- `--embedding-dim`: Atom embedding dimension (default 16). Increasing this value can enhance model expressiveness but will increase computation and memory usage
- `--embed-size`: Hidden layer sizes for readout MLP (default [128, 128, 128]). Can specify multiple values, e.g., `--embed-size 128 256 128` means three hidden layers with sizes 128, 256, 128
- `--output-size`: Output size for atom readout MLP (default 8). This is the feature dimension for each atom, used for subsequent energy and force prediction
- `--lmax`: Maximum order of spherical harmonics (default 2). Controls the highest order of irreducible representations. Increasing `lmax` can capture higher-order geometric information but will significantly increase computation. Common values: 1 (fast, suitable for simple systems), 2 (recommended, balanced performance and accuracy), 3 (high accuracy, but computationally expensive)
- `--irreps-output-conv-channels`: Number of channels for irreps_output_conv (optional, default None, will use channel_in from config, typically 64). Together with `--lmax` determines the irreps form. For example:
  - `lmax=2, channels=64` → "64x0e + 64x1o + 64x2e"
  - `lmax=1, channels=64` → "64x0e + 64x1o"
  - `lmax=3, channels=64` → "64x0e + 64x1o + 64x2e + 64x3o"
  - Increasing channel number can improve model capacity but will significantly increase memory and computation
- `--function-type`: Radial basis function type (default 'gaussian'). Options:
  - `gaussian`: Gaussian basis functions (default, smooth and easy to optimize, recommended for most scenarios)
  - `bessel`: Bessel basis functions (suitable for periodic systems)
  - `fourier`: Fourier basis functions (suitable for periodic boundary conditions)
  - `cosine`: Cosine basis functions
  - `smooth_finite`: Smooth finite support basis functions
- `--tensor-product-mode`: Equivariant tensor product implementation mode (default 'spherical'). **This framework supports eight equivariant tensor product modes**, options:
  - `spherical`: Use e3nn spherical harmonics tensor product (default, high precision, standard implementation)
  - `spherical-save`: channelwise edge conv (e3nn backend, fewer params)
  - `spherical-save-cue`: channelwise edge conv (cuEquivariance backend, requires `pip install -e ".[cue]"`, GPU accelerated)
  - `partial-cartesian`: Cartesian coordinates + e3nn CG coefficients (strictly equivariant, 17.4% parameter reduction)
  - `partial-cartesian-loose`: Approximate equivariant tensor product (norm product approximation, faster, 17.3% parameter reduction, not strictly equivariant)
  - `pure-cartesian`: Pure Cartesian tensor product (\(3^L\) representation, strictly equivariant, very slow, fails at lmax≥4, not recommended)
  - `pure-cartesian-sparse`: Sparse pure Cartesian tensor product (strictly equivariant, 29.6% parameter reduction, requires `--max-rank-other` and `--k-policy`)
  - `pure-cartesian-ictd`: ICTD irreps internal representation (strictly equivariant, 72.1% parameter reduction, fastest, recommended for large-scale training)
  
  For detailed performance comparison and recommended scenarios, see the [Tensor Product Mode Comparison](#tensor-product-mode-comparison) section.

**Tensor Product Mode Specific Parameters (only for pure-cartesian-sparse and pure-cartesian-ictd):**
- `--max-rank-other`: Maximum rank for sparse tensor product (only for `pure-cartesian-sparse` mode, default 1). Only interactions where `min(L1, L2) <= max_rank_other` are allowed. Increasing this value allows more interactions but increases parameters and computation
- `--k-policy`: K policy for sparse tensor product (only for `pure-cartesian-sparse` mode, default 'k0'). Options:
  - `k0`: Only keep k=0 (promotes higher rank, recommended)
  - `k1`: Only keep k=1 (contracts to lower rank)
  - `both`: Keep both k=0 and k=1 (more interactions, but more computation)
- `--ictd-tp-path-policy`: Path policy for ICTD tensor products (only for `pure-cartesian-ictd` mode, default 'full'). Options:
  - `full`: Keep all CG-allowed (l1,l2->l3) paths (recommended, best performance)
  - `max_rank_other`: Only keep paths where `min(l1,l2) <= --ictd-tp-max-rank-other` (reduces parameters)
- `--ictd-tp-max-rank-other`: Maximum rank for ICTD path pruning (only for `pure-cartesian-ictd` mode, when `--ictd-tp-path-policy=max_rank_other`, default None). For example, setting to 1 only keeps scalar/vector couplings

**Long-range parameters** (`mff-train` / `mff-evaluate` / `mff-active-learn`):

Control reciprocal / latent long-range paths. If not explicitly passed, CLI restores from checkpoint; when training without checkpoint, defaults are used. See [Active Learning: Recommended long-range setup](#recommended-long-range-setup) for reciprocal-spectral-v1 and slab examples.

**isolated-far-field-v1 CLI parameters** (nonperiodic isolated systems; O(N+E), no runtime reciprocal/tree source):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--long-range-mode` | — | Must be `isolated-far-field-v1` |
| `--long-range-boundary` | `nonperiodic` | Must be `nonperiodic`; periodic/slab not supported |
| `--long-range-hidden-dim` | `64` | Hidden dimension for source head and energy head |
| `--long-range-neutralize` / `--no-long-range-neutralize` | neutralize | Per-graph zero-mean for latent source |
| `--long-range-filter-hidden-dim` | `64` | Hidden dimension for shell stat MLP |
| `--long-range-source-channels` | `1` | Latent source channels; `latent_dim = max(source_channels, min(hidden_dim, 16))` |
| `--long-range-energy-partition` | `potential` | `potential` or `uniform`; how energy is partitioned to atoms |

**Parameters NOT used by isolated-far-field-v1** (ignored or error if passed): `--long-range-backend`, `--long-range-reciprocal-backend`, `--long-range-kmax`, `--long-range-mesh-size`, `--long-range-slab-padding-factor`, `--long-range-include-k0`, `--long-range-green-mode`, `--long-range-assignment`, `--long-range-theta`, `--long-range-leaf-size`, `--long-range-multipole-order`.

**SWA and EMA Parameters:**
- `--swa-start-epoch`: Epoch to start SWA (Stochastic Weight Averaging) (optional, default None). After enabling, `a` and `b` will directly switch to `--swa-a` and `--swa-b` values at this epoch, and reset early stopping counter (`best_val_loss` reset to inf, `patience_counter` reset to 0). If not set, continuous linear growth/decay strategy is used
- `--swa-a`: Energy weight `a` for SWA phase (must be used with `--swa-start-epoch`). When reaching `swa-start-epoch`, `a` will directly switch to this value and stop growing
- `--swa-b`: Force weight `b` for SWA phase (must be used with `--swa-start-epoch`). When reaching `swa-start-epoch`, `b` will directly switch to this value and stop decaying
- `--ema-start-epoch`: Epoch to start EMA (Exponential Moving Average) (optional, default None). EMA model is the exponential moving average of main model parameters, typically enabled in later training stages (recommended to start at ~60%-80% of total epochs). If not set, EMA functionality is disabled
- `--ema-decay`: EMA decay coefficient (default 0.999, range 0-1). Larger values are smoother but respond slower. Typical values: 0.999 (very smooth, recommended), 0.99 (faster response)
- `--use-ema-for-validation`: Use EMA model for validation (instead of main model). If enabled, validation will use EMA weights for forward propagation, typically resulting in more stable validation results
- `--save-ema-model`: Save EMA model weights in checkpoint. If enabled, checkpoint will contain `e3trans_ema_state_dict`, allowing EMA model recovery from checkpoint

**Validation Acceleration Parameters:**
- `--compile-val`: Use `torch.compile` on e3trans during validation, `none` (default) or `e3trans`
- `--compile-val-mode`: Compilation mode, e.g., `reduce-overhead`, `max-autotune`
- `--compile-val-fullgraph`: Force full graph compilation
- `--compile-val-dynamic`: Dynamic shapes

**Distributed Training Parameters:**
- `--distributed`: Enable distributed training (DDP mode). Requires `torchrun` or `torch.distributed.launch`. When enabled, training will automatically use multi-GPU parallelization
- `--local-rank`: Local process rank (default -1, usually automatically set by `torchrun`, no need to specify manually). If not set, will read from environment variable `LOCAL_RANK`
- `--backend`: Distributed backend (default 'nccl', options: 'nccl' or 'gloo'). 'nccl' for GPU training (recommended), 'gloo' for CPU training
- `--init-method`: Distributed initialization method (default 'env://'). Usually uses environment variable initialization, no need to modify

**Atomic Reference Energy (E0) Parameters:**
- `--atomic-energy-file`: Path to CSV file containing atomic reference energies (optional, default None). CSV file should contain `Atom` and `E0` columns. If not set, will use `{data-dir}/fitted_E0.csv` (fitted from training set using least squares)
- `--atomic-energy-keys`: List of atomic numbers (optional, must be used together with `--atomic-energy-values`). For example: `--atomic-energy-keys 1 6 7 8` represents H, C, N, O
- `--atomic-energy-values`: Corresponding atomic reference energy values (eV, optional, must be used together with `--atomic-energy-keys`). For example: `--atomic-energy-values -430.53 -821.03 -1488.19 -2044.35`

**Training Output:**

All output files are saved in the current working directory:

**Model Checkpoints (saved in `checkpoint/` directory):**
- `checkpoint/combined_model_{tensor_product_mode}.pth` - Final model (best validation performance model, filename includes tensor product mode suffix, e.g., `_spherical`, `_pure_cartesian`, etc.)
- `checkpoint/combined_model_epoch{epoch}_batch_count{batch_count}_{tensor_product_mode}.pth` - Periodically saved intermediate models (frequency controlled by `--dump-frequency`, default every 250 batches)

**Validation Results (saved in `validation/` directory, only when `--save-val-csv` is enabled):**
- `validation/val_energy_epoch{epoch}_batch{batch_count}.csv` - Validation set energy predictions (contains Target_Energy, Predicted_Energy, Delta columns)
- `validation/val_force_epoch{epoch}_batch{batch_count}.csv` - Validation set force predictions (contains Fx_True, Fy_True, Fz_True, Fx_Pred, Fy_Pred, Fz_Pred columns)

**Log and Record Files (saved in current directory):**
- `training_YYYYMMDD_HHMMSS.log` - Training log (contains detailed batch-level information, uses RotatingFileHandler, max 1GB per file, keeps 5 backups)
- `loss.csv` - Loss records (contains training and validation metrics for each validation point, such as loss, RMSE, MAE, etc.; when stress training is enabled, also includes stress_loss, stress_rmse, stress_mae; when physical tensor training is enabled, also includes train_phys_loss, val_phys_loss)

### Step 3: Evaluate Model

#### 3.1 Static Evaluation (Compute RMSE and MAE)

**Note: Evaluation must use the same model hyperparameters as training!**

```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --test-prefix test \
    --output-prefix test \
    --batch-size 1 \
    --max-atomvalue 10 \
    --embedding-dim 16 \
    --lmax 2 \
    --irreps-output-conv-channels 64 \
    --function-type gaussian \
    --tensor-product-mode spherical \
    --device cuda
```

**Parameter Description:**
- `--checkpoint`: Model checkpoint path
- `--test-prefix`: Test data file prefix
- `--output-prefix`: Output file prefix
- `--use-h5`: Use H5Dataset (if preprocessed), otherwise use OnTheFlyDataset
- `--batch-size`: Batch size (default 1)
- `--compile`: Inference acceleration, `none` (default) or `e3trans` (use `torch.compile` on e3trans layer)
- `--compile-mode`: Compilation mode, e.g., `reduce-overhead`, `max-autotune`
- `--max-atomvalue`: Must be the same as training
- `--embedding-dim`: Must be the same as training
- `--lmax`: Must be the same as training
- `--irreps-output-conv-channels`: Must be the same as training (if set during training)
- `--function-type`: Must be the same as training
- `--tensor-product-mode`: Must be the same as training (supports eight modes: `spherical`, `spherical-save`, `spherical-save-cue`, `partial-cartesian`, `partial-cartesian-loose`, `pure-cartesian`, `pure-cartesian-sparse`, `pure-cartesian-ictd`)
- `--output-physical-tensors`: Physical tensor output control (`auto`=use checkpoint's inference_output_physical_tensors, `true`=always output, `false`=never; use `false` when MD/LAMMPS only needs energy and forces)

**Output Files:**
- `test_loss.csv` - Test set loss metrics (includes phys_loss when physical tensors are enabled)
- `test_energy.csv` - Test set energy predictions
- `test_force.csv` - Test set force predictions

#### 3.2 Molecular Dynamics (MD) Simulation

Use ASE's Langevin thermostat for molecular dynamics simulation. MD simulation will automatically skip static evaluation and proceed directly to dynamics calculation.

**Basic Usage:**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --md-sim \
    --md-input start.xyz \
    --device cuda
```

**Complete Parameter Example:**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --md-sim \
    --md-input molecule.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 10000 \
    --md-friction 0.01 \
    --md-relax-fmax 0.05 \
    --md-log-interval 10 \
    --md-output md_traj.xyz \
    --device cuda
```

**MD Parameter Description:**

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--md-input` | `start_structure.xyz` | Input structure file (XYZ format) |
| `--md-temperature` | 300.0 | Simulation temperature (K) |
| `--md-timestep` | 1.0 | Timestep (fs) |
| `--md-steps` | 10000 | Total number of steps |
| `--md-friction` | 0.01 | Langevin friction coefficient |
| `--md-relax-fmax` | 0.05 | Pre-optimization force convergence threshold (eV/Å) |
| `--md-log-interval` | 10 | Log and trajectory recording interval |
| `--md-output` | `md_traj.xyz` | Output trajectory file |
| `--md-no-relax` | False | Skip initial structure optimization |

**MD Workflow:**
1. Load initial structure (`--md-input`)
2. Optional: Use BFGS optimizer for structure optimization (unless `--md-no-relax` is used)
3. Initialize Maxwell-Boltzmann velocity distribution
4. Run Langevin dynamics simulation
5. Periodically save trajectory and logs

**Output Files:**
- `md_traj.xyz` - MD trajectory (or file specified by `--md-output`)
- `md_traj_log.txt` - Energy/temperature log (filename based on `--md-output`)
- `relaxed_structure.xyz` - Optimized initial structure (if pre-optimization was performed)
- `md_relax.log` - Pre-optimization process log

**Example 1: Standard MD Simulation (300K, 10 ps)**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input molecule.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 10000 \
    --md-output md_300K.xyz \
    --device cuda
```

**Example 2: High Temperature MD Simulation (500K, 50 ps)**
```bash
# Run 50 ps at 500K
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input reactant.xyz \
    --md-temperature 500 \
    --md-timestep 0.5 \
    --md-steps 100000 \
    --md-friction 0.005 \
    --md-output md_500K.xyz \
    --device cuda
```

**Example 3: Skip Pre-optimization and Run MD Directly**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input pre_relaxed.xyz \
    --md-no-relax \
    --md-temperature 300 \
    --md-steps 50000 \
    --device cuda
```

**Example 4: Long MD Simulation (100 ps)**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input system.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 100000 \
    --md-log-interval 50 \
    --md-output md_long.xyz \
    --device cuda
```

**Notes:**
1. **Timestep Selection**: Usually 0.5-2.0 fs is safe. For light atoms (H), recommend using smaller timestep (0.5 fs)
2. **Friction Coefficient**: 0.01 is a common value. Smaller values (0.001-0.005) are suitable for long simulations, larger values (0.05-0.1) are suitable for rapid equilibration
3. **Pre-optimization**: Recommend optimizing initial structure unless it's already optimized
4. **Periodic Boundary Conditions**: If input structure contains cell information, PBC will be automatically used
5. **Energy Units**: All energies are in eV, forces are in eV/Å

#### 3.3 NEB (Nudged Elastic Band) Calculation

Use ASE's NEB method to find transition states and energy barriers for chemical reactions. NEB calculation will automatically skip static evaluation and proceed directly to path optimization.

**Basic Usage:**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --device cuda
```

**Complete Parameter Example:**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --neb \
    --neb-initial initial.xyz \
    --neb-final final.xyz \
    --neb-images 15 \
    --neb-fmax 0.03 \
    --neb-output neb.traj \
    --device cuda
```

**NEB Parameter Description:**

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--neb-initial` | `initial.xyz` | Initial structure (reactant) |
| `--neb-final` | `final.xyz` | Final structure (product) |
| `--neb-images` | 10 | Number of intermediate images (excluding endpoints) |
| `--neb-fmax` | 0.05 | Force convergence threshold (eV/Å), uses NEB projected forces |
| `--neb-output` | `neb.traj` | Output trajectory file (ASE Trajectory format) |

**NEB Workflow:**
1. Load initial and final structures
2. Create NEB image chain (initial + N intermediate images + final)
3. Linear interpolation to generate initial path
4. Optimize path using FIRE optimizer (Climbing Image NEB)
5. Output optimized path and barrier information

**Output Files:**
- `neb.traj` - NEB optimization trajectory (ASE Trajectory format, contains all images)
- `neb.log` - Optimization log file (contains fmax for each step)

**Output Information Includes:**
- Forward barrier: E_saddle - E_initial
- Reverse barrier: E_saddle - E_final
- Reaction energy: E_final - E_initial
- Detailed logs for each optimization step

**Prepare Input Files Example:**
```bash
# Create initial structure (reactant)
cat > initial.xyz << EOF
3

O  0.000  0.000  0.000
H  0.960  0.000  0.000
H -0.240  0.930  0.000
EOF

# Create final structure (product)
cat > final.xyz << EOF
3

O  0.000  0.000  0.000
H  1.500  0.000  0.000
H -0.240  0.930  0.000
EOF
```

**Example 1: Standard NEB Calculation**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 10 \
    --neb-fmax 0.05 \
    --device cuda
```

**Example 2: High Precision NEB Calculation**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 20 \
    --neb-fmax 0.02 \
    --neb-output neb_high_res.traj \
    --dtype float64 \
    --device cuda
```

**Example 3: Quick NEB Calculation (Fewer Images)**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 5 \
    --neb-fmax 0.1 \
    --device cuda
```

**View and Analyze NEB Results:**
```python
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np

# Read NEB trajectory
images = read('neb.traj', index=':')

# Extract energies
energies = [img.get_potential_energy() for img in images]

# Plot energy curve
plt.figure(figsize=(10, 6))
plt.plot(range(len(energies)), energies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Image Index', fontsize=12)
plt.ylabel('Energy (eV)', fontsize=12)
plt.title('NEB Energy Profile', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('neb_profile.png', dpi=300)
plt.close()

# Calculate barriers and reaction energy
e_initial = energies[0]
e_saddle = max(energies)
e_final = energies[-1]

forward_barrier = e_saddle - e_initial
reverse_barrier = e_saddle - e_final
reaction_energy = e_final - e_initial

print(f"Initial energy:    {e_initial:.4f} eV")
print(f"Saddle point:      {e_saddle:.4f} eV")
print(f"Final energy:      {e_final:.4f} eV")
print(f"Forward barrier:   {forward_barrier:.4f} eV")
print(f"Reverse barrier:   {reverse_barrier:.4f} eV")
print(f"Reaction energy:   {reaction_energy:.4f} eV")

# Visualize structures (using ASE GUI)
from ase.visualize import view
view(images)
```

**Using ASE to Analyze NEB Results:**
```python
from ase.io import read
from ase.neb import NEBTools

# Read trajectory
images = read('neb.traj', index=':')

# Use NEBTools for analysis
nebtools = NEBTools(images)

# Get barrier
barrier = nebtools.get_barrier()[0]  # Forward barrier
print(f"Forward barrier: {barrier:.4f} eV")

# Get saddle point index
saddle_index = nebtools.get_fitted_pes()[1]
print(f"Saddle point at image: {saddle_index}")

# Plot fitted potential energy surface
nebtools.plot_band()
```

**Notes:**
1. **Structure Requirements**: Initial and final structures must have the same number of atoms and atom types
2. **Pre-optimization**: Recommend optimizing initial and final structures separately first to ensure they are local minima
3. **Number of Images**: More images (15-20) provide higher path resolution but require more computation. 10 images is usually a good starting point
4. **Convergence Criterion**: `--neb-fmax` uses NEB projected forces (not raw atomic forces), which is the correct convergence criterion
5. **Climbing Image**: Climbing Image NEB (CI-NEB) is automatically enabled to precisely locate saddle points
6. **Periodic Boundary Conditions**: If input structures contain cell information, PBC will be automatically used
7. **Interpolation Method**: Uses improved tangent method for path interpolation
8. **Optimizer**: Uses FIRE optimizer, which is particularly effective for NEB optimization

#### 3.4 Phonon Spectrum Calculation

Calculate phonon spectrum (Hessian matrix, vibrational frequencies). Supports both non-periodic and periodic systems, uses ASE on-the-fly neighbor building by default.

**Basic Usage:**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --device cuda
```

**Complete Parameter Example:**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output my_phonon \
    --max-radius 5.0 \
    --atomic-energy-file fitted_E0.csv \
    --device cuda
```

**Phonon Spectrum Parameter Description:**

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--phonon` | False | Enable phonon spectrum calculation (required flag) |
| `--phonon-input` | `structure.xyz` | Input structure file (XYZ format) |
| `--phonon-relax-fmax` | 0.01 | Structure optimization force convergence threshold (eV/Å) |
| `--phonon-output` | `phonon` | Output file prefix (generates `{prefix}_hessian.npy` and `{prefix}_frequencies.txt`) |
| `--phonon-no-relax` | False | Skip structure optimization (if structure is already optimized) |
| `--max-radius` | 5.0 | Maximum radius for neighbor search (Å) |
| `--atomic-energy-file` | `fitted_E0.csv` | Atomic reference energy file (CSV format, contains Atom and E0 columns) |

**Phonon Spectrum Workflow:**
1. Load input structure (`--phonon-input`)
2. Optional: Use BFGS optimizer for structure optimization (unless `--phonon-no-relax` is used)
3. Use ASE on-the-fly neighbor list to build graph structure (automatically handles periodic boundary conditions)
4. Compute Hessian matrix (second derivative of energy with respect to positions, shape: `[N_atoms * 3, N_atoms * 3]`)
5. Calculate phonon frequencies from Hessian matrix (via dynamical matrix diagonalization)
6. Save results to files

**Output Files:**
- `{prefix}_hessian.npy` - Hessian matrix (NumPy format, units: eV/Å²)
- `{prefix}_frequencies.txt` - Phonon frequency list (units: cm⁻¹, negative values indicate imaginary frequencies/unstable modes)

**Output File Format:**

`{prefix}_frequencies.txt` format example:
```
# Phonon frequencies (cm⁻¹)
# Negative values indicate imaginary frequencies (unstable modes)
# Index    Frequency (cm⁻¹)
     0       1234.567890
     1        987.654321
     2        456.789012
   ...
```

**Example 1: Non-Periodic System (Molecule)**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input water.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output water_phonon \
    --device cuda
```

**Example 2: Periodic System (Crystal)**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input crystal.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output crystal_phonon \
    --max-radius 5.0 \
    --device cuda
```

**Example 3: Skip Structure Optimization (Already Optimized Structure)**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input pre_relaxed.xyz \
    --phonon-no-relax \
    --phonon-output phonon \
    --device cuda
```

**Example 4: High Precision Calculation (Stricter Optimization)**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --phonon-relax-fmax 0.005 \
    --phonon-output phonon_high_precision \
    --dtype float64 \
    --device cuda
```

**View and Analyze Results:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Read Hessian matrix
hessian = np.load('phonon_hessian.npy')
print(f"Hessian shape: {hessian.shape}")
print(f"Hessian range: [{hessian.min():.6f}, {hessian.max():.6f}] eV/Å²")

# Check symmetry
is_symmetric = np.allclose(hessian, hessian.T, atol=1e-4)
print(f"Hessian symmetric: {is_symmetric}")

# Read frequencies
frequencies = []
with open('phonon_frequencies.txt', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                freq = float(parts[1])
                frequencies.append(freq)

frequencies = np.array(frequencies)
print(f"\nTotal modes: {len(frequencies)}")
print(f"Real modes (≥0): {np.sum(frequencies >= 0)}")
print(f"Imaginary modes (<0): {np.sum(frequencies < 0)}")
print(f"Frequency range: [{frequencies.min():.2f}, {frequencies.max():.2f}] cm⁻¹")

# Plot frequency distribution
plt.figure(figsize=(10, 6))
plt.hist(frequencies[frequencies >= 0], bins=50, alpha=0.7, label='Real frequencies')
if np.any(frequencies < 0):
    plt.hist(frequencies[frequencies < 0], bins=20, alpha=0.7, label='Imaginary frequencies', color='red')
plt.xlabel('Frequency (cm⁻¹)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Phonon Frequency Distribution', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phonon_frequencies.png', dpi=300)
plt.close()
```

**Notes:**
1. **Structure Optimization**: Recommend optimizing structure to energy minimum before calculation (enabled by default). If structure is not optimized, imaginary frequencies (negative frequencies) may appear
2. **Periodic Boundary Conditions**: If input structure contains cell information (`Lattice` attribute), PBC and on-the-fly neighbor building will be automatically used
3. **Neighbor Building**: Uses ASE on-the-fly neighbor list by default, automatically handles non-periodic and periodic systems
4. **Large Systems**: For systems with >100 atoms, Hessian calculation may be slow (O(N²) complexity)
5. **Atomic Energies**: Need to provide correct atomic reference energies (via `--atomic-energy-file` or `fitted_E0.csv` generated during training)
6. **Imaginary Frequencies**: If imaginary frequencies (negative values) appear, it indicates the structure is unstable, need further optimization or check the structure
7. **Hessian Symmetry**: Theoretically Hessian matrix should be symmetric, numerical error is usually < 1e-4
8. **Units**:
   - Hessian matrix: eV/Å²
   - Frequency: cm⁻¹
   - Real frequency: positive values (stable modes)
   - Imaginary frequency: negative values (unstable modes)

**Dependency Requirements:**
```bash
pip install scipy  # Required for phonon spectrum calculation (for matrix diagonalization)
pip install ase    # Structure processing and neighbor lists (usually already installed)
```

## Cold Start: Generate Initial Dataset (mff-init-data)

When you only have seed structures and no labeled data, `mff-init-data` generates an initial dataset in one step: **perturb → DFT label → preprocess**:

```bash
mff-init-data --structures water.xyz ethanol.xyz \
    --n-perturb 15 --rattle-std 0.05 \
    --label-type pyscf --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --output-dir data

# Periodic systems
mff-init-data --structures POSCAR.vasp \
    --n-perturb 20 --rattle-std 0.02 --cell-scale-range 0.03 \
    --label-type vasp --vasp-xc PBE --vasp-encut 500 \
    --output-dir data
```

Key parameters: `--n-perturb` (perturbation count), `--rattle-std` (displacement σ, Å), `--cell-scale-range` (cell scaling range for periodic systems), `--min-dist` (minimum interatomic distance filter). See [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md) for full details.

---

## Active Learning (mff-active-learn)

The active learning module implements a DPGen2-style workflow: **Train ensemble → Explore (MD/NEB) → Select by force deviation → Label (DFT) → Merge data → Repeat**, to automatically sample under-sampled regions of the potential energy surface and expand the training set. It supports **single-node** (parallel labeling with multiple workers) and **HPC** (SLURM: one job per structure). The same DFT script template can be used for local runs (`local-script`) and cluster runs (`slurm`).

The ASE active-learning path also supports long-range checkpoints:

- Recommended setup: `reciprocal-spectral-v1 + mesh_fft + poisson + potential + cic`
- Supported boundaries: `periodic` and `slab`
- `slab` means `x/y periodic + z vacuum padding`
- `isolated-far-field-v1`: nonperiodic isolated systems; O(N+E), lightweight global-minus-local far-field correction, no runtime reciprocal/tree source
- `isolated-far-field-v2`: nonperiodic isolated systems; sparse explicit far shells + physical kernel + coarse multi-bin tail, still exported with no runtime reciprocal/tree source
- Verified entry points: `IdentityLabeler`, `ModelDeviCalculator`, `run_ase_md`, and `run_active_learning_loop`

### Basic usage

Required arguments: `--explore-type` (`ase` or `lammps`), `--label-type` (see table below). Single-stage example:

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --md-steps 500 --n-iterations 5
```

View all options:

```bash
mff-active-learn --help
```

### Recommended long-range setup

If your checkpoint uses the LES-style reciprocal long-range path, the recommended training setup is:

```bash
# periodic
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd \
    --long-range-mode reciprocal-spectral-v1 \
    --long-range-reciprocal-backend mesh_fft \
    --long-range-boundary periodic \
    --long-range-mesh-size 16 \
    --long-range-green-mode poisson \
    --long-range-energy-partition potential \
    --long-range-assignment cic

# slab
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd \
    --long-range-mode reciprocal-spectral-v1 \
    --long-range-reciprocal-backend mesh_fft \
    --long-range-boundary slab \
    --long-range-mesh-size 16 \
    --long-range-slab-padding-factor 2 \
    --long-range-green-mode poisson \
    --long-range-energy-partition potential \
    --long-range-assignment cic
```

**isolated far-field heads (nonperiodic isolated systems):**

`isolated-far-field-v1` is the lighter global-minus-local far-field correction, while `isolated-far-field-v2` uses sparse explicit far shells, a screened-Coulomb-style physical kernel multiplied by a learned radial gate, and a configurable coarse tail. `v2` is typically closer to `O(N log N + E + E_far)` than the old dense all-pairs implementation, although worst-case behavior can still degrade toward `O(N^2)`. Exported `core.pt` still runs directly via `pair_style mff/torch` with `long_range_runtime_backend=none`.

```bash
# isolated-far-field-v1: recommended
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd \
    --long-range-mode isolated-far-field-v1 \
    --long-range-boundary nonperiodic \
    --long-range-energy-partition potential

# isolated-far-field-v2: sparse explicit far shells + physical kernel + coarse multi-bin tail
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd \
    --long-range-mode isolated-far-field-v2 \
    --long-range-boundary nonperiodic \
    --long-range-energy-partition potential
```

**isolated-far-field-v1 CLI parameters** (`mff-train` / `mff-evaluate` / `mff-active-learn`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--long-range-mode` | — | Must be `isolated-far-field-v1` |
| `--long-range-boundary` | `nonperiodic` | Must be `nonperiodic`; periodic/slab not supported |
| `--long-range-hidden-dim` | `64` | Hidden dimension for source head and energy head |
| `--long-range-neutralize` / `--no-long-range-neutralize` | neutralize | Per-graph zero-mean for latent source |
| `--long-range-filter-hidden-dim` | `64` | Hidden dimension for shell stat MLP |
| `--long-range-source-channels` | `1` | Latent source channels; `latent_dim = max(source_channels, min(hidden_dim, 16))` |
| `--long-range-energy-partition` | `potential` | `potential` or `uniform`; how energy is partitioned to atoms |

**Parameters NOT used by isolated-far-field-v1** (ignored or error if passed): `--long-range-backend`, `--long-range-reciprocal-backend`, `--long-range-kmax`, `--long-range-mesh-size`, `--long-range-slab-padding-factor`, `--long-range-include-k0`, `--long-range-green-mode`, `--long-range-assignment`, `--long-range-theta`, `--long-range-leaf-size`, `--long-range-multipole-order`.

**isolated-far-field-v2 CLI parameters** (`mff-train` / `mff-evaluate` / `mff-active-learn`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--long-range-mode` | — | Must be `isolated-far-field-v2` |
| `--long-range-boundary` | `nonperiodic` | Must be `nonperiodic`; periodic/slab not supported |
| `--long-range-hidden-dim` | `64` | Hidden dimension for the source head, shell encoder, and energy head. It indirectly controls the learned radial gate capacity but does not change the kernel family itself. |
| `--long-range-neutralize` / `--no-long-range-neutralize` | neutralize | Whether to apply per-graph zero-mean normalization to the far latent source. Keeping this enabled is usually more stable. |
| `--long-range-energy-partition` | `potential` | `potential` or `uniform`; controls how the graph-level long-range correction is partitioned back to atoms. |
| `--long-range-far-source-dim` | `16` | Width of the invariant far latent source projected from the local representation before shell/tail aggregation. |
| `--long-range-far-num-shells` | `3` | Number of explicit sparse far shells; must be `>= 1`. The explicit far cutoff is no longer entered separately; it is derived internally as `far_shell_growth ** far_num_shells`. |
| `--long-range-far-shell-growth` | `2.0` | Geometric shell growth factor; must be `> 1`. For example, with `3` shells and growth `2.0`, the explicit shells are roughly `(r_c,2r_c]`, `(2r_c,4r_c]`, `(4r_c,8r_c]`, so the explicit far cutoff becomes `8 * cutoff`. It also sets the scale used by the current coarse tail bins. |
| `--long-range-far-tail` / `--no-long-range-far-tail` | tail enabled | Whether to enable the coarse multi-bin tail. |
| `--long-range-far-tail-bins` | `2` | Number of coarse tail bins; must be `>= 1`. When tail is enabled, the first `N-1` bins follow geometric ranges like `(R_max, growth*R_max]`, `(growth*R_max, growth^2*R_max]`, and the last bin is `(growth^(N-1)*R_max, +inf)`. |
| `--long-range-far-stats` | `mean,count,mean_r,rms_r` | Currently the only supported shell statistics bundle. In the current implementation, the `count` slot is effectively a kernel-weight sum rather than a raw atom count. |
| `--long-range-far-source-norm` / `--no-long-range-far-source-norm` | enabled | Whether to apply LayerNorm to the projected far source before shell/tail aggregation. |
| `--long-range-far-gate-init` | `0.0` | Initial scalar gate on the shell-encoded far features, used to keep the long-range correction conservative at initialization. |

`isolated-far-field-v2` also has one **implementation detail that is currently fixed and not exposed as a CLI knob**:
- The physical kernel is `exp(-kappa r) / r × learned radial gate`, where both `kappa` and the gate parameters are learned inside the model and automatically serialized into checkpoints / `core.pt`

`isolated-far-field-v2` also **derives** the explicit far cutoff internally:
- `far_max_radius_multiplier = far_shell_growth ** far_num_shells`
- This keeps the outer edge of the last explicit shell exactly aligned with the tail start, avoiding either empty explicit shells or an unencoded gap between the explicit-shell region and the tail region

`tail_bins` also has a clear **cost/benefit trade-off**:
- Increasing `tail_bins` makes the tail-statistics loop longer and widens `shell_features`, which in turn increases the input size of the downstream `shell_encoder` and `energy_head`. In practice, the extra cost is usually closer to linear in `tail_bins`, rather than changing `E_far` the way the explicit-shell radius does.
- A good starting range is usually `2` or `3`: `2` is a solid baseline, while `3-4` is reasonable when you specifically want finer stratification beyond `R_max`.
- For many systems, especially when the explicit shells already cover most of the important far field, `tail_bins=1-2` is enough. Very large values are usually not the first knob to turn because the farthest bins are still coarse summary statistics, so returns often diminish.

**How each CLI uses these arguments:**
- `mff-train`: the flags above define the actual `v2` architecture and are written into checkpoint metadata. For reproducibility, it is best to set them explicitly when you start a training run.
- `mff-evaluate`: the same flags can override the saved `v2` metadata in a checkpoint. If you do not pass them, evaluation restores the checkpoint configuration automatically. In routine evaluation, overriding is usually unnecessary unless you are doing ablations or debugging.
- `mff-active-learn`: the flags are forwarded into every retraining call to `mff-train`, so the active-learning loop keeps the `v2` architecture stable across iterations.

**Typical recommended commands for `isolated-far-field-v2`:**

```bash
# 1) mff-train: explicitly pin the v2 architecture
mff-train \
  --data-dir data \
  --tensor-product-mode pure-cartesian-ictd \
  --long-range-mode isolated-far-field-v2 \
  --long-range-boundary nonperiodic \
  --long-range-hidden-dim 64 \
  --long-range-neutralize \
  --long-range-energy-partition potential \
  --long-range-far-source-dim 16 \
  --long-range-far-num-shells 3 \
  --long-range-far-shell-growth 2.0 \
  --long-range-far-tail \
  --long-range-far-tail-bins 2 \
  --long-range-far-stats mean,count,mean_r,rms_r \
  --long-range-far-source-norm \
  --long-range-far-gate-init 0.0
```

```bash
# 2) mff-evaluate: usually restore from checkpoint; only pass structural overrides when needed
mff-evaluate \
  --checkpoint model_v2.pth \
  --tensor-product-mode pure-cartesian-ictd \
  --long-range-mode isolated-far-field-v2 \
  --long-range-boundary nonperiodic \
  --long-range-energy-partition potential
```

```bash
# 3) mff-active-learn: forward the v2 architecture into every internal mff-train call
mff-active-learn \
  --work-dir al_work \
  --data-dir data \
  --explore-type ase \
  --explore-mode md \
  --label-type pyscf \
  --tensor-product-mode pure-cartesian-ictd \
  --long-range-mode isolated-far-field-v2 \
  --long-range-boundary nonperiodic \
  --long-range-hidden-dim 64 \
  --long-range-energy-partition potential \
  --long-range-far-source-dim 16 \
  --long-range-far-num-shells 3 \
  --long-range-far-shell-growth 2.0 \
  --long-range-far-tail \
  --long-range-far-tail-bins 2 \
  --long-range-far-stats mean,count,mean_r,rms_r \
  --long-range-far-source-norm \
  --long-range-far-gate-init 0.0
```

**Parameters NOT used by isolated-far-field-v2** (ignored or error if passed): `--long-range-backend`, `--long-range-reciprocal-backend`, `--long-range-kmax`, `--long-range-mesh-size`, `--long-range-slab-padding-factor`, `--long-range-include-k0`, `--long-range-green-mode`, `--long-range-assignment`, `--long-range-theta`, `--long-range-leaf-size`, `--long-range-multipole-order`, `--long-range-source-channels`, `--long-range-filter-hidden-dim`.

You can then pass such checkpoints directly into `mff-active-learn`; no extra active-learning-specific long-range flag is required.

For slab active learning, prefer an initial structure with explicit `pbc="T T F"` in `extxyz` / `xyz`, so the z direction stays open.

### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--work-dir` | `al_work` | Active learning working directory |
| `--data-dir` | `data` | Training data directory (typically contains `processed_train.h5` plus raw files such as variable-length `read_train.h5` / `raw_energy_train.h5`) |
| `--init-structure` | Auto from data-dir | One or more initial structure paths (multi-structure parallel exploration), or a directory of .xyz files |
| `--init-checkpoint` | None | Optional warm-start checkpoint(s). Iteration 0 can skip training and explore directly; 1 checkpoint=bootstrap mode, `n_models` checkpoints=full ensemble |
| `--n-models` | 4 | Number of ensemble models |
| `--n-iterations` | 20 | Max iterations per stage (when not using --stages) |
| `--explore-type` | Required | Exploration backend: `ase` or `lammps` |
| `--explore-mode` | `md` | Exploration mode: `md` (molecular dynamics) or `neb` (elastic band) |
| `--explore-n-workers` | `1` | Parallel workers for multi-structure exploration; `1`=sequential, `>1`=concurrent threads (ThreadPoolExecutor) |
| `--label-type` | Required | Labeling method, see table below |
| `--md-temperature` | 300 | MD temperature (K) |
| `--md-steps` | 10000 | MD steps |
| `--md-timestep` | 1.0 | MD timestep (fs) |
| `--md-friction` | 0.01 | Langevin friction |
| `--md-relax-fmax` | 0.05 | Pre-relax force convergence (eV/Å) |
| `--md-log-interval` | 10 | Trajectory log interval |
| `--level-f-lo` / `--level-f-hi` | 0.05 / 0.5 | Force deviation selection thresholds (eV/Å) |
| `--conv-accuracy` | 0.9 | Convergence criterion ratio |
| `--epochs` | mff-train default | Training epochs per model per iteration |
| `--train-n-gpu` | 1 | GPUs per ensemble model per node. 1=single process (CPU/single GPU compatible), >1=torchrun DDP |
| `--train-max-parallel` | 0 (auto) | Max models trained simultaneously. 0=auto (available_gpus // train_n_gpu), 1=sequential. Forced to 1 for multi-node |
| `--train-nnodes` | 1 | Nodes **per model**. SLURM auto-computes `total_nodes // nnodes` for parallel training |
| `--train-master-addr` | auto | Rendezvous address. `auto`=resolve from SLURM or local hostname |
| `--train-master-port` | 29500 | Base port. Parallel models auto-offset (`+slot`) to avoid collisions |
| `--train-launcher` | auto | Launcher: `auto` / `local` / `slurm`. SLURM auto-assigns disjoint `--nodelist` per model |
| `--resume` | Off | Resume active learning from `work_dir/al_state.json` and existing `iterations/iter_*` artifacts |
| `--stages` | None | Path to multi-stage JSON file |
| `--device` | `cuda` | Inference device |
| `--max-radius` | 5.0 | Neighbor search cutoff (Å) |
| `--atomic-energy-file` | `data/fitted_E0.csv` | Atomic reference energy CSV |
| `--neb-initial` / `--neb-final` | None | Initial/final structures for NEB mode |

### Multi-layer candidate filtering

Candidates pass through three filtering layers before labeling, significantly reducing DFT cost and improving training-set diversity:

| Layer | Name | Description |
|-------|------|-------------|
| **Layer 0** | Fail recovery | Optionally promote the least extreme `fail` frames into the candidate pool |
| **Layer 1** | Uncertainty gate | Keep frames with `level_f_lo ≤ max_devi_f < level_f_hi` (DPGen2 trust window) |
| **Layer 2** | Diversity selection | Structural fingerprint (SOAP / deviation histogram) + FPS for maximum coverage |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--diversity-metric` | `soap` | Fingerprint: `soap` (requires dscribe), `devi_hist` (zero extra inference), `none` (skip) |
| `--max-candidates-per-iter` | 50 | Max candidates to keep after diversity selection |
| `--soap-rcut` | 5.0 | SOAP cutoff radius (Å) |
| `--soap-nmax` | 8 | SOAP radial expansion order |
| `--soap-lmax` | 6 | SOAP angular expansion order |
| `--soap-sigma` | 0.5 | SOAP Gaussian smearing width |
| `--devi-hist-bins` | 32 | Number of bins for `devi_hist` fingerprint |
| `--fail-strategy` | `discard` | `discard` (drop fail frames) or `sample_topk` (promote mildest fail frames) |
| `--fail-max-select` | 10 | Number of fail frames to promote with `sample_topk` |

> SOAP requires `dscribe`: `pip install dscribe` or `pip install molecular_force_field[al]`. Falls back to `devi_hist` if not installed.

### Label types (--label-type)

| Type | Description | Typical use |
|------|-------------|-------------|
| `identity` | Use current ML model (no DFT) | Debug, quick pipeline test |
| `pyscf` | PySCF (no external binary) | Small molecules, local validation |
| `vasp` | VASP (ASE interface) | Plane-wave DFT, single-node or in-script MPI |
| `cp2k` | CP2K (ASE interface) | Gaussian+plane-wave, single-node |
| `espresso` | Quantum Espresso pw.x (ASE interface) | Single-node QE |
| `gaussian` | Gaussian g16/g09 (ASE interface) | Single-node |
| `orca` | ORCA (ASE interface) | Single-node |
| `script` | User script: `script_path input.xyz output.xyz` | Any DFT/code |
| `local-script` | Same template as SLURM, **run locally** | Single-node, same script |
| `slurm` | Same template as local-script, **one sbatch job per structure** | HPC multi-node |

### CLI command examples

**Quick test (no DFT, ML bootstrap):**

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type identity \
    --md-temperature 300 --md-steps 200 --n-iterations 2 --epochs 5 --n-models 2
```

**PySCF (local, no external DFT binary):**

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 8 --md-steps 500 --n-iterations 5
```

**Start iteration 0 directly from an existing checkpoint:**

```bash
# Single-checkpoint bootstrap:
# iteration 0 skips training and goes directly to MD -> candidate/diversity -> labeling
mff-active-learn --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --init-checkpoint warm_start.pth \
    --n-models 4 \
    --md-steps 1000 --n-iterations 10

# Full warm-start ensemble:
# provide n_models checkpoints to keep ensemble deviation in iteration 0
mff-active-learn --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --init-checkpoint model_0.pth model_1.pth model_2.pth model_3.pth \
    --n-models 4 \
    --md-steps 1000 --n-iterations 10
```

**Resume interrupted active learning:**

```bash
mff-active-learn --work-dir al_work --data-dir data \
    --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --resume
```

**VASP (single-node, ASE interface):**

```bash
export ASE_VASP_COMMAND="mpirun -np 4 vasp_std"
export VASP_PP_PATH=/path/to/potpaw_PBE

mff-active-learn --explore-type ase --label-type vasp \
    --vasp-xc PBE --vasp-encut 500 --vasp-kpts 4 4 4 \
    --label-n-workers 8 --label-threads-per-worker 4 \
    --md-steps 1000 --n-iterations 10
```

**CP2K:**

```bash
mff-active-learn --explore-type ase --label-type cp2k \
    --cp2k-xc PBE --cp2k-cutoff 600 --md-steps 500 --n-iterations 5
```

**Quantum Espresso:**

```bash
mff-active-learn --explore-type ase --label-type espresso \
    --qe-pseudo-dir /path/to/pseudos \
    --qe-pseudopotentials '{"H":"H.pbe.UPF","O":"O.pbe.UPF"}' \
    --qe-ecutwfc 60 --md-steps 500 --n-iterations 5
```

**Gaussian:**

```bash
mff-active-learn --explore-type ase --label-type gaussian \
    --gaussian-method b3lyp --gaussian-basis 6-31+G* --gaussian-nproc 8 \
    --md-steps 500 --n-iterations 5
```

**ORCA:**

```bash
mff-active-learn --explore-type ase --label-type orca \
    --orca-simpleinput "B3LYP def2-TZVP TightSCF" --orca-nproc 8 \
    --md-steps 500 --n-iterations 5
```

**User script (any DFT):**

```bash
mff-active-learn --explore-type ase --label-type script --label-script ./my_dft.sh \
    --md-steps 500 --n-iterations 5
```

**local-script (run script template locally):**

```bash
mff-active-learn --explore-type ase --label-type local-script \
    --local-script-template dft_job.sh \
    --label-n-workers 4 --md-steps 500 --n-iterations 3
```

**SLURM (HPC, one job per structure):**

```bash
mff-active-learn --explore-type ase --label-type slurm \
    --slurm-template dft_job.sh --slurm-partition cpu \
    --slurm-nodes 1 --slurm-ntasks 32 --slurm-time 04:00:00
```

**Multi-stage JSON (e.g. 300K then 600K):**

```bash
mff-active-learn --explore-type ase --label-type pyscf --stages stages.json
```

**NEB exploration:**

```bash
mff-active-learn --explore-type ase --explore-mode neb \
    --neb-initial reactant.xyz --neb-final product.xyz \
    --label-type pyscf --pyscf-method b3lyp --n-iterations 5
```

**Parallel ensemble training (auto GPU assignment):**

```bash
# 8 GPUs, 4 models × 1 GPU each → 4 models in parallel (auto)
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --n-models 4 --train-n-gpu 1 --md-steps 500 --n-iterations 5

# 8 GPUs, 4 models × 2 GPUs DDP each → 4 models in parallel (auto 8÷2=4)
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --n-models 4 --train-n-gpu 2 --md-steps 500 --n-iterations 5

# Manually cap parallelism: 8 GPUs, 1 GPU/model, max 2 concurrent
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --n-models 4 --train-n-gpu 1 --train-max-parallel 2
```

**Multi-node DDP training (HPC/SLURM):**

```bash
# Scenario 1: 2 SLURM nodes × 4 GPUs, 4 models × 2 nodes each → sequential (2÷2=1)
#SBATCH --nodes=2 --gres=gpu:4
mff-active-learn --explore-type ase --label-type vasp \
    --train-n-gpu 4 --train-nnodes 2 --n-models 4

# Scenario 2: 8 SLURM nodes × 4 GPUs, 4 models × 2 nodes each → 4 in parallel (8÷2=4)
#SBATCH --nodes=8 --gres=gpu:4
mff-active-learn --explore-type ase --label-type vasp \
    --train-n-gpu 4 --train-nnodes 2 --n-models 4

# Scenario 3: 2 nodes × 8 GPUs, 4 models × 4 GPUs each → 4 in parallel (cross-node)
# 2 models per node (8÷4=2), 2 nodes × 2 = 4 parallel slots
#SBATCH --nodes=2 --gres=gpu:8
mff-active-learn --explore-type ase --label-type vasp \
    --train-n-gpu 4 --train-nnodes 1 --n-models 4
```

> **Resource assignment (three auto-switching modes):**
>
> | Condition | Mode | Strategy |
> |-----------|------|----------|
> | `nnodes=1`, no SLURM or single-node | **Local** | Local GPU pool split by `n_gpu` |
> | `nnodes=1`, SLURM multi-node | **Cross-node** | Each node's GPUs split by `n_gpu`, models dispatched via `srun --nodelist` |
> | `nnodes>1` | **Multi-node DDP** | SLURM nodes grouped by `nnodes`, each group trains one model |
>
> - Each concurrent model gets unique port `master_port + slot`
> - Cross-node mode uses `srun --nodes=1 --nodelist=<node>` with `CUDA_VISIBLE_DEVICES` per model
> - Without SLURM (`launcher=local`), multi-node defaults to sequential
>
> **DDP safety guarantees:**
> - Only rank 0 writes checkpoints (`is_main_process` guard)
> - All ranks synchronize via `dist.barrier()` after checkpoint writes
> - `dist.destroy_process_group()` is called on all ranks at exit

### Concurrency and error handling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--label-n-workers` | 1 | Number of structures to run in parallel (processes) |
| `--label-threads-per-worker` | 1 (when n_workers>1) or auto | Threads per structure (e.g. PySCF/VASP OpenMP) |
| `--label-error-handling` | `raise` | `raise` (stop on first failure) or `skip` (skip failed structures) |

Recommendation: `n_workers × threads_per_worker ≤ total CPU cores`.

### Workflow (what each iteration does)

Each iteration (until the stage’s `max_iters` or convergence) roughly does:

1. **Train ensemble**: Train `n_models` models (different seeds) on current `data_dir`, save checkpoints to `work_dir/checkpoint/`.
2. **Explore**: Run MD (or NEB) from the initial structure using one model; trajectory written to `work_dir/iterations/iter_<i>/explore_traj.xyz`.
3. **Select (multi-layer)**:
   - **Layer 0** (fail recovery): if `--fail-strategy sample_topk`, promote the mildest fail frames.
   - **Layer 1** (uncertainty gate): keep frames with `level_f_lo ≤ max_devi_f < level_f_hi`.
   - **Layer 2** (diversity): SOAP / deviation-histogram fingerprint + FPS, capped at `--max-candidates-per-iter`.
4. **Label**: Run DFT (or identity/script) on selected configs; write extended XYZ (energy, forces) to `iterations/iter_<i>/labeled/`.
5. **Merge**: Merge new labeled data into `data_dir` (update `read_train.h5`, `raw_energy_train.h5`, `processed_train.h5`, etc.) for the next round.
   `read_train.h5` stores raw atomic data in variable-length `sample_i` groups by default, without fixed `max-atom` padding.

With `--stages`, stages run in order; iterations repeat within each stage and data accumulates across stages.

### Working directory layout (--work-dir)

Typical layout (default `al_work`):

```
al_work/
├── checkpoint/           # Ensemble checkpoints (model_0_*.pth, model_1_*.pth, ...)
├── init.xyz              # Initial structure (from --init-structure or data_dir)
└── iterations/
    ├── iter_0/
    │   ├── explore_traj_0.xyz # Multi-structure: sub-trajectory for structure 0
    │   ├── explore_traj_1.xyz # Multi-structure: sub-trajectory for structure 1
    │   ├── explore_traj.xyz   # Combined trajectory (all structures)
    │   └── labeled/           # This iteration’s DFT-labeled extended XYZs
    ├── iter_1/
    │   └── ...
    └── ...
```

### DFT backend parameters (summary)

**PySCF** (`--label-type pyscf`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pyscf-method` | `b3lyp` | Method: b3lyp, pbe, hf, mp2, etc. |
| `--pyscf-basis` | `sto-3g` | Basis: sto-3g, 6-31g*, def2-svp, etc. |
| `--pyscf-charge` | 0 | Total charge |
| `--pyscf-spin` | 0 | 2S (unpaired electrons) |
| `--pyscf-max-memory` | 4000 | Max memory (MB) |
| `--pyscf-conv-tol` | 1e-9 | SCF convergence threshold |

**VASP** (`--label-type vasp`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vasp-xc` | `PBE` | XC functional: PBE, LDA, HSE06, etc. |
| `--vasp-encut` | None | Plane-wave cutoff (eV) |
| `--vasp-kpts` | `1 1 1` | k-point mesh |
| `--vasp-ediff` | 1e-6 | SCF convergence (eV) |
| `--vasp-ismear` | 0 | Smearing: 0=Gaussian, -5=tetrahedron |
| `--vasp-sigma` | 0.05 | Smearing width (eV) |
| `--vasp-command` | Override ASE_VASP_COMMAND | Run command |
| `--vasp-cleanup` | False | Remove run dirs after success |

**CP2K**: `--cp2k-xc`, `--cp2k-basis-set`, `--cp2k-cutoff`, `--cp2k-max-scf`, `--cp2k-charge`, `--cp2k-command`, `--cp2k-cleanup`.  
**Quantum Espresso**: `--qe-pseudo-dir` (required), `--qe-pseudopotentials` (required, JSON), `--qe-ecutwfc`, `--qe-kpts`, `--qe-command`, `--qe-cleanup`.  
**Gaussian**: `--gaussian-method`, `--gaussian-basis`, `--gaussian-charge`, `--gaussian-mult`, `--gaussian-nproc`, `--gaussian-mem`, `--gaussian-command`, `--gaussian-cleanup`.  
**ORCA**: `--orca-simpleinput`, `--orca-nproc`, `--orca-charge`, `--orca-mult`, `--orca-command`, `--orca-cleanup`.

### Script template placeholders (local-script / slurm)

`--local-script-template` and `--slurm-template` use the same placeholders:

| Placeholder | Description |
|-------------|-------------|
| `{run_dir}` | Run directory for current structure |
| `{input_xyz}` | Input XYZ path |
| `{output_xyz}` | Output extended XYZ path (script must write here) |
| `{job_name}` | Job/task name |
| `{partition}` | SLURM partition (slurm only) |
| `{nodes}` | Node count (slurm only) |
| `{ntasks}` | Task count (slurm only) |
| `{time}` | Time limit (slurm only) |
| `{mem}` | Memory (slurm only) |

The script must produce `{output_xyz}` as extended XYZ with `Properties=species:S:1:pos:R:3:energy:R:1:forces:R:3`, energy in eV, forces in eV/Å.

### SLURM parameters (--label-type slurm)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--slurm-template` | Required | Path to job script template |
| `--slurm-partition` | `cpu` | Partition/queue name |
| `--slurm-nodes` | 1 | Nodes per job |
| `--slurm-ntasks` | 32 | Tasks per job |
| `--slurm-time` | `02:00:00` | Wall-clock limit |
| `--slurm-mem` | `64G` | Memory per job |
| `--slurm-max-concurrent` | 200 | Max concurrent jobs in queue |
| `--slurm-poll-interval` | 30 | Seconds between squeue polls |
| `--slurm-extra` | None | Extra sbatch args, e.g. `--account=myproject` |
| `--slurm-cleanup` | False | Remove run dirs after success |

For `slurm` and `local-script` labelers, if a structure’s `output.xyz` already exists
(e.g. after resume), that structure is skipped.

### Environment variables

| Variable | Description |
|----------|-------------|
| `ASE_VASP_COMMAND` | VASP run command, e.g. `mpirun -np 4 vasp_std` |
| `VASP_PP_PATH` | VASP pseudopotential directory |
| `ASE_CP2K_COMMAND` | CP2K run command |
| `CP2K_DATA_DIR` | CP2K data directory |
| `ASE_GAUSSIAN_COMMAND` | Gaussian command, e.g. `g16 < PREFIX.com > PREFIX.log` |

ORCA and QE can use `--orca-command` / `--qe-command` or have executables on PATH.

### FAQ

- **Where does the initial structure come from?** If `--init-structure` is not set, the first structure is extracted from available training data in `--data-dir`, preferring `train.xyz` and otherwise falling back to `processed_train.h5` / `read_train.h5`. You can pass multiple files or a directory for **multi-structure parallel exploration**: `--init-structure A.xyz B.xyz` or `--init-structure structures/`.
- **How do I start active learning from an existing checkpoint?** Use `--init-checkpoint`. With a single checkpoint, iteration 0 enters bootstrap mode: training is skipped, MD exploration starts immediately, and explored frames are sent directly to candidate/diversity/labeling. With `n_models` checkpoints, iteration 0 still skips training but keeps full ensemble deviation.
- **How to resume after interruption?** Re-run the command with `--resume`. The loop loads `work_dir/al_state.json` and reuses existing checkpoints, `explore_traj.xyz`, `model_devi.out`, `candidate.xyz`, `labeled.xyz`, and `merge.done`, so completed steps are not repeated. SLURM labelers still skip structures whose `output.xyz` already exists.
- **Output XYZ format?** Labeler output must be extended XYZ with `Properties=species:S:1:pos:R:3:energy:R:1:forces:R:3`, energy in eV, forces in eV/Å.
- **Training hyperparameters?** `--epochs` is passed to the internal `mff-train`; the CLI currently only exposes `--epochs` (see training section for others).

### Multi-stage JSON (--stages)

When using `--stages stages.json`, single-stage flags like `--md-*` and `--n-iterations` are ignored. Example JSON:

```json
[
  {"name": "300K", "temperature": 300, "nsteps": 500, "max_iters": 3,
   "level_f_lo": 0.05, "level_f_hi": 0.5, "conv_accuracy": 0.9},
  {"name": "600K", "temperature": 600, "nsteps": 1000, "max_iters": 3,
   "level_f_lo": 0.05, "level_f_hi": 0.5, "conv_accuracy": 0.9}
]
```

### More information

For **DFT backend options** (PySCF/VASP/CP2K/QE/Gaussian/ORCA), **script template placeholders**, **SLURM parameters**, **environment variables**, and **FAQ**, see [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md).

## Python API Usage

### Example 1: Data Preprocessing

```python
from molecular_force_field.data.preprocessing import (
    extract_data_blocks,
    fit_baseline_energies,
    compute_correction,
    save_set,
    save_to_h5_parallel
)

# Extract data blocks (supports parsing energy / pbc / Lattice / Properties)
all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs = extract_data_blocks('data.xyz')

# Fit baseline energies
keys = np.array([1, 6, 7, 8], dtype=np.int64)
fitted_values = fit_baseline_energies(
    train_blocks, train_raw_E, keys,
    initial_values=np.array([-0.01] * len(keys))
)

# Compute correction energies
train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)

# Save data (including pbc information; raw H5 uses variable-length sample_i groups by default)
save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs)

# Preprocess H5 files (precompute neighbor lists)
save_to_h5_parallel('train', max_radius=5.0, num_workers=8)
```

### Example 2: Train Model

```python
import torch
from torch.utils.data import DataLoader
from molecular_force_field.models import E3_TransformerLayer_multi, MainNet
from molecular_force_field.data import H5Dataset
from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.utils.config import ModelConfig

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = H5Dataset('train')
val_dataset = H5Dataset('val')

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn_h5,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn_h5,
    num_workers=2,
    pin_memory=True
)

# Model configuration
config = ModelConfig()

# num_interaction notes:
# - Meaning: number of interaction (convolution) layers in E3_TransformerLayer_multi and cartesian variants
# - Range: minimum 2, default 2; n layers will chain n convolutions
# - Effect: concatenates f1..fn and expands f_combine_product to (n-1)*32x0e
#
# Example: set to 3 interactions
#   num_interaction=3

# Initialize model
model = MainNet(
    input_size=config.input_dim_weight,
    hidden_sizes=config.main_hidden_sizes4,
    output_size=1
).to(device)

e3trans = E3_TransformerLayer_multi(
    max_embed_radius=config.max_radius,
    main_max_radius=config.max_radius_main,
    main_number_of_basis=config.number_of_basis_main,
    irreps_input=config.get_irreps_input_conv_main(),
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
    num_interaction=2,
    device=device
).to(device)

# Create trainer
trainer = Trainer(
    model=model,
    e3trans=e3trans,
    train_loader=train_loader,
    val_loader=val_loader,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    config=config,
    learning_rate=2e-4,
    epoch_numbers=1000,
    checkpoint_path='combined_model.pth',
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

# Start training
trainer.run_training()
```

### Example 3: Evaluate Model

```python
import torch
from torch.utils.data import DataLoader
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.data import OnTheFlyDataset
from molecular_force_field.data.collate import on_the_fly_collate
from molecular_force_field.evaluation.evaluator import Evaluator
from molecular_force_field.utils.config import ModelConfig

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# Load test dataset
test_dataset = OnTheFlyDataset(
    'read_test.h5',
    'raw_energy_test.h5',
    'cell_test.h5',
    max_radius=5.0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=on_the_fly_collate,
    num_workers=10
)

# Evaluate
evaluator = Evaluator(
    model=e3trans,
    dataset=test_dataset,
    device=device,
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

metrics = evaluator.evaluate(test_loader, output_prefix='test')
print(f"Energy RMSE: {metrics['energy_rmse']:.6f}")
print(f"Force RMSE: {metrics['force_rmse']:.6f}")
```


### Example 4: Using ASE Calculator for MD Simulation

```python
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase import units
from ase.md import MDLogger
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
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
    num_interaction=2,
    function_type_main=config.function_type,
    device=device
).to(device)

e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# Create calculator
atomic_energies_dict = {
    1: -430.53299511,
    6: -821.03326787,
    7: -1488.18856918,
    8: -2044.3509823
}
calc = MyE3NNCalculator(e3trans, atomic_energies_dict, device, max_radius=5.0)

# Read structure
atoms = read('structure.xyz')
atoms.set_calculator(calc)

# Optional: Pre-optimize structure
print("Relaxing structure...")
opt = BFGS(atoms, logfile='relax.log')
opt.run(fmax=0.05)
write('relaxed_structure.xyz', atoms)

# Initialize velocity distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Set MD parameters
timestep = 1.0  # fs
temperature = 300.0  # K
friction = 0.01
n_steps = 10000
log_interval = 10

# Create Langevin dynamics
dyn = Langevin(
    atoms, 
    timestep * units.fs, 
    temperature_K=temperature, 
    friction=friction
)

# Add logging and trajectory recording
dyn.attach(MDLogger(dyn, atoms, 'md_log.txt', header=True, mode="w"), interval=log_interval)
dyn.attach(lambda: write('md_traj.xyz', atoms, append=True), interval=log_interval)

# Run MD
print(f"Starting MD: {n_steps} steps ({n_steps * timestep / 1000:.2f} ps)")
dyn.run(n_steps)
print("MD simulation completed!")
```

### Example 5: Using ASE NEB to Calculate Reaction Barriers

```python
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch

# Load model (same as above)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# Create calculator
atomic_energies_dict = {
    1: -430.53299511,
    6: -821.03326787,
    7: -1488.18856918,
    8: -2044.3509823
}
calc = MyE3NNCalculator(e3trans, atomic_energies_dict, device, max_radius=5.0)

# Read initial and final structures
initial = read('reactant.xyz')
final = read('product.xyz')

# Optional: Pre-optimize endpoints
print("Optimizing initial structure...")
initial.set_calculator(calc)
opt_initial = BFGS(initial)
opt_initial.run(fmax=0.05)

print("Optimizing final structure...")
final.set_calculator(calc)
opt_final = BFGS(final)
opt_final.run(fmax=0.05)

# Create NEB images
n_images = 10  # Number of intermediate images
images = [initial]
images += [initial.copy() for _ in range(n_images)]
images += [final]

# Set calculator and cell
for image in images:
    if initial.cell.any():
        image.set_cell(initial.cell)
        image.set_pbc(initial.pbc)
    else:
        image.set_cell([100, 100, 100])
        image.set_pbc(False)
    image.set_calculator(calc)

# Create NEB object
neb = NEB(images, climb=True, method='improvedtangent', allow_shared_calculator=True)
neb.interpolate()

# Optimize path
optimizer = FIRE(neb, trajectory='neb.traj')

# Define logging function
def log_neb_status():
    energies = [img.get_potential_energy() for img in images]
    neb_forces = neb.get_forces()
    n_images_total = len(images)
    n_atoms = len(images[0])
    forces_reshaped = neb_forces.reshape(n_images_total, n_atoms, 3)
    intermediate_forces = forces_reshaped[1:-1]
    max_force = (intermediate_forces**2).sum(axis=2).max()**0.5
    print(
        f"NEB step {optimizer.nsteps:4d} | "
        f"E_min = {min(energies):.6f} eV | "
        f"E_max = {max(energies):.6f} eV | "
        f"Barrier = {max(energies) - energies[0]:.4f} eV | "
        f"F_max = {max_force:.4f} eV/Å"
    )

optimizer.attach(log_neb_status, interval=1)

# Run optimization
print("Starting NEB optimization...")
optimizer.run(fmax=0.05)

# Output final results
energies = [img.get_potential_energy() for img in images]
e_initial = energies[0]
e_saddle = max(energies)
e_final = energies[-1]

print("\n" + "=" * 60)
print("NEB Results:")
print(f"  Initial energy:    {e_initial:.4f} eV")
print(f"  Saddle point:      {e_saddle:.4f} eV")
print(f"  Final energy:      {e_final:.4f} eV")
print(f"  Forward barrier:   {e_saddle - e_initial:.4f} eV")
print(f"  Reverse barrier:   {e_saddle - e_final:.4f} eV")
print(f"  Reaction energy:   {e_final - e_initial:.4f} eV")
print("=" * 60)
```

### Example 5a: LAMMPS LibTorch Interface (USER-MFFTORCH, HPC Recommended)

**USER-MFFTORCH** is a custom LAMMPS package providing `pair_style mff/torch`, loading TorchScript models via LibTorch C++ API. **No Python at runtime**, suitable for HPC and production deployment.

**Model support**: `pure-cartesian-ictd` series and `spherical-save-cue` only. Element order and cutoff must match export.

**Step 1: Export core.pt** (one-time, requires Python):
```bash
mff-export-core \
  --checkpoint model.pth \
  --elements H O \
  --device cuda \
  --dtype float32 \
  --e0-csv fitted_E0.csv \
  --out core.pt
```

If the checkpoint was trained with ZBL enabled, the exported `core.pt` automatically contains the same ZBL short-range repulsion; you do not need to pass any extra `--zbl-*` flags during export.

**Checkpoint auto-restore notes:**

- `mff-export-core` now restores model-structure hyperparameters from the checkpoint by default, such as `tensor_product_mode`, `max_radius`, and `num_interaction`
- `mff-export-core` now **embeds E0 by default**; only `--no-embed-e0` disables it and exports pure network energy
- New checkpoints also store `atomic_energy_keys/atomic_energy_values`, so checkpoint E0 is usually enough; if `--e0-csv` is passed explicitly, `--e0-csv` wins
- If you pass conflicting CLI values explicitly, the **CLI wins**
- Recommended practice: only pass these structure arguments when you intentionally want to override the checkpoint configuration
- Older checkpoints without saved E0 still fall back to the legacy local `fitted_E0.csv` path
- This does not modify the checkpoint itself; it only changes parameter-resolution priority during export

**Step 2: Build LAMMPS**: Enable `PKG_KOKKOS` and `PKG_USER-MFFTORCH`. See [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md).

**Step 3: Run LAMMPS** (pure LAMMPS, no Python):
```bash
# Set LibTorch dynamic library path
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"

# Run with Kokkos GPU
lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
```

**LAMMPS input example** (`in.mfftorch`):
```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O

velocity all create 300 42
fix 1 all nve
thermo 20
run 200
```

Additional note:

- `pair_style mff/torch` has no separate `zbl` runtime keyword
- if `core.pt` was exported from a checkpoint with ZBL enabled, the short-range repulsion is already embedded in the TorchScript model
- at runtime you only need to keep the `core.pt` path and element order consistent with export

**USER-MFFTORCH notes**:

- `isolated-far-field-v1` and `isolated-far-field-v2` checkpoints export with `long_range_runtime_backend=none`; no `--export-reciprocal-source` needed. The current `v2` export includes sparse explicit far shells, an internal screened-Coulomb × learned-radial-gate kernel, and a configurable coarse tail, all embedded directly in the core forward path.
- The current `mfftorch` path is validated for 4 common combinations:
  1. No external field, no physical tensors: energy + forces only
  2. External field, no physical tensors: energy + forces with runtime `field` / `mfield` / `field6` / `field9`
  3. No external field, with physical tensors: energy + forces + tensor outputs via `compute mff/torch/phys`
  4. External field, with physical tensors: both runtime field input and tensor outputs enabled
- Runtime external field support:
  - `field v_Ex v_Ey v_Ez`: rank-1 `1o` external tensor, appropriate for electric-field-like polar vectors
  - `mfield v_Bx v_By v_Bz`: rank-1 `1e` external tensor, appropriate for magnetic-field-like axial vectors
  - If an exported `core.pt` declares both rank-1 electric and magnetic fields, specify both `field ... mfield ...` in the same `pair_style mff/torch` line
  - `field6` / `field9`: rank-2 external tensor input
  - If exported metadata says `external_tensor_irrep=1o`, use `field`; if it says `1e`, use `mfield`
  - `field6` / `field9` remain mutually exclusive with `field` / `mfield`
- Runtime multi-fidelity support:
  - If exported `core.pt` keeps runtime fidelity input, pass `fidelity 1` in `pair_style mff/torch`
  - Equal-style variable form `fidelity v_fid` is also supported
  - If `core.pt` was exported with `--export-fidelity-id`, the fidelity branch is frozen and runtime `fidelity` should not be passed
- The current fixed-schema physical quantities exposed in LAMMPS are:
  - global: `charge`, `dipole`, `polarizability`, `quadrupole`
  - per-atom: `charge_per_atom`, `dipole_per_atom`, `polarizability_per_atom`, `quadrupole_per_atom`, `born_effective_charge_per_atom`
- Missing physical heads are allowed: the corresponding mask becomes 0 and the output block is filled with 0
- Typical fixed-schema `l` compatibility:
  - `charge` / `charge_per_atom`: `l=0`
  - `dipole` / `dipole_per_atom`: `l=1`
  - `polarizability`: usually a rank-2 Cartesian tensor, typically `l=0+2`
  - `quadrupole`: usually `l=2`
  - `born_effective_charge_per_atom`: usually a full rank-2 Cartesian tensor, typically `l=0+1+2`
- Arbitrary custom physical head names are not auto-exposed to `compute mff/torch/phys` yet

Example:

```lammps
pair_style mff/torch 5.0 cuda fidelity 1
pair_coeff * * /path/to/core.pt H O
```

or

```lammps
variable fid equal 1
pair_style mff/torch 5.0 cuda fidelity v_fid
pair_coeff * * /path/to/core.pt H O
```
- The current LibTorch/LAMMPS export path assumes `channels_out == 1` for exposed physical heads
- `born_effective_charge_per_atom` is read through `compute ... atom born_effective_charge`; the full tensor has 9 Cartesian components, and individual components such as `xx` can be selected directly

**Recommended GPU test commands**:

```bash
# 1) No field, no physical tensors: energy/force only
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --elements H O --cutoff 5.0 --steps 50

# 2) External field, no physical tensors
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --elements H O \
  --field-values 0.0 0.0 0.01 --cutoff 5.0 --steps 50

# 3) No field, with physical tensors
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --dummy-phys-heads --test-phys-compute \
  --elements H O --cutoff 5.0 --steps 50

# 4) External field, with physical tensors
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --dummy-phys-heads --test-phys-compute \
  --elements H O --field-values 0.0 0.0 0.01 --cutoff 5.0 --steps 50
```

**isolated-far-field-v1 LAMMPS smoke (open-boundary nonperiodic)**:

```bash
# Single-GPU smoke: LAMMPS vs Python eager consistency
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --test-isolated-far-field \
  --elements H --dtype float32 --cutoff 5.0

# Throughput compare: baseline vs isolated-far-field-v1 (open-boundary MD)
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --test-isolated-far-field \
  --compare-throughput --elements H O --dtype float32 --cutoff 5.0 --steps 200
```

**isolated-far-field-v2 LAMMPS smoke (open-boundary nonperiodic)**:

```bash
# Single-GPU smoke: LAMMPS vs Python eager consistency
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --test-isolated-far-field-v2 \
  --elements H --dtype float32 --cutoff 5.0

# Throughput compare: baseline vs isolated-far-field-v2 (open-boundary MD)
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --test-isolated-far-field-v2 \
  --compare-throughput --elements H O --dtype float32 --cutoff 5.0 --steps 200
```

**Two-GPU / multi-GPU note**:

- If your cluster uses one GPU per MPI rank, the recommended workflow is: use the script to generate `core.pt` and `in.corept`, then launch LAMMPS with your site-standard MPI/GPU binding
- A common pattern is:

```bash
mpirun -np 2 --allow-run-as-root --bind-to none \
  bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; \
  /path/to/lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in /tmp/mff_test/in.corept'
```

**spherical-save-cue export note**: Default export uses pure PyTorch implementation (`force_naive`); `core.pt` does not depend on cuEquivariance custom ops and runs in any LibTorch environment.

See [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md) for full documentation.

### Example 5b: LAMMPS ML-IAP Unified Interface

ML-IAP unified is LAMMPS's machine learning potential interface, ~1.7x faster than fix external and supports GPU acceleration. Export checkpoint to ML-IAP format first.

**Model support**: Only these five models support ML-IAP (support `precomputed_edge_vec`): `e3nn_layers`, `e3nn_layers_channelwise`, `cue_layers_channelwise` (spherical-save-cue), `pure_cartesian_ictd_layers`, `pure_cartesian_ictd_layers_full`. Others (e.g., pure-cartesian, pure-cartesian-sparse) are not supported.

**Step 1: Export model**
```bash
python -m molecular_force_field.cli.export_mliap your_checkpoint.pth \
    --elements H O \
    --atomic-energy-keys 1 8 \
    --atomic-energy-values -13.6 -75.0 \
    --output model-mliap.pt
```

If the checkpoint was trained with ZBL enabled, the exported `model-mliap.pt` automatically inherits the same ZBL short-range repulsion; no extra `--zbl-*` flags are needed during export.

**Automatic TorchScript behavior (important)**:

- For `spherical-save-cue`, `python -m molecular_force_field.cli.export_mliap` now **automatically enables TorchScript export**, even if `--torchscript` is not specified explicitly.
- This is the default safe behavior because the plain Python pickle path is not stable for that mode.
- You can therefore use the normal export command directly; if the checkpoint mode is `spherical-save-cue`, the CLI will switch to the safe TorchScript-backed export path automatically.
- For `pure-cartesian-ictd` and `pure-cartesian-ictd-save`, you may still pass `--torchscript` explicitly when desired.
- `export_mliap` also restores model-structure hyperparameters from the checkpoint by default; if conflicting CLI arguments are passed explicitly, the **CLI wins**
- For new checkpoints, `export_mliap` also restores `atomic_energy_keys/atomic_energy_values` from the checkpoint by default; if `--atomic-energy-keys/--atomic-energy-values` are passed explicitly, the CLI wins
- Older checkpoints without saved E0 still fall back to the legacy local `fitted_E0.csv` path
- This does not change export capability itself; it simply makes the no-manual-architecture-arguments path safe by default

**Step 2: Drive LAMMPS from Python**
```python
import torch
import lammps
from lammps.mliap import activate_mliappy, load_unified

lmp = lammps.lammps()
activate_mliappy(lmp)

model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)

lmp.commands_string("""
units metal
atom_style atomic
read_data your_system.data
pair_style mliap unified model-mliap.pt 0
pair_coeff * * H O
velocity all create 300 12345
fix 1 all nve
thermo 100
run 1000
""")
lmp.close()
```

**Step 3: Pure LAMMPS input file**

Create `run.py`:
```python
import torch
import lammps
from lammps.mliap import activate_mliappy, load_unified

lmp = lammps.lammps()
activate_mliappy(lmp)
model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)
lmp.file("input.lammps")
lmp.close()
```

Likewise, ML-IAP has no extra runtime ZBL keyword. If `model-mliap.pt` was exported from a checkpoint with ZBL enabled, the short-range repulsion is already included and will be used automatically at runtime.

`input.lammps` example:
```
units metal
atom_style atomic
read_data system.data

pair_style mliap unified model-mliap.pt 0
pair_coeff * * H O

velocity all create 300 12345
fix 1 all nvt temp 300 300 0.1
thermo 100
dump 1 all xyz 100 traj.xyz
run 10000
```

Run:
```bash
export DYLD_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib"
python run.py
```

**Notes**: Element order in `pair_coeff * * H O` must match `--elements`; `pair_style mliap unified model.pt 0` trailing `0` means no ghost neighbors (use `1` for multi-layer message passing); use `units metal` (eV, Angstrom). On macOS set `DYLD_LIBRARY_PATH`. LAMMPS must be built with `PKG_ML-IAP=ON`, `MLIAP_ENABLE_PYTHON=ON`. See `molecular_force_field/docs/INSTALL_LAMMPS_PYTHON.md`.

**Kokkos GPU**: If LAMMPS is built with Kokkos+CUDA, run `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.lammps`; use `activate_mliappy_kokkos(lmp)` when driving from Python. See `LAMMPS_INTERFACE.md`.

### Example 5c: Large-Scale Multi-GPU Inference (inference_ddp)

For very large systems (e.g., 100k+ atoms), use DDP inference for multi-GPU parallel computation. **Only supports `pure-cartesian-ictd` mode**. Current version uses random graph for testing; real structures need to be wired in code.

```bash
torchrun --nproc_per_node=2 -m molecular_force_field.cli.inference_ddp \
  --checkpoint model.pth \
  --atoms 100000 \
  --forces
```

**Parameters**:
- `--atoms`: Number of atoms (for random test graph, default 50000)
- `--checkpoint`: Model checkpoint path
- `--forces`: Also compute and output forces
- `--partition`: Graph partition strategy, `modulo` (default) or `spatial`

### Example 5d: Thermal Conductivity Workflow (MLFF -> IFC2/IFC3 -> intrinsic BTE -> Callaway)

This workflow targets **crystalline systems** and does not use Green-Kubo integration. The recommended route is:

1. Use the MLFF to compute displaced-supercell forces and build `IFC2/IFC3`
2. Use `phono3py` to solve the intrinsic lattice thermal conductivity (`intrinsic BTE`)
3. Add Callaway-style engineering scattering on top of the intrinsic result for grain-size, defect, and interface studies

This has a different purpose from `mff-evaluate --phonon`:

- `mff-evaluate --phonon`: Hessian, stability, imaginary modes, and frequency sanity checks
- `thermal_transport bte`: actual thermal-transport workflow

#### Install dependencies

```bash
pip install -e ".[thermal]"
```

Or manually: `pip install phonopy phono3py spglib scipy`

#### Entry point

```bash
python -m molecular_force_field.cli.thermal_transport --help
```

Two subcommands are provided:

- `bte`: generate `IFC2/IFC3` and run intrinsic BTE
- `callaway`: add engineering scattering on top of the intrinsic `phono3py` result

#### Step 1: Run intrinsic BTE

```bash
python -m molecular_force_field.cli.thermal_transport bte \
  --checkpoint best_model.pth \
  --structure relaxed.cif \
  --supercell 4 4 4 \
  --phonon-supercell 4 4 4 \
  --mesh 16 16 16 \
  --temperatures 300 400 500 600 700 \
  --output-dir thermal_bte \
  --device cuda \
  --atomic-energy-file fitted_E0.csv
```

**What this does**:

1. Restores model hyperparameters and `tensor_product_mode` from the checkpoint
2. Reads the crystal structure and optionally relaxes it first with `--relax-fmax`
3. Uses `phono3py` to generate FC2/FC3 displaced supercells
4. Evaluates MLFF forces for all displaced supercells through the existing ASE calculator
5. Produces `fc2.hdf5` and `fc3.hdf5`
6. Runs intrinsic BTE in `RTA` by default, or iterative `LBTE` with `--lbte`

**Main outputs**:

- `fc2.hdf5`
- `fc3.hdf5`
- `fc2_forces.hdf5`
- `fc3_forces.hdf5`
- `fc2_forces.txt`
- `fc3_forces.txt`
- `thermal_workflow_metadata.json`
- `kappa-*.hdf5`

**Recommended practice**:

- Start from a structure already close to the target equilibrium phase
- Check phonon stability first with `mff-evaluate --phonon`
- Converge `--supercell`, `--phonon-supercell`, and `--mesh`
- Start with `RTA`, then move to `LBTE` once the workflow is stable

#### Step 2: Add Callaway engineering scattering on top of intrinsic BTE

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway \
  --component xx \
  --grain-size-nm 200 \
  --point-defect-coeff 1.0e-4
```

The current post-process uses `phono3py` mode conductivity and linewidths, then applies Matthiessen-style engineering scattering:

- boundary scattering from `grain_size_nm` and `specularity`
- point-defect scattering via `point_defect_coeff * omega^4`
- dislocation scattering via `dislocation_coeff * omega^2`
- interface scattering via `interface_coeff * omega`

**Outputs**:

- `<prefix>.csv`
- `<prefix>.json`
- `<prefix>_summary.json`

#### Step 3: Fit engineering scattering parameters to experiment

If experimental thermal-conductivity data already exists, fit only the extrinsic scattering terms instead of rerunning BTE:

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway_fit \
  --fit-experiment-csv exp_kappa.csv \
  --fit-component xx \
  --fit-parameters grain_size_nm,point_defect_coeff
```

The experiment CSV should contain at least:

- `temperature`
- one conductivity column such as `xx`, `yy`, `zz`, or a custom column passed by `--fit-column`

Physical interpretation:

- intrinsic BTE remains fixed
- only extrinsic scattering parameters are fitted
- the fitted parameters can then be reused to scan grain size, defect level, and interface quality

#### Recommended engineering workflow

1. Validate MLFF phonon stability near equilibrium
2. Run `bte` to obtain intrinsic `kappa(T)`
3. Fit only the Callaway extrinsic parameters to experiment
4. Reuse those parameters to scan process windows
5. Export the resulting `k(T)` curves into COMSOL, ANSYS, or your own thermal design workflow

For the standalone detailed document, see `THERMAL_TRANSPORT.md`.

### Example 6: Phonon Spectrum Calculation (Hessian and Frequencies)

```python
from ase.io import read
from ase.optimize import BFGS
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.evaluation.evaluator import Evaluator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch
import numpy as np

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
config.load_atomic_energies_from_file('fitted_E0.csv')

e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# Create calculator and evaluator
ref_energies_dict = {
    k.item(): v.item()
    for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
}
calc = MyE3NNCalculator(e3trans, ref_energies_dict, device, max_radius=5.0)

class SimpleDataset:
    def restore_force(self, x):
        return x
    def restore_energy(self, x):
        return x

dataset = SimpleDataset()
evaluator = Evaluator(
    model=e3trans,
    dataset=dataset,
    device=device,
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

# Read structure
atoms = read('structure.xyz')
atoms.calc = calc

# Optional: Optimize structure
print("Relaxing structure...")
optimizer = BFGS(atoms, logfile=None)
optimizer.run(fmax=0.01)
print(f"Relaxation completed. Final forces: max={atoms.get_forces().max():.6f} eV/Å")

# Prepare data
pos = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=device)
A = torch.tensor([atomic_numbers[symbol] for symbol in atoms.get_chemical_symbols()], 
                 dtype=torch.long, device=device)
batch_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)

# Build graph (using ASE on-the-fly neighbor list)
cell_array = atoms.cell.array
pbc_flags = atoms.pbc
if cell_array is None or np.abs(cell_array).sum() <= 1e-9:
    cell_array = np.eye(3) * 100.0
    pbc_flags = [False, False, False]

atoms_nl = atoms.copy()
atoms_nl.cell = cell_array
atoms_nl.pbc = pbc_flags

edge_src_np, edge_dst_np, edge_shifts_np = neighbor_list('ijS', atoms_nl, max_radius=5.0)
edge_src = torch.tensor(edge_src_np, dtype=torch.long, device=device)
edge_dst = torch.tensor(edge_dst_np, dtype=torch.long, device=device)
edge_shifts = torch.tensor(edge_shifts_np, dtype=torch.float64, device=device)
cell = torch.tensor(cell_array, dtype=torch.get_default_dtype(), device=device).unsqueeze(0)

# Compute Hessian matrix
print("Computing Hessian matrix...")
hessian = evaluator.compute_hessian(
    pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell
)

print(f"Hessian shape: {hessian.shape}")
print(f"Hessian range: [{hessian.min():.6f}, {hessian.max():.6f}] eV/Å²")

# Check symmetry
is_symmetric = torch.allclose(hessian, hessian.T, atol=1e-4)
print(f"Hessian symmetric: {is_symmetric}")

# Get atomic masses
masses = torch.tensor([atomic_masses[atomic_numbers[symbol]] 
                      for symbol in atoms.get_chemical_symbols()],
                     dtype=torch.get_default_dtype())

# Compute phonon spectrum
print("Computing phonon spectrum...")
frequencies = evaluator.compute_phonon_spectrum(
    hessian, masses, output_prefix='phonon'
)

print(f"\nPhonon calculation completed!")
print(f"Total modes: {len(frequencies)}")
print(f"Real modes: {np.sum(frequencies >= 0)}")
print(f"Imaginary modes: {np.sum(frequencies < 0)}")
print(f"Frequency range: [{frequencies.min():.2f}, {frequencies.max():.2f}] cm⁻¹")

# Analyze results
if np.any(frequencies < 0):
    print(f"\n⚠️  Warning: {np.sum(frequencies < 0)} imaginary frequencies detected!")
    print("   This indicates the structure may not be at a local minimum.")
    print("   Consider further optimization or checking the structure.")
else:
    print("\n✅ All frequencies are real (structure is stable).")
```

## Frequently Asked Questions

### Q: How to view all available command-line parameters?

```bash
mff-preprocess --help
mff-train --help
mff-evaluate --help
mff-export-core --help   # LAMMPS LibTorch export
mff-lammps --help        # LAMMPS fix external interface
python -m molecular_force_field.cli.export_mliap --help  # LAMMPS ML-IAP export
```

### Q: How to use LAMMPS LibTorch interface?

LAMMPS LibTorch interface (USER-MFFTORCH) loads TorchScript models via `pair_style mff/torch` in C++ with LibTorch. **No Python at runtime**, suitable for HPC and production.

**Quick steps**:
1. Export: `mff-export-core --checkpoint model.pth --elements H O --e0-csv fitted_E0.csv --out core.pt`
2. Build LAMMPS: Enable `PKG_KOKKOS`, `PKG_USER-MFFTORCH`. See [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)
3. Run: `lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch`

**Supported models**: `pure-cartesian-ictd` series, `spherical-save-cue`. See [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md) for full documentation.

### Q: How to export ML-IAP format?

ML-IAP is used for LAMMPS `pair_style mliap unified`, faster than fix external and supports Kokkos GPU:

```bash
python -m molecular_force_field.cli.export_mliap checkpoint.pth \
  --elements H O --atomic-energy-keys 1 8 --atomic-energy-values -13.6 -75.0 \
  --output model-mliap.pt
```

Supported models: `spherical`, `spherical-save`, `spherical-save-cue`, `pure-cartesian-ictd`, `pure-cartesian-ictd-save`. See [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md).

Additional notes:

- For `spherical-save-cue`, the CLI now auto-enables TorchScript export; you do not need to add `--torchscript` manually
- `pure-cartesian` and `pure-cartesian-sparse` are still not supported by `export_mliap`
- `export_mliap` restores model-structure hyperparameters from the checkpoint by default; if conflicting CLI arguments are passed explicitly, the **CLI wins**

### Q: How do I compute thermal conductivity?

The recommended crystalline workflow is:

1. `MLFF -> IFC2/IFC3`
2. `IFC2/IFC3 -> intrinsic BTE`
3. `intrinsic BTE -> Callaway engineering scattering`

Entry point:

```bash
python -m molecular_force_field.cli.thermal_transport --help
```

Typical commands:

```bash
python -m molecular_force_field.cli.thermal_transport bte \
  --checkpoint best_model.pth \
  --structure relaxed.cif \
  --supercell 4 4 4 \
  --phonon-supercell 4 4 4 \
  --mesh 16 16 16 \
  --temperatures 300 400 500 600 700 \
  --output-dir thermal_bte \
  --device cuda \
  --atomic-energy-file fitted_E0.csv
```

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway \
  --component xx \
  --grain-size-nm 200 \
  --point-defect-coeff 1.0e-4
```

See `THERMAL_TRANSPORT.md` for the detailed workflow.

### Q: How to resume training from a previous checkpoint?

The trainer will **automatically detect and load** checkpoint files. If the file specified by `--checkpoint` already exists, training state will be automatically restored.

**Automatically restored content:**
- ✅ Model weights (`e3trans_state_dict`)
- ✅ Epoch number (continues from interrupted epoch)
- ✅ Batch count (cumulative batch number)
- ✅ Loss weights `a` and `b`
- ✅ Best validation loss (for early stopping, based on `a × energy loss + b × force loss`)
- ✅ Early stopping counter (patience counter)
- ✅ Model-structure hyperparameters (priority: `explicit CLI > checkpoint.model_hyperparameters / checkpoint top-level metadata > defaults`)

**Recommended usage:**

- When resuming training, you usually only need to keep passing `--checkpoint` plus runtime/training options; most model-structure arguments can be omitted
- Only pass `--embedding-dim`, `--output-size`, `--lmax`, `--num-interaction`, `--max-radius`, `--tensor-product-mode`, `--invariant-channels`, etc. when you intentionally want to override the checkpoint config
- If CLI and checkpoint conflict, the current behavior is: **CLI overrides checkpoint**
- Very old checkpoints without `model_hyperparameters` may still require some manual arguments

**Usage example:**

```bash
# First training (interrupted at epoch 50)
mff-train \
    --data-dir data \
    --epochs 100 \
    --checkpoint my_model.pth \
    --device cuda

# After interruption, run the same command again to continue
# Will automatically continue from epoch 51
mff-train \
    --data-dir data \
    --epochs 100 \
    --checkpoint my_model.pth \
    --device cuda

# Log will show:
# ================================================================================
# Checkpoint Loaded Successfully!
#   Resuming from epoch: 51
#   Batch count: 12500
#   Loss weights: a=1.0234, b=9.7845
#   Best validation loss: 0.012345
#   Early stopping patience counter: 3/20
# ================================================================================
```

**Notes:**
1. If you do not override them explicitly, training now restores model-structure hyperparameters from the checkpoint automatically
2. Ensure using **the same data directory** and **the same device**
3. You can adjust `--epochs` to extend training (e.g., from 100 to 200)
4. **Cannot** change learning rate scheduler initial settings when resuming (will reinitialize)

### Q: How should `mff-evaluate` be used now?

The recommended default is the minimal command:

```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --test-prefix test \
    --output-prefix test \
    --use-h5
```

Notes:

- `mff-evaluate` now restores `tensor_product_mode` and model-structure hyperparameters from the checkpoint by default
- For new checkpoints, `mff-evaluate` also restores `atomic_energy_keys/atomic_energy_values` from the checkpoint by default
- If you explicitly pass conflicting options such as `--tensor-product-mode`, `--embedding-dim`, `--output-size`, or `--invariant-channels`, the **CLI wins**
- If you explicitly pass `--atomic-energy-file` or `--atomic-energy-keys/--atomic-energy-values`, those E0 inputs also override the checkpoint
- Older checkpoints without saved E0 still follow the legacy local `fitted_E0.csv` fallback path
- Therefore, only pass those structure parameters when you intentionally want to override the checkpoint configuration

**How to restart training from scratch (ignore existing checkpoint):**
```bash
# Method 1: Use a different checkpoint name
mff-train --checkpoint new_model.pth ...

# Method 2: Delete or rename old checkpoint
mv my_model.pth my_model_backup.pth
mff-train --checkpoint my_model.pth ...
```

### Q: How to adjust model hyperparameters?

You can adjust model architecture by modifying the `ModelConfig` class in `molecular_force_field/utils/config.py`, or by passing parameters directly when using the Python API.

### Q: How to adjust model hyperparameters?

You can adjust model architecture via CLI parameters:

**Atom embedding parameters:**
```bash
# Default settings (suitable for common elements like H, C, N, O)
mff-train --max-atomvalue 10 --embedding-dim 16

# Include more element types (e.g., metals)
mff-train --max-atomvalue 50 --embedding-dim 32

# Increase embedding dimension to enhance model expressiveness
mff-train --max-atomvalue 10 --embedding-dim 32
```

**Model capacity parameters (lmax and channels):**
```bash
# Default settings (lmax=2, 64 channels, gaussian basis functions)
# Generates: "64x0e + 64x1o + 64x2e"
mff-train --lmax 2 --irreps-output-conv-channels 64

# Lower maximum order (simpler systems, faster training)
# Generates: "64x0e + 64x1o"
mff-train --lmax 1 --irreps-output-conv-channels 64

# Increase maximum order (more complex systems, stronger expressiveness)
# Generates: "64x0e + 64x1o + 64x2e + 64x3o"
mff-train --lmax 3 --irreps-output-conv-channels 64

# Increase channel number (larger model capacity)
# Generates: "128x0e + 128x1o + 128x2e"
mff-train --lmax 2 --irreps-output-conv-channels 128

# Increase both lmax and channels (maximum model capacity, suitable for complex systems)
# Generates: "128x0e + 128x1o + 128x2e + 128x3o"
mff-train --lmax 3 --irreps-output-conv-channels 128

# Reduce model capacity (suitable for simple systems or memory-constrained)
# Generates: "32x0e + 32x1o"
mff-train --lmax 1 --irreps-output-conv-channels 32
```

**Radial basis function type:**
```bash
# Default: Gaussian basis functions (recommended, smooth and easy to optimize)
mff-train --function-type gaussian

# Bessel basis functions (suitable for periodic systems)
mff-train --function-type bessel

# Fourier basis functions (suitable for periodic boundary conditions)
mff-train --function-type fourier

# Cosine basis functions
mff-train --function-type cosine

# Smooth finite support basis functions
mff-train --function-type smooth_finite
```

**Notes:**
- `--max-atomvalue` must be greater than or equal to the maximum atomic number in the dataset
- `--lmax` controls the maximum order of spherical harmonics, affecting model's ability to express angular information
  - `lmax=0`: Only scalar
  - `lmax=1`: Scalar + vector
  - `lmax=2`: Scalar + vector + rank-2 tensor
  - `lmax=3`: Includes rank-3 tensor
  - Higher lmax can capture more complex geometric features but significantly increases computation
- `--function-type` selects radial basis function type:
  - `gaussian`: Most commonly used, suitable for most systems
  - `bessel`: Suitable for systems with periodicity (e.g., crystals)
  - `fourier`: Suitable for periodic boundary conditions
  - Usually the default `gaussian` is sufficient
- Increasing `--embedding-dim`, `--lmax`, and `--irreps-output-conv-channels` will significantly increase memory usage and training time
- Evaluation must use the same hyperparameter values as training
- These parameters affect model performance, recommend adjusting based on dataset size and complexity

**Complete example: Large model configuration**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --max-atomvalue 20 \
    --embedding-dim 32 \
    --lmax 3 \
    --irreps-output-conv-channels 128 \
    --function-type gaussian \
    --batch-size 4 \
    --device cuda
```



### Q: How to adjust learning rate?

You can adjust learning rate strategy during training with the following parameters:

**Basic learning rate settings:**
```bash
mff-train \
    --learning-rate 2e-4 \      # Target learning rate (value reached after warmup)
    --min-learning-rate 1e-5    # Minimum learning rate (prevents too small)
```

**Learning rate warmup:**
```bash
mff-train \
    --warmup-batches 1000 \     # First 1000 batches for warmup
    --warmup-start-ratio 0.1    # Start from 10% of target learning rate, linearly increase to 100%
```
During warmup, learning rate linearly increases from `learning_rate × warmup_start_ratio` to `learning_rate`. For example, if `--learning-rate 2e-4` and `--warmup-start-ratio 0.1`, learning rate will linearly increase from `2e-5` to `2e-4`. This helps stabilize training in early stages.

**Learning rate decay:**
```bash
mff-train \
    --lr-decay-patience 1000 \  # Check every 1000 batches, decay if no improvement
    --lr-decay-factor 0.98      # Multiply learning rate by 0.98 each time it decays
```
For example: initial learning rate 2e-4, after every 1000 batches if validation metrics don't improve, learning rate becomes 2e-4 × 0.98 = 1.96e-4, after another 1000 batches becomes 1.96e-4 × 0.98 = 1.92e-4, and so on.

**Complete examples:**
```bash
# Fast convergence settings (larger learning rate, fast warmup)
mff-train \
    --learning-rate 5e-4 \
    --min-learning-rate 1e-5 \
    --warmup-batches 500 \
    --warmup-start-ratio 0.1

# Stable training settings (smaller learning rate, longer warmup)
mff-train \
    --learning-rate 1e-4 \
    --min-learning-rate 1e-6 \
    --warmup-batches 2000 \
    --warmup-start-ratio 0.05

# Fine-tuning settings (small learning rate, slow decay)
mff-train \
    --learning-rate 1e-5 \
    --lr-decay-factor 0.99 \
    --lr-decay-patience 2000 \
    --warmup-start-ratio 0.2
```

### Q: How to adjust loss weights?

Loss function: `Total loss = a × energy loss + b × force loss`

**Initial weight settings:**
```bash
# Default settings (energy weight 1.0, force weight 10.0)
mff-train -a 1.0 -b 10.0

# More emphasis on energy (if energy prediction is poor)
mff-train -a 2.0 -b 5.0

# More emphasis on forces (if force prediction is poor)
mff-train -a 0.5 -b 20.0
```

**Automatic weight adjustment:**
```bash
# Adjust every 500 batches (more frequent)
mff-train -a 1.0 -b 10.0 --update-param 500

# Adjust every 2000 batches (more stable)
mff-train -a 1.0 -b 10.0 --update-param 2000
```

**Adjustment rate control:**
```bash
# Slow adjustment (suitable for very long training, 200k+ batches)
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.005 --weight-b-decay 0.995

# Medium adjustment (more stable, suitable for most cases)
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.01 --weight-b-decay 0.99

# Fast adjustment (suitable for short training, <50k batches)
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.02 --weight-b-decay 0.98

# Very fast adjustment (default)
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.05 --weight-b-decay 0.98
```

Automatic adjustment rule: Every `--update-param` batches, `a` is multiplied by `--weight-a-growth` (increase), `b` is multiplied by `--weight-b-decay` (decrease), to gradually balance the importance of energy and forces. **Note: `a` and `b` are clamped to [1, 1000] by default to prevent excessive drift in long training.**

**Optional: Set ranges for a / b (Clamp)** (suitable for long training to prevent excessive weight drift):
```bash
# Limit a to [0.5, 10], b to [0.001, 100]
mff-train \
  -a 1.0 -b 10.0 \
  --a-min 0.5 --a-max 10 \
  --b-min 0.001 --b-max 100
```

**Rate selection recommendations:**
- **Slow (1.005/0.995)**: Suitable for very long training (>200k batches), weight changes are gradual, training is more stable
- **Medium (1.01/0.99)**: Suitable for most training scenarios (50k-200k batches), balances stability and adjustment speed
- **Fast (1.02/0.98)**: Suitable for short training (<50k batches), fast weight adjustment, but may cause instability in later training stages
- **Default (1.05/0.98)**: More aggressive; can converge faster but may be less stable—switch to medium/slow if oscillating

### Q: How to adjust optimizer parameters?

**Gradient clipping (prevent gradient explosion):**
```bash
# Stricter gradient clipping (suitable when training is unstable)
mff-train --max-grad-norm 0.3

# More lenient gradient clipping (suitable when training is stable)
mff-train --max-grad-norm 1.0
```

**Optimizer stability parameters:**
```bash
# Clamp v_hat more frequently (suitable when training is unstable)
mff-train --vhat-clamp-interval 1000 --max-vhat-growth 3.0

# More lenient v_hat control (suitable when training is stable)
mff-train --vhat-clamp-interval 5000 --max-vhat-growth 10.0
```

**Monitor gradients:**
```bash
# Log gradient statistics every 100 batches (for debugging)
mff-train --grad-log-interval 100

# Log every 1000 batches (reduce log volume)
mff-train --grad-log-interval 1000
```

### Q: How to choose appropriate batch size?

Batch size affects training speed and memory usage:

```bash
# Small batch (suitable for small GPU memory or small datasets)
mff-train --batch-size 4

# Medium batch (default, suitable for most cases)
mff-train --batch-size 8

# Large batch (suitable when GPU memory is sufficient, can accelerate training)
mff-train --batch-size 16
```

**Note:** Batch size affects gradient estimation stability. If batch is too small (e.g., 2 or 4), may need to reduce learning rate; if batch is too large, may need to increase learning rate.

### Q: How to adjust validation and save frequency?

```bash
# More frequent validation and saving (suitable for rapid iteration)
mff-train --dump-frequency 100

# Default frequency (balance performance and monitoring)
mff-train --dump-frequency 250

# Less frequent validation (suitable for long training, reduce I/O overhead)
mff-train --dump-frequency 500
```

**Note:** `--dump-frequency` controls the frequency of validation and model saving. More frequent validation can detect problems earlier but will increase training time.

### Q: What are the data format requirements?

Input XYZ files need to contain:
- Atomic coordinates (x, y, z)
- Atom types/atomic numbers
- Forces (Fx, Fy, Fz)
- Energy information (in Properties line)
- Cell information (in Lattice attribute, optional)
- **Stress/virial** (optional, for periodic stress training): In comment line, `stress="..."` or `virial="..."` (6 or 9 components)

### Q: How to use multi-GPU parallel training?

Multi-GPU training can significantly accelerate training, especially on large datasets.

**Method 1: Use `--n-gpu` (recommended, simplest):**

```bash
# 4-GPU training, auto-launches torchrun DDP
mff-train --n-gpu 4 --data-dir data --epochs 1000 --batch-size 8

# Multi-node: 2 nodes × 4 GPUs (auto-detects SLURM)
mff-train --n-gpu 4 --nnodes 2 --data-dir data --epochs 1000
```

> `--n-gpu 1` (default): behavior is identical to the original `mff-train`.
> `--n-gpu N` (N > 1) or `--nnodes > 1`: auto-relaunches via `torchrun --nproc_per_node=N --distributed ...`.
> Manual `torchrun ... --distributed` usage is still fully supported.

| Param | Default | Description |
|-------|---------|-------------|
| `--n-gpu` | 1 | GPUs to use. >1 auto-launches torchrun DDP |
| `--nnodes` | 1 | Number of nodes. >1 multi-node DDP |
| `--master-addr` | auto | Rendezvous address |
| `--master-port` | 29500 | Rendezvous port |
| `--launcher` | auto | auto / local / slurm |

**Method 2: Manual torchrun (fully backward-compatible):**

```bash
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**Notes:**
- Effective batch size per GPU = `--batch-size`, total = `--batch-size × GPUs`
- Recommend scaling learning rate with total batch size (linear scaling)
- Logs, checkpoints, CSV files only saved on rank 0
- Validation results auto-aggregated across all ranks
- `dist.barrier()` ensures checkpoint write completion before proceeding

**Example: 4-GPU training, batch size=8 per GPU, total=32**
```bash
mff-train --n-gpu 4 --data-dir data --batch-size 8 --learning-rate 4e-3
```

### Q: How to accelerate training?

**1. Use optimized tensor product modes:**
```bash
# Pure-Cartesian-ICTD: Fewest parameters (72.1% reduction), fastest speed (CPU: up to 4.12x, GPU: up to 2.10x)
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-ictd

# Pure-Cartesian-Sparse: 29.6% parameter reduction, stable speed (CPU: 0.53x-1.39x, GPU: 0.46x-1.17x)
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-sparse

# Partial-Cartesian-Loose: Non-strict equivariant (norm product approximation), faster (CPU: 0.17x-1.37x, GPU: 0.21x-1.52x), 17.3% parameter reduction
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian-loose

# Partial-Cartesian: Strictly equivariant, 17.4% parameter reduction
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian
```

**2. Use preprocessed data:**
```bash
# Use --preprocess-h5 to precompute neighbor lists during preprocessing
mff-preprocess --input-file 2000.xyz --preprocess-h5

# Use preprocessed data during training (auto-detected)
mff-train --data-dir data
```

**3. Use GPU:**
```bash
mff-train --device cuda
```

**4. Increase parallel processing:**
```bash
# Increase number of parallel processes for data loading
mff-train --num-workers 16

# Note: num_workers will automatically adjust based on CPU cores, DataLoader uses half threads during training
```

**5. Use float32 (if precision is sufficient):**
```bash
# float32 is about 2x faster than float64, but slightly lower precision
mff-train --dtype float32
```

**6. Increase batch size (if GPU memory allows):**
```bash
# Larger batches can better utilize GPU
mff-train --batch-size 16
```

**7. Reduce validation frequency (during long training):**
```bash
# Reduce I/O overhead
mff-train --dump-frequency 500
```

**8. Use multi-GPU parallel training (most effective):**
```bash
# Use 4 GPUs for parallel training, speedup close to 4x
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --batch-size 8
```

**9. Use torch.compile during validation (optional):**
```bash
mff-train --compile-val e3trans --data-dir data
```

### Q: Which tensor product mode should I choose?

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Publishing papers/High precision requirements | `spherical` | Uses e3nn strict spherical harmonics, highest precision, standard implementation |
| Parameter optimization (maximum) | `pure-cartesian-ictd` | 72.1% parameter reduction, fastest speed (CPU: up to 4.12x, GPU: up to 2.10x), strictly equivariant |
| Parameter optimization (balanced) | `pure-cartesian-sparse` | 29.6% parameter reduction, stable speed (CPU: 0.53x-1.39x, GPU: 0.46x-1.17x), strictly equivariant |
| GPU memory constrained | `pure-cartesian-ictd` or `pure-cartesian-sparse` | 72.1% or 29.6% parameter reduction, lower memory usage |
| Strict equivariance requirements | `spherical`, `partial-cartesian`, `pure-cartesian-sparse` or `pure-cartesian-ictd` | All these modes are strictly equivariant |
| Fast experimental iteration | `pure-cartesian-ictd` | Fastest speed (CPU: up to 4.12x, GPU: up to 2.10x), strictly equivariant |
| First attempt/comparison benchmark | `spherical` | Verify model correctness, performance benchmark |
| Large-scale model deployment | `pure-cartesian-ictd` | Fewest parameters (72.1% reduction), fastest speed (CPU: up to 4.12x, GPU: up to 2.10x) |

**Switch modes:**
```bash
# Spherical mode (default)
mff-train --input-file data.xyz

# Other modes
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian-loose
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-sparse
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-ictd
```

**Note:** Training and evaluation must use the same mode!

### Q: How to use SWA and EMA?

**SWA (Stochastic Weight Averaging)** is used to fix loss weights in later training stages, avoiding weight drift:
```bash
# From epoch 100, fix a to 100, b to 10
mff-train \
    --input-file data.xyz \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 100 --swa-a 100 --swa-b 10
```

**EMA (Exponential Moving Average)** is used to maintain sliding average of model parameters, improving stability:
```bash
# Enable EMA from epoch 150, decay coefficient 0.999
mff-train \
    --input-file data.xyz \
    --ema-start-epoch 150 --ema-decay 0.999 \
    --use-ema-for-validation \
    --save-ema-model
```

**Use both SWA and EMA:**
```bash
mff-train \
    --input-file data.xyz \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 100 --swa-a 100 --swa-b 10 \
    --ema-start-epoch 150 --ema-decay 0.999 --use-ema-for-validation --save-ema-model
```

**Recommended settings:**
- SWA start time: 50%-70% of total epochs
- EMA start time: 60%-80% of total epochs
- EMA decay coefficient: 0.999 (default, larger is smoother)

**Note:** When SWA starts, it will reset early stopping counter (patience_counter) and best validation loss (best_val_loss) to adapt to new loss weights.

### Q: How to monitor training progress?

**Console output:**
- Validation results are output to console in real-time
- Summary information for each epoch is output to console

**Log files:**
- `training_YYYYMMDD_HHMMSS.log` - Contains all detailed information
  - Batch-level training metrics (every N batches, controlled by `--energy-log-frequency`)
  - Detailed results during validation
  - Gradient statistics (every N batches, controlled by `--grad-log-interval`)

**CSV files:**
- `training_YYYYMMDD_HH_loss.csv` - Training loss records
- `val_energy_epoch{epoch}_batch{batch_count}.csv` - Validation set energy predictions
- `val_force_epoch{epoch}_batch{batch_count}.csv` - Validation set force predictions

**Real-time monitoring:**
```bash
# View log file in real-time
tail -f training_*.log

# View latest validation results
tail -f training_*.log | grep "Validation"
```



## Complete Training Examples

### Example 1: Quick Test (Small Dataset, Fast Verification)

```bash
mff-train \
    --input-file small_test.xyz \
    --data-dir data \
    --epochs 100 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --warmup-batches 100 \
    --dump-frequency 50 \
    --device cuda \
    --dtype float32
```

**Use case:** Quickly verify code and parameter settings are correct

### Example 2: Standard Training (Medium Dataset, Balanced Settings)

```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 2e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --patience 20 \
    --dump-frequency 250 \
    --device cuda \
    --dtype float64
```

**Use case:** Recommended configuration for most cases

### Example 3: Large-Scale Training (Large Dataset, Long Training)

```bash
mff-train \
    --input-file large_dataset.xyz \
    --data-dir data \
    --epochs 5000 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --min-learning-rate 5e-6 \
    --warmup-batches 2000 \
    --warmup-start-ratio 0.05 \
    --lr-decay-patience 2000 \
    --lr-decay-factor 0.99 \
    -a 1.0 \
    -b 10.0 \
    --update-param 2000 \
    --patience 50 \
    --dump-frequency 500 \
    --max-grad-norm 0.5 \
    --vhat-clamp-interval 2000 \
    --device cuda \
    --dtype float64 \
    --num-workers 16
```

**Use case:** Large-scale datasets, need long training to achieve best performance

### Example 4: Fine-Tuning (Near Convergence, Small Step Adjustments)

```bash
mff-train \
    --data-dir data \
    --checkpoint combined_model.pth \
    --epochs 500 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --min-learning-rate 1e-6 \
    --warmup-batches 500 \
    --warmup-start-ratio 0.2 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.995 \
    -a 0.5 \
    -b 15.0 \
    --update-param 2000 \
    --patience 30 \
    --dump-frequency 100 \
    --device cuda \
    --dtype float64
```

**Use case:** Continue training from existing checkpoint, fine-tuning

### Example 5: CPU Training (No GPU Environment)

```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 500 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --warmup-batches 500 \
    --dump-frequency 100 \
    --device cpu \
    --dtype float32 \
    --num-workers 4
```

**Use case:** Training on machines without GPU, use float32 and smaller batch size to accelerate

## Notes

1. **Memory Usage**: When using `H5Dataset`, data will be preloaded into memory, ensure sufficient memory. If memory is insufficient, consider using `OnTheFlyDataset` (but training speed will be slower).

2. **GPU Memory**: If encountering GPU memory shortage (OOM errors), you can:
   - Reduce `--batch-size` (e.g., from 8 to 4)
   - Use `--dtype float32` instead of `float64`
   - Reduce `--max-radius` (reduce number of neighbors)

3. **Data Format**: Ensure XYZ file format is correct, especially energy and force units. Energy is usually in eV, forces in eV/Å.

4. **Checkpoints**: Checkpoints will be saved periodically during training (frequency controlled by `--dump-frequency`), recommend regular backups. Final model will be saved at path specified by `--checkpoint`.

5. **Logs**: Training logs are saved in `training_*.log` files, containing detailed training information. Console only shows validation results and important information, detailed logs please check log files.

6. **Early Stopping**: If **weighted validation loss** (`a × energy loss + b × force loss`) does not improve for `--patience` consecutive epochs, training will automatically stop. This can prevent overfitting and save time. Early stopping criterion is consistent with total loss during training.

7. **Automatic Weight Adjustment**: `a` and `b` will automatically adjust during training (every `--update-param` batches). `a` is multiplied by `--weight-a-growth` (default 1.05, i.e., 5% increase), `b` is multiplied by `--weight-b-decay` (default 0.98, i.e., 2% decrease), to balance the importance of energy and forces. If training is unstable, you can:
   - Increase `--update-param` (reduce adjustment frequency)
   - Use slower adjustment rate (`--weight-a-growth 1.005 --weight-b-decay 0.995`)
   - Fix weights (set `--update-param` to a very large value, e.g., 1000000)

8. **Learning Rate Strategy**: Learning rate goes through three stages:
   - **Warmup stage** (first `--warmup-batches` batches): Linearly increases from `learning_rate × warmup_start_ratio` to `learning_rate`
   - **Stable stage**: Maintains `learning_rate`
   - **Decay stage**: If validation metrics don't improve, multiplied by `--lr-decay-factor` after every `--lr-decay-patience` batches



## Tensor Product Mode Comparison

**FusedSCEquiTensorPot supports eight equivariant tensor product implementation modes**, each with different characteristics in speed, parameter count, and equivariance:

1. **`spherical`**: e3nn-based spherical harmonics method (default, standard implementation)
2. **`spherical-save`**: channelwise edge conv (e3nn backend, fewer params)
3. **`spherical-save-cue`**: channelwise edge conv (cuEquivariance backend, optional, GPU accelerated)
4. **`partial-cartesian`**: Cartesian coordinates + e3nn CG coefficients (strictly equivariant)
5. **`partial-cartesian-loose`**: Approximate equivariant (norm product approximation)
6. **`pure-cartesian`**: Pure Cartesian \(3^L\) representation (strictly equivariant, very slow, not recommended)
7. **`pure-cartesian-sparse`**: Sparse pure Cartesian (strictly equivariant, parameter-optimized)
8. **`pure-cartesian-ictd`**: ICTD irreps internal representation (strictly equivariant, fastest, fewest parameters)

All modes maintain O(3) equivariance (including rotation and reflection). Comparison data below:

### Mode Comparison Overview

| Feature | Spherical | Partial-Cartesian | Partial-Cartesian-Loose | Pure-Cartesian | Pure-Cartesian-Sparse | Pure-Cartesian-ICTD |
|---------|-----------|-------------------|------------------------|----------------|----------------------|---------------------|
| Implementation | e3nn spherical harmonics | Cartesian coordinates + e3nn CG coefficients | Non-strict equivariant (norm product approximation, uses e3nn Irreps) | Pure Cartesian (3^L, δ/ε, fully self-implemented) | Sparse pure Cartesian (δ/ε, fully self-implemented) | ICTD irreps internal representation (fully self-implemented) |
| Equivariance | ✅ Strictly equivariant | ✅ Strictly equivariant | ⚠️ Approximately equivariant | ✅ Strictly equivariant | ✅ Strictly equivariant | ✅ Strictly equivariant |
| Speed (CPU, lmax=2) | 1.00x (baseline) | 1.06x | 1.33x | 0.06x (very slow) | 1.39x | **4.12x (fastest)** |
| Speed (GPU, lmax=2)* | 1.00x (baseline) | 0.75x | 1.15x | 0.06x (very slow) | 1.17x | **2.10x (fastest)** |
| Parameters (lmax=2) | 6,540,634 (100%) | 5,404,938 (82.6%) | 5,406,026 (82.7%) | 33,626,186 (514.0%) | 4,606,026 (70.4%) | 1,824,497 (27.9%) |
| Parameter change | - | **17.4% reduction** | **17.3% reduction** | +414% | **29.6% reduction** | **72.1% reduction** |
| Equivariance error (O(3), lmax=2) | ~1e-15 | ~1e-14 | ~1e-15 | ~1e-14 | ~1e-15 | ~1e-7 |
| Recommended scenario | Default, highest precision | Strictly equivariant, balanced performance | Fast iteration (CPU), non-strict equivariance acceptable | Not recommended (slow) | Parameter optimization, strictly equivariant | **Fewest parameters, GPU fastest, strictly equivariant** |

*GPU speed: Total training time (forward+backward) speedup ratio, relative to spherical. Test environment: RTX 3090, float64, N=32, E=256.

### Usage

**Select mode during training:**
```bash
# Spherical mode (default, highest precision)
mff-train --input-file data.xyz --data-dir output

# Cartesian mode (fast, fewer parameters)
mff-train --input-file data.xyz --data-dir output --tensor-product-mode partial-cartesian
```

**Must use same mode during evaluation:**
```bash
# If training used cartesian
mff-evaluate --checkpoint model.pth --tensor-product-mode partial-cartesian
```

### Detailed Performance Comparison

#### CPU Test Results (channels=64, lmax=0 to 6, 32 atoms, 256 edges, float64)

**Total training time speedup ratio (forward+backward, relative to spherical):**

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             1.06x |                  1.13x |           0.36x |                 1.07x |                **2.97x** |
| 1    |             1.05x |                  1.37x |           0.13x |                 1.02x |                **3.33x** |
| 2    |             1.06x |                  1.33x |           0.06x |                 1.39x |                **4.12x** |
| 3    |             0.58x |                  0.70x |           0.02x |                 1.05x |                **2.68x** |
| 4    |             0.37x |                  0.43x |        **FAILED** |                 0.97x |                **2.20x** |
| 5    |             0.23x |                  0.28x |        **FAILED** |                 0.78x |                **1.81x** |
| 6    |             0.16x |                  0.17x |        **FAILED** |                 0.53x |                **1.58x** |

**Parameter count comparison (lmax=0 to 6):**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**Equivariance error comparison (O(3), including parity):**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 3.33e-15  |        8.24e-16  |            6.39e-14  |      3.41e-15  |           4.34e-15  |           1.69e-12 |
| 2    | 3.88e-15  |        1.87e-14  |            7.73e-16  |      3.08e-15  |           3.56e-14  |           7.73e-08 |
| 4    | 1.27e-15  |        6.83e-15  |            1.00e-15  |       **FAILED** |           1.24e-14  |           3.50e-05 |
| 6    | 3.26e-15  |        2.01e-15  |            5.82e-16  |       **FAILED** |           1.51e-15  |           1.00e-06 |

#### GPU Test Results (channels=64, lmax=0 to 6, RTX 3090, float64, N=32, E=256)

**Total training time speedup ratio (forward+backward, relative to spherical):**

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             0.96x |                  1.52x |           0.54x |                 1.17x |                **1.92x** |
| 1    |             0.85x |                  1.33x |           0.16x |                 1.02x |                **1.97x** |
| 2    |             0.75x |                  1.15x |           0.06x |                 1.17x |                **2.10x** |
| 3    |             0.56x |                  0.81x |           0.02x |                 1.15x |                **1.91x** |
| 4    |             0.38x |                  0.51x |        **FAILED** |                 0.99x |                **1.78x** |
| 5    |             0.26x |                  0.32x |        **FAILED** |                 0.75x |                **1.44x** |
| 6    |             0.17x |                  0.21x |        **FAILED** |                 0.46x |                **1.05x** |

**Parameter count comparison (lmax=0 to 6):**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**Equivariance error comparison (O(3), including parity):**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 1.03e-15  |        6.37e-16  |            1.20e-15  |      3.27e-15  |           9.31e-16  |           5.24e-08 |
| 2    | 9.27e-16  |        1.97e-16  |            8.96e-16  |      3.18e-14  |           1.30e-15  |           7.18e-07 |
| 4    | 9.16e-16  |        7.76e-14  |            4.26e-16  |       **FAILED** |           6.19e-16  |           7.16e-07 |
| 6    | 4.77e-16  |        7.02e-16  |            5.65e-16  |       **FAILED** |           5.72e-16  |           1.11e-07 |

**Performance Analysis:**
- ✅ **All modes pass O(3) equivariance tests** (including parity/reflection), equivariance error < 1e-6
- 🚀 **`pure-cartesian-ictd` is fastest on CPU at all lmax** (up to **4.12x speedup** at lmax=2)
- 🚀 **`pure-cartesian-ictd` is fastest on GPU at all lmax** (up to **2.10x speedup** at lmax=2)
- 🚀 **`pure-cartesian-sparse` performs stably on CPU/GPU** (0.53x - 1.39x, close to baseline)
- 💾 **`pure-cartesian-ictd` always has fewest parameters** (27-32% of spherical)
- 💾 **`pure-cartesian-sparse` has moderate parameters** (70-88% of spherical)
- ⚠️ **`pure-cartesian` is very slow and fails at lmax≥4** (not recommended)
- 📊 **CPU test environment**: channels=64, lmax=0-6, 32 atoms, 256 edges, float64
- 📊 **GPU test environment**: channels=64, lmax=0-6, RTX 3090, float64, 32 atoms, 256 edges



### Recommended Usage Scenarios

**Use Spherical (default):**
- ✅ Need highest precision and compatibility
- ✅ Research/publishing papers (standard e3nn implementation)
- ✅ First attempt/comparison benchmark
- ✅ Small-scale datasets

**Use Partial-Cartesian:**
- ✅ Need strict equivariance but fewer parameters (17.4% reduction)
- ✅ GPU memory constrained scenarios
- ✅ Need to balance performance and equivariance

**Use Partial-Cartesian-Loose:**
- ✅ Fast experimental iteration (faster speed, CPU: 0.17x-1.37x, GPU: 0.21x-1.52x)
- ⚠️ Scenarios where strict equivariance is not highly required (uses norm product approximation, not strictly equivariant)
- ⚠️ Note: Although equivariance error < 1e-6, theoretically not strictly equivariant

**Use Pure-Cartesian-Sparse (recommended for CPU/GPU training):**
- ✅ **Stable performance on CPU** (0.53x - 1.39x, close to baseline)
- ✅ **Faster on GPU** (**1.17x speedup** at lmax=2)
- ✅ Moderate parameters (29.6% reduction, 70-88% of spherical)
- ✅ Strictly equivariant (error ~1e-15, highest precision)
- ✅ Best choice for balancing parameters and performance

**Use Pure-Cartesian-ICTD (recommended for CPU/GPU training):**
- ✅ **Fastest on CPU** (up to **4.12x speedup** at lmax=2, 1.58x - 4.12x at all lmax)
- ✅ **Fastest on GPU** (up to **2.10x speedup** at lmax=2, 1.05x - 2.10x at all lmax)
- ✅ Fewest parameters (72.1% reduction, 27-32% of spherical)
- ✅ Strictly equivariant (error ~1e-7 to 1e-6, acceptable)
- ✅ Extremely memory-constrained scenarios
- ✅ Large-scale model deployment
- 📊 **Best CPU performance**: Most obvious advantage at lmax ≤ 3 (2.68x - 4.12x speedup)
- 📊 **Best GPU performance**: Most obvious advantage at lmax ≤ 3 (1.91x - 2.10x speedup)

### Real Task Test Results

**Dataset**: NEB (Nudged Elastic Band) data for five nitrogen oxide and carbon structure reaction paths, truncated to fmax=0.2, total 2,788 data points. Test set: 1-2 complete or incomplete data points selected for each reaction.

**Test configuration**: 64 channels, lmax=2, float64

<table>
<thead>
<tr>
<th style="text-align:center">Method</th>
<th style="text-align:center">Configuration</th>
<th style="text-align:center">Mode</th>
<th style="text-align:center">Energy RMSE<br/>(mev/atom)</th>
<th style="text-align:center">Force RMSE<br/>(mev/Å)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align:center;vertical-align:middle"><strong>MACE</strong></td>
<td style="text-align:center">Lmax=2, 64ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.13</td>
<td style="text-align:center">11.6</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 128ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.12</td>
<td style="text-align:center">11.3</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 198ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.24</td>
<td style="text-align:center">15.1</td>
</tr>
<tr>
<td rowspan="4" style="text-align:center;vertical-align:middle"><strong>FSCETP</strong></td>
<td rowspan="4" style="text-align:center;vertical-align:middle">Lmax=2, 64ch</td>
<td style="text-align:center"><strong>spherical</strong></td>
<td style="text-align:center"><strong>0.044</strong> ⭐</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center"><strong>partial-cartesian</strong></td>
<td style="text-align:center">0.045</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian-loose</td>
<td style="text-align:center">0.048</td>
<td style="text-align:center">8.4</td>
</tr>
<tr>
<td style="text-align:center">pure-cartesian-ictd</td>
<td style="text-align:center">0.046</td>
<td style="text-align:center">9.0</td>
</tr>
</tbody>
</table>

**Result Analysis**:
- **Energy accuracy comparison**: FSCETP reduces energy RMSE by 66.2% compared to MACE (64ch) (0.044 vs 0.13 mev/atom)
- **Force accuracy comparison**: FSCETP reduces force RMSE by 36.2% compared to MACE (64ch) (7.4 vs 11.6 mev/Å)
- **Best performance mode**: `spherical` and `partial-cartesian` modes achieve optimal accuracy (energy: 0.044-0.045 mev/atom, force: 7.4 mev/Å)
- **Accuracy and efficiency balance**: `pure-cartesian-ictd` maintains near-optimal accuracy (energy: 0.046 mev/atom, force: 9.0 mev/Å) while reducing parameters by 72.1% and training speed by 2.10x (GPU, lmax=2)

**Not recommended to use Pure-Cartesian:**
- ❌ Very slow (CPU: 0.02x-0.36x, GPU: 0.02x-0.54x), largest parameters (+414%)
- ❌ Fails at lmax≥4 (insufficient memory)
- ❌ Only for research purposes, not recommended for practical use

### Complete Example: Cartesian Mode + Multi-GPU + SWA + EMA

```bash
torchrun --nproc_per_node=4 -m molecular_force_field.cli.train \
    --input-file large_dataset.xyz \
    --data-dir data \
    --tensor-product-mode partial-cartesian \
    --distributed \
    --epochs 2000 \
    --batch-size 8 \
    --lmax 2 \
    --irreps-output-conv-channels 64 \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 1000 --swa-a 100 --swa-b 10 \
    --ema-start-epoch 1500 --ema-decay 0.999 --use-ema-for-validation --save-ema-model \
    --patience 50 \
    --device cuda
```

### Python API Usage

```python
from molecular_force_field.models import CartesianTransformerLayer
from molecular_force_field.utils.config import ModelConfig

config = ModelConfig(
    max_atomvalue=10,
    embedding_dim=16,
    lmax=2,
    channel_in=64,
)

# Create Cartesian model
model = CartesianTransformerLayer(
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
    function_type_main=config.function_type,
    lmax=config.lmax,
    device='cuda'
).to('cuda')

# Forward pass
energy = model(pos, Z, batch, edge_src, edge_dst, edge_shifts, cell)
forces = -torch.autograd.grad(energy.sum(), pos)[0]
```

### Cartesian Tensor Product API

If you need to use Cartesian tensor product layers separately:

```python
from molecular_force_field.models import CartesianFullyConnectedTensorProduct

# Create tensor product (similar to e3nn.o3.FullyConnectedTensorProduct)
tp = CartesianFullyConnectedTensorProduct(
    irreps_in1="64x0e + 64x1o + 64x2e",
    irreps_in2="1x0e + 1x1o + 1x2e",
    irreps_out="64x0e + 64x1o + 64x2e",
    shared_weights=True,
    internal_weights=True
)

# Forward pass
out = tp(x1, x2)

# Or use external weights
tp_ext = CartesianFullyConnectedTensorProduct(
    irreps_in1="64x0e + 64x1o + 64x2e",
    irreps_in2="1x0e + 1x1o + 1x2e",
    irreps_out="64x0e",
    shared_weights=False,
    internal_weights=False
)
weights = weight_network(edge_features)  # shape: (..., tp_ext.weight_numel)
out = tp_ext(x1, x2, weights)
```
