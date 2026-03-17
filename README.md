# FusedSCEquiTensorPot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19063206-blue)](https://doi.org/10.5281/zenodo.19063206)

**FusedSCEquiTensorPot** is an E(3)-equivariant neural potential for atomistic modeling with multiple tensor-product backends, explicit external-field conditioning, physical-tensor supervision, multi-fidelity training, and direct LAMMPS deployment.

## Overview

- **Backends**: spherical, channelwise spherical (`spherical-save-cue`), partial Cartesian, sparse Cartesian, ICTD, and strict-parity `pure-cartesian-ictd-o3`.
- **Field-aware learning**: electric field (`1o`), magnetic field (`1e`), and rank-aware tensor inputs can be embedded into equivariant message passing.
- **Physical tensor targets**: charge, dipole, polarizability, quadrupole, BEC, and magnetic moment.
- **Multi-fidelity**: graph-level fidelity conditioning, delta-learning (`delta-baseline`), per-fidelity weighting, and per-fidelity metrics.
- **Deployment**: export to `core.pt`, run through `USER-MFFTORCH` or ML-IAP, and support runtime field / fidelity control in LAMMPS.

## What This Repository Focuses On

- Multiple Cartesian and spherical equivariant trunks under one training/evaluation stack.
- External-field-aware tensor learning, including explicit parity-sensitive O(3) modeling.
- End-to-end workflow from preprocessing and training to `core.pt` export and LAMMPS runtime.
- Research-oriented extensions such as long-range prototypes, active learning, NEB, phonons, and thermal transport.

Dataset notes and conversion examples (rMD17, ANI-1x, QM7-X, SPICE, generic HDF5) are in [USAGE.md](USAGE.md#步骤-0-常用数据集与-extxyz-转换) (中文) and [USAGE_EN.md](USAGE_EN.md#step-0-common-datasets-and-extxyz-conversion) (English).

## Installation

```bash
pip install -e .
```

For a reproducible Linux CUDA setup with pinned PyTorch/cuEquivariance/PyG wheels:

```bash
bash scripts/install_pt271_cu128.sh
pip install -e .
```

Optional extras:

- `pip install -e ".[cue]"` for `spherical-save-cue`
- `pip install -e ".[pyg]"` for faster PyG scatter / neighbor ops
- `pip install -e ".[al]"` for SOAP-based active learning diversity
- `pip install -e ".[thermal]"` for thermal transport (`phono3py`, `scipy`)

## Quick Start

### 1. Preprocess Data

```bash
mff-preprocess --input-file data.xyz --output-dir data
```

To skip neighbor list preprocessing (for quick sanity-check):

```bash
mff-preprocess --input-file data.xyz --output-dir data --skip-h5
```

If your `extxyz` uses custom field names, you can override them explicitly:

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

### 2. Train

Minimal training:

```bash
mff --train --data-dir data --epochs 1000 --batch-size 8 --device cuda
```

Recommended backbone examples:

```bash
# Memory-efficient ICTD
mff --train --data-dir data --device cuda --tensor-product-mode pure-cartesian-ictd

# Full-parity O(3) ICTD
mff --train --data-dir data --device cuda --tensor-product-mode pure-cartesian-ictd-o3

# Sparse Cartesian
mff --train --data-dir data --device cuda --tensor-product-mode pure-cartesian-sparse
```

Field-aware training examples:

```bash
# Electric field + dipole/polarizability
mff --train --data-dir data --tensor-product-mode pure-cartesian-sparse \
  --external-tensor-rank 1 --external-field-file data/efield.npy \
  --physical-tensors dipole,polarizability \
  --dipole-file data/dipole.npy --polarizability-file data/pol.npy \
  --physical-tensor-weights "dipole:2.0,polarizability:1.0"

# Magnetic field (1e) + magnetic moment
mff --train --data-dir data --tensor-product-mode pure-cartesian-ictd-o3 \
  --external-tensor-rank 1 --external-tensor-irrep 1e \
  --o3-irrep-preset auto \
  --o3-active-irreps '0e,1e,2e' \
  --external-field-file data/bfield.npy \
  --physical-tensors magnetic_moment \
  --magnetic-moment-file data/magnetic_moment.npy

# Simultaneous electric field (1o) + magnetic field (1e)
mff --train --data-dir data --tensor-product-mode pure-cartesian-ictd-o3 \
  --external-tensor-rank 1 --external-tensor-irrep 1o \
  --external-field-file data/efield.npy \
  --magnetic-field-file data/bfield.npy \
  --o3-irrep-preset auto \
  --o3-active-irreps '0e,1e,1o,2e'

```

Optional ZBL short-range repulsion:

```bash
mff --train --data-dir data --tensor-product-mode pure-cartesian-ictd \
  --zbl-enabled \
  --zbl-inner-cutoff 0.6 \
  --zbl-outer-cutoff 1.2 \
  --zbl-exponent 0.23 \
  --zbl-energy-scale 1.0
```

## Multi-Fidelity

Supported modes:
- `spherical-save-cue`
- `pure-cartesian-ictd`
- `pure-cartesian-ictd-o3`
- `pure-cartesian-sparse`
- `pure-cartesian-sparse-save`

Conditioning-only multi-fidelity:

```bash
mff --train \
  --data-dir data \
  --tensor-product-mode pure-cartesian-ictd \
  --num-fidelity-levels 2 \
  --fidelity-id-file data/train_fidelity_id.npy \
  --fidelity-loss-weights '0:1.0,1:3.0'
```

Delta-learning multi-fidelity:

```bash
mff --train \
  --data-dir data \
  --tensor-product-mode pure-cartesian-ictd-o3 \
  --num-fidelity-levels 2 \
  --multi-fidelity-mode delta-baseline \
  --fidelity-id-file data/train_fidelity_id.npy \
  --fidelity-loss-weights '0:1.0,1:3.0' \
  --delta-regularization-weight 1e-4
```

Merge multiple processed HDF5 files into one multi-fidelity dataset:

```bash
mff --merge-multifidelity \
  --inputs data/processed_pbe.h5 data/processed_hse.h5 \
  --fidelity-ids 0 1 \
  --output-h5 data/processed_train_mf.h5 \
  --output-fidelity-npy data/train_fidelity_id.npy
```

Train with LES-style long-range (`mesh_fft`, recommended first-stage settings):

```bash
# 3D periodic reciprocal long-range
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-reciprocal-backend mesh_fft \
  --long-range-boundary periodic \
  --long-range-mesh-size 16 \
  --long-range-green-mode poisson \
  --long-range-energy-partition potential \
  --long-range-assignment cic

# Slab reciprocal long-range: x/y periodic + z vacuum padding
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

Notes:
- Supported training architectures: `pure-cartesian-ictd`, `spherical-save-cue`
- Recommended first use: keep `--long-range-green-mode poisson`
- `ASE` active learning now supports the same `periodic` / `slab` boundary semantics for the Python calculator path

By default, dynamic loss weights `a/b` are clamped to `[1, 1000]` (they change during training). You can override the range:

```bash
mff-train --data-dir data --a 10.0 --b 100.0 --update-param 750 --weight-a-growth 1.05 --weight-b-decay 0.98 --a-max 1000 --b-min 1 --b-max 1000 
```

Optional: override baseline atomic energies (E0):

```bash
# from CSV (Atom,E0)
mff-train --data-dir data --atomic-energy-file data/fitted_E0.csv

# or directly from CLI
mff-train --data-dir data --atomic-energy-keys 1 6 7 8 --atomic-energy-values -430.53 -821.03 -1488.19 -2044.35
```

### 3. Evaluation

Evaluate a trained model. The recommended default is to let `mff-evaluate` restore model-structure hyperparameters and `tensor_product_mode` from the checkpoint automatically:

```bash
mff-evaluate --checkpoint combined_model.pth --test-prefix test --output-prefix test --use-h5
```

If you explicitly pass conflicting structure arguments such as `--tensor-product-mode`, `--embedding-dim`, `--output-size`, or `--invariant-channels`, the CLI takes precedence over the checkpoint. For new checkpoints, `mff-evaluate` can also restore `atomic_energy_keys/atomic_energy_values` directly from the checkpoint; older checkpoints still fall back to local `fitted_E0.csv` behavior. Only pass those arguments when you intentionally want to override the checkpoint configuration.

Outputs include:
- `test_loss.csv`
- `test_energy.csv`
- `test_force.csv`

Optional: use `--compile e3trans` to accelerate evaluation with `torch.compile`.

For molecular dynamics simulation:

```bash
mff-evaluate --checkpoint combined_model.pth --md-sim
```

For NEB (Nudged Elastic Band) calculations:

```bash
mff-evaluate --checkpoint combined_model.pth --neb
```

For phonon spectrum (Hessian, vibrational frequencies):

```bash
mff-evaluate --checkpoint combined_model.pth --phonon --phonon-input structure.xyz
```

Optional: stress training (PBC with stress/virial in XYZ):

```bash
mff-train --data-dir data -c 0.1 --input-file pbc_with_stress.xyz
```

### 4. Active Learning (Optional) 🔄

> **Grow your training set automatically** where the potential is under-sampled: one CLI runs the full **train → explore → select → label (DFT) → merge** loop. Works on a single machine (PySCF, VASP, …) or on HPC (SLURM, one job per structure).

```bash
# Local: PySCF, 8 parallel workers
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 8 --md-steps 500 --n-iterations 5

# HPC: SLURM, one job per structure
mff-active-learn --explore-type ase --label-type slurm \
    --slurm-template dft_job.sh --slurm-partition cpu \
    --slurm-nodes 1 --slurm-ntasks 32 --slurm-time 04:00:00
```

📖 **Full CLI & options**: [USAGE.md](USAGE.md#主动学习-mff-active-learn) (中文) · [USAGE_EN.md](USAGE_EN.md#active-learning-mff-active-learn) (English) · [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md) (backends, multi-stage, FAQ).

Long-range aware active learning is also supported through the ASE calculator path. In practice, use checkpoints trained with:

```bash
--long-range-mode reciprocal-spectral-v1 \
--long-range-reciprocal-backend mesh_fft \
--long-range-green-mode poisson \
--long-range-energy-partition potential \
--long-range-assignment cic
```

For slab systems, additionally set:

```bash
--long-range-boundary slab \
--long-range-slab-padding-factor 2
```

## LAMMPS Integration

FusedSCEquiTensorPot supports three LAMMPS integration methods:

| Method | Speed | Requirements | Use Case |
|--------|-------|---------------|----------|
| **USER-MFFTORCH (LibTorch pure C++)** | Fastest, no Python/GIL | LAMMPS built with KOKKOS + USER-MFFTORCH | HPC, clusters, production |
| **ML-IAP unified** | Faster (~1.7x vs fix external) | LAMMPS built with ML-IAP | Recommended, GPU support |
| **fix external / pair_style python** | Slower | Standard LAMMPS + Python | Quick validation, no ML-IAP |

### LibTorch Interface (USER-MFFTORCH, HPC Recommended)

**USER-MFFTORCH** loads TorchScript models via LibTorch C++ API directly. **No Python at runtime**, suitable for HPC and production deployment.

1. **Export core.pt** (one-time, requires Python):
   ```bash
   mff-export-core --checkpoint model.pth --elements H O --device cuda \
     --e0-csv fitted_E0.csv --out core.pt
   ```

   `mff-export-core` restores structure hyperparameters such as `tensor_product_mode`, `max_radius`, and `num_interaction` from the checkpoint by default. It now embeds E0 by default as well. New checkpoints store `atomic_energy_keys/atomic_energy_values`, so checkpoint E0 is usually enough; if `--e0-csv` is passed explicitly, the CLI wins. Older checkpoints still fall back to local `fitted_E0.csv`. Use `--no-embed-e0` only if you explicitly want to export network energy without E0.
   If the checkpoint was trained with ZBL enabled, the exported `core.pt` will include the same short-range ZBL correction automatically.

2. **Build LAMMPS**: Enable `PKG_KOKKOS` and `PKG_USER-MFFTORCH`. See [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md).

3. **Run** (pure LAMMPS, no Python):
   ```bash
   lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
   ```

**LAMMPS input example**:
```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O
```

If `core.pt` came from a checkpoint with ZBL enabled, no extra LAMMPS keyword is needed: the ZBL short-range repulsion is already embedded in the exported TorchScript model.

For checkpoints exported with external-field architecture, USER-MFFTORCH supports runtime rank-1 external tensors and follows the exported irrep semantics:

```lammps
variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01
pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez
pair_coeff * * /path/to/core.pt H O
```

For magnetic-field-style `1e` checkpoints, use `mfield` instead of `field`:

```lammps
variable Bx equal 0.0
variable By equal 0.0
variable Bz equal 0.01
pair_style mff/torch 5.0 cuda mfield v_Bx v_By v_Bz
pair_coeff * * /path/to/core.pt H O
```

The rank-1 variables are re-evaluated on each force call, so time-dependent equal-style variables are supported. `field` is the runtime keyword for `1o`-style vectors such as electric field; `mfield` is the runtime keyword for `1e`-style axial vectors such as magnetic field. Current limitation: runtime external tensors are implemented for rank-1 and rank-2.

When a `core.pt` is exported with simultaneous rank-1 electric and magnetic fields, provide both keywords in the same `pair_style` line:

```lammps
variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01
variable Bx equal 0.0
variable By equal 0.0
variable Bz equal 0.02
pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez mfield v_Bx v_By v_Bz
pair_coeff * * /path/to/core.pt H O
```

For multi-fidelity `core.pt`, runtime fidelity is passed through `pair_style mff/torch fidelity ...`:

```lammps
pair_style mff/torch 5.0 cuda fidelity 1
pair_coeff * * /path/to/core.pt H O
```

or with an equal-style variable:

```lammps
variable fid equal 1
pair_style mff/torch 5.0 cuda fidelity v_fid
pair_coeff * * /path/to/core.pt H O
```

If `core.pt` was exported with `--export-fidelity-id`, the fidelity branch is frozen during export and you should not pass `fidelity` at runtime.

`field6` / `field9` remain mutually exclusive with `field` / `mfield`.

For rank-2 runtime external tensors, USER-MFFTORCH supports both:
- `field9`: full `3x3` tensor in row-major order `xx xy xz yx yy yz zx zy zz`
- `field6`: symmetric `3x3` shorthand in order `xx yy zz xy xz yz`

Example:

```lammps
variable Txx equal 1.0
variable Txy equal 0.0
variable Txz equal 0.0
variable Tyx equal 0.0
variable Tyy equal 1.0
variable Tyz equal 0.0
variable Tzx equal 0.0
variable Tzy equal 0.0
variable Tzz equal 1.0
pair_style mff/torch 5.0 cuda field9 v_Txx v_Txy v_Txz v_Tyx v_Tyy v_Tyz v_Tzx v_Tzy v_Tzz
pair_coeff * * /path/to/core.pt H O
```

**Model support**: `pure-cartesian-ictd` series and `spherical-save-cue` only.

### ML-IAP Interface

Export ML-IAP format (requires LAMMPS built with ML-IAP):

```bash
python -m molecular_force_field.cli.export_mliap checkpoint.pth --elements H O \
  --atomic-energy-keys 1 8 --atomic-energy-values -13.6 -75.0 --output model-mliap.pt
```

Supported models: `spherical`, `spherical-save`, `spherical-save-cue`, `pure-cartesian-ictd`, `pure-cartesian-ictd-save`.

Notes:

- `spherical-save-cue` is automatically exported through the TorchScript path in `export_mliap`, even if `--torchscript` is not specified explicitly. This is now the default safe behavior because the plain Python pickle path is not stable for this mode.
- `pure-cartesian` and `pure-cartesian-sparse` are still not supported by `export_mliap`.
- `export_mliap` also restores structure hyperparameters from the checkpoint by default. If conflicting CLI values are passed explicitly, the CLI wins.
- For new checkpoints, `export_mliap` can also restore `atomic_energy_keys/atomic_energy_values` directly from the checkpoint. Older checkpoints still fall back to local `fitted_E0.csv`.
- If the checkpoint was trained with ZBL enabled, the exported `model-mliap.pt` carries the same ZBL correction automatically.

### Thermal Transport Workflow

For crystalline systems, the recommended thermal-conductivity route is:

1. `MLFF -> IFC2/IFC3`
2. `IFC2/IFC3 -> intrinsic lattice thermal conductivity` via `phono3py`
3. `intrinsic BTE -> engineering scattering / fast generalization` via a Callaway-style post-process

This workflow is intentionally separate from `mff-evaluate --phonon`. The phonon mode is useful for Hessian and stability checks, while the thermal workflow is meant for actual transport calculations.

Install thermal deps: `pip install -e ".[thermal]"`

Minimal intrinsic BTE example:

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

Minimal Callaway post-process example:

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway \
  --component xx \
  --grain-size-nm 200 \
  --point-defect-coeff 1.0e-4
```

Outputs include `fc2.hdf5`, `fc3.hdf5`, `kappa-*.hdf5`, and Callaway CSV/JSON summaries.

For the detailed workflow, fitting strategy, and engineering notes, see [`THERMAL_TRANSPORT.md`](THERMAL_TRANSPORT.md).

See [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md) for full documentation.

## Project Structure

```
rebuild/
├── molecular_force_field/              # Main package
│   ├── models/                          # Model definitions (eight tensor product modes)
│   │   ├── e3nn_layers.py              # Spherical mode (e3nn-based)
│   │   ├── e3nn_layers_channelwise.py  # spherical-save
│   │   ├── cartesian_e3_layers.py      # partial-cartesian, partial-cartesian-loose
│   │   ├── pure_cartesian*.py          # pure-cartesian, pure-cartesian-sparse
│   │   ├── pure_cartesian_ictd*.py      # pure-cartesian-ictd
│   │   ├── cue_layers*.py              # spherical-save-cue (cuEquivariance)
│   │   ├── mlp.py, losses.py
│   │   └── ...
│   ├── data/                            # Dataset and preprocessing
│   │   ├── datasets.py, preprocessing.py, collate.py
│   │   └── ...
│   ├── utils/                           # Configuration, graph utilities
│   │   ├── config.py, graph_utils.py, scatter.py, checkpoint_metadata.py
│   │   └── ...
│   ├── training/                        # Trainer
│   │   ├── trainer.py, schedulers.py
│   │   └── ...
│   ├── evaluation/                      # Evaluator, ASE Calculator
│   │   ├── evaluator.py, calculator.py
│   │   └── ...
│   ├── active_learning/                 # Active learning loop
│   │   ├── loop.py                     # Main AL loop (train → explore → select → label → merge)
│   │   ├── train_ensemble.py           # Multi-model parallel training (DDP, cross-node)
│   │   ├── labeling.py                 # DFT labelers (PySCF, VASP, script, SLURM, ...)
│   │   ├── diversity_selector.py       # SOAP / devi_hist + FPS
│   │   ├── exploration.py, model_devi.py, data_merge.py, stage_scheduler.py
│   │   ├── init_data.py                # Cold-start perturbation
│   │   └── ...
│   ├── thermal/                         # Thermal transport (IFC2/IFC3, BTE, Callaway)
│   │   ├── model_loader.py, callaway.py
│   │   └── ...
│   ├── interfaces/                      # LAMMPS potential, ML-IAP
│   │   ├── lammps_potential.py         # fix external / pair_style python
│   │   └── lammps_mliap.py             # ML-IAP unified
│   └── cli/                             # Command-line interfaces
│       ├── train.py                     # mff-train (supports --n-gpu, --nnodes)
│       ├── preprocess.py                # mff-preprocess
│       ├── evaluate.py                  # mff-evaluate (static/MD/NEB/phonon)
│       ├── active_learning.py           # mff-active-learn
│       ├── init_data.py                 # mff-init-data (cold-start)
│       ├── lammps_interface.py          # mff-lammps (fix external)
│       ├── export_libtorch_core.py      # mff-export-core
│       ├── export_mliap.py              # ML-IAP export
│       ├── inference_ddp.py             # Large-scale multi-GPU inference
│       ├── thermal_transport.py         # IFC2/IFC3, BTE, Callaway
│       └── evaluate_pes_coverage.py     # PES coverage (SOAP)
├── lammps_user_mfftorch/                # LAMMPS LibTorch package (USER-MFFTORCH)
│   ├── src/USER-MFFTORCH/               # pair_style mff/torch source
│   └── docs/BUILD_AND_RUN.md            # Build and run guide
├── scripts/                             # Install scripts, smoke tests
├── test/                                # Unit tests, benchmarks
└── docs/                                # Additional docs (LAMMPS, thermal)
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- e3nn >= 0.5.0
- ASE >= 3.22.0
- See `requirements.txt` for full list

## 🎯 Choosing Tensor Product Mode

The library supports **eight equivariant tensor product modes**, each optimized for different use cases:

1. **`spherical`**: e3nn-based spherical harmonics (default, standard implementation)
2. **`spherical-save`**: channelwise edge conv (e3nn backend; fewer params)
3. **`spherical-save-cue`**: channelwise edge conv (cuEquivariance backend; optional, GPU accelerated)
4. **`partial-cartesian`**: Cartesian coordinates + CG coefficients (strictly equivariant)
5. **`partial-cartesian-loose`**: Approximate equivariant (norm product approximation)
6. **`pure-cartesian`**: Pure Cartesian \(3^L\) representation (strictly equivariant, very slow)
7. **`pure-cartesian-sparse`**: Sparse pure Cartesian (strictly equivariant, parameter-optimized)
8. **`pure-cartesian-ictd`**: ICTD irreps internal representation (strictly equivariant, fastest, fewest parameters)

All modes maintain O(3) equivariance (including rotation and reflection). Performance comparison:

| Mode | Equivariance | Speed (CPU) l=0-6 | Speed (GPU) common configuration | Parameters | Equivariance Error | Use Case |
|------|--------------|-------------------|----------------------------------|------------|---------------------|----------|
| `spherical` | Strict | 1.00x (baseline) | 1.00x (baseline) | 100% (baseline) | ~1e-15 | Default, maximum compatibility, research/publication |
| `spherical-save-cue` | Strict | - | 16x | 32.6% (-67.4%) | ~1e-15 | Different GCN structure, design for highest speed MD. **CAN NOT compare with other modes** |
| `partial-cartesian` | Strict | 0.16x-1.06x | 0.75x | 82.6% (-17.4%) | ~1e-14 | Strict equivariance with fewer parameters |
| `partial-cartesian-loose` | Approximate | 0.17x-1.37x | 1.15x | 82.7% (-17.3%) | ~1e-15 | Fast iteration, approximate equivariance acceptable |
| `pure-cartesian-sparse` | SO3 Strict | 0.53x-1.39x | 1.17x | 70.4% (-29.6%) | ~1e-15 | Best balance: fewer params, stable performance |
| `pure-cartesian-ictd` | Strict | **1.58x-4.12x (fastest)** | **5.0x** | **27.9% (-72.1%)** | ~1e-12 | **Best overall**: fewest params, fastest on CPU/GPU, strictly equivariant |
| `pure-cartesian` | Strict | 0.02x-0.36x (slowest) | 0.06x | 514.0% (+414%) | ~1e-14 | ❌ Not recommended (too slow, too many params) |

*CPU benchmark: channels=64, lmax=0-6, 32 atoms, 256 edges, float64. Speed shown is total training time (forward+backward) acceleration ratio relative to spherical.  
*GPU benchmark: channels=64, lmax=0-6, 32 atoms, 256 edges, RTX 3090, float64. Speed shown is total training time (forward+backward) acceleration ratio relative to spherical.  
*`spherical-save-cue` uses a different GCN structure and cannot be directly compared with other modes.  
All modes pass O(3) equivariance tests (including parity/reflection, error < 1e-6).

### Quick Recommendations

#### CPU Environment (Recommended)
- **Speed + Memory**: Use `pure-cartesian-ictd` (**1.58x-4.12x faster**, 72.1% fewer parameters, all lmax)
- **High Precision**: Use `spherical` or `pure-cartesian-sparse` (equivariance error ~1e-15)
- **Best Balance**: Use `pure-cartesian-sparse` (0.53x-1.39x, 29.6% fewer params, strict equivariance)
- **Standard Baseline**: Use `spherical` (highest precision, standard implementation)

#### GPU Environment (Recommended for Training)
- **Speed + Memory**: Use `pure-cartesian-ictd` (**5.0x faster**, 72.1% fewer parameters, lmax≤3)
- **High Precision**: Use `spherical` or `pure-cartesian-sparse` (equivariance error ~1e-15)
- **Best Balance**: Use `pure-cartesian-sparse` (**1.17x faster**, 29.6% fewer params, strict equivariance)
- **Avoid**: `pure-cartesian` (too slow, fails at lmax≥4)

For detailed performance comparison and recommendations, see [USAGE.md](USAGE.md#tensor-product-mode-comparison).

### Real-World Task Performance

**Dataset**: Five nitrogen oxide and carbon structure reaction pathways from NEB (Nudged Elastic Band) calculations, filtered to fmax=0.2, totaling 2,788 structures. Test set: 1-2 complete or incomplete structures per reaction.

<table>
<thead>
<tr>
<th style="text-align:center">Model</th>
<th style="text-align:center">Configuration</th>
<th style="text-align:center">Mode</th>
<th style="text-align:center">Energy RMSE<br/>(mev/atom)</th>
<th style="text-align:center">Force RMSE<br/>(mev/Å)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align:center;vertical-align:middle"><strong>MACE correction=3</strong></td>
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
<td style="text-align:center">0.12</td>
<td style="text-align:center">15.1</td>
</tr>
<tr>
<td rowspan="6" style="text-align:center;vertical-align:middle"><strong>FSCETP</strong></td>
<td rowspan="6" style="text-align:center;vertical-align:middle">Lmax=2, 64ch</td>
<td style="text-align:center">spherical</td>
<td style="text-align:center">0.044</td>
<td style="text-align:center">7.4</td>
</tr>
<tr>
<td style="text-align:center">spherical-save-cue</td>
<td style="text-align:center">0.076</td>
<td style="text-align:center">8.0</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian</td>
<td style="text-align:center">0.045</td>
<td style="text-align:center">7.4</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian-loose</td>
<td style="text-align:center">0.048</td>
<td style="text-align:center">8.4</td>
</tr>
<tr>
<td style="text-align:center"><strong>pure-cartesian-sparse</strong></td>
<td style="text-align:center"><strong>0.044</strong> ⭐</td>
<td style="text-align:center"><strong>6.5</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center">pure-cartesian-ictd</td>
<td style="text-align:center">0.046</td>
<td style="text-align:center">9.0</td>
</tr>
</tbody>
</table>

**Key Findings**:
- **Energy Accuracy**: FSCETP achieves **66.2% lower** energy RMSE than MACE correction=3 (64ch) (0.044 vs 0.13 mev/atom)
- **Force Accuracy**: FSCETP achieves **43.9% lower** force RMSE than MACE correction=3 (64ch) (6.5 vs 11.6 mev/Å) with `pure-cartesian-sparse`
- **Best Performance**: `pure-cartesian-sparse` achieves the best force RMSE (6.5 mev/Å) with competitive energy (0.044 mev/atom)
- **Efficiency**: `pure-cartesian-ictd` achieves competitive accuracy (Energy: 0.046, Force: 9.0) with **72.1% fewer parameters** and **5.0x faster** training speed

## 📚 Documentation

- [USAGE.md](USAGE.md) - Full CLI and hyperparameter reference (Chinese)
- [USAGE_EN.md](USAGE_EN.md) - Full CLI and hyperparameter reference (English)
- [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md) - LAMMPS integration guide (LibTorch, ML-IAP, fix external)
- [THERMAL_TRANSPORT.md](THERMAL_TRANSPORT.md) - MLFF thermal conductivity workflow (`IFC2/IFC3 -> BTE -> Callaway`)
- [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md) - LibTorch interface build and run


## 📄 License

MIT License

## 🙏 Acknowledgments

This framework implements **eight equivariant tensor product modes**:
- **`spherical` and `spherical-save` modes**: Built on [e3nn](https://github.com/e3nn/e3nn) for spherical harmonics-based tensor products
- **`spherical-save-cue` mode**: Uses [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) for GPU-accelerated channelwise spherical convolution
- **`partial-cartesian` and `partial-cartesian-loose` modes**: Partially use e3nn's Clebsch-Gordan coefficients and Irreps framework
- **Self-implemented Cartesian modes**: `pure-cartesian`, `pure-cartesian-sparse`, `pure-cartesian-ictd` are independently implemented without e3nn dependencies

Other dependencies and inspirations:
- Uses [ASE](https://wiki.fysik.dtu.dk/ase/) for molecular simulations
- Inspired by NequIP, MACE, and other equivariant neural potentials

## Citation

If you use this library in your research, please cite:

```bibtex
@software{fused_sc_equitensorpot,
  title = {FusedSCEquiTensorPot},
  version = {0.1.0},
  url = {https://github.com/Parity-LRX/FusedSCEquiTensorPot}
}
```
