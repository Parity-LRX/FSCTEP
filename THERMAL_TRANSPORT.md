# Thermal Transport Workflow

This document adds a separate thermal-transport workflow on top of the existing MLFF codebase without changing training, model definitions, or LAMMPS production paths.

The intended pipeline is:

1. `MLFF -> IFC2/IFC3`
2. `IFC2/IFC3 -> intrinsic lattice thermal conductivity` with `phono3py`
3. `intrinsic BTE -> engineering scattering / fast generalization` with a Callaway-style post-process

## Scope

This workflow is meant for crystalline systems where phonon transport is still the right language.

It does **not** use Green-Kubo integration. That keeps the cost much lower than long MD for every temperature and microstructure point, and it matches the goal of using the MLFF as a force-constant engine.

## Installation

The thermal workflow is opt-in. Install the thermal extra to get phono3py, phonopy, spglib, and scipy:

```bash
pip install -e ".[thermal]"
```

Or install the phonon stack manually:

```bash
pip install phonopy phono3py spglib scipy
```

## Entry Point

Run the new CLI directly:

```bash
python -m molecular_force_field.cli.thermal_transport --help
```

Subcommands:

- `bte`: generate IFC2/IFC3 from the MLFF and run intrinsic BTE
- `callaway`: add engineering scattering on top of the intrinsic `phono3py` result

## 1. Intrinsic BTE From MLFF

Minimal example:

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

What this does:

1. Loads the existing MLFF checkpoint in inference mode
2. Optionally relaxes the input structure if `--relax-fmax` is given
3. Uses `phono3py` to generate displaced supercells
4. Evaluates MLFF forces for every displaced supercell through the existing ASE calculator
5. Produces `fc2.hdf5` and `fc3.hdf5`
6. Runs intrinsic BTE in `RTA` by default, or `LBTE` with `--lbte`

Outputs in `--output-dir`:

- `fc2.hdf5`
- `fc3.hdf5`
- `fc2_forces.hdf5`
- `fc3_forces.hdf5`
- `fc2_forces.txt`
- `fc3_forces.txt`
- `thermal_workflow_metadata.json`
- `kappa-*.hdf5` written by `phono3py`

Recommended practice:

- Use a structure already relaxed near the target phase
- Converge `--supercell`, `--phonon-supercell`, and `--mesh`
- Use `--lbte` only after the `RTA` workflow is stable
- Validate phonons first with your existing `mff-evaluate --phonon`

## 2. Engineering Scattering With Callaway

Once `phono3py` has produced a `kappa-*.hdf5`, add grain-boundary or defect scattering with:

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway \
  --component xx \
  --grain-size-nm 200 \
  --point-defect-coeff 1.0e-4
```

The current post-process uses the intrinsic `phono3py` mode conductivity and linewidths, then applies Matthiessen-type engineering scattering:

- boundary scattering from `grain_size_nm` and `specularity`
- point-defect scattering via `point_defect_coeff * omega^4`
- dislocation scattering via `dislocation_coeff * omega^2`
- interface scattering via `interface_coeff * omega`

This keeps the intrinsic anharmonic physics in the BTE result, while making microstructure scans cheap.

Outputs:

- `<prefix>.csv`
- `<prefix>.json`
- `<prefix>_summary.json`

## 3. Fit Engineering Parameters To Experiment

If you have experimental conductivity data, fit the extrinsic parameters instead of re-running BTE:

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway_fit \
  --fit-experiment-csv exp_kappa.csv \
  --fit-component xx \
  --fit-parameters grain_size_nm,point_defect_coeff
```

Expected CSV columns:

- `temperature`
- one conductivity column such as `xx`, `yy`, `zz`, or a custom column passed by `--fit-column`

The intended interpretation is:

- intrinsic BTE stays fixed
- only extrinsic scattering parameters are fitted
- those fitted parameters can then be reused to scan grain size, defect level, or interface quality

## Suggested Engineering Workflow

1. Use your existing MLFF validation and phonon checks to ensure the model is stable near equilibrium.
2. Run `bte` on a converged supercell/mesh to obtain intrinsic `kappa(T)`.
3. Fit only extrinsic Callaway parameters to one experimental dataset.
4. Reuse those fitted parameters to generalize across process windows:
   - grain size
   - defect concentration
   - interface quality
5. Export the resulting `k(T)` curves into COMSOL, ANSYS, or your own design pipeline.

## Notes

- This workflow is additive and does not replace the current `mff-evaluate` phonon mode.
- The ASE calculator is reused as-is, so the MLFF remains the single source of forces.
- The Callaway post-process assumes the `phono3py` HDF5 file contains `mode_kappa`, `gamma`, `frequency`, and `group_velocity`.
