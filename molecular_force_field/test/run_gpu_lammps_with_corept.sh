#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
One-click GPU test:
  pth(.pth) -> export core.pt (TorchScript, optional embed E0 / fp32) -> run MD with LAMMPS (USER-MFFTORCH)

Requirements:
- Built LAMMPS executable (with KOKKOS + USER-MFFTORCH)
- Python env with torch (CUDA) for exporting core.pt

Usage:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --pth /path/to/model.pth \
    --elements H O \
    --e0-csv /path/to/fitted_E0.csv \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

Also supports auto dummy pth (option 1 pure-cartesian-ictd):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

Also supports auto dummy pth (option 1b pure-cartesian-ictd-o3):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd-o3 \
    --elements H O \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

Option 2 spherical-save-cue (requires cuEquivariance):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-cue \
    --elements H O \
    --cutoff 5.0 \
    --steps 200

Runtime external field smoke test (rank-1):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --field-values 0.0 0.0 0.01 \
    --cutoff 5.0 \
    --steps 50

Runtime magnetic field smoke test (rank-1 axial):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --mfield-values 0.0 0.0 0.01 \
    --cutoff 5.0 \
    --steps 50

Runtime simultaneous electric + magnetic field smoke test (rank-1 + rank-1):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --field-values 0.0 0.0 0.01 \
    --mfield-values 0.0 0.0 0.02 \
    --cutoff 5.0 \
    --steps 50

Runtime external field smoke test (rank-2 full 3x3, row-major):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --field9-values 1 0 0 0 1 0 0 0 1 \
    --cutoff 5.0 \
    --steps 50

Runtime external field + physical tensor compute smoke test (including per-atom BEC):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --dummy-phys-heads \
    --test-phys-compute \
    --elements H O \
    --field-values 0.0 0.0 0.01 \
    --cutoff 5.0 \
    --steps 50

PBC minimum-image smoke test with dummy checkpoint/core:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H \
    --dtype float32 \
    --cutoff 5.0 \
    --test-pbc-minimum-image

2D slab lateral-PBC smoke test with dummy checkpoint/core:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H \
    --dtype float32 \
    --cutoff 5.0 \
    --test-pbc-slab

2D slab z-open sanity with dummy checkpoint/core:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H \
    --dtype float32 \
    --cutoff 5.0 \
    --test-pbc-slab-z-open

Triclinic PBC smoke test with dummy checkpoint/core:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H \
    --dtype float32 \
    --cutoff 5.0 \
    --test-pbc-triclinic

Long-range module smoke test (latent-coulomb, export + LAMMPS):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-long-range \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Reciprocal long-range smoke test (reciprocal-spectral-v1 mesh/FFT, export + LAMMPS):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-reciprocal-long-range \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Open-boundary tree/FMM long-range smoke test (latent-coulomb tree_fmm, export + LAMMPS):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-tree-fmm-long-range \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Open-boundary isolated far-field smoke test (isolated-far-field-v1, export + LAMMPS):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-isolated-far-field \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Open-boundary isolated far-field v2 smoke (isolated-far-field-v2, export + LAMMPS):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-isolated-far-field-v2 \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Feature-space FFT + reciprocal-source smoke test:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-feature-spectral-fft \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Feature-space FFT triclinic smoke test:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-feature-spectral-fft-triclinic \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Feature-space FFT slab smoke test:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-feature-spectral-fft-slab \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Feature-space FFT slab z-open sanity:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-feature-spectral-fft-slab-z-open \
    --elements H \
    --dtype float32 \
    --cutoff 5.0

Feature-space FFT MPI consistency sanity:
  CUDA_VISIBLE_DEVICES=0,1 \
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --test-feature-spectral-fft-mpi-consistency \
    --elements H \
    --dtype float32 \
    --cutoff 5.0 \
    --gpu 1 \
    --np 2

Throughput compare: baseline vs long-range enabled:
  CUDA_VISIBLE_DEVICES=0,1 \
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /path/to/lmp \
    --dummy-ictd \
    --test-feature-spectral-fft \
    --compare-throughput \
    --elements H O \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200 \
    --gpu 1 \
    --np 2

Throughput compare for isolated far-field on open boundary:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /path/to/lmp \
    --dummy-ictd \
    --test-isolated-far-field \
    --compare-throughput \
    --elements H O \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

No external field + physical tensor compute smoke test:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --dummy-phys-heads \
    --test-phys-compute \
    --elements H O \
    --cutoff 5.0 \
    --steps 50

Multi-GPU (4 GPUs, LAMMPS must be built with MPI):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /path/to/lmp \
    --dummy-ictd \
    --elements H O \
    --gpu 1 \
    --np 4 \
    --steps 200

  # SLURM: use srun:
  bash ... --gpu 1 --np 4 --mpi-cmd srun

Options:
  --lmp <path>        LAMMPS executable
  --pth <path>        Real checkpoint (.pth)
  --dummy-ictd        Auto-generate pure-cartesian-ictd dummy checkpoint when --pth not provided
  --dummy-ictd-o3     Auto-generate pure-cartesian-ictd-o3 dummy checkpoint when --pth not provided
  --dummy-cue         Auto-generate spherical-save-cue dummy checkpoint when --pth not provided (requires cuEquivariance)
  --dummy-phys-heads  In --dummy-ictd / --dummy-ictd-o3 mode, also add fixed-schema physical tensor heads
  --dummy-e0          Auto-generate fitted_E0.csv for embed E0 test in dummy mode (default on)
  --no-dummy-e0       Disable auto dummy E0
  --elements ...      Element order (LAMMPS type order)
  --e0-csv <path>     fitted_E0.csv (Atom,E0 columns); embed into core.pt if provided
  --dtype float32|float64   core.pt export precision (default: follow pth)
  --cutoff <A>        pair_style cutoff (Angstrom)
  --mode <mode>       Model mode for export (e.g. spherical-save-cue); else from checkpoint
  --steps <N>         MD steps (default 200)
  --out-dir <dir>     Output dir (default mktemp)
  --gpu <g>           GPUs per MPI rank passed to `lmp -k on g <g>` (default 1)
  --np <N>            MPI processes for multi-GPU (default 1; recommended one rank per GPU)
  --mpi-cmd <cmd>     MPI launcher for multi-GPU (default mpirun; use srun for SLURM)
  --n1 <int>          Random type1 atoms (default 2000)
  --n2 <int>          Random type2 atoms (default 1000)
  --box <float>       Box side length (default auto)
  --native-ops        spherical-save-cue: keep native cuEquivariance ops (requires MFF_CUSTOM_OPS_LIB)
  --field-values Ex Ey Ez   Enable runtime electric field via pair_style field v_Ex v_Ey v_Ez
  --mfield-values Bx By Bz Enable runtime magnetic field via pair_style mfield v_Bx v_By v_Bz
  --field6-values xx yy zz xy xz yz
                      Enable symmetric rank-2 runtime external field via pair_style field6
  --field9-values xx xy xz yx yy yz zx zy zz
                      Enable full rank-2 runtime external field via pair_style field9 (row-major)
  --test-phys-compute Inject compute mff/torch/phys checks into the generated LAMMPS input
  --test-pbc-minimum-image
                     Run a dedicated two-atom PBC equivalence test:
                     one cross-boundary pair vs one equivalent in-box pair.
                     Requires exactly one element in --elements.
  --test-pbc-slab
                     Run a dedicated two-atom slab smoke with `boundary p p f`:
                     compare one x-cross-boundary pair vs one equivalent in-box pair.
                     This validates lateral periodic geometry only, not a slab Green's function.
                     Requires exactly one element in --elements.
  --test-pbc-slab-z-open
                     Run a dedicated `boundary p p f` sanity:
                     compare a far-separated z pair against a true short-range z pair.
                     They must NOT be treated as equivalent, which catches accidental z-periodic wrapping.
                     Requires exactly one element in --elements.
  --test-pbc-triclinic
                     Run a dedicated two-atom triclinic PBC equivalence test.
                     Requires exactly one element in --elements.
  --test-long-range  Create dummy checkpoint with long-range module (latent-coulomb),
                     export core.pt, run PBC minimum-image smoke test.
                     Requires --dummy-ictd and exactly one element in --elements.
  --test-reciprocal-long-range
                     Create dummy checkpoint with reciprocal-spectral-v1 long-range module
                     using the mesh_fft reciprocal backend, export core.pt with
                     reciprocal_source, run PBC minimum-image smoke test.
                     Can be combined with --test-pbc-slab / --test-pbc-slab-z-open.
                     Requires --dummy-ictd and exactly one element in --elements.
  --test-tree-fmm-long-range
                     Create dummy checkpoint with latent-coulomb + tree_fmm,
                     export core.pt with runtime long-range source, run a dedicated
                     open-boundary single-rank LAMMPS-vs-Python eager smoke, and when
                     --np > 1 also compare np=1 vs np>1 LAMMPS consistency.
                     Requires --dummy-ictd and exactly one element.
  --test-isolated-far-field
                     Create dummy checkpoint with isolated-far-field-v1,
                     export core.pt, run a dedicated open-boundary single-rank
                     LAMMPS-vs-Python eager smoke.
                     Requires --dummy-ictd or --dummy-cue and exactly one element.
  --test-isolated-far-field-v2
                     Create dummy checkpoint with isolated-far-field-v2,
                     export core.pt, run a dedicated open-boundary single-rank
                     LAMMPS-vs-Python eager smoke.
                     Requires --dummy-ictd or --dummy-cue and exactly one element.
  --test-feature-spectral-fft
                     Create dummy checkpoint with feature-space FFT block,
                     export core.pt with reciprocal_source, run smoke or full MD.
                     With --force-md-run or --compare-throughput: multi-element (e.g. H O) allowed.
                     Without: runs PBC minimum-image test, requires exactly one element.
  --test-feature-spectral-fft-triclinic
                     Create dummy checkpoint with feature-space FFT block,
                     export core.pt, run triclinic PBC smoke test.
                     Requires --dummy-ictd or --dummy-cue and exactly one element in --elements.
  --test-feature-spectral-fft-slab
                     Create dummy checkpoint with feature-space FFT block,
                     export core.pt, run the `boundary p p f` slab smoke.
                     This exercises the current vacuum-padded slab reciprocal backend.
                     Requires --dummy-ictd or --dummy-cue and exactly one element in --elements.
  --test-feature-spectral-fft-slab-z-open
                     Create dummy checkpoint with feature-space FFT block,
                     export core.pt, then run the slab z-open sanity test.
                     This checks that the current slab reciprocal path does not wrap z as periodic.
                     Requires --dummy-ictd or --dummy-cue and exactly one element in --elements.
  --test-feature-spectral-fft-mpi-consistency
                     Create dummy checkpoint with feature-space FFT block,
                     export core.pt, run np=1 and np>1 orthogonal PBC smoke,
                     then compare single-rank vs MPI energies/forces.
                     Requires --dummy-ictd or --dummy-cue, exactly one element, and --np > 1.
  --compare-throughput
                     Run two full MD throughput cases with the same random system:
                     one baseline (no long-range feature/reciprocal path) and one with the selected
                     long-range/feature-FFT model option enabled, then compare loop time and throughput.
                     Currently intended for --dummy-ictd / --dummy-cue model generation.
  --tree-fmm-gpu-aware-mpi
                     Request GPU-aware MPI for USER-MFFTORCH runtime. The runtime now
                     performs conservative capability checks and may auto-fallback.
  --tree-fmm-assume-gpu-aware-mpi
                     Force-enable GPU-aware MPI without runtime capability checks.
                     Only use this if you know your MPI build supports device pointers.
  --tree-fmm-device-local-eval
                     Backward-compatible alias for the CUDA-first pure-GPU local tree path.
                     On CUDA runs this path is now the default.
  --tree-fmm-reuse-position-tol <float>
                     Export `MFF_TREE_FMM_REUSE_POSITION_TOL=<float>` so nearby MD steps
                     can reuse cached tree topology when displacements stay below this threshold.

Notes:
- Script creates random system (create_atoms), uses pair_style mff/torch with -sf kk for Kokkos CUDA.
- Multi-GPU recommended mode is one rank per GPU, e.g. `CUDA_VISIBLE_DEVICES=0,1 --gpu 1 --np 2`.
- The script binds one visible GPU to each MPI rank using local-rank environment variables.
- If multi-GPU hangs: reduce system size with `--n1 500 --n2 250` or inspect MPI rank/GPU binding.
- For custom system: edit input section, replace with read_data for your data file.
- Four common test combinations:
  1) No field, no physical tensor:
     --dummy-ictd or --dummy-ictd-o3
  2) Field, no physical tensor:
     --dummy-ictd / --dummy-ictd-o3 + one of:
       --field-values
       --mfield-values
       --field-values + --mfield-values
       --field6-values
       --field9-values
  3) No field, with physical tensor:
     --dummy-ictd / --dummy-ictd-o3 --dummy-phys-heads --test-phys-compute
  4) Field, with physical tensor:
     --dummy-ictd / --dummy-ictd-o3 --dummy-phys-heads --test-phys-compute + one rank-1/2 field configuration
- --dummy-phys-heads currently adds fixed-schema heads only:
    dipole, dipole_per_atom, polarizability, polarizability_per_atom,
    born_effective_charge_per_atom
  and is intended to validate the current mfftorch/LAMMPS export path.
- Current LAMMPS mfftorch physical tensor interface exposes only fixed-schema quantities:
    charge, dipole, polarizability, quadrupole
    charge_per_atom, dipole_per_atom, polarizability_per_atom, quadrupole_per_atom,
    born_effective_charge_per_atom
  Missing heads are allowed: masks become 0 and corresponding outputs are filled with 0.
- Rank/l compatibility of the fixed schema:
    charge / charge_per_atom correspond to l=0 scalar outputs
    dipole / dipole_per_atom correspond to l=1 vector outputs
    polarizability is expected as a rank-2 Cartesian tensor (typically l=0+2)
    quadrupole is expected as a rank-2 traceless tensor (typically l=2)
    born_effective_charge_per_atom is expected as a full rank-2 Cartesian tensor (typically l=0+1+2)
- External field compatibility:
    --field-values tests rank-1 electric-like `1o`
    --mfield-values tests rank-1 magnetic-like `1e`
    --field-values + --mfield-values tests simultaneous rank-1 multi-field packing
    --field6-values / --field9-values test rank-2 external tensor input
- Arbitrary custom physical head names are not auto-exposed to LAMMPS yet.
  If your model trains a custom scalar/tensor head outside the fixed schema above,
  the model can still run, but compute mff/torch/phys will not automatically expose it.
- Current LibTorch/LAMMPS export path assumes channels_out == 1 for exposed physical heads.
- `reciprocal-spectral-v1` keeps the legacy `direct_kspace` path in Python, but this smoke script now
  exercises the LES-style `mesh_fft` reciprocal backend and exports reciprocal_source for USER-MFFTORCH.
- `--test-tree-fmm-long-range` exercises the new `latent-coulomb + tree_fmm` open-boundary runtime path.
- With `--np 1` it compares single-rank LAMMPS against Python eager; with `--np > 1` it also compares
  single-rank and MPI LAMMPS results for the same open-boundary geometry.
- `--test-isolated-far-field` exercises the new `isolated-far-field-v1` open-boundary correction path
  through plain `core.pt -> pair_style mff/torch`, without runtime reciprocal/tree source handling.
- When paired with `--compare-throughput`, `--test-isolated-far-field` also switches both baseline and
  enabled runs to the open-boundary MD template so the comparison stays geometry-consistent.
- `--test-isolated-far-field-v2` does the same for the explicit-shell `isolated-far-field-v2` path.
- `tree_fmm` runtime tuning flags exposed by this script:
    `--tree-fmm-gpu-aware-mpi`, `--tree-fmm-assume-gpu-aware-mpi`,
    `--tree-fmm-device-local-eval`, `--tree-fmm-reuse-position-tol <tol>`
- `--test-feature-spectral-fft` exports a core with reciprocal_source and exercises the new
  distributed-contract reciprocal solver. Use `CUDA_VISIBLE_DEVICES=0,1 --np 2 --gpu 1` for a simple MPI sanity smoke.
- The current performance-oriented implementation batches `phi/gx/gy/gz` through one reciprocal field path,
  which reduces repeated transpose/Allgather overhead without changing the external `core.pt` contract.
- `--test-phys-compute` enables physical tensor caching and therefore reintroduces extra host-side output traffic;
  leave it off when checking reciprocal performance.
- `--test-pbc-slab` runs the same two-atom equivalence idea under `boundary p p f` and only checks
  lateral periodic wrapping (x/y), not nonperiodic-z long-range physics.
- `--test-pbc-slab-z-open` is the companion negative test: a large z separation should remain large and
  must not collapse through a fake minimum image along the nonperiodic direction.
- `--test-feature-spectral-fft-slab` reuses the same slab geometry with reciprocal_source export enabled.
  It verifies the current vacuum-padded slab backend and `core.pt -> USER-MFFTORCH` plumbing under mixed boundary settings.
- `--test-feature-spectral-fft-slab-z-open` extends the same slab backend smoke with an explicit nonperiodic-z sanity check.
- `--test-feature-spectral-fft-triclinic` extends the same smoke to a triclinic box.
- `--test-feature-spectral-fft-mpi-consistency` compares `np=1` and `np>1` results for the same
  orthogonal feature-FFT smoke case to catch rank-splitting regressions.
- `--compare-throughput` is for MD throughput, not geometry smoke. It re-runs this script internally
  in a baseline configuration and an enabled-long-range configuration, then reports relative slowdown.
  When paired with `--test-tree-fmm-long-range`, it now uses an open-boundary MD template instead of the default periodic box.
EOF
}

LMP=""
PTH=""
DUMMY_ICTD=0
DUMMY_ICTD_O3=0
DUMMY_CUE=0
DUMMY_PHYS_HEADS=0
DUMMY_E0=1
MODE=""
NATIVE_OPS=0
ELEMENTS=()
E0CSV=""
DTYPE=""
CUTOFF="5.0"
STEPS="200"
OUT_DIR=""
GPU_N="1"
NP=""          # MPI processes for multi-GPU; default = GPU_N
MPI_CMD="mpirun"
N1="2000"
N2="1000"
BOX=""
FIELD_VALUES=()
MFIELD_VALUES=()
FIELD6_VALUES=()
FIELD9_VALUES=()
FIDELITY_VALUE=""
EXPORT_FIDELITY_ID=""
NUM_FIDELITY_LEVELS=0
MULTI_FIDELITY_MODE="conditioning"
TEST_PHYS_COMPUTE=0
TEST_PBC_MINIMUM_IMAGE=0
TEST_PBC_SLAB=0
TEST_PBC_SLAB_Z_OPEN=0
TEST_PBC_TRICLINIC=0
TEST_LONG_RANGE=0
TEST_RECIPROCAL_LONG_RANGE=0
TEST_TREE_FMM_LONG_RANGE=0
TEST_ISOLATED_FAR_FIELD=0
TEST_ISOLATED_FAR_FIELD_V2=0
TEST_FEATURE_SPECTRAL_FFT=0
TEST_FEATURE_SPECTRAL_FFT_SLAB=0
TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN=0
TEST_FEATURE_SPECTRAL_FFT_TRICLINIC=0
TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY=0
COMPARE_THROUGHPUT=0
FORCE_MD_RUN=0
OPEN_BOUNDARY_MD=0
TREE_FMM_GPU_AWARE_MPI=0
TREE_FMM_ASSUME_GPU_AWARE_MPI=0
TREE_FMM_DEVICE_LOCAL_EVAL=0
TREE_FMM_REUSE_POSITION_TOL=""
TREE_FMM_THROUGHPUT_AUTO_TUNED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmp) LMP="${2:-}"; shift 2;;
    --pth) PTH="${2:-}"; shift 2;;
    --dummy-ictd) DUMMY_ICTD=1; shift;;
    --dummy-ictd-o3) DUMMY_ICTD_O3=1; shift;;
    --dummy-cue) DUMMY_CUE=1; shift;;
    --dummy-phys-heads) DUMMY_PHYS_HEADS=1; shift;;
    --dummy-e0) DUMMY_E0=1; shift;;
    --no-dummy-e0) DUMMY_E0=0; shift;;
    --elements)
      shift
      ELEMENTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do ELEMENTS+=("$1"); shift; done
      ;;
    --e0-csv) E0CSV="${2:-}"; shift 2;;
    --dtype) DTYPE="${2:-}"; shift 2;;
    --cutoff) CUTOFF="${2:-}"; shift 2;;
    --mode) MODE="${2:-}"; shift 2;;
    --native-ops) NATIVE_OPS=1; shift;;
    --steps) STEPS="${2:-}"; shift 2;;
    --out-dir) OUT_DIR="${2:-}"; shift 2;;
    --gpu) GPU_N="${2:-}"; shift 2;;
    --np) NP="${2:-}"; shift 2;;
    --mpi-cmd) MPI_CMD="${2:-}"; shift 2;;
    --n1) N1="${2:-}"; shift 2;;
    --n2) N2="${2:-}"; shift 2;;
    --box) BOX="${2:-}"; shift 2;;
    --field-values)
      shift
      FIELD_VALUES=("${1:-}" "${2:-}" "${3:-}")
      shift 3
      ;;
    --mfield-values)
      shift
      MFIELD_VALUES=("${1:-}" "${2:-}" "${3:-}")
      shift 3
      ;;
    --field6-values)
      shift
      FIELD6_VALUES=("${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}")
      shift 6
      ;;
    --field9-values)
      shift
      FIELD9_VALUES=("${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}" "${8:-}" "${9:-}")
      shift 9
      ;;
    --fidelity-value) FIDELITY_VALUE="${2:-}"; shift 2;;
    --export-fidelity-id) EXPORT_FIDELITY_ID="${2:-}"; shift 2;;
    --num-fidelity-levels) NUM_FIDELITY_LEVELS="${2:-}"; shift 2;;
    --multi-fidelity-mode) MULTI_FIDELITY_MODE="${2:-}"; shift 2;;
    --test-phys-compute) TEST_PHYS_COMPUTE=1; shift;;
    --test-pbc-minimum-image) TEST_PBC_MINIMUM_IMAGE=1; shift;;
    --test-pbc-slab) TEST_PBC_SLAB=1; shift;;
    --test-pbc-slab-z-open) TEST_PBC_SLAB_Z_OPEN=1; shift;;
    --test-pbc-triclinic) TEST_PBC_TRICLINIC=1; shift;;
    --test-long-range) TEST_LONG_RANGE=1; shift;;
    --test-reciprocal-long-range) TEST_RECIPROCAL_LONG_RANGE=1; shift;;
    --test-tree-fmm-long-range) TEST_TREE_FMM_LONG_RANGE=1; shift;;
    --test-isolated-far-field) TEST_ISOLATED_FAR_FIELD=1; shift;;
    --test-isolated-far-field-v2) TEST_ISOLATED_FAR_FIELD_V2=1; shift;;
    --test-feature-spectral-fft) TEST_FEATURE_SPECTRAL_FFT=1; shift;;
    --test-feature-spectral-fft-slab) TEST_FEATURE_SPECTRAL_FFT_SLAB=1; shift;;
    --test-feature-spectral-fft-slab-z-open) TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN=1; shift;;
    --test-feature-spectral-fft-triclinic) TEST_FEATURE_SPECTRAL_FFT_TRICLINIC=1; shift;;
    --test-feature-spectral-fft-mpi-consistency) TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY=1; shift;;
    --compare-throughput) COMPARE_THROUGHPUT=1; shift;;
    --tree-fmm-gpu-aware-mpi) TREE_FMM_GPU_AWARE_MPI=1; shift;;
    --tree-fmm-assume-gpu-aware-mpi) TREE_FMM_ASSUME_GPU_AWARE_MPI=1; shift;;
    --tree-fmm-device-local-eval) TREE_FMM_DEVICE_LOCAL_EVAL=1; shift;;
    --tree-fmm-reuse-position-tol) TREE_FMM_REUSE_POSITION_TOL="${2:-}"; shift 2;;
    --force-md-run) FORCE_MD_RUN=1; shift;;
    --open-boundary-md) OPEN_BOUNDARY_MD=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

if [[ -z "$LMP" || ${#ELEMENTS[@]} -eq 0 ]]; then
  echo "Must provide --lmp --elements"
  usage
  exit 2
fi
if [[ -z "$PTH" && $DUMMY_ICTD -ne 1 && $DUMMY_ICTD_O3 -ne 1 && $DUMMY_CUE -ne 1 ]]; then
  echo "Must provide --pth, or use --dummy-ictd / --dummy-ictd-o3 / --dummy-cue to auto-generate"
  usage
  exit 2
fi
if [[ $((DUMMY_ICTD + DUMMY_ICTD_O3 + DUMMY_CUE)) -gt 1 ]]; then
  echo "Cannot use more than one of --dummy-ictd, --dummy-ictd-o3, and --dummy-cue"
  exit 2
fi
if [[ $DUMMY_PHYS_HEADS -eq 1 && $DUMMY_ICTD -ne 1 && $DUMMY_ICTD_O3 -ne 1 ]]; then
  echo "--dummy-phys-heads currently requires --dummy-ictd or --dummy-ictd-o3"
  exit 2
fi
HAS_RANK1_FIELD=0
[[ ${#FIELD_VALUES[@]} -eq 3 ]] && HAS_RANK1_FIELD=1
[[ ${#MFIELD_VALUES[@]} -eq 3 ]] && HAS_RANK1_FIELD=1
RANK2_MODE_COUNT=0
[[ ${#FIELD6_VALUES[@]} -eq 6 ]] && RANK2_MODE_COUNT=$((RANK2_MODE_COUNT+1))
[[ ${#FIELD9_VALUES[@]} -eq 9 ]] && RANK2_MODE_COUNT=$((RANK2_MODE_COUNT+1))
if [[ $RANK2_MODE_COUNT -gt 1 ]]; then
  echo "Use only one of --field6-values or --field9-values"
  exit 2
fi
if [[ $HAS_RANK1_FIELD -eq 1 && $RANK2_MODE_COUNT -gt 0 ]]; then
  echo "Do not combine --field6-values/--field9-values with --field-values/--mfield-values"
  exit 2
fi
if [[ ${#FIELD_VALUES[@]} -ne 0 && ${#FIELD_VALUES[@]} -ne 3 ]]; then
  echo "--field-values expects exactly 3 values: Ex Ey Ez"
  exit 2
fi
if [[ ${#MFIELD_VALUES[@]} -ne 0 && ${#MFIELD_VALUES[@]} -ne 3 ]]; then
  echo "--mfield-values expects exactly 3 values: Bx By Bz"
  exit 2
fi
if [[ ${#FIELD6_VALUES[@]} -ne 0 && ${#FIELD6_VALUES[@]} -ne 6 ]]; then
  echo "--field6-values expects exactly 6 values: xx yy zz xy xz yz"
  exit 2
fi
if [[ ${#FIELD9_VALUES[@]} -ne 0 && ${#FIELD9_VALUES[@]} -ne 9 ]]; then
  echo "--field9-values expects exactly 9 values in row-major order"
  exit 2
fi
if [[ -n "$FIDELITY_VALUE" ]]; then
  if ! [[ "$FIDELITY_VALUE" =~ ^-?[0-9]+$ ]]; then
    echo "--fidelity-value expects an integer"
    exit 2
  fi
fi
if [[ -n "$EXPORT_FIDELITY_ID" ]]; then
  if ! [[ "$EXPORT_FIDELITY_ID" =~ ^-?[0-9]+$ ]]; then
    echo "--export-fidelity-id expects an integer"
    exit 2
  fi
fi
if ! [[ "$NUM_FIDELITY_LEVELS" =~ ^[0-9]+$ ]]; then
  echo "--num-fidelity-levels expects a non-negative integer"
  exit 2
fi
if [[ "$MULTI_FIDELITY_MODE" != "conditioning" && "$MULTI_FIDELITY_MODE" != "delta-baseline" ]]; then
  echo "--multi-fidelity-mode must be conditioning or delta-baseline"
  exit 2
fi
if [[ -n "$EXPORT_FIDELITY_ID" && -n "$FIDELITY_VALUE" ]]; then
  echo "Do not combine --export-fidelity-id with --fidelity-value; fixed-fidelity export does not accept runtime fidelity"
  exit 2
fi
if [[ "$NUM_FIDELITY_LEVELS" != "0" && -z "$EXPORT_FIDELITY_ID" && -z "$FIDELITY_VALUE" ]]; then
  echo "With --num-fidelity-levels > 0, provide either --fidelity-value for runtime fidelity or --export-fidelity-id to freeze a branch"
  exit 2
fi
if [[ ${#FIELD_VALUES[@]} -ne 0 && $DUMMY_CUE -eq 1 ]]; then
  echo "Runtime external field is currently supported only for pure-cartesian-ictd / pure-cartesian-ictd-o3"
  exit 2
fi
if [[ ${#MFIELD_VALUES[@]} -ne 0 && $DUMMY_CUE -eq 1 ]]; then
  echo "Runtime magnetic field is currently supported only for pure-cartesian-ictd / pure-cartesian-ictd-o3"
  exit 2
fi
if [[ ${#FIELD6_VALUES[@]} -ne 0 && $DUMMY_CUE -eq 1 ]]; then
  echo "Runtime external field is currently supported only for pure-cartesian-ictd / pure-cartesian-ictd-o3"
  exit 2
fi
if [[ ${#FIELD9_VALUES[@]} -ne 0 && $DUMMY_CUE -eq 1 ]]; then
  echo "Runtime external field is currently supported only for pure-cartesian-ictd / pure-cartesian-ictd-o3"
  exit 2
fi
if [[ $DUMMY_PHYS_HEADS -eq 1 ]]; then
  TEST_PHYS_COMPUTE=1
fi
if [[ $COMPARE_THROUGHPUT -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--compare-throughput currently requires --dummy-ictd or --dummy-cue so baseline and enabled checkpoints can be generated consistently"
    exit 2
  fi
  LR_MODEL_COUNT=0
  [[ $TEST_LONG_RANGE -eq 1 ]] && LR_MODEL_COUNT=$((LR_MODEL_COUNT+1))
  [[ $TEST_RECIPROCAL_LONG_RANGE -eq 1 ]] && LR_MODEL_COUNT=$((LR_MODEL_COUNT+1))
  [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 ]] && LR_MODEL_COUNT=$((LR_MODEL_COUNT+1))
  [[ $TEST_ISOLATED_FAR_FIELD -eq 1 ]] && LR_MODEL_COUNT=$((LR_MODEL_COUNT+1))
  [[ $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]] && LR_MODEL_COUNT=$((LR_MODEL_COUNT+1))
  if [[ $TEST_FEATURE_SPECTRAL_FFT -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_TRICLINIC -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY -eq 1 ]]; then
    LR_MODEL_COUNT=$((LR_MODEL_COUNT+1))
  fi
  if [[ $LR_MODEL_COUNT -ne 1 ]]; then
    echo "--compare-throughput expects exactly one long-range model selector: --test-long-range, --test-reciprocal-long-range, --test-tree-fmm-long-range, --test-isolated-far-field, --test-isolated-far-field-v2, or one feature-spectral-fft variant"
    exit 2
  fi
  if [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
    # Only auto-tune when user has not customized any of --steps/--n1/--n2.
    # If user explicitly passes e.g. --steps 200 --n1 1024 --n2 512, honor them.
    if [[ "$STEPS" == "200" && "$N1" == "2000" && "$N2" == "1000" ]]; then
      STEPS="20"
      N1="256"
      N2="128"
      TREE_FMM_THROUGHPUT_AUTO_TUNED=1
    fi
  fi
fi
if [[ $DUMMY_ICTD_O3 -eq 1 && ($TEST_LONG_RANGE -eq 1 || $TEST_RECIPROCAL_LONG_RANGE -eq 1 || $TEST_TREE_FMM_LONG_RANGE -eq 1 || $TEST_ISOLATED_FAR_FIELD -eq 1 || $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 || $TEST_FEATURE_SPECTRAL_FFT -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_TRICLINIC -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY -eq 1 || $COMPARE_THROUGHPUT -eq 1) ]]; then
  echo "--dummy-ictd-o3 currently supports the core field/phys smoke paths only; long-range and throughput modes still use --dummy-ictd"
  exit 2
fi
MD_ONLY_MODE=0
if [[ $FORCE_MD_RUN -eq 1 || $COMPARE_THROUGHPUT -eq 1 ]]; then
  MD_ONLY_MODE=1
fi
if [[ $TEST_PBC_MINIMUM_IMAGE -eq 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
  echo "--test-pbc-minimum-image currently requires exactly one element in --elements"
  exit 2
fi
if [[ $TEST_PBC_SLAB -eq 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
  echo "--test-pbc-slab currently requires exactly one element in --elements"
  exit 2
fi
if [[ $TEST_PBC_SLAB_Z_OPEN -eq 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
  echo "--test-pbc-slab-z-open currently requires exactly one element in --elements"
  exit 2
fi
if [[ $TEST_PBC_TRICLINIC -eq 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
  echo "--test-pbc-triclinic currently requires exactly one element in --elements"
  exit 2
fi
PBC_TEST_COUNT=0
[[ $TEST_PBC_MINIMUM_IMAGE -eq 1 ]] && PBC_TEST_COUNT=$((PBC_TEST_COUNT+1))
[[ $TEST_PBC_SLAB -eq 1 ]] && PBC_TEST_COUNT=$((PBC_TEST_COUNT+1))
[[ $TEST_PBC_SLAB_Z_OPEN -eq 1 ]] && PBC_TEST_COUNT=$((PBC_TEST_COUNT+1))
[[ $TEST_PBC_TRICLINIC -eq 1 ]] && PBC_TEST_COUNT=$((PBC_TEST_COUNT+1))
if [[ $PBC_TEST_COUNT -gt 1 ]]; then
  echo "Use only one of --test-pbc-minimum-image, --test-pbc-slab, --test-pbc-slab-z-open, or --test-pbc-triclinic"
  exit 2
fi
if [[ $TEST_LONG_RANGE -eq 1 && ($TEST_RECIPROCAL_LONG_RANGE -eq 1 || $TEST_TREE_FMM_LONG_RANGE -eq 1 || $TEST_ISOLATED_FAR_FIELD -eq 1 || $TEST_ISOLATED_FAR_FIELD_V2 -eq 1) ]]; then
  echo "Use only one of --test-long-range, --test-reciprocal-long-range, --test-tree-fmm-long-range, --test-isolated-far-field, or --test-isolated-far-field-v2"
  exit 2
fi
if [[ $TEST_RECIPROCAL_LONG_RANGE -eq 1 && ($TEST_TREE_FMM_LONG_RANGE -eq 1 || $TEST_ISOLATED_FAR_FIELD -eq 1 || $TEST_ISOLATED_FAR_FIELD_V2 -eq 1) ]]; then
  echo "Use only one of --test-reciprocal-long-range, --test-tree-fmm-long-range, --test-isolated-far-field, or --test-isolated-far-field-v2"
  exit 2
fi
if [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 && $TEST_ISOLATED_FAR_FIELD -eq 1 ]]; then
  echo "Use only one of --test-tree-fmm-long-range or --test-isolated-far-field"
  exit 2
fi
if [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 && $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]]; then
  echo "Use only one of --test-tree-fmm-long-range or --test-isolated-far-field-v2"
  exit 2
fi
if [[ $TEST_ISOLATED_FAR_FIELD -eq 1 && $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]]; then
  echo "Use only one of --test-isolated-far-field or --test-isolated-far-field-v2"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT -eq 1 && ($TEST_LONG_RANGE -eq 1 || $TEST_RECIPROCAL_LONG_RANGE -eq 1) ]]; then
  echo "Use --test-feature-spectral-fft separately from the legacy long-range smoke flags"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT -eq 1 && $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
  echo "Use --test-feature-spectral-fft separately from --test-tree-fmm-long-range"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_TRICLINIC -eq 1 && ($TEST_LONG_RANGE -eq 1 || $TEST_RECIPROCAL_LONG_RANGE -eq 1) ]]; then
  echo "Use --test-feature-spectral-fft-triclinic separately from the legacy long-range smoke flags"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_TRICLINIC -eq 1 && $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
  echo "Use --test-feature-spectral-fft-triclinic separately from --test-tree-fmm-long-range"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 && ($TEST_LONG_RANGE -eq 1 || $TEST_RECIPROCAL_LONG_RANGE -eq 1) ]]; then
  echo "Use --test-feature-spectral-fft-slab separately from the legacy long-range smoke flags"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 && $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
  echo "Use --test-feature-spectral-fft-slab separately from --test-tree-fmm-long-range"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 && ($TEST_LONG_RANGE -eq 1 || $TEST_RECIPROCAL_LONG_RANGE -eq 1) ]]; then
  echo "Use --test-feature-spectral-fft-slab-z-open separately from the legacy long-range smoke flags"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 && $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
  echo "Use --test-feature-spectral-fft-slab-z-open separately from --test-tree-fmm-long-range"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY -eq 1 && ($TEST_LONG_RANGE -eq 1 || $TEST_RECIPROCAL_LONG_RANGE -eq 1) ]]; then
  echo "Use --test-feature-spectral-fft-mpi-consistency separately from the legacy long-range smoke flags"
  exit 2
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY -eq 1 && $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
  echo "Use --test-feature-spectral-fft-mpi-consistency separately from --test-tree-fmm-long-range"
  exit 2
fi
if [[ $TEST_LONG_RANGE -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 ]]; then
    echo "--test-long-range requires --dummy-ictd"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-long-range requires exactly one element in --elements (or use --force-md-run / --compare-throughput for multi-element MD)"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 ]]; then
    TEST_PBC_MINIMUM_IMAGE=1
  fi
fi
if [[ $TEST_RECIPROCAL_LONG_RANGE -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 ]]; then
    echo "--test-reciprocal-long-range requires --dummy-ictd"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-reciprocal-long-range requires exactly one element in --elements (or use --force-md-run / --compare-throughput for multi-element MD)"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 ]]; then
    TEST_PBC_MINIMUM_IMAGE=1
  fi
fi
if [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 ]]; then
    echo "--test-tree-fmm-long-range requires --dummy-ictd"
    exit 2
  fi
  if [[ ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-tree-fmm-long-range currently requires exactly one element in --elements"
    exit 2
  fi
fi
if [[ $TEST_ISOLATED_FAR_FIELD -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--test-isolated-far-field requires --dummy-ictd or --dummy-cue"
    exit 2
  fi
  if [[ $COMPARE_THROUGHPUT -ne 1 && $FORCE_MD_RUN -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-isolated-far-field requires exactly one element in --elements (or use --force-md-run / --compare-throughput for multi-element MD)"
    exit 2
  fi
fi
if [[ $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--test-isolated-far-field-v2 requires --dummy-ictd or --dummy-cue"
    exit 2
  fi
  if [[ $COMPARE_THROUGHPUT -ne 1 && $FORCE_MD_RUN -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-isolated-far-field-v2 requires exactly one element in --elements (or use --force-md-run / --compare-throughput for multi-element MD)"
    exit 2
  fi
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--test-feature-spectral-fft requires --dummy-ictd or --dummy-cue"
    exit 2
  fi
  # Single-element required only for PBC minimum-image smoke; MD/throughput flow supports multi-element
  if [[ $MD_ONLY_MODE -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-feature-spectral-fft requires exactly one element in --elements (or use --force-md-run / --compare-throughput for multi-element MD)"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 ]]; then
    TEST_PBC_MINIMUM_IMAGE=1
  fi
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_TRICLINIC -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--test-feature-spectral-fft-triclinic requires --dummy-ictd or --dummy-cue"
    exit 2
  fi
  if [[ $COMPARE_THROUGHPUT -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-feature-spectral-fft-triclinic requires exactly one element in --elements (PBC triclinic test uses single-type data)"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 ]]; then
    TEST_PBC_TRICLINIC=1
  fi
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--test-feature-spectral-fft-slab requires --dummy-ictd or --dummy-cue"
    exit 2
  fi
  if [[ $COMPARE_THROUGHPUT -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-feature-spectral-fft-slab requires exactly one element in --elements (PBC slab test uses single-type data)"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 ]]; then
    TEST_PBC_SLAB=1
  fi
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--test-feature-spectral-fft-slab-z-open requires --dummy-ictd or --dummy-cue"
    exit 2
  fi
  if [[ $COMPARE_THROUGHPUT -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-feature-spectral-fft-slab-z-open requires exactly one element in --elements (PBC slab z-open test uses single-type data)"
    exit 2
  fi
  if [[ $MD_ONLY_MODE -ne 1 ]]; then
    TEST_PBC_SLAB_Z_OPEN=1
  fi
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY -eq 1 ]]; then
  if [[ $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
    echo "--test-feature-spectral-fft-mpi-consistency requires --dummy-ictd or --dummy-cue"
    exit 2
  fi
  if [[ $COMPARE_THROUGHPUT -ne 1 && ${#ELEMENTS[@]} -ne 1 ]]; then
    echo "--test-feature-spectral-fft-mpi-consistency requires exactly one element in --elements (PBC minimum-image MPI test uses single-type data)"
    exit 2
  fi
  if [[ $COMPARE_THROUGHPUT -ne 1 && (-z "$NP" || "$NP" -le 1) ]]; then
    echo "--test-feature-spectral-fft-mpi-consistency requires --np > 1"
    exit 2
  fi
fi

FINAL_PBC_TEST_COUNT=0
[[ $TEST_PBC_MINIMUM_IMAGE -eq 1 ]] && FINAL_PBC_TEST_COUNT=$((FINAL_PBC_TEST_COUNT+1))
[[ $TEST_PBC_SLAB -eq 1 ]] && FINAL_PBC_TEST_COUNT=$((FINAL_PBC_TEST_COUNT+1))
[[ $TEST_PBC_SLAB_Z_OPEN -eq 1 ]] && FINAL_PBC_TEST_COUNT=$((FINAL_PBC_TEST_COUNT+1))
[[ $TEST_PBC_TRICLINIC -eq 1 ]] && FINAL_PBC_TEST_COUNT=$((FINAL_PBC_TEST_COUNT+1))
if [[ $FINAL_PBC_TEST_COUNT -gt 1 ]]; then
  echo "Resolved test mode is ambiguous; choose only one of orthogonal, slab, or triclinic PBC smoke."
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(python - <<'PY'
import tempfile
print(tempfile.mkdtemp(prefix="mff-corept-lmp-"))
PY
)"
fi
mkdir -p "$OUT_DIR"

CORE_PT="$OUT_DIR/core.pt"

if [[ -n "$TREE_FMM_REUSE_POSITION_TOL" ]]; then
  python - <<'PY' "$TREE_FMM_REUSE_POSITION_TOL"
import sys
value = float(sys.argv[1])
if value < 0.0:
    raise SystemExit("tree_fmm reuse_position_tol must be >= 0")
PY
fi

[[ $TREE_FMM_GPU_AWARE_MPI -eq 1 ]] && export MFF_TREE_FMM_GPU_AWARE_MPI=1
[[ $TREE_FMM_ASSUME_GPU_AWARE_MPI -eq 1 ]] && export MFF_TREE_FMM_ASSUME_GPU_AWARE_MPI=1
[[ $TREE_FMM_DEVICE_LOCAL_EVAL -eq 1 ]] && export MFF_TREE_FMM_DEVICE_LOCAL_EVAL=1
[[ -n "$TREE_FMM_REUSE_POSITION_TOL" ]] && export MFF_TREE_FMM_REUSE_POSITION_TOL="$TREE_FMM_REUSE_POSITION_TOL"

append_common_cli_args() {
  local -n _out="$1"
  _out+=(--lmp "$LMP")
  if [[ -n "$PTH" ]]; then _out+=(--pth "$PTH"); fi
  [[ $DUMMY_ICTD -eq 1 ]] && _out+=(--dummy-ictd)
  [[ $DUMMY_ICTD_O3 -eq 1 ]] && _out+=(--dummy-ictd-o3)
  [[ $DUMMY_CUE -eq 1 ]] && _out+=(--dummy-cue)
  [[ $DUMMY_PHYS_HEADS -eq 1 ]] && _out+=(--dummy-phys-heads)
  if [[ $DUMMY_E0 -eq 0 ]]; then _out+=(--no-dummy-e0); fi
  _out+=(--elements "${ELEMENTS[@]}")
  [[ -n "$E0CSV" ]] && _out+=(--e0-csv "$E0CSV")
  [[ -n "$DTYPE" ]] && _out+=(--dtype "$DTYPE")
  _out+=(--cutoff "$CUTOFF")
  [[ -n "$MODE" ]] && _out+=(--mode "$MODE")
  [[ $NATIVE_OPS -eq 1 ]] && _out+=(--native-ops)
  _out+=(--steps "$STEPS")
  _out+=(--gpu "$GPU_N")
  [[ -n "$NP" ]] && _out+=(--np "$NP")
  _out+=(--mpi-cmd "$MPI_CMD")
  _out+=(--n1 "$N1" --n2 "$N2")
  [[ -n "$BOX" ]] && _out+=(--box "$BOX")
  [[ $TREE_FMM_GPU_AWARE_MPI -eq 1 ]] && _out+=(--tree-fmm-gpu-aware-mpi)
  [[ $TREE_FMM_ASSUME_GPU_AWARE_MPI -eq 1 ]] && _out+=(--tree-fmm-assume-gpu-aware-mpi)
  [[ $TREE_FMM_DEVICE_LOCAL_EVAL -eq 1 ]] && _out+=(--tree-fmm-device-local-eval)
  [[ -n "$TREE_FMM_REUSE_POSITION_TOL" ]] && _out+=(--tree-fmm-reuse-position-tol "$TREE_FMM_REUSE_POSITION_TOL")
  [[ ${#FIELD_VALUES[@]} -eq 3 ]] && _out+=(--field-values "${FIELD_VALUES[@]}")
  [[ ${#MFIELD_VALUES[@]} -eq 3 ]] && _out+=(--mfield-values "${MFIELD_VALUES[@]}")
  [[ ${#FIELD6_VALUES[@]} -eq 6 ]] && _out+=(--field6-values "${FIELD6_VALUES[@]}")
  [[ ${#FIELD9_VALUES[@]} -eq 9 ]] && _out+=(--field9-values "${FIELD9_VALUES[@]}")
  [[ -n "$FIDELITY_VALUE" ]] && _out+=(--fidelity-value "$FIDELITY_VALUE")
  [[ -n "$EXPORT_FIDELITY_ID" ]] && _out+=(--export-fidelity-id "$EXPORT_FIDELITY_ID")
  [[ "$NUM_FIDELITY_LEVELS" != "0" ]] && _out+=(--num-fidelity-levels "$NUM_FIDELITY_LEVELS")
  [[ "$MULTI_FIDELITY_MODE" != "conditioning" ]] && _out+=(--multi-fidelity-mode "$MULTI_FIDELITY_MODE")
  [[ $TEST_PHYS_COMPUTE -eq 1 ]] && _out+=(--test-phys-compute)
  [[ $OPEN_BOUNDARY_MD -eq 1 ]] && _out+=(--open-boundary-md)
  return 0
}

run_throughput_compare() {
  local script_path="$1"
  local base_dir="$2"
  local enabled_dir="$3"
  local base_log="$base_dir/console.log"
  local enabled_log="$enabled_dir/console.log"
  local -a base_args=()
  local -a enabled_args=()

  append_common_cli_args base_args
  append_common_cli_args enabled_args
  base_args+=(--out-dir "$base_dir" --force-md-run)
  enabled_args+=(--out-dir "$enabled_dir" --force-md-run)

  if [[ $TEST_LONG_RANGE -eq 1 ]]; then
    enabled_args+=(--test-long-range)
  elif [[ $TEST_RECIPROCAL_LONG_RANGE -eq 1 ]]; then
    enabled_args+=(--test-reciprocal-long-range)
  elif [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
    base_args+=(--open-boundary-md)
    enabled_args+=(--open-boundary-md)
    enabled_args+=(--test-tree-fmm-long-range)
  elif [[ $TEST_ISOLATED_FAR_FIELD -eq 1 ]]; then
    base_args+=(--open-boundary-md)
    enabled_args+=(--open-boundary-md)
    enabled_args+=(--test-isolated-far-field)
  elif [[ $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]]; then
    base_args+=(--open-boundary-md)
    enabled_args+=(--open-boundary-md)
    enabled_args+=(--test-isolated-far-field-v2)
  else
    enabled_args+=(--test-feature-spectral-fft)
  fi

  if [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
    echo "[throughput] tree_fmm compare uses open-boundary MD and the CUDA runtime now prefers the pure-GPU local tree path."
    if [[ $TREE_FMM_DEVICE_LOCAL_EVAL -eq 1 || $TREE_FMM_GPU_AWARE_MPI -eq 1 || -n "$TREE_FMM_REUSE_POSITION_TOL" ]]; then
      echo "[throughput] tree_fmm runtime env: gpu_aware_mpi=$TREE_FMM_GPU_AWARE_MPI assume_gpu_aware=$TREE_FMM_ASSUME_GPU_AWARE_MPI device_local_eval=$TREE_FMM_DEVICE_LOCAL_EVAL reuse_tol=${TREE_FMM_REUSE_POSITION_TOL:-0}"
    fi
    echo "[throughput] if you want a larger benchmark, explicitly raise --steps/--n1/--n2 after this sanity run."
    if [[ $TREE_FMM_THROUGHPUT_AUTO_TUNED -eq 1 ]]; then
      echo "[throughput] auto-tuned tree_fmm workload to --steps $STEPS --n1 $N1 --n2 $N2 for a faster sanity benchmark"
    fi
  elif [[ $TEST_ISOLATED_FAR_FIELD -eq 1 ]]; then
    echo "[throughput] isolated-far-field compare uses open-boundary MD for both baseline and enabled runs."
    echo "[throughput] if you want a larger benchmark, explicitly raise --steps/--n1/--n2 after this sanity run."
  elif [[ $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]]; then
    echo "[throughput] isolated-far-field-v2 compare uses open-boundary MD for both baseline and enabled runs."
    echo "[throughput] if you want a larger benchmark, explicitly raise --steps/--n1/--n2 after this sanity run."
  fi

  echo "[throughput] baseline run: no long-range feature/recriprocal path"
  if ! {
    printf '[throughput] baseline command: bash %q' "$script_path"
    printf ' %q' "${base_args[@]}"
    printf '\n'
    bash "$script_path" "${base_args[@]}"
  } > "$base_log" 2>&1; then
    echo "[throughput] baseline run failed; see $base_log"
    return 1
  fi
  echo "[throughput] enabled run: selected long-range / feature-FFT path"
  if ! {
    printf '[throughput] enabled command: bash %q' "$script_path"
    printf ' %q' "${enabled_args[@]}"
    printf '\n'
    bash "$script_path" "${enabled_args[@]}"
  } > "$enabled_log" 2>&1; then
    echo "[throughput] enabled run failed; see $enabled_log"
    return 1
  fi

  python - <<'PY' "$base_log" "$enabled_log"
import re
import sys
from pathlib import Path

base_log = Path(sys.argv[1])
enabled_log = Path(sys.argv[2])
pat = re.compile(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")

def parse(path: Path):
    text = path.read_text()
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"未在 {path} 中找到 LAMMPS loop time")
    loop = float(m.group(1))
    procs = int(m.group(2))
    steps = int(m.group(3))
    atoms = int(m.group(4))
    steps_per_s = steps / loop if loop > 0 else float("inf")
    katom_steps_per_s = (steps * atoms) / loop / 1000.0 if loop > 0 else float("inf")
    return {
        "loop": loop,
        "procs": procs,
        "steps": steps,
        "atoms": atoms,
        "steps_per_s": steps_per_s,
        "katom_steps_per_s": katom_steps_per_s,
    }

base = parse(base_log)
enabled = parse(enabled_log)
slowdown = enabled["loop"] / base["loop"] if base["loop"] > 0 else float("inf")
throughput_ratio = enabled["steps_per_s"] / base["steps_per_s"] if base["steps_per_s"] > 0 else 0.0

print("[throughput] baseline")
print(f"  loop_time_s         = {base['loop']:.6f}")
print(f"  steps_per_s         = {base['steps_per_s']:.6f}")
print(f"  katom_steps_per_s   = {base['katom_steps_per_s']:.6f}")
print("[throughput] long-range enabled")
print(f"  loop_time_s         = {enabled['loop']:.6f}")
print(f"  steps_per_s         = {enabled['steps_per_s']:.6f}")
print(f"  katom_steps_per_s   = {enabled['katom_steps_per_s']:.6f}")
print("[throughput] compare")
print(f"  slowdown_x          = {slowdown:.6f}")
print(f"  throughput_ratio    = {throughput_ratio:.6f}")
PY
}

run_mpi_lammps() {
  local mpi_cmd="$1"
  local mpi_np="$2"
  local lmp_bin="$3"
  shift 3
  local -a lmp_args=("$@")

  local rank_wrapper='
rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-${LOCAL_RANK:-0}}}}}"
orig="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$orig" ]]; then
  IFS="," read -r -a gpus <<< "$orig"
  if (( rank < ${#gpus[@]} )); then
    export CUDA_VISIBLE_DEVICES="${gpus[$rank]}"
  else
    export CUDA_VISIBLE_DEVICES="$rank"
  fi
else
  export CUDA_VISIBLE_DEVICES="$rank"
fi
exec "$@"
'

  export CUDA_DEVICE_ORDER=PCI_BUS_ID

  if [[ "$mpi_cmd" == "srun" ]]; then
    "$mpi_cmd" -n "$mpi_np" --cpu-bind=none bash -c "$rank_wrapper" bash "$lmp_bin" "${lmp_args[@]}"
    return
  fi

  local -a mpi_extra=(-x LD_LIBRARY_PATH -x CUDA_DEVICE_ORDER)
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && mpi_extra+=(-x CUDA_VISIBLE_DEVICES)
  [[ -n "${MFF_TREE_FMM_GPU_AWARE_MPI:-}" ]] && mpi_extra+=(-x MFF_TREE_FMM_GPU_AWARE_MPI)
  [[ -n "${MFF_TREE_FMM_ASSUME_GPU_AWARE_MPI:-}" ]] && mpi_extra+=(-x MFF_TREE_FMM_ASSUME_GPU_AWARE_MPI)
  [[ -n "${MFF_TREE_FMM_DEVICE_LOCAL_EVAL:-}" ]] && mpi_extra+=(-x MFF_TREE_FMM_DEVICE_LOCAL_EVAL)
  [[ -n "${MFF_TREE_FMM_REUSE_POSITION_TOL:-}" ]] && mpi_extra+=(-x MFF_TREE_FMM_REUSE_POSITION_TOL)
  "$mpi_cmd" -np "$mpi_np" "${mpi_extra[@]}" --bind-to none \
    bash -c "$rank_wrapper" bash "$lmp_bin" "${lmp_args[@]}"
}

if [[ $COMPARE_THROUGHPUT -eq 1 && $FORCE_MD_RUN -ne 1 ]]; then
  BASELINE_OUT_DIR="$OUT_DIR/throughput_baseline"
  ENABLED_OUT_DIR="$OUT_DIR/throughput_enabled"
  mkdir -p "$BASELINE_OUT_DIR" "$ENABLED_OUT_DIR"
  run_throughput_compare "${BASH_SOURCE[0]}" "$BASELINE_OUT_DIR" "$ENABLED_OUT_DIR"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi

FEATURE_SPECTRAL_EXPORT=0
if [[ $TEST_FEATURE_SPECTRAL_FFT -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_TRICLINIC -eq 1 || $TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY -eq 1 ]]; then
  FEATURE_SPECTRAL_EXPORT=1
fi

if [[ -z "$PTH" && $DUMMY_ICTD -eq 1 ]]; then
  if [[ $TEST_RECIPROCAL_LONG_RANGE -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with reciprocal-spectral-v1)"
  elif [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with latent-coulomb tree_fmm)"
  elif [[ $TEST_ISOLATED_FAR_FIELD -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with isolated-far-field-v1)"
  elif [[ $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with isolated-far-field-v2)"
  elif [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with feature-space FFT block, slab z-open sanity)"
  elif [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with feature-space FFT block, slab smoke)"
  elif [[ $TEST_FEATURE_SPECTRAL_FFT -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with feature-space FFT block)"
  elif [[ $TEST_LONG_RANGE -eq 1 ]]; then
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint (with long-range latent-coulomb)"
  else
    echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint"
  fi
  PTH="$OUT_DIR/dummy_ictd.pth"
  EXTERNAL_IRREP_PY="None"
  EXTERNAL_SPECS_PY="None"
  if [[ ${#FIELD_VALUES[@]} -eq 3 && ${#MFIELD_VALUES[@]} -eq 3 ]]; then
    EXTERNAL_SPECS_PY="[{\"name\": \"external_field\", \"rank\": 1, \"irrep\": \"1o\"}, {\"name\": \"magnetic_field\", \"rank\": 1, \"irrep\": \"1e\"}]"
  elif [[ ${#MFIELD_VALUES[@]} -eq 3 ]]; then
    EXTERNAL_IRREP_PY="'1e'"
  elif [[ ${#FIELD_VALUES[@]} -eq 3 ]]; then
    EXTERNAL_IRREP_PY="'1o'"
  fi
  python - <<PY
import torch
from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd
out = r"$PTH"
external_rank = None
if ${#FIELD_VALUES[@]} == 3 or ${#MFIELD_VALUES[@]} == 3:
    external_rank = 1
elif ${#FIELD6_VALUES[@]} == 6 or ${#FIELD9_VALUES[@]} == 9:
    external_rank = 2
physical_tensor_outputs = None
if $DUMMY_PHYS_HEADS == 1:
    physical_tensor_outputs = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "dipole_per_atom": {"ls": [1], "channels_out": 1, "reduce": "none"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
        "polarizability_per_atom": {"ls": [0, 2], "channels_out": 1, "reduce": "none"},
        "born_effective_charge_per_atom": {"ls": [0, 1, 2], "channels_out": 1, "reduce": "none"},
    }
if $TEST_RECIPROCAL_LONG_RANGE == 1:
    long_range_mode = "reciprocal-spectral-v1"
    long_range_boundary = "slab" if ($TEST_PBC_SLAB == 1 or $TEST_PBC_SLAB_Z_OPEN == 1) else "periodic"
    long_range_filter_hidden_dim = 32
    long_range_kmax = 2
    long_range_backend = "dense_pairwise"
    long_range_theta = 0.5
    long_range_leaf_size = 32
elif $TEST_TREE_FMM_LONG_RANGE == 1:
    long_range_mode = "latent-coulomb"
    long_range_boundary = "nonperiodic"
    long_range_filter_hidden_dim = 64
    long_range_kmax = 2
    long_range_backend = "tree_fmm"
    long_range_theta = 0.4
    long_range_leaf_size = 4
elif $TEST_ISOLATED_FAR_FIELD == 1:
    long_range_mode = "isolated-far-field-v1"
    long_range_boundary = "nonperiodic"
    long_range_filter_hidden_dim = 64
    long_range_kmax = 2
    long_range_backend = "dense_pairwise"
    long_range_theta = 0.5
    long_range_leaf_size = 32
elif $TEST_ISOLATED_FAR_FIELD_V2 == 1:
    long_range_mode = "isolated-far-field-v2"
    long_range_boundary = "nonperiodic"
    long_range_filter_hidden_dim = 64
    long_range_kmax = 2
    long_range_backend = "dense_pairwise"
    long_range_theta = 0.5
    long_range_leaf_size = 32
elif $TEST_LONG_RANGE == 1:
    long_range_mode = "latent-coulomb"
    long_range_boundary = "periodic"
    long_range_filter_hidden_dim = 64
    long_range_kmax = 2
    long_range_backend = "dense_pairwise"
    long_range_theta = 0.5
    long_range_leaf_size = 32
else:
    long_range_mode = "none"
    long_range_boundary = "nonperiodic"
    long_range_filter_hidden_dim = 64
    long_range_kmax = 2
    long_range_backend = "dense_pairwise"
    long_range_theta = 0.5
    long_range_leaf_size = 32
feature_spectral_mode = "fft" if $FEATURE_SPECTRAL_EXPORT == 1 else "none"
feature_spectral_boundary = "slab" if ($TEST_FEATURE_SPECTRAL_FFT_SLAB == 1 or $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN == 1) else "periodic"
_make_dummy_checkpoint_pure_cartesian_ictd(
    out,
    device=torch.device("cpu"),
    external_tensor_rank=external_rank,
    external_tensor_irrep=$EXTERNAL_IRREP_PY,
    external_tensor_specs=$EXTERNAL_SPECS_PY,
    num_fidelity_levels=int("$NUM_FIDELITY_LEVELS"),
    multi_fidelity_mode="$MULTI_FIDELITY_MODE",
    physical_tensor_outputs=physical_tensor_outputs,
    long_range_mode=long_range_mode,
    long_range_hidden_dim=32,
    long_range_boundary=long_range_boundary,
    long_range_neutralize=True,
    long_range_filter_hidden_dim=long_range_filter_hidden_dim,
    long_range_kmax=long_range_kmax,
    long_range_mesh_size=16,
    long_range_slab_padding_factor=2,
    long_range_include_k0=False,
    long_range_source_channels=1,
    long_range_backend=long_range_backend,
    long_range_reciprocal_backend="mesh_fft" if $TEST_RECIPROCAL_LONG_RANGE == 1 else "direct_kspace",
    long_range_energy_partition="potential",
    long_range_green_mode="poisson",
    long_range_assignment="cic",
    long_range_theta=long_range_theta,
    long_range_leaf_size=long_range_leaf_size,
    long_range_multipole_order=0,
    feature_spectral_mode=feature_spectral_mode,
    feature_spectral_bottleneck_dim=8,
    feature_spectral_mesh_size=16,
    feature_spectral_filter_hidden_dim=32,
    feature_spectral_boundary=feature_spectral_boundary,
    feature_spectral_slab_padding_factor=2,
    feature_spectral_neutralize=True,
    feature_spectral_include_k0=False,
    feature_spectral_gate_init=0.0,
)
print("wrote", out)
PY
fi
if [[ -z "$PTH" && $DUMMY_ICTD_O3 -eq 1 ]]; then
  echo "[0/3] Generating pure-cartesian-ictd-o3 dummy checkpoint"
  PTH="$OUT_DIR/dummy_ictd_o3.pth"
  EXTERNAL_IRREP_PY="None"
  EXTERNAL_SPECS_PY="None"
  O3_ACTIVE_IRREPS_PY="None"
  if [[ ${#FIELD_VALUES[@]} -eq 3 && ${#MFIELD_VALUES[@]} -eq 3 ]]; then
    EXTERNAL_SPECS_PY="[{\"name\": \"external_field\", \"rank\": 1, \"irrep\": \"1o\"}, {\"name\": \"magnetic_field\", \"rank\": 1, \"irrep\": \"1e\"}]"
    O3_ACTIVE_IRREPS_PY="'0e,1e,1o,2e'"
  elif [[ ${#MFIELD_VALUES[@]} -eq 3 ]]; then
    EXTERNAL_IRREP_PY="'1e'"
    O3_ACTIVE_IRREPS_PY="'0e,1e,2e'"
  elif [[ ${#FIELD_VALUES[@]} -eq 3 ]]; then
    EXTERNAL_IRREP_PY="'1o'"
    O3_ACTIVE_IRREPS_PY="'0e,1o,2e'"
  fi
  python - <<PY
import torch
from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd_o3
out = r"$PTH"
external_rank = None
if ${#FIELD_VALUES[@]} == 3 or ${#MFIELD_VALUES[@]} == 3:
    external_rank = 1
elif ${#FIELD6_VALUES[@]} == 6 or ${#FIELD9_VALUES[@]} == 9:
    external_rank = 2
physical_tensor_outputs = None
if $DUMMY_PHYS_HEADS == 1:
    physical_tensor_outputs = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "dipole_per_atom": {"ls": [1], "channels_out": 1, "reduce": "none"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
        "polarizability_per_atom": {"ls": [0, 2], "channels_out": 1, "reduce": "none"},
        "born_effective_charge_per_atom": {"irreps": ["0e", "1e", "2e"], "channels_out": 1, "reduce": "none"},
    }
_make_dummy_checkpoint_pure_cartesian_ictd_o3(
    out,
    device=torch.device("cpu"),
    external_tensor_rank=external_rank,
    external_tensor_irrep=$EXTERNAL_IRREP_PY,
    external_tensor_specs=$EXTERNAL_SPECS_PY,
    num_fidelity_levels=int("$NUM_FIDELITY_LEVELS"),
    multi_fidelity_mode="$MULTI_FIDELITY_MODE",
    physical_tensor_outputs=physical_tensor_outputs,
    o3_irrep_preset="auto",
    o3_active_irreps=$O3_ACTIVE_IRREPS_PY,
)
print("wrote", out)
PY
fi
if [[ -z "$PTH" && $DUMMY_CUE -eq 1 ]]; then
  if [[ $TEST_ISOLATED_FAR_FIELD -eq 1 ]]; then
    echo "[0/3] Generating spherical-save-cue dummy checkpoint (with isolated-far-field-v1)"
  elif [[ $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 ]]; then
    echo "[0/3] Generating spherical-save-cue dummy checkpoint (with isolated-far-field-v2)"
  elif [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN -eq 1 ]]; then
    echo "[0/3] Generating spherical-save-cue dummy checkpoint (with feature-space FFT block, slab z-open sanity)"
  elif [[ $TEST_FEATURE_SPECTRAL_FFT_SLAB -eq 1 ]]; then
    echo "[0/3] Generating spherical-save-cue dummy checkpoint (with feature-space FFT block, slab smoke)"
  elif [[ $TEST_FEATURE_SPECTRAL_FFT -eq 1 ]]; then
    echo "[0/3] Generating spherical-save-cue dummy checkpoint (with feature-space FFT block)"
  else
    echo "[0/3] Generating spherical-save-cue dummy checkpoint"
  fi
  PTH="$OUT_DIR/dummy_cue.pth"
  python - <<PY
import torch
from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint_spherical_save_cue
out = r"$PTH"
if $TEST_ISOLATED_FAR_FIELD == 1:
    long_range_mode = "isolated-far-field-v1"
    long_range_boundary = "nonperiodic"
elif $TEST_ISOLATED_FAR_FIELD_V2 == 1:
    long_range_mode = "isolated-far-field-v2"
    long_range_boundary = "nonperiodic"
else:
    long_range_mode = "none"
    long_range_boundary = "nonperiodic"
_make_dummy_checkpoint_spherical_save_cue(
    out,
    device=torch.device("cpu"),
    long_range_mode=long_range_mode,
    long_range_hidden_dim=32,
    long_range_boundary=long_range_boundary,
    long_range_neutralize=True,
    long_range_filter_hidden_dim=64,
    long_range_kmax=2,
    long_range_mesh_size=16,
    long_range_slab_padding_factor=2,
    long_range_include_k0=False,
    long_range_source_channels=1,
    long_range_backend="dense_pairwise",
    long_range_reciprocal_backend="direct_kspace",
    long_range_energy_partition="potential",
    long_range_green_mode="poisson",
    long_range_assignment="cic",
    long_range_theta=0.5,
    long_range_leaf_size=32,
    long_range_multipole_order=0,
    feature_spectral_mode="fft" if $FEATURE_SPECTRAL_EXPORT == 1 else "none",
    feature_spectral_bottleneck_dim=8,
    feature_spectral_mesh_size=16,
    feature_spectral_filter_hidden_dim=32,
    feature_spectral_boundary="slab" if ($TEST_FEATURE_SPECTRAL_FFT_SLAB == 1 or $TEST_FEATURE_SPECTRAL_FFT_SLAB_Z_OPEN == 1) else "periodic",
    feature_spectral_slab_padding_factor=2,
    feature_spectral_neutralize=True,
    feature_spectral_include_k0=False,
    feature_spectral_gate_init=0.0,
)
print("wrote", out)
PY
fi

echo "[1/3] Exporting core.pt (TorchScript, embed E0 optional)"
if [[ ($DUMMY_ICTD -eq 1 || $DUMMY_CUE -eq 1) && -z "$E0CSV" && $DUMMY_E0 -eq 1 ]]; then
  # Generate simple fitted_E0.csv (Atom,E0) for embed E0 test
  E0CSV="$OUT_DIR/fitted_E0.csv"
  echo "Atom,E0" > "$E0CSV"
  # For H/O etc. integration test; unknown elements skipped (E0=0).
  python - "${ELEMENTS[@]}" <<'PY' >> "$E0CSV"
import sys

sym2Z = {
  "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
  "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
}

seen = set()
for s in sys.argv[1:]:
  s = (s or "").strip()
  if not s or s in seen:
    continue
  seen.add(s)
  z = sym2Z.get(s)
  if z is None:
    continue
  # Simple test value: E0 = -0.1 * Z (eV)
  print(f"{z},{-0.1*z:.8f}")
PY
  echo "[0/3] Generated dummy E0 CSV: $E0CSV"
fi

EXPORT_ARGS=(--checkpoint "$PTH" --elements "${ELEMENTS[@]}" --device cuda --max-radius "$CUTOFF" --out "$CORE_PT" --embed-e0)
if [[ -n "$MODE" ]]; then EXPORT_ARGS+=(--mode "$MODE"); fi
if [[ $NATIVE_OPS -eq 1 ]]; then EXPORT_ARGS+=(--native-ops); fi
if [[ -n "$DTYPE" ]]; then EXPORT_ARGS+=(--dtype "$DTYPE"); fi
if [[ -n "$E0CSV" ]]; then EXPORT_ARGS+=(--e0-csv "$E0CSV"); fi
if [[ -n "$EXPORT_FIDELITY_ID" ]]; then EXPORT_ARGS+=(--export-fidelity-id "$EXPORT_FIDELITY_ID"); fi
if [[ $FEATURE_SPECTRAL_EXPORT -eq 1 || $TEST_RECIPROCAL_LONG_RANGE -eq 1 || $TEST_TREE_FMM_LONG_RANGE -eq 1 ]]; then EXPORT_ARGS+=(--export-reciprocal-source); fi
# Export always on cuda:0 to avoid MPI env interference
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/molecular_force_field/cli/export_libtorch_core.py" "${EXPORT_ARGS[@]}"

echo "[2/3] Writing LAMMPS input file"
if [[ -z "$BOX" ]]; then
  BOX="$(python - <<PY
import math
n = int("$N1")+int("$N2")
box = max(60.0, 3.0*(n**(1.0/3.0))*2.5)
print(f"{box:.3f}")
PY
)"
fi

run_pbc_minimum_image_test() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local element="$4"
  local cutoff="$5"
  local gpu_n="$6"
  local mpi_np="$7"
  local mpi_cmd="$8"

  cat > "$workdir/two_atoms_cross.data" <<'EOF'
LAMMPS data file for MFFTorch PBC test: cross-boundary pair

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms

1 1 0.1 5.0 5.0
2 1 9.9 5.0 5.0
EOF

  cat > "$workdir/two_atoms_inside.data" <<'EOF'
LAMMPS data file for MFFTorch PBC test: same minimum-image pair inside box

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms

1 1 0.1 5.0 5.0
2 1 0.3 5.0 5.0
EOF

  cat > "$workdir/in.cross" <<EOF
units metal
atom_style atomic
boundary p p p

read_data $workdir/two_atoms_cross.data

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/cross.dump id type x y z fx fy fz modify sort id
EOF

  cat > "$workdir/in.inside" <<EOF
units metal
atom_style atomic
boundary p p p

read_data $workdir/two_atoms_inside.data

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/inside.dump id type x y z fx fy fz modify sort id
EOF

  local -a lmp_args_cross=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.cross")
  local -a lmp_args_inside=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.inside")

  if [[ "$mpi_np" -gt 1 ]]; then
    echo "[3/3] PBC minimum-image test (multi-GPU): $mpi_cmd -np $mpi_np ..."
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_cross[@]}" > "$workdir/cross.log" 2>&1
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_inside[@]}" > "$workdir/inside.log" 2>&1
  else
    echo "[3/3] PBC minimum-image test (single-GPU)"
    "$lmp_bin" "${lmp_args_cross[@]}" > "$workdir/cross.log" 2>&1
    "$lmp_bin" "${lmp_args_inside[@]}" > "$workdir/inside.log" 2>&1
  fi

  python - <<'PY' "$workdir/cross.log" "$workdir/inside.log" "$workdir/cross.dump" "$workdir/inside.dump"
import sys
from pathlib import Path

cross_log, inside_log, cross_dump, inside_dump = map(Path, sys.argv[1:5])

def parse_pe(log_path: Path) -> float:
    lines = log_path.read_text().splitlines()
    step_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            step_idx = i
    if step_idx is None or step_idx + 1 >= len(lines):
        raise RuntimeError(f"未在 {log_path} 中找到 thermo PotEng 输出")
    vals = lines[step_idx + 1].split()
    return float(vals[1])

def parse_dump_forces(dump_path: Path):
    lines = dump_path.read_text().splitlines()
    idx = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    atoms = []
    for line in lines[idx + 1: idx + 3]:
        parts = line.split()
        atoms.append((int(parts[0]), [float(parts[5]), float(parts[6]), float(parts[7])]))
    atoms.sort(key=lambda x: x[0])
    return [f for _, f in atoms]

pe_cross = parse_pe(cross_log)
pe_inside = parse_pe(inside_log)
forces_cross = parse_dump_forces(cross_dump)
forces_inside = parse_dump_forces(inside_dump)

energy_diff = abs(pe_cross - pe_inside)
max_force_diff = 0.0
for fa, fb in zip(forces_cross, forces_inside):
    max_force_diff = max(max_force_diff, max(abs(a - b) for a, b in zip(fa, fb)))

print(f"cross PotEng   = {pe_cross:.12e}")
print(f"inside PotEng  = {pe_inside:.12e}")
print(f"|dE|           = {energy_diff:.12e}")
print(f"max |dF_atom|  = {max_force_diff:.12e}")

ENERGY_TOL = 1e-6
FORCE_TOL = 1e-5
if energy_diff > ENERGY_TOL or max_force_diff > FORCE_TOL:
    print("\nFAILED: 跨边界构型与盒内等价构型不一致。")
    sys.exit(1)

print("\nPASSED: 跨边界构型与盒内等价构型在容差内一致。")
PY
}

run_pbc_slab_test() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local element="$4"
  local cutoff="$5"
  local gpu_n="$6"
  local mpi_np="$7"
  local mpi_cmd="$8"

  cat > "$workdir/two_atoms_slab_cross.data" <<'EOF'
LAMMPS data file for MFFTorch slab test: lateral cross-boundary pair with nonperiodic z

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0

Atoms

1 1 0.1 5.0 10.0
2 1 9.9 5.0 10.0
EOF

  cat > "$workdir/two_atoms_slab_inside.data" <<'EOF'
LAMMPS data file for MFFTorch slab test: same lateral minimum-image pair inside box

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0

Atoms

1 1 0.1 5.0 10.0
2 1 0.3 5.0 10.0
EOF

  cat > "$workdir/in.slab.cross" <<EOF
units metal
atom_style atomic
boundary p p f

read_data $workdir/two_atoms_slab_cross.data

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/slab_cross.dump id type x y z fx fy fz modify sort id
EOF

  cat > "$workdir/in.slab.inside" <<EOF
units metal
atom_style atomic
boundary p p f

read_data $workdir/two_atoms_slab_inside.data

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/slab_inside.dump id type x y z fx fy fz modify sort id
EOF

  local -a lmp_args_cross=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.slab.cross")
  local -a lmp_args_inside=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.slab.inside")

  if [[ "$mpi_np" -gt 1 ]]; then
    echo "[3/3] 2D slab lateral-PBC test (multi-GPU): $mpi_cmd -np $mpi_np ..."
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_cross[@]}" > "$workdir/slab_cross.log" 2>&1
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_inside[@]}" > "$workdir/slab_inside.log" 2>&1
  else
    echo "[3/3] 2D slab lateral-PBC test (single-GPU)"
    "$lmp_bin" "${lmp_args_cross[@]}" > "$workdir/slab_cross.log" 2>&1
    "$lmp_bin" "${lmp_args_inside[@]}" > "$workdir/slab_inside.log" 2>&1
  fi

  python - <<'PY' "$workdir/slab_cross.log" "$workdir/slab_inside.log" "$workdir/slab_cross.dump" "$workdir/slab_inside.dump"
import sys
from pathlib import Path

cross_log, inside_log, cross_dump, inside_dump = map(Path, sys.argv[1:5])

def parse_pe(log_path: Path) -> float:
    lines = log_path.read_text().splitlines()
    step_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            step_idx = i
    if step_idx is None or step_idx + 1 >= len(lines):
        raise RuntimeError(f"未在 {log_path} 中找到 thermo PotEng 输出")
    vals = lines[step_idx + 1].split()
    return float(vals[1])

def parse_dump_forces(dump_path: Path):
    lines = dump_path.read_text().splitlines()
    idx = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    atoms = []
    for line in lines[idx + 1: idx + 3]:
        parts = line.split()
        atoms.append((int(parts[0]), [float(parts[5]), float(parts[6]), float(parts[7])]))
    atoms.sort(key=lambda x: x[0])
    return [f for _, f in atoms]

pe_cross = parse_pe(cross_log)
pe_inside = parse_pe(inside_log)
forces_cross = parse_dump_forces(cross_dump)
forces_inside = parse_dump_forces(inside_dump)

energy_diff = abs(pe_cross - pe_inside)
max_force_diff = 0.0
for fa, fb in zip(forces_cross, forces_inside):
    max_force_diff = max(max_force_diff, max(abs(a - b) for a, b in zip(fa, fb)))

print(f"slab cross PotEng   = {pe_cross:.12e}")
print(f"slab inside PotEng  = {pe_inside:.12e}")
print(f"|dE|                = {energy_diff:.12e}")
print(f"max |dF_atom|       = {max_force_diff:.12e}")

ENERGY_TOL = 1e-6
FORCE_TOL = 1e-5
if energy_diff > ENERGY_TOL or max_force_diff > FORCE_TOL:
    print("\nFAILED: `boundary p p f` 下横向跨边界构型与盒内等价构型不一致。")
    sys.exit(1)

print("\nPASSED: `boundary p p f` 下横向跨边界构型与盒内等价构型在容差内一致。")
PY
}

run_pbc_slab_z_open_test() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local element="$4"
  local cutoff="$5"
  local gpu_n="$6"
  local mpi_np="$7"
  local mpi_cmd="$8"

  cat > "$workdir/two_atoms_slab_zfar.data" <<'EOF'
LAMMPS data file for MFFTorch slab z-open sanity: large z separation must stay large

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0

Atoms

1 1 5.0 5.0 0.1
2 1 5.0 5.0 19.9
EOF

  cat > "$workdir/two_atoms_slab_znear.data" <<'EOF'
LAMMPS data file for MFFTorch slab z-open sanity: true short-range z pair

2 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0

Atoms

1 1 5.0 5.0 10.0
2 1 5.0 5.0 10.2
EOF

  cat > "$workdir/in.slab.zfar" <<EOF
units metal
atom_style atomic
boundary p p f

read_data $workdir/two_atoms_slab_zfar.data

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/slab_zfar.dump id type x y z fx fy fz modify sort id
EOF

  cat > "$workdir/in.slab.znear" <<EOF
units metal
atom_style atomic
boundary p p f

read_data $workdir/two_atoms_slab_znear.data

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/slab_znear.dump id type x y z fx fy fz modify sort id
EOF

  local -a lmp_args_far=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.slab.zfar")
  local -a lmp_args_near=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.slab.znear")

  if [[ "$mpi_np" -gt 1 ]]; then
    echo "[3/3] 2D slab z-open sanity (multi-GPU): $mpi_cmd -np $mpi_np ..."
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_far[@]}" > "$workdir/slab_zfar.log" 2>&1
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_near[@]}" > "$workdir/slab_znear.log" 2>&1
  else
    echo "[3/3] 2D slab z-open sanity (single-GPU)"
    "$lmp_bin" "${lmp_args_far[@]}" > "$workdir/slab_zfar.log" 2>&1
    "$lmp_bin" "${lmp_args_near[@]}" > "$workdir/slab_znear.log" 2>&1
  fi

  python - <<'PY' "$workdir/slab_zfar.log" "$workdir/slab_znear.log" "$workdir/slab_zfar.dump" "$workdir/slab_znear.dump"
import sys
from pathlib import Path

zfar_log, znear_log, zfar_dump, znear_dump = map(Path, sys.argv[1:5])

def parse_pe(log_path: Path) -> float:
    lines = log_path.read_text().splitlines()
    step_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            step_idx = i
    if step_idx is None or step_idx + 1 >= len(lines):
        raise RuntimeError(f"未在 {log_path} 中找到 thermo PotEng 输出")
    vals = lines[step_idx + 1].split()
    return float(vals[1])

def parse_force_magnitudes(dump_path: Path):
    lines = dump_path.read_text().splitlines()
    idx = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    vals = []
    for line in lines[idx + 1: idx + 3]:
        parts = line.split()
        fx, fy, fz = map(float, parts[5:8])
        vals.append(max(abs(fx), abs(fy), abs(fz)))
    return max(vals)

pe_far = parse_pe(zfar_log)
pe_near = parse_pe(znear_log)
force_far = parse_force_magnitudes(zfar_dump)
force_near = parse_force_magnitudes(znear_dump)

energy_diff = abs(pe_far - pe_near)
force_diff = abs(force_far - force_near)

print(f"slab z-far PotEng    = {pe_far:.12e}")
print(f"slab z-near PotEng   = {pe_near:.12e}")
print(f"|dE|                 = {energy_diff:.12e}")
print(f"max |F| far          = {force_far:.12e}")
print(f"max |F| near         = {force_near:.12e}")
print(f"|d maxF|             = {force_diff:.12e}")

ENERGY_MIN_DIFF = 1e-3
FORCE_MIN_DIFF = 1e-4
if energy_diff < ENERGY_MIN_DIFF and force_diff < FORCE_MIN_DIFF:
    print("\nFAILED: `boundary p p f` 的 z-open sanity 没有区分远距 z 对与近距 z 对；可能把 z 错当成了周期方向。")
    sys.exit(1)

print("\nPASSED: `boundary p p f` 的 z-open sanity 通过，远距 z 对没有被误当成最小像近邻。")
PY
}

run_pbc_triclinic_test() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local element="$4"
  local cutoff="$5"
  local gpu_n="$6"
  local mpi_np="$7"
  local mpi_cmd="$8"

  cat > "$workdir/in.triclinic.cross" <<EOF
units metal
atom_style atomic
boundary p p p

region box prism 0 10 0 10 0 10 2.0 1.0 3.0
create_box 1 box
create_atoms 1 single 3.04 10.0 1.0 units box
create_atoms 1 single 1.32 0.4 1.0 units box
mass 1 1.0

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/triclinic_cross.dump id type x y z fx fy fz modify sort id
EOF

  cat > "$workdir/in.triclinic.inside" <<EOF
units metal
atom_style atomic
boundary p p p

region box prism 0 10 0 10 0 10 2.0 1.0 3.0
create_box 1 box
create_atoms 1 single 2.04 5.0 1.0 units box
create_atoms 1 single 2.32 5.4 1.0 units box
mass 1 1.0

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $workdir/triclinic_inside.dump id type x y z fx fy fz modify sort id
EOF

  local -a lmp_args_cross=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.triclinic.cross")
  local -a lmp_args_inside=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$workdir/in.triclinic.inside")

  if [[ "$mpi_np" -gt 1 ]]; then
    echo "[3/3] Triclinic PBC test (multi-GPU): $mpi_cmd -np $mpi_np ..."
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_cross[@]}" > "$workdir/triclinic_cross.log" 2>&1
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args_inside[@]}" > "$workdir/triclinic_inside.log" 2>&1
  else
    echo "[3/3] Triclinic PBC test (single-GPU)"
    "$lmp_bin" "${lmp_args_cross[@]}" > "$workdir/triclinic_cross.log" 2>&1
    "$lmp_bin" "${lmp_args_inside[@]}" > "$workdir/triclinic_inside.log" 2>&1
  fi

  python - <<'PY' "$workdir/triclinic_cross.log" "$workdir/triclinic_inside.log" "$workdir/triclinic_cross.dump" "$workdir/triclinic_inside.dump"
import sys
from pathlib import Path

cross_log, inside_log, cross_dump, inside_dump = map(Path, sys.argv[1:5])

def parse_pe(log_path: Path) -> float:
    lines = log_path.read_text().splitlines()
    step_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            step_idx = i
    if step_idx is None or step_idx + 1 >= len(lines):
        raise RuntimeError(f"未在 {log_path} 中找到 thermo PotEng 输出")
    vals = lines[step_idx + 1].split()
    return float(vals[1])

def parse_dump_forces(dump_path: Path):
    lines = dump_path.read_text().splitlines()
    idx = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    atoms = []
    for line in lines[idx + 1: idx + 3]:
        parts = line.split()
        atoms.append((int(parts[0]), [float(parts[5]), float(parts[6]), float(parts[7])]))
    atoms.sort(key=lambda x: x[0])
    return [f for _, f in atoms]

pe_cross = parse_pe(cross_log)
pe_inside = parse_pe(inside_log)
forces_cross = parse_dump_forces(cross_dump)
forces_inside = parse_dump_forces(inside_dump)

energy_diff = abs(pe_cross - pe_inside)
max_force_diff = 0.0
for fa, fb in zip(forces_cross, forces_inside):
    max_force_diff = max(max_force_diff, max(abs(a - b) for a, b in zip(fa, fb)))

print(f"triclinic cross PotEng   = {pe_cross:.12e}")
print(f"triclinic inside PotEng  = {pe_inside:.12e}")
print(f"|dE|                     = {energy_diff:.12e}")
print(f"max |dF_atom|            = {max_force_diff:.12e}")

ENERGY_TOL = 1e-6
FORCE_TOL = 1e-5
if energy_diff > ENERGY_TOL or max_force_diff > FORCE_TOL:
    print("\nFAILED: 三斜晶胞跨边界构型与盒内等价构型不一致。")
    sys.exit(1)

print("\nPASSED: 三斜晶胞跨边界构型与盒内等价构型在容差内一致。")
PY
}

run_pbc_minimum_image_mpi_consistency_test() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local element="$4"
  local cutoff="$5"
  local gpu_n="$6"
  local mpi_np="$7"
  local mpi_cmd="$8"

  local single_dir="$workdir/mpi_consistency_single"
  local multi_dir="$workdir/mpi_consistency_multi"
  mkdir -p "$single_dir" "$multi_dir"

  echo "[3/3] Feature FFT MPI consistency: running single-rank reference"
  run_pbc_minimum_image_test "$single_dir" "$lmp_bin" "$core_pt" "$element" "$cutoff" "1" "1" "$mpi_cmd"
  echo "[3/3] Feature FFT MPI consistency: running MPI case ($mpi_np ranks)"
  run_pbc_minimum_image_test "$multi_dir" "$lmp_bin" "$core_pt" "$element" "$cutoff" "$gpu_n" "$mpi_np" "$mpi_cmd"

  python - <<'PY' "$single_dir" "$multi_dir"
import sys
from pathlib import Path

single_dir = Path(sys.argv[1])
multi_dir = Path(sys.argv[2])

def parse_pe(log_path: Path) -> float:
    lines = log_path.read_text().splitlines()
    step_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            step_idx = i
    if step_idx is None or step_idx + 1 >= len(lines):
        raise RuntimeError(f"未在 {log_path} 中找到 thermo PotEng 输出")
    vals = lines[step_idx + 1].split()
    return float(vals[1])

def parse_dump_forces(dump_path: Path):
    lines = dump_path.read_text().splitlines()
    idx = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    atoms = []
    for line in lines[idx + 1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        atoms.append((int(parts[0]), [float(parts[5]), float(parts[6]), float(parts[7])]))
    atoms.sort(key=lambda x: x[0])
    return [f for _, f in atoms]

def max_force_diff(fa, fb):
    out = 0.0
    for a, b in zip(fa, fb):
        out = max(out, max(abs(x - y) for x, y in zip(a, b)))
    return out

cases = ["cross", "inside"]
ENERGY_TOL = 1e-6
FORCE_TOL = 1e-5
worst_de = 0.0
worst_df = 0.0

for case in cases:
    pe_single = parse_pe(single_dir / f"{case}.log")
    pe_multi = parse_pe(multi_dir / f"{case}.log")
    f_single = parse_dump_forces(single_dir / f"{case}.dump")
    f_multi = parse_dump_forces(multi_dir / f"{case}.dump")
    de = abs(pe_single - pe_multi)
    df = max_force_diff(f_single, f_multi)
    worst_de = max(worst_de, de)
    worst_df = max(worst_df, df)
    print(f"{case} single PotEng = {pe_single:.12e}")
    print(f"{case} mpi    PotEng = {pe_multi:.12e}")
    print(f"{case} |dE|         = {de:.12e}")
    print(f"{case} max |dF|     = {df:.12e}")

if worst_de > ENERGY_TOL or worst_df > FORCE_TOL:
    print("\nFAILED: np=1 与 np>1 的 feature-FFT PBC 结果不一致。")
    sys.exit(1)

print("\nPASSED: np=1 与 np>1 的 feature-FFT PBC 结果在容差内一致。")
PY
}

run_nonperiodic_tree_fmm_case() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local element="$4"
  local cutoff="$5"
  local gpu_n="$6"
  local mpi_np="$7"
  local mpi_cmd="$8"
  local case_name="${9}"

  local data_file="$workdir/${case_name}.data"
  local input_file="$workdir/in.${case_name}"
  local dump_file="$workdir/${case_name}.dump"
  local log_file="$workdir/${case_name}.log"

  cat > "$data_file" <<'EOF'
LAMMPS data file for MFFTorch open-boundary tree_fmm test

4 atoms
1 atom types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0

Atoms

1 1 4.0 5.0 5.0
2 1 6.0 5.0 5.0
3 1 14.0 5.0 5.0
4 1 16.0 5.0 5.0
EOF

  cat > "$input_file" <<EOF
units metal
atom_style atomic
boundary f f f

read_data $data_file

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch $cutoff cuda
pair_coeff * * $core_pt $element

thermo_style custom step pe
thermo 1
run 0

write_dump all custom $dump_file id type x y z fx fy fz modify sort id
EOF

  local -a lmp_args=(-k on g "$gpu_n" -sf kk -pk kokkos newton off neigh full -in "$input_file")
  if [[ "$mpi_np" -gt 1 ]]; then
    run_mpi_lammps "$mpi_cmd" "$mpi_np" "$lmp_bin" "${lmp_args[@]}" > "$log_file" 2>&1
  else
    "$lmp_bin" "${lmp_args[@]}" > "$log_file" 2>&1
  fi
}

run_nonperiodic_python_reference_test() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local checkpoint_pt="$4"
  local element="$5"
  local cutoff="$6"
  local e0_csv="${7}"
  local case_name="${8}"
  local banner="${9}"

  echo "[3/3] $banner"
  run_nonperiodic_tree_fmm_case "$workdir" "$lmp_bin" "$core_pt" "$element" "$cutoff" "1" "1" "mpirun" "$case_name"

  python - <<'PY' "$workdir/${case_name}.log" "$workdir/${case_name}.dump" "$workdir" "$checkpoint_pt" "$cutoff" "$element" "$e0_csv" "$case_name"
import sys
from pathlib import Path
import csv

from ase import Atoms
import torch

from molecular_force_field.active_learning.model_loader import build_e3trans_from_checkpoint
from molecular_force_field.evaluation.calculator import MyE3NNCalculator

log_path = Path(sys.argv[1])
dump_path = Path(sys.argv[2])
workdir = Path(sys.argv[3])
checkpoint = Path(sys.argv[4])
cutoff = float(sys.argv[5])
element = sys.argv[6]
e0_csv = Path(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else None
case_name = sys.argv[8] if len(sys.argv) > 8 else "nonperiodic"

def parse_pe(path: Path) -> float:
    lines = path.read_text().splitlines()
    step_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            step_idx = i
    if step_idx is None or step_idx + 1 >= len(lines):
        raise RuntimeError(f"未在 {path} 中找到 thermo PotEng 输出")
    vals = lines[step_idx + 1].split()
    return float(vals[1])

def parse_dump_forces(path: Path):
    lines = path.read_text().splitlines()
    idx = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    atoms = []
    for line in lines[idx + 1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        atoms.append((int(parts[0]), [float(parts[5]), float(parts[6]), float(parts[7])]))
    atoms.sort(key=lambda x: x[0])
    return [f for _, f in atoms]

lmp_energy = parse_pe(log_path)
lmp_forces = parse_dump_forces(dump_path)

atoms = Atoms(
    symbols=[element, element, element, element],
    positions=[
        [4.0, 5.0, 5.0],
        [6.0, 5.0, 5.0],
        [14.0, 5.0, 5.0],
        [16.0, 5.0, 5.0],
    ],
    cell=[20.0, 20.0, 20.0],
    pbc=[False, False, False],
)
device = torch.device("cpu")
model, _ = build_e3trans_from_checkpoint(
    str(checkpoint),
    device,
    atomic_energy_file=str(e0_csv) if e0_csv is not None and e0_csv.exists() else None,
)

ref_dict = {}
if e0_csv is not None and e0_csv.exists():
    with e0_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_dict[int(row["Atom"])] = float(row["E0"])
else:
    raw_ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    aek = raw_ckpt.get("atomic_energy_keys")
    aev = raw_ckpt.get("atomic_energy_values")
    if aek is not None and aev is not None:
        if isinstance(aek, torch.Tensor):
            aek = aek.tolist()
        if isinstance(aev, torch.Tensor):
            aev = aev.tolist()
        ref_dict = {int(k): float(v) for k, v in zip(aek, aev)}

calc = MyE3NNCalculator(model, ref_dict, device, cutoff)
atoms.calc = calc
py_energy = float(atoms.get_potential_energy())
py_forces = atoms.get_forces().tolist()

energy_diff = abs(lmp_energy - py_energy)
max_force_diff = 0.0
for fa, fb in zip(lmp_forces, py_forces):
    max_force_diff = max(max_force_diff, max(abs(a - b) for a, b in zip(fa, fb)))

print(f"LAMMPS PotEng        = {lmp_energy:.12e}")
print(f"Python eager energy  = {py_energy:.12e}")
print(f"|dE|                 = {energy_diff:.12e}")
print(f"max |dF_atom|        = {max_force_diff:.12e}")

ENERGY_TOL = 5e-4
FORCE_TOL = 5e-4
if energy_diff > ENERGY_TOL or max_force_diff > FORCE_TOL:
    print(f"\nFAILED: open-boundary {case_name} runtime 与 Python eager 参考不一致。")
    sys.exit(1)

print(f"\nPASSED: open-boundary {case_name} runtime 与 Python eager 参考在容差内一致。")
PY
}

run_nonperiodic_tree_fmm_mpi_consistency_test() {
  local workdir="$1"
  local lmp_bin="$2"
  local core_pt="$3"
  local element="$4"
  local cutoff="$5"
  local gpu_n="$6"
  local mpi_np="$7"
  local mpi_cmd="$8"

  local single_dir="$workdir/tree_fmm_mpi_single"
  local multi_dir="$workdir/tree_fmm_mpi_multi"
  mkdir -p "$single_dir" "$multi_dir"

  echo "[3/3] Tree FMM MPI consistency: running single-rank reference"
  run_nonperiodic_tree_fmm_case "$single_dir" "$lmp_bin" "$core_pt" "$element" "$cutoff" "1" "1" "$mpi_cmd" "tree_fmm"
  echo "[3/3] Tree FMM MPI consistency: running MPI case ($mpi_np ranks)"
  run_nonperiodic_tree_fmm_case "$multi_dir" "$lmp_bin" "$core_pt" "$element" "$cutoff" "$gpu_n" "$mpi_np" "$mpi_cmd" "tree_fmm"

  python - <<'PY' "$single_dir" "$multi_dir"
import sys
from pathlib import Path

single_dir = Path(sys.argv[1])
multi_dir = Path(sys.argv[2])

def parse_pe(log_path: Path) -> float:
    lines = log_path.read_text().splitlines()
    step_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            step_idx = i
    if step_idx is None or step_idx + 1 >= len(lines):
        raise RuntimeError(f"未在 {log_path} 中找到 thermo PotEng 输出")
    vals = lines[step_idx + 1].split()
    return float(vals[1])

def parse_dump_forces(dump_path: Path):
    lines = dump_path.read_text().splitlines()
    idx = lines.index("ITEM: ATOMS id type x y z fx fy fz")
    atoms = []
    for line in lines[idx + 1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        atoms.append((int(parts[0]), [float(parts[5]), float(parts[6]), float(parts[7])]))
    atoms.sort(key=lambda x: x[0])
    return [f for _, f in atoms]

def max_force_diff(fa, fb):
    out = 0.0
    for a, b in zip(fa, fb):
        out = max(out, max(abs(x - y) for x, y in zip(a, b)))
    return out

pe_single = parse_pe(single_dir / "tree_fmm.log")
pe_multi = parse_pe(multi_dir / "tree_fmm.log")
f_single = parse_dump_forces(single_dir / "tree_fmm.dump")
f_multi = parse_dump_forces(multi_dir / "tree_fmm.dump")
de = abs(pe_single - pe_multi)
df = max_force_diff(f_single, f_multi)

print(f"tree_fmm single PotEng = {pe_single:.12e}")
print(f"tree_fmm mpi    PotEng = {pe_multi:.12e}")
print(f"tree_fmm |dE|         = {de:.12e}")
print(f"tree_fmm max |dF|     = {df:.12e}")

ENERGY_TOL = 5e-4
FORCE_TOL = 5e-4
if de > ENERGY_TOL or df > FORCE_TOL:
    print("\nFAILED: np=1 与 np>1 的 tree_fmm open-boundary 结果不一致。")
    sys.exit(1)

print("\nPASSED: np=1 与 np>1 的 tree_fmm open-boundary 结果在容差内一致。")
PY
}

FIELD_LMP="pair_style mff/torch $CUTOFF cuda"
if [[ ${#FIELD_VALUES[@]} -eq 3 || ${#MFIELD_VALUES[@]} -eq 3 ]]; then
  FIELD_LMP=""
fi
if [[ ${#FIELD_VALUES[@]} -eq 3 ]]; then
  FIELD_LMP=$'variable Ex equal '"${FIELD_VALUES[0]}"$'\n'
  FIELD_LMP+=$'variable Ey equal '"${FIELD_VALUES[1]}"$'\n'
  FIELD_LMP+=$'variable Ez equal '"${FIELD_VALUES[2]}"$'\n'
fi
if [[ ${#MFIELD_VALUES[@]} -eq 3 ]]; then
  FIELD_LMP+=$'variable Bx equal '"${MFIELD_VALUES[0]}"$'\n'
  FIELD_LMP+=$'variable By equal '"${MFIELD_VALUES[1]}"$'\n'
  FIELD_LMP+=$'variable Bz equal '"${MFIELD_VALUES[2]}"$'\n'
fi
if [[ ${#FIELD_VALUES[@]} -eq 3 || ${#MFIELD_VALUES[@]} -eq 3 ]]; then
  FIELD_LMP+=$'pair_style mff/torch '"$CUTOFF"$' cuda'
  if [[ ${#FIELD_VALUES[@]} -eq 3 ]]; then
    FIELD_LMP+=$' field v_Ex v_Ey v_Ez'
  fi
  if [[ ${#MFIELD_VALUES[@]} -eq 3 ]]; then
    FIELD_LMP+=$' mfield v_Bx v_By v_Bz'
  fi
  if [[ -n "$FIDELITY_VALUE" ]]; then
    FIELD_LMP+=$' fidelity '"$FIDELITY_VALUE"
  fi
elif [[ ${#FIELD6_VALUES[@]} -eq 6 ]]; then
  FIELD_LMP=$'variable Txx equal '"${FIELD6_VALUES[0]}"$'\n'
  FIELD_LMP+=$'variable Tyy equal '"${FIELD6_VALUES[1]}"$'\n'
  FIELD_LMP+=$'variable Tzz equal '"${FIELD6_VALUES[2]}"$'\n'
  FIELD_LMP+=$'variable Txy equal '"${FIELD6_VALUES[3]}"$'\n'
  FIELD_LMP+=$'variable Txz equal '"${FIELD6_VALUES[4]}"$'\n'
  FIELD_LMP+=$'variable Tyz equal '"${FIELD6_VALUES[5]}"$'\n'
  FIELD_LMP+=$'pair_style mff/torch '"$CUTOFF"$' cuda field6 v_Txx v_Tyy v_Tzz v_Txy v_Txz v_Tyz'
  if [[ -n "$FIDELITY_VALUE" ]]; then
    FIELD_LMP+=$' fidelity '"$FIDELITY_VALUE"
  fi
elif [[ ${#FIELD9_VALUES[@]} -eq 9 ]]; then
  FIELD_LMP=$'variable Txx equal '"${FIELD9_VALUES[0]}"$'\n'
  FIELD_LMP+=$'variable Txy equal '"${FIELD9_VALUES[1]}"$'\n'
  FIELD_LMP+=$'variable Txz equal '"${FIELD9_VALUES[2]}"$'\n'
  FIELD_LMP+=$'variable Tyx equal '"${FIELD9_VALUES[3]}"$'\n'
  FIELD_LMP+=$'variable Tyy equal '"${FIELD9_VALUES[4]}"$'\n'
  FIELD_LMP+=$'variable Tyz equal '"${FIELD9_VALUES[5]}"$'\n'
  FIELD_LMP+=$'variable Tzx equal '"${FIELD9_VALUES[6]}"$'\n'
  FIELD_LMP+=$'variable Tzy equal '"${FIELD9_VALUES[7]}"$'\n'
  FIELD_LMP+=$'variable Tzz equal '"${FIELD9_VALUES[8]}"$'\n'
  FIELD_LMP+=$'pair_style mff/torch '"$CUTOFF"$' cuda field9 v_Txx v_Txy v_Txz v_Tyx v_Tyy v_Tyz v_Tzx v_Tzy v_Tzz'
  if [[ -n "$FIDELITY_VALUE" ]]; then
    FIELD_LMP+=$' fidelity '"$FIDELITY_VALUE"
  fi
elif [[ -n "$FIDELITY_VALUE" ]]; then
  FIELD_LMP+=$' fidelity '"$FIDELITY_VALUE"
fi

PHYS_LMP=""
DUMP_FREQ=10
THERMO_FREQ=20
if [[ "$STEPS" =~ ^[0-9]+$ ]]; then
  if (( STEPS > 0 && STEPS < DUMP_FREQ )); then DUMP_FREQ=$STEPS; fi
  if (( STEPS > 0 && STEPS < THERMO_FREQ )); then THERMO_FREQ=$STEPS; fi
fi
if [[ $DUMP_FREQ -le 0 ]]; then DUMP_FREQ=1; fi
if [[ $THERMO_FREQ -le 0 ]]; then THERMO_FREQ=1; fi
if [[ $TEST_PHYS_COMPUTE -eq 1 ]]; then
  PHYS_LMP+=$'compute mffg all mff/torch/phys global\n'
  PHYS_LMP+=$'compute mffgm all mff/torch/phys global/mask\n'
  PHYS_LMP+=$'compute mffd all mff/torch/phys global dipole\n'
  PHYS_LMP+=$'compute mffdx all mff/torch/phys global dipole x\n'
  PHYS_LMP+=$'compute mffp all mff/torch/phys global polarizability\n'
  PHYS_LMP+=$'compute mffpxx all mff/torch/phys global polarizability xx\n'
  PHYS_LMP+=$'compute mffa all mff/torch/phys atom\n'
  PHYS_LMP+=$'compute mffam all mff/torch/phys atom/mask\n'
  PHYS_LMP+=$'compute mffad all mff/torch/phys atom dipole\n'
  PHYS_LMP+=$'compute mffadx all mff/torch/phys atom dipole x\n'
  PHYS_LMP+=$'compute mffab all mff/torch/phys atom born_effective_charge\n'
  PHYS_LMP+=$'compute mffabxx all mff/torch/phys atom born_effective_charge xx\n'
  PHYS_LMP+=$'thermo_style custom step pe c_mffgm[2] c_mffgm[3] c_mffdx c_mffpxx c_mffam[5]\n'
  PHYS_LMP+=$'dump 1 all custom '"$DUMP_FREQ"$' dump.phys id type x y z c_mffadx c_mffad[1] c_mffad[2] c_mffad[3] c_mffa[1] c_mffa[2] c_mffa[3] c_mffa[4] c_mffabxx c_mffab[1] c_mffab[2] c_mffab[3] c_mffab[4] c_mffab[5] c_mffab[6] c_mffab[7] c_mffab[8] c_mffab[9]\n'
fi

ATOM_TYPE_COUNT=${#ELEMENTS[@]}
if [[ $ATOM_TYPE_COUNT -le 0 ]]; then
  echo "Internal error: expected at least one element before writing MD input"
  exit 2
fi

MASS_BLOCK="$(python - "${ELEMENTS[@]}" <<'PY'
import sys

masses = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797, "Na": 22.98976928,
    "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
}

for i, sym in enumerate(sys.argv[1:], start=1):
    print(f"mass {i} {masses.get(sym, 1.0)}")
PY
)"

ATOM_CREATE_BLOCK="$(python - "$ATOM_TYPE_COUNT" "$N1" "$N2" <<'PY'
import sys

n_types = int(sys.argv[1])
n1 = int(sys.argv[2])
n2 = int(sys.argv[3])

lines = []
if n_types == 1:
    total = max(n1, 0) + max(n2, 0)
    lines.append(f"create_atoms 1 random {total} 12345 box")
else:
    if n1 > 0:
        lines.append(f"create_atoms 1 random {n1} 12345 box")
    if n2 > 0:
        lines.append(f"create_atoms 2 random {n2} 12346 box")
print("\n".join(lines))
PY
)"

if [[ $OPEN_BOUNDARY_MD -eq 1 || ($TEST_TREE_FMM_LONG_RANGE -eq 1 && $MD_ONLY_MODE -eq 1) ]]; then
  MD_BOUNDARY="f f f"
  MD_FIX_BLOCK=$'fix mffwall all wall/reflect xlo EDGE xhi EDGE ylo EDGE yhi EDGE zlo EDGE zhi EDGE\nfix 1 all nve'
else
  MD_BOUNDARY="p p p"
  MD_FIX_BLOCK='fix 1 all nve'
fi

cat > "$OUT_DIR/in.corept" <<EOF
units metal
atom_style atomic
boundary $MD_BOUNDARY

region box block 0 $BOX 0 $BOX 0 $BOX
create_box $ATOM_TYPE_COUNT box
$ATOM_CREATE_BLOCK
$MASS_BLOCK

neighbor 1.0 bin

$FIELD_LMP
pair_coeff * * $CORE_PT ${ELEMENTS[*]}

velocity all create 300 42
$MD_FIX_BLOCK
$PHYS_LMP
thermo $THERMO_FREQ
run $STEPS
EOF

echo "[3/3] Running LAMMPS (Kokkos+CUDA)"
export LD_LIBRARY_PATH="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
):${LD_LIBRARY_PATH:-}"

if [[ $NATIVE_OPS -eq 1 && -z "${MFF_CUSTOM_OPS_LIB:-}" ]]; then
  echo "[3/3] --native-ops: auto-detect cuEquivariance ops and libpython path"
  eval "$(python - <<'PY'
import pathlib, sys, sysconfig

extra_ld = []

# 1) Find the torch extension .so (registers TorchScript custom ops)
try:
    import cuequivariance_ops_torch
    pkg = pathlib.Path(cuequivariance_ops_torch.__file__).parent
    so_files = sorted(f for f in pkg.rglob("*.so") if "__pycache__" not in str(f))
    if so_files:
        joined = ":".join(str(f) for f in so_files)
        print('export MFF_CUSTOM_OPS_LIB="{}"'.format(joined))
except ImportError:
    print("echo 'WARNING: cuequivariance_ops_torch not installed'", flush=True)

# 2) Find libcue_ops.so and add its dir to LD_LIBRARY_PATH
try:
    import cuequivariance_ops
    pkg2 = pathlib.Path(cuequivariance_ops.__file__).parent
    for f in pkg2.rglob("libcue_ops*"):
        extra_ld.append(str(f.parent))
except ImportError:
    pass

# 3) Find libpython (needed by CPython extension in pure-C++ process)
libdir = sysconfig.get_config_var("LIBDIR") or ""
if libdir:
    extra_ld.append(libdir)

if extra_ld:
    dirs = ":".join(sorted(set(extra_ld)))
    print('export LD_LIBRARY_PATH="{}:${{LD_LIBRARY_PATH:-}}"'.format(dirs))
PY
)"
  echo "  MFF_CUSTOM_OPS_LIB=${MFF_CUSTOM_OPS_LIB:-<not set>}"
fi

# Multi-GPU: use MPI when np > 1
if [[ -z "$NP" ]]; then
  NP="1"
fi

if [[ $TEST_PBC_MINIMUM_IMAGE -eq 1 ]]; then
  run_pbc_minimum_image_test "$OUT_DIR" "$LMP" "$CORE_PT" "${ELEMENTS[0]}" "$CUTOFF" "$GPU_N" "$NP" "$MPI_CMD"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi
if [[ $TEST_FEATURE_SPECTRAL_FFT_MPI_CONSISTENCY -eq 1 ]]; then
  run_pbc_minimum_image_mpi_consistency_test "$OUT_DIR" "$LMP" "$CORE_PT" "${ELEMENTS[0]}" "$CUTOFF" "$GPU_N" "$NP" "$MPI_CMD"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi
if [[ $TEST_PBC_SLAB -eq 1 ]]; then
  run_pbc_slab_test "$OUT_DIR" "$LMP" "$CORE_PT" "${ELEMENTS[0]}" "$CUTOFF" "$GPU_N" "$NP" "$MPI_CMD"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi
if [[ $TEST_PBC_SLAB_Z_OPEN -eq 1 ]]; then
  run_pbc_slab_z_open_test "$OUT_DIR" "$LMP" "$CORE_PT" "${ELEMENTS[0]}" "$CUTOFF" "$GPU_N" "$NP" "$MPI_CMD"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi
if [[ $TEST_PBC_TRICLINIC -eq 1 ]]; then
  run_pbc_triclinic_test "$OUT_DIR" "$LMP" "$CORE_PT" "${ELEMENTS[0]}" "$CUTOFF" "$GPU_N" "$NP" "$MPI_CMD"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi
if [[ $TEST_TREE_FMM_LONG_RANGE -eq 1 && $MD_ONLY_MODE -ne 1 ]]; then
  run_nonperiodic_python_reference_test "$OUT_DIR" "$LMP" "$CORE_PT" "$PTH" "${ELEMENTS[0]}" "$CUTOFF" "${E0CSV:-}" "tree_fmm" "Open-boundary tree_fmm smoke (single-rank LAMMPS vs Python eager)"
  if [[ "$NP" -gt 1 ]]; then
    run_nonperiodic_tree_fmm_mpi_consistency_test "$OUT_DIR" "$LMP" "$CORE_PT" "${ELEMENTS[0]}" "$CUTOFF" "$GPU_N" "$NP" "$MPI_CMD"
  fi
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi
if [[ $TEST_ISOLATED_FAR_FIELD -eq 1 && $MD_ONLY_MODE -ne 1 ]]; then
  run_nonperiodic_python_reference_test "$OUT_DIR" "$LMP" "$CORE_PT" "$PTH" "${ELEMENTS[0]}" "$CUTOFF" "${E0CSV:-}" "isolated_far_field" "Open-boundary isolated-far-field-v1 smoke (single-rank LAMMPS vs Python eager)"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi
if [[ $TEST_ISOLATED_FAR_FIELD_V2 -eq 1 && $MD_ONLY_MODE -ne 1 ]]; then
  run_nonperiodic_python_reference_test "$OUT_DIR" "$LMP" "$CORE_PT" "$PTH" "${ELEMENTS[0]}" "$CUTOFF" "${E0CSV:-}" "isolated_far_field_v2" "Open-boundary isolated-far-field-v2 smoke (single-rank LAMMPS vs Python eager)"
  echo "DONE. out_dir=$OUT_DIR"
  exit 0
fi

LMP_ARGS=(-k on g "$GPU_N" -sf kk -pk kokkos newton off neigh full -in "$OUT_DIR/in.corept")

if [[ "$NP" -gt 1 ]]; then
  echo "[3/3] Multi-GPU run: $MPI_CMD -np $NP $LMP ${LMP_ARGS[*]}"
  run_mpi_lammps "$MPI_CMD" "$NP" "$LMP" "${LMP_ARGS[@]}"
else
  echo "[3/3] Single-GPU run"
  "$LMP" "${LMP_ARGS[@]}"
fi

if [[ $TEST_PHYS_COMPUTE -eq 1 ]]; then
  if [[ ! -s "$OUT_DIR/dump.phys" ]]; then
    echo "ERROR: expected physical tensor dump file was not created: $OUT_DIR/dump.phys"
    exit 1
  fi
  echo "[3/3] Physical tensor dump written: $OUT_DIR/dump.phys"
fi

echo "DONE. out_dir=$OUT_DIR"
