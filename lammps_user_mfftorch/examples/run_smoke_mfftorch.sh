#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: bash run_smoke_mfftorch.sh /path/to/lmp [cuda|cpu]"
  exit 2
fi

LMP_EXE="$1"
DEVICE="${2:-cuda}"

OUT_DIR="${OUT_DIR:-$(pwd)/mfftorch_smoke_out}"
mkdir -p "$OUT_DIR"

run_lammps() {
  local input_file="$1"
  if [[ "$DEVICE" == "cuda" ]]; then
    "$LMP_EXE" -k on g 1 -sf kk -pk kokkos newton off neigh full -in "$input_file"
  else
    "$LMP_EXE" -in "$input_file"
  fi
}

echo "[1/4] 生成 dummy checkpoint + core.pt（普通版 + physical tensor 版）"
if [[ ! -f "$OUT_DIR/core.pt" || ! -f "$OUT_DIR/core_phys.pt" ]]; then
  python - <<PY
import os
import torch
from molecular_force_field.interfaces.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd

out_dir = r"$OUT_DIR"
ckpt_plain = os.path.join(out_dir, "dummy.pth")
ckpt_phys = os.path.join(out_dir, "dummy_phys.pth")

_make_dummy_checkpoint_pure_cartesian_ictd(ckpt_plain, device=torch.device("cpu"))
_make_dummy_checkpoint_pure_cartesian_ictd(
    ckpt_phys,
    device=torch.device("cpu"),
    physical_tensor_outputs={
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "dipole_per_atom": {"ls": [1], "channels_out": 1, "reduce": "none"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
        "polarizability_per_atom": {"ls": [0, 2], "channels_out": 1, "reduce": "none"},
    },
)
print("dummy checkpoint:", ckpt_plain)
print("dummy checkpoint with physical tensors:", ckpt_phys)
PY
  python -m molecular_force_field.cli.export_libtorch_core \
    --checkpoint "$OUT_DIR/dummy.pth" --device "$DEVICE" --elements H O --out "$OUT_DIR/core.pt"
  python -m molecular_force_field.cli.export_libtorch_core \
    --checkpoint "$OUT_DIR/dummy_phys.pth" --device "$DEVICE" --elements H O --out "$OUT_DIR/core_phys.pt"
fi

echo "[2/4] 写入普通 LAMMPS smoke 输入文件（无 physical tensors）"
cat > "$OUT_DIR/in.smoke_plain" <<EOF
units metal
atom_style atomic
boundary p p p

region box block 0 40 0 40 0 40
create_box 2 box
create_atoms 1 random 200 12345 box
create_atoms 2 random 100 12346 box
mass 1 1.008
mass 2 15.999

neighbor 1.0 bin

pair_style mff/torch 5.0 $DEVICE
pair_coeff * * $OUT_DIR/core.pt H O

velocity all create 300 42
fix 1 all nve
thermo 10
run 20
EOF

echo "[3/4] 写入 physical tensor LAMMPS smoke 输入文件"
cat > "$OUT_DIR/in.smoke_phys" <<EOF
units metal
atom_style atomic
boundary p p p

region box block 0 40 0 40 0 40
create_box 2 box
create_atoms 1 random 60 22345 box
create_atoms 2 random 40 22346 box
mass 1 1.008
mass 2 15.999

neighbor 1.0 bin

pair_style mff/torch 5.0 $DEVICE
pair_coeff * * $OUT_DIR/core_phys.pt H O

compute mffg all mff/torch/phys global
compute mffgm all mff/torch/phys global/mask
compute mffd all mff/torch/phys global dipole
compute mffdx all mff/torch/phys global dipole x
compute mffp all mff/torch/phys global polarizability
compute mffpxx all mff/torch/phys global polarizability xx
compute mffam all mff/torch/phys atom/mask
compute mffa all mff/torch/phys atom
compute mffad all mff/torch/phys atom dipole
compute mffadx all mff/torch/phys atom dipole x

thermo_style custom step pe c_mffgm[2] c_mffgm[3] c_mffdx c_mffpxx
thermo 5
dump 1 all custom 10 dump.mffphys id type x y z c_mffadx c_mffad[1] c_mffad[2] c_mffad[3] c_mffa[1] c_mffa[2] c_mffa[3] c_mffa[4]

run 20
EOF

echo "[4/4] 运行 LAMMPS"
echo "OUT_DIR=$OUT_DIR"
echo "INPUT(plain)=$OUT_DIR/in.smoke_plain"
run_lammps "$OUT_DIR/in.smoke_plain"
echo "INPUT(phys)=$OUT_DIR/in.smoke_phys"
run_lammps "$OUT_DIR/in.smoke_phys"

echo "Smoke tests finished."

