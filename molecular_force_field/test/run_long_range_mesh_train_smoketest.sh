#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
LES 风格 mesh/FFT long-range 训练 + 导出 smoke test。

做什么：
1) 生成极小的 periodic / slab extxyz toy 数据
2) 预处理成 HDF5
3) 运行 `long_range_mode=reciprocal-spectral-v1` + `mesh_fft` 的训练 dry run
4) 导出 `core.pt`，校验 `core.pt.json` 中的 long-range 元数据

用法：
  bash molecular_force_field/test/run_long_range_mesh_train_smoketest.sh

  bash molecular_force_field/test/run_long_range_mesh_train_smoketest.sh \
    --mode ictd \
    --boundary both \
    --device cpu

参数：
  --mode ictd|cue|both         测试模型（默认 both）
  --boundary periodic|slab|both
                              边界模式（默认 both）
  --device cpu|cuda            训练 / 导出设备（默认 cpu）
  --dtype float32|float64      dtype（默认 float32）
  --epochs N                   epoch 数（默认 1）
  --batch-size N               batch size（默认 1）
  --max-radius R               邻域截断（默认 5.0）
  --mesh-size N                reciprocal mesh size（默认 8）
  --filter-hidden-dim N        spectral filter hidden dim（默认 16）
  --source-channels N          latent source channels（默认 1）
  --slab-padding-factor N      slab 真空 padding 倍数（默认 2）
  --out-dir DIR                输出目录（默认 mktemp）
EOF
}

MODE="both"
BOUNDARY="both"
DEVICE="cpu"
DTYPE="float32"
EPOCHS="1"
BATCH_SIZE="1"
MAX_RADIUS="5.0"
MESH_SIZE="8"
FILTER_HIDDEN_DIM="16"
SOURCE_CHANNELS="1"
SLAB_PADDING_FACTOR="2"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2;;
    --boundary) BOUNDARY="${2:-}"; shift 2;;
    --device) DEVICE="${2:-}"; shift 2;;
    --dtype) DTYPE="${2:-}"; shift 2;;
    --epochs) EPOCHS="${2:-}"; shift 2;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2;;
    --max-radius) MAX_RADIUS="${2:-}"; shift 2;;
    --mesh-size) MESH_SIZE="${2:-}"; shift 2;;
    --filter-hidden-dim) FILTER_HIDDEN_DIM="${2:-}"; shift 2;;
    --source-channels) SOURCE_CHANNELS="${2:-}"; shift 2;;
    --slab-padding-factor) SLAB_PADDING_FACTOR="${2:-}"; shift 2;;
    --out-dir) OUT_DIR="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 2;;
  esac
done

case "$MODE" in
  ictd|cue|both) ;;
  *) echo "--mode 只能是 ictd|cue|both"; exit 2;;
esac

case "$BOUNDARY" in
  periodic|slab|both) ;;
  *) echo "--boundary 只能是 periodic|slab|both"; exit 2;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(python - <<'PY'
import tempfile
print(tempfile.mkdtemp(prefix="mff-long-range-mesh-smoke-"))
PY
)"
fi
mkdir -p "$OUT_DIR"

PERIODIC_XYZ="$OUT_DIR/toy_periodic.xyz"
SLAB_XYZ="$OUT_DIR/toy_slab.xyz"

echo "[1/4] 生成 toy extxyz 数据"
python - <<PY
from pathlib import Path

periodic = Path(r"$PERIODIC_XYZ")
slab = Path(r"$SLAB_XYZ")

periodic_frames = [
    (0.10, [('H', 0.10, 0.10, 0.10,  0.10, 0.02, -0.01), ('H', 0.83, 0.27, 0.14, -0.10, -0.02, 0.01)]),
    (0.20, [('H', 0.16, 0.11, 0.12,  0.20, 0.01, -0.02), ('H', 0.79, 0.31, 0.19, -0.20, -0.01, 0.02)]),
]
slab_frames = [
    (0.12, [('H', 0.12, 0.14, 0.25,  0.08, 0.01, -0.01), ('H', 0.81, 0.29, 0.34, -0.08, -0.01, 0.01)]),
    (0.18, [('H', 0.18, 0.18, 0.28,  0.11, 0.02, -0.02), ('H', 0.76, 0.33, 0.39, -0.11, -0.02, 0.02)]),
]

with periodic.open("w", encoding="utf-8") as f:
    for energy, atoms in periodic_frames:
        f.write(f"{len(atoms)}\n")
        f.write(
            f'Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" energy="{energy}" '
            'Properties=species:S:1:pos:R:3:force:R:3\n'
        )
        for sym, x, y, z, fx, fy, fz in atoms:
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n")

with slab.open("w", encoding="utf-8") as f:
    for energy, atoms in slab_frames:
        f.write(f"{len(atoms)}\n")
        f.write(
            f'Lattice="10 0 0 0 10 0 0 0 20" pbc="T T F" energy="{energy}" '
            'Properties=species:S:1:pos:R:3:force:R:3\n'
        )
        for sym, x, y, z, fx, fy, fz in atoms:
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n")
print(periodic)
print(slab)
PY

preprocess_dataset() {
  local input_xyz="$1"
  local out_dir="$2"
  python -m molecular_force_field.cli.preprocess \
    --input-file "$input_xyz" \
    --output-dir "$out_dir" \
    --train-ratio 0.5 \
    --seed 7 \
    --elements H \
    --max-atom 2 \
    --max-radius "$MAX_RADIUS" \
    --num-workers 1
}

echo "[2/4] 预处理 periodic / slab toy 数据"
preprocess_dataset "$PERIODIC_XYZ" "$OUT_DIR/data_periodic"
preprocess_dataset "$SLAB_XYZ" "$OUT_DIR/data_slab"

run_case() {
  local mode="$1"
  local boundary="$2"
  local data_dir="$3"
  local tag="$4"
  local ckpt="$OUT_DIR/${tag}.pth"
  local mode_suffix=""
  case "$mode" in
    pure-cartesian-ictd) mode_suffix="_pure_cartesian_ictd" ;;
    spherical-save-cue) mode_suffix="_spherical_save_cue" ;;
    *) echo "unsupported mode for smoke output naming: $mode" >&2; exit 2 ;;
  esac
  local train_ckpt="$REPO_ROOT/checkpoint/${tag}${mode_suffix}.pth"
  local core="$OUT_DIR/${tag}.pt"
  local -a train_args=(
    -m molecular_force_field.cli.train
    --data-dir "$data_dir"
    --checkpoint "$ckpt"
    --tensor-product-mode "$mode"
    --long-range-mode reciprocal-spectral-v1
    --long-range-boundary "$boundary"
    --long-range-reciprocal-backend mesh_fft
    --long-range-green-mode poisson
    --long-range-assignment cic
    --long-range-mesh-size "$MESH_SIZE"
    --long-range-slab-padding-factor "$SLAB_PADDING_FACTOR"
    --long-range-filter-hidden-dim "$FILTER_HIDDEN_DIM"
    --long-range-source-channels "$SOURCE_CHANNELS"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers 0
    --device "$DEVICE"
    --dtype "$DTYPE"
    --dump-frequency 1
    --patience 1
    --energy-log-frequency 1
    --no-save-val-csv
  )

  echo "[3/4][$tag] 训练 smoke"
  python "${train_args[@]}"

  echo "[4/4][$tag] 导出 core.pt 并校验元数据"
  python -m molecular_force_field.cli.export_libtorch_core \
    --checkpoint "$train_ckpt" \
    --elements H \
    --device "$DEVICE" \
    --out "$core" \
    --embed-e0 \
    --export-reciprocal-source

  python - <<PY
import json
from pathlib import Path

meta = json.loads(Path(r"$core.json").read_text(encoding="utf-8"))
assert meta["export_reciprocal_source"] is True
assert int(meta["reciprocal_source_channels"]) == int($SOURCE_CHANNELS)
assert str(meta["long_range_boundary"]) == "$boundary"
assert int(meta["long_range_mesh_size"]) == int($MESH_SIZE)
assert int(meta["long_range_slab_padding_factor"]) == int($SLAB_PADDING_FACTOR)
assert str(meta["long_range_reciprocal_backend"]) == "mesh_fft"
assert str(meta["long_range_green_mode"]) == "poisson"
if "$boundary" == "slab":
    assert str(meta["reciprocal_source_boundary"]) == "slab"
print("validated", r"$core.json")
PY
}

run_mode_boundary() {
  local mode="$1"
  local mode_tag="$2"
  case "$BOUNDARY" in
    periodic)
      run_case "$mode" "periodic" "$OUT_DIR/data_periodic" "${mode_tag}_periodic"
      ;;
    slab)
      run_case "$mode" "slab" "$OUT_DIR/data_slab" "${mode_tag}_slab"
      ;;
    both)
      run_case "$mode" "periodic" "$OUT_DIR/data_periodic" "${mode_tag}_periodic"
      run_case "$mode" "slab" "$OUT_DIR/data_slab" "${mode_tag}_slab"
      ;;
  esac
}

case "$MODE" in
  ictd)
    run_mode_boundary "pure-cartesian-ictd" "ictd_long_range_mesh_smoke"
    ;;
  cue)
    run_mode_boundary "spherical-save-cue" "cue_long_range_mesh_smoke"
    ;;
  both)
    run_mode_boundary "pure-cartesian-ictd" "ictd_long_range_mesh_smoke"
    run_mode_boundary "spherical-save-cue" "cue_long_range_mesh_smoke"
    ;;
esac

echo "DONE. out_dir=$OUT_DIR"
