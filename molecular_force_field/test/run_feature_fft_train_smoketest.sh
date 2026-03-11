#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Feature-space FFT 训练 smoke test（支持单卡 / 多卡）。

做什么：
1) 自动生成一个极小的 periodic extxyz toy 数据集（非退化、非共线几何）
2) 预处理成 processed_train.h5 / processed_val.h5
3) 运行带 feature-space FFT block 的 `mff-train` dry run
4) 支持：
   - `pure-cartesian-ictd`
   - `spherical-save-cue`
   - 或两者都测
5) 支持单 GPU，也支持通过 `--n-gpu` 走 train.py 的自动 torchrun/DDP

用法：
  bash molecular_force_field/test/run_feature_fft_train_smoketest.sh

  bash molecular_force_field/test/run_feature_fft_train_smoketest.sh \
    --mode both \
    --device cuda \
    --dtype float32 \
    --n-gpu 2

  bash molecular_force_field/test/run_feature_fft_train_smoketest.sh \
    --mode cue \
    --device cuda \
    --dtype float32 \
    --n-gpu 4

参数：
  --mode ictd|cue|both     测试模式（默认 both）
  --device cpu|cuda        训练设备（默认 cuda）
  --dtype float32|float64  训练 dtype（默认 float32）
  --n-gpu N                GPU 数；N>1 时自动触发 DDP（默认 1）
  --epochs N               epoch 数（默认 1）
  --batch-size N           batch size（默认 1）
  --max-radius R           邻域截断（默认 5.0）
  --mesh-size N            FFT 网格边长（默认 8）
  --bottleneck-dim N       低秩 bottleneck 通道数（默认 4）
  --filter-hidden-dim N    FFT filter MLP hidden dim（默认 16）
  --out-dir DIR            输出目录（默认 mktemp）
  --launcher auto|local|slurm
                           透传给 mff-train（默认 auto）

说明：
- 多 GPU 模式下，脚本会先单进程预处理数据，再调用 `mff-train --n-gpu N`。
- `spherical-save-cue` 需要 cuEquivariance 可用。
- 建议在 GPU 机器上配合 `CUDA_VISIBLE_DEVICES` 控制实际可见卡。
EOF
}

MODE="both"
DEVICE="cuda"
DTYPE="float32"
N_GPU="1"
EPOCHS="1"
BATCH_SIZE="1"
MAX_RADIUS="5.0"
MESH_SIZE="8"
BOTTLENECK_DIM="4"
FILTER_HIDDEN_DIM="16"
OUT_DIR=""
LAUNCHER="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2;;
    --device) DEVICE="${2:-}"; shift 2;;
    --dtype) DTYPE="${2:-}"; shift 2;;
    --n-gpu) N_GPU="${2:-}"; shift 2;;
    --epochs) EPOCHS="${2:-}"; shift 2;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2;;
    --max-radius) MAX_RADIUS="${2:-}"; shift 2;;
    --mesh-size) MESH_SIZE="${2:-}"; shift 2;;
    --bottleneck-dim) BOTTLENECK_DIM="${2:-}"; shift 2;;
    --filter-hidden-dim) FILTER_HIDDEN_DIM="${2:-}"; shift 2;;
    --out-dir) OUT_DIR="${2:-}"; shift 2;;
    --launcher) LAUNCHER="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 2;;
  esac
done

case "$MODE" in
  ictd|cue|both) ;;
  *) echo "--mode 只能是 ictd|cue|both"; exit 2;;
esac

if [[ "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
  echo "--device 只能是 cpu|cuda"
  exit 2
fi

if [[ "$N_GPU" != "1" && "$DEVICE" != "cuda" ]]; then
  echo "--n-gpu > 1 只能与 --device cuda 一起使用"
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(python - <<'PY'
import tempfile
print(tempfile.mkdtemp(prefix="mff-feature-fft-train-smoke-"))
PY
)"
fi
mkdir -p "$OUT_DIR"

TOY_XYZ="$OUT_DIR/toy_periodic.xyz"
DATA_DIR="$OUT_DIR/data"
mkdir -p "$DATA_DIR"

echo "[1/3] 生成 toy periodic extxyz 数据集"
python - <<PY
from pathlib import Path

xyz = Path(r"$TOY_XYZ")
frames = [
    (0.10, [('H', 0.10, 0.10, 0.10,  0.10, 0.02, -0.01), ('H', 0.83, 0.27, 0.14, -0.10, -0.02, 0.01)]),
    (0.20, [('H', 0.16, 0.11, 0.12,  0.20, 0.01, -0.02), ('H', 0.79, 0.31, 0.19, -0.20, -0.01, 0.02)]),
    (0.15, [('H', 0.21, 0.14, 0.11,  0.15, 0.03, -0.01), ('H', 0.76, 0.34, 0.22, -0.15, -0.03, 0.01)]),
    (0.25, [('H', 0.28, 0.16, 0.13,  0.25, 0.02, -0.02), ('H', 0.72, 0.38, 0.24, -0.25, -0.02, 0.02)]),
]
with xyz.open("w", encoding="utf-8") as f:
    for energy, atoms in frames:
        f.write(f"{len(atoms)}\\n")
        f.write(
            f'Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" energy="{energy}" '
            'Properties=species:S:1:pos:R:3:force:R:3\\n'
        )
        for sym, x, y, z, fx, fy, fz in atoms:
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\\n")
print(xyz)
PY

echo "[2/3] 预处理 toy 数据"
python -m molecular_force_field.cli.preprocess \
  --input-file "$TOY_XYZ" \
  --output-dir "$DATA_DIR" \
  --train-ratio 0.5 \
  --seed 7 \
  --elements H \
  --max-atom 2 \
  --max-radius "$MAX_RADIUS" \
  --num-workers 1

run_mode() {
  local mode="$1"
  local tag="$2"
  local ckpt="$OUT_DIR/${tag}.pth"
  local -a args=(
    -m molecular_force_field.cli.train
    --data-dir "$DATA_DIR"
    --checkpoint "$ckpt"
    --tensor-product-mode "$mode"
    --feature-spectral-mode fft
    --feature-spectral-bottleneck-dim "$BOTTLENECK_DIM"
    --feature-spectral-mesh-size "$MESH_SIZE"
    --feature-spectral-filter-hidden-dim "$FILTER_HIDDEN_DIM"
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

  if [[ "$N_GPU" != "1" ]]; then
    args+=(--n-gpu "$N_GPU" --launcher "$LAUNCHER")
  fi

  echo "[3/3][$tag] 运行训练 smoke"
  python "${args[@]}"
}

case "$MODE" in
  ictd)
    run_mode "pure-cartesian-ictd" "ictd_feature_fft_smoke"
    ;;
  cue)
    run_mode "spherical-save-cue" "cue_feature_fft_smoke"
    ;;
  both)
    run_mode "pure-cartesian-ictd" "ictd_feature_fft_smoke"
    run_mode "spherical-save-cue" "cue_feature_fft_smoke"
    ;;
esac

echo "DONE. out_dir=$OUT_DIR"
