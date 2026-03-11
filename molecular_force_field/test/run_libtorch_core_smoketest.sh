#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
在 GPU 机器上验证 “TorchScript core.pt + (可选) 纯 C++ LibTorch” 链路。

做什么：
1) 从 checkpoint 导出可被 C++ 直接加载的 TorchScript：core.pt（torch.jit.save）
2) Python 侧用 CUDA 做一次 forward + dE/dpos forces 校验与计时
3) （可选）编译并运行 libtorch_smoketest（C++/LibTorch）在 CUDA 上再测一遍

用法：
  bash molecular_force_field/scripts/run_libtorch_core_smoketest.sh \
    --checkpoint /path/to/ckpt.pth --elements H O --device cuda

也支持不提供 checkpoint（自动生成 pure-cartesian-ictd dummy checkpoint）：
  bash molecular_force_field/scripts/run_libtorch_core_smoketest.sh \
    --elements H O --device cuda

常用参数：
  --N 512            原子数
  --E 16384          边数
  --warmup 10        预热迭代
  --iters 50         计时迭代
  --out-dir /tmp/x   输出目录（默认：mktemp）
  --skip-cpp         只跑 Python，不编译 C++
  --skip-python      只导出 + 跑 C++
EOF
}

CHECKPOINT=""
ELEMENTS=("H" "O")
DEVICE="cuda"
N=512
E=16384
WARMUP=10
ITERS=50
OUT_DIR=""
SKIP_CPP=0
SKIP_PY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="${2:-}"; shift 2;;
    --elements)
      shift
      ELEMENTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ELEMENTS+=("$1"); shift
      done
      ;;
    --device) DEVICE="${2:-}"; shift 2;;
    --N) N="${2:-}"; shift 2;;
    --E) E="${2:-}"; shift 2;;
    --warmup) WARMUP="${2:-}"; shift 2;;
    --iters) ITERS="${2:-}"; shift 2;;
    --out-dir) OUT_DIR="${2:-}"; shift 2;;
    --skip-cpp) SKIP_CPP=1; shift;;
    --skip-python) SKIP_PY=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数：$1"; usage; exit 2;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(python - <<'PY'
import tempfile
print(tempfile.mkdtemp(prefix="mliap-libtorch-smoke-"))
PY
)"
fi

CORE_PT="$OUT_DIR/core.pt"

if [[ -z "$CHECKPOINT" ]]; then
  echo "[0/3] 未提供 checkpoint，生成 pure-cartesian-ictd dummy checkpoint"
  CHECKPOINT="$OUT_DIR/dummy.pth"
  python - <<PY
import torch
from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd
out = r"$CHECKPOINT"
_make_dummy_checkpoint_pure_cartesian_ictd(out, device=torch.device("cpu"))
print("dummy checkpoint:", out)
PY
fi

echo "[1/3] 导出 TorchScript core.pt"
python "$REPO_ROOT/molecular_force_field/scripts/export_libtorch_core.py" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --elements "${ELEMENTS[@]}" \
  --out "$CORE_PT"

echo "  core.pt = $CORE_PT"

if [[ $SKIP_PY -eq 0 ]]; then
  echo "[2/3] Python (TorchScript) CUDA 自测 + 计时"
  python - <<PY
import time
import torch

core_path = r"$CORE_PT"
device_req = r"$DEVICE"
device = device_req
if device_req == "cuda" and not torch.cuda.is_available():
    print("WARN: CUDA 不可用，改用 CPU")
    device = "cpu"

core = torch.jit.load(core_path, map_location=device)
core.eval()

N = int(r"$N")
E = int(r"$E")
warmup = int(r"$WARMUP")
iters = int(r"$ITERS")

dtype = torch.float32
pos = torch.zeros(N, 3, device=device, dtype=dtype, requires_grad=True)
A = torch.randint(1, 9, (N,), device=device, dtype=torch.long)
batch = torch.zeros(N, device=device, dtype=torch.long)
edge_src = torch.randint(0, N, (E,), device=device, dtype=torch.long)
edge_dst = torch.randint(0, N, (E,), device=device, dtype=torch.long)
edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 100.0
rij = torch.randn(E, 3, device=device, dtype=dtype)

def run_once():
    edge_vec = pos[edge_dst] - pos[edge_src] + rij
    atom_e = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec)
    E_total = atom_e.sum()
    (grad_pos,) = torch.autograd.grad(E_total, pos, create_graph=False)
    forces = -grad_pos
    return atom_e, E_total, forces

for _ in range(warmup):
    atom_e, E_total, forces = run_once()
    if device == "cuda":
        torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(iters):
    atom_e, E_total, forces = run_once()
if device == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

atom_e_shape = tuple(atom_e.shape)
E_val = float(E_total.detach().to("cpu"))
F_norm = float(forces.detach().to("cpu").norm())
print(f"device={device} N={N} E={E} iters={iters} avg={(t1-t0)*1000/iters:.3f} ms/iter atom_e={atom_e_shape} E_total={E_val} |F|={F_norm}")
PY
fi

if [[ $SKIP_CPP -eq 0 ]]; then
  echo "[3/3] C++/LibTorch 编译 + CUDA 自测"
  PREFIX="$(python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"
  BUILD_DIR="$OUT_DIR/build-libtorch-smoketest"
  cmake -S "$REPO_ROOT/libtorch_smoketest" -B "$BUILD_DIR" -DCMAKE_PREFIX_PATH="$PREFIX"
  cmake --build "$BUILD_DIR" -j
  "$BUILD_DIR/libtorch_smoketest" \
    --model "$CORE_PT" \
    --device "$DEVICE" \
    --N "$N" \
    --E "$E" \
    --warmup 5 \
    --iters 20
fi

echo "DONE. 输出目录：$OUT_DIR"

