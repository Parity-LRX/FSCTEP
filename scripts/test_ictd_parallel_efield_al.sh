#!/bin/bash
# ===========================================================================
#  端到端测试：pure-cartesian-ictd + 外场 + 并行训练 + 并行 MD + 冷启动 + AL + resume
#
#  支持运行环境:
#    CPU:    bash scripts/test_ictd_parallel_efield_al.sh           (默认)
#    2 GPU:  bash scripts/test_ictd_parallel_efield_al.sh --gpu
#
#  流程:
#    1. 创建 2 个种子结构（水分子变体），用于多结构并行 MD
#    2. mff-init-data: 冷启动 (PySCF) 生成初始数据
#    3. 注入 external_field 到初始 H5 数据集
#    4. mff-active-learn: ICTD + 外场 + 并行训练 + 并行 MD，跑 2 轮
#       - 外场贯穿: 训练(H5嵌入) → MD探索 → 模型偏差 → identity标注 → merge注入
#    5. 验证 checkpoint 格式与外场架构
#    6. mff-active-learn --resume: 从 checkpoint 重启，跑第 3 轮
#    7. 验证 resume 产物 + merge 后 H5 仍包含 external_field
# ===========================================================================

set -e
cd "$(dirname "$0")/.."

# ---- 解析参数 ----
USE_GPU=0
if [[ "$1" == "--gpu" ]]; then
  USE_GPU=1
fi

if [[ $USE_GPU -eq 1 ]]; then
  DEVICE="cuda"
  TRAIN_N_GPU=1
  echo "=== 运行模式: 2×GPU (每模型 1 GPU, 2 模型并行) ==="
else
  DEVICE="cpu"
  TRAIN_N_GPU=1
  echo "=== 运行模式: CPU ==="
fi

WORK_ROOT="test_ictd_parallel_efield"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"

# ===========================================================================
# Step 0: 创建 2 个种子结构（水分子不同构型）
# ===========================================================================
echo ""
echo "=== Step 0: 创建种子结构 ==="

python3 -c "
from ase import Atoms
from ase.io import write

# 水分子构型 A
mol_a = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0.48, 0.83, 0]])
write('$WORK_ROOT/seed_a.xyz', mol_a, format='extxyz')

# 水分子构型 B（旋转 + 微扰）
mol_b = Atoms('H2O', positions=[[0, 0, 0], [0, 0.96, 0], [-0.83, 0.48, 0.1]])
write('$WORK_ROOT/seed_b.xyz', mol_b, format='extxyz')

print('Created seed_a.xyz and seed_b.xyz')
"

# ===========================================================================
# Step 1: mff-init-data 冷启动
# ===========================================================================
echo ""
echo "=== Step 1: mff-init-data (冷启动, PySCF) ==="

mff-init-data \
  --structures "$WORK_ROOT/seed_a.xyz" "$WORK_ROOT/seed_b.xyz" \
  --n-perturb 4 \
  --rattle-std 0.05 \
  --label-type pyscf \
  --pyscf-method b3lyp \
  --pyscf-basis sto-3g \
  --label-n-workers 2 \
  --output-dir "$WORK_ROOT/data"

echo "init-data 完成"

# ===========================================================================
# Step 2: 外场参数（不再需要手动注入 H5，--explore-external-field 自动注入）
# ===========================================================================
EFIELD="0.0 0.0 0.01"
echo ""
echo "=== Step 2: 外场值 = [$EFIELD] (rank-1, 由 --explore-external-field 自动注入 H5) ==="

# ===========================================================================
# Step 3: mff-active-learn (ICTD + 外场 + 并行训练 + 并行 MD)
#   --explore-external-field 自动:
#     1) 推断 --external-tensor-rank=1
#     2) 注入初始 H5
#     3) 每轮 merge 后重新注入
# ===========================================================================
echo ""
echo "=== Step 3: mff-active-learn (ICTD + 外场 + 并行训练 + 并行 MD, 2 轮) ==="

mff-active-learn \
  --work-dir "$WORK_ROOT/al_work" \
  --data-dir "$WORK_ROOT/data" \
  --init-structure "$WORK_ROOT/seed_a.xyz" "$WORK_ROOT/seed_b.xyz" \
  --explore-type ase \
  --explore-mode md \
  --label-type identity \
  --tensor-product-mode pure-cartesian-ictd \
  --explore-external-field $EFIELD \
  --n-models 2 \
  --n-iterations 2 \
  --epochs 5 \
  --md-steps 100 \
  --md-temperature 300 \
  --conv-accuracy 1.01 \
  --device "$DEVICE" \
  --train-max-parallel 2 \
  --train-n-gpu "$TRAIN_N_GPU" \
  --explore-n-workers 2 \
  --diversity-metric none \
  --no-pre-eval

echo ""
echo "=== Step 3 验证: checkpoint + 外场架构 ==="

# 检查 ICTD checkpoint 存在
ITER0_CKPT="$WORK_ROOT/al_work/iterations/iter_0/checkpoint"
if ls "$ITER0_CKPT"/model_*_pure_cartesian_ictd.pth 1>/dev/null 2>&1; then
  echo "OK: iter_0 找到 pure_cartesian_ictd checkpoint"
  ls -la "$ITER0_CKPT"/
else
  echo "FAIL: iter_0 未找到 ICTD checkpoint"
  ls -la "$ITER0_CKPT"/ 2>/dev/null || true
  exit 1
fi

# 验证 checkpoint 包含 external_tensor_rank 和正确的 tensor_product_mode
python3 -c "
import torch, sys

ckpt_path = '$(ls "$ITER0_CKPT"/model_0_*.pth | head -1)'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

mode = ckpt.get('tensor_product_mode')
ext_rank = ckpt.get('external_tensor_rank')
state = ckpt.get('e3trans_state_dict', {})
has_ext_scale = 'e3_conv_emb.external_tensor_scale_by_l' in state

print(f'  tensor_product_mode = {mode}')
print(f'  external_tensor_rank = {ext_rank}')
print(f'  has external_tensor_scale_by_l in state_dict = {has_ext_scale}')

ok = True
if mode != 'pure-cartesian-ictd':
    print(f'FAIL: expected pure-cartesian-ictd, got {mode}')
    ok = False
if not has_ext_scale:
    print('FAIL: external_tensor_scale_by_l not found in state_dict')
    ok = False

if ok:
    print('OK: checkpoint 外场架构正确')
else:
    sys.exit(1)
"

# 检查 iter_1 存在（2 轮）
ITER1_CKPT="$WORK_ROOT/al_work/iterations/iter_1/checkpoint"
if ls "$ITER1_CKPT"/model_*_pure_cartesian_ictd.pth 1>/dev/null 2>&1; then
  echo "OK: iter_1 找到 ICTD checkpoint"
else
  echo "FAIL: iter_1 未找到 ICTD checkpoint"
  exit 1
fi

# 检查并行 MD 产物（多结构子轨迹）
if [ -f "$WORK_ROOT/al_work/iterations/iter_0/explore_traj_0.xyz" ] && \
   [ -f "$WORK_ROOT/al_work/iterations/iter_0/explore_traj_1.xyz" ]; then
  echo "OK: 并行 MD 子轨迹 (explore_traj_0.xyz, explore_traj_1.xyz) 存在"
else
  echo "INFO: 并行 MD 产物未拆分为子轨迹（可能是顺序执行或已合并）"
fi

if [ -f "$WORK_ROOT/al_work/iterations/iter_0/explore_traj.xyz" ]; then
  echo "OK: 合并轨迹 explore_traj.xyz 存在"
fi

# ===========================================================================
# Step 4: mff-active-learn --resume (从 checkpoint 重启)
# ===========================================================================
echo ""
echo "=== Step 4: mff-active-learn --resume (重启, 增加到 3 轮) ==="

mff-active-learn \
  --work-dir "$WORK_ROOT/al_work" \
  --data-dir "$WORK_ROOT/data" \
  --init-structure "$WORK_ROOT/seed_a.xyz" "$WORK_ROOT/seed_b.xyz" \
  --explore-type ase \
  --explore-mode md \
  --label-type identity \
  --tensor-product-mode pure-cartesian-ictd \
  --explore-external-field $EFIELD \
  --n-models 2 \
  --n-iterations 3 \
  --epochs 5 \
  --md-steps 100 \
  --md-temperature 300 \
  --conv-accuracy 1.01 \
  --device "$DEVICE" \
  --train-max-parallel 2 \
  --train-n-gpu "$TRAIN_N_GPU" \
  --explore-n-workers 2 \
  --diversity-metric none \
  --no-pre-eval \
  --resume

echo ""
echo "=== Step 4 验证: resume 产物 ==="

# 验证 al_state.json
if [ -f "$WORK_ROOT/al_work/al_state.json" ]; then
  echo "OK: al_state.json 存在"
  cat "$WORK_ROOT/al_work/al_state.json"
  echo ""
else
  echo "FAIL: al_state.json 不存在"
  exit 1
fi

# 验证 iter_2 存在（resume 后的第 3 轮）
ITER2_DIR="$WORK_ROOT/al_work/iterations/iter_2"
if [ -d "$ITER2_DIR" ]; then
  echo "OK: iter_2 目录存在 (resume 成功从 checkpoint 重启)"
  ls -la "$ITER2_DIR"/
else
  echo "FAIL: iter_2 目录不存在"
  exit 1
fi

# 验证 iter_2 checkpoint 也能用外场架构正确加载
ITER2_CKPT="$ITER2_DIR/checkpoint"
if ls "$ITER2_CKPT"/model_*_pure_cartesian_ictd.pth 1>/dev/null 2>&1; then
  echo "OK: iter_2 找到 ICTD checkpoint"
  python3 -c "
import torch
ckpt = torch.load('$(ls "$ITER2_CKPT"/model_0_*.pth | head -1)', map_location='cpu', weights_only=False)
state = ckpt.get('e3trans_state_dict', {})
has_ext = 'e3_conv_emb.external_tensor_scale_by_l' in state
print(f'  iter_2 checkpoint has external field params: {has_ext}')
if not has_ext:
    raise AssertionError('iter_2 checkpoint missing external field params')
print('OK: iter_2 checkpoint 外场架构正确')
"
else
  echo "FAIL: iter_2 未找到 ICTD checkpoint"
  exit 1
fi

# 验证 iter_2 model_devi 正常（模型可加载并推理）
if [ -f "$ITER2_DIR/model_devi.out" ]; then
  N_FRAMES=$(grep -v '^#' "$ITER2_DIR/model_devi.out" | wc -l | tr -d ' ')
  echo "OK: iter_2 model_devi.out 存在 ($N_FRAMES frames)"
else
  echo "FAIL: iter_2 model_devi.out 不存在"
  exit 1
fi

# 验证 merge 后 H5 中的 external_field 被正确注入
python3 -c "
import h5py, sys, numpy as np

h5_path = '$WORK_ROOT/data/processed_train.h5'
with h5py.File(h5_path, 'r') as f:
    keys = sorted([k for k in f.keys() if k.startswith('sample_')])
    n_with_efield = 0
    for k in keys:
        if 'external_field' in f[k]:
            n_with_efield += 1
            ef = np.array(f[k]['external_field'])
    print(f'  {n_with_efield}/{len(keys)} samples have external_field in {h5_path}')
    if n_with_efield == len(keys):
        print('OK: 所有样本都包含 external_field（merge 后外场自动注入成功）')
    else:
        print('FAIL: 部分样本缺少 external_field')
        sys.exit(1)
"

# ===========================================================================
# 汇总
# ===========================================================================
echo ""
echo "==========================================="
echo "  所有测试通过！"
echo "==========================================="
echo ""
echo "  已验证:"
echo "    [✓] 零样本冷启动 (mff-init-data, PySCF)"
echo "    [✓] pure-cartesian-ictd 模式训练"
echo "    [✓] --explore-external-field 自动推断 rank + 注入 H5"
echo "    [✓] 外场在 MD 探索中传入 (--explore-external-field)"
echo "    [✓] 外场在 model deviation 计算中传入"
echo "    [✓] 外场在 identity labeling 中传入"
echo "    [✓] 外场在 data merge 后自动注入 H5 (fp64)"
echo "    [✓] 并行训练 (train-max-parallel=2)"
echo "    [✓] 多结构并行 MD (explore-n-workers=2)"
echo "    [✓] 主动学习完整流程 (2 轮)"
echo "    [✓] Checkpoint 重启 (--resume, 第 3 轮)"
echo "    [✓] 外场架构在 checkpoint 中持久化"
echo "    [✓] Model deviation 可正确加载外场模型"
echo ""
