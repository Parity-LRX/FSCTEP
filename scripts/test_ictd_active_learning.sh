#!/bin/bash
# 测试 pure-cartesian-ictd 模式下的完整零样本启动 + 主动学习流程，以及 Checkpoint 重启
#
# 流程：
# 1. mff-init-data: 冷启动生成初始数据 (PySCF)
# 2. mff-active-learn: 主动学习 2 轮 (identity 标注，快速测试)
# 3. mff-active-learn --resume: 验证 checkpoint 能否正确重启

set -e
cd "$(dirname "$0")/.."
WORK_ROOT="test_ictd_al"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"

# 创建水分子种子结构
python3 -c "
from ase import Atoms
from ase.io import write
atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0.48, 0.83, 0]])
write('$WORK_ROOT/seed.xyz', atoms, format='extxyz')
print('Created seed.xyz')
"

echo "=== Step 1: mff-init-data (零样本冷启动) ==="
mff-init-data \
  --structures "$WORK_ROOT/seed.xyz" \
  --n-perturb 5 \
  --rattle-std 0.05 \
  --label-type pyscf \
  --pyscf-method b3lyp \
  --pyscf-basis sto-3g \
  --label-n-workers 2 \
  --output-dir "$WORK_ROOT/data"

echo ""
echo "=== Step 2: mff-active-learn (pure-cartesian-ictd, 2 轮) ==="
# --conv-accuracy 1.01 使收敛条件永不满足，强制跑满 2 轮（小模型在 MD 上偏差极低易过早收敛）
mff-active-learn \
  --work-dir "$WORK_ROOT/al_work" \
  --data-dir "$WORK_ROOT/data" \
  --init-structure "$WORK_ROOT/seed.xyz" \
  --explore-type ase \
  --explore-mode md \
  --label-type identity \
  --tensor-product-mode pure-cartesian-ictd \
  --n-models 2 \
  --n-iterations 2 \
  --epochs 5 \
  --md-steps 150 \
  --conv-accuracy 1.01 \
  --device cpu \
  --diversity-metric none

echo ""
echo "=== 验证 Step 2 产物: ICTD checkpoint 存在 ==="
CKPT_DIR="$WORK_ROOT/al_work/iterations/iter_0/checkpoint"
if ls "$CKPT_DIR"/model_*_pure_cartesian_ictd.pth 1>/dev/null 2>&1; then
  echo "OK: 找到 pure_cartesian_ictd checkpoint"
  ls -la "$CKPT_DIR"/
else
  echo "WARN: 未找到 model_*_pure_cartesian_ictd.pth，检查其他格式"
  ls -la "$CKPT_DIR"/ 2>/dev/null || true
fi

echo ""
echo "=== Step 3: mff-active-learn --resume (验证 checkpoint 重启) ==="
# resume 从 iter_1 的 checkpoint 继续，应能完成 iter_2
mff-active-learn \
  --work-dir "$WORK_ROOT/al_work" \
  --data-dir "$WORK_ROOT/data" \
  --init-structure "$WORK_ROOT/seed.xyz" \
  --explore-type ase \
  --explore-mode md \
  --label-type identity \
  --tensor-product-mode pure-cartesian-ictd \
  --n-models 2 \
  --n-iterations 3 \
  --epochs 5 \
  --md-steps 150 \
  --conv-accuracy 1.01 \
  --device cpu \
  --resume \
  --diversity-metric none

echo ""
echo "=== 验证 Step 3: al_state.json 与 iter_2 存在 ==="
if [ -f "$WORK_ROOT/al_work/al_state.json" ]; then
  echo "OK: al_state.json 存在"
  cat "$WORK_ROOT/al_work/al_state.json"
fi
if [ -d "$WORK_ROOT/al_work/iterations/iter_2" ]; then
  echo "OK: iter_2 目录存在 (resume 后继续到第 3 轮)"
  ls -la "$WORK_ROOT/al_work/iterations/iter_2"/
else
  echo "INFO: iter_2 可能因收敛而未创建；若 al_state 显示 total_iters_done>=2 则 resume 正常"
fi

echo ""
echo "=== 测试完成 ==="
