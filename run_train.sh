#!/bin/bash
# ============================================================
# 完整流水线训练启动脚本（MuSc → DataBridge → AnomalyNCD）
# 用法：bash run_train.sh [input_dir]
# ============================================================
set -u

cd /root/DCproject2026

# 覆盖系统里非法的 OMP_NUM_THREADS=0（会导致 OpenMP 报错）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# 输入目录（默认为测试数据）
INPUT_DIR="${1:-/root/autodl-tmp/DCproject/uploads/test_input}"

# 日志放数据盘根目录（注意：不要放 results/ 里，引擎每次会清空该目录）
LOG=/root/autodl-tmp/DCproject/train_log.txt

echo "=============================================="
echo "启动完整流水线，输入目录: $INPUT_DIR"
echo "日志文件: $LOG"
echo "=============================================="

python -u core/run_pipeline.py --input_dir "$INPUT_DIR" 2>&1 | tee "$LOG"
