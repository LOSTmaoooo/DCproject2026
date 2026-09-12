#!/bin/bash
# ============================================================
# 标准 benchmark 流程：AeBAD 做 base + mtd 做 novel（复现学长那套）
# 用法：bash run_standard_mtd.sh
# ============================================================
set -u
cd /root/DCproject2026/libs/AnomalyNCD

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

echo "=== 标准流程训练：base=AeBAD_crop, novel=mtd ==="
python -u examples/anomalyncd_main.py \
    --runner_name "mtd_musc_crop" \
    --dataset "mtd" \
    --category "MTD" \
    --dataset_path "data/mtd_anomaly_detection/test" \
    --anomaly_map_path "data/mtd_musc_anomaly_map" \
    --binary_data_path "data/mtd_musc" \
    --crop_data_path "data/mtd_musc_crop" \
    --base_data_path "data/AeBAD_crop" 2>&1 | tee /tmp/standard_mtd.log
