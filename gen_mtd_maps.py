# -*- coding: utf-8 -*-
# 生成标准流程所需的 mtd anomaly maps（格式：mtd_musc_anomaly_map/MTD/{anomaly_type}/*.png）
# 等价于 AnomalyNCD README 里的 generate_anomaly_maps：全局归一化 + 转 PNG
import sys, os, glob
sys.path.insert(0, '/root/DCproject2026')
import numpy as np, cv2

DATA = '/root/DCproject2026/libs/AnomalyNCD/data'
SRC = os.path.join(DATA, 'mtd_anomaly_detection', 'test')   # 6 类（含 MT_Free 正常）
TMP = '/tmp/mtd_standard_maps'
OUT = os.path.join(DATA, 'mtd_musc_anomaly_map', 'MTD')

# 1. MuSc 生成 .npy（mirror 结构）
from core.musc_wrapper import MuScWrapper
musc = MuScWrapper('/root/DCproject2026/libs/MuSc/configs/musc.yaml')
musc.generate_anomaly_maps(SRC, TMP)
print(f'[1/2] MuSc 生成完成 -> {TMP}')

# 2. 全局归一化 + 转 PNG（保留图间差异）
all_maps = sorted(glob.glob(os.path.join(TMP, '**', '*_map.npy'), recursive=True))
gmin, gmax = np.inf, -np.inf
for f in all_maps:
    m = np.load(f)
    gmin = min(gmin, float(m.min()))
    gmax = max(gmax, float(m.max()))
print(f'[2/2] 全局 min={gmin:.4f}, max={gmax:.4f}, 共 {len(all_maps)} 张')

for f in all_maps:
    rel = os.path.relpath(f, TMP)                       # e.g. MT_Blowhole/exp1_xxx_map.npy
    anomaly_type = os.path.dirname(rel)
    basename = os.path.basename(rel).replace('_map.npy', '.png')
    m = np.load(f)
    norm = (m - gmin) / (gmax - gmin)
    dst_dir = os.path.join(OUT, anomaly_type)
    os.makedirs(dst_dir, exist_ok=True)
    cv2.imwrite(os.path.join(dst_dir, basename), np.clip(norm * 255, 0, 255).astype(np.uint8))

# 统计
for d in sorted(glob.glob(os.path.join(OUT, '*'))):
    print(f'  {os.path.basename(d)}: {len(os.listdir(d))} 张')
print('完成：anomaly maps 已生成到', OUT)
