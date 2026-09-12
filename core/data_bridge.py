# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：数据桥接器 (Data Bridge)
# 文件名：data_bridge.py
# 功能：
#   基于语义前缀 (known_normal, known_crack, unknown_new 等) 进行智能路由。
#   将 MuSc 的输出精确转换为 AnomalyNCD (MTD格式) 要求的目录结构和图片格式。
# =================================================================================================

import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm

class DataBridge:
    def __init__(self):
        pass

    def prepare_ncd_dataset(self, raw_images_dir, maps_dir, output_base_dir, category_name="custom_dataset"):
        print(f"[DataBridge] Starting semantic routing for AnomalyNCD. Category: {category_name}")

        # --- 1. 定义核心输出路径 (双保险兼容底层读取习惯) ---
        base_data_root_img = os.path.join(output_base_dir, "normal_ref", category_name, "images", "good")
        base_data_root_trn = os.path.join(output_base_dir, "normal_ref", category_name, "train", "good")
        base_data_root_msk = os.path.join(output_base_dir, "normal_ref", category_name, "masks", "good")
        images_root    = os.path.join(output_base_dir, "images", category_name)
        maps_root      = os.path.join(output_base_dir, "anomaly_maps", category_name)

        # --- 防御性编程：强制创建所有空壳目录 ---
        # 无论这次上传有没有正常样本，先把架子搭好，防止下游 os.listdir 找不到文件夹崩溃
        for d in [base_data_root_img, base_data_root_trn, base_data_root_msk, images_root, maps_root]:
            os.makedirs(d, exist_ok=True)

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        processed_count = {"normal": 0, "anomaly": 0}

        # --- 1.5 收集所有 anomaly map 的全局 min/max ---
        # 修复：不能用逐张 min-max 归一化，否则每张图的 max 都会变成 255，
        # 导致 MEBin 的 get_search_range 拿到 min_th=max_th=255，阈值搜索退化、二值化全黑。
        global_map_min = np.inf
        global_map_max = -np.inf
        for mroot, mdirs, mfiles in os.walk(maps_dir):
            for mf in mfiles:
                if mf.endswith("_map.npy"):
                    am = np.load(os.path.join(mroot, mf))
                    global_map_min = min(global_map_min, float(am.min()))
                    global_map_max = max(global_map_max, float(am.max()))
        if global_map_max <= global_map_min:
            global_map_min, global_map_max = 0.0, 1.0  # 退化保护，避免除零

        # --- 2. 递归遍历原图目录进行路由 ---
        for root, dirs, files in os.walk(raw_images_dir):
            for file in files:
                if not file.lower().endswith(valid_exts):
                    continue

                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, raw_images_dir)
                
                rel_dir = os.path.dirname(rel_path)
                basename = os.path.splitext(file)[0]
                
                # 【修复 1】提取最底层文件夹名作为类别，避免外层嵌套目录干扰
                actual_category = os.path.basename(rel_dir) if rel_dir else "unknown_default"
                anomaly_type = actual_category.replace(os.sep, "_") 

                # 只有 known_normal 作为 AnomalyNCD 的正常参考类，其余类别都进入待聚类样本侧。
                if anomaly_type == "known_normal":
                    for d in [base_data_root_img, base_data_root_trn, base_data_root_msk]:
                        os.makedirs(d, exist_ok=True)

                    shutil.copy2(img_path, os.path.join(base_data_root_img, file))
                    shutil.copy2(img_path, os.path.join(base_data_root_trn, file))

                    orig_img = cv2.imread(img_path)
                    if orig_img is not None:
                        h, w = orig_img.shape[:2]
                        blank_mask = np.zeros((h, w), dtype=np.uint8)
                        mask_save_name = file if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) else f"{basename}.png"
                        mask_save_path = os.path.join(base_data_root_msk, mask_save_name)
                        cv2.imwrite(mask_save_path, blank_mask)

                    processed_count["normal"] += 1

                else:
                    dest_img_path = os.path.join(images_root, anomaly_type, file)
                    dest_map_path = os.path.join(maps_root, anomaly_type, f"{basename}.png")
                    
                    os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
                    os.makedirs(os.path.dirname(dest_map_path), exist_ok=True)
                    
                    shutil.copy2(img_path, dest_img_path)
                    
                    map_path = os.path.join(maps_dir, rel_dir, f"{basename}_map.npy")
                    if os.path.exists(map_path):
                        anomaly_map = np.load(map_path)
                        # 全局归一化：保留图间分数差异（逐张 min-max 会让每张 max=255，MEBin 退化）
                        norm_map = (anomaly_map - global_map_min) / (global_map_max - global_map_min)
                        map_uint8 = np.clip(norm_map * 255, 0, 255).astype(np.uint8)
                        cv2.imwrite(dest_map_path, map_uint8)
                    else:
                        blank_map = np.zeros((256, 256), dtype=np.uint8)
                        cv2.imwrite(dest_map_path, blank_map)
                        
                    processed_count["anomaly"] += 1

        print(f"[DataBridge] Routing complete: {processed_count['normal']} normal ref, {processed_count['anomaly']} anomalies.")
        
        # --- 核心修正：适配 AnomalyNCD 不对称的路径读取逻辑 ---
        # base 使用 AeBAD（4 类异常，作为 labeled 先验），符合 AnomalyNCD 原始设计。
        # 之前用 known_normal（正常类）做 base，偏离设计导致 cls 不收敛。
        aebad_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'libs', 'AnomalyNCD', 'data', 'AeBAD_crop'
        )
        return {
            'base_data_path':    aebad_path,
            'images_path':       os.path.join(output_base_dir, "images", category_name),

            # anomaly_maps 保持在外层根目录，因为 NCD 底层会自动拼接 category_name
            'anomaly_maps_path': os.path.join(output_base_dir, "anomaly_maps"),
            'category_name':     category_name
        }