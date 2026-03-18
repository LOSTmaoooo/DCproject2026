# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：数据桥接器 (Data Bridge)
# 文件名：data_bridge.py
# 功能：
#   负责连接 MuSc 和 AnomalyNCD 两个系统的数据格式。
#   主要任务是将 MuSc 生成的 .npy 格式热力图，转换为 AnomalyNCD 所需的特定目录结构和 PNG 格式图片。
# =================================================================================================

import os
import sys
import shutil
import numpy as np
import cv2  # OpenCV 库，用于图像处理
from tqdm import tqdm

# --- 路径配置 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCD_PATH = os.path.join(PROJECT_ROOT, 'libs', 'AnomalyNCD')
sys.path.append(NCD_PATH)

# --- 常量定义 ---
# 这些常量需要与 AnomalyNCDWrapper 中的配置保持一致
CATEGORY_NAME = "unknown"    # 类别名
ANOMALY_TYPE  = "unknown"    # 异常类型名 (用于未标记数据)

class DataBridge:
    """
    类：数据桥接器
    
    Data Bridge 的存在是为了解耦 MuSc 和 AnomalyNCD。
    MuSc 专注于生成热力图，而不关心下游任务所需的文件夹结构。
    本类负责将两者适配起来。
    """
    
    def __init__(self):
        pass

    def prepare_ncd_dataset(self, raw_images_dir, maps_dir, output_base_dir):
        """
        方法：准备 NCD 数据集
        
        功能：
            遍历原始图片和对应的热力图，将它们复制/转换到目标目录，
            构建从 MuSc 输出到 AnomalyNCD 输入的管道。
            
        输入要求：
            raw_images_dir: 包含 jpg/png 原图的文件夹。
            maps_dir: 包含 _map.npy 文件的文件夹 (由 MuSc 生成)。
            
        输出结构 (AnomalyNCD MTD/Custom 格式):
           output_base_dir/
             |-- images/
             |     |-- unknown/
             |           |-- img1.png
             |-- anomaly_maps/
                   |-- unknown/
                         |-- unknown/
                               |-- img1.png  (注意：这里必须是 PNG 灰度图，不能是 npy)
        
        Args:
            raw_images_dir (str): 源图像目录。
            maps_dir (str): 源热力图目录。
            output_base_dir (str): 目标根目录。
            
        Returns:
            dict: 包含 'images_path' 和 'anomaly_maps_path' 的字典，供后续步骤使用。
            或 None (如果失败)。
        """
        print(f"[DataBridge] Starting data preparation for AnomalyNCD...")

        # --- 1. 创建目标目录结构 ---
        
        # 构建 images 路径：output/images/unknown/
        images_root     = os.path.join(output_base_dir, "images")
        out_images_dir  = os.path.join(images_root, ANOMALY_TYPE)

        # 构建 anomaly_maps 路径：output/anomaly_maps/unknown/unknown/
        maps_root       = os.path.join(output_base_dir, "anomaly_maps")
        out_maps_dir    = os.path.join(maps_root, CATEGORY_NAME, ANOMALY_TYPE)

        # 创建目录 (如果父目录不存在会自动创建)
        for d in [out_images_dir, out_maps_dir]:
            os.makedirs(d, exist_ok=True)

        # --- 2. 扫描源文件 ---
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        # 获取所有支持格式的图片文件列表，并排序
        image_files = sorted(
            f for f in os.listdir(raw_images_dir) if f.lower().endswith(valid_exts)
        )

        if not image_files:
            print(f"[DataBridge] Error: No images found in {raw_images_dir}")
            return None

        processed_count = 0

        # --- 3. 遍历处理每张图片 ---
        for img_name in tqdm(image_files, desc="Bridging Data"):
            # 获取文件名 (不含扩展名)，例如 "img001"
            base_name = os.path.splitext(img_name)[0]
            
            # 构建源文件完整路径
            img_path  = os.path.join(raw_images_dir, img_name)
            # 推断应存在的热力图文件名：<原文件名>_map.npy
            map_path  = os.path.join(maps_dir, f"{base_name}_map.npy")

            # 检查热力图是否存在
            if not os.path.exists(map_path):
                print(f"[DataBridge] Warning: Map not found for {img_name}, skipping.")
                continue

            # --- 动作 A: 复制原图 ---
            # 直接将原图复制到目标 images 目录
            dest_img_path = os.path.join(out_images_dir, img_name)
            shutil.copy2(img_path, dest_img_path)

            # --- 动作 B: 转换异常热力图 (.npy -> .png) ---
            # AnomalyNCD 的 MEBin 模块通常使用 opencv 读取灰度图，因此必须转为 PNG
            
            # 加载 float32 的 npy 数组
            anomaly_map = np.load(map_path)
            
            # 数据归一化到 [0, 1] 区间
            map_min, map_max = float(anomaly_map.min()), float(anomaly_map.max())
            if map_max > map_min:
                norm_map = (anomaly_map - map_min) / (map_max - map_min)
            else:
                norm_map = np.zeros_like(anomaly_map)
                
            # 缩放到 [0, 255] 并转为 uint8 类型
            map_uint8 = (norm_map * 255).astype(np.uint8)

            # 写入 PNG 文件
            dest_map_path = os.path.join(out_maps_dir, f"{base_name}.png")
            cv2.imwrite(dest_map_path, map_uint8)

            processed_count += 1

        print(f"[DataBridge] Prepared {processed_count} image-map pairs.")
        print(f"[DataBridge]   images_path      -> {images_root}")
        print(f"[DataBridge]   anomaly_maps_path-> {maps_root}")

        # 返回构建好的路径供 AnomalyNCD 使用
        return {
            'images_path':       images_root,
            'anomaly_maps_path': maps_root,
        }

if __name__ == "__main__":
    # 简单的本地测试代码
    bridge = DataBridge()
    # ... 测试逻辑 ...
