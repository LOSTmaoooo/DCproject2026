# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：MuSc 核心算法封装器 (MuSc Wrapper)
# 文件名：musc_wrapper.py
# 功能：
#   1. 封装 MuSc (Multi-Scale Clustering?) 异常检测算法。
#   2. 管理深度学习模型 (DINOv2/CLIP) 的加载与推理。
#   3. 实现图像数据的预处理、特征提取、多尺度特征聚合与异常评分计算。
#   4. 提供统一的 API `generate_anomaly_maps` 供上层引擎调用。
# =================================================================================================

import os
import sys
import torch
import numpy as np
import cv2
import torch.nn.functional as F     # PyTorch 的函数式接口，包含插值、激活函数等
from PIL import Image               # Python Imaging Library，用于图像读取
from torchvision import transforms  # PyTorch 视觉库，用于图像预处理
from torch.utils.data import Dataset, DataLoader # PyTorch 数据加载工具
from tqdm import tqdm               # 进度条库

# -------------------------------------------------------------------------------------------------
# 路径配置与动态导入 (Dynamic Import Setup)
# -------------------------------------------------------------------------------------------------
# 目的：为了避免不同模块间同名包 (如 utils, models) 的冲突，执行清理和重新注入。
# -------------------------------------------------------------------------------------------------

# 1. 清理 sys.modules 中可能存在的冲突模块
#    这是一种防御性编程，防止之前导入的 AnomalyNCD 的 utils 模块干扰 MuSc 的导入。
for mod in ['utils', 'models', 'datasets']:
    if mod in sys.modules:
        del sys.modules[mod]

# 2. 定位 MuSc 库路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #获取项目根路径
MUSC_PATH = os.path.join(PROJECT_ROOT, 'libs', 'MuSc')

# 3. 将 MuSc 路径插入到 sys.path 的最前面 (索引 0)
#    这样 import models 时，Python 会优先从 libs/MuSc/models 加载，而不是其他地方。
if MUSC_PATH not in sys.path:
    sys.path.insert(0, MUSC_PATH)

# 4. 导入 MuSc 的核心组件
try:
    from models.musc import MuSc
    from utils.load_config import load_yaml
    # 导入具体的计算模块：
    # LNAMD: Local Neighborhood Aggregation via MD (局部邻域聚合)
    # MSM:   Multi-Scale Map (多尺度映射评分)
    from models.modules._LNAMD import LNAMD
    from models.modules._MSM import MSM
except ImportError as e:
    print(f"[MuScWrapper] Import Error: {e}. Please check permissions or path: {MUSC_PATH}")

# ==========================================
# 类：自定义推理数据集 (Inference Dataset)
# ==========================================
class InferenceDataset(Dataset):
    """
    类功能：
        为 PyTorch DataLoader 提供标准的数据接口。
        专用于"无标签推理"场景，即只有一个图片目录，没有掩模(Mask)或标签文件。
    """
    def __init__(self, image_dir, resize=256, imagesize=224, clip_transformer=None):
        """
        初始化数据集。
        Args:
            image_dir: 图片文件夹路径。
            resize: 图片预处理时的缩放尺寸。
            imagesize: Crop 后的最终输入尺寸。
            clip_transformer: 如果使用 CLIP 模型，传入其特定的预处理函数；否则使用 ImageNet 标准处理。
        """
        self.image_dir = image_dir
        # 定义支持的图片格式元组
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        # 列表推导式：遍历目录，筛选出符合扩展名的文件，并拼接完整路径
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        ]
        
        # 构建预处理流水线 (Transform Pipeline)
        if clip_transformer is None:
            # 标准 ImageNet 预处理：Resize -> CenterCrop -> ToTensor -> Normalize
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),  # 调整大小
                transforms.CenterCrop(imagesize),     # 中心裁剪
                transforms.ToTensor(),                # 转为浮点张量 [0, 1]
                # 标准化：减均值，除标准差
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # 使用模型自带的预处理
            self.transform = clip_transformer

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        按索引获取一个样本。
        Returns:
            dict: 包含处理后的图像张量、伪造的掩模、图像路径等。
        """
        img_path = self.image_paths[idx]
        # 使用 PIL 读取图像并转为 RGB 模式 (防止读入灰度图或RGBA图导致维度错误)
        image = Image.open(img_path).convert("RGB")
        # 应用预处理
        image_tensor = self.transform(image)
        
        # 创建一个全零的掩模 Tensor，形状为 [1, H, W]
        # 仅为了兼容某些可能需要 mask 字段的底层接口，实际推理中不使用真值。
        dummy_mask = torch.zeros([1, image_tensor.shape[1], image_tensor.shape[2]])
        
        return {
            "image": image_tensor,
            "mask": dummy_mask,
            "is_anomaly": 0,    # 默认为0 (未知/正常)
            "image_path": img_path
        }

# ==========================================
# 类：MuSc 算法封装器 (MuSc Wrapper)
# ==========================================
class MuScWrapper:
    """
    类功能：
        管理 MuSc 模型的生命周期。
        核心逻辑是 generate_anomaly_maps，它实现了从图片到异常热力图的全过程。
    """
    
    def __init__(self, config_path, device=None):
        """
        初始化模型。
        Args:
            config_path: YAML 配置文件路径。
            device: 运行设备 (None, int, or str)。
        """
        print(f"[MuScWrapper] Loading configuration from: {config_path}")
        self.cfg = load_yaml(config_path) # 加载配置字典
        
        # --- 设备选择逻辑 ---
        if device is None:
             # 如果未指定，优先使用 CUDA GPU，否则使用 CPU
             self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
             self.device = torch.device(f"cuda:{device}")
        else:
             self.device = torch.device(device)
             
        # 更新配置中的 device 字段，因为底层 MuSc 代码可能依赖该配置
        if self.device.type == 'cpu':
             self.cfg['device'] = 'cpu' 
        else:
             # 获取 GPU 索引号 (int)
             self.cfg['device'] = self.device.index if self.device.index is not None else 0

        # --- 模型加载 ---
        print(f"[MuScWrapper] Initializing MuSc Model on {self.device}...")
        # 实例化 MuSc 主类，传入配置和随机种子
        self.musc_model = MuSc(self.cfg, seed=42)
        
        # --- Backbone 识别 ---
        # 识别使用的是 DINOv2 还是 CLIP 作为特征提取器
        if hasattr(self.musc_model, 'dino_model'):
            self.backbone = self.musc_model.dino_model
            self.model_type = 'dino'
        elif hasattr(self.musc_model, 'clip_model'):
            self.backbone = self.musc_model.clip_model
            self.model_type = 'clip'
        else:
            raise ValueError("Unknown backbone type in MuSc model")

        # 提取关键属性方便后续调用
        self.preprocess = self.musc_model.preprocess   # 预处理函数
        self.features_list = self.musc_model.features_list # 需要提取特征的层索引 (例如 [5, 11, 17])
        self.r_list = self.musc_model.r_list               # 邻域聚合半径列表 (aggregating radius)
        self.batch_size = self.cfg['models']['batch_size']
        self.image_size = self.cfg['datasets']['img_resize']
        
    def _extract_features(self, images):
        """
        内部方法：从 Backbone 网络中提取指定层的特征图。
        Args:
            images: 输入图像张量 [B, C, H, W]
        Returns:
            list of tensors: 每一层的特征图列表。
        """
        # 针对不同模型架构调用不同的 API
        
        # 场景 A: DINOv2 模型
        if 'dinov2' in self.musc_model.model_name and self.model_type == 'dino':
            # get_intermediate_layers 返回指定层的输出
            # n 参数索引通常是 0-based，这里做了转换
            patch_tokens = self.backbone.get_intermediate_layers(x=images, n=[l-1 for l in self.features_list], return_class_token=False)
            patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
            
            # DINOv2 可能不返回 CLS token，这里手动添加一个假的 CLS token 以保持形状对齐
            # 形状操作：[B, N, C] -> concat([B, 1, C], [B, N, C])
            fake_cls = [torch.zeros_like(p)[:, 0:1, :] for p in patch_tokens]
            patch_tokens = [torch.cat([fake_cls[i], patch_tokens[i]], dim=1) for i in range(len(patch_tokens))]
            
        # 场景 B: 普通 DINO 模型
        elif 'dino' in self.musc_model.model_name and self.model_type == 'dino':
            patch_tokens_all = self.backbone.get_intermediate_layers(x=images, n=max(self.features_list))
            patch_tokens = [patch_tokens_all[l-1].cpu() for l in self.features_list]
            
        # 场景 C: CLIP 模型
        else: 
            # encode_image 返回图像编码和所有层的 token
            _, patch_tokens = self.backbone.encode_image(images, self.features_list)
            patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
            
        return patch_tokens

    def generate_anomaly_maps(self, input_dir, output_dir):
        """
        核心 API：生成异常热力图。
        
        流程：
        1. 构建 DataLoader。
        2. Extract: 遍历所有图片，提取特征并缓存到内存 (patch_tokens_list)。
        3. Compute: 使用 LNAMD 和 MSM 模块计算每个 patch 的异常分数。
        4. Resize & Save: 将分数插值回原图大小，保存为 .npy 文件。
        
        Args:
            input_dir: 图片输入路径。
            output_dir: 结果输出路径。
            
        Returns:
            list: 保存的文件路径列表。
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"[MuScWrapper] Starting batch generation for: {input_dir}")
        
        # --- 步骤 1: 数据准备 ---
        dataset = InferenceDataset(
            image_dir=input_dir, 
            resize=self.image_size, 
            imagesize=self.image_size,
            clip_transformer=self.preprocess
        )
        
        if len(dataset) == 0:
            print(f"[MuScWrapper] Warning: No images found in {input_dir}")
            return []
            
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True # 锁页内存，加速从 CPU 到 GPU 的传输
        )
        
        # --- 步骤 2: 特征提取 (Feature Extraction) ---
        print("[MuScWrapper] Step 1/2: Extracting features...")
        patch_tokens_list = [] # 存储所有批次的特征：[Batch1_Features, Batch2_Features, ...]
        image_paths_all = []   # 存储对应的图片路径
        
        with torch.no_grad(): # 上下文管理器：禁用梯度计算，节省显存
            for batch in tqdm(dataloader, desc="Extracting"):
                images = batch["image"].to(self.device)
                paths = batch["image_path"]
                
                # 调用内部方法提取特征
                batch_features = self._extract_features(images)
                
                patch_tokens_list.append(batch_features) 
                image_paths_all.extend(paths)
                
        # --- 步骤 3: 异常评分计算 (Computing Scores) ---
        print("[MuScWrapper] Step 2/2: Computing Scores (LNAMD + MSM)...")
        
        # anomaly_maps_r: 存储不同半径 r 下计算得到的异常图，最后求平均
        anomaly_maps_r = torch.tensor([]).double() 
        
        # 获取特征维度 C (channel)
        feature_dim = patch_tokens_list[0][0].shape[-1]
        
        # 循环遍历每一个聚合半径 r
        for r in self.r_list:
            print(f'Processing aggregation degree r={r}...')
            
            # --- 子步骤 3.1: LNAMD (局部邻域聚合) ---
            # 初始化 LNAMD 算子
            LNAMD_r = LNAMD(device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features_list)
            
            Z_layers = {} # 用于按层归类聚合后的特征
            
            # 遍历之前缓存的所有批次特征
            for batch_idx, batch_feats in enumerate(patch_tokens_list):
                batch_feats_gpu = [f.to(self.device) for f in batch_feats]
                
                with torch.no_grad():
                     # LNAMD._embed: 执行基于距离的局部特征聚合
                     # 返回 features 形状: [B, N_patches, num_layers, C]
                    features = LNAMD_r._embed(batch_feats_gpu)
                    
                    # 特征归一化 (L2 Norm)
                    features /= features.norm(dim=-1, keepdim=True)
                    
                    # 将特征拆分并存入 Z_layers 字典
                    for l_idx in range(len(self.features_list)):
                        l_key = str(l_idx)
                        if l_key not in Z_layers: Z_layers[l_key] = []
                        # features[:, :, l_idx, :] 取出当前层的所有 patch 特征
                        Z_layers[l_key].append(features[:, :, l_idx, :])
            
            # --- 子步骤 3.2: MSM (多尺度映射) ---
            anomaly_maps_l = torch.tensor([]).double()
            
            for l_key in Z_layers.keys():
                # 拼接所有批次的数据 -> [Total_Images, N_patches, C]
                Z = torch.cat(Z_layers[l_key], dim=0).to(self.device)
                
                # MSM: 计算每个 patch 在全集中的相对异常度 (基于余弦相似度等)
                # scores 形状: [Total_Images, N_patches]
                # topmin_min/max 用于控制背景过滤的阈值
                scores = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3)
                
                # 收集每一层的结果
                anomaly_maps_l = torch.cat((anomaly_maps_l, scores.unsqueeze(0).cpu()), dim=0)
                
                # 手动清理显存
                del Z
                torch.cuda.empty_cache()
            
            # 对所有层 (Layers) 的分数求平均
            anomaly_maps_l = torch.mean(anomaly_maps_l, dim=0) 
            # 存入 radius 列表
            anomaly_maps_r = torch.cat((anomaly_maps_r, anomaly_maps_l.unsqueeze(0)), dim=0)

        # --- 步骤 4: 结果融合与保存 ---
        # 对不同 r 的结果求平均，得到最终分数
        anomaly_maps_final = torch.mean(anomaly_maps_r, dim=0).to(self.device)
        
        # 准备上采样参数
        N, L_patches = anomaly_maps_final.shape
        H_feat = int(np.sqrt(L_patches)) # 特征图边长 (如 14)
        
        # 双线性插值 (Bilinear Interpolation) 上采样到原始图片尺寸
        # view() 改变形状: [N, L] -> [N, 1, H_feat, H_feat]
        anomaly_maps_resized = F.interpolate(
            anomaly_maps_final.view(N, 1, H_feat, H_feat),
            size=self.image_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        # 保存文件
        saved_paths = []
        anomaly_maps_np = anomaly_maps_resized.squeeze().cpu().numpy() # 转回 CPU numpy 数组
        
        # Handle single image batch case where squeeze removes too many dims
        if len(anomaly_maps_np.shape) == 2:
            anomaly_maps_np = anomaly_maps_np[np.newaxis, ...]
            
        print("[MuScWrapper] Saving anomaly maps...")
        for i in range(len(image_paths_all)):
            img_path = image_paths_all[i]
            basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # 获取单张图的热力图
            map_data = anomaly_maps_np[i]
            
            # 构建保存路径
            save_path = os.path.join(output_dir, f"{basename}_map.npy")
            
            # 保存为 NumPy 格式 (.npy) 保留浮点精度
            np.save(save_path, map_data)
            saved_paths.append(save_path)
            
        print(f"[MuScWrapper] Done. Saved {len(saved_paths)} maps to {output_dir}")
        return saved_paths


